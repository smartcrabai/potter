//! Blender 5.2.2 mask layer sampling (`mask_rasterize.cc`).
//!
//! A spline is one layer: its body is filled (depth 0) and its feather ring is a strip of quads
//! whose depth ramps from 0 at the body edge to 1 at the ring's outer edge, interpolated with
//! mean-value weights (`barycentric_weights_v2_quad`). A sample's depth is the smallest depth among
//! the faces that contain it, and the layer value is `1 - depth` eased by `PROP_SMOOTH`
//! (`3v² - 2v³`), as `BKE_maskrasterize_handle_sample` does for the default falloff.

use glam::DVec2;

use super::evaluate::Contour;

/// A feather quad: body edge `[0]`-`[1]` and ring edge `[3]`-`[2]`, in Blender's vertex order.
#[derive(Clone, Copy, Debug)]
struct Quad {
    vertices: [DVec2; 4],
    minimum: DVec2,
    maximum: DVec2,
}

impl Quad {
    fn new(vertices: [DVec2; 4]) -> Self {
        let minimum = vertices
            .iter()
            .fold(DVec2::INFINITY, |minimum, vertex| minimum.min(*vertex));
        let maximum = vertices
            .iter()
            .fold(DVec2::NEG_INFINITY, |maximum, vertex| maximum.max(*vertex));
        Self {
            vertices,
            minimum,
            maximum,
        }
    }

    fn contains_bounds(&self, point: DVec2) -> bool {
        point.x >= self.minimum.x
            && point.x <= self.maximum.x
            && point.y >= self.minimum.y
            && point.y <= self.maximum.y
    }
}

/// One spline's layer: a filled body and its feather quads.
#[derive(Debug)]
pub(super) struct Layer {
    body: Vec<DVec2>,
    quads: Vec<Quad>,
}

impl Layer {
    /// Builds the layer from a contour. Body edge `j` joins ring edge `j`, and the ring wraps, so an
    /// open spline's feather closes like its body.
    pub(super) fn new(contour: Contour) -> Self {
        let quads = contour.feather.map_or_else(Vec::new, |ring| {
            let count = contour.body.len();
            (0..count)
                .map(|index| {
                    let next = (index + 1) % count;
                    Quad::new([
                        contour.body[index],
                        contour.body[next],
                        ring[next],
                        ring[index],
                    ])
                })
                .collect()
        });
        Self {
            body: contour.body,
            quads,
        }
    }

    /// The layer's value at a normalized sample point, in `[0, 1]` before union clamping.
    pub(super) fn value(&self, point: DVec2) -> f64 {
        let depth = if point_inside_polygon(point, &self.body) {
            0.0
        } else {
            self.quads
                .iter()
                .filter(|quad| quad.contains_bounds(point))
                .filter_map(|quad| quad_depth(point, &quad.vertices))
                .fold(1.0_f64, f64::min)
        };
        let value = 1.0 - depth;
        3.0 * value * value - 2.0 * value * value * value
    }
}

/// Depth of a feather quad at `point`, if the quad covers it. The quad is covered when either of
/// its two triangles is; the depth is the mean-value interpolation of the ring weights.
fn quad_depth(point: DVec2, vertices: &[DVec2; 4]) -> Option<f64> {
    if inside_triangle(point, vertices[0], vertices[1], vertices[2])
        || inside_triangle(point, vertices[0], vertices[2], vertices[3])
    {
        let weights = quad_weights(point, vertices);
        Some(weights[2] + weights[3])
    } else {
        None
    }
}

/// Inclusive point-in-triangle test for either winding.
fn inside_triangle(point: DVec2, first: DVec2, second: DVec2, third: DVec2) -> bool {
    let sides = [
        (second - first).perp_dot(point - first),
        (third - second).perp_dot(point - second),
        (first - third).perp_dot(point - third),
    ];
    let negative = sides.iter().any(|side| *side < 0.0);
    let positive = sides.iter().any(|side| *side > 0.0);
    !(negative && positive)
}

/// `barycentric_weights_v2_quad`: mean-value weights of `point` in a quad, with Blender's handling
/// of points on an edge and of degenerate quads.
fn quad_weights(point: DVec2, vertices: &[DVec2; 4]) -> [f64; 4] {
    let directions = vertices.map(|vertex| vertex - point);
    let lengths = directions.map(DVec2::length);
    if let Some(index) = lengths.iter().position(|length| *length < FLOAT_EPSILON) {
        let mut weights = [0.0; 4];
        weights[index] = 1.0;
        return weights;
    }
    let next = |index: usize| (index + 1) % 4;
    let areas: [f64; 4] =
        std::array::from_fn(|index| directions[index].perp_dot(directions[next(index)]));
    let dots: [f64; 4] =
        std::array::from_fn(|index| directions[index].dot(directions[next(index)]));
    let products: [f64; 4] = std::array::from_fn(|index| lengths[index] * lengths[next(index)]);
    let on_edge =
        (0..4).find(|&index| areas[index].abs() < 1.0e-12 * products[index] && dots[index] <= 0.0);
    if let Some(index) = on_edge {
        let other = next(index);
        let sum = lengths[index] + lengths[other];
        let mut weights = [0.0; 4];
        weights[index] = lengths[other] / sum;
        weights[other] = lengths[index] / sum;
        return weights;
    }
    let half_tangents: [f64; 4] = std::array::from_fn(|index| {
        if areas[index] == 0.0 {
            0.0
        } else {
            ((products[index] - dots[index]) / areas[index]).abs()
        }
    });
    let mut weights: [f64; 4] = std::array::from_fn(|index| {
        let previous = (index + 3) % 4;
        (half_tangents[previous] + half_tangents[index]) / lengths[index]
    });
    let total: f64 = weights.iter().sum();
    if total != 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
        if weights.iter().all(|weight| weight.is_finite()) {
            return weights;
        }
    }
    [0.25; 4]
}

/// `FLT_EPSILON`, below which a quad vertex coincides with the sample.
const FLOAT_EPSILON: f64 = f32::EPSILON as f64;

/// Even-odd containment of `point` in the closed polygon `vertices`.
fn point_inside_polygon(point: DVec2, vertices: &[DVec2]) -> bool {
    let mut inside = false;
    let Some(&last) = vertices.last() else {
        return false;
    };
    let mut previous = last;
    for &current in vertices {
        if (current.y > point.y) != (previous.y > point.y) {
            let amount = (point.y - current.y) / (previous.y - current.y);
            let intersection_x = current.x + (previous.x - current.x) * amount;
            if point.x < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}
