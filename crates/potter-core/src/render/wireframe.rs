//! Wireframe preview: Blender 5.2.2's viewport wireframe overlay (`draw/engines/overlay`), not the
//! Workbench render, which draws nothing in wireframe shading (`workbench_state.cc` sets `xray_alpha`
//! to 0 below `OB_SOLID`). Every mesh edge is one line:
//!
//! - colour: the facing of its corner normal, `|n · V|`, mixes the theme's rim and interior colours in
//!   gamma 2.2 with weight 0.35 (`overlay_wireframe_vert.glsl`, `wire_color_get`);
//! - visibility: `wd` is the edge's cosine band (`extract_mesh_vbo_edge_fac.cc`), tested against the
//!   overlay threshold (`wire_discard_threshold_get`, `is_edge_sharpness_visible`);
//! - x-ray: wires are not occluded by surfaces, since `xray_alpha_wire` is below 1 by default;
//! - depth: each pixel keeps its nearest line (`LESS_EQUAL` depth test against a cleared buffer);
//! - anti-aliasing: the centre pixel's coverage, and the 4-neighbour cross, blended by depth
//!   (`overlay_antialiasing.bsl.hh`); the blend is premultiplied and composited in linear light over the
//!   theme background;
//! - display: the overlay framebuffer is shown with a 2.2 gamma and no view transform (checked: the
//!   `Standard` and `AgX` captures are byte-identical).
//!
//! The model is validated against a Blender 5.2.2 viewport capture in `raster.rs`'s tests (fixture
//! `tests/fixtures/workbench_wireframe`). GPU line rasterisation on dense, curved lines is not
//! reproduced bit-exactly: see the tolerance notes in that test.

use std::collections::HashMap;

use glam::DVec3;

/// `view_3d.wire` of the factory theme (`userdef_default_theme.c`, `space_view3d.wire`), the rim colour.
pub(crate) const THEME_WIRE_SRGB: [f64; 3] = [0.0, 0.0, 0.0];

/// `space.gradients.high_gradient` of the factory theme (`TH_BACK`, `space_view3d.back` = `0x3d3d3d`), the
/// interior colour and the viewport background. The byte is stored as `0x3d / 255` in the theme.
pub(crate) const THEME_BACK_SRGB: [f64; 3] = [61.0 / 255.0, 61.0 / 255.0, 61.0 / 255.0];

/// `overlay.wireframe_threshold`, default 1.0 (`DNA_view3d_types.h`): every edge is kept.
pub(crate) const WIRE_THRESHOLD: f64 = 1.0;

/// `1 / sqrt(pi) * 1.05` (`overlay_shader_shared.hh`, `DISC_RADIUS`).
pub(crate) const DISC_RADIUS: f64 = 0.564_189_583_547_756_3 * 1.05;
/// `0.5 - DISC_RADIUS` (`LINE_SMOOTH_START`).
pub(crate) const LINE_SMOOTH_START: f64 = 0.5 - DISC_RADIUS;
/// `0.5 + DISC_RADIUS` (`LINE_SMOOTH_END`).
pub(crate) const LINE_SMOOTH_END: f64 = 0.5 + DISC_RADIUS;

/// Half a pixel: a fragment is written where its pixel centre is within this distance of the line
/// (the GPU line rule for 1-pixel lines, as the prototype models it).
const LINE_HALF_WIDTH: f64 = 0.5;

/// The overlay's gamma for its colour blend and for display (`pow(·, 1/2.2)`).
const OVERLAY_GAMMA: f64 = 2.2;

/// `1 - 1/255` (`wire_discard_threshold_get`): the largest `wd` a threshold of 1 keeps, as the 8-bit
/// edge factor is stored up to `254/255`.
const WD_STEP: f64 = 254.0 / 255.0;

/// `theme.sizes.vert` (`overlay_private.hh`, `vertex_size_get`): the theme vertex size (3 in the factory
/// theme, `space_view3d`) times √2/2, at least 1. A loose vertex is drawn as a square of `2 · vert` pixels.
pub(crate) fn vertex_size() -> f64 {
    (3.0 * std::f64::consts::SQRT_2 / 2.0).max(1.0)
}

/// A mesh edge or a stroke, in world space.
#[derive(Clone, Debug)]
pub(crate) struct WireEdge {
    pub(crate) start: DVec3,
    pub(crate) end: DVec3,
    /// World corner normals of `start` and `end` in the first face that uses the edge; `None` for a
    /// stroke, which takes `colour` as it is.
    pub(crate) normals: Option<[DVec3; 2]>,
    /// Linear colour of a stroke (`normals` is `None`).
    pub(crate) colour: [f64; 3],
    /// The edge factor `wd` (`extract_mesh_vbo_edge_fac.cc`): 0 for a boundary or non-manifold edge.
    pub(crate) wd: f64,
}

/// Accumulates the mesh edges of one object, each once, in the order their first polygon is visited.
#[derive(Default)]
pub(crate) struct WireEdgeBuilder {
    edges: Vec<PendingEdge>,
    index: HashMap<(u32, u32), usize>,
}

struct PendingEdge {
    start: DVec3,
    end: DVec3,
    normals: [DVec3; 2],
    faces: Vec<DVec3>,
}

impl WireEdgeBuilder {
    /// Adds the boundary edges of one polygon: `vertices` are its vertex ids, `positions` their world
    /// positions, and `normals` its world corner normals (`None` for a flat face, whose normal is the
    /// polygon's).
    pub(crate) fn add_polygon(
        &mut self,
        vertices: &[u32],
        positions: &[DVec3],
        normals: Option<&[DVec3]>,
    ) {
        let count = vertices.len();
        if count < 3 || positions.len() != count {
            return;
        }
        let face_normal = polygon_normal(positions);
        let corner_normal = |index: usize| {
            normals
                .and_then(|values| values.get(index).copied())
                .unwrap_or(face_normal)
        };
        for (corner, &a) in vertices.iter().enumerate() {
            let next = (corner + 1) % count;
            let b = vertices[next];
            let key = (a.min(b), a.max(b));
            if let Some(&slot) = self.index.get(&key) {
                self.edges[slot].faces.push(face_normal);
                continue;
            }
            self.index.insert(key, self.edges.len());
            self.edges.push(PendingEdge {
                start: positions[corner],
                end: positions[next],
                normals: [corner_normal(corner), corner_normal(next)],
                faces: vec![face_normal],
            });
        }
    }

    /// Whether the edge between two vertex ids is already on a polygon.
    pub(crate) fn contains(&self, a: u32, b: u32) -> bool {
        self.index.contains_key(&(a.min(b), a.max(b)))
    }

    /// The edges in insertion order, with their `wd`.
    pub(crate) fn finish(self) -> Vec<WireEdge> {
        self.edges
            .into_iter()
            .map(|edge| WireEdge {
                start: edge.start,
                end: edge.end,
                normals: Some(edge.normals),
                colour: [0.0; 3],
                wd: edge_factor(&edge.faces),
            })
            .collect()
    }
}

/// Newell normal of a polygon, normalised; zero for a degenerate polygon.
pub(crate) fn polygon_normal(positions: &[DVec3]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for (index, current) in positions.iter().enumerate() {
        let next = positions[(index + 1) % positions.len()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    normal.normalize_or_zero()
}

/// `wd` of an edge from the normals of the faces that use it (`edge_factor_calc`): an edge shared by
/// two faces is `clamp(200 (cos - 1) + 1, 0, 1) * 254/255`, and any other edge is 0 (always visible).
pub(crate) fn edge_factor(faces: &[DVec3]) -> f64 {
    match faces {
        [first, second] => {
            let cosine = first.dot(*second);
            (200.0 * (cosine - 1.0) + 1.0).clamp(0.0, 1.0) * WD_STEP
        }
        _ => 0.0,
    }
}

/// Whether an edge with factor `wd` is drawn at `threshold` (`is_edge_sharpness_visible`):
/// `wd <= sqrt(|threshold|) * (1 - 1/255)`.
pub(crate) fn edge_visible(wd: f64, threshold: f64) -> bool {
    wd <= threshold.abs().sqrt() * WD_STEP
}

/// Smoothstep from `edge0` to `edge1` (`smoothstep` in GLSL, for either ordering of the edges).
pub(crate) fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Coverage of a line onto a pixel centre `distance` pixels from it, with a zero inner kernel (theme
/// pixel size 1): `smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, |distance|)`.
pub(crate) fn line_coverage(distance: f64) -> f64 {
    smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, distance.abs())
}

/// `srgb_to_linearrgb` of `BLI_color` (the piecewise sRGB decode).
pub(crate) fn srgb_to_linear(value: f64) -> f64 {
    if value < 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear theme colours of the wire pass: `(rim, interior)`.
pub(crate) fn theme_linear() -> ([f64; 3], [f64; 3]) {
    (
        THEME_WIRE_SRGB.map(srgb_to_linear),
        THEME_BACK_SRGB.map(srgb_to_linear),
    )
}

/// The colour of a line at `facing` (`|n · V|`, clamped to [0, 1]) in linear light
/// (`overlay_wireframe_vert.glsl`): rim and interior are taken to gamma 1/2.2, mixed by 0.35 for the
/// front colour, mixed with the rim by `facing`, and taken back to linear.
pub(crate) fn facing_colour(facing: f64, rim: [f64; 3], interior: [f64; 3]) -> [f64; 3] {
    let facing = facing.abs().clamp(0.0, 1.0);
    let gamma = 1.0 / OVERLAY_GAMMA;
    std::array::from_fn(|channel| {
        let rim_gamma = rim[channel].powf(gamma);
        let interior_gamma = interior[channel].powf(gamma);
        let front = rim_gamma + (interior_gamma - rim_gamma) * 0.35;
        let colour = rim_gamma + (front - rim_gamma) * facing;
        colour.powf(OVERLAY_GAMMA)
    })
}

/// The 8-bit value the overlay framebuffer shows for a linear value: `round(255 · v^(1/2.2))`.
pub(crate) fn display_byte(linear: f64) -> u8 {
    (255.0 * linear.clamp(0.0, 1.0).powf(1.0 / OVERLAY_GAMMA))
        .round()
        .clamp(0.0, 255.0) as u8
}

/// The viewport background as displayed: the theme interior colour, opaque.
pub(crate) fn background_rgba() -> [u8; 4] {
    let (_, interior) = theme_linear();
    let value = display_byte(interior[0]);
    [value, value, value, 255]
}

/// The colour of a point's pixel at `fac` (`overlay_wireframe_frag.glsl`, POINTS): the rim and interior
/// are taken to their square roots, mixed by 0.35 for the front colour, mixed with the rim by
/// `fac = 4 · dist²` (dist is the distance to the centre in units of the point size), and squared.
pub(crate) fn point_colour(fac: f64, rim: [f64; 3], interior: [f64; 3]) -> [f64; 3] {
    let fac = fac.clamp(0.0, 1.0);
    std::array::from_fn(|channel| {
        let rim_root = rim[channel].sqrt();
        let interior_root = interior[channel].sqrt();
        let front = rim_root + (interior_root - rim_root) * 0.35;
        let mixed = front + (rim_root - front) * fac;
        mixed * mixed
    })
}

/// A point square on the screen: its centre (continuous pixel coordinates) and its side in pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PointSquare {
    pub(crate) centre_x: f64,
    pub(crate) centre_y: f64,
    pub(crate) side: f64,
}

/// A projected point: pixel coordinates (row 0 at the top) and its depth, which orders lines.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScreenPoint {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) depth: f64,
}

/// A projected line with its linear colours at each end.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WireSegment {
    pub(crate) a: ScreenPoint,
    pub(crate) b: ScreenPoint,
    pub(crate) colour_a: [f64; 3],
    pub(crate) colour_b: [f64; 3],
}

/// The line record of one pixel: the last-written fragment of the nearest line (`pack_line_data`).
#[derive(Clone, Copy)]
struct Record {
    colour: [f64; 3],
    /// Signed distance from the pixel centre to the line, in pixels.
    distance: f64,
    /// Unit normal of the line (`perp`).
    direction: [f64; 2],
    /// Window depth, in [0, 1]; the cleared buffer is 1.
    depth: f64,
}

/// Rasterises the segments and returns each pixel's linear colour, background included.
pub(crate) fn composite(
    segments: &[WireSegment],
    points: &[PointSquare],
    width: usize,
    height: usize,
) -> Vec<[f64; 3]> {
    let (rim, background) = theme_linear();
    let mut records: Vec<Option<Record>> = vec![None; width * height];
    for segment in segments {
        draw_segment(&mut records, width, height, segment);
    }
    let mut out = vec![background; width * height];
    for (index, pixel) in out.iter_mut().enumerate() {
        let (x, y) = (index % width, index / width);
        let neighbours = neighbour_records(&records, width, height, x, y);
        if records[index].is_none() && neighbours.iter().all(Option::is_none) {
            continue;
        }
        *pixel = anti_alias(records[index].as_ref(), &neighbours, background);
    }
    // Points are opaque squares written over the lines (no line records, so no anti-aliasing).
    for point in points {
        let half = point.side * 0.5;
        let min_x = (point.centre_x - half).floor().max(0.0) as usize;
        let max_x = ((point.centre_x + half).ceil().max(0.0) as usize).min(width.saturating_sub(1));
        let min_y = (point.centre_y - half).floor().max(0.0) as usize;
        let max_y =
            ((point.centre_y + half).ceil().max(0.0) as usize).min(height.saturating_sub(1));
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let (dx, dy) = (
                    px as f64 + 0.5 - point.centre_x,
                    py as f64 + 0.5 - point.centre_y,
                );
                if dx.abs() > half || dy.abs() > half {
                    continue;
                }
                let dist = dx.abs().max(dy.abs()) / point.side;
                out[py * width + px] = point_colour(4.0 * dist * dist, rim, background);
            }
        }
    }
    out
}

fn draw_segment(
    records: &mut [Option<Record>],
    width: usize,
    height: usize,
    segment: &WireSegment,
) {
    let (ax, ay, bx, by) = (segment.a.x, segment.a.y, segment.b.x, segment.b.y);
    let (dx, dy) = (bx - ax, by - ay);
    let length = dx.hypot(dy);
    if !length.is_finite() || length <= 0.0 {
        return;
    }
    let (ux, uy) = (dx / length, dy / length);
    let perp = [-uy, ux];
    let major = ux.abs().max(uy.abs());
    let min_x = (ax.min(bx).floor() - 2.0).max(0.0) as usize;
    let max_x = ((ax.max(bx).ceil() + 2.0).max(0.0) as usize).min(width.saturating_sub(1));
    let min_y = (ay.min(by).floor() - 2.0).max(0.0) as usize;
    let max_y = ((ay.max(by).ceil() + 2.0).max(0.0) as usize).min(height.saturating_sub(1));
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let (rx, ry) = (px as f64 + 0.5 - ax, py as f64 + 0.5 - ay);
            let t = rx * ux + ry * uy;
            if !(-LINE_HALF_WIDTH..=length + LINE_HALF_WIDTH).contains(&t) {
                continue;
            }
            let distance = rx * perp[0] + ry * perp[1];
            // One fragment per major-axis step: the band is half a pixel measured along the major axis
            // (`-0.5 * major <= d < 0.5 * major`), half-open at the tie. Measured on the native line
            // rasteriser of the capture's Metal GPU, it reproduces isolated lines at every angle to 2%.
            if !(-LINE_HALF_WIDTH * major..LINE_HALF_WIDTH * major).contains(&distance) {
                continue;
            }
            let along = (t / length).clamp(0.0, 1.0);
            let colour: [f64; 3] = std::array::from_fn(|channel| {
                segment.colour_a[channel]
                    + (segment.colour_b[channel] - segment.colour_a[channel]) * along
            });
            let depth = segment.a.depth + (segment.b.depth - segment.a.depth) * along;
            // The depth test is LESS_EQUAL against the cleared buffer: a nearer line replaces the
            // record, and an equal depth goes to the later fragment.
            let index = py * width + px;
            if let Some(current) = records[index]
                && depth > current.depth
            {
                continue;
            }
            records[index] = Some(Record {
                colour,
                distance,
                direction: perp,
                depth,
            });
        }
    }
}

/// The records of the 4-neighbours in the order `+x, -x, +y, -y`, when they exist.
fn neighbour_records(
    records: &[Option<Record>],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> [Option<(Record, i32, i32)>; 4] {
    [(1, 0), (-1, 0), (0, 1), (0, -1)].map(|(ox, oy): (i32, i32)| {
        let nx = x as i64 + i64::from(ox);
        let ny = y as i64 + i64::from(oy);
        if !(0..width as i64).contains(&nx) || !(0..height as i64).contains(&ny) {
            return None;
        }
        records[ny as usize * width + nx as usize].map(|record| (record, ox, oy))
    })
}

/// The overlay AA of one pixel (`overlay_antialiasing.bsl.hh`, `frag_main`): the centre's coverage, then
/// each neighbour whose line runs across its offset, blended by depth, then the dense-line fix, then
/// premultiplied over the background.
fn anti_alias(
    centre: Option<&Record>,
    neighbours: &[Option<(Record, i32, i32)>; 4],
    background: [f64; 3],
) -> [f64; 3] {
    let (mut colour, mut alpha, mut depth) = ([0.0_f64; 3], 0.0_f64, 1.0_f64);
    if let Some(record) = centre {
        let coverage = line_coverage(record.distance);
        colour = record.colour.map(|value| value * coverage);
        alpha = coverage;
        depth = record.depth;
    }
    for (neighbour, ox, oy) in neighbours.iter().flatten() {
        // A line only reaches an offset it runs across: a horizontal offset needs a near-vertical line.
        let dir_horizontal = neighbour.direction[0].abs() > neighbour.direction[1].abs();
        let offset_horizontal = *ox != 0;
        if dir_horizontal != offset_horizontal {
            continue;
        }
        let distance = neighbour.distance
            + (f64::from(*ox) * -neighbour.direction[0] + f64::from(*oy) * -neighbour.direction[1]);
        let coverage = line_coverage(distance);
        if coverage <= 0.0 {
            continue;
        }
        let neighbour_colour = neighbour.colour.map(|value| value * coverage);
        if neighbour.depth < depth {
            // Alpha over: neighbour in front of the centre's colour.
            colour = std::array::from_fn(|channel| {
                neighbour_colour[channel] + colour[channel] * (1.0 - coverage)
            });
            alpha = coverage + alpha * (1.0 - coverage);
            depth = neighbour.depth;
        } else {
            // Alpha under: the centre's colour in front of the neighbour's.
            colour = std::array::from_fn(|channel| {
                colour[channel] + neighbour_colour[channel] * (1.0 - alpha)
            });
            alpha += coverage * (1.0 - alpha);
        }
    }
    // Dense-line fix: with the centre a line, the un-premultiplied colour is blended in as the count of
    // line neighbours grows past two (`lines_raw`, `blend`).
    if centre.is_some() && alpha > 0.0 {
        let count = neighbours.iter().flatten().count() as f64;
        let blend = (0.25 * count * 2.0 - 1.0).clamp(0.0, 1.0);
        colour = std::array::from_fn(|channel| {
            let unpremultiplied = colour[channel] / alpha;
            colour[channel] + (unpremultiplied - colour[channel]) * blend
        });
    }
    std::array::from_fn(|channel| colour[channel] + background[channel] * (1.0 - alpha))
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    #[test]
    fn background_is_the_theme_interior_as_displayed() {
        // The overlay shows the theme byte 0x3d as 63 (measured on the Blender 5.2.2 capture).
        assert_eq!(background_rgba(), [63, 63, 63, 255]);
    }

    #[test]
    fn display_follows_the_measured_gamma_table() {
        // Background samples, theme value to displayed byte, from the Blender 5.2.2 capture.
        for (theme, shown) in [(0.25, 66), (0.5, 127), (0.9, 229), (1.0, 255), (0.0, 0)] {
            assert_eq!(display_byte(srgb_to_linear(theme)), shown, "theme {theme}");
        }
    }

    #[test]
    fn line_coverage_matches_the_smooth_kernel() {
        assert!((line_coverage(0.0) - 0.98).abs() < 0.01);
        assert!(line_coverage(LINE_SMOOTH_END).abs() < 1e-9);
    }

    #[test]
    fn edge_factor_and_visibility_follow_the_cosine_band() {
        let up = DVec3::Z;
        assert!(
            edge_factor(&[up]).abs() < 1e-12,
            "boundary edges are always visible"
        );
        // Right angle: cos 0 gives 0, flat edges give the top of the band.
        assert!(edge_factor(&[up, DVec3::X]).abs() < 1e-12);
        assert!((edge_factor(&[up, up]) - WD_STEP).abs() < 1e-12);
        assert!(edge_visible(WD_STEP, WIRE_THRESHOLD));
        // A threshold of 0.5 hides nearly flat edges (wd above sqrt(0.5) * 254/255).
        assert!(!edge_visible(WD_STEP, 0.5));
        assert!(edge_visible(0.0, 0.5));
    }

    #[test]
    fn polygon_normal_of_a_unit_quad_is_up() {
        let quad = [
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.0, 1.0, 1.0),
        ];
        assert!((polygon_normal(&quad) - DVec3::Z).length() < 1e-12);
    }

    #[test]
    fn builder_keeps_each_edge_once_with_its_first_face_normals() {
        let positions = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let mut builder = WireEdgeBuilder::default();
        builder.add_polygon(&[0, 1, 2, 3], &positions, None);
        builder.add_polygon(&[1, 0, 4, 5], &positions, None);
        let edges = builder.finish();
        // Four edges of the first quad, plus three new ones from the second (edge 0-1 is shared).
        assert_eq!(edges.len(), 7);
        let first = edges[0].normals.map_or(DVec3::ZERO, |normals| normals[0]);
        assert!((first - DVec3::Z).length() < 1e-12);
    }

    #[test]
    fn point_rim_is_black_and_the_centre_is_the_interior_front_colour() {
        let (rim, interior) = theme_linear();
        // A point's rim (fac 1) is the theme wire colour, black; its centre is the front colour, a little
        // lighter than the interior (the capture shows 19 to 23 across a point's interior).
        assert_eq!(display_byte(point_colour(1.0, rim, interior)[0]), 0);
        assert!(display_byte(point_colour(0.0, rim, interior)[0]) > 15);
    }

    #[test]
    fn facing_colour_is_black_edge_on_and_front_colour_facing_the_viewer() {
        let (rim, interior) = theme_linear();
        let edge_on = facing_colour(0.0, rim, interior);
        assert!(edge_on.iter().all(|value| value.abs() < 1e-12));
        let front = facing_colour(1.0, rim, interior);
        // Rim black, interior 0x3d: front is 0.35 of the interior in gamma space, about 22/255 displayed.
        assert_eq!(display_byte(front[0]), 22);
    }
}

/// Oracle checks against the Blender 5.2.2 viewport captures in `tests/fixtures/workbench_wireframe`.
///
/// What the model provably reproduces, and what it does not:
/// - Background: every pixel farther than 2.5 px from any line or point matches the capture within one
///   level (8-bit rounding of the theme value; the overlay's gamma display rounds one level differently
///   at a few theme values).
/// - View transform: the Standard and `AgX` captures of the same scene are byte-identical, so overlays are
///   not view-transformed, and Wire output is not either.
/// - Points: the interior of each loose-vertex square (within 0.25 of its side from the centre) matches
///   within one level (`overlay_wireframe_frag.glsl`, POINTS).
/// - Curves: pixels on a curve's centre line match within one level, except one pixel at 9 levels.
/// - Cube line centres (pixels within 0.3 px of a cube edge, x < 400): 95 pixels; 93 match within three
///   levels. The two outliers are at the corner where the cube's vertical edge meets the top edge, a line
///   end where GPU line rasterisation is not reproduced.
///
/// Known gaps, with evidence (also in CHANGELOG and docs/spec.md):
/// 1. AA fringe ink on Apple Metal lines. Coincident lines add ink in the capture: a front/back pair adds
///    7.4% ink per unit length over a single edge, and an identical-depth duplicate pair adds 8.2%. The
///    model, which keeps the nearest line per pixel, adds none. The wire pass has no blend state and
///    writes colour and line record from the same fragment (`overlay_wireframe.hh:79`,
///    `overlay_wireframe_frag.glsl:37–41`), so the source does not explain it. The cube's ink is 22% below
///    the capture, and the sphere's 11% below. These are not asserted.
/// 2. Sphere line centres. Of 1132 pixels within 0.3 px of a sphere edge, 430 differ from the capture by
///    more than three levels (up to 46): the facing colour of shared edges. No choice of adjacent face
///    (first, last, most or least facing) fixes it, so the facing rule for shared edges is not known. Not
///    asserted.
/// 3. Point-cloud outline ring. Point-cloud dots at radii 0.05 to 0.2 show the vertex-sized square plus a
///    partial outline whose radius follows no rule that fits. The square alone gives a lower error than the
///    best ring fit (0.84 against 1.24). Dots are drawn as the square; the ring is not modelled. Point-cloud
///    and stroke colours are black (the theme wire colour), not measured.
#[cfg(test)]
mod oracle_tests {
    use std::{
        error::Error,
        fs,
        io::BufReader,
        path::{Path, PathBuf},
    };

    use glam::DVec3;
    use serde_json::Value;

    use crate::render::raster::{Camera, Projection, StereoMode, rasterize_wireframe};

    use super::*;

    type TestResult<T = ()> = Result<T, Box<dyn Error>>;

    /// Ortho scale of the capture's camera: the window matrix's half width is 8.6399 (landscape sensor fit).
    const ORTHO_SCALE: f64 = 2.0 * 8.6399;
    const WIDTH: u32 = 818;
    const HEIGHT: u32 = 695;
    /// The object region of the captures. The viewport corners carry UI arrows, which are not compared.
    const OBJECT_REGION: (std::ops::Range<usize>, std::ops::Range<usize>) = (120..700, 120..600);
    /// The whole frame, for scenes whose objects sit outside the object region (the point-cloud scene's curve).
    const FULL_FRAME: (std::ops::Range<usize>, std::ops::Range<usize>) = (0..818, 0..695);

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/workbench_wireframe")
            .join(name)
    }

    fn point_value(value: &Value) -> TestResult<DVec3> {
        let coords: Vec<f64> = value
            .as_array()
            .ok_or("point is not an array")?
            .iter()
            .map(|coordinate| coordinate.as_f64().ok_or("coordinate is not a number"))
            .collect::<std::result::Result<_, _>>()?;
        match coords.as_slice() {
            &[x, y, z] => Ok(DVec3::new(x, y, z)),
            _ => Err("point must have three coordinates".into()),
        }
    }

    /// The fixture's mesh edges: world endpoints, the first face's corner normals, and `wd`.
    fn cube_sphere_edges() -> TestResult<Vec<WireEdge>> {
        let records: Vec<Value> =
            serde_json::from_str(&fs::read_to_string(fixture("wire_edges_cube_sphere.json"))?)?;
        records
            .iter()
            .map(|record| {
                let normals = record["normals"]
                    .as_array()
                    .ok_or("normals missing")?
                    .iter()
                    .map(point_value)
                    .collect::<TestResult<Vec<_>>>()?;
                let [normal_a, normal_b] = normals.as_slice() else {
                    return Err::<WireEdge, Box<dyn Error>>("two normals per edge".into());
                };
                Ok(WireEdge {
                    start: point_value(&record["start"])?,
                    end: point_value(&record["end"])?,
                    normals: Some([*normal_a, *normal_b]),
                    colour: [0.0; 3],
                    wd: record["wd"].as_f64().ok_or("wd missing")?,
                })
            })
            .collect()
    }

    /// The capture's RGB pixels, row 0 at the top, and its size.
    fn load_capture(name: &str) -> TestResult<Vec<[u8; 3]>> {
        let mut reader =
            png::Decoder::new(BufReader::new(fs::File::open(fixture(name))?)).read_info()?;
        let mut buffer = vec![
            0;
            reader
                .output_buffer_size()
                .ok_or("PNG output buffer size is unknown")?
        ];
        let info = reader.next_frame(&mut buffer)?;
        if (info.width, info.height) != (WIDTH, HEIGHT) {
            return Err(format!("{name} is {}x{}", info.width, info.height).into());
        }
        let channels = info.color_type.samples();
        Ok(buffer
            .chunks_exact(channels)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect())
    }

    fn camera() -> Camera {
        Camera {
            position: DVec3::new(0.0, -12.0, 0.0),
            target: DVec3::ZERO,
            up: DVec3::Z,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic {
                height: ORTHO_SCALE,
            },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        }
    }

    /// Projects world points with the capture's own matrices (window times view), independently of
    /// Potter's projection, into continuous pixel coordinates with row 0 at the top.
    fn capture_projector(name: &str) -> TestResult<impl Fn(DVec3) -> [f64; 2]> {
        let json: Value = serde_json::from_str(&fs::read_to_string(fixture(name))?)?;
        let matrix = |key: &str| -> TestResult<[[f64; 4]; 4]> {
            let rows = json[key].as_array().ok_or("matrix missing")?;
            let mut out = [[0.0; 4]; 4];
            for (r, row) in rows.iter().enumerate() {
                for (c, value) in row.as_array().ok_or("row missing")?.iter().enumerate() {
                    out[r][c] = value.as_f64().ok_or("entry missing")?;
                }
            }
            Ok(out)
        };
        let window = matrix("window_matrix")?;
        let view = matrix("view_matrix")?;
        let product: [[f64; 4]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|c| (0..4).map(|k| window[r][k] * view[k][c]).sum())
        });
        let width = WIDTH as f64;
        let height = HEIGHT as f64;
        Ok(move |p: DVec3| {
            let v = [p.x, p.y, p.z, 1.0];
            let clip: [f64; 4] =
                std::array::from_fn(|r| (0..4).map(|k| product[r][k] * v[k]).sum());
            let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
            [
                f64::midpoint(x, 1.0) * width,
                f64::midpoint(1.0, -y) * height,
            ]
        })
    }

    fn segment_distance(px: f64, py: f64, a: [f64; 2], b: [f64; 2]) -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let length_squared = dx * dx + dy * dy;
        let t = if length_squared == 0.0 {
            0.0
        } else {
            (((px - a[0]) * dx + (py - a[1]) * dy) / length_squared).clamp(0.0, 1.0)
        };
        (px - (a[0] + t * dx)).hypot(py - (a[1] + t * dy))
    }

    /// Distance from every pixel centre to the nearest segment, computed inside each segment's box.
    fn line_distance(segments: &[([f64; 2], [f64; 2])]) -> Vec<f64> {
        let (width, height) = (WIDTH as usize, HEIGHT as usize);
        let mut out = vec![f64::MAX; width * height];
        for &(a, b) in segments {
            let reach = 4.0;
            let x0 = (a[0].min(b[0]) - reach).floor().max(0.0) as usize;
            let x1 = ((a[0].max(b[0]) + reach).ceil().max(0.0) as usize).min(width - 1);
            let y0 = (a[1].min(b[1]) - reach).floor().max(0.0) as usize;
            let y1 = ((a[1].max(b[1]) + reach).ceil().max(0.0) as usize).min(height - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = segment_distance(x as f64 + 0.5, y as f64 + 0.5, a, b);
                    let slot = &mut out[y * width + x];
                    *slot = slot.min(d);
                }
            }
        }
        out
    }

    /// Distance from every pixel centre to the nearest point centre.
    fn point_distance(centres: &[[f64; 2]]) -> Vec<f64> {
        let (width, height) = (WIDTH as usize, HEIGHT as usize);
        let mut out = vec![f64::MAX; width * height];
        for centre in centres {
            let reach = 10.0;
            let x0 = (centre[0] - reach).floor().max(0.0) as usize;
            let x1 = ((centre[0] + reach).ceil().max(0.0) as usize).min(width - 1);
            let y0 = (centre[1] - reach).floor().max(0.0) as usize;
            let y1 = ((centre[1] + reach).ceil().max(0.0) as usize).min(height - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = (x as f64 + 0.5 - centre[0]).hypot(y as f64 + 0.5 - centre[1]);
                    let slot = &mut out[y * width + x];
                    *slot = slot.min(d);
                }
            }
        }
        out
    }

    /// Per-channel largest difference between the model's pixel and the capture's, over the region.
    fn errors_where(
        rgba: &[u8],
        capture: &[[u8; 3]],
        region: (std::ops::Range<usize>, std::ops::Range<usize>),
        keep: impl Fn(usize, usize) -> bool,
    ) -> Vec<u8> {
        let width = WIDTH as usize;
        let mut out = Vec::new();
        for y in region.1 {
            for x in region.0.clone() {
                if !keep(x, y) {
                    continue;
                }
                let index = y * width + x;
                let model = [rgba[index * 4], rgba[index * 4 + 1], rgba[index * 4 + 2]];
                let want = capture[index];
                out.push(
                    model
                        .iter()
                        .zip(want)
                        .map(|(a, b)| a.abs_diff(b))
                        .max()
                        .unwrap_or(0),
                );
            }
        }
        out
    }

    fn segments_of(
        edges: &[WireEdge],
        project: &impl Fn(DVec3) -> [f64; 2],
    ) -> Vec<([f64; 2], [f64; 2])> {
        edges
            .iter()
            .map(|edge| (project(edge.start), project(edge.end)))
            .filter(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() > 1e-9)
            .collect()
    }

    #[test]
    fn overlay_is_not_view_transformed() -> TestResult {
        // The Standard and AgX captures of the same scene are byte-identical.
        assert_eq!(
            load_capture("oracle_cube_sphere.png")?,
            load_capture("oracle_cube_sphere_agx.png")?
        );
        Ok(())
    }

    #[test]
    fn background_matches_the_blender_capture() -> TestResult {
        let edges = cube_sphere_edges()?;
        let project = capture_projector("oracle_cube_sphere.json")?;
        let capture = load_capture("oracle_cube_sphere.png")?;
        let output = rasterize_wireframe(&edges, &[], &camera(), WIDTH, HEIGHT)?;
        let distance = line_distance(&segments_of(&edges, &project));
        let errors = errors_where(&output.rgba, &capture, OBJECT_REGION, |x, y| {
            distance[y * WIDTH as usize + x] > 2.5
        });
        let count = errors.len();
        assert_ne!(count, 0);
        assert!(
            errors.iter().all(|&error| error <= 1),
            "background error {:?}",
            errors.iter().max()
        );
        Ok(())
    }

    #[test]
    fn cube_line_centres_match_the_blender_capture() -> TestResult {
        let edges = cube_sphere_edges()?;
        let project = capture_projector("oracle_cube_sphere.json")?;
        let capture = load_capture("oracle_cube_sphere.png")?;
        let output = rasterize_wireframe(&edges, &[], &camera(), WIDTH, HEIGHT)?;
        let cube_segments: Vec<_> = segments_of(&edges, &project)
            .into_iter()
            .filter(|(a, b)| a[0] < 400.0 && b[0] < 400.0)
            .collect();
        let distance = line_distance(&cube_segments);
        let errors = errors_where(&output.rgba, &capture, OBJECT_REGION, |x, y| {
            x < 400 && distance[y * WIDTH as usize + x] <= 0.3
        });
        assert!(
            errors.len() >= 90,
            "too few cube centre pixels: {}",
            errors.len()
        );
        // All but two corner pixels match within three levels; no pixel exceeds 23.
        assert!(
            errors.iter().filter(|&&error| error > 3).count() <= 2,
            "cube centres {errors:?}"
        );
        assert!(errors.iter().all(|&error| error <= 23));
        Ok(())
    }

    #[test]
    fn points_and_curves_match_the_blender_capture() -> TestResult {
        let project = capture_projector("points_curves/pc.json")?;
        let capture = load_capture("points_curves/pc.png")?;
        // The curve of the point-cloud scene: (-1.5, 0, 5.5) to (1.5, 0, 5.5) to (1.5, 0, 6.5), black.
        let curve_points = [
            DVec3::new(-1.5, 0.0, 5.5),
            DVec3::new(1.5, 0.0, 5.5),
            DVec3::new(1.5, 0.0, 6.5),
        ];
        let edges: Vec<WireEdge> = curve_points
            .windows(2)
            .map(|pair| WireEdge {
                start: pair[0],
                end: pair[1],
                normals: None,
                colour: [0.0; 3],
                wd: 0.0,
            })
            .collect();
        // Four point-cloud points at z = 3.5, with radii 0.05 to 0.2 (drawn as squares of 2 x vert).
        let points: Vec<DVec3> = [-1.5, -0.5, 0.5, 1.5]
            .iter()
            .map(|&x| DVec3::new(x, 0.0, 3.5))
            .collect();
        let output = rasterize_wireframe(&edges, &points, &camera(), WIDTH, HEIGHT)?;
        let segments = segments_of(&edges, &project);
        let line_d = line_distance(&segments);
        let centres: Vec<[f64; 2]> = points.iter().map(|&point| project(point)).collect();
        let point_d = point_distance(&centres);
        let side = 2.0 * vertex_size();
        let interior = errors_where(&output.rgba, &capture, FULL_FRAME, |x, y| {
            point_d[y * WIDTH as usize + x] <= 0.25 * side
        });
        let count = interior.len();
        assert_ne!(count, 0);
        assert!(
            interior.iter().all(|&error| error <= 1),
            "point interiors {interior:?}"
        );
        let curve = errors_where(&output.rgba, &capture, FULL_FRAME, |x, y| {
            line_d[y * WIDTH as usize + x] <= 0.5
        });
        assert!(curve.len() >= 150, "too few curve pixels: {}", curve.len());
        assert!(
            curve.iter().filter(|&&error| error > 1).count() <= 2,
            "curve centres {curve:?}"
        );
        assert!(curve.iter().all(|&error| error <= 9));
        Ok(())
    }

    #[test]
    fn edge_factor_of_a_manifold_pair_follows_the_cosine_band() {
        // A flat pair (cos 1) keeps the top of the band; a right angle (cos 0) is 0.
        assert!((edge_factor(&[DVec3::Z, DVec3::Z]) - WD_STEP).abs() < 1e-12);
        assert!(edge_factor(&[DVec3::Z, DVec3::X]).abs() < 1e-12);
    }
}
