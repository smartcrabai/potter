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
pub(crate) fn composite(segments: &[WireSegment], width: usize, height: usize) -> Vec<[f64; 3]> {
    let (_, background) = theme_linear();
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
        *pixel = anti_alias(&records[index], &neighbours, background);
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
    centre: &Option<Record>,
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
        let first = edges[0]
            .normals
            .map(|normals| normals[0])
            .unwrap_or(DVec3::ZERO);
        assert!((first - DVec3::Z).length() < 1e-12);
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

/// The Blender 5.2.2 viewport oracle of the cube and sphere (`tests/fixtures/workbench_wireframe`):
/// the model's per-pixel result is compared with the capture, by pixel class.
///
/// What the model gets right, and what it does not:
/// - background: pixels the model leaves at the theme colour match the capture within 4 levels; 11 of
///   about 194 thousand sit beside the sphere's silhouette, where the capture has a faint line the model
///   does not draw;
/// - cube edges: the cube's lines match the capture within 9 levels except for 2 pixels at a corner,
///   where a line's end is off by a fraction of a pixel (the fixed GPU line rule is not reproduced);
/// - sphere edges: the mean error is 2.4 levels, and the ink (the darkness the lines add) is about 9%
///   below the capture. Overlapping curved lines are where the GPU's line rasteriser and this model
///   disagree most, and the cause is not known: the candidates tried (draw order, facing choice, depth
///   ordering, the dense-line fix, the inclusion width, the coverage kernel, sub-pixel projection) do
///   not close the ink gap. The ink bound below pins the current deficit so a change is noticed.
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

    /// Ortho scale of the capture's camera: its half width is 8.6399 (window matrix `m00` of the
    /// oracle, landscape sensor fit), so the scale is twice that.
    const ORTHO_SCALE: f64 = 2.0 * 8.6399;
    const WIDTH: u32 = 818;
    const HEIGHT: u32 = 695;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/workbench_wireframe")
            .join(name)
    }

    fn point(value: &Value) -> TestResult<DVec3> {
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

    /// The fixture's edges: world endpoints, the first face's corner normals, and `wd`.
    fn load_edges() -> TestResult<Vec<WireEdge>> {
        let records: Vec<Value> =
            serde_json::from_str(&fs::read_to_string(fixture("wire_edges_cube_sphere.json"))?)?;
        records
            .iter()
            .map(|record| {
                let normals = record["normals"]
                    .as_array()
                    .ok_or("normals missing")?
                    .iter()
                    .map(point)
                    .collect::<TestResult<Vec<_>>>()?;
                let [normal_a, normal_b] = normals.as_slice() else {
                    return Err::<WireEdge, Box<dyn Error>>("two normals per edge".into());
                };
                Ok(WireEdge {
                    start: point(&record["start"])?,
                    end: point(&record["end"])?,
                    normals: Some([*normal_a, *normal_b]),
                    colour: [0.0; 3],
                    wd: record["wd"].as_f64().ok_or("wd missing")?,
                })
            })
            .collect()
    }

    /// The capture's RGB pixels, row 0 at the top.
    fn load_capture() -> TestResult<Vec<[u8; 3]>> {
        let mut reader = png::Decoder::new(BufReader::new(fs::File::open(fixture(
            "oracle_cube_sphere.png",
        ))?))
        .read_info()?;
        let mut buffer = vec![
            0;
            reader
                .output_buffer_size()
                .ok_or("PNG output buffer size is unknown")?
        ];
        let info = reader.next_frame(&mut buffer)?;
        assert_eq!((info.width, info.height), (WIDTH, HEIGHT));
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

    /// Per-class error statistics, the classes as the model's own pixels define them.
    #[derive(Default, Debug)]
    struct Stats {
        count: usize,
        sum: u64,
        max: u8,
        over_nine: usize,
    }

    impl Stats {
        fn add(&mut self, error: u8) {
            self.count += 1;
            self.sum += u64::from(error);
            self.max = self.max.max(error);
            if error > 9 {
                self.over_nine += 1;
            }
        }

        fn mean(&self) -> f64 {
            self.sum as f64 / self.count.max(1) as f64
        }
    }

    #[test]
    fn wireframe_matches_the_blender_viewport_capture() -> TestResult {
        let edges = load_edges()?;
        let capture = load_capture()?;
        let output = rasterize_wireframe(&edges, &camera(), WIDTH, HEIGHT)?;
        let width = WIDTH as usize;
        let height = HEIGHT as usize;
        let model = |index: usize| -> [u8; 3] {
            [
                output.rgba[index * 4],
                output.rgba[index * 4 + 1],
                output.rgba[index * 4 + 2],
            ]
        };
        let background = background_rgba();
        let background = [background[0], background[1], background[2]];
        let is_background = |x: usize, y: usize| -> bool {
            let index = y * width + x;
            if model(index) != background {
                return false;
            }
            [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                .iter()
                .all(|&(nx, ny)| {
                    nx >= width || ny >= height || model(ny * width + nx) == background
                })
        };
        let mut interior = Stats::default();
        let mut interior_beyond_one = 0;
        let mut cube = Stats::default();
        let mut sphere = Stats::default();
        let mut ink_model = [0_i64; 2];
        let mut ink_capture = [0_i64; 2];
        for y in 150..550 {
            for x in 150..650 {
                let index = y * width + x;
                let want = capture[index];
                let side = usize::from(x >= 400);
                ink_model[side] += background.iter().map(|&v| i64::from(v)).sum::<i64>()
                    - model(index).iter().map(|&v| i64::from(v)).sum::<i64>();
                ink_capture[side] += background.iter().map(|&v| i64::from(v)).sum::<i64>()
                    - want.iter().map(|&v| i64::from(v)).sum::<i64>();
                let error = model(index)
                    .iter()
                    .zip(want)
                    .map(|(a, b)| a.abs_diff(b))
                    .max()
                    .unwrap_or(0);
                if is_background(x, y) {
                    let deviation = want
                        .iter()
                        .zip(background)
                        .map(|(a, b)| a.abs_diff(b))
                        .max()
                        .unwrap_or(0);
                    interior.add(deviation);
                    if deviation > 1 {
                        interior_beyond_one += 1;
                    }
                } else if x < 400 {
                    cube.add(error);
                } else {
                    sphere.add(error);
                }
            }
        }
        // Background: exact up to 4 levels; 11 pixels beside the silhouette carry a faint line.
        assert!(interior.max <= 4, "interior {interior:?}");
        assert!(
            interior_beyond_one <= 11,
            "interior beyond one: {interior_beyond_one}"
        );
        // Cube edges: mean 3.0, within 9 levels except two corner pixels that reach 23.
        assert!(cube.mean() <= 3.5, "cube {cube:?}");
        assert!(cube.over_nine <= 2 && cube.max <= 23, "cube {cube:?}");
        // Sphere edges: mean 2.4 measured; bounded with a margin.
        assert!(sphere.mean() <= 3.0, "sphere {sphere:?}");
        // Ink: the model adds about 9% less darkness on the sphere and 22% less on the cube than the
        // capture (see the module note). Pinned, so that a change to the line model shows up here.
        let ratio = |side: usize| {
            100.0 * (ink_model[side] - ink_capture[side]) as f64 / ink_capture[side].max(1) as f64
        };
        assert!((ratio(1) + 8.9).abs() <= 1.0, "sphere ink {:.2}%", ratio(1));
        assert!((ratio(0) + 21.7).abs() <= 2.0, "cube ink {:.2}%", ratio(0));
        Ok(())
    }

    #[test]
    fn edge_factor_of_a_manifold_pair_follows_the_cosine_band() -> TestResult {
        // A flat pair (cos 1) keeps the top of the band; a right angle (cos 0) is 0.
        assert!((edge_factor(&[DVec3::Z, DVec3::Z]) - WD_STEP).abs() < 1e-12);
        assert!(edge_factor(&[DVec3::Z, DVec3::X]).abs() < 1e-12);
        Ok(())
    }
}
