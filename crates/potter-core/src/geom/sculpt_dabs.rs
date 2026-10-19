//! Blender 5.2.2 dab brushes: one framework for every sculpt brush that Potter applies per dab.
//!
//! Blender applies one `do_brush_action` per mouse event of a DOTS stroke. Each sample here is one
//! such dab at its `position`, applied in order to the positions the previous dabs left behind.
//! Vertex normals are the stroke-start normals for every dab: Blender's pbvh normals are not
//! refreshed between dabs of one stroke, which the Inflate and Layer oracles confirm. Area normals
//! and centers read the current positions only for brushes that accumulate (`BRUSH_ACCUMULATE`);
//! otherwise they read the stroke-start positions, as Blender's `orig_position_data_lookup` does.
//! Mirror symmetry runs each dab once per pass, in Blender's order, against the positions the
//! earlier passes left behind. The strength of every pass is scaled by Blender's symmetry feather.
//!
//! Potter's stroke has no brush settings, so the Blender 5.2.2 essentials defaults for each brush
//! are fixed here: the constants in the family modules, the strength and pressure a sample carries,
//! and the falloff the caller selects.
//!
//! Blender sources: `mesh/brushes/*.cc` for each brush; `sculpt.cc` `brush_strength` (2329),
//! `calc_area_normal_and_center` (2185), `calc_brush_plane` (3048), `calc_brush_cube_distances`
//! (7503), `apply_hardness_to_distances` (7543); `brush.cc` `BKE_brush_calc_curve_factors` (1478).

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};

use super::sculpt::{DabBrush, Falloff, Stroke, StrokeSample};
use crate::geom::Mesh;

mod crease_pinch;
mod elastic;
mod grab;
mod offset;
mod plane;
mod smooth;

/// Clay Strips `plane_offset` default: the plane sits this many radii above the area center.
const CLAY_STRIPS_PLANE_OFFSET: f64 = 0.15;
/// Clay Strips `tip_roundness` default: the rounded-square tip's corner radius.
const CLAY_STRIPS_TIP_ROUNDNESS: f64 = 0.15;
/// `normal_radius_factor` default of Clay Strips (asset value 1.2). It sets both the area normal
/// radius and the area center radius (`area_normal_and_center_get_position_radius`, non-plane
/// branch).
const CLAY_STRIPS_AREA_RADIUS_FACTOR: f64 = 1.2;
/// Layer `height` default: the displacement factor scales this along the stroke-start normal.
const LAYER_HEIGHT: f64 = 0.05;
/// Layer `hardness` default.
const LAYER_HARDNESS: f64 = 0.35;

/// One stroke's per-vertex state, shared by every dab brush. Vectors are indexed in
/// `mesh.vertices` order.
struct Field {
    /// Current positions. Each dab moves them in place (Blender's pbvh positions).
    positions: Vec<DVec3>,
    /// Stroke-start positions. Falloff of Layer and the area of non-accumulating brushes use them.
    original: Vec<DVec3>,
    /// Stroke-start vertex normals (`normals_calc_verts`).
    normals: Vec<DVec3>,
    /// `1 - mask` per vertex (Blender's `fill_factor_from_hide_and_mask`).
    mask_factors: Vec<f64>,
    /// Blender's interior neighbors per vertex (`vertex_neighbors`), sorted; empty when Blender has none.
    neighbors: Vec<Vec<usize>>,
    /// Layer's per-vertex displacement factor, accumulated across the stroke.
    displacement: Vec<f64>,
    falloff: Falloff,
    /// Blender's symmetry passes as per-axis sign vectors; pass 0 is the unflipped brush.
    passes: Vec<DVec3>,
    /// Unit vector toward the viewer. Blender's area normal groups normals by it, and the grab
    /// delta lies on the plane perpendicular to it. `None` means Potter has no view.
    view: Option<DVec3>,
    /// Grab-family per-stroke state: Blender `StrokeCache` fields that persist between dabs.
    grab: grab::State,
    /// Whether a Smooth call has already run this stroke (`smooth::smooth_dab` reads the first call
    /// from its start positions).
    smooth_started: bool,
}

/// One dab's values as Blender's `StrokeCache` holds them, before any symmetry pass.
#[derive(Clone, Copy)]
struct DabInput {
    /// `cache->location`: the dab center, or the stroke-start center for anchored brushes.
    center: DVec3,
    /// `cache->radius`.
    radius: f64,
    /// `cache->pressure`: the sample's raw pressure.
    pressure: f64,
    /// `cache->grab_delta`: the move from the previous dab's center (from the stroke-start center
    /// for anchored brushes), projected off the view direction, or zero for the first dab. An
    /// explicit stroke delta replaces it for the brushes that accept one.
    grab: DVec3,
    /// `BKE_brush_alpha_get`: the sample's strength before it is squared into `alpha`.
    brush_alpha: f64,
    /// `cache->bstrength`: `brush_strength` for this dab, symmetry feather included.
    strength: f64,
}

/// Blender's `calc_area_normal_and_center` result for one dab.
#[derive(Clone, Copy)]
struct AreaPlane {
    normal: DVec3,
    center: DVec3,
}

/// Apply every sample as one dab, in order, running each dab's symmetry passes in Blender's order.
/// Returns whether any vertex moved.
pub(super) fn apply_dabs(
    mesh: &mut Mesh,
    stroke: &Stroke,
    dab: DabBrush,
    normals: &[DVec3],
    masks: &BTreeMap<u32, f64>,
) -> bool {
    let passes = symmetry_passes(symmetry_mask(&stroke.symmetry));
    let original: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let index: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(position, vertex)| (vertex.id, position))
        .collect();
    let mut field = Field {
        positions: original.clone(),
        original: original.clone(),
        normals: normals.to_vec(),
        // Blender's `fill_factor_from_hide_and_mask`: a masked vertex keeps `1 - mask` of each dab.
        mask_factors: mesh
            .vertices
            .iter()
            .map(|vertex| {
                1.0 - masks
                    .get(&vertex.id)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0)
            })
            .collect(),
        neighbors: vertex_neighbors(mesh, &index),
        displacement: vec![0.0; original.len()],
        falloff: stroke.falloff,
        passes,
        view: stroke.view.map(DVec3::normalize),
        grab: grab::State::default(),
        smooth_started: false,
    };
    // Blender's `need_delta_from_anchored_origin` brushes measure every dab from the stroke's
    // first dab, which also becomes the dab center (`brush_delta_update`, sculpt.cc 4277-4312).
    let anchored = anchors_to_stroke_start(dab);
    let origin = stroke
        .samples
        .first()
        .map_or(DVec3::ZERO, |sample| sample.position);
    let mut previous: Option<DVec3> = None;
    for sample in &stroke.samples {
        if restores_each_dab(dab) {
            // Blender's `restore_from_undo_step_if_necessary` (sculpt.cc 5160-5180): the mesh returns
            // to the stroke-start state before each dab, so a dab replaces the one before it.
            field.positions.clone_from(&field.original);
        }
        let (center, reference) = if anchored {
            (origin, origin)
        } else {
            (sample.position, previous.unwrap_or(sample.position))
        };
        previous = Some(sample.position);
        let feather = symmetry_feather(center, sample.radius, &field.passes);
        let input = DabInput {
            center,
            radius: sample.radius,
            pressure: sample.pressure,
            grab: stroke
                .delta
                .unwrap_or_else(|| grab_on_view_plane(sample.position - reference, field.view)),
            brush_alpha: sample.strength,
            strength: brush_strength(dab, sample, feather),
        };
        apply_dab(&mut field, dab, &input);
    }
    let changed = field
        .positions
        .iter()
        .zip(&original)
        .any(|(after, before)| after != before);
    if changed {
        for (vertex, co) in mesh.vertices.iter_mut().zip(field.positions) {
            vertex.co = co;
        }
    }
    changed
}

fn apply_dab(field: &mut Field, dab: DabBrush, input: &DabInput) {
    match dab {
        DabBrush::ClayStrips => clay_strips_dab(field, input),
        DabBrush::Inflate => inflate_dab(field, input),
        DabBrush::Layer => layer_dab(field, input),
        DabBrush::Draw | DabBrush::DrawSharp | DabBrush::Blob | DabBrush::Nudge => {
            offset::apply(field, dab, input);
        }
        DabBrush::Flatten | DabBrush::Fill | DabBrush::Scrape => plane::apply(field, dab, input),
        DabBrush::Crease | DabBrush::Pinch => crease_pinch::apply(field, dab, input),
        DabBrush::Smooth => smooth::apply(field, dab, input),
        DabBrush::Grab | DabBrush::SnakeHook | DabBrush::Thumb | DabBrush::Rotate => {
            grab::apply(field, dab, input);
        }
        DabBrush::Elastic => elastic::apply(field, dab, input),
    }
}

/// Blender's `brush_strength` (`sculpt.cc` 2329) for one dab of `brush`, with `overlap` = 1 (DOTS
/// dabs without spacing attenuation) and the symmetry feather. `alpha` is the strength squared,
/// except for the grab-type brushes, which use the strength as is.
fn brush_strength(brush: DabBrush, sample: &StrokeSample, feather: f64) -> f64 {
    let root_alpha = sample.strength;
    let alpha = root_alpha * root_alpha;
    let flip = brush_flip(brush);
    let pressure = if uses_pressure_strength(brush) {
        sample.pressure
    } else {
        1.0
    };
    match brush {
        DabBrush::ClayStrips => 0.3 * alpha * flip * pressure.powf(1.5) * feather,
        DabBrush::Inflate => {
            let base = if flip > 0.0 { 0.25 } else { 0.125 };
            base * alpha * flip * pressure * feather
        }
        DabBrush::Pinch => {
            let base = if flip > 0.0 { 1.0 } else { 0.25 };
            base * alpha * flip * pressure * feather
        }
        DabBrush::Layer
        | DabBrush::Draw
        | DabBrush::DrawSharp
        | DabBrush::Blob
        | DabBrush::Crease => alpha * flip * pressure * feather,
        DabBrush::Smooth => flip * alpha * pressure * feather,
        DabBrush::Nudge
        | DabBrush::Flatten
        | DabBrush::Fill
        | DabBrush::Scrape
        | DabBrush::Rotate
        | DabBrush::Thumb => alpha * pressure * feather,
        DabBrush::Grab | DabBrush::SnakeHook | DabBrush::Elastic => root_alpha * feather,
    }
}

/// Blender's `brush_flip` (`sculpt.cc` 2316): -1 for a SUBTRACT brush (`BRUSH_DIR_IN`), else 1.
/// Crease Polish's asset is SUBTRACT: its oracle moves the top vertex inward. Draw Sharp reports the
/// same raw direction value in the asset, and its sign is pinned by `draw_sharp_sphere`. Potter has
/// no invert input, so the asset direction is the only flip.
fn brush_flip(brush: DabBrush) -> f64 {
    match brush {
        DabBrush::Crease | DabBrush::DrawSharp => -1.0,
        _ => 1.0,
    }
}

/// Whether the brush's `use_pressure_strength` is on in the Blender 5.2.2 essentials asset. When it
/// is off, Blender's `brush_strength` sees pressure 1.
fn uses_pressure_strength(brush: DabBrush) -> bool {
    !matches!(brush, DabBrush::Grab | DabBrush::Thumb | DabBrush::Elastic)
}

/// Brushes whose dabs measure from the stroke's first dab (`need_delta_from_anchored_origin`,
/// sculpt.cc 4170-4190; Rotate's `cache->location` stays pinned to the first dab, sculpt.cc
/// 5831-5837). Pose and Boundary, which share the first rule, are not dab brushes here.
fn anchors_to_stroke_start(brush: DabBrush) -> bool {
    matches!(
        brush,
        DabBrush::Grab | DabBrush::Thumb | DabBrush::Elastic | DabBrush::Rotate
    )
}

/// Brushes whose mesh returns to the stroke-start state before each dab
/// (`restore_from_undo_step_if_necessary`, sculpt.cc 5160-5180).
fn restores_each_dab(brush: DabBrush) -> bool {
    matches!(
        brush,
        DabBrush::Grab | DabBrush::Thumb | DabBrush::Rotate | DabBrush::Elastic
    )
}

/// Blender unprojects the mouse at the depth of the stroke's first dab, so a grab delta lies on
/// the plane perpendicular to the view. Without a view, the delta is the 3D move as given.
fn grab_on_view_plane(delta: DVec3, view: Option<DVec3>) -> DVec3 {
    match view {
        Some(view) => delta - view * delta.dot(view),
        None => delta,
    }
}

/// Blender's `symm` bit mask for the stroke's mirror axes: X = 1, Y = 2, Z = 4. `validate_stroke`
/// has already rejected any other axis name, so an unknown name contributes no bit.
fn symmetry_mask(axes: &[String]) -> u8 {
    axes.iter().fold(0, |mask, axis| {
        mask | match axis.as_str() {
            "x" => 1,
            "y" => 2,
            "z" => 4,
            _ => 0,
        }
    })
}

/// Blender's `is_symmetry_iteration_valid` (`paint_intern.hh`): the symmetry passes of a `symm`
/// mask, as per-axis sign vectors. Pass 0 is the unflipped brush. Blender's
/// `do_symmetrical_brush_actions` runs the whole brush once per pass, in order, with the location
/// and grab delta flipped.
fn symmetry_passes(symm: u8) -> Vec<DVec3> {
    let sign = |bit: u8, i: u8| if i & bit != 0 { -1.0 } else { 1.0 };
    (0..=symm)
        .filter(|&i| {
            i == 0
                || ((symm & i) != 0 && !(symm == 5 && i == 3) && !(symm == 6 && (i == 3 || i == 5)))
        })
        .map(|i| DVec3::new(sign(1, i), sign(2, i), sign(4, i)))
        .collect()
}

/// Blender's `calc_symmetry_feather` with the default `use_symmetry_feather`: each symmetry pass
/// contributes an overlap of how close its mirrored dab center is to the original, and every
/// pass's strength is divided by the total overlap (`sculpt.cc` `calc_overlap`).
fn symmetry_feather(center: DVec3, radius: f64, passes: &[DVec3]) -> f64 {
    let overlap: f64 = passes
        .iter()
        .map(|sign| {
            let distance_sq = (center * *sign).distance_squared(center);
            if distance_sq <= 4.0 * radius * radius {
                (2.0 * radius - distance_sq.sqrt()) / (2.0 * radius)
            } else {
                0.0
            }
        })
        .sum();
    1.0 / overlap
}

/// Blender's `calc_vert_neighbors_interior` (`sculpt.cc` 8092) over the whole mesh: the two corners
/// next to a vertex in each face (`append_neighbors_to_vector`, `sculpt.cc` 554). Boundary
/// vertices (endpoints of edges used by fewer than two faces, loose edges included;
/// `create_boundary_info`, `sculpt.cc` 598) keep only boundary-edge neighbors, and none when they
/// have exactly two (a corner).
fn vertex_neighbors(mesh: &Mesh, index: &BTreeMap<u32, usize>) -> Vec<Vec<usize>> {
    let key = |a: u32, b: u32| if a <= b { (a, b) } else { (b, a) };
    let mut face_uses: BTreeMap<(u32, u32), usize> = mesh
        .edges
        .iter()
        .map(|edge| (key(edge.vertices[0], edge.vertices[1]), 0))
        .collect();
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); mesh.vertices.len()];
    for face in &mesh.faces {
        let corners = &face.vertices;
        let count = corners.len();
        for corner in 0..count {
            let vert = corners[corner];
            let next = corners[(corner + 1) % count];
            let prev = corners[(corner + count - 1) % count];
            *face_uses.entry(key(vert, next)).or_insert(0) += 1;
            if let Some(&i) = index.get(&vert) {
                around[i].extend([prev, next]);
            }
        }
    }
    let mut boundary = vec![false; mesh.vertices.len()];
    for edge in &mesh.edges {
        if face_uses
            .get(&key(edge.vertices[0], edge.vertices[1]))
            .is_some_and(|&uses| uses < 2)
        {
            for id in edge.vertices {
                if let Some(&i) = index.get(&id) {
                    boundary[i] = true;
                }
            }
        }
    }
    let mut neighbors = vec![Vec::new(); mesh.vertices.len()];
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        let mut ids = std::mem::take(&mut around[i]);
        ids.sort_unstable();
        ids.dedup();
        if boundary[i] && ids.len() == 2 {
            continue;
        }
        let mut list: Vec<usize> = ids
            .iter()
            .filter(|&&id| {
                !boundary[i]
                    || face_uses
                        .get(&key(vertex.id, id))
                        .is_some_and(|&uses| uses < 2)
            })
            .filter_map(|id| index.get(id).copied())
            .collect();
        list.sort_unstable();
        neighbors[i] = list;
    }
    neighbors
}

/// Blender's `calc_area_normal_and_center` over the mesh vertices. Normals are summed with the
/// smooth weight inside `normal_radius`, and each vertex inside `position_radius` is pulled toward
/// `center` by the same weight before averaging. The current positions are used when `accumulate`
/// is set, the stroke-start positions otherwise. Blender splits the vertices into two groups by
/// `dot(view, normal) <= 0` (`flip_index`, sculpt.cc 1538) and takes the first group with a nonzero
/// normal sum for the normal and with a vertex for the center. Without a view there is one group.
/// Returns `None` when no group has a nonzero normal sum.
fn area_plane(
    field: &Field,
    center: DVec3,
    normal_radius: f64,
    position_radius: f64,
    accumulate: bool,
) -> Option<AreaPlane> {
    let positions = if accumulate {
        &field.positions
    } else {
        &field.original
    };
    let mut normal_sums = [DVec3::ZERO; 2];
    let mut center_sums = [DVec3::ZERO; 2];
    let mut counts = [0_usize; 2];
    for (position, normal) in positions.iter().zip(&field.normals) {
        let group = match field.view {
            Some(view) if normal.dot(view) <= 0.0 => 1,
            _ => 0,
        };
        let distance = position.distance(center);
        if distance <= normal_radius {
            normal_sums[group] += *normal * Falloff::Smooth.weight(distance / normal_radius);
        }
        if distance <= position_radius {
            let weight = Falloff::Smooth.weight(distance / position_radius);
            center_sums[group] += center + (*position - center) * (1.0 - weight);
            counts[group] += 1;
        }
    }
    let normal = normal_sums
        .iter()
        .map(|sum| sum.normalize_or_zero())
        .find(|normal| *normal != DVec3::ZERO)?;
    let center = match counts.iter().position(|&count| count > 0) {
        Some(group) => center_sums[group] / counts[group] as f64,
        None => center,
    };
    Some(AreaPlane { normal, center })
}

/// Blender's `calc_brush_cube_distances` for the Clay Strips tip: a square of half-size one with
/// corners rounded by `roundness`, returning one outside the square.
fn cube_distance(x: f64, y: f64, roundness: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let hardness = 1.0 - roundness;
    let (low, high) = (ax.min(ay), ax.max(ay));
    if high > 1.0 {
        1.0
    } else if low > hardness {
        DVec2::new(ax, ay).distance(DVec2::splat(hardness)) / roundness
    } else if high > hardness {
        (high - hardness) / roundness
    } else {
        0.0
    }
}

/// Blender's `apply_hardness_to_distances`: distances inside `hardness * radius` collapse to zero,
/// and the rest of the range is stretched back to `[0, radius]`.
fn hardness_distance(distance: f64, radius: f64, hardness: f64) -> f64 {
    if distance < hardness * radius {
        0.0
    } else {
        (distance / radius - hardness) / (1.0 - hardness) * radius
    }
}

/// Blender 5.2.2 Clay Strips for one dab, run once per symmetry pass. The plane comes from the area
/// normal and center around the dab (`calc_brush_plane`, AREA), computed once in the main pass; a
/// mirror pass flips that normal, center, and the grab delta instead of recomputing them
/// (`sculpt.cc`, non-main-pass branch). The strip frame has x = normal × grab, y = normal × x, and
/// z = -normal, so vertices below the plane have positive local z. Each vertex moves along the
/// plane normal by `strength * radius` times its depth and tip factors (`clay_strips.cc`).
fn clay_strips_dab(field: &mut Field, dab: &DabInput) {
    // Clay Strips has no strip direction before the second dab, and `calc_node_mask` returns no
    // nodes for a zero grab delta.
    if dab.grab == DVec3::ZERO {
        return;
    }
    let radius = dab.radius * CLAY_STRIPS_AREA_RADIUS_FACTOR;
    let Some(area) = area_plane(field, dab.center, radius, radius, true) else {
        return;
    };
    for sign in &field.passes {
        let plane_normal = area.normal * *sign;
        let plane_center =
            area.center * *sign + plane_normal * (dab.radius * CLAY_STRIPS_PLANE_OFFSET);
        let grab = dab.grab * *sign;
        let x_axis = plane_normal.cross(grab).normalize_or_zero();
        if x_axis == DVec3::ZERO {
            continue;
        }
        let y_axis = plane_normal.cross(x_axis).normalize_or_zero();
        let z_axis = -plane_normal;
        for (index, position) in field.positions.iter_mut().enumerate() {
            let offset = *position - plane_center;
            let local_z = offset.dot(z_axis) / dab.radius;
            let depth = (local_z * (1.0 - local_z)).max(0.0) * field.mask_factors[index];
            if depth == 0.0 {
                continue;
            }
            let tip = cube_distance(
                offset.dot(x_axis) / dab.radius,
                offset.dot(y_axis) / dab.radius,
                CLAY_STRIPS_TIP_ROUNDNESS,
            );
            if tip >= 1.0 {
                continue;
            }
            let factor = depth * field.falloff.weight(tip);
            *position += plane_normal * (dab.strength * dab.radius * factor);
        }
    }
}

/// Blender 5.2.2 Inflate for one dab: each vertex moves along its stroke-start normal by
/// `strength * radius` times the falloff and mask factors (`inflate.cc`).
fn inflate_dab(field: &mut Field, dab: &DabInput) {
    for sign in &field.passes {
        let center = dab.center * *sign;
        for (index, position) in field.positions.iter_mut().enumerate() {
            let distance = position.distance(center);
            if distance >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(distance / dab.radius) * field.mask_factors[index];
            *position += field.normals[index] * (dab.strength * dab.radius * factor);
        }
    }
}

/// Blender 5.2.2 Layer for one dab (`layer.cc`, non-persistent base). Falloff uses the stroke-start
/// positions. A per-vertex displacement factor accumulates across the stroke, clamped to the mask
/// limit `1 - mask`, and each vertex moves toward `base + normal * height * factor`.
fn layer_dab(field: &mut Field, dab: &DabInput) {
    for sign in &field.passes {
        let center = dab.center * *sign;
        for index in 0..field.positions.len() {
            let distance = field.original[index].distance(center);
            if distance >= dab.radius {
                continue;
            }
            let hardened = hardness_distance(distance, dab.radius, LAYER_HARDNESS);
            if hardened >= dab.radius {
                continue;
            }
            let factor = field.falloff.weight(hardened / dab.radius) * field.mask_factors[index];
            if factor == 0.0 {
                continue;
            }
            let limit = field.mask_factors[index];
            let displacement = field.displacement[index];
            let grown = displacement + factor * dab.strength * (1.05 - displacement.abs());
            field.displacement[index] = grown.clamp(-limit, limit);
            let target = field.original[index]
                + field.normals[index] * (LAYER_HEIGHT * field.displacement[index]);
            field.positions[index] += (target - field.positions[index]) * factor;
        }
    }
}
