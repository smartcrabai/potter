//! Blender 5.2.2 Boundary (`sculpt_boundary.cc`) on the dab framework.
//!
//! The first dab finds one mesh boundary loop: the boundary vertex nearest the active vertex (the
//! vertex nearest the stroke start), found by a flood fill within the brush radius, and the loop
//! that vertex belongs to, grown along boundary vertices (`data_init_mesh`). A propagation then
//! assigns every vertex near the loop to its nearest loop vertex, counting steps across the mesh
//! until the distance walked from the initial vertex's own cone exceeds the radius
//! (`edit_data_init_mesh`). Each deform type moves vertices from their stroke-start positions by
//! the falloff of those steps, scaled by the grab delta or the displacement of the grab from the
//! loop's pivot, every dab (`do_*_brush`). Boundary is not restored between dabs.
//!
//! Potter's brush has the Blender 5.2.2 essentials defaults for the settings it does not expose:
//! `boundary_falloff_type` CONSTANT, `boundary_offset` 0, and no `deform_target` change. The falloff
//! is then one for every propagated vertex short of the last step, and zero on the last step
//! (`BKE_brush_curve_strength` with a CONSTANT curve).
//!
//! Mirror symmetry runs each dab once per pass, in `Field::passes` order. Each pass finds its own
//! loop on the stroke's first dab, from the mirror image of the active vertex, and deforms it with
//! the mirror image of the grab. A vertex outside a pass's symmetry area gets no factor, so the
//! passes split the mesh between them (`filter_verts_outside_symmetry_area`).
//!
//! Blender computes the propagation distance in single precision. Potter uses double precision,
//! so a distance that sits exactly on the radius can stop one step apart from Blender.

use std::collections::VecDeque;
use std::f64::consts::PI;

use glam::DVec3;

use super::topology::Topology;
use super::{DabInput, Field, nearest_vertex};
use crate::geom::sculpt::BoundaryDeform;

mod bend;
mod expand;
mod grab;
mod inflate;
mod smooth;
mod twist;

/// Boundary state that persists between dabs: Blender's `cache->boundaries[symm_area]` for each
/// symmetry pass, in `Field::passes` order. Empty until the stroke's first dab.
#[derive(Default)]
pub(super) struct State {
    /// `None` for a pass whose loop was not found.
    passes: Vec<Option<Box<Boundary>>>,
}

/// Blender's `SculptBoundary`, for the single loop the stroke found.
pub(super) struct Boundary {
    /// `boundary.verts`: the loop's vertices in the order the loop grew from the initial one. The
    /// initial vertex appears twice, as Blender's flood fill adds it again when a neighbor reaches it.
    verts: Vec<usize>,
    /// `boundary.pivot_position`: the vertex the propagation reached last from the initial vertex's
    /// cone, the reference direction of the bend and displacement.
    pivot: DVec3,
    /// `max_propagation_steps`: the number of propagation steps before the radius ran out.
    max_steps: u32,
    /// `boundary.edit_info`.
    edit: EditInfo,
    /// `cache->initial_location_symm`: the stroke's first location, mirrored for this pass.
    initial_location: DVec3,
    /// Data each deform type precomputes once per stroke.
    deform: DeformData,
}

/// Blender's `edit_info`, indexed by vertex position.
pub(super) struct EditInfo {
    /// `original_vertex_i`: the loop vertex the propagation reached this vertex from.
    pub(super) original: Vec<Option<usize>>,
    /// `propagation_steps_num`: steps from the loop, `None` when the propagation never reached it.
    pub(super) steps: Vec<Option<u32>>,
    /// `strength_factor`: the mask times the falloff of the vertex's step, zero when unreached, and
    /// zero outside the pass's symmetry area.
    pub(super) factor: Vec<f64>,
}

/// Per-stroke data that only some deform types need, computed once with the loop.
pub(super) enum DeformData {
    /// Grab, Inflate, and Smooth precompute nothing.
    None,
    Bend(bend::Data),
    Expand(expand::Data),
    Twist(twist::Data),
}

/// Apply one Boundary dab, once per symmetry pass in `Field::passes` order. Each pass's loop is
/// found on the stroke's first dab only.
pub(super) fn apply(field: &mut Field, deform: BoundaryDeform, input: &DabInput) {
    let mut state = std::mem::take(&mut field.boundary);
    if state.passes.is_empty() {
        state.passes = init_passes(field, deform, input);
    }
    for (pass, slot) in state.passes.iter().enumerate() {
        let Some(boundary) = slot else {
            continue;
        };
        let dab = mirrored(input, field.passes[pass]);
        let amount = deform_amount(boundary, deform, &dab);
        match deform {
            BoundaryDeform::Bend => bend::apply(field, boundary, amount),
            BoundaryDeform::Expand => expand::apply(field, boundary, amount),
            BoundaryDeform::Inflate => inflate::apply(field, boundary, amount),
            BoundaryDeform::Grab => grab::apply(field, boundary, &dab, amount),
            BoundaryDeform::Twist => twist::apply(field, boundary, amount),
            BoundaryDeform::Smooth => smooth::apply(field, boundary, amount),
        }
    }
    field.boundary = state;
}

/// Blender's `get_mesh_strength` (`sculpt_boundary.cc` 3263): the per-dab amount each deform type
/// receives. Bend and Twist turn by an angle from the grab's displacement over the radius; Expand
/// and Inflate move by the displacement; Grab and Smooth use the strength itself.
fn deform_amount(boundary: &Boundary, deform: BoundaryDeform, input: &DabInput) -> f64 {
    match deform {
        BoundaryDeform::Bend | BoundaryDeform::Twist => {
            let disp = input.strength * displacement(boundary, input.grab);
            (disp / input.radius) * PI
        }
        BoundaryDeform::Expand | BoundaryDeform::Inflate => {
            input.strength * displacement(boundary, input.grab)
        }
        BoundaryDeform::Grab | BoundaryDeform::Smooth => input.strength,
    }
}

/// Blender's `displacement_from_grab_delta_get`: the grab's signed distance from the plane through
/// the stroke's first location, perpendicular to the direction from the pivot.
fn displacement(boundary: &Boundary, grab: DVec3) -> f64 {
    let normal = (boundary.initial_location - boundary.pivot).normalize_or_zero();
    normal.dot(grab)
}

/// Blender's `rotate_v3_v3v3fl` (`rotate_normalized_v3_v3v3fl` after normalizing the axis): `point`
/// rotated by `angle` about `axis`. A zero axis gives `cos(angle) * point`, as Blender's matrix does.
pub(super) fn rotate(point: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let axis = axis.normalize_or_zero();
    let (sin, cos) = angle.sin_cos();
    point * cos + axis.cross(point) * sin + axis * (axis.dot(point) * (1.0 - cos))
}

/// Blender's `data_init_mesh` for one symmetry pass: the loop around the boundary vertex `initial`,
/// with its propagation and deform data. `input` is the pass's mirrored dab. `None` when no editable
/// loop is in reach.
fn find(
    field: &Field,
    deform: BoundaryDeform,
    input: &DabInput,
    initial: usize,
) -> Option<Boundary> {
    let topology = &field.topology;
    let radius = input.radius;
    let closest = closest_boundary_vertex(topology, &field.original, initial, radius)?;
    if !is_editable(topology, initial) {
        return None;
    }
    let verts = boundary_loop(topology, closest);
    let (mut edit, pivot, max_steps) = propagate(
        topology,
        &field.original,
        &field.mask_factors,
        &verts,
        closest,
        radius,
    );
    // `filter_verts_outside_symmetry_area`, against the loop's initial vertex.
    let mirror = mirror_axes(&field.passes);
    for (index, factor) in edit.factor.iter_mut().enumerate() {
        if !in_symmetry_area(field.original[index], field.original[closest], mirror) {
            *factor = 0.0;
        }
    }
    let mut boundary = Boundary {
        verts,
        pivot,
        max_steps,
        edit,
        initial_location: input.center,
        deform: DeformData::None,
    };
    boundary.deform = match deform {
        BoundaryDeform::Bend => DeformData::Bend(bend::init(field, &boundary)),
        BoundaryDeform::Expand => DeformData::Expand(expand::init(field, &boundary)),
        BoundaryDeform::Twist => DeformData::Twist(twist::init(field, &boundary)),
        BoundaryDeform::Grab | BoundaryDeform::Inflate | BoundaryDeform::Smooth => DeformData::None,
    };
    Some(boundary)
}

/// Blender's `init_boundary_mesh` for every symmetry pass, on the stroke's first dab. The main pass
/// starts from the active vertex, the vertex nearest the stroke's first location. A mirror pass
/// starts from the vertex nearest the mirror image of the active vertex, if it lies within the
/// radius. Each pass then finds its loop from its mirrored dab.
fn init_passes(
    field: &Field,
    deform: BoundaryDeform,
    input: &DabInput,
) -> Vec<Option<Box<Boundary>>> {
    let active = nearest_vertex(&field.original, input.center);
    field
        .passes
        .iter()
        .enumerate()
        .map(|(pass, &sign)| {
            let initial = if pass == 0 {
                Some(active)
            } else {
                nearest_within(&field.original, sign * field.original[active], input.radius)
            };
            let dab = mirrored(input, sign);
            initial
                .and_then(|initial| find(field, deform, &dab, initial))
                .map(Box::new)
        })
        .collect()
}

/// The dab as one symmetry pass sees it: Blender's `cache_calc_brushdata_symm` flips the location,
/// the grab delta, and the initial location. The initial location is the dab center for anchored
/// brushes, so the center is what gets flipped.
fn mirrored(input: &DabInput, sign: DVec3) -> DabInput {
    DabInput {
        center: input.center * sign,
        grab: input.grab * sign,
        ..*input
    }
}

/// Blender's `nearest_vert_calc_mesh`: the vertex nearest `location` when it lies within `radius`
/// of it. Blender tests the mesh nodes the sphere touches; testing the vertex itself agrees with it
/// whenever the nearest vertex is inside the sphere.
fn nearest_within(positions: &[DVec3], location: DVec3, radius: f64) -> Option<usize> {
    let vertex = nearest_vertex(positions, location);
    (positions[vertex].distance(location) <= radius).then_some(vertex)
}

/// The mirror axes of a stroke's symmetry passes: an axis is mirrored when some pass flips it.
fn mirror_axes(passes: &[DVec3]) -> [bool; 3] {
    std::array::from_fn(|axis| passes.iter().any(|sign| sign[axis] < 0.0))
}

/// Blender's `check_vertex_pivot_symmetry` (`sculpt.cc` 671): on each mirror axis, a vertex must
/// lie on the pivot's side. A pivot on the mirror plane admits only vertices at or below it.
fn in_symmetry_area(vertex: DVec3, pivot: DVec3, mirror: [bool; 3]) -> bool {
    (0..3).all(|axis| {
        let (co, pv) = (vertex[axis], pivot[axis]);
        !mirror[axis] || (!(pv == 0.0 && co > 0.0) && co * pv >= 0.0)
    })
}

/// Blender's `check_counts` with `is_vert_in_editable_boundary_mesh`: a vertex can grow the loop
/// when it has more than two neighbors and at most two of them are boundary vertices. Corners and
/// non-manifold vertices cannot.
fn is_editable(topology: &Topology, vertex: usize) -> bool {
    let neighbors = &topology.ring[vertex];
    if neighbors.len() <= 2 {
        return false;
    }
    neighbors
        .iter()
        .filter(|&&neighbor| topology.boundary[neighbor])
        .count()
        <= 2
}

/// Blender's `get_closest_boundary_vert_mesh`: the boundary vertex reached from `initial` with the
/// fewest steps, where the flood fill expands only through vertices within `radius` of `initial`.
/// Ties go to the vertex the flood fill reached first.
fn closest_boundary_vertex(
    topology: &Topology,
    positions: &[DVec3],
    initial: usize,
    radius: f64,
) -> Option<usize> {
    if topology.boundary[initial] {
        return Some(initial);
    }
    let origin = positions[initial];
    let radius_squared = radius * radius;
    let mut steps = vec![0_u32; positions.len()];
    let mut visited = vec![false; positions.len()];
    let mut queue = VecDeque::from([initial]);
    let mut closest = None;
    let mut closest_steps = u32::MAX;
    while let Some(from) = queue.pop_front() {
        for &to in &topology.ring[from] {
            if visited[to] {
                continue;
            }
            visited[to] = true;
            steps[to] = steps[from] + 1;
            if topology.boundary[to] && steps[to] < closest_steps {
                closest_steps = steps[to];
                closest = Some(to);
            }
            if origin.distance_squared(positions[to]) < radius_squared {
                queue.push_back(to);
            }
        }
    }
    closest
}

/// Blender's `indices_init_mesh`: the loop through `start`, grown along boundary vertices. A loop
/// vertex that is not editable is listed but does not grow the loop further.
fn boundary_loop(topology: &Topology, start: usize) -> Vec<usize> {
    let mut verts = vec![start];
    let mut visited = vec![false; topology.boundary.len()];
    let mut queue = VecDeque::from([start]);
    while let Some(from) = queue.pop_front() {
        for &to in &topology.ring[from] {
            if visited[to] {
                continue;
            }
            visited[to] = true;
            if !topology.boundary[to] {
                continue;
            }
            verts.push(to);
            if is_editable(topology, to) {
                queue.push_back(to);
            }
        }
    }
    verts
}

/// Blender's `edit_data_init_mesh` with the CONSTANT falloff of the essentials defaults. Returns
/// the edit info, the pivot, and the number of propagation steps.
fn propagate(
    topology: &Topology,
    positions: &[DVec3],
    mask_factors: &[f64],
    verts: &[usize],
    initial: usize,
    radius: f64,
) -> (EditInfo, DVec3, u32) {
    let count = positions.len();
    let mut original = vec![None; count];
    let mut steps = vec![None; count];
    let mut current = VecDeque::new();
    for &vertex in verts {
        original[vertex] = Some(vertex);
        steps[vertex] = Some(0_u32);
        current.push_back(vertex);
    }
    let mut next = VecDeque::new();
    let mut pivot = DVec3::ZERO;
    let mut accumulated = 0.0_f64;
    let mut step_count = 0_u32;
    let max_steps = loop {
        if accumulated > radius || current.is_empty() {
            break step_count;
        }
        while let Some(from) = current.pop_front() {
            for &to in &topology.ring[from] {
                if steps[to].is_some() {
                    continue;
                }
                let from_original = original[from];
                original[to] = from_original;
                steps[to] = steps[from].map(|step| step + 1);
                next.push_back(to);
                // Only the cone of the initial vertex's loop vertex measures the distance.
                if from_original == Some(initial) {
                    pivot = positions[to];
                    accumulated += positions[from].distance(pivot);
                }
            }
        }
        current.append(&mut next);
        step_count += 1;
    };
    // `init_falloff_mesh` with the CONSTANT curve: one until the last step, zero at it. A masked
    // vertex keeps `1 - mask` of that.
    let factor = (0..count)
        .map(|index| match steps[index] {
            Some(step) if step < max_steps => mask_factors[index],
            _ => 0.0,
        })
        .collect();
    (
        EditInfo {
            original,
            steps,
            factor,
        },
        pivot,
        max_steps,
    )
}
