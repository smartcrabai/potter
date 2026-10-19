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
//! `boundary_falloff_type` CONSTANT, `boundary_offset` 0, no `deform_target` change, and no
//! symmetry. The falloff is then one for every propagated vertex short of the last step, and zero
//! on the last step (`BKE_brush_curve_strength` with a CONSTANT curve).
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

/// Boundary state that persists between dabs.
#[derive(Default)]
pub(super) enum State {
    /// The first dab has not run.
    #[default]
    Unstarted,
    /// The first dab found no editable boundary, so the stroke deforms nothing.
    Missing,
    /// The loop the stroke deforms.
    Found(Box<Boundary>),
}

/// Blender's `SculptBoundary`, for the single loop the stroke found.
pub(super) struct Boundary {
    /// `boundary.pivot_position`: the vertex the propagation reached last from the initial vertex's
    /// cone, the reference direction of the bend and displacement.
    pivot: DVec3,
    /// `max_propagation_steps`: the number of propagation steps before the radius ran out.
    max_steps: u32,
    /// `boundary.edit_info`.
    edit: EditInfo,
    /// `cache->initial_location`: the stroke's first location (Blender's cursor at stroke start).
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
    /// `strength_factor`: the mask times the falloff of the vertex's step, zero when unreached.
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

/// Apply one Boundary dab. The loop is found on the stroke's first dab only.
pub(super) fn apply(field: &mut Field, deform: BoundaryDeform, input: &DabInput) {
    let state = std::mem::take(&mut field.boundary);
    let state = match state {
        State::Unstarted => match find(field, deform, input) {
            Some(boundary) => State::Found(Box::new(boundary)),
            None => State::Missing,
        },
        other => other,
    };
    if let State::Found(boundary) = &state {
        let amount = deform_amount(boundary, deform, input);
        match deform {
            BoundaryDeform::Bend => bend::apply(field, boundary, amount),
            BoundaryDeform::Expand => expand::apply(field, boundary, amount),
            BoundaryDeform::Inflate => inflate::apply(field, boundary, amount),
            BoundaryDeform::Grab => grab::apply(field, boundary, input, amount),
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

/// Blender's `data_init_mesh` for one stroke: the loop around the boundary vertex nearest the
/// active vertex, with its propagation and deform data. `None` when no editable loop is in reach.
fn find(field: &Field, deform: BoundaryDeform, input: &DabInput) -> Option<Boundary> {
    let topology = &field.topology;
    let initial = nearest_vertex(&field.original, input.center);
    let radius = input.radius;
    let closest = closest_boundary_vertex(topology, &field.original, initial, radius)?;
    if !is_editable(topology, initial) {
        return None;
    }
    let verts = boundary_loop(topology, closest);
    let (edit, pivot, max_steps) = propagate(
        topology,
        &field.original,
        &field.mask_factors,
        &verts,
        closest,
        radius,
    );
    let mut boundary = Boundary {
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
