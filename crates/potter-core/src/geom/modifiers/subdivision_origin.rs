// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org) source/blender/blenkernel/intern/subdiv_mesh.cc, GPL-2.0-or-later.

//! Element origins of a subdivided mesh under Blender's interpolation rules, for the attribute remap.
//!
//! A face-sampled element mixes its data from the coarse corners of its base face through Blender's four ptex
//! slots, which also place its position. `quad_weights_from_uv` (`subdiv_mesh.cc:966`) weights the slots at ptex
//! corners (0,0), (1,0), (1,1), and (0,1). A quad's slots are its corners. An n-gon's slots for ptex face `p` are
//! corner `p`, the midpoints of `(p, p+1)` and `(p-1, p)`, and the face centre, which mixes all corners with equal
//! weight (`vert_interpolation_from_corner`, `subdiv_mesh.cc:498`; `loop_interpolation_from_corner`,
//! `subdiv_mesh.cc:657`). Each slot is a stage: Blender converts it to the attribute type before the outer mix, so
//! integer, quaternion, and byte data round or average per slot (`provenance::Stage`).
//!
//! A coarse corner copies its vertex (`vert_corner`, `subdiv_mesh.cc:1093`), so it has no stages. A loose-edge
//! vertex is one mix of its edge's ends (`subdiv_mesh_vert_of_loose_edge_interpolate`, `subdiv_mesh.cc:1434`),
//! and a loose end point copies its vertex (`subdiv_mesh_vert_loose`, `subdiv_mesh.cc:1357`). Both stay flat.
//!
//! Edges follow `subdiv_mesh_edge` (`subdiv_mesh.cc:1226`): a subedge copies its coarse edge's data, and an
//! internal edge has none (`subdiv_copy_edge_data`, `subdiv_mesh.cc:1199`). Each subdivided face copies its
//! coarse face (`subdiv_mesh_face`, `subdiv_mesh.cc:1331`).

use serde_json::Value;

use super::subdivision_limit::{Sample, Samples};
use super::subdivision_loose::LooseEdges;
use super::{Polygon, usize_as_f64};
use crate::error::{ErrorCode, PotError, Result};
use crate::geom::provenance::{Mixing, Origin, Stage, Staged, needs_corner_sources};
use crate::geom::{Mesh, uv_layers};

/// Keys subdivision writes itself, so the remap skips them and their output entries are kept.
/// `crease_edge` holds the subdivided creases (`subdivision_uv::subdivided_mesh`), `uv_layers` the limit
/// UVs (`write_loop_uvs`), and `custom_normal` is re-encoded from the subdivided loops.
pub(super) const WRITTEN_KEYS: [&str; 3] = ["crease_edge", uv_layers::ATTRIBUTE, "custom_normal"];

/// Keeps only the [`WRITTEN_KEYS`] attributes. The level loop reads nothing else, so every other store is
/// remapped once from the input after the last level.
pub(super) fn retain_written(mesh: &mut Mesh) {
    mesh.attributes
        .retain(|key, _| WRITTEN_KEYS.contains(&key.as_str()));
}

/// Removes the [`WRITTEN_KEYS`] entries from `mesh`, returning them for [`restore_written`].
pub(super) fn take_written(mesh: &mut Mesh) -> Vec<(&'static str, Value)> {
    WRITTEN_KEYS
        .iter()
        .filter_map(|&key| mesh.attributes.remove(key).map(|value| (key, value)))
        .collect()
}

/// Puts back the entries that [`take_written`] removed.
pub(super) fn restore_written(mesh: &mut Mesh, written: Vec<(&'static str, Value)>) {
    for (key, value) in written {
        mesh.attributes.insert(key.to_owned(), value);
    }
}

/// Each output edge's source as an explicit `(input edge, weight)` list for the remap: a subedge copies its
/// coarse edge (`subdiv_mesh_edge`, `subdiv_mesh.cc:1226`), and an internal edge has none (`subdiv_copy_edge_data`,
/// `subdiv_mesh.cc:1199`). `edge_sources` is the order builder's coarse-edge map.
pub(super) fn edge_origins(edge_sources: Vec<Option<usize>>) -> Vec<Vec<(usize, f64)>> {
    edge_sources
        .into_iter()
        .map(|source| source.map_or_else(Vec::new, |edge| vec![(edge, 1.0)]))
        .collect()
}

/// The origin and staged mixes of every element of `output`, the subdivided mesh before its storage is
/// permuted into Blender's order. `polygons` and `loop_starts` describe `mesh` as subdivision read it;
/// `samples` and `loose` place each output element on `mesh`. Corner sources are built only when the remap
/// reads them.
pub(super) fn build(
    mesh: &Mesh,
    polygons: &[Polygon],
    loop_starts: &[usize],
    samples: &Samples,
    loose: &LooseEdges,
    output: &Mesh,
) -> Result<(Origin, Staged)> {
    let consistent = samples.vertices.len() == output.vertices.len()
        && samples.corners.len() == output.faces.len()
        && samples
            .corners
            .iter()
            .zip(&output.faces)
            .all(|(corners, face)| corners.len() == face.vertices.len());
    if !consistent {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "subdivision samples do not describe the subdivided mesh",
        ));
    }
    let mut vertex_stages: Vec<Option<Mixing>> = Vec::with_capacity(samples.vertices.len());
    let mut vertex_sources: Vec<Vec<(usize, f64)>> = Vec::with_capacity(samples.vertices.len());
    for (vertex, sample) in samples.vertices.iter().enumerate() {
        if let Some(sample) = sample {
            let face = &polygons[sample.face].0;
            let mixed = remap_sources(
                slots(*sample, polygons, loop_starts, Element::Vertex),
                |corner| face[corner],
            );
            vertex_sources.push(Staged::flatten(&mixed));
            vertex_stages.push((!is_copy(&mixed)).then_some(Mixing::Slots(mixed)));
        } else {
            let sources = match loose.sources(vertex) {
                Some(sources) => sources,
                // A base vertex on no face and no loose edge is copied by `subdiv_mesh_vert_loose`.
                None if vertex < mesh.vertices.len() => vec![(vertex, 1.0)],
                None => Vec::new(),
            };
            // An interior loose-edge vertex is one `mix2` of its edge's ends; an end point is a copy.
            vertex_stages.push((sources.len() == 2).then(|| Mixing::Pair(sources.clone())));
            vertex_sources.push(sources);
        }
    }
    // Corner stages exist only when the remap reads corner sources; otherwise the origin carries an empty list.
    let corner_stages: Option<Vec<Vec<Stage>>> =
        needs_corner_sources(mesh, &WRITTEN_KEYS).then(|| {
            let offsets: Vec<usize> = polygons
                .iter()
                .scan(0, |total, (vertices, _)| {
                    let start = *total;
                    *total += vertices.len();
                    Some(start)
                })
                .collect();
            samples
                .corners
                .iter()
                .flatten()
                .map(|sample| {
                    remap_sources(
                        slots(*sample, polygons, loop_starts, Element::Corner),
                        |corner| offsets[sample.face] + corner,
                    )
                })
                .collect()
        });
    let corner_sources: Vec<Vec<(usize, f64)>> =
        corner_stages.as_ref().map_or_else(Vec::new, |stages| {
            stages
                .iter()
                .map(|stages| Staged::flatten(stages))
                .collect()
        });
    let origin = Origin {
        vertices: vertex_sources,
        faces: samples
            .corners
            .iter()
            .map(|corners| corners.first().map(|sample| sample.face))
            .collect(),
        corners: Some(corner_sources),
        ..Default::default()
    };
    let staged = Staged {
        vertices: vertex_stages,
        corners: corner_stages,
    };
    Ok((origin, staged))
}

/// Maps the storage corners of a stage list to other indices (base vertices or flat output corners).
fn remap_sources(stages: Vec<Stage>, index: impl Fn(usize) -> usize) -> Vec<Stage> {
    stages
        .into_iter()
        .map(|stage| Stage {
            weight: stage.weight,
            sources: stage
                .sources
                .into_iter()
                .map(|(corner, weight)| (index(corner), weight))
                .collect(),
        })
        .collect()
}

/// A sample on a coarse corner: one full-weight source. Blender copies such a vertex (`vert_corner`), so
/// it is not staged.
fn is_copy(stages: &[Stage]) -> bool {
    matches!(stages, [stage] if stage.weight > 1.0 - 1.0e-9 && stage.sources.len() == 1)
}

/// The element a slot set is mixed for. Blender builds the third slot's two-source midpoint in a different order
/// for vertices and loops: `last_indices = {first_loop, last_loop}` for a vertex (`subdiv_mesh.cc:535`) and
/// `{last_loop, first_loop}` for a loop (`subdiv_mesh.cc:691`). Slerp depends on that order when the two
/// quaternions have a negative dot product.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Element {
    Vertex,
    Corner,
}

/// Blender's mixing slots of a face-sampled element, in the sample's loop frame (loop `j` is storage corner
/// `(start + j) mod size`), as stages over the face's storage corners. Zero-weight stages are dropped.
fn slots(
    sample: Sample,
    polygons: &[Polygon],
    loop_starts: &[usize],
    element: Element,
) -> Vec<Stage> {
    let size = polygons[sample.face].0.len();
    let start = loop_starts[sample.face];
    let local = sample.in_loop_frame(start, size);
    let (u, v) = (local.u, local.v);
    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
    let corner = |loop_index: usize| (start + loop_index) % size;
    let mut stages: Vec<Stage> = if size == 4 {
        weights
            .into_iter()
            .enumerate()
            .map(|(loop_index, weight)| Stage {
                weight,
                sources: vec![(corner(loop_index), 1.0)],
            })
            .collect()
    } else {
        let ptex = local.ptex;
        let next = (ptex + 1) % size;
        let previous = (ptex + size - 1) % size;
        let centre = 1.0 / usize_as_f64(size);
        vec![
            Stage {
                weight: weights[0],
                sources: vec![(corner(ptex), 1.0)],
            },
            Stage {
                weight: weights[1],
                sources: vec![(corner(ptex), 0.5), (corner(next), 0.5)],
            },
            Stage {
                weight: weights[2],
                sources: (0..size)
                    .map(|loop_index| (corner(loop_index), centre))
                    .collect(),
            },
            Stage {
                weight: weights[3],
                sources: match element {
                    Element::Vertex => vec![(corner(ptex), 0.5), (corner(previous), 0.5)],
                    Element::Corner => vec![(corner(previous), 0.5), (corner(ptex), 0.5)],
                },
            },
        ]
    };
    stages.retain(|stage| stage.weight > 0.0);
    stages
}
