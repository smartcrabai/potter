// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org) source/blender/blenkernel/intern/subdiv_mesh.cc, GPL-2.0-or-later.

//! Element origins of a subdivided mesh under Blender's interpolation rules, for the attribute remap.
//!
//! Every output element has a ptex sample of its base face (`Samples`). Blender mixes its data from the
//! coarse corners of that face with the same four ptex slots that place its position.
//! `quad_weights_from_uv` (`subdiv_mesh.cc:966`) weights the slots at ptex corners (0,0), (1,0), (1,1),
//! and (0,1). A quad's slots are its corners (`vert_interpolation_from_face`, `subdiv_mesh.cc:452`). An
//! n-gon's slots for ptex face `p` are corner `p`, the midpoints of `(p, p+1)` and `(p-1, p)`, and the face
//! centre, which mixes all corners with equal weight (`vert_interpolation_from_corner`, `subdiv_mesh.cc:498`;
//! `loop_interpolation_from_corner`, `subdiv_mesh.cc:657`; centre in `vert_interpolation_from_face`,
//! `subdiv_mesh.cc:452`, and `loop_interpolation_from_face`, `subdiv_mesh.cc:611`). Vertices and corners
//! use these slots alike, so both are weights over the base face's storage vertices and corners.
//!
//! Edges follow `subdiv_mesh_edge` (`subdiv_mesh.cc:1226`): a subedge copies its coarse edge's data, and an
//! internal edge has none (`subdiv_copy_edge_data`, `subdiv_mesh.cc:1199`). Each subdivided face copies its
//! coarse face (`subdiv_mesh_face`, `subdiv_mesh.cc:1331`). A loose-edge vertex mixes its coarse edge's
//! stored ends by `(1 - u, u)` (`subdiv_mesh_vert_of_loose_edge_interpolate`, `subdiv_mesh.cc:1434`), and a
//! loose end point copies its vertex (`subdiv_mesh_vert_loose`, `subdiv_mesh.cc:1357`).

use std::collections::HashMap;

use serde_json::{Map, Value};

use super::subdivision_limit::{Sample, Samples};
use super::subdivision_loose::LooseEdges;
use super::{Polygon, blend_weights, usize_as_f64};
use crate::error::{ErrorCode, PotError, Result};
use crate::geom::provenance::Origin;
use crate::geom::{Edge, Mesh, edge_key, uv_layers};

/// Keys subdivision writes itself, so the remap skips them and their output entries are kept.
/// `crease_edge` holds the subdivided creases (`subdivision_uv::subdivided_mesh`), `uv_layers` the limit
/// UVs (`write_loop_uvs`), and `custom_normal` is re-encoded from the subdivided loops.
pub(super) const WRITTEN_KEYS: [&str; 3] = ["crease_edge", uv_layers::ATTRIBUTE, "custom_normal"];

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

/// The origin of every element of `output`, the subdivided mesh before its storage is permuted into
/// Blender's order. `polygons`, `edges`, and `loop_starts` describe `mesh` as subdivision read it, and
/// `samples` and `loose` place each output element on it. Indices are pre-permutation output storage.
pub(super) fn build(
    mesh: &Mesh,
    polygons: &[Polygon],
    edges: &[[usize; 2]],
    loop_starts: &[usize],
    samples: &Samples,
    loose: &LooseEdges,
    output: &Mesh,
) -> Result<Origin> {
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
    let offsets: Vec<usize> = polygons
        .iter()
        .scan(0, |total, (vertices, _)| {
            let start = *total;
            *total += vertices.len();
            Some(start)
        })
        .collect();
    let vertex_sources: Vec<Vec<(usize, f64)>> = samples
        .vertices
        .iter()
        .enumerate()
        .map(|(vertex, sample)| match sample {
            Some(sample) => blend_weights(
                slots(*sample, polygons, loop_starts)
                    .into_iter()
                    .map(|(corner, weight)| (polygons[sample.face].0[corner], weight)),
            ),
            None => match loose.sources(vertex) {
                Some(sources) => sources,
                // A base vertex on no face and no loose edge is copied by `subdiv_mesh_vert_loose`.
                None if vertex < mesh.vertices.len() => vec![(vertex, 1.0)],
                None => Vec::new(),
            },
        })
        .collect();
    let corner_sources: Vec<Vec<(usize, f64)>> = samples
        .corners
        .iter()
        .flatten()
        .map(|sample| {
            blend_weights(
                slots(*sample, polygons, loop_starts)
                    .into_iter()
                    .map(|(corner, weight)| (offsets[sample.face] + corner, weight)),
            )
        })
        .collect();
    let edge_index: HashMap<(usize, usize), usize> = {
        let mut index = HashMap::with_capacity(edges.len());
        for (position, &[first, second]) in edges.iter().enumerate() {
            index.entry(edge_key(first, second)).or_insert(position);
        }
        index
    };
    let vertex_of_id: HashMap<u32, usize> = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    // An output edge lies on a base edge when both ends blend only that edge's two vertices (its
    // subedges, Blender's boundary copies). Any other output edge has no source.
    let source_edge = |edge: &Edge| -> Option<usize> {
        let first = *vertex_of_id.get(&edge.vertices[0])?;
        let second = *vertex_of_id.get(&edge.vertices[1])?;
        let mut ends: Vec<usize> = vertex_sources[first]
            .iter()
            .chain(&vertex_sources[second])
            .map(|(source, _)| *source)
            .collect();
        ends.sort_unstable();
        ends.dedup();
        match ends[..] {
            [a, b] => edge_index.get(&edge_key(a, b)).copied(),
            _ => None,
        }
    };
    let edge_sources: Vec<Option<usize>> = output.edges.iter().map(source_edge).collect();
    Ok(Origin {
        vertices: vertex_sources,
        faces: samples
            .corners
            .iter()
            .map(|corners| corners.first().map(|sample| sample.face))
            .collect(),
        corners: Some(corner_sources),
        edges: Some(edge_sources),
    })
}

/// The storage corners of a base face whose data a sample mixes, with Blender's slot weights. The
/// sample is taken in Blender's loop frame, where loop `j` is storage corner `(start + j) mod size`.
fn slots(sample: Sample, polygons: &[Polygon], loop_starts: &[usize]) -> Vec<(usize, f64)> {
    let size = polygons[sample.face].0.len();
    let start = loop_starts[sample.face];
    let local = sample.in_loop_frame(start, size);
    let (u, v) = (local.u, local.v);
    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
    let corner = |loop_index: usize| (start + loop_index) % size;
    if size == 4 {
        return weights
            .into_iter()
            .enumerate()
            .map(|(loop_index, weight)| (corner(loop_index), weight))
            .collect();
    }
    let ptex = local.ptex;
    let next = (ptex + 1) % size;
    let previous = (ptex + size - 1) % size;
    let centre = weights[2] / usize_as_f64(size);
    let mut mixed = vec![
        (corner(ptex), weights[0]),
        (corner(ptex), 0.5 * weights[1]),
        (corner(next), 0.5 * weights[1]),
        (corner(previous), 0.5 * weights[3]),
        (corner(ptex), 0.5 * weights[3]),
    ];
    mixed.extend((0..size).map(|loop_index| (corner(loop_index), centre)));
    mixed
}

/// Moves the positional legacy arrays that `remap_attributes` wrote in pre-permutation order into
/// Blender's order: `blender_edge_flags` by edge, `blender_polygon_smooth` by face, and
/// `blender_skin_vertices` plus the vertex-keyed `skin_radii` and `skin_roots` by vertex. Each `order`
/// maps a Blender index to its pre-permutation index. `blender_attributes` is reordered by
/// `subdivision_order::reorder_blender_attributes`.
pub(super) fn reorder_legacy(
    mesh: &mut Mesh,
    vertices: &[usize],
    edges: &[usize],
    faces: &[usize],
) {
    for (key, order) in [
        ("blender_edge_flags", edges),
        ("blender_polygon_smooth", faces),
        ("blender_skin_vertices", vertices),
    ] {
        if let Some(items) = mesh.attributes.get_mut(key).and_then(Value::as_array_mut) {
            permute_list(items, order);
        }
    }
    let mut position = vec![0; vertices.len()];
    for (new, &old) in vertices.iter().enumerate() {
        position[old] = new;
    }
    for key in ["skin_radii", "skin_roots"] {
        if let Some(object) = mesh.attributes.get_mut(key).and_then(Value::as_object_mut) {
            rekey(object, &position);
        }
    }
}

fn permute_list(items: &mut Vec<Value>, order: &[usize]) {
    if items.len() == order.len() {
        *items = order.iter().map(|&index| items[index].clone()).collect();
    }
}

/// Re-keys a `"{index}"`-keyed object from pre-permutation to Blender vertex indices.
fn rekey(object: &mut Map<String, Value>, position: &[usize]) {
    let rekeyed: Map<String, Value> = object
        .iter()
        .map(|(key, value)| {
            let key = key
                .parse::<usize>()
                .ok()
                .and_then(|old| position.get(old))
                .map_or_else(|| key.clone(), |&new| new.to_string());
            (key, value.clone())
        })
        .collect();
    *object = rekeyed;
}
