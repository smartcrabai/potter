// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_foreach.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.

//! Blender's output array order for one subdivided mesh. `subdiv_foreach_ctx_init_offsets` lays out
//! coarse vertices, per-coarse-edge subdivision vertices, and per-coarse-face interior vertices;
//! boundary edges are indexed by coarse-edge storage order, internal edges by coarse-face callback
//! order, and faces/loops by the loop emission index. These permutations map Potter's level-staged
//! output arrays into that layout without comparing positions at runtime.

use std::collections::HashMap;

use crate::geom::{Mesh, edge_key};

use super::{Polygon, subdivision_limit::Samples};

/// Storage permutations and loop/edge endpoint remaps from Potter's current output into Blender's
/// emitted order. Each vector maps a Blender destination index to an index in the current mesh.
pub(super) struct OutputOrder {
    pub(super) vertices: Vec<usize>,
    pub(super) edges: Vec<usize>,
    pub(super) edge_vertices: Vec<[usize; 2]>,
    pub(super) faces: Vec<usize>,
    pub(super) corners: Vec<usize>,
    /// Per face in Potter's pre-permutation storage order.
    pub(super) face_starts: Vec<usize>,
    /// Per output edge in pre-permutation storage: the base edge it copies, or `None` if inner.
    pub(super) edge_sources: Vec<Option<usize>>,
}

/// Builds Blender's order from the base topology and the final ptex samples.
///
/// `base_edges` retains Blender's coarse edge storage order and orientation. `loop_starts` is the
/// first-loop corner of each input face. `loose_targets[vertex]` is Blender's vertex index for a
/// loose-only vertex, if any.
pub(super) fn build(
    base_faces: &[Polygon],
    base_edges: &[[usize; 2]],
    base_vertex_count: usize,
    loop_starts: &[usize],
    samples: &Samples,
    ordinals: &[usize],
    loose_targets: &[Option<usize>],
    output: &Mesh,
    levels: u32,
) -> Result<OutputOrder, &'static str> {
    let grid = 1_usize
        .checked_shl(levels)
        .ok_or("subdivision grid overflow")?;
    let resolution = grid + 1;
    let edge_points_per_edge = resolution - 2;
    let face_vertex_offset = base_vertex_count + base_edges.len() * edge_points_per_edge;
    let mut face_vertex_starts = Vec::with_capacity(base_faces.len());
    let mut face_vertex_count = 0;
    for (face, _) in base_faces {
        face_vertex_starts.push(face_vertex_offset + face_vertex_count);
        let count = if face.len() == 4 {
            edge_points_per_edge * edge_points_per_edge
        } else {
            let patch_resolution = (resolution >> 1) + 1;
            1 + face.len() * (patch_resolution - 1) * (patch_resolution - 2)
        };
        face_vertex_count += count;
    }

    let edge_index: HashMap<(usize, usize), usize> = base_edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge[0], edge[1]), index))
        .collect();
    let mut vertex_new_to_old = vec![usize::MAX; output.vertices.len()];
    for old in 0..base_vertex_count.min(output.vertices.len()) {
        put(&mut vertex_new_to_old, old, old)?;
    }
    for (old, sample) in samples.vertices.iter().enumerate().skip(base_vertex_count) {
        let target = match sample {
            Some(sample) => sample_vertex(
                *sample,
                base_faces,
                base_edges,
                &edge_index,
                loop_starts,
                base_vertex_count,
                edge_points_per_edge,
                resolution,
                &face_vertex_starts,
            )?,
            None => loose_targets
                .get(old)
                .copied()
                .flatten()
                .ok_or("loose output vertex has no Blender-order index")?,
        };
        put(&mut vertex_new_to_old, target, old)?;
    }
    if vertex_new_to_old.contains(&usize::MAX) {
        return Err("Blender vertex order is not a permutation of the output vertices");
    }
    let mut vertex_old_to_new = vec![usize::MAX; vertex_new_to_old.len()];
    for (new, &old) in vertex_new_to_old.iter().enumerate() {
        vertex_old_to_new[old] = new;
    }

    let output_face_count = output.faces.len();
    if ordinals.len() != output_face_count * 4 {
        return Err("loop ordinals do not cover four corners per output face");
    }
    let mut face_new_to_old = vec![usize::MAX; output_face_count];
    let mut corner_new_to_old = vec![usize::MAX; ordinals.len()];
    let mut face_starts = Vec::with_capacity(output_face_count);
    let mut corner_offset = 0;
    for (old_face, face) in output.faces.iter().enumerate() {
        let count = face.vertices.len();
        if count != 4 {
            return Err("subdivision output face is not a quad");
        }
        let start = (0..count)
            .min_by_key(|&corner| ordinals[corner_offset + corner])
            .unwrap_or(0);
        face_starts.push(start);
        let face_rank = ordinals[corner_offset] / 4;
        put(&mut face_new_to_old, face_rank, old_face)?;
        for new_corner in 0..count {
            let old_corner = corner_offset + (new_corner + start) % count;
            put(&mut corner_new_to_old, ordinals[old_corner], old_corner)?;
        }
        corner_offset += count;
    }
    if face_new_to_old.contains(&usize::MAX) || corner_new_to_old.contains(&usize::MAX) {
        return Err("Blender face/loop order is not a permutation of the output");
    }

    let boundary_edge_count = base_edges.len() * (resolution - 1);
    let mut edge_endpoints = Vec::with_capacity(output.edges.len());
    for (edge, &[first, second]) in base_edges.iter().enumerate() {
        for segment in 0..resolution - 1 {
            edge_endpoints.push([
                edge_vertex(first, second, edge, segment, resolution, base_vertex_count),
                edge_vertex(
                    first,
                    second,
                    edge,
                    segment + 1,
                    resolution,
                    base_vertex_count,
                ),
            ]);
        }
    }
    for (face_index, (face, _)) in base_faces.iter().enumerate() {
        let start = loop_starts[face_index];
        let vertices = rotate(face, start);
        let mut face_edge = Vec::with_capacity(vertices.len());
        for (corner, &vertex) in vertices.iter().enumerate() {
            face_edge.push(
                *edge_index
                    .get(&edge_key(vertex, vertices[(corner + 1) % vertices.len()]))
                    .ok_or("base face edge is missing from the edge order")?,
            );
        }
        let face_base = face_vertex_starts[face_index];
        if vertices.len() == 4 {
            append_quad_edges(
                &mut edge_endpoints,
                face_base,
                &vertices,
                &face_edge,
                base_edges,
                base_vertex_count,
                resolution,
            );
        } else {
            append_ngon_edges(
                &mut edge_endpoints,
                face_base,
                &vertices,
                &face_edge,
                base_edges,
                base_vertex_count,
                resolution,
            );
        }
    }
    if edge_endpoints.len() != output.edges.len() {
        return Err("Blender edge order does not match the output edge count");
    }
    let output_id_to_old: HashMap<u32, usize> = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut expected_edges = HashMap::with_capacity(edge_endpoints.len());
    for (index, &[a, b]) in edge_endpoints.iter().enumerate() {
        let key = edge_key(a, b);
        if expected_edges.insert(key, (index, [a, b])).is_some() {
            return Err("Blender edge order contains duplicate endpoint pairs");
        }
    }
    let mut edge_new_to_old = vec![usize::MAX; output.edges.len()];
    let mut edge_target_endpoints = vec![[usize::MAX; 2]; output.edges.len()];
    for (old_edge, edge) in output.edges.iter().enumerate() {
        let old_a = *output_id_to_old
            .get(&edge.vertices[0])
            .ok_or("output edge has a missing first endpoint")?;
        let old_b = *output_id_to_old
            .get(&edge.vertices[1])
            .ok_or("output edge has a missing second endpoint")?;
        let key = edge_key(vertex_old_to_new[old_a], vertex_old_to_new[old_b]);
        let &(new_edge, endpoints) = expected_edges
            .get(&key)
            .ok_or("output edge is absent from Blender's edge order")?;
        put(&mut edge_new_to_old, new_edge, old_edge)?;
        edge_target_endpoints[new_edge] = endpoints;
    }
    if edge_new_to_old.contains(&usize::MAX) {
        return Err("Blender edge order is not a permutation of the output edges");
    }
    // subdiv_foreach_boundary_edges (subdiv_foreach.cc L941-966) emits `resolution - 1` edges per coarse
    // edge with that coarse index. Inner edges carry ORIGINDEX_NONE, and subdiv_copy_edge_data
    // (subdiv_mesh.cc L1199-1222) gives them defaults instead of copying.
    let mut edge_sources = vec![None; output.edges.len()];
    for (new_edge, &old_edge) in edge_new_to_old.iter().enumerate() {
        edge_sources[old_edge] =
            (new_edge < boundary_edge_count).then(|| new_edge / (resolution - 1));
    }
    Ok(OutputOrder {
        vertices: vertex_new_to_old,
        edges: edge_new_to_old,
        edge_vertices: edge_target_endpoints,
        faces: face_new_to_old,
        corners: corner_new_to_old,
        face_starts,
        edge_sources,
    })
}

fn put(values: &mut [usize], index: usize, value: usize) -> Result<(), &'static str> {
    let slot = values
        .get_mut(index)
        .ok_or("Blender element index is out of range")?;
    if *slot != usize::MAX {
        return Err("Blender element order has a duplicate index");
    }
    *slot = value;
    Ok(())
}

/// `items` in the order `order` lists their indices. `order` is a permutation of `0..items.len()` (checked by
/// [`build`]), so each item moves once and nothing is cloned.
pub(super) fn permute<T>(items: Vec<T>, order: &[usize]) -> Vec<T> {
    let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
    order
        .iter()
        .filter_map(|&index| slots.get_mut(index)?.take())
        .collect()
}

fn rotate(values: &[usize], start: usize) -> Vec<usize> {
    (0..values.len())
        .map(|index| values[(index + start) % values.len()])
        .collect()
}

/// A base face read from its first loop: `Rotated[k]` is the vertex of loop `k`, without copying the face.
struct Rotated<'a> {
    face: &'a [usize],
    start: usize,
}

impl Rotated<'_> {
    fn len(&self) -> usize {
        self.face.len()
    }
}

impl std::ops::Index<usize> for Rotated<'_> {
    type Output = usize;

    fn index(&self, index: usize) -> &usize {
        &self.face[(index + self.start) % self.face.len()]
    }
}

fn sample_vertex(
    sample: super::subdivision_limit::Sample,
    faces: &[Polygon],
    edges: &[[usize; 2]],
    edge_index: &HashMap<(usize, usize), usize>,
    loop_starts: &[usize],
    base_vertex_count: usize,
    edge_points_per_edge: usize,
    resolution: usize,
    face_vertex_starts: &[usize],
) -> Result<usize, &'static str> {
    let (face, _) = &faces[sample.face];
    let vertices = Rotated {
        face: face.as_slice(),
        start: loop_starts[sample.face],
    };
    // Blender's ptex indices count from the face's first loop, so quads and n-gons both read the loop frame.
    let sample = sample.in_loop_frame(loop_starts[sample.face], vertices.len());
    if vertices.len() == 4 {
        let denominator = resolution - 1;
        let x = (sample.u * denominator as f64).round() as usize;
        let y = (sample.v * denominator as f64).round() as usize;
        let last = denominator;
        if x == 0 && y == 0 {
            return Ok(vertices[0]);
        }
        if x == last && y == 0 {
            return Ok(vertices[1]);
        }
        if x == last && y == last {
            return Ok(vertices[2]);
        }
        if x == 0 && y == last {
            return Ok(vertices[3]);
        }
        let boundary = if y == 0 {
            Some((vertices[0], vertices[1], x as f64 / denominator as f64))
        } else if x == last {
            Some((vertices[1], vertices[2], y as f64 / denominator as f64))
        } else if y == last {
            Some((vertices[3], vertices[2], x as f64 / denominator as f64))
        } else if x == 0 {
            Some((vertices[0], vertices[3], y as f64 / denominator as f64))
        } else {
            None
        };
        if let Some((first, second, t)) = boundary {
            return edge_sample(
                first,
                second,
                t,
                edges,
                edge_index,
                base_vertex_count,
                resolution,
            );
        }
        let width = edge_points_per_edge;
        return Ok(face_vertex_starts[sample.face] + (y - 1) * width + (x - 1));
    }
    let patch_resolution = (resolution >> 1) + 1;
    let denominator = patch_resolution - 1;
    let mut ptex = sample.ptex;
    let mut x = (sample.u * denominator as f64).round() as usize;
    let mut y = (sample.v * denominator as f64).round() as usize;
    let last = denominator;
    let n = vertices.len();
    if x == 0 && y == 0 {
        return Ok(vertices[ptex]);
    }
    if y == 0 {
        return edge_sample(
            vertices[ptex],
            vertices[(ptex + 1) % n],
            0.5 * sample.u,
            edges,
            edge_index,
            base_vertex_count,
            resolution,
        );
    }
    if x == 0 {
        return edge_sample(
            vertices[(ptex + n - 1) % n],
            vertices[ptex],
            1.0 - 0.5 * sample.v,
            edges,
            edge_index,
            base_vertex_count,
            resolution,
        );
    }
    if x == last && y == last {
        return Ok(face_vertex_starts[sample.face]);
    }
    // `subdiv_foreach_inner_vertices_special` omits the top edge of each ptex patch and stores that
    // radial seam as the right edge of the previous ptex patch.
    if y == last {
        ptex = (ptex + n - 1) % n;
        y = x;
        x = last;
    }
    let per_patch = (patch_resolution - 1) * (patch_resolution - 2);
    let row_width = patch_resolution - 1;
    Ok(face_vertex_starts[sample.face] + 1 + ptex * per_patch + (y - 1) * row_width + (x - 1))
}

fn edge_sample(
    first: usize,
    second: usize,
    t: f64,
    edges: &[[usize; 2]],
    edge_index: &HashMap<(usize, usize), usize>,
    base_vertex_count: usize,
    resolution: usize,
) -> Result<usize, &'static str> {
    let edge = *edge_index
        .get(&edge_key(first, second))
        .ok_or("sample edge is absent from the base edge order")?;
    let [edge_first, edge_second] = edges[edge];
    let t = if edge_first == first && edge_second == second {
        t
    } else {
        1.0 - t
    };
    let segment = (t * (resolution - 1) as f64).round() as usize;
    Ok(edge_vertex(
        edge_first,
        edge_second,
        edge,
        segment,
        resolution,
        base_vertex_count,
    ))
}

/// The output vertex at `segment` of base edge `edge` (`first` to `second`), where segment 0 and the last
/// segment are the edge's endpoints and the rest are the edge's `resolution - 2` interior points.
pub(super) fn edge_vertex(
    first: usize,
    second: usize,
    edge: usize,
    segment: usize,
    resolution: usize,
    base_vertices: usize,
) -> usize {
    if segment == 0 {
        first
    } else if segment == resolution - 1 {
        second
    } else {
        base_vertices + edge * (resolution - 2) + segment - 1
    }
}

fn append_row(edges: &mut Vec<[usize; 2]>, start: usize, width: usize) {
    for index in 0..width.saturating_sub(1) {
        edges.push([start + index, start + index + 1]);
    }
}

fn append_column(edges: &mut Vec<[usize; 2]>, start: usize, width: usize) {
    for index in 0..width {
        edges.push([start + index, start + index + width]);
    }
}

fn append_quad_edges(
    edges: &mut Vec<[usize; 2]>,
    face_base: usize,
    face: &[usize],
    face_edges: &[usize],
    base_edges: &[[usize; 2]],
    base_vertices: usize,
    resolution: usize,
) {
    let width = resolution - 2;
    append_row(edges, face_base, width);
    for row in 0..width.saturating_sub(1) {
        let start = face_base + row * width;
        append_column(edges, start, width);
        append_row(edges, start + width, width);
    }
    for (corner, &edge) in face_edges.iter().enumerate() {
        let (side_start, stride) = match corner {
            0 => (face_base as isize, 1),
            1 => ((face_base + resolution - 3) as isize, width as isize),
            2 => ((face_base + width * width - 1) as isize, -1),
            _ => (
                (face_base + width * (width - 1)) as isize,
                -(width as isize),
            ),
        };
        let flip = base_edges[edge][1] == face[corner];
        let edge_start = base_vertices + edge * width;
        for index in 0..width {
            let boundary = edge_start + if flip { width - index - 1 } else { index };
            let inner = (side_start + stride * index as isize) as usize;
            edges.push([boundary, inner]);
        }
    }
}

fn inner_edges_per_patch(resolution: usize) -> usize {
    if resolution < 2 {
        0
    } else {
        (resolution - 2) * resolution + (resolution - 1) * (resolution - 1)
    }
}

fn append_ngon_edges(
    edges: &mut Vec<[usize; 2]>,
    face_base: usize,
    face: &[usize],
    face_edges: &[usize],
    base_edges: &[[usize; 2]],
    base_vertices: usize,
    resolution: usize,
) {
    let patch_resolution = (resolution >> 1) + 1;
    let inner = patch_resolution - 2;
    let width = patch_resolution - 1;
    let per_patch = width * inner;
    let patch_start = |corner: usize| face_base + 1 + corner * per_patch;
    for corner in 0..face.len() {
        let start = patch_start(corner);
        append_row(edges, start, inner + 1);
        for row in 0..inner.saturating_sub(1) {
            let row_start = start + row * width;
            append_column(edges, row_start, width);
            append_row(edges, row_start + width, width);
        }
    }
    for corner in 0..face.len() {
        let current = patch_start(corner) + inner;
        let next = patch_start((corner + 1) % face.len()) + per_patch - patch_resolution + 1;
        for row in 0..inner {
            edges.push([current + row * width, next + row]);
        }
    }
    if patch_resolution >= 3 {
        for corner in 0..face.len() {
            edges.push([face_base, patch_start(corner) + per_patch - 1]);
        }
    }
    for (corner, &edge) in face_edges.iter().enumerate() {
        let flip = base_edges[edge][1] == face[corner];
        let edge_start = base_vertices + edge * (resolution - 2);
        let start = patch_start(corner);
        for index in 0..patch_resolution - 1 {
            let boundary = edge_start + if flip { resolution - index - 3 } else { index };
            let inner_vertex = if patch_resolution >= 3 {
                start + index
            } else {
                face_base
            };
            edges.push([boundary, inner_vertex]);
        }
        if patch_resolution >= 3 {
            let previous_edge = face_edges[(corner + face.len() - 1) % face.len()];
            let previous_flip = base_edges[previous_edge][1] == face[corner];
            let previous_start = base_vertices + previous_edge * (resolution - 2);
            for index in 0..inner {
                let boundary = previous_start
                    + if previous_flip {
                        resolution - index - 3
                    } else {
                        index
                    };
                edges.push([boundary, start + width * index]);
            }
        }
    }
}

/// Moves every positional array that `remap_attributes` wrote in pre-permutation storage into Blender's order:
/// `blender_attributes` entries by their domain, `blender_edge_flags` by edge, `blender_polygon_smooth` by face,
/// and `blender_skin_vertices` by vertex. The vertex-keyed `skin_radii` and `skin_roots` are re-keyed to the
/// new vertex indices.
pub(super) fn reorder_positional_attributes(mesh: &mut Mesh, order: &OutputOrder) {
    for (key, permutation) in [
        ("blender_edge_flags", &order.edges),
        ("blender_polygon_smooth", &order.faces),
        ("blender_skin_vertices", &order.vertices),
    ] {
        if let Some(items) = mesh
            .attributes
            .get_mut(key)
            .and_then(serde_json::Value::as_array_mut)
            && items.len() == permutation.len()
        {
            *items = permute(std::mem::take(items), permutation);
        }
    }
    if let Some(entries) = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(serde_json::Value::as_array_mut)
    {
        for entry in entries {
            let permutation = match entry.get("domain").and_then(serde_json::Value::as_str) {
                Some("POINT") => &order.vertices,
                Some("EDGE") => &order.edges,
                Some("FACE") => &order.faces,
                _ => continue,
            };
            let Some(values) = entry
                .get_mut("values")
                .and_then(serde_json::Value::as_array_mut)
            else {
                continue;
            };
            if values.len() == permutation.len() {
                *values = permute(std::mem::take(values), permutation);
            }
        }
    }
    let mut position = vec![0; order.vertices.len()];
    for (new, &old) in order.vertices.iter().enumerate() {
        position[old] = new;
    }
    for key in ["skin_radii", "skin_roots"] {
        if let Some(object) = mesh
            .attributes
            .get_mut(key)
            .and_then(serde_json::Value::as_object_mut)
        {
            rekey(object, &position);
        }
    }
}

/// Re-keys a `"{index}"`-keyed object from pre-permutation to Blender vertex indices.
fn rekey(object: &mut serde_json::Map<String, serde_json::Value>, position: &[usize]) {
    *object = std::mem::take(object)
        .into_iter()
        .map(|(key, value)| {
            let key = key
                .parse::<usize>()
                .ok()
                .and_then(|old| position.get(old))
                .map_or(key, |&new| new.to_string());
            (key, value)
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{OutputOrder, permute, reorder_positional_attributes};
    use crate::geom::Mesh;

    #[test]
    fn positional_blender_entries_follow_element_permutations() {
        let mut mesh = Mesh::new();
        mesh.attributes.insert(
            "blender_attributes".to_owned(),
            json!([
                {"name": "p", "domain": "POINT", "data_type": "FLOAT", "values": [0.0, 1.0, 2.0]},
                {"name": "e", "domain": "EDGE", "data_type": "FLOAT", "values": [10.0, 11.0]},
                {"name": "f", "domain": "FACE", "data_type": "FLOAT", "values": [20.0]},
                {"name": "c", "domain": "CORNER", "data_type": "FLOAT", "values": [30.0, 31.0, 32.0, 33.0]},
                {"name": "short", "domain": "POINT", "data_type": "FLOAT", "values": [7.0]}
            ]),
        );
        let order = OutputOrder {
            vertices: vec![2, 0, 1],
            edges: vec![1, 0],
            edge_vertices: Vec::new(),
            faces: vec![0],
            corners: Vec::new(),
            face_starts: Vec::new(),
            edge_sources: Vec::new(),
        };
        reorder_positional_attributes(&mut mesh, &order);
        let entries = mesh.attributes["blender_attributes"]
            .as_array()
            .expect("entries stay an array");
        assert_eq!(entries[0]["values"], json!([2.0, 0.0, 1.0]));
        assert_eq!(entries[1]["values"], json!([11.0, 10.0]));
        assert_eq!(entries[2]["values"], json!([20.0]));
        // Corner rows are permuted by the corner order, not by this function.
        assert_eq!(entries[3]["values"], json!([30.0, 31.0, 32.0, 33.0]));
        // A length that matches no element domain is left as stored.
        assert_eq!(entries[4]["values"], json!([7.0]));
    }

    #[test]
    fn permute_moves_items_by_index() {
        assert_eq!(
            permute(vec!["a", "b", "c"], &[2, 0, 1]),
            vec!["c", "a", "b"]
        );
    }
}
