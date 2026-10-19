// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_skin.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

use super::{
    DVec3, ErrorCode, HashMap, HashSet, Mesh, Modifier, Origin, Polygon, PotError, Result, Value,
    bool_param, build_mesh, id_indices, invalid_parameter, number_param, positions, validated,
};

use std::ptr::NonNull;

use crate::geom::heap_simple::HeapSimple;

#[repr(C)]
struct PotterSkinHull {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn potter_skin_hull_compute(
        coordinates: *const f32,
        vertex_count: usize,
        result: *mut *mut PotterSkinHull,
    ) -> i32;
    fn potter_skin_hull_triangle_index_count(hull: *const PotterSkinHull) -> usize;
    fn potter_skin_hull_copy_triangle_indices(
        hull: *const PotterSkinHull,
        destination: *mut u32,
        capacity: usize,
    ) -> i32;
    fn potter_skin_hull_free(hull: *mut PotterSkinHull);
}

struct SkinHullHandle(NonNull<PotterSkinHull>);

impl Drop for SkinHullHandle {
    fn drop(&mut self) {
        unsafe { potter_skin_hull_free(self.0.as_ptr()) };
    }
}

fn bullet_hull_faces(points: &[DVec3], input: &[usize]) -> Result<Vec<Vec<usize>>> {
    let mut coordinates = Vec::with_capacity(input.len() * 3);
    for &index in input {
        let point = points[index];
        let coordinate = [point.x as f32, point.y as f32, point.z as f32];
        if !coordinate.into_iter().all(f32::is_finite) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "Skin hull input cannot be represented as Blender float coordinates",
            ));
        }
        coordinates.extend(coordinate);
    }

    let mut raw_hull = std::ptr::null_mut();
    let status =
        unsafe { potter_skin_hull_compute(coordinates.as_ptr(), input.len(), &raw mut raw_hull) };
    if status != 0 {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("Skin Bullet convex hull failed with native status {status}"),
        ));
    }
    let hull = SkinHullHandle(NonNull::new(raw_hull).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "Skin Bullet convex hull returned no result",
        )
    })?);
    let index_count = unsafe { potter_skin_hull_triangle_index_count(hull.0.as_ptr()) };
    if index_count % 3 != 0 {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "Skin Bullet convex hull returned an invalid triangle index count",
        ));
    }
    let mut triangle_indices = vec![0_u32; index_count];
    let status = unsafe {
        potter_skin_hull_copy_triangle_indices(
            hull.0.as_ptr(),
            triangle_indices.as_mut_ptr(),
            triangle_indices.len(),
        )
    };
    if status != 0 {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            format!("Skin Bullet convex hull copy failed with native status {status}"),
        ));
    }

    triangle_indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| {
            triangle
                .iter()
                .map(|index| {
                    input.get(*index as usize).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "Skin Bullet convex hull returned an invalid original vertex index",
                        )
                    })
                })
                .collect()
        })
        .collect()
}

pub(super) fn skin_with_warnings(
    mesh: &Mesh,
    modifier: &Modifier,
    warnings: &mut Vec<PotError>,
) -> Result<Mesh> {
    match skin(mesh, modifier) {
        Err(error)
            if error
                .details
                .get("modifier_warning")
                .and_then(Value::as_bool)
                == Some(true) =>
        {
            warnings.push(error);
            Ok(mesh.clone())
        }
        result => result,
    }
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
pub(super) fn skin(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let source_positions = positions(mesh);
    let source_indices = id_indices(mesh);
    let radii = mesh.attributes.get("skin_radii").and_then(Value::as_object);
    let roots = mesh.attributes.get("skin_roots").and_then(Value::as_object);
    let skin_vertices = mesh
        .attributes
        .get("blender_skin_vertices")
        .and_then(Value::as_array);
    let node_loose: Vec<bool> = (0..mesh.vertices.len())
        .map(|index| {
            skin_vertices
                .and_then(|vertices| vertices.get(index))
                .and_then(|vertex| vertex.get("loose"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect();
    let branch_smoothing = number_param(modifier, "branch_smoothing")?;
    if !(0.0..=1.0).contains(&branch_smoothing) {
        return Err(invalid_parameter(
            modifier,
            "branch_smoothing",
            "a number from 0 to 1",
        ));
    }
    let smooth_shade = bool_param(modifier, "use_smooth_shade")?;
    let symmetry_axes = [
        bool_param(modifier, "use_x_symmetry")?,
        bool_param(modifier, "use_y_symmetry")?,
        bool_param(modifier, "use_z_symmetry")?,
    ];
    let mut node_positions = source_positions.clone();
    let mut node_origins: Vec<Vec<(usize, f64)>> = (0..mesh.vertices.len())
        .map(|index| vec![(index, 1.0)])
        .collect();
    let mut node_radii = Vec::with_capacity(mesh.vertices.len() + mesh.edges.len() * 4);
    let mut node_roots = Vec::with_capacity(mesh.vertices.len());
    // Without skin attributes Blender adds its default layer, which roots the first vertex
    // (BKE_mesh_ensure_skin_customdata).
    let default_layer = radii.is_none() && roots.is_none();
    for vertex_index in 0..mesh.vertices.len() {
        node_radii.push(skin_radius(radii, vertex_index, modifier)?);
        node_roots.push(if default_layer {
            vertex_index == 0
        } else {
            roots
                .and_then(|roots| roots.get(&vertex_index.to_string()))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        });
    }
    let mut degrees = vec![0_usize; mesh.vertices.len()];
    let mut edge_endpoints = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let first = *source_indices.get(&edge.vertices[0]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        let second = *source_indices.get(&edge.vertices[1]).ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "edges referencing existing vertices")
        })?;
        let length = source_positions[first].distance(source_positions[second]);
        if length <= f64::EPSILON {
            return Err(invalid_parameter(
                modifier,
                "mesh",
                "skeleton edges with non-zero length",
            ));
        }
        degrees[first] += 1;
        degrees[second] += 1;
        edge_endpoints.push([first, second]);
    }
    let mut edge_chains = Vec::with_capacity(mesh.edges.len());
    for [first, second] in edge_endpoints.iter().copied() {
        let first_radius = node_radii[first];
        let second_radius = node_radii[second];
        // Blender's `calc_edge_subdivisions`: `half_v2` of each end, summed, over the f32 length.
        let average_radius =
            skin_half_radius_sum(first_radius) + skin_half_radius_sum(second_radius);
        let edge_length = skin_float_distance(source_positions[first], source_positions[second]);
        let first_is_branch = degrees[first] > 2;
        let second_is_branch = degrees[second] > 2;
        let loose_branch =
            (first_is_branch && node_loose[first]) || (second_is_branch && node_loose[second]);
        let mut subdivisions = if loose_branch {
            if first_is_branch && second_is_branch {
                2
            } else {
                0
            }
        } else if average_radius > 0.0 {
            (edge_length / average_radius).min(128.0) as usize
        } else if average_radius < 0.0 && edge_length / average_radius <= -1.0 {
            // Blender's subdivision count truncates to -1 or below here, which its output vertex
            // count assertion forbids.
            return Err(invalid_parameter(
                modifier,
                "skin_radii",
                "edge radii whose average keeps the subdivision count non-negative",
            ));
        } else {
            // A zero average and a ratio in (-1, 0) both give no subdivisions.
            0
        };
        if first_is_branch && second_is_branch && subdivisions < 2 {
            subdivisions = 2;
        }
        // Blender: radrat = half(r1) / half(r0), then (radrat + 1) / 2 when finite, else 1.
        let ratio = skin_half_radius_sum(second_radius) / skin_half_radius_sum(first_radius);
        let radius_ratio = if ratio.is_finite() {
            (ratio + 1.0) / 2.0
        } else {
            1.0
        };
        let first_position = skin_float_point(source_positions[first]);
        let second_position = skin_float_point(source_positions[second]);
        let first_radius32 = first_radius.map(|component| component as f32);
        let second_radius32 = second_radius.map(|component| component as f32);
        let mut chain = Vec::with_capacity(subdivisions + 2);
        chain.push(first);
        for subdivision in 0..subdivisions {
            // Blender's `r = (j + 1) / float(edge_subd + 1)` and `t = powf(r, radrat)`, in f32.
            let r = (subdivision + 1) as f32 / (subdivisions + 1) as f32;
            let t = r.powf(radius_ratio);
            let index = node_positions.len();
            let position = skin_float_interp(first_position, second_position, t);
            node_positions.push(skin_float_widen(position));
            let weight = f64::from(t);
            node_origins.push(vec![(first, 1.0 - weight), (second, weight)]);
            let radius = skin_float_interp(first_radius32, second_radius32, t);
            node_radii.push(radius.map(f64::from));
            chain.push(index);
        }
        chain.push(second);
        edge_chains.push(chain);
    }
    let mut adjacency = vec![Vec::new(); source_positions.len()];
    for (edge_index, [first, second]) in edge_endpoints.iter().copied().enumerate() {
        adjacency[first].push((edge_index, second));
        adjacency[second].push((edge_index, first));
    }
    let edge_mats = edge_matrices(
        modifier,
        &source_positions,
        &edge_endpoints,
        &adjacency,
        &node_roots,
        &degrees,
    )?;
    // Blender's frames (MOD_skin.cc end_node_frames, connection_node_frames, subdivide_base): a leaf
    // takes its edge's frame; a chain node the bisector of its incoming and outgoing edge frames; a
    // seam node (both edges leave it, or neither does) one frame per edge, offset along that edge
    // by half its radius sum; a subdivision node its edge's frame.
    let mut frames: Vec<[DVec3; 4]> = Vec::new();
    let mut frame_nodes: Vec<usize> = Vec::new();
    let mut node_frame = vec![0_usize; node_positions.len()];
    let mut seam_frames: Vec<Option<[usize; 2]>> = vec![None; source_positions.len()];
    let mut box_frames: Vec<Option<[usize; 2]>> = vec![None; source_positions.len()];
    for (vertex, incident) in adjacency.iter().enumerate() {
        let center = node_positions[vertex];
        let radius = node_radii[vertex];
        match incident.as_slice() {
            [] => {
                // Blender's solitary-node box uses frames offset along +Z and -Z.
                let offset = skin_half_radius_sum(radius);
                let first = frames.len();
                frames.push(frame_corners(
                    center,
                    radius,
                    DVec3::X,
                    DVec3::Y,
                    skin_float_scale(DVec3::Z, offset),
                ));
                frame_nodes.push(vertex);
                let second = frames.len();
                frames.push(frame_corners(
                    center,
                    radius,
                    DVec3::X,
                    DVec3::Y,
                    skin_float_scale(DVec3::Z, -offset),
                ));
                frame_nodes.push(vertex);
                box_frames[vertex] = Some([first, second]);
            }
            [(edge_index, _)] => {
                let [_, y_axis, z_axis] = edge_mats[*edge_index].1;
                node_frame[vertex] = frames.len();
                frames.push(frame_corners(center, radius, y_axis, z_axis, [0.0; 3]));
                frame_nodes.push(vertex);
            }
            [(first_edge, _), (second_edge, _)] => {
                if (edge_mats[*first_edge].0 == vertex) == (edge_mats[*second_edge].0 == vertex) {
                    let offset = skin_half_radius_sum(radius);
                    let mut ids = [0_usize; 2];
                    for (slot, edge_index) in [*first_edge, *second_edge].into_iter().enumerate() {
                        let (origin, [x_axis, y_axis, z_axis]) = edge_mats[edge_index];
                        let direction = if origin == vertex { x_axis } else { -x_axis };
                        ids[slot] = frames.len();
                        frames.push(frame_corners(
                            center,
                            radius,
                            y_axis,
                            z_axis,
                            skin_float_scale(direction, offset),
                        ));
                        frame_nodes.push(vertex);
                    }
                    seam_frames[vertex] = Some(ids);
                } else {
                    let (incoming, outgoing) = if edge_mats[*first_edge].0 == vertex {
                        (*second_edge, *first_edge)
                    } else {
                        (*first_edge, *second_edge)
                    };
                    let [y_axis, z_axis] =
                        bisector_yz(edge_mats[incoming].1, edge_mats[outgoing].1[0]);
                    node_frame[vertex] = frames.len();
                    frames.push(frame_corners(center, radius, y_axis, z_axis, [0.0; 3]));
                    frame_nodes.push(vertex);
                }
            }
            _ if incident.len() > 2 => {}
            _ => {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "skin node has an unexpected degree",
                ));
            }
        }
    }
    for (edge_index, chain) in edge_chains.iter().enumerate() {
        let [_, y_axis, z_axis] = edge_mats[edge_index].1;
        for &index in chain.iter().skip(1).take(chain.len().saturating_sub(2)) {
            node_frame[index] = frames.len();
            frames.push(frame_corners(
                node_positions[index],
                node_radii[index],
                y_axis,
                z_axis,
                [0.0; 3],
            ));
            frame_nodes.push(index);
        }
    }
    let is_seam = |node: usize| seam_frames.get(node).is_some_and(Option::is_some);
    // The frame a chain uses at a node: a seam node has one per incident edge.
    let frame_at = |node: usize, edge_index: usize| -> usize {
        match seam_frames.get(node).copied().flatten() {
            Some([first_frame, second_frame]) => {
                if adjacency[node][0].0 == edge_index {
                    first_frame
                } else {
                    second_frame
                }
            }
            None => node_frame[node],
        }
    };
    let mut branch_rings = Vec::new();
    for (vertex, incident) in adjacency.iter().enumerate() {
        if degrees[vertex] <= 2 {
            continue;
        }
        let mut rings = Vec::with_capacity(incident.len());
        for &(edge_index, _) in incident {
            let chain = &edge_chains[edge_index];
            let neighbor = if edge_endpoints[edge_index][0] == vertex {
                chain.get(1)
            } else {
                chain.get(chain.len().saturating_sub(2))
            }
            .copied()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "skin branch has no neighboring frame",
                )
            })?;
            // Blender's collect_hull_frames: a seam neighbour contributes its first frame.
            rings.push(
                seam_frames
                    .get(neighbor)
                    .copied()
                    .flatten()
                    .map_or(node_frame[neighbor], |frames| frames[0]),
            );
        }
        branch_rings.push(rings);
    }
    let merged_corners = merge_branch_frame_corners(&mut frames, &branch_rings);
    let mut points = Vec::with_capacity(frames.len() * 4);
    let mut origins = Vec::with_capacity(frames.len() * 4);
    let mut frame_vertex_ids = vec![[0_usize; 4]; frames.len()];
    for (frame_index, (frame, node)) in frames.iter().zip(&frame_nodes).enumerate() {
        for corner in 0..4 {
            if merged_corners[frame_index][corner].is_none() {
                frame_vertex_ids[frame_index][corner] = points.len();
                points.push(frame[corner]);
                origins.push(node_origins[*node].clone());
            }
        }
    }
    for frame_index in 0..frames.len() {
        for corner in 0..4 {
            let Some((target_frame, target_corner)) = merged_corners[frame_index][corner] else {
                continue;
            };
            let target = frame_vertex_ids
                .get(target_frame)
                .and_then(|ids| ids.get(target_corner))
                .copied()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "skin frame merge target is missing",
                    )
                })?;
            frame_vertex_ids[frame_index][corner] = target;
        }
    }
    let mut faces = Vec::<Polygon>::new();
    let mut branch_faces = Vec::new();
    let mut branch_frame_states = vec![(false, [false; 4]); frames.len()];
    for ring_frames in &branch_rings {
        let rings: Vec<[usize; 4]> = ring_frames
            .iter()
            .map(|frame| frame_vertex_ids[*frame])
            .collect();
        branch_hull_faces(
            &points,
            &rings,
            ring_frames,
            &mut branch_faces,
            &mut branch_frame_states,
        )?;
    }
    let mut detached_frames: Vec<(usize, [bool; 4])> = branch_frame_states
        .into_iter()
        .enumerate()
        .filter_map(|(frame, (detached, inside))| detached.then_some((frame, inside)))
        .collect();
    // skin_fix_hull_topology walks skin nodes in index order and each node's frames in order
    // (MOD_skin.cc:1676-1690), so repairs run in (node, frame) order rather than frame order.
    detached_frames.sort_by_key(|&(frame, _)| (frame_nodes[frame], frame));
    branch_faces = merge_triangle_faces(&points, branch_faces, symmetry_axes);
    let mut welded = Vec::new();
    let mut tagged_edges = TaggedEdges::default();
    repair_detached_branch_frames(
        &mut points,
        &mut origins,
        &mut frame_vertex_ids,
        &mut branch_faces,
        &detached_frames,
        &mut welded,
        &mut tagged_edges,
    )?;
    smooth_branch_hull_points(
        &mut points,
        &branch_faces,
        &frame_vertex_ids,
        branch_smoothing,
    );
    faces.extend(branch_faces.into_iter().map(|face| (face, 0)));
    for ids in seam_frames.iter().flatten() {
        let [first, second] = *ids;
        let second_ids = bridge_order(&points, frame_vertex_ids[first], frame_vertex_ids[second]);
        connect_frames(&points, frame_vertex_ids[first], second_ids, &mut faces);
    }
    for ids in box_frames.iter().flatten() {
        let [first, second] = *ids;
        let first_ids = frame_vertex_ids[first];
        let second_ids = frame_vertex_ids[second];
        connect_frames(&points, first_ids, second_ids, &mut faces);
        faces.push((
            vec![first_ids[3], first_ids[2], first_ids[1], first_ids[0]],
            0,
        ));
        faces.push((second_ids.to_vec(), 0));
    }
    for (edge_index, chain) in edge_chains.iter().enumerate() {
        for pair in chain.windows(2) {
            let (first, second) = (pair[0], pair[1]);
            if degrees.get(first).is_some_and(|degree| *degree > 2)
                || degrees.get(second).is_some_and(|degree| *degree > 2)
            {
                continue;
            }
            let first_ids = frame_vertex_ids[frame_at(first, edge_index)];
            let second_ids = frame_vertex_ids[frame_at(second, edge_index)];
            // Blender bridges every connection that touches a seam (skin_output_connections).
            let second_ids = if is_seam(first) || is_seam(second) {
                bridge_order(&points, first_ids, second_ids)
            } else {
                second_ids
            };
            connect_frames(&points, first_ids, second_ids, &mut faces);
        }
    }
    let mut capped = HashSet::new();
    for [first, second] in edge_endpoints.iter().copied() {
        for vertex in [first, second] {
            if degrees[vertex] != 1 || !capped.insert(vertex) {
                continue;
            }
            let base = frame_vertex_ids[node_frame[vertex]];
            // Blender's cap frame has its normal opposite the edge direction, which points into the
            // tube (MOD_skin.cc, FLIP_NORMAL), so the cap faces away from the tube.
            let other = if vertex == first { second } else { first };
            let into_tube = source_positions[other] - source_positions[vertex];
            let cap_normal =
                (points[base[1]] - points[base[0]]).cross(points[base[2]] - points[base[0]]);
            let flipped = into_tube.dot(cap_normal) < 0.0;
            let cap = if flipped {
                vec![base[0], base[1], base[2], base[3]]
            } else {
                vec![base[3], base[2], base[1], base[0]]
            };
            faces.push((cap, 0));
        }
    }
    let polygons: Vec<Vec<usize>> = faces.into_iter().map(|(face, _)| face).collect();
    // Blender's final hull_merge_triangles deletes every edge still tagged BM_ELEM_TAG, which
    // removes all faces around it (MOD_skin.cc:1632). The tags come from the triangle split in
    // skin_fix_hole_no_good_verts.
    let (kept, killed): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
        merge_triangle_faces(&points, polygons, symmetry_axes)
            .into_iter()
            .partition(|face| !tagged_edges.0.iter().any(|&edge| face_has_edge(face, edge)));
    let loose_edges = skin_loose_edges(&kept, &killed, &tagged_edges.0);
    let faces: Vec<Polygon> = kept.into_iter().map(|face| (face, 0)).collect();
    let (points, origins, faces, loose_edges) =
        drop_welded_points(points, origins, faces, &welded, loose_edges);
    let origin = Origin {
        vertices: origins,
        source_vertices: None,
        faces: vec![None; faces.len()],
        corners: None,
    };
    let mut result = build_mesh(points, faces, loose_edges, &origin, mesh)?;
    // Skin's fresh BMesh has no point CD_ORIGINDEX layer, so the index carrier is dropped.
    result.attributes.remove(crate::geom::ORIGINDEX_ATTRIBUTE);
    result.attributes.remove("blender_skin_vertices");
    result.attributes.remove("skin_radii");
    result.attributes.remove("skin_roots");
    // Shading lives in `shade_smooth`; a remapped `blender_polygon_smooth` would conflict.
    result.attributes.remove("blender_polygon_smooth");
    if smooth_shade {
        crate::geom::normals::shade_faces_smooth(&mut result);
    }
    // Blender's Skin output has no UV layers, so UV data is dropped (reported as attribute loss).
    let uv_names: Vec<String> = crate::geom::uv_layers::read_layers(&result)
        .into_iter()
        .map(|layer| layer.name)
        .collect();
    result.attributes.remove(crate::geom::uv_layers::ATTRIBUTE);
    if let Some(entries) = result
        .attributes
        .get_mut("blender_attributes")
        .and_then(serde_json::Value::as_array_mut)
    {
        entries.retain(|entry| {
            !entry
                .get("name")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| uv_names.iter().any(|uv| uv == name))
        });
    }
    drop_partial_vertex_groups(mesh, &origin, &mut result);
    validated(&result)?;
    Ok(result)
}

/// Blender's `skin_merge_close_frame_verts`: merge near corners of frames surrounding each branch.
fn merge_branch_frame_corners(
    frames: &mut [[DVec3; 4]],
    branch_rings: &[Vec<usize>],
) -> Vec<[Option<(usize, usize)>; 4]> {
    let mut merges = vec![[None; 4]; frames.len()];
    let mut targets = vec![[false; 4]; frames.len()];
    for rings in branch_rings {
        for first in 0..rings.len() {
            let first_frame = rings[first];
            let first_side = frame_side_length(frames[first_frame]);
            for first_corner in 0..4 {
                if merges[first_frame][first_corner].is_some() {
                    continue;
                }
                for &second_frame in &rings[first + 1..] {
                    let threshold = first_side.min(frame_side_length(frames[second_frame])) * 0.5;
                    for second_corner in 0..4 {
                        if merges[second_frame][second_corner].is_some()
                            || targets[second_frame][second_corner]
                        {
                            continue;
                        }
                        let distance = skin_float_distance(
                            frames[first_frame][first_corner],
                            frames[second_frame][second_corner],
                        );
                        if distance < threshold {
                            let midpoint = skin_float_midpoint(
                                frames[first_frame][first_corner],
                                frames[second_frame][second_corner],
                            );
                            frames[first_frame][first_corner] = midpoint;
                            frames[second_frame][second_corner] = midpoint;
                            merges[second_frame][second_corner] = Some((first_frame, first_corner));
                            targets[first_frame][first_corner] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    merges
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn frame_side_length(frame: [DVec3; 4]) -> f32 {
    (skin_float_distance(frame[0], frame[1]) + skin_float_distance(frame[1], frame[2])) * 0.5
}

fn branch_hull_faces(
    points: &[DVec3],
    rings: &[[usize; 4]],
    ring_frames: &[usize],
    faces: &mut Vec<Vec<usize>>,
    frame_states: &mut [(bool, [bool; 4])],
) -> Result<()> {
    if rings.len() != ring_frames.len() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "Skin branch ring frames do not match their vertex indices",
        ));
    }
    let mut input: Vec<usize> = rings.iter().flatten().copied().collect();
    input.sort_unstable();
    input.dedup();
    let hull_start = faces.len();
    faces.extend(bullet_hull_faces(points, &input)?);
    let mut hull_vertices = HashSet::new();
    let mut hull_edges = HashSet::new();
    for face in faces.iter() {
        for corner in 0..face.len() {
            let first = face[corner];
            let second = face[(corner + 1) % face.len()];
            hull_edges.insert(ordered_edge(first, second));
        }
    }
    for face in &faces[hull_start..] {
        hull_vertices.extend(face.iter().copied());
    }

    let mut removed_faces = vec![false; faces.len()];
    for (ring, frame_index) in rings.iter().zip(ring_frames) {
        let Some((detached, inside)) = frame_states.get_mut(*frame_index) else {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "Skin branch frame index is out of range",
            ));
        };
        if *detached {
            continue;
        }

        let first_inside = (0..4)
            .filter(|corner| !hull_vertices.contains(&ring[*corner]))
            .min_by_key(|corner| ring[*corner]);
        if let Some(corner) = first_inside {
            inside[corner] = true;
            *detached = true;
        }
        if !*detached
            && (0..4).any(|corner| {
                !hull_edges.contains(&ordered_edge(ring[corner], ring[(corner + 1) % 4]))
            })
        {
            *detached = true;
        }

        if !*detached {
            let diagonal = if hull_edges.contains(&ordered_edge(ring[0], ring[2])) {
                Some(ordered_edge(ring[0], ring[2]))
            } else if hull_edges.contains(&ordered_edge(ring[1], ring[3])) {
                Some(ordered_edge(ring[1], ring[3]))
            } else {
                None
            };
            let contained_faces = diagonal.map(|diagonal| {
                let mut pair = [0; 2];
                let mut count = 0;
                for (index, face) in faces.iter().enumerate() {
                    if face_has_edge(face, diagonal) {
                        if count < pair.len() {
                            pair[count] = index;
                        }
                        count += 1;
                    }
                }
                (pair, count)
            });
            if let Some((pair, 2)) = contained_faces {
                removed_faces[pair[0]] = true;
                removed_faces[pair[1]] = true;
            } else {
                *detached = true;
            }
        }
    }

    let mut face_index = 0;
    faces.retain(|_| {
        let keep = !removed_faces[face_index];
        face_index += 1;
        keep
    });
    Ok(())
}

fn face_has_edge(face: &[usize], edge: (usize, usize)) -> bool {
    (0..face.len())
        .any(|corner| ordered_edge(face[corner], face[(corner + 1) % face.len()]) == edge)
}

fn ordered_edge(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn repair_detached_branch_frames(
    points: &mut Vec<DVec3>,
    origins: &mut Vec<Vec<(usize, f64)>>,
    frame_vertex_ids: &mut [[usize; 4]],
    faces: &mut Vec<Vec<usize>>,
    detached_frames: &[(usize, [bool; 4])],
    welded: &mut Vec<usize>,
    tagged: &mut TaggedEdges,
) -> Result<()> {
    for &(frame_index, inside) in detached_frames {
        let frame = frame_vertex_ids.get(frame_index).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "Skin detached frame index is out of range",
            )
        })?;
        let target = skin_hole_target_face(points, faces, frame);
        let has_degenerate_coincidence = target.is_some_and(|face| {
            !inside.iter().any(|is_inside| *is_inside)
                && faces[face].iter().any(|vertex| frame.contains(vertex))
        });

        for corner in 0..4 {
            if inside[corner] {
                continue;
            }
            let source = frame_vertex_ids[frame_index][corner];
            let duplicate = points.len();
            points.push(points[source]);
            origins.push(origins[source].clone());
            frame_vertex_ids[frame_index][corner] = duplicate;
        }

        if let Some(target) = target
            && !has_degenerate_coincidence
            && skin_fix_hole_no_good_verts(
                points,
                origins,
                faces,
                target,
                frame_vertex_ids[frame_index],
                welded,
                tagged,
            )?
        {
            continue;
        }
        faces.push(frame_vertex_ids[frame_index].to_vec());
    }
    Ok(())
}

fn subdivide_skin_target_face(
    points: &mut Vec<DVec3>,
    origins: &mut Vec<Vec<(usize, f64)>>,
    faces: &mut [Vec<usize>],
    target: usize,
) -> Result<(usize, usize, usize)> {
    let face = faces.get(target).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Skin target face index is out of range",
        )
    })?;
    let mut longest_edge = face
        .first()
        .zip(face.get(1))
        .map(|(first, second)| (*first, *second));
    let mut longest_length = 0.0;
    for corner in 0..face.len() {
        let first = face[corner];
        let second = face[(corner + 1) % face.len()];
        let length = skin_float_distance_squared(points[first], points[second]);
        if length >= longest_length {
            longest_length = length;
            longest_edge = Some((first, second));
        }
    }
    let (first, second) = longest_edge.ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Skin triangular target face has no edge",
        )
    })?;
    let a = points[first];
    let b = points[second];
    let midpoint = skin_float_midpoint(a, b);
    let mut midpoint_origin = std::collections::BTreeMap::<usize, f64>::new();
    for &(source, weight) in &origins[first] {
        *midpoint_origin.entry(source).or_default() += weight * 0.5;
    }
    for &(source, weight) in &origins[second] {
        *midpoint_origin.entry(source).or_default() += weight * 0.5;
    }
    let midpoint_index = points.len();
    points.push(midpoint);
    origins.push(midpoint_origin.into_iter().collect());

    let split_edge = ordered_edge(first, second);
    let mut found_edge = false;
    for face in faces {
        let mut subdivided = Vec::with_capacity(face.len() + 1);
        for corner in 0..face.len() {
            let current = face[corner];
            let next = face[(corner + 1) % face.len()];
            subdivided.push(current);
            if ordered_edge(current, next) == split_edge {
                subdivided.push(midpoint_index);
                found_edge = true;
            }
        }
        if subdivided.len() != face.len() {
            *face = subdivided;
        }
    }
    if !found_edge {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "Skin target face subdivision edge disappeared",
        ));
    }
    Ok((first, midpoint_index, second))
}

/// Blender's `skin_fix_hole_no_good_verts`: `target` is extruded, which deletes it and leaves a copy
/// with new vertices plus one side wall per edge. A triangular copy is split on its longest edge, and
/// its four corners are welded onto `frame` in the nearest order, so the side walls join the frame.
/// Returns `false` when the copy is not a quad; the copy then stays and the caller caps the frame.
fn skin_fix_hole_no_good_verts(
    points: &mut Vec<DVec3>,
    origins: &mut Vec<Vec<(usize, f64)>>,
    faces: &mut Vec<Vec<usize>>,
    target: usize,
    frame: [usize; 4],
    welded: &mut Vec<usize>,
    tagged: &mut TaggedEdges,
) -> Result<bool> {
    let ring = faces.get(target).cloned().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Skin target face index is out of range",
        )
    })?;
    let mut copy = Vec::with_capacity(ring.len());
    for &vertex in &ring {
        let position = points[vertex];
        let source = origins[vertex].clone();
        copy.push(points.len());
        points.push(position);
        origins.push(source);
    }
    for (corner, &vertex) in ring.iter().enumerate() {
        let next = (corner + 1) % ring.len();
        faces.push(vec![ring[next], copy[next], copy[corner], vertex]);
    }
    faces[target] = copy;
    if faces[target].len() == 3 {
        // Blender's triangle branch clears every edge tag, then tags the longest edge
        // (MOD_skin.cc:1389-1390). BM_edge_split gives both halves the tag (bmesh_construct.cc).
        tagged.0.clear();
        let (first, midpoint, second) = subdivide_skin_target_face(points, origins, faces, target)?;
        tagged.0.push(ordered_edge(first, midpoint));
        tagged.0.push(ordered_edge(midpoint, second));
    }
    if faces[target].len() > 4 {
        collapse_skin_face_corners(points, faces, target, welded, tagged);
    }
    if faces[target].len() != 4 {
        return Ok(false);
    }
    let quad = faces[target].clone();
    let order = skin_choose_quad_bridge_order(points, &quad, frame);
    let mapping: Vec<(usize, usize)> = quad
        .iter()
        .copied()
        .zip(order.map(|corner| frame[corner]))
        .collect();
    welded.extend(quad);
    faces.remove(target);
    tagged.weld(faces, &mapping);
    weld_skin_vertices(faces, &mapping);
    Ok(true)
}

/// Blender's `collapse_face_corners`: while `target` has more than four corners, its shortest edge
/// merges into the edge's first corner, which moves to the edge midpoint. That merge is a
/// `weld_verts`, so tagged edges follow it (see `TaggedEdges::weld`).
fn collapse_skin_face_corners(
    points: &mut [DVec3],
    faces: &mut [Vec<usize>],
    target: usize,
    welded: &mut Vec<usize>,
    tagged: &mut TaggedEdges,
) {
    while faces[target].len() > 4 {
        let face = faces[target].clone();
        let mut shortest = 0;
        let mut shortest_length = f32::MAX;
        for (corner, &first) in face.iter().enumerate() {
            let second = face[(corner + 1) % face.len()];
            let length = skin_float_distance_squared(points[first], points[second]);
            if length <= shortest_length {
                shortest_length = length;
                shortest = corner;
            }
        }
        let keep = face[shortest];
        let merge = face[(shortest + 1) % face.len()];
        points[keep] = skin_float_midpoint(points[keep], points[merge]);
        welded.push(merge);
        tagged.weld(faces, &[(merge, keep)]);
        for other in &mut *faces {
            for vertex in &mut *other {
                if *vertex == merge {
                    *vertex = keep;
                }
            }
            dedupe_cyclic(other);
        }
    }
}

/// Blender's `BM_ELEM_TAG` on the hull edges repaired by `skin_fix_hole_no_good_verts`, as ordered
/// point pairs. Only the triangle split tags edges, and nothing else in `MOD_skin.cc` clears them
/// before the final `hull_merge_triangles`, whose `BM_mesh_delete_hflag_tagged` deletes tagged
/// edges with every face around them (`MOD_skin.cc:1632`). Its face-tag reset is separate
/// (`MOD_skin.cc:1566`).
#[derive(Default)]
struct TaggedEdges(Vec<(usize, usize)>);

impl TaggedEdges {
    /// `bmo_weld_verts_exec` (`bmo_removedoubles.cc`) for a target map. A tagged edge with a welded
    /// corner moves onto the targets. A collapsed edge (both ends on one vertex) is deleted. When
    /// the target pair already exists, that edge survives and keeps only its own tag, because
    /// `BM_elem_flag_merge_ex` merges only `BM_ELEM_HIDDEN`, so the moved tag is lost. Otherwise
    /// `BM_edge_create(.., e, BM_CREATE_NOP)` copies the header flags, so the new edge is tagged.
    fn weld(&mut self, faces: &[Vec<usize>], mapping: &[(usize, usize)]) {
        let is_welded = |vertex: usize| mapping.iter().any(|&(from, _)| from == vertex);
        let target = |vertex: usize| {
            mapping
                .iter()
                .find(|&&(from, _)| from == vertex)
                .map_or(vertex, |&(_, to)| to)
        };
        let mut moved = Vec::new();
        self.0.retain(|&(first, second)| {
            if !is_welded(first) && !is_welded(second) {
                return true;
            }
            moved.push(ordered_edge(target(first), target(second)));
            false
        });
        for (first, second) in moved {
            if first == second
                || self.0.contains(&(first, second))
                || faces
                    .iter()
                    .any(|face| face_has_edge(face, (first, second)))
            {
                continue;
            }
            self.0.push((first, second));
        }
    }
}

/// Blender's `weld_verts` for a target map: each mapped corner takes its target, corners that become
/// adjacent collapse, and faces left with fewer than three corners are dropped.
fn weld_skin_vertices(faces: &mut Vec<Vec<usize>>, mapping: &[(usize, usize)]) {
    for face in &mut *faces {
        for vertex in &mut *face {
            if let Some(&(_, to)) = mapping.iter().find(|&&(from, _)| from == *vertex) {
                *vertex = to;
            }
        }
        dedupe_cyclic(face);
    }
    faces.retain(|face| face.len() >= 3);
}

/// Drops repeated corners of a polygon, including a repeat across its closing edge.
fn dedupe_cyclic(face: &mut Vec<usize>) {
    face.dedup();
    if face.len() > 1 && face.first() == face.last() {
        face.pop();
    }
}

/// The points, their origins, faces and loose edges after welded points are dropped.
type WeldedPoints = (
    Vec<DVec3>,
    Vec<Vec<(usize, f64)>>,
    Vec<Polygon>,
    Vec<[usize; 2]>,
);

/// Blender's weld removes the vertices it merges away. Drops the `welded` points, renumbering the
/// rest and every face and loose edge that uses them. Faces and loose edges are built from current
/// corners, which the weld has already moved, so they never reference a welded point.
fn drop_welded_points(
    points: Vec<DVec3>,
    origins: Vec<Vec<(usize, f64)>>,
    faces: Vec<Polygon>,
    welded: &[usize],
    loose_edges: Vec<[usize; 2]>,
) -> WeldedPoints {
    let mut dropped = vec![false; points.len()];
    for &index in welded {
        if let Some(flag) = dropped.get_mut(index) {
            *flag = true;
        }
    }
    let mut renumbered = vec![0_usize; points.len()];
    let mut kept_points = Vec::with_capacity(points.len());
    let mut kept_origins = Vec::with_capacity(origins.len());
    for (index, (point, origin)) in points.into_iter().zip(origins).enumerate() {
        if dropped[index] {
            continue;
        }
        renumbered[index] = kept_points.len();
        kept_points.push(point);
        kept_origins.push(origin);
    }
    let faces = faces
        .into_iter()
        .map(|(face, material)| {
            (
                face.into_iter().map(|vertex| renumbered[vertex]).collect(),
                material,
            )
        })
        .collect();
    let loose_edges = loose_edges
        .into_iter()
        .map(|[first, second]| [renumbered[first], renumbered[second]])
        .collect();
    (kept_points, kept_origins, faces, loose_edges)
}

/// Blender's final delete removes the faces around each tagged edge but leaves their other edges
/// in the mesh (`MOD_skin.cc:1632`). An edge of a removed face survives as a loose edge unless a
/// kept face still uses it. The nogood branch oracle has 69 edges for 68 face edges, which this
/// rule reproduces.
fn skin_loose_edges(
    kept: &[Vec<usize>],
    killed: &[Vec<usize>],
    tagged: &[(usize, usize)],
) -> Vec<[usize; 2]> {
    let mut loose: Vec<(usize, usize)> = Vec::new();
    for face in killed {
        for corner in 0..face.len() {
            let edge = ordered_edge(face[corner], face[(corner + 1) % face.len()]);
            if tagged.contains(&edge)
                || loose.contains(&edge)
                || kept.iter().any(|kept_face| face_has_edge(kept_face, edge))
            {
                continue;
            }
            loose.push(edge);
        }
    }
    loose
        .into_iter()
        .map(|(first, second)| [first, second])
        .collect()
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn skin_float_midpoint(first: DVec3, second: DVec3) -> DVec3 {
    DVec3::new(
        ((first.x as f32 + second.x as f32) * 0.5) as f64,
        ((first.y as f32 + second.y as f32) * 0.5) as f64,
        ((first.z as f32 + second.z as f32) * 0.5) as f64,
    )
}

fn skin_hole_target_face(
    points: &[DVec3],
    faces: &[Vec<usize>],
    frame: [usize; 4],
) -> Option<usize> {
    let frame_center = skin_frame_center(points, frame);
    let frame_normal = skin_float_normalize(skin_float_cross(
        skin_float_sub(
            skin_float_point(points[frame[3]]),
            skin_float_point(points[frame[1]]),
        ),
        skin_float_sub(
            skin_float_point(points[frame[2]]),
            skin_float_point(points[frame[0]]),
        ),
    ));
    let mut isect_target = None;
    let mut center_target = None;
    let mut best_isect_distance = f32::MAX;
    let mut best_center_distance = f32::MAX;

    for (face_index, face) in faces.iter().enumerate() {
        if let Some(&first) = face.first() {
            let mut previous = first;
            for &vertex in face.iter().skip(1) {
                if previous != first
                    && let Some(distance) = skin_ray_triangle(
                        frame_center,
                        frame_normal,
                        skin_float_point(points[first]),
                        skin_float_point(points[previous]),
                        skin_float_point(points[vertex]),
                    )
                    && distance < best_isect_distance
                {
                    isect_target = Some(face_index);
                    best_isect_distance = distance;
                }
                previous = vertex;
            }
        }

        let mut face_center = [0.0; 3];
        for &vertex in face {
            face_center = skin_float_add(face_center, skin_float_point(points[vertex]));
        }
        for component in &mut face_center {
            *component /= face.len() as f32;
        }
        let distance = skin_float_length(skin_float_sub(frame_center, face_center));
        if distance < best_center_distance {
            center_target = Some(face_index);
            best_center_distance = distance;
        }
    }

    if isect_target.is_none() || best_center_distance < best_isect_distance / 2.0 {
        center_target
    } else {
        isect_target
    }
}

fn skin_choose_quad_bridge_order(
    points: &[DVec3],
    target: &[usize],
    frame: [usize; 4],
) -> [usize; 4] {
    let mut best = [0; 4];
    let mut shortest = f32::MAX;
    for reversed in 0..2 {
        for rotation in 0..4 {
            let mut order = [0; 4];
            for (corner, slot) in order.iter_mut().enumerate() {
                *slot = if reversed == 0 {
                    (corner + rotation) % 4
                } else {
                    3 - ((corner + rotation) % 4)
                };
            }
            let mut length = 0.0;
            for corner in 0..4 {
                length += skin_float_distance_squared(
                    points[target[corner]],
                    points[frame[order[corner]]],
                );
            }
            if length < shortest {
                shortest = length;
                best = order;
            }
        }
    }
    best
}

fn skin_ray_triangle(
    origin: [f32; 3],
    direction: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<f32> {
    let edge1 = skin_float_sub(v1, v0);
    let edge2 = skin_float_sub(v2, v0);
    let p = skin_float_cross(direction, edge2);
    let determinant = skin_float_dot(edge1, p);
    if determinant > -1.0e-8 && determinant < 1.0e-8 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = skin_float_sub(origin, v0);
    let u = inverse * skin_float_dot(s, p);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = skin_float_cross(s, edge1);
    let v = inverse * skin_float_dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = inverse * skin_float_dot(edge2, q);
    (distance >= 0.0).then_some(distance)
}

fn skin_frame_center(points: &[DVec3], frame: [usize; 4]) -> [f32; 3] {
    let mut center = skin_float_add(
        skin_float_point(points[frame[0]]),
        skin_float_point(points[frame[1]]),
    );
    center = skin_float_add(center, skin_float_point(points[frame[2]]));
    center = skin_float_add(center, skin_float_point(points[frame[3]]));
    center.map(|component| component * 0.25)
}

fn skin_float_point(point: DVec3) -> [f32; 3] {
    [point.x as f32, point.y as f32, point.z as f32]
}

fn skin_float_add(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    ]
}

fn skin_float_sub(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn skin_float_cross(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn skin_float_dot(first: [f32; 3], second: [f32; 3]) -> f32 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn skin_float_distance_squared(first: DVec3, second: DVec3) -> f32 {
    let difference = skin_float_sub(skin_float_point(first), skin_float_point(second));
    skin_float_dot(difference, difference)
}

fn skin_float_distance(first: DVec3, second: DVec3) -> f32 {
    skin_float_distance_squared(first, second).sqrt()
}

/// Blender's `mul_v3_v3fl`: `axis * scalar` in `f32`.
fn skin_float_scale(axis: DVec3, scalar: f32) -> [f32; 3] {
    skin_float_point(axis).map(|component| component * scalar)
}

fn skin_float_widen(point: [f32; 3]) -> DVec3 {
    DVec3::new(
        f64::from(point[0]),
        f64::from(point[1]),
        f64::from(point[2]),
    )
}

/// Blender's `half_v2`: the seam and solitary-box offset, `(r0 + r1) * 0.5` in `f32`.
fn skin_half_radius_sum(radius: [f64; 2]) -> f32 {
    f32::midpoint(radius[0] as f32, radius[1] as f32)
}

fn skin_float_length(vector: [f32; 3]) -> f32 {
    skin_float_dot(vector, vector).sqrt()
}

fn skin_float_normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = skin_float_length(vector);
    if length == 0.0 {
        [0.0; 3]
    } else {
        [vector[0] / length, vector[1] / length, vector[2] / length]
    }
}

/// Blender's `normalize_v3`: `vector * (1 / length)` in `f32`, zero when the squared length is at
/// most `1e-35`. Edge frames use this; `skin_float_normalize` keeps its own hole-target behavior.
fn skin_float_normalize_v3(vector: [f32; 3]) -> [f32; 3] {
    let squared = skin_float_dot(vector, vector);
    if squared > 1.0e-35 {
        let scale = 1.0 / squared.sqrt();
        vector.map(|component| component * scale)
    } else {
        [0.0; 3]
    }
}

/// Blender's `normalize_v3(to - from)`: the unit direction from one vertex to another, in `f32`.
fn skin_float_edge_direction(from: DVec3, to: DVec3) -> [f32; 3] {
    skin_float_normalize_v3(skin_float_sub(skin_float_point(to), skin_float_point(from)))
}

/// Blender's `angle_normalized_v3v3`: the angle between two unit vectors, from their chord length.
fn skin_float_angle(first: [f32; 3], second: [f32; 3]) -> f32 {
    // Blender's `safe_asinf` of half the chord length.
    let half_asin = |chord: f32| (chord / 2.0).clamp(-1.0, 1.0).asin();
    if skin_float_dot(first, second) >= 0.0 {
        2.0 * half_asin(skin_float_length(skin_float_sub(second, first)))
    } else {
        let negated = second.map(|component| -component);
        std::f32::consts::PI - 2.0 * half_asin(skin_float_length(skin_float_sub(negated, first)))
    }
}

/// Blender's `rotate_normalized_v3_v3v3fl`: `point` rotated by `angle` about the unit `axis`.
fn skin_float_rotate(point: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let costheta = angle.cos();
    let sintheta = angle.sin();
    let [ax, ay, az] = axis;
    let [px, py, pz] = point;
    let one_minus = 1.0 - costheta;
    [
        ((costheta + one_minus * ax * ax) * px)
            + ((one_minus * ax * ay - az * sintheta) * py)
            + ((one_minus * ax * az + ay * sintheta) * pz),
        ((one_minus * ax * ay + az * sintheta) * px)
            + ((costheta + one_minus * ay * ay) * py)
            + ((one_minus * ay * az - ax * sintheta) * pz),
        ((one_minus * ax * az - ay * sintheta) * px)
            + ((one_minus * ay * az + ax * sintheta) * py)
            + ((costheta + one_minus * az * az) * pz),
    ]
}

/// Blender's `interp_v3_v3v3`: `(1 - t) * a + t * b` per component in `f32`. Positions use three
/// components; radii use the two of `MVertSkin`.
fn skin_float_interp<const N: usize>(first: [f32; N], second: [f32; N], t: f32) -> [f32; N] {
    let s = 1.0 - t;
    std::array::from_fn(|index| s * first[index] + t * second[index])
}

/// Blender's `BM_face_calc_area` for a triangle, in the same `f32` operation order: the Newell sum of
/// `add_newell_cross_v3_v3v3` over the loops, then `len_v3(n) * 0.5f`. The order matters because the score of
/// two candidate quads can differ by one f32 ulp, and the heap pops by that score.
fn triangle_area32(points: &[glam::Vec3], triangle: [usize; 3]) -> f32 {
    let mut normal = [0.0_f32; 3];
    for corner in 0..3 {
        let previous = points[triangle[corner]].to_array();
        let current = points[triangle[(corner + 1) % 3]].to_array();
        normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
        normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
        normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
    }
    dot32(normal, normal).sqrt() * 0.5
}

/// Blender's `BM_face_normal_update` for a triangle (`normal_tri_v3` followed by `normalize_v3`), in `f32`.
fn triangle_normal32(points: &[glam::Vec3], triangle: [usize; 3]) -> [f32; 3] {
    let v1 = points[triangle[0]].to_array();
    let v2 = points[triangle[1]].to_array();
    let v3 = points[triangle[2]].to_array();
    let n1 = [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
    let n2 = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]];
    let normal = [
        n1[1] * n2[2] - n1[2] * n2[1],
        n1[2] * n2[0] - n1[0] * n2[2],
        n1[0] * n2[1] - n1[1] * n2[0],
    ];
    // `normalize_v3_v3_length` with unit length 1: scale by `1 / len` only above Blender's 1e-35 threshold.
    let squared = dot32(normal, normal);
    if squared > 1.0e-35 {
        let scale = 1.0_f32 / squared.sqrt();
        normal.map(|component| component * scale)
    } else {
        [0.0; 3]
    }
}

/// Blender's `dot_v3v3`, left to right in `f32`.
fn dot32(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The three corners of a triangular face.
fn triangle_of(face: &[usize]) -> [usize; 3] {
    [face[0], face[1], face[2]]
}

/// Blender's `hull_merge_triangles` (MOD_skin.cc:1557-1632). Each edge used by two triangles offers a quad.
/// Candidates are scored in `f32` at build time and kept in a `BLI_heapsimple`; the heap pops the best first.
/// A pop takes its quad only when both triangles are still free and they share no neighbouring face
/// (`BM_face_share_face_check`). Edges are visited in first-appearance order over the faces, which stands in
/// for the `BMesh` edge order.
fn merge_triangle_faces(
    points: &[DVec3],
    faces: Vec<Vec<usize>>,
    symmetry_axes: [bool; 3],
) -> Vec<Vec<usize>> {
    let points32: Vec<glam::Vec3> = points.iter().map(|point| point.as_vec3()).collect();
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut edge_order: Vec<(usize, usize)> = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        for edge in face_edges(face) {
            let adjacent = edge_faces.entry(edge).or_default();
            if adjacent.is_empty() {
                edge_order.push(edge);
            }
            adjacent.push(face_index);
        }
    }
    let mut heap = HeapSimple::new();
    for (edge_position, &edge) in edge_order.iter().enumerate() {
        let adjacent = &edge_faces[&edge];
        if adjacent.len() != 2 {
            continue;
        }
        let (first, second) = (adjacent[0], adjacent[1]);
        if faces[first].len() != 3 || faces[second].len() != 3 {
            continue;
        }
        let (first_tri, second_tri) = (triangle_of(&faces[first]), triangle_of(&faces[second]));
        let Some(quad) = quad_from_triangles(first_tri, second_tri, [edge.0, edge.1]) else {
            continue;
        };
        let mut score = (triangle_area32(&points32, first_tri)
            + triangle_area32(&points32, second_tri))
            * dot32(
                triangle_normal32(&points32, first_tri),
                triangle_normal32(&points32, second_tri),
            );
        if quad_crosses_symmetry_plane(points, quad, symmetry_axes) {
            if quad_is_symmetric(points, quad, symmetry_axes) {
                score *= 10.0;
            } else {
                continue;
            }
        }
        if !quad_is_convex(points, quad) {
            continue;
        }
        heap.insert(-score, edge_position);
    }

    let mut all_faces = faces;
    let mut consumed = vec![false; all_faces.len()];
    while let Some(edge_position) = heap.pop_min() {
        let edge = edge_order[edge_position];
        let adjacent = edge_faces[&edge].clone();
        if adjacent.len() != 2 {
            continue;
        }
        let (first, second) = (adjacent[0], adjacent[1]);
        if consumed[first] || consumed[second] {
            continue;
        }
        if share_neighbouring_face(&all_faces, &edge_faces, first, second) {
            continue;
        }
        let first_tri = triangle_of(&all_faces[first]);
        let second_tri = triangle_of(&all_faces[second]);
        let Some(quad) = quad_from_triangles(first_tri, second_tri, [edge.0, edge.1]) else {
            continue;
        };
        consumed[first] = true;
        consumed[second] = true;
        let quad_face = all_faces.len();
        for quad_edge in face_edges(&quad) {
            edge_faces.entry(quad_edge).or_default().push(quad_face);
        }
        all_faces.push(quad.to_vec());
        consumed.push(false);
    }
    all_faces
        .into_iter()
        .zip(consumed)
        .filter_map(|(face, merged)| (!merged).then_some(face))
        .collect()
}

/// Blender's `BM_face_share_face_check`: whether a third face shares an edge with both faces.
fn share_neighbouring_face(
    faces: &[Vec<usize>],
    edge_faces: &HashMap<(usize, usize), Vec<usize>>,
    first: usize,
    second: usize,
) -> bool {
    let neighbours = |face: usize| -> HashSet<usize> {
        face_edges(&faces[face])
            .filter_map(|edge| edge_faces.get(&edge))
            .flatten()
            .copied()
            .filter(|other| *other != face)
            .collect()
    };
    let first_neighbours = neighbours(first);
    neighbours(second)
        .iter()
        .any(|other| first_neighbours.contains(other))
}

/// The undirected edges of a polygon, as ordered vertex pairs.
fn face_edges(face: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..face.len()).map(|corner| ordered_edge(face[corner], face[(corner + 1) % face.len()]))
}

fn quad_from_triangles(
    first: [usize; 3],
    second: [usize; 3],
    shared: [usize; 2],
) -> Option<[usize; 4]> {
    let opposite = second
        .iter()
        .copied()
        .find(|vertex| !first.contains(vertex))?;
    let mut quad = Vec::with_capacity(4);
    for index in 0..3 {
        let vertex = first[index];
        quad.push(vertex);
        if [vertex, first[(index + 1) % 3]]
            .iter()
            .all(|vertex| shared.contains(vertex))
        {
            quad.push(opposite);
        }
    }
    quad.try_into().ok()
}

fn quad_crosses_symmetry_plane(
    points: &[DVec3],
    quad: [usize; 4],
    symmetry_axes: [bool; 3],
) -> bool {
    for axis in 0..3 {
        if !symmetry_axes[axis] {
            continue;
        }
        let left = quad.iter().any(|vertex| points[*vertex][axis] < 0.0);
        let right = quad.iter().any(|vertex| points[*vertex][axis] > 0.0);
        if left && right {
            return true;
        }
    }
    false
}

fn quad_is_symmetric(points: &[DVec3], quad: [usize; 4], symmetry_axes: [bool; 3]) -> bool {
    let threshold_squared = 1.0e-8;
    for axis in 0..3 {
        if !symmetry_axes[axis] {
            continue;
        }
        let mut reflected = points[quad[0]];
        reflected[axis] = -reflected[axis];
        if reflected.distance_squared(points[quad[1]]) < threshold_squared {
            let mut other = points[quad[2]];
            other[axis] = -other[axis];
            if other.distance_squared(points[quad[3]]) < threshold_squared {
                return true;
            }
        } else if reflected.distance_squared(points[quad[3]]) < threshold_squared {
            let mut other = points[quad[2]];
            other[axis] = -other[axis];
            if other.distance_squared(points[quad[1]]) < threshold_squared {
                return true;
            }
        }
    }
    false
}

fn quad_is_convex(points: &[DVec3], quad: [usize; 4]) -> bool {
    let diagonal_a = points[quad[0]] - points[quad[2]];
    let diagonal_b = points[quad[1]] - points[quad[3]];
    let plane = diagonal_a.cross(diagonal_b);
    if plane.length_squared() < 1.0e-16 {
        return false;
    }
    let project = |point: DVec3| point - plane * (point.dot(plane) / plane.length_squared());
    let projected = quad.map(|vertex| project(points[vertex]));
    for index in 0..4 {
        let previous = projected[(index + 3) % 4];
        let current = projected[index];
        let next = projected[(index + 1) % 4];
        if (current - previous).cross(next - current).dot(plane) <= 0.0 {
            return false;
        }
    }
    true
}

fn smooth_branch_hull_points(
    points: &mut [DVec3],
    faces: &[Vec<usize>],
    frame_vertex_ids: &[[usize; 4]],
    smoothing: f64,
) {
    if smoothing == 0.0 {
        return;
    }
    let mut adjacency = vec![Vec::new(); points.len()];
    for face in faces {
        for index in 0..face.len() {
            let first = face[index];
            let second = face[(index + 1) % face.len()];
            if !adjacency[first].contains(&second) {
                adjacency[first].push(second);
            }
            if !adjacency[second].contains(&first) {
                adjacency[second].push(first);
            }
        }
    }
    let mut is_frame_vertex = vec![false; points.len()];
    for frame in frame_vertex_ids {
        for &vertex in frame {
            is_frame_vertex[vertex] = true;
        }
    }
    let source = points.to_vec();
    for (index, neighbors) in adjacency.iter().enumerate() {
        if neighbors.is_empty() {
            continue;
        }
        let average = (source[index]
            + neighbors
                .iter()
                .map(|neighbor| source[*neighbor])
                .sum::<DVec3>())
            / (neighbors.len() + 1) as f64;
        let weight = if is_frame_vertex[index] {
            smoothing * 0.5
        } else {
            smoothing
        };
        points[index] = source[index].lerp(average, weight);
    }
}
/// Blender's `build_edge_mats`: every root's edges go onto one stack, and each edge takes the frame
/// of the side that reaches it first. Returns (origin vertex, [x, y, z]) per edge.
fn edge_matrices(
    modifier: &Modifier,
    vertex_positions: &[DVec3],
    edge_endpoints: &[[usize; 2]],
    adjacency: &[Vec<(usize, usize)>],
    node_roots: &[bool],
    degrees: &[usize],
) -> Result<Vec<(usize, [DVec3; 3])>> {
    let mut mats: Vec<Option<(usize, [DVec3; 3])>> = vec![None; edge_endpoints.len()];
    // (parent vertex, edge, parent frame); a root's entries carry no frame, since roots start fresh.
    let mut stack: Vec<(usize, usize, [DVec3; 3])> = Vec::new();
    for (root, &is_root) in node_roots.iter().enumerate() {
        if is_root {
            for &(edge_index, _) in &adjacency[root] {
                stack.push((root, edge_index, [DVec3::ZERO; 3]));
            }
        }
    }
    while let Some((parent, edge_index, parent_mat)) = stack.pop() {
        if mats[edge_index].is_some() {
            continue;
        }
        let [first, second] = edge_endpoints[edge_index];
        let child = if first == parent { second } else { first };
        let mat = if node_roots[parent] || degrees[parent] > 2 {
            calc_edge_mat(vertex_positions[parent], vertex_positions[child])
        } else {
            propagate_frame(
                parent_mat,
                vertex_positions[parent],
                vertex_positions[child],
            )
        };
        mats[edge_index] = Some((parent, mat));
        for &(next_edge, _) in &adjacency[child] {
            stack.push((child, next_edge, mat));
        }
    }
    mats.into_iter().collect::<Option<Vec<_>>>().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "No valid root vertex found (you need one per mesh island you want to skin)",
            serde_json::json!({"modifier_id":modifier.id,"modifier_warning":true}),
        )
    })
}

/// Blender's `calc_edge_mat`: a root's frame, with world Z as the up reference; an edge along Z uses
/// the fixed X/Y frame. Evaluated in `f32` in Blender's operation order, widened at the output.
fn calc_edge_mat(from: DVec3, to: DVec3) -> [DVec3; 3] {
    const Z_UP: [f32; 3] = [0.0, 0.0, 1.0];
    let x_axis = skin_float_edge_direction(from, to);
    let dot = skin_float_dot(x_axis, Z_UP);
    // Blender's FLT_EPSILON pole tolerance, exclusive at both ends.
    let mat = if dot > -1.0 + f32::EPSILON && dot < 1.0 - f32::EPSILON {
        let y_axis = skin_float_normalize_v3(skin_float_cross(Z_UP, x_axis));
        let z_axis = skin_float_normalize_v3(skin_float_cross(x_axis, y_axis));
        [x_axis, y_axis, z_axis]
    } else {
        [x_axis, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    };
    mat.map(skin_float_widen)
}

/// Blender's `build_emats_stack`: an edge's frame is its parent edge's frame rotated onto its
/// direction, evaluated in `f32`. Like Blender, the rotated axes are not renormalized.
fn propagate_frame(parent: [DVec3; 3], from: DVec3, to: DVec3) -> [DVec3; 3] {
    let [parent_x, parent_y, parent_z] = parent.map(skin_float_point);
    let x_axis = skin_float_edge_direction(from, to);
    let angle = skin_float_angle(parent_x, x_axis);
    let axis = skin_float_normalize_v3(skin_float_cross(parent_x, x_axis));
    [
        skin_float_widen(x_axis),
        skin_float_widen(skin_float_rotate(parent_y, axis, angle)),
        skin_float_widen(skin_float_rotate(parent_z, axis, angle)),
    ]
}

/// Blender's `connection_node_mat`: the y and z axes halfway between a chain node's incoming frame
/// and the outgoing edge direction, evaluated in `f32`.
fn bisector_yz(incoming: [DVec3; 3], outgoing_x: DVec3) -> [DVec3; 2] {
    let [incoming_x, incoming_y, incoming_z] = incoming.map(skin_float_point);
    let outgoing_x = skin_float_point(outgoing_x);
    let angle = skin_float_angle(incoming_x, outgoing_x) / 2.0;
    let axis = skin_float_normalize_v3(skin_float_cross(incoming_x, outgoing_x));
    [
        skin_float_widen(skin_float_rotate(incoming_y, axis, angle)),
        skin_float_widen(skin_float_rotate(incoming_z, axis, angle)),
    ]
}

/// Blender's `create_frame`: the four corners of a cross-section, offset along the edge direction.
/// Evaluated in `f32` in Blender's operation order; `offset` is the already-scaled edge offset.
fn frame_corners(
    center: DVec3,
    radius: [f64; 2],
    y_axis: DVec3,
    z_axis: DVec3,
    offset: [f32; 3],
) -> [DVec3; 4] {
    let co = skin_float_point(center);
    let ry = skin_float_scale(y_axis, radius[0] as f32);
    let rz = skin_float_scale(z_axis, radius[1] as f32);
    [
        skin_float_sub(skin_float_add(co, ry), rz),
        skin_float_sub(skin_float_sub(co, ry), rz),
        skin_float_add(skin_float_sub(co, ry), rz),
        skin_float_add(skin_float_add(co, ry), rz),
    ]
    .map(|corner| skin_float_widen(skin_float_add(corner, offset)))
}

/// Blender's `skin_choose_quad_bridge_order`: the rotation or reversal of `second` whose corners lie
/// nearest `first`; the earliest candidate wins ties.
fn bridge_order(points: &[DVec3], first: [usize; 4], second: [usize; 4]) -> [usize; 4] {
    let mut best = [0_usize; 4];
    let mut best_length = f64::INFINITY;
    for reverse in [false, true] {
        for start in 0..4 {
            let order = std::array::from_fn::<usize, 4, _>(|j| {
                if reverse {
                    3 - (j + start) % 4
                } else {
                    (j + start) % 4
                }
            });
            let length: f64 = (0..4)
                .map(|j| points[first[j]].distance_squared(points[second[order[j]]]))
                .sum();
            if length < best_length {
                best_length = length;
                best = order;
            }
        }
    }
    best.map(|index| second[index])
}

/// Blender's `connect_frames`: four quads between two frames, reversed as a whole when they face the
/// centre of the band.
fn connect_frames(
    points: &[DVec3],
    first: [usize; 4],
    second: [usize; 4],
    faces: &mut Vec<Polygon>,
) {
    let quads: Vec<[usize; 4]> = (0..4)
        .map(|side| {
            let next = (side + 1) % 4;
            [second[side], second[next], first[next], first[side]]
        })
        .collect();
    let centre_of =
        |quad: &[usize; 4]| quad.iter().map(|index| points[*index]).sum::<DVec3>() * 0.25;
    let centre = quads.iter().map(centre_of).sum::<DVec3>() * 0.25;
    let facing: f64 = quads
        .iter()
        .map(|quad| {
            let normal = (points[quad[2]] - points[quad[0]])
                .cross(points[quad[3]] - points[quad[1]])
                .normalize_or_zero();
            normal.dot(centre - centre_of(quad))
        })
        .sum();
    for quad in quads {
        if facing > 0.0 {
            faces.push((vec![quad[3], quad[2], quad[1], quad[0]], 0));
        } else {
            faces.push((quad.to_vec(), 0));
        }
    }
}

/// Blender interpolates a vertex group onto a blended skin vertex only when every source vertex
/// has it; a group present on one source alone is dropped rather than zero-filled.
fn drop_partial_vertex_groups(source: &Mesh, origin: &Origin, result: &mut Mesh) {
    let Some(groups) = result
        .attributes
        .get_mut("vertex_groups")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let source_groups = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    for (name, weights) in groups.iter_mut() {
        let Some(weights) = weights.as_object_mut() else {
            continue;
        };
        let source_weights = source_groups
            .and_then(|groups| groups.get(name))
            .and_then(Value::as_object);
        // Whether each source vertex, by storage index, has an entry for this group.
        let has_group: Vec<bool> = source
            .vertices
            .iter()
            .map(|vertex| {
                source_weights
                    .is_some_and(|weights| weights.contains_key(&format!("v{}", vertex.id)))
            })
            .collect();
        for (output, sources) in origin.vertices.iter().enumerate() {
            if sources.len() < 2 {
                continue;
            }
            if !sources.iter().all(|(index, _)| has_group[*index]) {
                weights.remove(&format!("v{}", result.vertices[output].id));
            }
        }
    }
}

fn skin_radius(
    radii: Option<&serde_json::Map<String, Value>>,
    vertex_index: usize,
    modifier: &Modifier,
) -> Result<[f64; 2]> {
    let Some(value) = radii.and_then(|radii| radii.get(&vertex_index.to_string())) else {
        return Ok([0.25, 0.25]);
    };
    let Some(values) = value.as_array() else {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two finite radii per vertex",
        ));
    };
    if values.len() != 2 {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two finite radii per vertex",
        ));
    }
    let first = values
        .first()
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid_parameter(modifier, "skin_radii", "two finite radii per vertex"))?;
    let second = values
        .get(1)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid_parameter(modifier, "skin_radii", "two finite radii per vertex"))?;
    if !first.is_finite() || !second.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two finite radii per vertex",
        ));
    }
    Ok([first, second])
}

#[cfg(test)]
mod tagged_edge_tests {
    use super::TaggedEdges;

    #[test]
    fn weld_moves_a_tag_onto_a_new_target_edge() {
        let mut tags = TaggedEdges(vec![(4, 5)]);
        tags.weld(&[vec![0, 1, 2]], &[(4, 1)]);
        assert_eq!(tags.0, vec![(1, 5)]);
    }

    #[test]
    fn weld_loses_a_tag_when_the_target_edge_already_exists() {
        let mut tags = TaggedEdges(vec![(4, 5)]);
        tags.weld(&[vec![1, 5, 2]], &[(4, 1)]);
        assert_eq!(tags.0, Vec::<(usize, usize)>::new());
    }

    #[test]
    fn weld_deletes_a_tagged_edge_that_collapses() {
        let mut tags = TaggedEdges(vec![(4, 5)]);
        tags.weld(&[], &[(4, 5)]);
        assert_eq!(tags.0, Vec::<(usize, usize)>::new());
    }
}
