// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_skin.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

use super::*;

use std::ptr::NonNull;

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
    let status = unsafe {
        potter_skin_hull_compute(coordinates.as_ptr(), input.len(), &mut raw_hull)
    };
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
        .chunks_exact(3)
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
    let mut edge_lengths = Vec::with_capacity(mesh.edges.len());
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
        edge_lengths.push(length);
    }
    let mut edge_chains = Vec::with_capacity(mesh.edges.len());
    for (edge_index, [first, second]) in edge_endpoints.iter().copied().enumerate() {
        let first_radius = node_radii[first];
        let second_radius = node_radii[second];
        let average_radius =
            (first_radius[0] + first_radius[1] + second_radius[0] + second_radius[1]) * 0.5;
        let first_is_branch = degrees[first] > 2;
        let second_is_branch = degrees[second] > 2;
        let loose_branch = (first_is_branch && node_loose[first])
            || (second_is_branch && node_loose[second]);
        let mut subdivisions = if loose_branch {
            if first_is_branch && second_is_branch {
                2
            } else {
                0
            }
        } else if average_radius > 0.0 {
            (edge_lengths[edge_index] / average_radius).min(128.0) as usize
        } else {
            // Blender crashes on a negative subdivision count; keep the finite radii accepted.
            0
        };
        if first_is_branch && second_is_branch && subdivisions < 2 {
            subdivisions = 2;
        }
        let first_half = (first_radius[0] + first_radius[1]) * 0.5;
        let second_half = (second_radius[0] + second_radius[1]) * 0.5;
        // Blender: radrat = half(r1) / half(r0), then (radrat + 1) / 2 when finite, else 1.
        let ratio = second_half / first_half;
        let radius_ratio = if ratio.is_finite() {
            (ratio + 1.0) * 0.5
        } else {
            1.0
        };
        let mut chain = Vec::with_capacity(subdivisions + 2);
        chain.push(first);
        for subdivision in 0..subdivisions {
            let ratio = (subdivision + 1) as f64 / (subdivisions + 1) as f64;
            let fraction = ratio.powf(radius_ratio);
            let index = node_positions.len();
            node_positions.push(source_positions[first].lerp(source_positions[second], fraction));
            node_origins.push(vec![(first, 1.0 - fraction), (second, fraction)]);
            node_radii.push([
                first_radius[0] + (second_radius[0] - first_radius[0]) * fraction,
                first_radius[1] + (second_radius[1] - first_radius[1]) * fraction,
            ]);
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
                let offset = (radius[0] + radius[1]) * 0.5;
                let first = frames.len();
                frames.push(frame_corners(
                    center,
                    radius,
                    DVec3::X,
                    DVec3::Y,
                    DVec3::Z * offset,
                ));
                frame_nodes.push(vertex);
                let second = frames.len();
                frames.push(frame_corners(
                    center,
                    radius,
                    DVec3::X,
                    DVec3::Y,
                    DVec3::Z * -offset,
                ));
                frame_nodes.push(vertex);
                box_frames[vertex] = Some([first, second]);
            }
            [(edge_index, _)] => {
                let [_, y_axis, z_axis] = edge_mats[*edge_index].1;
                node_frame[vertex] = frames.len();
                frames.push(frame_corners(
                    center,
                    radius,
                    y_axis,
                    z_axis,
                    DVec3::ZERO,
                ));
                frame_nodes.push(vertex);
            }
            [(first_edge, _), (second_edge, _)] => {
                if (edge_mats[*first_edge].0 == vertex) == (edge_mats[*second_edge].0 == vertex) {
                    let offset = (radius[0] + radius[1]) * 0.5;
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
                            direction * offset,
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
                    frames.push(frame_corners(
                        center,
                        radius,
                        y_axis,
                        z_axis,
                        DVec3::ZERO,
                    ));
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
                DVec3::ZERO,
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
                PotError::new(ErrorCode::InternalError, "skin branch has no neighboring frame")
            })?;
            rings.push(frame_at(neighbor, edge_index));
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
                    PotError::new(ErrorCode::InternalError, "skin frame merge target is missing")
                })?;
            frame_vertex_ids[frame_index][corner] = target;
        }
    }
    let mut branch_faces = Vec::new();
    let mut detached_frames = Vec::new();
    for ring_frames in &branch_rings {
        let rings: Vec<[usize; 4]> = ring_frames
            .iter()
            .map(|frame| frame_vertex_ids[*frame])
            .collect();
        let hull = branch_hull_faces(&points, &rings, &mut branch_faces)?;
        for (&frame, (detached, inside)) in ring_frames.iter().zip(hull.frame_states) {
            if detached {
                detached_frames.push((frame, inside));
            }
        }
    }
    branch_faces = merge_triangle_faces(&points, branch_faces, symmetry_axes);
    repair_detached_branch_frames(
        &mut points,
        &mut origins,
        &mut frame_vertex_ids,
        &mut branch_faces,
        &detached_frames,
    );
    smooth_branch_hull_points(&mut points, &branch_faces, branch_smoothing);
    faces.extend(branch_faces.into_iter().map(|face| (face, 0)));
    for ids in seam_frames.iter().flatten() {
        let [first, second] = *ids;
        let second_ids = bridge_order(
            &points,
            frame_vertex_ids[first],
            frame_vertex_ids[second],
        );
        connect_frames(&points, frame_vertex_ids[first], second_ids, &mut faces);
    }
    for ids in box_frames.iter().flatten() {
        let [first, second] = *ids;
        let first_ids = frame_vertex_ids[first];
        let second_ids = frame_vertex_ids[second];
        connect_frames(&points, first_ids, second_ids, &mut faces);
        faces.push((vec![first_ids[3], first_ids[2], first_ids[1], first_ids[0]], 0));
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
    let mut faces: Vec<Polygon> = merge_triangle_faces(&points, polygons, symmetry_axes)
        .into_iter()
        .map(|face| (face, 0))
        .collect();
    let origin = Origin {
        vertices: origins,
        faces: vec![None; faces.len()],
        corners: None,
        raw_normals: false,
    };
    let mut result = build_mesh(points, faces, Vec::new(), &origin, mesh)?;
    result.attributes.remove("blender_skin_vertices");
    result.attributes.remove("skin_radii");
    result.attributes.remove("skin_roots");
    // Shading lives in `shade_smooth`; a remapped `blender_polygon_smooth` would conflict.
    result.attributes.remove("blender_polygon_smooth");
    if smooth_shade {
        crate::geom::normals::shade_faces_smooth(&mut result);
    }
    // Blender's Skin output has no UV layers, so UV data is dropped (reported as attribute loss).
    let uv_names: Vec<String> = result
        .attributes
        .get("blender_uv_layers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| layer.get("name").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
        .collect();
    result.attributes.remove("blender_uv_layers");
    result.attributes.remove("uv_map");
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
                for second in first + 1..rings.len() {
                    let second_frame = rings[second];
                    let threshold = first_side.min(frame_side_length(frames[second_frame])) * 0.5;
                    for second_corner in 0..4 {
                        if merges[second_frame][second_corner].is_some()
                            || targets[second_frame][second_corner]
                        {
                            continue;
                        }
                        let distance = frames[first_frame][first_corner]
                            .distance(frames[second_frame][second_corner]);
                        if distance < threshold {
                            let midpoint = (frames[first_frame][first_corner]
                                + frames[second_frame][second_corner])
                                * 0.5;
                            frames[first_frame][first_corner] = midpoint;
                            frames[second_frame][second_corner] = midpoint;
                            merges[second_frame][second_corner] =
                                Some((first_frame, first_corner));
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

fn frame_side_length(frame: [DVec3; 4]) -> f64 {
    (frame[0].distance(frame[1]) + frame[1].distance(frame[2])) * 0.5
}
struct BranchHullBuild {
    frame_states: Vec<(bool, [bool; 4])>,
}

fn branch_hull_faces(
    points: &[DVec3],
    rings: &[[usize; 4]],
    faces: &mut Vec<Vec<usize>>,
) -> Result<BranchHullBuild> {
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
    let mut frame_states = Vec::with_capacity(rings.len());
    for ring in rings {
        let mut inside = [false; 4];
        let first_inside = (0..4)
            .filter(|corner| !hull_vertices.contains(&ring[*corner]))
            .min_by_key(|corner| ring[*corner]);
        let mut detached = false;
        if let Some(corner) = first_inside {
            inside[corner] = true;
            detached = true;
        }
        if !detached
            && (0..4).any(|corner| {
                !hull_edges.contains(&ordered_edge(ring[corner], ring[(corner + 1) % 4]))
            })
        {
            detached = true;
        }

        if !detached {
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
                detached = true;
            }
        }
        frame_states.push((detached, inside));
    }

    let mut face_index = 0;
    faces.retain(|_| {
        let keep = !removed_faces[face_index];
        face_index += 1;
        keep
    });
    Ok(BranchHullBuild { frame_states })
}

fn face_has_edge(face: &[usize], edge: (usize, usize)) -> bool {
    (0..face.len()).any(|corner| {
        ordered_edge(face[corner], face[(corner + 1) % face.len()]) == edge
    })
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
        {
            if faces[target].len() == 3 {
                subdivide_skin_target_face(points, origins, faces, target)?;
            }
            if faces[target].len() == 4 {
                let target_ring = faces[target].clone();
                let frame = frame_vertex_ids[frame_index];
                let order = skin_choose_quad_bridge_order(points, target_ring.as_slice(), frame);
                for corner in 0..4 {
                    let next = (corner + 1) % 4;
                    append_welded_face(
                        faces,
                        [
                            target_ring[next],
                            frame[order[next]],
                            frame[order[corner]],
                            target_ring[corner],
                        ],
                    );
                }
                continue;
            }
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
) -> Result<()> {
    let face = faces.get(target).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Skin target face index is out of range",
        )
    })?;
    let mut longest_edge = None;
    let mut longest_length = f32::NEG_INFINITY;
    for corner in 0..face.len() {
        let first = face[corner];
        let second = face[(corner + 1) % face.len()];
        let length = skin_float_distance_squared(points[first], points[second]);
        if length > longest_length {
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
    let midpoint = DVec3::new(
        ((a.x as f32 + b.x as f32) * 0.5) as f64,
        ((a.y as f32 + b.y as f32) * 0.5) as f64,
        ((a.z as f32 + b.z as f32) * 0.5) as f64,
    );
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
    Ok(())
}

fn append_welded_face(faces: &mut Vec<Vec<usize>>, corners: [usize; 4]) {
    let mut face = Vec::with_capacity(4);
    for vertex in corners {
        if face.last() != Some(&vertex) {
            face.push(vertex);
        }
    }
    if face.first() == face.last() {
        face.pop();
    }
    if face.len() < 3
        || face
            .iter()
            .enumerate()
            .any(|(index, vertex)| face[..index].contains(vertex))
    {
        return;
    }
    faces.push(face);
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
            for corner in 0..4 {
                order[corner] = if reversed == 0 {
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


fn triangle_normal(points: &[DVec3], triangle: [usize; 3]) -> DVec3 {
    (points[triangle[1]] - points[triangle[0]])
        .cross(points[triangle[2]] - points[triangle[0]])
        .normalize_or_zero()
}

fn triangle_area(points: &[DVec3], triangle: [usize; 3]) -> f64 {
    (points[triangle[1]] - points[triangle[0]])
        .cross(points[triangle[2]] - points[triangle[0]])
        .length()
        * 0.5
}

/// Blender's heap-based merge pass over adjacent triangle faces.
fn merge_triangle_faces(
    points: &[DVec3],
    faces: Vec<Vec<usize>>,
    symmetry_axes: [bool; 3],
) -> Vec<Vec<usize>> {
    let mut triangle_faces = Vec::new();
    let mut triangles = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.len() == 3 {
            triangle_faces.push(face_index);
            triangles.push([face[0], face[1], face[2]]);
        }
    }
    let mut candidates = Vec::<(f64, usize, usize, [usize; 4])>::new();
    for first in 0..triangles.len() {
        for second in first + 1..triangles.len() {
            let shared: Vec<usize> = triangles[first]
                .iter()
                .copied()
                .filter(|vertex| triangles[second].contains(vertex))
                .collect();
            if shared.len() != 2 {
                continue;
            }
            let Some(quad) =
                quad_from_triangles(triangles[first], triangles[second], [shared[0], shared[1]])
            else {
                continue;
            };
            let mut score = (triangle_area(points, triangles[first])
                + triangle_area(points, triangles[second]))
                * triangle_normal(points, triangles[first])
                    .dot(triangle_normal(points, triangles[second]));
            if quad_crosses_symmetry_plane(points, quad, symmetry_axes) {
                if quad_is_symmetric(points, quad, symmetry_axes) {
                    score *= 10.0;
                } else {
                    continue;
                }
            }
            if quad_is_convex(points, quad) {
                candidates.push((score, first, second, quad));
            }
        }
    }
    candidates.sort_by(|first, second| second.0.total_cmp(&first.0));
    let mut merged_triangles = vec![false; triangles.len()];
    let mut merged_faces = vec![false; faces.len()];
    let mut quads = Vec::new();
    for (_, first, second, quad) in candidates {
        if merged_triangles[first] || merged_triangles[second] {
            continue;
        }
        merged_triangles[first] = true;
        merged_triangles[second] = true;
        merged_faces[triangle_faces[first]] = true;
        merged_faces[triangle_faces[second]] = true;
        quads.push(quad.to_vec());
    }
    let mut output: Vec<Vec<usize>> = faces
        .into_iter()
        .enumerate()
        .filter_map(|(index, face)| (!merged_faces[index]).then_some(face))
        .collect();
    output.extend(quads);
    output
}
fn quad_from_triangles(
    first: [usize; 3],
    second: [usize; 3],
    shared: [usize; 2],
) -> Option<[usize; 4]> {
    let opposite = second.iter().copied().find(|vertex| !first.contains(vertex))?;
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

fn smooth_branch_hull_points(points: &mut [DVec3], faces: &[Vec<usize>], smoothing: f64) {
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
        points[index] = source[index].lerp(average, smoothing * 0.5);
    }
}
/// Blender's build_edge_mats: every root's edges go onto one stack, and each edge takes the frame
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
        let direction = (vertex_positions[child] - vertex_positions[parent]).normalize();
        let mat = if node_roots[parent] || degrees[parent] > 2 {
            calc_edge_mat(direction)
        } else {
            propagate_frame(parent_mat, direction)
        };
        mats[edge_index] = Some((parent, mat));
        for &(next_edge, _) in &adjacency[child] {
            stack.push((child, next_edge, mat));
        }
    }
    mats.into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            invalid_parameter(modifier, "mesh", "a root vertex in every connected skeleton")
        })
}

/// Blender's calc_edge_mat: a root's frame, with world Z as the up reference; an edge along Z uses
/// the fixed X/Y frame.
fn calc_edge_mat(x_axis: DVec3) -> [DVec3; 3] {
    let dot = x_axis.dot(DVec3::Z);
    if (-1.0 + f64::EPSILON..1.0 - f64::EPSILON).contains(&dot) {
        let y_axis = DVec3::Z.cross(x_axis).normalize_or_zero();
        [
            x_axis,
            y_axis,
            x_axis.cross(y_axis).normalize_or_zero(),
        ]
    } else {
        [x_axis, DVec3::X, DVec3::Y]
    }
}

/// Blender's build_emats_stack: an edge's frame is its parent edge's frame rotated onto its direction.
fn propagate_frame(parent: [DVec3; 3], x_axis: DVec3) -> [DVec3; 3] {
    let dot = parent[0].dot(x_axis).clamp(-1.0, 1.0);
    let rotation = DQuat::from_axis_angle(parent[0].cross(x_axis).normalize_or_zero(), dot.acos());
    [
        x_axis,
        (rotation * parent[1]).normalize_or_zero(),
        (rotation * parent[2]).normalize_or_zero(),
    ]
}

/// Blender's connection_node_mat: the y and z axes halfway between a chain node's incoming frame
/// and the outgoing edge direction.
fn bisector_yz(incoming: [DVec3; 3], outgoing_x: DVec3) -> [DVec3; 2] {
    let angle = incoming[0].dot(outgoing_x).clamp(-1.0, 1.0).acos() / 2.0;
    let rotation = DQuat::from_axis_angle(incoming[0].cross(outgoing_x).normalize_or_zero(), angle);
    [rotation * incoming[1], rotation * incoming[2]]
}

/// Blender's create_frame: the four corners of a cross-section, offset along the edge direction.
fn frame_corners(
    center: DVec3,
    radius: [f64; 2],
    y_axis: DVec3,
    z_axis: DVec3,
    offset: DVec3,
) -> [DVec3; 4] {
    let y = y_axis * radius[0];
    let z = z_axis * radius[1];
    [
        center + y - z,
        center - y - z,
        center - y + z,
        center + y + z,
    ]
    .map(|corner| corner + offset)
}

/// Blender's skin_choose_quad_bridge_order: the rotation or reversal of `second` whose corners lie
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

/// Blender's connect_frames: four quads between two frames, reversed as a whole when they face the
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
    let first = values.first().and_then(Value::as_f64).ok_or_else(|| {
        invalid_parameter(modifier, "skin_radii", "two finite radii per vertex")
    })?;
    let second = values.get(1).and_then(Value::as_f64).ok_or_else(|| {
        invalid_parameter(modifier, "skin_radii", "two finite radii per vertex")
    })?;
    if !first.is_finite() || !second.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "skin_radii",
            "two finite radii per vertex",
        ));
    }
    Ok([first, second])
}
