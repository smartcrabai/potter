// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender Explode modifier port (`MOD_explode.cc`, Blender 5.2).

use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DVec3};
use serde_json::{Map, Value};

use crate::{
    blender_f32,
    blender_rng::BlenderRng,
    geom::{
        clnor::{self, CornerNormalCode, NormalCodeOptions},
        Edge, Face, Mesh, Vertex,
        color_layers::{self, ColorDomain},
        edge_key,
        provenance::{Origin, remap_attributes},
        triangulate::tessellate_ngon,
        uv_layers,
    },
    sim::{ParticleLifeState, ParticleState},
};

use super::simulation::{bool_param, invalid, number_param, value_or_default};

type EdgeKey = (usize, usize);

#[derive(Clone)]
struct TessFace {
    source_face: usize,
    vertices: Vec<usize>,
    source_corners: Vec<usize>,
}

#[derive(Clone)]
struct Piece {
    tess_face: usize,
    vertices: Vec<usize>,
    particle: usize,
    corner_slots: Vec<usize>,
    /// UV-corner sources in tessface-local slots for Blender's exceptional remap tables.
    uv_corners: Option<Vec<Vec<(usize, f64)>>>,
}

#[derive(Clone, Copy)]
struct KdNode {
    co: [f32; 3],
    index: usize,
    left: Option<usize>,
    right: Option<usize>,
    axis: usize,
}

/// Splits and transforms mesh fragments using Blender's particle-to-face, edge-cut and
/// vertex-to-particle-group rules.
pub fn explode(
    mesh: &Mesh,
    particles: &[ParticleState],
    system_seed: u32,
    target_world: DMat4,
    params: &Map<String, Value>,
    frame: f64,
) -> crate::error::Result<Mesh> {
    let parents: Vec<&ParticleState> = particles
        .iter()
        .filter(|particle| !particle.is_child)
        .collect();
    if parents.is_empty() {
        return Ok(mesh.clone());
    }

    let show_alive = bool_param("explode", params, "show_alive")?;
    let show_dead = bool_param("explode", params, "show_dead")?;
    let show_unborn = bool_param("explode", params, "show_unborn")?;
    let use_edge_cut = bool_param("explode", params, "use_edge_cut")?;
    let use_size = bool_param("explode", params, "use_size")?;
    let invert_group = bool_param("explode", params, "invert_vertex_group")?;
    let protect = number_param("explode", params, "protect")?;
    if !(0.0..=1.0).contains(&protect) {
        return Err(invalid("protect must be in [0, 1]"));
    }
    let group_name = value_or_default("explode", params, "vertex_group")?
        .as_str()
        .ok_or_else(|| invalid("vertex_group must be a string"))?;
    let requested_uv = value_or_default("explode", params, "particle_uv")?
        .as_str()
        .ok_or_else(|| invalid("particle_uv must be a string"))?;

    let inverse_world = target_world.inverse();
    let world32 = blender_f32::mat4(target_world);
    let inverse32 = blender_f32::mat4(inverse_world);
    let vertex_indices: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let source_corner_offsets = corner_offsets(mesh);
    let tess_faces = tessellate(mesh, &vertex_indices, &positions)?;

    let emitter_points: Vec<[f32; 3]> = parents
        .iter()
        .map(|particle| {
            let local = inverse_world.transform_point3(DVec3::from_array(particle.birth_position));
            [local.x as f32, local.y as f32, local.z as f32]
        })
        .collect();
    let mut tree: Vec<KdNode> = emitter_points
        .iter()
        .enumerate()
        .map(|(index, co)| KdNode {
            co: *co,
            index,
            left: None,
            right: None,
            axis: 0,
        })
        .collect();
    let root = balance(&mut tree, 0, 0);

    let vertex_group = if group_name.is_empty() {
        None
    } else {
        mesh.attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .and_then(|groups| groups.get(group_name))
            .and_then(Value::as_object)
    };
    let mut protected = vec![false; mesh.vertices.len()];
    if let Some(weights) = vertex_group {
        let mut rng = BlenderRng::seed_random(system_seed);
        let protect = protect as f32;
        let weights: Vec<f32> = mesh
            .vertices
            .iter()
            .map(|vertex| {
                let weight = weights
                    .get(&format!("v{}", vertex.id))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                if invert_group {
                    1.0_f32 - weight
                } else {
                    weight
                }
            })
            .collect();
        for (index, weight) in weights.into_iter().enumerate() {
            let mut val = rng.get_float();
            val = (1.0_f32 - protect) * val + protect * 0.5_f32;
            protected[index] = val < weight;
        }
    }

    let mut face_particles = Vec::with_capacity(tess_faces.len());
    for tess_face in &tess_faces {
        let center = face_center(&tess_face.vertices, &positions);
        let particle = nearest(&tree, root, center);
        let face_is_protected = tess_face.vertices.iter().any(|vertex| protected[*vertex]);
        face_particles.push(if face_is_protected {
            parents.len()
        } else {
            particle
        });
    }

    let (pieces, split_points, split_edges) = if use_edge_cut {
        cut_edges(&tess_faces, &face_particles, &positions)?
    } else {
        (
            tess_faces
                .iter()
                .enumerate()
                .map(|(index, tess_face)| Piece {
                    tess_face: index,
                    vertices: tess_face.vertices.clone(),
                    particle: face_particles[index],
                    corner_slots: (0..tess_face.vertices.len()).collect(),
                    uv_corners: None,
                })
                .collect(),
            Vec::new(),
            Vec::new(),
        )
    };
    let mut point_origins: Vec<Vec<(usize, f64)>> = (0..mesh.vertices.len())
        .map(|index| vec![(index, 1.0)])
        .collect();
    // Blender copies point custom data from the higher-index edge endpoint.
    point_origins.extend(split_edges.iter().map(|(_, second)| vec![(*second, 1.0)]));

    let mut output = Mesh::default();
    let mut vertex_groups = BTreeMap::<(usize, usize), u32>::new();
    let mut origin = Origin::default();
    let mut output_corner_copies = Vec::new();
    let mut output_uv_sources = Vec::new();
    let mut particle_ages = Vec::new();
    let point_position = |index: usize| {
        if index < positions.len() {
            positions[index]
        } else {
            split_points[index - positions.len()]
        }
    };
    for piece in &pieces {
        let particle = (piece.particle < parents.len()).then_some(piece.particle);
        if let Some(index) = particle {
            let visible = match parents[index].life_state {
                ParticleLifeState::Alive => show_alive,
                ParticleLifeState::Dead => show_dead,
                ParticleLifeState::Unborn => show_unborn,
            };
            if !visible {
                continue;
            }
        }
        let unborn = particle.is_some_and(|index| frame < parents[index].birth_frame);
        let group = if particle.is_none() || unborn {
            parents.len()
        } else {
            particle.unwrap_or(parents.len())
        };
        let mut out_vertices = Vec::with_capacity(piece.vertices.len());
        let mut copy_corner_sources = Vec::with_capacity(piece.vertices.len());
        let mut uv_corner_sources = Vec::with_capacity(piece.vertices.len());
        let tess_face = &tess_faces[piece.tess_face];
        let source_face_offset = source_corner_offsets[tess_face.source_face];
        for (corner, &point) in piece.vertices.iter().enumerate() {
            let key = (point, group);
            let id = if let Some(id) = vertex_groups.get(&key) {
                *id
            } else {
                let id = take_id(&mut output.next_id.vertex)?;
                let co = if let Some(index) = particle.filter(|_| !unborn) {
                    let state = parents[index];
                    let size = use_size.then_some(state.size as f32);
                    DVec3::from_array(
                        blender_f32::explode_fragment(
                            blender_f32::vec3(point_position(point).to_array()),
                            (&world32, &inverse32),
                            (
                                blender_f32::vec3(state.birth_position),
                                blender_f32::quat_from_xyzw(state.birth_rotation),
                            ),
                            (
                                blender_f32::vec3(state.position),
                                blender_f32::quat_from_xyzw(state.rotation),
                            ),
                            size,
                        )
                        .map(f64::from),
                    )
                } else {
                    point_position(point)
                };
                output.vertices.push(Vertex { id, co });
                origin.vertices.push(point_origins[point].clone());
                vertex_groups.insert(key, id);
                id
            };
            out_vertices.push(id);
            let uv_sources = if let Some(uv_corners) = &piece.uv_corners {
                uv_corners[corner]
                    .iter()
                    .map(|(source_corner, weight)| {
                        (
                            source_face_offset + tess_face.source_corners[*source_corner],
                            *weight,
                        )
                    })
                    .collect()
            } else {
                source_corner_weights(
                    point,
                    piece.tess_face,
                    &tess_faces,
                    &split_edges,
                    mesh,
                    &source_corner_offsets,
                )?
            };
            uv_corner_sources.push(uv_sources);
            copy_corner_sources.push(
                tess_face
                    .source_corners
                    .get(piece.corner_slots[corner])
                    .map(|source_corner| vec![(source_face_offset + *source_corner, 1.0)])
                    .unwrap_or_default(),
            );
        }
        rotate_valid_face(
            &mut out_vertices,
            &mut copy_corner_sources,
            &mut uv_corner_sources,
        );
        let source_face = tess_faces[piece.tess_face].source_face;
        let face_id = take_id(&mut output.next_id.face)?;
        output.faces.push(Face {
            id: face_id,
            vertices: out_vertices,
            material_index: mesh.faces[source_face].material_index,
        });
        origin.faces.push(Some(source_face));
        output_corner_copies.extend(copy_corner_sources);
        output_uv_sources.extend(uv_corner_sources);
        particle_ages.push(particle_age(particle.map(|index| parents[index]), frame));
    }

    let mut seen_edges = BTreeSet::new();
    for face in &output.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            let key = edge_key(first, second);
            if seen_edges.insert(key) {
                let id = take_id(&mut output.next_id.edge)?;
                output.edges.push(Edge {
                    id,
                    vertices: [key.0, key.1],
                });
            }
        }
    }
    origin.corners = Some(output_corner_copies);
    remap_attributes(mesh, &mut output, &origin)?;
    // Blender rebuilds Explode loops from legacy MFace data, so their custom-normal codes are Auto.
    if clnor::corner_codes(mesh)?.is_some() {
        let automatic =
            vec![CornerNormalCode::Auto; output.faces.iter().map(|face| face.vertices.len()).sum()];
        clnor::write_codes(
            &mut output,
            Some(automatic.as_slice()),
            NormalCodeOptions::default(),
        )?;
    }
    if let Some(corner_sources) = origin.corners.as_deref() {
        fill_triangle_byte_color_padding(mesh, &mut output, corner_sources)?;
    }
    // remap_attributes carries layer names, pins and face presence; Blender's mask-23 corner sources then replace the face values.
    remap_uv_layers(mesh, &mut output, &output_uv_sources)?;
    apply_particle_uv(mesh, &mut output, requested_uv, &particle_ages);
    output
        .validate()
        .map_err(|error| invalid(&format!("explode produced invalid geometry: {error}")))?;
    Ok(output)
}

fn tessellate(
    mesh: &Mesh,
    vertex_indices: &BTreeMap<u32, usize>,
    positions: &[DVec3],
) -> crate::error::Result<Vec<TessFace>> {
    let mut output = Vec::new();
    for (source_face, face) in mesh.faces.iter().enumerate() {
        let vertices: Vec<usize> = face
            .vertices
            .iter()
            .map(|id| {
                vertex_indices
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("explode face references a missing vertex"))
            })
            .collect::<crate::error::Result<_>>()?;
        match vertices.len() {
            3 => output.push(TessFace {
                source_face,
                vertices,
                source_corners: vec![0, 1, 2],
            }),
            4 => {
                let mut corners = vec![0, 1, 2, 3];
                let mut vertices = vertices;
                if vertices[2] == 0 || vertices[3] == 0 {
                    vertices.rotate_left(2);
                    corners.rotate_left(2);
                }
                output.push(TessFace {
                    source_face,
                    vertices,
                    source_corners: corners,
                });
            }
            count if count > 4 => {
                let corners: Vec<[f32; 3]> = vertices
                    .iter()
                    .map(|index| {
                        let co = positions[*index];
                        [co.x as f32, co.y as f32, co.z as f32]
                    })
                    .collect();
                for triangle in tessellate_ngon(&corners) {
                    let tri_vertices = triangle.iter().map(|corner| vertices[*corner]).collect();
                    output.push(TessFace {
                        source_face,
                        vertices: tri_vertices,
                        source_corners: triangle.to_vec(),
                    });
                }
            }
            _ => {
                return Err(invalid(
                    "explode faces must contain at least three vertices",
                ));
            }
        }
    }
    Ok(output)
}

fn corner_offsets(mesh: &Mesh) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(mesh.faces.len());
    let mut offset = 0;
    for face in &mesh.faces {
        offsets.push(offset);
        offset += face.vertices.len();
    }
    offsets
}

fn face_center(vertices: &[usize], positions: &[DVec3]) -> [f32; 3] {
    let mut center = [0.0_f32; 3];
    for &index in vertices {
        let point = positions[index];
        center[0] += point.x as f32;
        center[1] += point.y as f32;
        center[2] += point.z as f32;
    }
    let scale = 1.0_f32 / vertices.len() as f32;
    [center[0] * scale, center[1] * scale, center[2] * scale]
}

fn balance(nodes: &mut [KdNode], offset: usize, axis: usize) -> Option<usize> {
    if nodes.len() <= 1 {
        return (!nodes.is_empty()).then_some(offset);
    }
    let median = nodes.len() / 2;
    let mut left = 0_isize;
    let mut right = nodes.len() - 1;
    while right > left as usize {
        let pivot = nodes[right].co[axis];
        let mut i = left - 1;
        let mut j = right as isize;
        loop {
            i += 1;
            while nodes[i as usize].co[axis] < pivot {
                i += 1;
            }
            j -= 1;
            while j > left && nodes[j as usize].co[axis] > pivot {
                j -= 1;
            }
            if i >= j {
                break;
            }
            nodes.swap(i as usize, j as usize);
        }
        nodes.swap(i as usize, right);
        if i as usize >= median {
            right = i as usize - 1;
        }
        if i as usize <= median {
            left = i + 1;
        }
    }
    let (before, rest) = nodes.split_at_mut(median);
    let (middle, after) = rest.split_at_mut(1);
    middle[0].axis = axis;
    let left = balance(before, offset, (axis + 1) % 3);
    let right = balance(after, offset + median + 1, (axis + 1) % 3);
    middle[0].left = left;
    middle[0].right = right;
    Some(offset + median)
}

fn nearest(nodes: &[KdNode], root: Option<usize>, point: [f32; 3]) -> usize {
    let Some(root) = root else {
        return 0;
    };
    // `balance` produces a median-balanced tree, so its traversal depth is bounded by usize bits.
    let mut stack = [0_usize; usize::BITS as usize + 1];
    let mut stack_len = 1;
    stack[0] = root;
    let mut nearest_index = 0;
    let mut min_distance = f32::MAX;
    while stack_len > 0 {
        stack_len -= 1;
        let node = nodes[stack[stack_len]];
        let delta = node.co[node.axis] - point[node.axis];
        let plane_distance = delta * delta;
        let (near_child, far_child) = if delta < 0.0 {
            (node.right, node.left)
        } else {
            (node.left, node.right)
        };
        if plane_distance < min_distance {
            let x = node.co[0] - point[0];
            let y = node.co[1] - point[1];
            let z = node.co[2] - point[2];
            let distance = (x * x + y * y) + z * z;
            if distance < min_distance {
                min_distance = distance;
                nearest_index = node.index;
            }
            if let Some(child) = far_child {
                stack[stack_len] = child;
                stack_len += 1;
            }
        }
        if let Some(child) = near_child {
            stack[stack_len] = child;
            stack_len += 1;
        }
    }
    nearest_index
}

fn cut_edges(
    tess_faces: &[TessFace],
    face_particles: &[usize],
    positions: &[DVec3],
) -> crate::error::Result<(Vec<Piece>, Vec<DVec3>, Vec<EdgeKey>)> {
    let mut vertex_particles = vec![usize::MAX; positions.len()];
    for (index, face) in tess_faces.iter().enumerate() {
        for &vertex in &face.vertices {
            vertex_particles[vertex] = face_particles[index];
        }
    }
    let mut edge_splits = BTreeMap::<EdgeKey, usize>::new();
    let mut split_edges = Vec::<EdgeKey>::new();
    let mut masks = Vec::with_capacity(tess_faces.len());
    let mut points = Vec::new();
    for face in tess_faces {
        let mut mask = if face.vertices.len() == 3 { 16 } else { 0 };
        for edge in 0..face.vertices.len() {
            let first = face.vertices[edge];
            let second = face.vertices[(edge + 1) % face.vertices.len()];
            let split = if face.vertices.len() == 3 {
                let (a, b) = match edge {
                    0 => (0, 1),
                    1 => (1, 2),
                    _ => (0, 2),
                };
                vertex_particles[face.vertices[a]] != vertex_particles[face.vertices[b]]
            } else {
                vertex_particles[first] != vertex_particles[second]
            };
            if split {
                let key = edge_key(first, second);
                if let std::collections::btree_map::Entry::Vacant(entry) = edge_splits.entry(key) {
                    let id = positions.len() + points.len();
                    points.push((positions[key.0] + positions[key.1]) * 0.5);
                    entry.insert(id);
                    split_edges.push(key);
                }
                mask |= 1 << edge;
            }
        }
        if face.vertices.len() == 4 && mask == 15 {
            let key = edge_key(face.vertices[0], face.vertices[2]);
            if let std::collections::btree_map::Entry::Vacant(entry) = edge_splits.entry(key) {
                let id = positions.len() + points.len();
                points.push((positions[key.0] + positions[key.1]) * 0.5);
                entry.insert(id);
                split_edges.push(key);
            }
        }
        masks.push(mask);
    }
    let mut pieces = Vec::new();
    for (tess_index, face) in tess_faces.iter().enumerate() {
        let mask = masks[tess_index];
        let get_edge = |first: usize, second: usize| -> crate::error::Result<usize> {
            edge_splits
                .get(&edge_key(first, second))
                .copied()
                .ok_or_else(|| invalid("explode edge-cut midpoint is missing"))
        };
        let mut add = |vertices: Vec<usize>, particle: usize| {
            pieces.push(build_piece(tess_index, vertices, particle, None));
        };
        if face.vertices.len() == 3 {
            let original = &face.vertices;
            match mask {
                19 | 21 | 22 => {
                    let order = match mask {
                        19 => [1, 2, 0],
                        21 => [0, 1, 2],
                        _ => [2, 0, 1],
                    };
                    let [a, b, c] = order.map(|corner| original[corner]);
                    let ab = get_edge(a, b)?;
                    let ac = get_edge(a, c)?;
                    add(vec![a, ab, ac], vertex_particles[a]);
                    add(vec![ab, b, c, ac], vertex_particles[b]);
                }
                23 => {
                    let [a, b, c] = [original[0], original[1], original[2]];
                    let ab = get_edge(a, b)?;
                    let bc = get_edge(b, c)?;
                    let ac = get_edge(a, c)?;
                    add(vec![a, ab, bc, ac], vertex_particles[a]);
                    // Blender 5.2 overwrites the second split triangle's UV row with the third's, then leaves the third row copied.
                    let source_slots = vec![
                        vec![(0, 0.5), (2, 0.5)],
                        vec![(1, 0.5), (2, 0.5)],
                        vec![(2, 1.0)],
                    ];
                    pieces.push(build_piece(
                        tess_index,
                        vec![ab, b, bc],
                        vertex_particles[b],
                        Some(source_slots),
                    ));
                    let copied_slots = vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]];
                    pieces.push(build_piece(
                        tess_index,
                        vec![ac, bc, c],
                        vertex_particles[c],
                        Some(copied_slots),
                    ));
                }
                16 | 17 | 18 | 20 => add(original.clone(), vertex_particles[original[0]]),
                _ => {
                    return Err(invalid(
                        "explode triangle has an impossible edge-cut pattern",
                    ));
                }
            }
        } else {
            let original = &face.vertices;
            let order = match mask {
                0 | 3 | 10 | 11 | 15 => [0, 1, 2, 3],
                5..=7 => [1, 2, 3, 0],
                9 | 13 => [3, 0, 1, 2],
                12 | 14 => [2, 3, 0, 1],
                _ => return Err(invalid("explode quad has an impossible edge-cut pattern")),
            };
            let [a, b, c, d] = order.map(|corner| original[corner]);
            let edge = |x, y| get_edge(x, y);
            match mask {
                0 => add(original.clone(), vertex_particles[original[0]]),
                3 | 6 | 9 | 12 => {
                    let ab = edge(a, b)?;
                    let bc = edge(b, c)?;
                    add(vec![a, ab, bc, c], vertex_particles[a]);
                    add(vec![ab, b, bc], vertex_particles[b]);
                    add(vec![a, c, d], vertex_particles[a]);
                }
                5 | 10 => {
                    let bc = edge(b, c)?;
                    let ad = edge(a, d)?;
                    add(vec![a, b, bc, ad], vertex_particles[a]);
                    add(vec![ad, bc, c, d], vertex_particles[c]);
                }
                15 => {
                    let ab = edge(a, b)?;
                    let bc = edge(b, c)?;
                    let cd = edge(c, d)?;
                    let ad = edge(a, d)?;
                    let ac = edge(a, c)?;
                    add(vec![a, ab, ac, ad], vertex_particles[a]);
                    add(vec![ab, b, bc, ac], vertex_particles[b]);
                    add(vec![ac, bc, c, cd], vertex_particles[c]);
                    add(vec![ad, ac, cd, d], vertex_particles[d]);
                }
                7 | 11 | 13 | 14 => {
                    let ab = edge(a, b)?;
                    let bc = edge(b, c)?;
                    let ad = edge(a, d)?;
                    add(vec![a, ab, bc, ad], vertex_particles[a]);
                    add(vec![ab, b, bc], vertex_particles[b]);
                    add(vec![ad, bc, c, d], vertex_particles[d]);
                }
                _ => return Err(invalid("explode quad has an unsupported edge-cut pattern")),
            }
        }
    }
    Ok((pieces, points, split_edges))
}

/// A split piece of `tess_face` with corners in their own order, validated as Blender stores them.
fn build_piece(
    tess_face: usize,
    vertices: Vec<usize>,
    particle: usize,
    uv_corners: Option<Vec<Vec<(usize, f64)>>>,
) -> Piece {
    let corner_slots = (0..vertices.len()).collect();
    let mut piece = Piece {
        tess_face,
        vertices,
        particle,
        corner_slots,
        uv_corners,
    };
    validate_piece(&mut piece);
    piece
}

fn validate_piece(piece: &mut Piece) {
    if piece.vertices.len() == 3 && piece.vertices[2] == 0 {
        piece.vertices.rotate_left(1);
        piece.corner_slots.rotate_left(1);
        if let Some(uv_corners) = &mut piece.uv_corners {
            uv_corners.rotate_left(1);
        }
    } else if piece.vertices.len() == 4 && (piece.vertices[2] == 0 || piece.vertices[3] == 0) {
        piece.vertices.rotate_left(2);
        piece.corner_slots.rotate_left(2);
        if let Some(uv_corners) = &mut piece.uv_corners {
            uv_corners.rotate_left(2);
        }
    }
}

fn source_corner_weights(
    point: usize,
    tess_index: usize,
    tess_faces: &[TessFace],
    split_edges: &[EdgeKey],
    mesh: &Mesh,
    offsets: &[usize],
) -> crate::error::Result<Vec<(usize, f64)>> {
    let face = &tess_faces[tess_index];
    let offset = offsets[face.source_face];
    if point < mesh.vertices.len() {
        let corner = face
            .vertices
            .iter()
            .position(|vertex| *vertex == point)
            .ok_or_else(|| invalid("explode corner is not in its source tessface"))?;
        return Ok(vec![(offset + face.source_corners[corner], 1.0)]);
    }
    let (first, second) = *split_edges
        .get(point - mesh.vertices.len())
        .ok_or_else(|| invalid("explode split vertex has no edge provenance"))?;
    let first_corner = face
        .vertices
        .iter()
        .position(|vertex| *vertex == first)
        .ok_or_else(|| invalid("explode split edge is outside its source tessface"))?;
    let second_corner = face
        .vertices
        .iter()
        .position(|vertex| *vertex == second)
        .ok_or_else(|| invalid("explode split edge is outside its source tessface"))?;
    Ok(vec![
        (offset + face.source_corners[first_corner], 0.5),
        (offset + face.source_corners[second_corner], 0.5),
    ])
}

fn rotate_valid_face(
    vertices: &mut [u32],
    copy_corners: &mut [Vec<(usize, f64)>],
    uv_corners: &mut [Vec<(usize, f64)>],
) {
    let rotation = if vertices.len() == 3 && vertices[2] == 0 {
        1
    } else if vertices.len() == 4 && (vertices[2] == 0 || vertices[3] == 0) {
        2
    } else {
        0
    };
    vertices.rotate_left(rotation);
    copy_corners.rotate_left(rotation);
    uv_corners.rotate_left(rotation);
}

/// Explode copies tessface corner data; Blender applies midpoint interpolation only to UVs.
fn remap_uv_layers(
    source: &Mesh,
    output: &mut Mesh,
    corner_sources: &[Vec<(usize, f64)>],
) -> crate::error::Result<()> {
    let source_layers = uv_layers::read_layers(source);
    if source_layers.is_empty() {
        return Ok(());
    }
    let output_corner_count = output
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if corner_sources.len() != output_corner_count {
        return Err(invalid(
            "explode UV provenance does not match output corners",
        ));
    }
    let source_offsets = corner_offsets(source);
    let source_corner_count = source
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    let mut store = uv_layers::read(output);
    for source_layer in &source_layers {
        let target_layer = store
            .layers
            .iter_mut()
            .find(|layer| layer.name == source_layer.name)
            .ok_or_else(|| invalid("explode lost a source UV layer during provenance remap"))?;
        let mut source_values = vec![[0.0; 2]; source_corner_count];
        for (face_index, face) in source.faces.iter().enumerate() {
            let Some(Some(values)) = source_layer.faces.get(face_index) else {
                continue;
            };
            if values.len() != face.vertices.len() {
                return Err(invalid("explode source UV corner count is invalid"));
            }
            let start = source_offsets[face_index];
            source_values[start..start + values.len()].copy_from_slice(values);
        }
        let mut corner_offset = 0;
        for (face_index, face) in output.faces.iter().enumerate() {
            let corner_count = face.vertices.len();
            if target_layer.faces[face_index].is_some() {
                let mut values = Vec::with_capacity(corner_count);
                for sources in &corner_sources[corner_offset..corner_offset + corner_count] {
                    let mut uv = [0.0; 2];
                    for (source_corner, weight) in sources {
                        let value = source_values
                            .get(*source_corner)
                            .ok_or_else(|| invalid("explode UV source corner is missing"))?;
                        uv[0] += value[0] * *weight;
                        uv[1] += value[1] * *weight;
                    }
                    values.push(uv);
                }
                target_layer.faces[face_index] = Some(values);
            }
            corner_offset += corner_count;
        }
    }
    uv_layers::write(output, &store);
    Ok(())
}

/// Legacy tessfaces have an opaque-white fourth `BYTE_COLOR` slot even when the source is a triangle.
fn fill_triangle_byte_color_padding(
    source: &Mesh,
    output: &mut Mesh,
    corner_sources: &[Vec<(usize, f64)>],
) -> crate::error::Result<()> {
    let Some(data_types) = source
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    let mut store = color_layers::read(output);
    if !store.layers.iter().any(|layer| {
        layer.domain == ColorDomain::Corner
            && data_types.get(&layer.name).and_then(Value::as_str) == Some("BYTE_COLOR")
    }) {
        return Ok(());
    }
    let output_corner_count = output
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if corner_sources.len() != output_corner_count {
        return Err(invalid(
            "explode color provenance does not match output corners",
        ));
    }
    let mut offset = 0;
    for face in &output.faces {
        for corner in 0..face.vertices.len() {
            if !corner_sources[offset + corner].is_empty() {
                continue;
            }
            for layer in &mut store.layers {
                if layer.domain == ColorDomain::Corner
                    && data_types.get(&layer.name).and_then(Value::as_str) == Some("BYTE_COLOR")
                {
                    let value = layer
                        .values
                        .get_mut(offset + corner)
                        .ok_or_else(|| invalid("explode corner color slot is missing"))?;
                    *value = [1.0; 4];
                }
            }
        }
        offset += face.vertices.len();
    }
    color_layers::write(output, &store);
    Ok(())
}
fn particle_age(particle: Option<&ParticleState>, frame: f64) -> f32 {
    let mut age = particle.map_or(0.0, |particle| {
        let current = frame as f32;
        let birth = particle.birth_frame as f32;
        let death = particle.death_frame as f32;
        let lifetime = death - birth;
        (current - birth) / lifetime
    });
    age = age.clamp(0.001, 0.999);
    age
}

fn apply_particle_uv(source: &Mesh, output: &mut Mesh, requested: &str, ages: &[f32]) {
    if !uv_layers::has_uv_layers(source) || requested.is_empty() {
        return;
    }
    let mut layers = uv_layers::read_layers(output);
    if let Some(layer) = layers.iter_mut().find(|layer| layer.name == requested) {
        for (index, face) in output.faces.iter().enumerate() {
            layer.faces[index] = Some(vec![[f64::from(ages[index]), 0.5]; face.vertices.len()]);
        }
        uv_layers::write_layers(output, &layers);
    }
}

fn take_id(next: &mut u32) -> crate::error::Result<u32> {
    let id = *next;
    *next = next.checked_add(1).ok_or_else(|| {
        crate::error::PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "modifier output exhausted mesh element IDs",
        )
    })?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DVec3};
    use serde_json::{Map, json};

    use super::{TessFace, cut_edges, explode};
    use crate::{
        geom::{Face, Mesh, Vertex},
        sim::ParticleState,
    };

    fn particle(position: [f64; 3]) -> ParticleState {
        ParticleState {
            position,
            birth_position: position,
            ..ParticleState::default()
        }
    }

    #[test]
    fn alive_quad_faces_remain_quads_and_duplicate_per_particle_group() -> crate::error::Result<()>
    {
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(-3.0, 0.0, 0.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(-1.0, 0.0, 0.0),
                },
                Vertex {
                    id: 2,
                    co: DVec3::new(-1.0, 2.0, 0.0),
                },
                Vertex {
                    id: 3,
                    co: DVec3::new(-3.0, 2.0, 0.0),
                },
                Vertex {
                    id: 4,
                    co: DVec3::new(1.0, 0.0, 0.0),
                },
                Vertex {
                    id: 5,
                    co: DVec3::new(3.0, 0.0, 0.0),
                },
                Vertex {
                    id: 6,
                    co: DVec3::new(3.0, 2.0, 0.0),
                },
                Vertex {
                    id: 7,
                    co: DVec3::new(1.0, 2.0, 0.0),
                },
            ],
            faces: vec![
                Face {
                    id: 0,
                    vertices: vec![0, 1, 2, 3],
                    material_index: 0,
                },
                Face {
                    id: 1,
                    vertices: vec![4, 5, 6, 7],
                    material_index: 0,
                },
            ],
            ..Mesh::default()
        };
        let particles = [particle([-2.0, 1.0, 0.0]), particle([2.0, 1.0, 0.0])];
        let output = explode(
            &mesh,
            &particles,
            1,
            DMat4::IDENTITY,
            &Map::from_iter([("use_edge_cut".into(), json!(false))]),
            1.0,
        )?;
        assert_eq!(output.faces.len(), 2);
        assert!(output.faces.iter().all(|face| face.vertices.len() == 4));
        assert_eq!(output.vertices.len(), 8);
        Ok(())
    }

    #[test]
    fn protected_vertex_group_prevents_particle_assignment() -> crate::error::Result<()> {
        let mut mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 1,
                    co: DVec3::X,
                },
                Vertex {
                    id: 2,
                    co: DVec3::Y,
                },
            ],
            faces: vec![Face {
                id: 0,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            ..Mesh::default()
        };
        mesh.attributes.insert(
            "vertex_groups".into(),
            json!({"protect": {"v0": 1.0, "v1": 1.0, "v2": 1.0}}),
        );
        let particle = ParticleState {
            position: [5.0, 0.0, 0.0],
            birth_position: [0.0; 3],
            ..ParticleState::default()
        };
        let params = Map::from_iter([
            ("vertex_group".into(), json!("protect")),
            ("protect".into(), json!(1.0)),
        ]);
        let output = explode(&mesh, &[particle], 33, DMat4::IDENTITY, &params, 1.0)?;
        assert_eq!(output.faces.len(), 1);
        for (before, after) in mesh.vertices.iter().zip(&output.vertices) {
            assert_eq!(before.co, after.co);
        }
        Ok(())
    }

    #[test]
    fn edge_cut_triangle_uses_blender_two_edge_split_table() -> crate::error::Result<()> {
        let mut mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(0.0, 0.0, 0.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(1.0, 0.0, 0.0),
                },
                Vertex {
                    id: 2,
                    co: DVec3::new(1.0, 1.0, 0.0),
                },
                Vertex {
                    id: 3,
                    co: DVec3::new(0.0, 1.0, 0.0),
                },
            ],
            faces: vec![
                Face {
                    id: 0,
                    vertices: vec![0, 1, 2],
                    material_index: 0,
                },
                Face {
                    id: 1,
                    vertices: vec![0, 2, 3],
                    material_index: 0,
                },
            ],
            ..Mesh::default()
        };
        let source_colors = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ];
        mesh.attributes.insert(
            "FixtureCornerColor".into(),
            json!({
                "domain":"corner",
                "type":"color",
                "values":{
                    "f0":[source_colors[0],source_colors[1],source_colors[2]],
                    "f1":[source_colors[3],source_colors[4],source_colors[5]]
                }
            }),
        );
        mesh.attributes.insert(
            "color_layers".into(),
            json!({"layers":["FixtureCornerColor"],"active":"FixtureCornerColor","render":"FixtureCornerColor"}),
        );
        mesh.attributes.insert(
            "blender_metadata".into(),
            json!({"color_data_types":{"FixtureCornerColor":"BYTE_COLOR"}}),
        );
        let particles = [
            particle([2.0 / 3.0, 1.0 / 3.0, 0.0]),
            particle([1.0 / 3.0, 2.0 / 3.0, 0.0]),
        ];
        let params = Map::from_iter([("use_edge_cut".into(), json!(true))]);
        let output = explode(&mesh, &particles, 9, DMat4::IDENTITY, &params, 1.0)?;
        assert_eq!(output.faces.len(), 3);
        assert_eq!(
            output
                .faces
                .iter()
                .map(|face| face.vertices.len())
                .collect::<Vec<_>>(),
            vec![3, 4, 3]
        );
        assert_eq!(output.vertices.len(), 8);
        let colors = output.attributes["FixtureCornerColor"]["values"]
            .as_object()
            .ok_or_else(|| crate::error::PotError::invalid_argument("Explode corner color rows"))?;
        for face in &output.faces {
            let row = colors
                .get(&format!("f{}", face.id))
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    crate::error::PotError::invalid_argument("Explode face color row")
                })?;
            let mut white_slots = 0;
            for value in row {
                let color: [f64; 4] = serde_json::from_value(value.clone())
                    .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
                if color == [1.0; 4] {
                    white_slots += 1;
                } else {
                    assert!(
                        source_colors.contains(&color),
                        "Explode must copy tessface color slots, not interpolate split corners: {color:?}",
                    );
                }
            }
            assert_eq!(white_slots, usize::from(face.vertices.len() == 4));
        }
        Ok(())
    }
    #[test]
    fn fully_split_triangle_uses_blender_uv_overwrite_table() -> crate::error::Result<()> {
        let tess_faces = vec![
            TessFace {
                source_face: 0,
                vertices: vec![0, 1, 2],
                source_corners: vec![0, 1, 2],
            },
            TessFace {
                source_face: 1,
                vertices: vec![0, 3, 4],
                source_corners: vec![0, 1, 2],
            },
            TessFace {
                source_face: 2,
                vertices: vec![1, 5, 6],
                source_corners: vec![0, 1, 2],
            },
            TessFace {
                source_face: 3,
                vertices: vec![2, 7, 8],
                source_corners: vec![0, 1, 2],
            },
        ];
        let positions = (0..9)
            .map(|index| DVec3::new(index as f64, index as f64 * 0.2, 0.0))
            .collect::<Vec<_>>();
        let face_particles = [3, 0, 1, 2];
        let (pieces, _, _) = cut_edges(&tess_faces, &face_particles, &positions)?;
        let central: Vec<_> = pieces.iter().filter(|piece| piece.tess_face == 0).collect();
        assert_eq!(central.len(), 3);
        assert_eq!(central[0].vertices.len(), 4);
        let overwritten_uvs = vec![
            vec![(0, 0.5), (2, 0.5)],
            vec![(1, 0.5), (2, 0.5)],
            vec![(2, 1.0)],
        ];
        assert_eq!(central[1].uv_corners.as_ref(), Some(&overwritten_uvs));
        let copied_uvs = vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]];
        assert_eq!(central[2].uv_corners.as_ref(), Some(&copied_uvs));
        Ok(())
    }
    #[test]
    fn matches_blender_522_edge_cut_fragment_oracle() -> crate::error::Result<()> {
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::new(0.0, 0.0, 0.0),
                },
                Vertex {
                    id: 1,
                    co: DVec3::new(2.0, 0.0, 0.0),
                },
                Vertex {
                    id: 2,
                    co: DVec3::new(0.0, 2.0, 0.0),
                },
                Vertex {
                    id: 3,
                    co: DVec3::new(2.0, 2.0, 0.0),
                },
            ],
            faces: vec![
                Face {
                    id: 0,
                    vertices: vec![0, 1, 2],
                    material_index: 0,
                },
                Face {
                    id: 1,
                    vertices: vec![1, 3, 2],
                    material_index: 0,
                },
            ],
            ..Mesh::default()
        };
        let particles = [
            particle([0.0, 2.0, 0.0]),
            particle([0.0, 0.0, 0.0]),
            particle([2.0, 0.0, 0.0]),
            particle([0.0, 0.0, 0.0]),
        ];
        let params = Map::from_iter([("use_edge_cut".into(), json!(true))]);
        let output = explode(&mesh, &particles, 11, DMat4::IDENTITY, &params, 2.0)?;
        let expected_vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let expected_faces = [vec![0, 1, 2], vec![3, 4, 5, 6], vec![4, 7, 5]];
        assert_eq!(output.vertices.len(), expected_vertices.len());
        for (vertex, expected) in output.vertices.iter().zip(expected_vertices) {
            // Derived vertex-path bound: 221 ulp(1) first-order worst case for the operations in
            // MOD_explode.cc's edge-cut path (see tests/modifier_param_audit.rs). It excludes the
            // particle quaternion's own f32 construction error, which is not yet counted. The 1e-12
            // is Potter's f64 error.
            let bound = 221.0 * f64::from(f32::EPSILON) + 1.0e-12;
            let error = (vertex.co - DVec3::from_array(expected))
                .abs()
                .max_element();
            assert!(
                error <= bound,
                "vertex {:?} differs from {expected:?} by {error:e}, bound {bound:e}",
                vertex.co
            );
        }
        assert_eq!(output.faces.len(), expected_faces.len());
        for (face, expected) in output.faces.iter().zip(expected_faces) {
            assert_eq!(face.vertices, expected);
        }
        Ok(())
    }
}
