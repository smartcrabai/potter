// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_decimate_collapse.cc, source/blender/bmesh/tools/bmesh_decimate_dissolve.cc, source/blender/bmesh/tools/bmesh_decimate_unsubdivide.cc, source/blender/bmesh/intern/bmesh_polygon.cc, source/blender/bmesh/operators/bmo_dissolve.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_decimate.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{DVec3, Vec3};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::Modifier,
};

use super::{
    Mesh, Polygon, bool_param, clean_polygon, cyclic_pairs, decimate_compact, edge_key, id_indices,
    invalid_parameter, loose_edges, number_param, polygons, positions, unique_polygons,
};
#[derive(Clone, Copy, Default)]
struct DecimateQuadric([f64; 10]);

impl DecimateQuadric {
    fn add_plane(&mut self, normal: DVec3, distance: f64, weight: f64) {
        let [a, b, c] = normal.to_array();
        let d = distance;
        let plane = [
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ];
        for (value, coefficient) in self.0.iter_mut().zip(plane) {
            *value += coefficient * weight;
        }
    }

    fn add(self, other: Self) -> Self {
        let mut result = self;
        for (value, other_value) in result.0.iter_mut().zip(other.0) {
            *value += other_value;
        }
        result
    }

    fn evaluate(self, point: DVec3) -> f64 {
        let [x, y, z] = point.to_array();
        let q = self.0;
        ((q[0] * (x * x))
            + (q[1] * 2.0 * (x * y))
            + (q[2] * 2.0 * (x * z))
            + (q[3] * 2.0 * x)
            + (q[4] * (y * y))
            + (q[5] * 2.0 * (y * z))
            + (q[6] * 2.0 * y)
            + (q[7] * (z * z))
            + (q[8] * 2.0 * z))
            + q[9]
    }

    fn optimize(self) -> Option<DVec3> {
        let q = self.0;
        let determinant = q[0] * (q[4] * q[7] - q[5] * q[5]) - q[1] * (q[1] * q[7] - q[2] * q[5])
            + q[2] * (q[1] * q[5] - q[2] * q[4]);
        if determinant.abs() <= 1.0e-8 {
            return None;
        }
        let inverse_determinant = 1.0 / determinant;
        let inverse = [
            [
                (q[4] * q[7] - q[5] * q[5]) * inverse_determinant,
                (q[2] * q[5] - q[1] * q[7]) * inverse_determinant,
                (q[1] * q[5] - q[4] * q[2]) * inverse_determinant,
            ],
            [
                (q[2] * q[5] - q[1] * q[7]) * inverse_determinant,
                (q[0] * q[7] - q[2] * q[2]) * inverse_determinant,
                (q[1] * q[2] - q[0] * q[5]) * inverse_determinant,
            ],
            [
                (q[1] * q[5] - q[2] * q[4]) * inverse_determinant,
                (q[1] * q[2] - q[0] * q[5]) * inverse_determinant,
                (q[0] * q[4] - q[1] * q[1]) * inverse_determinant,
            ],
        ];
        let result = DVec3::new(
            -(inverse[0][0] * q[3] + inverse[0][1] * q[6] + inverse[0][2] * q[8]),
            -(inverse[1][0] * q[3] + inverse[1][1] * q[6] + inverse[1][2] * q[8]),
            -(inverse[2][0] * q[3] + inverse[2][1] * q[6] + inverse[2][2] * q[8]),
        );
        result.is_finite().then_some(result)
    }
}

pub(super) fn decimate_face_normal(points: &[DVec3], face: &[usize]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for (first, second) in cyclic_pairs(face) {
        normal += points[*first].cross(points[*second]);
    }
    normal.normalize_or_zero()
}

#[derive(Clone)]
struct CollapseFace {
    vertices: [usize; 3],
    edges: [usize; 3],
    material: u32,
    origin: usize,
    bmesh_order: usize,
    // BMesh retains the source polygon normal on the original face after triangulation.
    quadric_normal: Option<Vec3>,
}

#[derive(Clone, Copy)]
struct CollapseEdge {
    vertices: [usize; 2],
    active: bool,
    cut_origin: Option<usize>,
    input_loose: bool,
}

struct CollapseMesh {
    points: Vec<Vec3>,
    faces: Vec<CollapseFace>,
    edges: Vec<CollapseEdge>,
    vertex_edges: Vec<Vec<usize>>,
    edge_lookup: BTreeMap<(usize, usize), usize>,
}

fn ensure_collapse_edge(
    edges: &mut Vec<CollapseEdge>,
    lookup: &mut BTreeMap<(usize, usize), usize>,
    vertices: [usize; 2],
    cut_origin: Option<usize>,
) -> usize {
    let key = edge_key(vertices[0], vertices[1]);
    if let Some(index) = lookup.get(&key).copied() {
        if cut_origin.is_some() && edges[index].cut_origin.is_none() {
            edges[index].cut_origin = cut_origin;
        }
        return index;
    }
    let index = edges.len();
    edges.push(CollapseEdge {
        vertices,
        active: true,
        cut_origin,
        input_loose: false,
    });
    lookup.insert(key, index);
    index
}

fn collapse_quad_split_02(points: &[Vec3], vertices: [usize; 4]) -> bool {
    let [v1, v2, v3, v4] = [
        points[vertices[1]],
        points[vertices[2]],
        points[vertices[3]],
        points[vertices[0]],
    ];
    let cross = |first: Vec3, second: Vec3, third: Vec3| (second - first).cross(third - first);
    let normal_a = cross(v2, v3, v4);
    let normal_b = cross(v2, v4, v1);
    let mut normal = normal_a + normal_b;
    if normal.length_squared() <= f32::MIN_POSITIVE {
        return true;
    }
    normal = normal.normalize_or_zero();
    let normal_xy_length_squared = normal.x * normal.x + normal.y * normal.y;
    let (basis_x, basis_y) = if normal_xy_length_squared > f32::EPSILON {
        let inverse_length = normal_xy_length_squared.sqrt().recip();
        let basis_x = Vec3::new(normal.y * inverse_length, -normal.x * inverse_length, 0.0);
        let basis_y = Vec3::new(
            -normal.z * basis_x.y,
            normal.z * basis_x.x,
            normal.x * basis_x.y - normal.y * basis_x.x,
        );
        (basis_x, basis_y)
    } else {
        (
            Vec3::new(if normal.z < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
            Vec3::Y,
        )
    };
    let project = |point: Vec3| glam::Vec2::new(basis_x.dot(point), basis_y.dot(point));
    let [v1, v2, v3, v4] = [v1, v2, v3, v4].map(project);
    let cross2 = |first: glam::Vec2, second: glam::Vec2, third: glam::Vec2| {
        (second - first).perp_dot(third - first)
    };
    let area_234 = cross2(v2, v3, v4);
    let area_241 = cross2(v2, v4, v1);
    let area_123 = cross2(v1, v2, v3);
    let area_134 = cross2(v1, v3, v4);
    let epsilon = 1.0e-12_f32;
    if ((area_123 >= 0.0) != (area_134 >= 0.0))
        || area_123.abs() <= epsilon
        || area_134.abs() <= epsilon
    {
        return true;
    }
    if (area_234 >= 0.0) != (area_241 >= 0.0)
        || area_234.abs() <= epsilon
        || area_241.abs() <= epsilon
    {
        return false;
    }
    let length = |first: glam::Vec2, second: glam::Vec2| (second - first).length();
    let fac_24 = area_234.abs() / (length(v2, v3) + length(v3, v4) + length(v2, v4))
        + area_241.abs() / (length(v4, v1) + length(v1, v2) + length(v2, v4));
    let fac_13 = area_123.abs() / (length(v1, v2) + length(v2, v3) + length(v1, v3))
        + area_134.abs() / (length(v3, v4) + length(v4, v1) + length(v1, v3));
    fac_24 - fac_13 > 0.0
}

fn collapse_mesh_from_mesh(mesh: &Mesh) -> Result<CollapseMesh> {
    let vertex_indices = id_indices(mesh);
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect::<Vec<_>>();
    let mut edges = Vec::with_capacity(mesh.edges.len() + mesh.faces.len());
    let mut edge_lookup = BTreeMap::new();
    for edge in &mesh.edges {
        let vertices = [
            *vertex_indices.get(&edge.vertices[0]).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "decimate edge references a missing vertex",
                )
            })?,
            *vertex_indices.get(&edge.vertices[1]).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "decimate edge references a missing vertex",
                )
            })?,
        ];
        let index = edges.len();
        edges.push(CollapseEdge {
            vertices,
            active: true,
            cut_origin: None,
            input_loose: true,
        });
        edge_lookup.insert(edge_key(vertices[0], vertices[1]), index);
    }

    let mut source_edges = BTreeSet::new();
    // The original face's normal remains cached on the retained triangle in Blender's BMesh.
    let mut original_face_normals = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let original_normal = match face.vertices.len() {
            4 => {
                let vertices = [
                    *vertex_indices.get(&face.vertices[0]).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?,
                    *vertex_indices.get(&face.vertices[1]).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?,
                    *vertex_indices.get(&face.vertices[2]).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?,
                    *vertex_indices.get(&face.vertices[3]).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?,
                ];
                Some(
                    (points[vertices[0]] - points[vertices[2]])
                        .cross(points[vertices[1]] - points[vertices[3]])
                        .normalize_or_zero(),
                )
            }
            5.. => {
                let mut normal = Vec3::ZERO;
                for (first, second) in cyclic_pairs(&face.vertices) {
                    let first = vertex_indices.get(first).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?;
                    let second = vertex_indices.get(second).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "decimate face references a missing vertex",
                        )
                    })?;
                    normal += points[first].cross(points[second]);
                }
                Some(normal.normalize_or_zero())
            }
            _ => None,
        };
        original_face_normals.push(original_normal);
        for (first, second) in cyclic_pairs(&face.vertices) {
            let first = vertex_indices.get(first).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "decimate face references a missing vertex",
                )
            })?;
            let second = vertex_indices.get(second).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "decimate face references a missing vertex",
                )
            })?;
            source_edges.insert(edge_key(first, second));
        }
    }
    for edge in &mut edges {
        edge.input_loose = !source_edges.contains(&edge_key(edge.vertices[0], edge.vertices[1]));
    }

    let triangles = mesh
        .triangulate()
        .map_err(|error| super::mesh_error(&error))?;
    let mut face_ranges = Vec::with_capacity(mesh.faces.len());
    let mut faces = Vec::with_capacity(triangles.len());
    let mut triangle_offset = 0;
    for (origin, face) in mesh.faces.iter().enumerate() {
        let first_face = faces.len();
        let triangle_count = face.vertices.len().saturating_sub(2);
        for local_index in 0..triangle_count {
            let vertices = if face.vertices.len() == 4 {
                let quad = [
                    face.vertices[0],
                    face.vertices[1],
                    face.vertices[2],
                    face.vertices[3],
                ]
                .map(|id| vertex_indices.get(&id).copied().unwrap_or(usize::MAX));
                if collapse_quad_split_02(&points, quad) {
                    if local_index == 0 {
                        [quad[0], quad[1], quad[2]]
                    } else {
                        [quad[0], quad[2], quad[3]]
                    }
                } else if local_index == 0 {
                    [quad[1], quad[2], quad[3]]
                } else {
                    [quad[1], quad[3], quad[0]]
                }
            } else {
                let Some(triangle) = triangles.get(triangle_offset + local_index) else {
                    return Err(PotError::new(
                        ErrorCode::InvalidArgument,
                        "decimate triangulation omitted a face triangle",
                    ));
                };
                triangle.map(|id| vertex_indices.get(&id).copied().unwrap_or(usize::MAX))
            };
            if vertices.contains(&usize::MAX) {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "decimate triangulation references a missing vertex",
                ));
            }
            let edge_ids = [
                ensure_collapse_edge(
                    &mut edges,
                    &mut edge_lookup,
                    [vertices[0], vertices[1]],
                    None,
                ),
                ensure_collapse_edge(
                    &mut edges,
                    &mut edge_lookup,
                    [vertices[1], vertices[2]],
                    None,
                ),
                ensure_collapse_edge(
                    &mut edges,
                    &mut edge_lookup,
                    [vertices[2], vertices[0]],
                    None,
                ),
            ];
            for edge_index in edge_ids {
                if !source_edges.contains(&edge_key(
                    edges[edge_index].vertices[0],
                    edges[edge_index].vertices[1],
                )) {
                    edges[edge_index].cut_origin = Some(origin);
                }
            }
            faces.push(CollapseFace {
                vertices,
                edges: edge_ids,
                material: face.material_index,
                origin,
                bmesh_order: 0,
                quadric_normal: (local_index + 1 == triangle_count)
                    .then_some(original_face_normals[origin])
                    .flatten(),
            });
        }
        face_ranges.push((first_face, triangle_count));
        triangle_offset += triangle_count;
    }
    let mut bmesh_order = 0;
    for (first, count) in &face_ranges {
        faces[first + count - 1].bmesh_order = bmesh_order;
        bmesh_order += 1;
    }
    for (first, count) in &face_ranges {
        let end = first + count - 1;
        for face in faces.iter_mut().take(end).skip(*first) {
            face.bmesh_order = bmesh_order;
            bmesh_order += 1;
        }
    }
    let mut vertex_edges = vec![Vec::new(); points.len()];
    for (edge_index, edge) in edges.iter().enumerate() {
        vertex_edges[edge.vertices[0]].push(edge_index);
        vertex_edges[edge.vertices[1]].push(edge_index);
    }
    Ok(CollapseMesh {
        points,
        faces,
        edges,
        vertex_edges,
        edge_lookup,
    })
}

fn collapse_edge_face_links(mesh: &CollapseMesh) -> (Vec<[usize; 2]>, Vec<u8>) {
    let mut links = vec![[usize::MAX; 2]; mesh.edges.len()];
    let mut counts = vec![0_u8; mesh.edges.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for edge in face.edges {
            let slot = usize::from(counts[edge].min(2));
            if slot < 2 {
                links[edge][slot] = face_index;
            }
            counts[edge] = counts[edge].saturating_add(1);
        }
    }
    (links, counts)
}

fn collapse_triangle_normal(points: &[Vec3], vertices: [usize; 3]) -> Vec3 {
    (points[vertices[0]] - points[vertices[1]])
        .cross(points[vertices[1]] - points[vertices[2]])
        .normalize_or_zero()
}
fn collapse_midpoint_db(points: &[Vec3], first: usize, second: usize) -> DVec3 {
    (points[first].as_dvec3() + points[second].as_dvec3()) * 0.5
}

fn collapse_quadrics(mesh: &CollapseMesh) -> Vec<DecimateQuadric> {
    let mut quadrics = vec![DecimateQuadric::default(); mesh.points.len()];
    let mut face_order = (0..mesh.faces.len()).collect::<Vec<_>>();
    face_order.sort_unstable_by_key(|index| mesh.faces[*index].bmesh_order);
    for face_index in face_order {
        let face = &mesh.faces[face_index];
        let normal = face
            .quadric_normal
            .unwrap_or_else(|| collapse_triangle_normal(&mesh.points, face.vertices));
        if normal.length_squared() <= f32::MIN_POSITIVE {
            continue;
        }
        let center = (mesh.points[face.vertices[0]]
            + mesh.points[face.vertices[1]]
            + mesh.points[face.vertices[2]])
            / 3.0;
        let normal_db = normal.as_dvec3();
        let distance = -normal_db.dot(center.as_dvec3());
        let mut quadric = DecimateQuadric::default();
        quadric.add_plane(normal_db, distance, 1.0);
        for vertex in face.vertices {
            quadrics[vertex] = quadrics[vertex].add(quadric);
        }
    }
    let (links, counts) = collapse_edge_face_links(mesh);
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        if !edge.active || counts[edge_index] != 1 {
            continue;
        }
        let face = &mesh.faces[links[edge_index][0]];
        let edge_vector = mesh.points[edge.vertices[1]] - mesh.points[edge.vertices[0]];
        let normal = face
            .quadric_normal
            .unwrap_or_else(|| collapse_triangle_normal(&mesh.points, face.vertices));
        let boundary = edge_vector.cross(normal).as_dvec3();
        let length = boundary.length();
        if length <= f64::from(f32::EPSILON) {
            continue;
        }
        let boundary_normal = boundary / length;
        let center =
            ((mesh.points[edge.vertices[0]] + mesh.points[edge.vertices[1]]) * 0.5).as_dvec3();
        let distance = -boundary_normal.dot(center);
        let mut quadric = DecimateQuadric::default();
        quadric.add_plane(boundary_normal, distance, 100.0);
        for vertex in edge.vertices {
            quadrics[vertex] = quadrics[vertex].add(quadric);
        }
    }
    quadrics
}

fn collapse_safe_acos_approx(value: f32) -> f32 {
    let absolute = value.abs();
    let clamped = if absolute < 1.0 {
        1.0 - (1.0 - absolute)
    } else {
        1.0
    };
    let angle = (1.0 - clamped).sqrt()
        * (1.570_796_4
            + clamped * (-0.213_301 + clamped * (0.077_980_48 + clamped * -0.021_640_95)));
    if value < 0.0 {
        std::f32::consts::PI - angle
    } else {
        angle
    }
}

fn collapse_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let vertex_indices = id_indices(mesh);
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect::<Vec<_>>();
    let mut normals = vec![Vec3::ZERO; points.len()];
    for face in &mesh.faces {
        let Some(vertices) = face
            .vertices
            .iter()
            .map(|id| vertex_indices.get(id).copied())
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(last) = vertices.last().copied() else {
            continue;
        };
        let face_normal = if vertices.len() == 4 {
            (points[vertices[0]] - points[vertices[2]])
                .cross(points[vertices[1]] - points[vertices[3]])
                .normalize_or_zero()
        } else {
            let mut normal = points[last].cross(points[vertices[0]]);
            for corner in 0..vertices.len() - 1 {
                normal += points[vertices[corner]].cross(points[vertices[corner + 1]]);
            }
            normal.normalize_or_zero()
        };
        for corner in 0..vertices.len() {
            let vertex = vertices[corner];
            let previous = points[vertices[(corner + vertices.len() - 1) % vertices.len()]];
            let next = points[vertices[(corner + 1) % vertices.len()]];
            let to_previous = (points[vertex] - previous).normalize_or_zero();
            let to_next = (next - points[vertex]).normalize_or_zero();
            let angle = collapse_safe_acos_approx(-to_previous.dot(to_next));
            normals[vertex] += face_normal * angle;
        }
    }
    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }
    normals
}

fn collapse_edge_is_eligible(
    mesh: &CollapseMesh,
    edge_index: usize,
    links: &[[usize; 2]],
    counts: &[u8],
    weights: Option<&[f32]>,
) -> bool {
    let edge = mesh.edges[edge_index];
    if !edge.active
        || (counts[edge_index] != 1 && counts[edge_index] != 2)
        || weights.is_some_and(|values| {
            values[edge.vertices[0]] == 0.0 || values[edge.vertices[1]] == 0.0
        })
    {
        return false;
    }
    links[edge_index]
        .iter()
        .take(usize::from(counts[edge_index]))
        .all(|face| mesh.faces[*face].vertices.len() == 3)
}

fn collapse_edge_cost(
    mesh: &CollapseMesh,
    edge_index: usize,
    links: &[[usize; 2]],
    counts: &[u8],
    quadrics: &[DecimateQuadric],
    normals: &[Vec3],
    weights: Option<&[f32]>,
    weight_factor: f32,
) -> Option<f32> {
    if !collapse_edge_is_eligible(mesh, edge_index, links, counts, weights) {
        return None;
    }
    let edge = mesh.edges[edge_index];
    let first = edge.vertices[0];
    let second = edge.vertices[1];
    let quadric = quadrics[first].add(quadrics[second]);
    let optimized = quadric
        .optimize()
        .unwrap_or_else(|| collapse_midpoint_db(&mesh.points, first, second));
    let mut cost =
        (quadrics[first].evaluate(optimized) + quadrics[second].evaluate(optimized)) as f32;
    cost = cost.abs();
    let edge_vector = mesh.points[second] - mesh.points[first];
    if cost < 1.0e-12_f32 {
        let denominator = if weights.is_some() {
            (-edge_vector.length()).min(-f32::EPSILON)
        } else {
            (-edge_vector.length_squared()).min(-f32::EPSILON)
        };
        cost = normals[first].dot(normals[second]).abs() / denominator - cost;
        if let Some(weights) = weights {
            let edge_weight = weights[first] + weights[second];
            if edge_weight != 0.0 {
                cost *= 1.0 + edge_weight * weight_factor;
            }
        }
    } else if let Some(weights) = weights {
        let edge_weight = 2.0 - (weights[first] + weights[second]);
        if edge_weight != 0.0 {
            cost += edge_vector.length() * (edge_weight * weight_factor);
        }
    }
    Some(cost)
}

fn collapse_is_topologically_degenerate(
    mesh: &CollapseMesh,
    edge_index: usize,
    links: &[[usize; 2]],
    counts: &[u8],
) -> bool {
    let edge = mesh.edges[edge_index];
    let face_count = counts[edge_index];
    if (face_count != 1 && face_count != 2)
        || links[edge_index]
            .iter()
            .take(usize::from(face_count))
            .any(|face| mesh.faces[*face].vertices.len() != 3)
    {
        return true;
    }
    let [first, second] = edge.vertices;
    let mut opposite_vertices = [usize::MAX; 2];
    for (slot, face_index) in links[edge_index]
        .iter()
        .take(usize::from(face_count))
        .enumerate()
    {
        let Some(opposite) = mesh.faces[*face_index]
            .vertices
            .iter()
            .copied()
            .find(|vertex| *vertex != first && *vertex != second)
        else {
            return true;
        };
        if opposite_vertices[..slot].contains(&opposite) {
            return true;
        }
        opposite_vertices[slot] = opposite;
    }
    let mut first_neighbors = vec![false; mesh.points.len()];
    let mut second_neighbors = vec![false; mesh.points.len()];
    for (index, candidate) in mesh.edges.iter().enumerate() {
        if !candidate.active {
            continue;
        }
        if (candidate.vertices.contains(&first) || candidate.vertices.contains(&second))
            && (counts[index] == 0 || counts[index] > 2)
        {
            return true;
        }
        if candidate.vertices.contains(&first) {
            let neighbor = if candidate.vertices[0] == first {
                candidate.vertices[1]
            } else {
                candidate.vertices[0]
            };
            if neighbor != second {
                first_neighbors[neighbor] = true;
            }
        }
        if candidate.vertices.contains(&second) {
            let neighbor = if candidate.vertices[0] == second {
                candidate.vertices[1]
            } else {
                candidate.vertices[0]
            };
            if neighbor != first {
                second_neighbors[neighbor] = true;
            }
        }
    }
    let mut common = 0;
    for (neighbor, (first, second)) in first_neighbors.iter().zip(second_neighbors).enumerate() {
        if *first && second {
            if !opposite_vertices[..usize::from(face_count)].contains(&neighbor) {
                return true;
            }
            common += 1;
        }
    }
    common != usize::from(face_count)
}

fn collapse_would_flip(mesh: &CollapseMesh, edge_index: usize, position: Vec3) -> bool {
    for vertex in mesh.edges[edge_index].vertices {
        for face in &mesh.faces {
            if face.edges.contains(&edge_index) {
                continue;
            }
            let Some(corner) = face.vertices.iter().position(|current| *current == vertex) else {
                continue;
            };
            let previous = mesh.points[face.vertices[(corner + 2) % 3]];
            let next = mesh.points[face.vertices[(corner + 1) % 3]];
            let other = previous - next;
            let before = other.cross(previous - mesh.points[vertex]);
            let after = other.cross(previous - position);
            if before.dot(after) <= (before.length_squared() + after.length_squared()) * 0.01_f32 {
                return true;
            }
        }
    }
    false
}

fn collapse_reflect(point: Vec3, axis: usize) -> Vec3 {
    let mut reflected = point;
    reflected[axis] = -reflected[axis];
    reflected
}

#[derive(Clone, Copy)]
struct CollapseKdNode {
    edge_index: usize,
    axis: usize,
    center: Vec3,
    left: Option<usize>,
    right: Option<usize>,
}

fn collapse_edge_midpoint(mesh: &CollapseMesh, edge_index: usize) -> Vec3 {
    let edge = mesh.edges[edge_index];
    (mesh.points[edge.vertices[0]] + mesh.points[edge.vertices[1]]) * 0.5
}

fn collapse_kd_build(
    mesh: &CollapseMesh,
    indices: &mut [usize],
    depth: usize,
    nodes: &mut Vec<CollapseKdNode>,
) -> Option<usize> {
    if indices.is_empty() {
        return None;
    }
    let axis = depth % 3;
    indices.sort_unstable_by(|first, second| {
        collapse_edge_midpoint(mesh, *first)[axis]
            .total_cmp(&collapse_edge_midpoint(mesh, *second)[axis])
            .then_with(|| first.cmp(second))
    });
    let middle = indices.len() / 2;
    let edge_index = indices[middle];
    let left = collapse_kd_build(mesh, &mut indices[..middle], depth + 1, nodes);
    let right = collapse_kd_build(mesh, &mut indices[middle + 1..], depth + 1, nodes);
    let node_index = nodes.len();
    nodes.push(CollapseKdNode {
        edge_index,
        axis,
        center: collapse_edge_midpoint(mesh, edge_index),
        left,
        right,
    });
    Some(node_index)
}

fn collapse_edge_mirror_matches(
    mesh: &CollapseMesh,
    first_index: usize,
    second_index: usize,
    axis: usize,
) -> bool {
    let first = mesh.edges[first_index];
    let second = mesh.edges[second_index];
    let reflected_first = collapse_reflect(mesh.points[first.vertices[0]], axis);
    let reflected_second = collapse_reflect(mesh.points[first.vertices[1]], axis);
    let reflected_midpoint = (reflected_first + reflected_second) * 0.5;
    if collapse_edge_midpoint(mesh, second_index).distance_squared(reflected_midpoint) > 4.0e-10_f32
    {
        return false;
    }
    let reflected_direction = reflected_second - reflected_first;
    let direction = mesh.points[second.vertices[1]] - mesh.points[second.vertices[0]];
    if direction.dot(reflected_direction) > 0.0 {
        mesh.points[second.vertices[0]].distance_squared(reflected_first) <= 4.0e-10_f32
            && mesh.points[second.vertices[1]].distance_squared(reflected_second) <= 4.0e-10_f32
    } else {
        mesh.points[second.vertices[1]].distance_squared(reflected_first) <= 4.0e-10_f32
            && mesh.points[second.vertices[0]].distance_squared(reflected_second) <= 4.0e-10_f32
    }
}

fn collapse_kd_find(
    mesh: &CollapseMesh,
    nodes: &[CollapseKdNode],
    node_index: Option<usize>,
    query: Vec3,
    first_edge: usize,
    axis: usize,
    result: &mut Option<usize>,
) {
    let Some(node_index) = node_index else {
        return;
    };
    let node = nodes[node_index];
    if node.center.distance_squared(query) <= 4.0e-10_f32
        && collapse_edge_mirror_matches(mesh, first_edge, node.edge_index, axis)
    {
        *result = Some(node.edge_index);
        return;
    }
    let distance = query[node.axis] - node.center[node.axis];
    let (near, far) = if distance <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };
    collapse_kd_find(mesh, nodes, near, query, first_edge, axis, result);
    if result.is_none() && distance * distance <= 4.0e-10_f32 {
        collapse_kd_find(mesh, nodes, far, query, first_edge, axis, result);
    }
}

fn collapse_symmetry_map(mesh: &CollapseMesh, axis: usize) -> Vec<Option<usize>> {
    let mut mapping = vec![None; mesh.edges.len()];
    let mut indices = mesh
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| edge.active.then_some(index))
        .collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(indices.len());
    let root = collapse_kd_build(mesh, &mut indices, 0, &mut nodes);
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        if !edge.active || mapping[edge_index].is_some() {
            continue;
        }
        let reflected_midpoint = collapse_reflect(collapse_edge_midpoint(mesh, edge_index), axis);
        let mut mirror = None;
        collapse_kd_find(
            mesh,
            &nodes,
            root,
            reflected_midpoint,
            edge_index,
            axis,
            &mut mirror,
        );
        if let Some(mirror_index) = mirror {
            mapping[edge_index] = Some(mirror_index);
            mapping[mirror_index] = Some(edge_index);
        }
    }
    mapping
}

fn collapse_line_factor(position: Vec3, first: Vec3, second: Vec3) -> f32 {
    let direction = second - first;
    let length_squared = direction.length_squared();
    if length_squared <= f32::EPSILON {
        0.5
    } else {
        (position - first).dot(direction) / length_squared
    }
}

fn collapse_apply(
    mesh: &mut CollapseMesh,
    edge_index: usize,
    position: Vec3,
    quadrics: &mut [DecimateQuadric],
    normals: &mut [Vec3],
    weights: &mut Option<Vec<f32>>,
) {
    let [keep, clear] = mesh.edges[edge_index].vertices;
    let keep_edges = mesh.vertex_edges[keep].clone();
    let clear_edges = mesh.vertex_edges[clear].clone();
    let (links, counts) = collapse_edge_face_links(mesh);
    let remove_faces = links[edge_index]
        .iter()
        .take(usize::from(counts[edge_index]))
        .copied()
        .collect::<BTreeSet<_>>();
    let factor = collapse_line_factor(position, mesh.points[keep], mesh.points[clear]);
    if let Some(values) = weights {
        values[keep] = (values[keep] * (1.0 - factor) + values[clear] * factor).clamp(0.0, 1.0);
    }
    mesh.points[keep] = position;
    normals[keep] = (normals[keep] * (1.0 - factor) + normals[clear] * factor).normalize_or_zero();
    quadrics[keep] = quadrics[keep].add(quadrics[clear]);

    mesh.faces = mesh
        .faces
        .drain(..)
        .enumerate()
        .filter_map(|(index, mut face)| {
            if remove_faces.contains(&index) {
                return None;
            }
            for vertex in &mut face.vertices {
                if *vertex == clear {
                    *vertex = keep;
                }
            }
            Some(face)
        })
        .collect();

    let mut retained = BTreeMap::new();
    for index in &keep_edges {
        let edge = mesh.edges[*index];
        if edge.active && !edge.vertices.contains(&clear) {
            retained.insert(edge_key(edge.vertices[0], edge.vertices[1]), *index);
        }
    }
    let mut moved_edges = Vec::new();
    let mut removed_edges = Vec::new();
    for index in clear_edges {
        if !mesh.edges[index].active {
            continue;
        }
        let edge = mesh.edges[index];
        let other = if edge.vertices[0] == clear {
            edge.vertices[1]
        } else {
            edge.vertices[0]
        };
        if other == keep || index == edge_index {
            mesh.edges[index].active = false;
            removed_edges.push(index);
            continue;
        }
        let vertices = [
            if edge.vertices[0] == clear {
                keep
            } else {
                edge.vertices[0]
            },
            if edge.vertices[1] == clear {
                keep
            } else {
                edge.vertices[1]
            },
        ];
        let key = edge_key(vertices[0], vertices[1]);
        match retained.entry(key) {
            std::collections::btree_map::Entry::Occupied(_) => {
                mesh.edges[index].active = false;
                removed_edges.push(index);
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                mesh.edges[index].vertices = vertices;
                entry.insert(index);
                moved_edges.push(index);
            }
        }
    }
    mesh.edges[edge_index].active = false;
    for index in removed_edges {
        let edge = mesh.edges[index];
        for vertex in edge.vertices {
            if vertex != keep && vertex != clear {
                mesh.vertex_edges[vertex].retain(|candidate| *candidate != index);
            }
        }
    }
    let mut keep_disk = keep_edges
        .into_iter()
        .filter(|index| mesh.edges[*index].active)
        .collect::<Vec<_>>();
    keep_disk.extend(moved_edges);
    mesh.vertex_edges[keep] = keep_disk;
    mesh.vertex_edges[clear].clear();

    mesh.edge_lookup.clear();
    for (index, edge) in mesh.edges.iter().enumerate() {
        if edge.active {
            mesh.edge_lookup
                .insert(edge_key(edge.vertices[0], edge.vertices[1]), index);
        }
    }
    for face in &mut mesh.faces {
        face.edges = [
            mesh.edge_lookup[&edge_key(face.vertices[0], face.vertices[1])],
            mesh.edge_lookup[&edge_key(face.vertices[1], face.vertices[2])],
            mesh.edge_lookup[&edge_key(face.vertices[2], face.vertices[0])],
        ];
    }
}
fn collapse_heap_insert_or_update(heap: &mut DecimateHeap, edge_index: usize, cost: f32) {
    if heap.positions[edge_index].is_some() {
        heap.update(edge_index, cost);
        return;
    }
    heap.nodes[edge_index].value = cost;
    let position = heap.tree.len();
    heap.positions[edge_index] = Some(position);
    heap.tree.push(edge_index);
    heap.up(position);
}

fn collapse_heap_update_after_collapse(
    mesh: &CollapseMesh,
    keep: usize,
    quadrics: &[DecimateQuadric],
    normals: &[Vec3],
    weights: Option<&[f32]>,
    weight_factor: f32,
    heap: &mut DecimateHeap,
) {
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        if !edge.active {
            heap.remove(edge_index);
        }
    }
    let (links, counts) = collapse_edge_face_links(mesh);
    let mut touched = mesh.vertex_edges[keep].clone();
    for edge_index in &mesh.vertex_edges[keep] {
        for face_index in links[*edge_index]
            .iter()
            .take(usize::from(counts[*edge_index]))
        {
            let face = &mesh.faces[*face_index];
            if let Some(corner) = face.vertices.iter().position(|vertex| *vertex == keep)
                && face.edges[(corner + 2) % 3] == *edge_index
            {
                touched.push(face.edges[(corner + 1) % 3]);
            }
        }
    }
    for edge_index in touched {
        if let Some(cost) = collapse_edge_cost(
            mesh,
            edge_index,
            &links,
            &counts,
            quadrics,
            normals,
            weights,
            weight_factor,
        ) {
            collapse_heap_insert_or_update(heap, edge_index, cost);
        } else {
            heap.remove(edge_index);
        }
    }
}

fn collapse_quad_is_convex(points: &[Vec3], vertices: &[usize]) -> bool {
    if vertices.len() != 4 {
        return false;
    }
    let corners = [
        points[vertices[0]],
        points[vertices[1]],
        points[vertices[2]],
        points[vertices[3]],
    ];
    let mut normal = Vec3::ZERO;
    for index in 0..4 {
        normal += corners[index].cross(corners[(index + 1) % 4]);
    }
    if normal.length_squared() <= f32::MIN_POSITIVE {
        return false;
    }
    let normal = normal.normalize_or_zero();
    (0..4).all(|index| {
        let first = corners[(index + 1) % 4] - corners[index];
        let second = corners[(index + 2) % 4] - corners[(index + 1) % 4];
        first.cross(second).dot(normal) > f32::EPSILON
    })
}

pub(super) fn decimate_collapse(mesh: &Mesh, ratio: f64, modifier: &Modifier) -> Result<Mesh> {
    if mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let weights = decimate_group_weights(mesh, modifier)?;
    let weight_factor = number_param(modifier, "vertex_group_factor", 1.0)?;
    if !(0.0..=1.0).contains(&weight_factor) {
        return Err(invalid_parameter(
            modifier,
            "vertex_group_factor",
            "a number from 0 through 1",
        ));
    }
    let use_triangulate = bool_param(modifier, "use_collapse_triangulate", false)?;
    let symmetry_axis = if bool_param(modifier, "use_symmetry", false)? {
        Some(decimate_symmetry_axis(modifier)?)
    } else {
        None
    };
    let invert_weights = bool_param(modifier, "invert_vertex_group", false)?;
    if ratio < 1.0 && invert_weights && weights.is_some() && weight_factor > 0.0 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "DECIMATE COLLAPSE with an inverted vertex group is not supported",
            json!({"feature_id":"modifier.decimate.collapse_vertex_group_invert"}),
        ));
    }
    let mut weights: Option<Vec<f32>> =
        weights.map(|values| values.into_iter().map(|value| value as f32).collect());
    let weight_factor = weight_factor as f32;
    if weight_factor == 0.0 {
        weights = None;
    }
    let mut model = collapse_mesh_from_mesh(mesh)?;
    let target_faces = (model.faces.len() as f32 * ratio as f32) as usize;
    let mut quadrics = collapse_quadrics(&model);
    let mut normals = collapse_vertex_normals(mesh);
    let mut heap = DecimateHeap::new(vec![f32::MAX; model.edges.len()]);
    heap.tree.clear();
    heap.positions.fill(None);
    let (links, counts) = collapse_edge_face_links(&model);
    for edge_index in 0..model.edges.len() {
        if let Some(cost) = collapse_edge_cost(
            &model,
            edge_index,
            &links,
            &counts,
            &quadrics,
            &normals,
            weights.as_deref(),
            weight_factor,
        ) {
            collapse_heap_insert_or_update(&mut heap, edge_index, cost);
        }
    }
    while model.faces.len() > target_faces {
        let Some(edge_index) = heap.top() else {
            break;
        };
        if heap.value(edge_index) == Some(f32::MAX) {
            break;
        }
        heap.remove(edge_index);
        let (links, counts) = collapse_edge_face_links(&model);
        let mirror_index = if let Some(axis) = symmetry_axis {
            let symmetry = collapse_symmetry_map(&model, axis);
            let Some(mirror_index) = symmetry[edge_index] else {
                collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
                continue;
            };
            if mirror_index != edge_index {
                if heap.positions[mirror_index].is_none() {
                    collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
                    continue;
                }
                if model.edges[edge_index]
                    .vertices
                    .iter()
                    .any(|vertex| model.edges[mirror_index].vertices.contains(vertex))
                {
                    continue;
                }
                if collapse_is_topologically_degenerate(&model, edge_index, &links, &counts)
                    || collapse_is_topologically_degenerate(&model, mirror_index, &links, &counts)
                {
                    heap.remove(mirror_index);
                    collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
                    collapse_heap_insert_or_update(&mut heap, mirror_index, f32::MAX);
                    continue;
                }
            } else if collapse_is_topologically_degenerate(&model, edge_index, &links, &counts) {
                collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
                continue;
            }
            Some(mirror_index)
        } else {
            if collapse_is_topologically_degenerate(&model, edge_index, &links, &counts) {
                collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
                continue;
            }
            None
        };
        let edge = model.edges[edge_index];
        let quadric = quadrics[edge.vertices[0]].add(quadrics[edge.vertices[1]]);
        let optimized = quadric.optimize().unwrap_or_else(|| {
            collapse_midpoint_db(&model.points, edge.vertices[0], edge.vertices[1])
        });
        if !optimized.is_finite() {
            collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
            if let Some(mirror_index) = mirror_index
                && mirror_index != edge_index
            {
                heap.remove(mirror_index);
                collapse_heap_insert_or_update(&mut heap, mirror_index, f32::MAX);
            }
            continue;
        }
        let mut position = optimized.as_vec3();
        if mirror_index == Some(edge_index)
            && let Some(axis) = symmetry_axis
        {
            position[axis] = 0.0;
        }
        if collapse_would_flip(&model, edge_index, position) {
            collapse_heap_insert_or_update(&mut heap, edge_index, f32::MAX);
            if let Some(mirror_index) = mirror_index
                && mirror_index != edge_index
            {
                heap.remove(mirror_index);
                collapse_heap_insert_or_update(&mut heap, mirror_index, f32::MAX);
            }
            continue;
        }
        let keep = edge.vertices[0];
        collapse_apply(
            &mut model,
            edge_index,
            position,
            &mut quadrics,
            &mut normals,
            &mut weights,
        );
        collapse_heap_update_after_collapse(
            &model,
            keep,
            &quadrics,
            &normals,
            weights.as_deref(),
            weight_factor,
            &mut heap,
        );
        if let Some(mirror_index) = mirror_index
            && mirror_index != edge_index
            && heap.positions[mirror_index].is_some()
        {
            heap.remove(mirror_index);
            let mirror_position = collapse_reflect(position, symmetry_axis.unwrap_or(0));
            let mirror_keep = model.edges[mirror_index].vertices[0];
            collapse_apply(
                &mut model,
                mirror_index,
                mirror_position,
                &mut quadrics,
                &mut normals,
                &mut weights,
            );
            collapse_heap_update_after_collapse(
                &model,
                mirror_keep,
                &quadrics,
                &normals,
                weights.as_deref(),
                weight_factor,
                &mut heap,
            );
        }
    }

    let polygons = if use_triangulate {
        model
            .faces
            .iter()
            .map(|face| (face.vertices.to_vec(), face.material))
            .collect()
    } else {
        collapse_untriangulate(&model)
    };
    let (_, counts) = collapse_edge_face_links(&model);
    let loose = model
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            (edge.active && edge.input_loose && counts[index] == 0).then_some(edge.vertices)
        })
        .collect();
    let points = model.points.iter().map(|point| point.as_dvec3()).collect();
    decimate_compact(points, polygons, loose, &mesh.attributes)
}

fn collapse_untriangulate(mesh: &CollapseMesh) -> Vec<Polygon> {
    let (links, counts) = collapse_edge_face_links(mesh);
    let mut merged_faces = vec![false; mesh.faces.len()];
    let mut polygons = Vec::new();
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let Some(origin) = edge.cut_origin else {
            continue;
        };
        if !edge.active || counts[edge_index] != 2 {
            continue;
        }
        let first_index = links[edge_index][0];
        let second_index = links[edge_index][1];
        if merged_faces[first_index] || merged_faces[second_index] {
            continue;
        }
        let first = &mesh.faces[first_index];
        let second = &mesh.faces[second_index];
        if first.origin != origin
            || second.origin != origin
            || first.vertices.len() != 3
            || second.vertices.len() != 3
        {
            continue;
        }
        let Some(vertices) = super::decimate_merge_faces(
            &first.vertices,
            &second.vertices,
            edge_key(edge.vertices[0], edge.vertices[1]),
        ) else {
            continue;
        };
        if !collapse_quad_is_convex(&mesh.points, &vertices) {
            continue;
        }
        merged_faces[first_index] = true;
        merged_faces[second_index] = true;
        polygons.push((vertices, first.material));
    }
    for (index, face) in mesh.faces.iter().enumerate() {
        if !merged_faces[index] {
            polygons.push((face.vertices.to_vec(), face.material));
        }
    }
    unique_polygons(polygons)
}

fn decimate_symmetry_axis(modifier: &Modifier) -> Result<usize> {
    match modifier.params.get("symmetry_axis") {
        None => Ok(0),
        Some(Value::String(axis)) => match axis.to_ascii_uppercase().as_str() {
            "X" => Ok(0),
            "Y" => Ok(1),
            "Z" => Ok(2),
            _ => Err(invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
        },
        Some(Value::Number(axis)) => axis
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value < 3)
            .ok_or_else(|| invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
        Some(_) => Err(invalid_parameter(modifier, "symmetry_axis", "X, Y, or Z")),
    }
}

fn decimate_group_weights(mesh: &Mesh, modifier: &Modifier) -> Result<Option<Vec<f64>>> {
    let group_name = match modifier.params.get("vertex_group") {
        None => "",
        Some(Value::String(name)) => name.as_str(),
        Some(_) => {
            return Err(invalid_parameter(
                modifier,
                "vertex_group",
                "a vertex-group name",
            ));
        }
    };
    if group_name.is_empty() {
        return Ok(None);
    }
    let weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group_name))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid_parameter(modifier, "vertex_group", "a group present on the mesh")
        })?;
    let invert = bool_param(modifier, "invert_vertex_group", false)?;
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(invalid_parameter(
                    modifier,
                    "vertex_group",
                    "finite weights from 0 through 1",
                ));
            }
            Ok(if invert { 1.0 - weight } else { weight })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[derive(Clone, Copy, Debug)]
struct DecimateHeapNode {
    value: f32,
}

#[derive(Debug)]
struct DecimateHeap {
    nodes: Vec<DecimateHeapNode>,
    tree: Vec<usize>,
    positions: Vec<Option<usize>>,
}

impl DecimateHeap {
    fn new(values: Vec<f32>) -> Self {
        let mut heap = Self {
            positions: vec![None; values.len()],
            nodes: values
                .into_iter()
                .map(|value| DecimateHeapNode { value })
                .collect(),
            tree: Vec::new(),
        };
        for index in 0..heap.nodes.len() {
            heap.positions[index] = Some(heap.tree.len());
            heap.tree.push(index);
            heap.up(heap.tree.len() - 1);
        }
        heap
    }

    fn top(&self) -> Option<usize> {
        self.tree.first().copied()
    }

    fn value(&self, index: usize) -> Option<f32> {
        self.nodes.get(index).map(|node| node.value)
    }

    fn swap(&mut self, first: usize, second: usize) {
        self.tree.swap(first, second);
        self.positions[self.tree[first]] = Some(first);
        self.positions[self.tree[second]] = Some(second);
    }

    fn up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) >> 1;
            if self.nodes[self.tree[parent]].value < self.nodes[self.tree[index]].value {
                break;
            }
            self.swap(parent, index);
            index = parent;
        }
    }

    fn down(&mut self, mut index: usize) {
        loop {
            let left = (index << 1) + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < self.tree.len()
                && self.nodes[self.tree[left]].value < self.nodes[self.tree[smallest]].value
            {
                smallest = left;
            }
            if right < self.tree.len()
                && self.nodes[self.tree[right]].value < self.nodes[self.tree[smallest]].value
            {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.swap(index, smallest);
            index = smallest;
        }
    }

    fn update(&mut self, index: usize, value: f32) {
        let Some(position) = self.positions.get(index).copied().flatten() else {
            return;
        };
        let old = self.nodes[index].value;
        if value < old {
            self.nodes[index].value = value;
            self.up(position);
        } else if value > old {
            self.nodes[index].value = value;
            self.down(position);
        }
    }

    fn remove(&mut self, index: usize) {
        let Some(mut position) = self.positions.get(index).copied().flatten() else {
            return;
        };
        while position > 0 {
            let parent = (position - 1) >> 1;
            self.swap(parent, position);
            position = parent;
        }
        if let Some(removed) = self.tree.first().copied() {
            self.positions[removed] = None;
            if let Some(last) = self.tree.pop()
                && !self.tree.is_empty()
            {
                self.tree[0] = last;
                self.positions[last] = Some(0);
                self.down(0);
            }
        }
    }
}

type DecimateEdge = (usize, usize);
type DecimateEdgeUses = HashMap<DecimateEdge, Vec<(usize, usize, usize)>>;

fn decimate_dissolve_edge_uses(polygons: &[Polygon]) -> DecimateEdgeUses {
    let mut uses = HashMap::new();
    for (face_index, (face, _)) in polygons.iter().enumerate() {
        for (first, second) in cyclic_pairs(face) {
            uses.entry(edge_key(*first, *second))
                .or_insert_with(Vec::new)
                .push((face_index, *first, *second));
        }
    }
    uses
}

fn decimate_dissolve_edge_order(
    mesh: &Mesh,
    indices: &HashMap<u32, usize>,
) -> Result<Vec<DecimateEdge>> {
    mesh.edges
        .iter()
        .map(|edge| {
            let first = indices.get(&edge.vertices[0]).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "modifier edge references a missing vertex",
                )
            })?;
            let second = indices.get(&edge.vertices[1]).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "modifier edge references a missing vertex",
                )
            })?;
            Ok(edge_key(first, second))
        })
        .collect()
}

fn decimate_normalize_v3(vector: [f32; 3]) -> [f32; 3] {
    let dot = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if dot > 1.0e-35 {
        let inverse = 1.0 / dot.sqrt();
        [
            vector[0] * inverse,
            vector[1] * inverse,
            vector[2] * inverse,
        ]
    } else {
        [0.0; 3]
    }
}

fn decimate_saasin(value: f32) -> f32 {
    if value <= -1.0 {
        -std::f32::consts::FRAC_PI_2
    } else if value >= 1.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        value.asin()
    }
}

fn decimate_angle_normalized_v3v3(first: [f32; 3], second: [f32; 3]) -> f32 {
    let length = |a: [f32; 3], b: [f32; 3]| {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
    if dot >= 0.0 {
        2.0 * decimate_saasin(length(first, second) / 2.0)
    } else {
        let negated = [-second[0], -second[1], -second[2]];
        std::f32::consts::PI - 2.0 * decimate_saasin(length(first, negated) / 2.0)
    }
}

fn decimate_angle_v3v3v3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let first = decimate_normalize_v3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    let second = decimate_normalize_v3([c[0] - b[0], c[1] - b[1], c[2] - b[2]]);
    decimate_angle_normalized_v3v3(first, second)
}

fn decimate_cos_v3v3v3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let first = decimate_normalize_v3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    let second = decimate_normalize_v3([c[0] - b[0], c[1] - b[1], c[2] - b[2]]);
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn decimate_dissolve_normal(points: &[DVec3], face: &[usize]) -> [f32; 3] {
    let p = |index: usize| points[face[index]].as_vec3().to_array();
    match face.len() {
        4 => {
            let (v1, v2, v3, v4) = (p(0), p(1), p(2), p(3));
            let n1 = [v1[0] - v3[0], v1[1] - v3[1], v1[2] - v3[2]];
            let n2 = [v2[0] - v4[0], v2[1] - v4[1], v2[2] - v4[2]];
            decimate_normalize_v3([
                n1[1] * n2[2] - n1[2] * n2[1],
                n1[2] * n2[0] - n1[0] * n2[2],
                n1[0] * n2[1] - n1[1] * n2[0],
            ])
        }
        3 => {
            let (v1, v2, v3) = (p(0), p(1), p(2));
            let n1 = [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
            let n2 = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]];
            decimate_normalize_v3([
                n1[1] * n2[2] - n1[2] * n2[1],
                n1[2] * n2[0] - n1[0] * n2[2],
                n1[0] * n2[1] - n1[1] * n2[0],
            ])
        }
        length => {
            let mut normal = [0.0_f32; 3];
            let mut previous = p(length - 1);
            for index in 0..length {
                let current = p(index);
                normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
                normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
                normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
                previous = current;
            }
            decimate_normalize_v3(normal)
        }
    }
}

fn decimate_dissolve_face_normals(points: &[DVec3], polygons: &[Polygon]) -> Vec<[f32; 3]> {
    polygons
        .iter()
        .map(|(face, _)| decimate_dissolve_normal(points, face))
        .collect()
}

fn decimate_dissolve_edge_flags(
    mesh: &Mesh,
    indices: &HashMap<u32, usize>,
) -> (HashSet<DecimateEdge>, HashSet<DecimateEdge>) {
    let Some(flags) = mesh
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array)
    else {
        return (HashSet::new(), HashSet::new());
    };
    let mut sharp = HashSet::new();
    let mut seam = HashSet::new();
    for (index, edge) in mesh.edges.iter().enumerate() {
        let Some(metadata) = flags.get(index) else {
            continue;
        };
        let (Some(first), Some(second)) = (
            indices.get(&edge.vertices[0]).copied(),
            indices.get(&edge.vertices[1]).copied(),
        ) else {
            continue;
        };
        let edge = edge_key(first, second);
        if metadata.get("sharp").and_then(Value::as_bool) == Some(true) {
            sharp.insert(edge);
        }
        if metadata.get("seam").and_then(Value::as_bool) == Some(true) {
            seam.insert(edge);
        }
    }
    (sharp, seam)
}

fn decimate_dissolve_uv_seams(mesh: &Mesh, polygons: &[Polygon]) -> HashSet<DecimateEdge> {
    let Some(layers) = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
    else {
        return HashSet::new();
    };
    let mut seams = HashSet::new();
    for layer in layers {
        let Some(values) = layer.get("values").and_then(Value::as_array) else {
            continue;
        };
        let mut edge_values: HashMap<DecimateEdge, (&Value, &Value)> = HashMap::new();
        let mut loop_index = 0;
        for (face, _) in polygons {
            if loop_index + face.len() > values.len() {
                break;
            }
            for index in 0..face.len() {
                let first = face[index];
                let second = face[(index + 1) % face.len()];
                let first_uv = &values[loop_index + index];
                let second_uv = &values[loop_index + (index + 1) % face.len()];
                let (first_uv, second_uv) = if first < second {
                    (first_uv, second_uv)
                } else {
                    (second_uv, first_uv)
                };
                let edge = edge_key(first, second);
                if let Some((expected_first, expected_second)) = edge_values.get(&edge) {
                    if *expected_first != first_uv || *expected_second != second_uv {
                        seams.insert(edge);
                    }
                } else {
                    edge_values.insert(edge, (first_uv, second_uv));
                }
            }
            loop_index += face.len();
        }
    }
    seams
}

fn decimate_dissolve_is_delimiter(
    edge: DecimateEdge,
    uses: &DecimateEdgeUses,
    polygons: &[Polygon],
    delimit: &HashSet<&str>,
    sharp_edges: &HashSet<DecimateEdge>,
    seam_edges: &HashSet<DecimateEdge>,
    uv_seams: &HashSet<DecimateEdge>,
) -> bool {
    let Some(edge_uses) = uses.get(&edge).filter(|edge_uses| edge_uses.len() == 2) else {
        return false;
    };
    if (delimit.contains("SEAM") && seam_edges.contains(&edge))
        || (delimit.contains("SHARP") && sharp_edges.contains(&edge))
        || (delimit.contains("UV") && uv_seams.contains(&edge))
    {
        return true;
    }
    let [(first_face, first_start, _), (second_face, second_start, _)] = edge_uses.as_slice()
    else {
        return false;
    };
    (delimit.contains("MATERIAL") && polygons[*first_face].1 != polygons[*second_face].1)
        || (delimit.contains("NORMAL") && first_start == second_start)
}

fn decimate_dissolve_edge_cost(
    edge: DecimateEdge,
    uses: &DecimateEdgeUses,
    polygons: &[Polygon],
    normals: &[[f32; 3]],
    delimit: &HashSet<&str>,
    sharp_edges: &HashSet<DecimateEdge>,
    seam_edges: &HashSet<DecimateEdge>,
    uv_seams: &HashSet<DecimateEdge>,
) -> f32 {
    let Some(edge_uses) = uses.get(&edge).filter(|edge_uses| edge_uses.len() == 2) else {
        return f32::MAX;
    };
    if decimate_dissolve_is_delimiter(
        edge,
        uses,
        polygons,
        delimit,
        sharp_edges,
        seam_edges,
        uv_seams,
    ) {
        return f32::MAX;
    }
    let [first, second] = edge_uses.as_slice() else {
        return f32::MAX;
    };
    let first_normal = normals[first.0];
    let second_normal = normals[second.0];
    let mut cost = first_normal[0] * second_normal[0]
        + first_normal[1] * second_normal[1]
        + first_normal[2] * second_normal[2];
    if first.1 != second.1 {
        cost *= -1.0;
    }
    cost
}

fn decimate_dissolve_can_join(
    first: &[usize],
    second: &[usize],
    _edge: DecimateEdge,
) -> Option<Vec<usize>> {
    let first_edges: HashSet<DecimateEdge> =
        cyclic_pairs(first).map(|(a, b)| edge_key(*a, *b)).collect();
    let second_edges: HashSet<DecimateEdge> = cyclic_pairs(second)
        .map(|(a, b)| edge_key(*a, *b))
        .collect();
    let same_direction = cyclic_pairs(first).any(|(a, b)| {
        second_edges.contains(&edge_key(*a, *b))
            && cyclic_pairs(second).any(|(c, d)| c == a && d == b)
    });
    let mut second_oriented = second.to_vec();
    if same_direction {
        second_oriented.reverse();
    }
    let mut next_of: HashMap<usize, usize> = HashMap::new();
    let mut boundary_count = 0;
    let mut start = None;
    for (a, b) in cyclic_pairs(first) {
        if !second_edges.contains(&edge_key(*a, *b)) {
            if next_of.insert(*a, *b).is_some() {
                return None;
            }
            boundary_count += 1;
            start.get_or_insert(*a);
        }
    }
    for (a, b) in cyclic_pairs(&second_oriented) {
        if !first_edges.contains(&edge_key(*a, *b)) {
            if next_of.insert(*a, *b).is_some() {
                return None;
            }
            boundary_count += 1;
            start.get_or_insert(*a);
        }
    }
    let start = start?;
    if boundary_count < 3 {
        return None;
    }
    let mut merged = Vec::with_capacity(boundary_count);
    let mut current = start;
    loop {
        merged.push(current);
        current = *next_of.get(&current)?;
        if current == start {
            break;
        }
        if merged.len() > boundary_count {
            return None;
        }
    }
    (merged.len() == boundary_count).then_some(merged)
}

fn decimate_dissolve_active_edges(
    polygons: &[Polygon],
    loose: &[[usize; 2]],
    old_order: &[DecimateEdge],
) -> Vec<DecimateEdge> {
    let mut active = HashSet::new();
    for (face, _) in polygons {
        for (first, second) in cyclic_pairs(face) {
            active.insert(edge_key(*first, *second));
        }
    }
    for [first, second] in loose {
        active.insert(edge_key(*first, *second));
    }
    let mut seen = HashSet::new();
    let mut order = Vec::with_capacity(active.len());
    for edge in old_order {
        if active.contains(edge) && seen.insert(*edge) {
            order.push(*edge);
        }
    }
    for (face, _) in polygons {
        for (first, second) in cyclic_pairs(face) {
            let edge = edge_key(*first, *second);
            if seen.insert(edge) {
                order.push(edge);
            }
        }
    }
    for [first, second] in loose {
        let edge = edge_key(*first, *second);
        if seen.insert(edge) {
            order.push(edge);
        }
    }
    order
}

fn decimate_dissolve_adjacency(
    vertex_count: usize,
    edge_order: &[DecimateEdge],
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); vertex_count];
    for (first, second) in edge_order {
        adjacency[*first].push(*second);
        adjacency[*second].push(*first);
    }
    adjacency
}

fn decimate_dissolve_vertex_cost(
    vertex: usize,
    points: &[DVec3],
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
    normals: &[[f32; 3]],
    polygons: &[Polygon],
    delimit: &HashSet<&str>,
    sharp_edges: &HashSet<DecimateEdge>,
    seam_edges: &HashSet<DecimateEdge>,
    uv_seams: &HashSet<DecimateEdge>,
) -> f32 {
    let neighbors = &adjacency[vertex];
    let angle = if neighbors.len() == 2 {
        std::f32::consts::PI
            - decimate_angle_v3v3v3(
                points[neighbors[0]].as_vec3().to_array(),
                points[vertex].as_vec3().to_array(),
                points[neighbors[1]].as_vec3().to_array(),
            )
    } else {
        std::f32::consts::FRAC_PI_2
    };
    let Some(first_neighbor) = neighbors.first() else {
        return angle;
    };
    let edge = edge_key(vertex, *first_neighbor);
    if uses
        .get(&edge)
        .is_some_and(|edge_uses| edge_uses.len() == 2)
        && !adjacency[vertex].iter().any(|neighbor| {
            decimate_dissolve_is_delimiter(
                edge_key(vertex, *neighbor),
                uses,
                polygons,
                delimit,
                sharp_edges,
                seam_edges,
                uv_seams,
            )
        })
    {
        let Some(edge_uses) = uses.get(&edge).filter(|edge_uses| edge_uses.len() == 2) else {
            return angle;
        };
        let first_normal = normals[edge_uses[0].0];
        let second_normal = normals[edge_uses[1].0];
        let face_angle = decimate_angle_normalized_v3v3(first_normal, second_normal);
        let inverse_unit_angle = 1.0 / std::f32::consts::FRAC_PI_2;
        return ((angle * inverse_unit_angle) * (face_angle * inverse_unit_angle))
            * std::f32::consts::FRAC_PI_2;
    }
    angle
}

fn decimate_dissolve_merge_face(
    polygons: &mut Vec<Polygon>,
    first_index: usize,
    second_index: usize,
    edge: DecimateEdge,
) -> Option<usize> {
    let merged = decimate_dissolve_can_join(
        &polygons.get(first_index)?.0,
        &polygons.get(second_index)?.0,
        edge,
    )?;
    let material = polygons[first_index].1;
    polygons[first_index] = (merged, material);
    polygons.remove(second_index);
    Some(first_index)
}

fn decimate_dissolve_cross(first: [f32; 2], second: [f32; 2], third: [f32; 2]) -> f32 {
    (first[0] - second[0]) * (second[1] - third[1])
        + (first[1] - second[1]) * (third[0] - second[0])
}

fn decimate_dissolve_basis(normal: [f32; 3]) -> [[f32; 3]; 2] {
    let f = normal[0] * normal[0] + normal[1] * normal[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let n1 = [normal[1] * d, -normal[0] * d, 0.0];
        let n2 = [
            -normal[2] * n1[1],
            normal[2] * n1[0],
            normal[0] * n1[1] - normal[1] * n1[0],
        ];
        [n1, n2]
    } else {
        [
            [if normal[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }
}

fn decimate_dissolve_project(point: DVec3, center: DVec3, basis: &[[f32; 3]; 2]) -> [f32; 2] {
    let point = point.as_vec3().to_array();
    let center = center.as_vec3().to_array();
    let co = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    [
        basis[0][0] * co[0] + basis[0][1] * co[1] + basis[0][2] * co[2],
        basis[1][0] * co[0] + basis[1][1] * co[1] + basis[1][2] * co[2],
    ]
}

fn decimate_signum_i(value: f32) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

fn decimate_dissolve_point_in_triangle(point: [f32; 2], triangle: [[f32; 2]; 3]) -> bool {
    let side = |first: [f32; 2], second: [f32; 2]| {
        (first[0] - point[0]) * (second[1] - point[1])
            - (second[0] - point[0]) * (first[1] - point[1])
    };
    side(triangle[0], triangle[1]) >= 0.0
        && side(triangle[1], triangle[2]) >= 0.0
        && side(triangle[2], triangle[0]) >= 0.0
}

fn decimate_dissolve_loop_is_degenerate(
    points: &[DVec3],
    polygons: &[Polygon],
    adjacency: &[Vec<usize>],
    vertex: usize,
    face_index: usize,
) -> bool {
    let face = &polygons[face_index].0;
    let Some(index) = face.iter().position(|item| *item == vertex) else {
        return false;
    };
    if face.len() <= 3 {
        return false;
    }
    let basis = decimate_dissolve_basis(decimate_dissolve_normal(points, face));
    let center = points[vertex];
    let previous = face[(index + face.len() - 1) % face.len()];
    let next = face[(index + 1) % face.len()];
    let mut triangle = [
        decimate_dissolve_project(points[previous], center, &basis),
        [0.0, 0.0],
        decimate_dissolve_project(points[next], center, &basis),
    ];
    if adjacency[previous].len() != 2 {
        let adjacent = face[(index + face.len() - 2) % face.len()];
        let adjacent = decimate_dissolve_project(points[adjacent], center, &basis);
        if decimate_signum_i(decimate_dissolve_cross(adjacent, triangle[0], triangle[1]))
            != decimate_signum_i(decimate_dissolve_cross(adjacent, triangle[0], triangle[2]))
        {
            return true;
        }
    }
    if adjacency[next].len() != 2 {
        let adjacent = face[(index + 2) % face.len()];
        let adjacent = decimate_dissolve_project(points[adjacent], center, &basis);
        if decimate_signum_i(decimate_dissolve_cross(adjacent, triangle[2], triangle[1]))
            != decimate_signum_i(decimate_dissolve_cross(adjacent, triangle[2], triangle[0]))
        {
            return true;
        }
    }
    if decimate_dissolve_cross(triangle[0], triangle[1], triangle[2]) < 0.0 {
        triangle.swap(1, 2);
    }
    for offset in 2..face.len() - 1 {
        let point = face[(index + offset) % face.len()];
        if decimate_dissolve_point_in_triangle(
            decimate_dissolve_project(points[point], center, &basis),
            triangle,
        ) {
            return true;
        }
    }
    false
}

fn decimate_dissolve_is_degenerate(
    points: &[DVec3],
    polygons: &[Polygon],
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
    vertex: usize,
) -> bool {
    let neighbors = &adjacency[vertex];
    if neighbors.len() != 2 {
        return true;
    }
    if neighbors.iter().any(|neighbor| {
        uses.get(&edge_key(vertex, *neighbor))
            .is_none_or(Vec::is_empty)
    }) {
        return false;
    }
    let a = points[neighbors[0]].as_vec3().to_array();
    let b = points[vertex].as_vec3().to_array();
    let c = points[neighbors[1]].as_vec3().to_array();
    if decimate_cos_v3v3v3(a, b, c).abs() >= 1.0 - f32::EPSILON {
        return false;
    }
    for (face_index, (face, _)) in polygons.iter().enumerate() {
        if face.len() > 3
            && face.contains(&vertex)
            && decimate_dissolve_loop_is_degenerate(points, polygons, adjacency, vertex, face_index)
        {
            return true;
        }
    }
    false
}

fn decimate_dissolve_replace_vertex(
    polygons: Vec<Polygon>,
    loose: &mut Vec<[usize; 2]>,
    vertex: usize,
    target: usize,
    other: usize,
) -> Vec<Polygon> {
    let polygons = polygons
        .into_iter()
        .filter_map(|(face, material)| {
            clean_polygon(
                face.into_iter()
                    .map(|index| if index == vertex { target } else { index })
                    .collect(),
            )
            .map(|face| (face, material))
        })
        .collect::<Vec<_>>();
    let mut seen = HashSet::new();
    *loose = std::mem::take(loose)
        .into_iter()
        .filter_map(|[first, second]| {
            let first = if first == vertex { target } else { first };
            let second = if second == vertex { target } else { second };
            (first != second && seen.insert(edge_key(first, second))).then_some([first, second])
        })
        .collect();
    let new_edge = edge_key(target, other);
    let has_face_edge = polygons.iter().any(|(face, _)| {
        cyclic_pairs(face).any(|(first, second)| edge_key(*first, *second) == new_edge)
    });
    if !has_face_edge && seen.insert(new_edge) {
        loose.push([new_edge.0, new_edge.1]);
    }
    unique_polygons(polygons)
}

fn decimate_dissolve_remap_edge_set(
    edges: &mut HashSet<DecimateEdge>,
    killed_edge: DecimateEdge,
    retained_edge: DecimateEdge,
    new_edge: DecimateEdge,
) {
    edges.remove(&killed_edge);
    if edges.remove(&retained_edge) {
        edges.insert(new_edge);
    }
}

fn decimate_dissolve_apply_vertex(
    polygons: &mut Vec<Polygon>,
    loose: &mut Vec<[usize; 2]>,
    edge_order: &mut Vec<DecimateEdge>,
    vertex: usize,
    target: usize,
    other: usize,
    sharp_edges: &mut HashSet<DecimateEdge>,
    seam_edges: &mut HashSet<DecimateEdge>,
    uv_seams: &mut HashSet<DecimateEdge>,
) {
    *polygons =
        decimate_dissolve_replace_vertex(std::mem::take(polygons), loose, vertex, target, other);
    let killed_edge = edge_key(vertex, target);
    let retained_edge = edge_key(vertex, other);
    let new_edge = edge_key(target, other);
    decimate_dissolve_remap_edge_set(sharp_edges, killed_edge, retained_edge, new_edge);
    decimate_dissolve_remap_edge_set(seam_edges, killed_edge, retained_edge, new_edge);
    decimate_dissolve_remap_edge_set(uv_seams, killed_edge, retained_edge, new_edge);
    let mut seen = HashSet::new();
    let mut remapped = Vec::with_capacity(edge_order.len());
    for (first, second) in std::mem::take(edge_order) {
        let first = if first == vertex { target } else { first };
        let second = if second == vertex { target } else { second };
        if first != second {
            let edge = edge_key(first, second);
            if seen.insert(edge) {
                remapped.push(edge);
            }
        }
    }
    let new_edge = edge_key(target, other);
    if seen.insert(new_edge) {
        remapped.push(new_edge);
    }
    *edge_order = decimate_dissolve_active_edges(polygons, loose, &remapped);
}

fn decimate_dissolve_faces(
    points: &[DVec3],
    polygons: &mut Vec<Polygon>,
    edge_order: &[DecimateEdge],
    angle_limit_cos_neg: f32,
    delimit: &HashSet<&str>,
    sharp_edges: &HashSet<DecimateEdge>,
    seam_edges: &HashSet<DecimateEdge>,
    uv_seams: &HashSet<DecimateEdge>,
) -> Vec<usize> {
    let mut stamps: Vec<usize> = (0..polygons.len()).collect();
    let mut next_stamp = polygons.len();
    let mut uses = decimate_dissolve_edge_uses(polygons);
    let mut normals = decimate_dissolve_face_normals(points, polygons);
    let mut edge_indices = HashMap::with_capacity(edge_order.len());
    for (index, edge) in edge_order.iter().enumerate() {
        edge_indices.insert(*edge, index);
    }
    let mut heap = DecimateHeap::new(
        edge_order
            .iter()
            .map(|edge| {
                decimate_dissolve_edge_cost(
                    *edge,
                    &uses,
                    polygons,
                    &normals,
                    delimit,
                    sharp_edges,
                    seam_edges,
                    uv_seams,
                )
            })
            .collect(),
    );
    while let Some(edge_index) = heap.top() {
        if heap
            .value(edge_index)
            .is_none_or(|cost| cost >= angle_limit_cos_neg)
        {
            break;
        }
        let edge = edge_order[edge_index];
        let Some(edge_uses) = uses.get(&edge).filter(|edge_uses| edge_uses.len() == 2) else {
            heap.update(edge_index, f32::MAX);
            continue;
        };
        let (newer, older) = if stamps[edge_uses[0].0] > stamps[edge_uses[1].0] {
            (edge_uses[0].0, edge_uses[1].0)
        } else {
            (edge_uses[1].0, edge_uses[0].0)
        };
        if decimate_dissolve_merge_face(polygons, newer, older, edge).is_none() {
            heap.update(edge_index, f32::MAX);
            continue;
        }
        heap.remove(edge_index);
        stamps[newer] = next_stamp;
        next_stamp += 1;
        stamps.remove(older);
        let first_face = if older < newer { newer - 1 } else { newer };
        uses = decimate_dissolve_edge_uses(polygons);
        normals = decimate_dissolve_face_normals(points, polygons);
        let merged_face = &polygons[first_face].0;
        for (first, second) in cyclic_pairs(merged_face) {
            let edge = edge_key(*first, *second);
            if let Some(index) = edge_indices.get(&edge).copied() {
                heap.update(
                    index,
                    decimate_dissolve_edge_cost(
                        edge,
                        &uses,
                        polygons,
                        &normals,
                        delimit,
                        sharp_edges,
                        seam_edges,
                        uv_seams,
                    ),
                );
            }
        }
    }
    stamps
}

fn decimate_dissolve_collapse_faces(
    polygons: &mut Vec<Polygon>,
    stamps: &mut Vec<usize>,
    loose: &mut Vec<[usize; 2]>,
    vertex: usize,
    target: usize,
    old: usize,
) {
    let mut kept_stamps = Vec::with_capacity(stamps.len());
    let mut seen = HashSet::new();
    for ((face, _), stamp) in polygons.iter().zip(stamps.iter()) {
        let cleaned = clean_polygon(
            face.iter()
                .map(|index| if *index == vertex { target } else { *index })
                .collect(),
        );
        if let Some(cleaned) = cleaned {
            let mut key = cleaned;
            key.sort_unstable();
            if seen.insert(key) {
                kept_stamps.push(*stamp);
            }
        }
    }
    *polygons =
        decimate_dissolve_replace_vertex(std::mem::take(polygons), loose, vertex, target, old);
    *stamps = kept_stamps;
}

pub(super) fn decimate_dissolve(
    mesh: &Mesh,
    angle_limit: f64,
    modifier: &Modifier,
) -> Result<Mesh> {
    let dissolve_boundaries = bool_param(modifier, "use_dissolve_boundaries", false)?;
    if angle_limit == 0.0 || mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let points = positions(mesh);
    let mut polygons = polygons(mesh)?;
    let delimit: HashSet<_> = modifier
        .params
        .get("delimit")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let indices = id_indices(mesh);
    let mut edge_order = decimate_dissolve_edge_order(mesh, &indices)?;
    let (mut sharp_edges, mut seam_edges) = decimate_dissolve_edge_flags(mesh, &indices);
    let mut uv_seams = decimate_dissolve_uv_seams(mesh, &polygons);
    let mut loose = loose_edges(mesh)?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Blender stores Decimate angle limits as float32"
    )]
    let angle_limit = angle_limit as f32;
    let mut stamps = decimate_dissolve_faces(
        &points,
        &mut polygons,
        &edge_order,
        -angle_limit.cos(),
        &delimit,
        &sharp_edges,
        &seam_edges,
        &uv_seams,
    );
    edge_order = decimate_dissolve_active_edges(&polygons, &loose, &edge_order);
    if dissolve_boundaries {
        for vertex in 0..points.len() {
            let adjacency = decimate_dissolve_adjacency(points.len(), &edge_order);
            if adjacency[vertex].len() == 2 {
                let target = adjacency[vertex][0];
                let other = adjacency[vertex][1];
                decimate_dissolve_apply_vertex(
                    &mut polygons,
                    &mut loose,
                    &mut edge_order,
                    vertex,
                    target,
                    other,
                    &mut sharp_edges,
                    &mut seam_edges,
                    &mut uv_seams,
                );
            }
        }
    } else {
        let orient: HashMap<DecimateEdge, [usize; 2]> = mesh
            .edges
            .iter()
            .filter_map(|edge| {
                let first = *indices.get(&edge.vertices[0])?;
                let second = *indices.get(&edge.vertices[1])?;
                Some((edge_key(first, second), [first, second]))
            })
            .collect();
        let mut edge_table: Vec<[usize; 2]> = edge_order
            .iter()
            .map(|edge| orient.get(edge).copied().unwrap_or([edge.0, edge.1]))
            .collect();
        let mut disk: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
        for (id, [first, second]) in edge_table.iter().enumerate() {
            disk[*first].push(id);
            disk[*second].push(id);
        }
        let adjacency_of = |disk: &[Vec<usize>], table: &[[usize; 2]]| -> Vec<Vec<usize>> {
            disk.iter()
                .enumerate()
                .map(|(vertex, list)| {
                    list.iter()
                        .map(|id| {
                            if table[*id][0] == vertex {
                                table[*id][1]
                            } else {
                                table[*id][0]
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let live: Vec<usize> = (0..points.len())
            .filter(|vertex| !disk[*vertex].is_empty())
            .collect();
        let mut node_of: Vec<Option<usize>> = vec![None; points.len()];
        for (node, vertex) in live.iter().enumerate() {
            node_of[*vertex] = Some(node);
        }
        let mut uses = decimate_dissolve_edge_uses(&polygons);
        let mut adjacency = adjacency_of(&disk, &edge_table);
        let mut normals = decimate_dissolve_face_normals(&points, &polygons);
        let costs = live
            .iter()
            .map(|vertex| {
                decimate_dissolve_vertex_cost(
                    *vertex,
                    &points,
                    &adjacency,
                    &uses,
                    &normals,
                    &polygons,
                    &delimit,
                    &sharp_edges,
                    &seam_edges,
                    &uv_seams,
                )
            })
            .collect();
        let mut heap = DecimateHeap::new(costs);
        while let Some(node) = heap.top() {
            if heap.value(node).is_none_or(|cost| cost >= angle_limit) {
                break;
            }
            let vertex = live[node];
            let neighbors = &adjacency[vertex];
            if neighbors.len() != 2
                || decimate_dissolve_is_degenerate(&points, &polygons, &adjacency, &uses, vertex)
            {
                heap.update(node, f32::MAX);
                continue;
            }
            let target = neighbors[0];
            let old = neighbors[1];
            let e_kill = disk[vertex][0];
            let e_old = disk[vertex][1];
            decimate_dissolve_collapse_faces(
                &mut polygons,
                &mut stamps,
                &mut loose,
                vertex,
                target,
                old,
            );
            let killed_edge = edge_key(vertex, target);
            let retained_edge = edge_key(vertex, old);
            let new_edge = edge_key(target, old);
            decimate_dissolve_remap_edge_set(
                &mut sharp_edges,
                killed_edge,
                retained_edge,
                new_edge,
            );
            decimate_dissolve_remap_edge_set(&mut seam_edges, killed_edge, retained_edge, new_edge);
            decimate_dissolve_remap_edge_set(&mut uv_seams, killed_edge, retained_edge, new_edge);
            heap.remove(node);
            node_of[vertex] = None;
            for slot in &mut edge_table[e_old] {
                if *slot == vertex {
                    *slot = target;
                }
            }
            disk[target].retain(|id| *id != e_kill);
            disk[target].push(e_old);
            disk[vertex].clear();
            let spliced = disk[target].iter().copied().find(|id| {
                let ends = edge_table[*id];
                *id != e_old && (ends == [target, old] || ends == [old, target])
            });
            if let Some(e_splice) = spliced {
                disk[target].retain(|id| *id != e_splice);
                disk[old].retain(|id| *id != e_splice);
            }
            uses = decimate_dissolve_edge_uses(&polygons);
            adjacency = adjacency_of(&disk, &edge_table);
            normals = decimate_dissolve_face_normals(&points, &polygons);
            for end in edge_table[e_old] {
                if let Some(end_node) = node_of[end] {
                    heap.update(
                        end_node,
                        decimate_dissolve_vertex_cost(
                            end,
                            &points,
                            &adjacency,
                            &uses,
                            &normals,
                            &polygons,
                            &delimit,
                            &sharp_edges,
                            &seam_edges,
                            &uv_seams,
                        ),
                    );
                }
            }
            let mut radial: Vec<usize> = uses
                .get(&new_edge)
                .map(|edge_uses| edge_uses.iter().map(|(face, _, _)| *face).collect())
                .unwrap_or_default();
            radial.sort_by_key(|face| std::cmp::Reverse(stamps[*face]));
            for face_index in radial {
                let face = &polygons[face_index].0;
                let length = face.len();
                let Some(start) = (0..length)
                    .find(|index| edge_key(face[*index], face[(*index + 1) % length]) == new_edge)
                else {
                    continue;
                };
                let stop = (start + length - 1) % length;
                let mut cursor = (start + 2) % length;
                loop {
                    let candidate = face[cursor];
                    if let Some(candidate_node) = node_of[candidate]
                        && heap.value(candidate_node) == Some(f32::MAX)
                    {
                        heap.update(
                            candidate_node,
                            decimate_dissolve_vertex_cost(
                                candidate,
                                &points,
                                &adjacency,
                                &uses,
                                &normals,
                                &polygons,
                                &delimit,
                                &sharp_edges,
                                &seam_edges,
                                &uv_seams,
                            ),
                        );
                    }
                    cursor = (cursor + 1) % length;
                    if cursor == stop {
                        break;
                    }
                }
            }
        }
    }
    decimate_compact(points, polygons, loose, &mesh.attributes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecimateUnsubdivideMethod {
    InteriorFan,
    BoundaryFan,
    InteriorChain,
    WireChain,
}

fn decimate_unsubdivide_method(
    vertex: usize,
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
) -> Option<DecimateUnsubdivideMethod> {
    let edges = adjacency.get(vertex)?;
    if edges.len() > 4 {
        return None;
    }
    let mut boundary_edges = 0;
    let mut manifold_edges = 0;
    let mut wire_edges = 0;
    for neighbor in edges {
        match uses.get(&edge_key(vertex, *neighbor)).map_or(0, Vec::len) {
            0 => wire_edges += 1,
            1 => boundary_edges += 1,
            2 => manifold_edges += 1,
            _ => return None,
        }
    }
    if (edges.len() == 3 && boundary_edges == 0 && manifold_edges == 3)
        || (edges.len() == 4 && boundary_edges == 0 && manifold_edges == 4)
    {
        Some(DecimateUnsubdivideMethod::InteriorFan)
    } else if edges.len() == 3 && boundary_edges == 2 && manifold_edges == 1 {
        Some(DecimateUnsubdivideMethod::BoundaryFan)
    } else if edges.len() == 2 && wire_edges == 2 {
        Some(DecimateUnsubdivideMethod::WireChain)
    } else if edges.len() == 2 && manifold_edges == 2 {
        Some(DecimateUnsubdivideMethod::InteriorChain)
    } else {
        None
    }
}

fn decimate_unsubdivide_tag_neighbors(
    starts: &[usize],
    desired_tag: i8,
    tags: &mut [i8],
    adjacency: &[Vec<usize>],
) -> Vec<usize> {
    let mut newly_tagged = Vec::new();
    for vertex in starts {
        for neighbor in &adjacency[*vertex] {
            if tags[*neighbor] == 0 {
                tags[*neighbor] = desired_tag;
                newly_tagged.push(*neighbor);
            }
        }
    }
    newly_tagged
}

fn decimate_unsubdivide_incident_faces(
    vertex: usize,
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut faces = Vec::new();
    for neighbor in &adjacency[vertex] {
        if let Some(edge_uses) = uses.get(&edge_key(vertex, *neighbor)) {
            for (face_index, start, _) in edge_uses {
                if *start == vertex && seen.insert(*face_index) {
                    faces.push(*face_index);
                }
            }
        }
    }
    faces
}

fn decimate_unsubdivide_central_face(
    vertex: usize,
    polygons: &[Polygon],
    incident_faces: &[usize],
) -> (Vec<usize>, Vec<(usize, usize, usize)>) {
    let mut loops = Vec::with_capacity(incident_faces.len());
    for face_index in incident_faces {
        let face = &polygons[*face_index].0;
        if let Some(index) = face.iter().position(|item| *item == vertex) {
            loops.push((
                *face_index,
                face[(index + 1) % face.len()],
                face[(index + face.len() - 1) % face.len()],
            ));
        }
    }
    let mut start = 0;
    if let Some(index) = loops.iter().enumerate().find_map(|(index, current)| {
        loops
            .iter()
            .enumerate()
            .all(|(other_index, other)| other_index == index || other.2 != current.1)
            .then_some(index)
    }) {
        start = index;
    }
    let mut ordered = Vec::with_capacity(loops.len());
    let mut used = HashSet::new();
    let mut current = start;
    while used.insert(current) {
        ordered.push(loops[current]);
        let previous = loops[current].2;
        let Some(next) = loops.iter().enumerate().find_map(|(index, item)| {
            (!used.contains(&index) && item.1 == previous).then_some(index)
        }) else {
            break;
        };
        current = next;
    }
    let mut central = Vec::with_capacity(ordered.len() * 2);
    for (_, next, previous) in &ordered {
        if central.last() != Some(next) {
            central.push(*next);
        }
        if central.last() != Some(previous) {
            central.push(*previous);
        }
    }
    if central.len() > 1 && central.first() == central.last() {
        central.pop();
    }
    (central, ordered)
}

fn decimate_unsubdivide_faces_match(first: &[usize], second: &[usize]) -> bool {
    if first.len() != second.len() {
        return false;
    }
    for start in 0..second.len() {
        if second[start] != first[0] {
            continue;
        }
        if (0..first.len()).all(|index| first[index] == second[(start + index) % second.len()])
            || (0..first.len())
                .all(|index| first[index] == second[(start + second.len() - index) % second.len()])
        {
            return true;
        }
    }
    false
}

fn decimate_unsubdivide_fan_is_duplicate(
    vertex: usize,
    method: DecimateUnsubdivideMethod,
    polygons: &[Polygon],
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
    incident_faces: &[usize],
    central: &[usize],
    ordered_loops: &[(usize, usize, usize)],
) -> bool {
    let incident: HashSet<_> = incident_faces.iter().copied().collect();
    if ordered_loops.len() >= 3
        && polygons.iter().enumerate().any(|(index, (face, _))| {
            !incident.contains(&index) && decimate_unsubdivide_faces_match(central, face)
        })
    {
        return true;
    }
    let mut ears = Vec::new();
    let mut residuals = Vec::new();
    for (face_index, next, previous) in ordered_loops {
        let face = &polygons[*face_index].0;
        if face.len() > 3 {
            let Some(vertex_index) = face.iter().position(|item| *item == vertex) else {
                continue;
            };
            let ear = vec![*previous, vertex, *next];
            if polygons.iter().enumerate().any(|(index, (other, _))| {
                index != *face_index && decimate_unsubdivide_faces_match(&ear, other)
            }) || ears
                .iter()
                .any(|other: &Vec<usize>| decimate_unsubdivide_faces_match(&ear, other))
            {
                return true;
            }
            ears.push(ear);
            let mut residual = Vec::with_capacity(face.len() - 1);
            for offset in 1..face.len() {
                let item = face[(vertex_index + offset) % face.len()];
                if item != vertex {
                    residual.push(item);
                }
            }
            residuals.push((*face_index, residual));
        }
    }
    for first in 0..residuals.len() {
        for second in first + 1..residuals.len() {
            if residuals[first].1.len() == residuals[second].1.len()
                && decimate_unsubdivide_faces_match(&residuals[first].1, &residuals[second].1)
            {
                return true;
            }
        }
    }
    if method == DecimateUnsubdivideMethod::BoundaryFan {
        let mut boundary_neighbors = adjacency[vertex]
            .iter()
            .filter(|neighbor| {
                uses.get(&edge_key(vertex, **neighbor))
                    .is_some_and(|edge_uses| edge_uses.len() == 1)
            })
            .copied();
        if let (Some(first), Some(second)) = (boundary_neighbors.next(), boundary_neighbors.next())
            && uses
                .get(&edge_key(first, second))
                .is_some_and(|edge_uses| !edge_uses.is_empty())
        {
            return true;
        }
    }
    false
}

fn decimate_unsubdivide_apply_fan(
    vertex: usize,
    polygons: &mut Vec<Polygon>,
    edge_order: &mut Vec<DecimateEdge>,
    method: DecimateUnsubdivideMethod,
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
) -> bool {
    let incident_faces = decimate_unsubdivide_incident_faces(vertex, adjacency, uses);
    if incident_faces.is_empty() {
        return false;
    }
    let (central, ordered_loops) =
        decimate_unsubdivide_central_face(vertex, polygons, &incident_faces);
    if decimate_unsubdivide_fan_is_duplicate(
        vertex,
        method,
        polygons,
        adjacency,
        uses,
        &incident_faces,
        &central,
        &ordered_loops,
    ) {
        return false;
    }
    if central.len() < 3 {
        return false;
    }
    let incident: HashSet<_> = incident_faces.iter().copied().collect();
    let first_triangle = incident_faces
        .iter()
        .copied()
        .find(|index| polygons[*index].0.len() == 3);
    let material = incident_faces.first().map_or(0, |index| polygons[*index].1);
    let split_edges = ordered_loops
        .iter()
        .filter_map(|(face_index, next, previous)| {
            (polygons[*face_index].0.len() > 3).then_some(edge_key(*next, *previous))
        })
        .collect::<Vec<_>>();
    let mut next = Vec::with_capacity(polygons.len() + ordered_loops.len());
    let mut central_added = false;
    for (index, (face, face_material)) in std::mem::take(polygons).into_iter().enumerate() {
        if !incident.contains(&index) {
            next.push((face, face_material));
        } else if face.len() > 3 {
            next.push((
                face.into_iter().filter(|item| *item != vertex).collect(),
                face_material,
            ));
        } else if Some(index) == first_triangle {
            next.push((central.clone(), material));
            central_added = true;
        }
    }
    if !central_added {
        next.push((central, material));
    }
    *polygons = next;
    edge_order.retain(|(first, second)| *first != vertex && *second != vertex);
    for edge in split_edges {
        if !edge_order.contains(&edge) {
            edge_order.push(edge);
        }
    }
    true
}

fn decimate_unsubdivide_apply_chain(
    vertex: usize,
    polygons: &mut Vec<Polygon>,
    loose: &mut Vec<[usize; 2]>,
    edge_order: &mut Vec<DecimateEdge>,
    neighbors: &[usize],
) -> bool {
    if neighbors.len() != 2 {
        return false;
    }
    let target = neighbors[0];
    let other = neighbors[1];
    *polygons =
        decimate_dissolve_replace_vertex(std::mem::take(polygons), loose, vertex, target, other);
    let mut seen = HashSet::new();
    let mut remapped = Vec::with_capacity(edge_order.len());
    for (first, second) in std::mem::take(edge_order) {
        let first = if first == vertex { target } else { first };
        let second = if second == vertex { target } else { second };
        if first != second {
            let edge = edge_key(first, second);
            if seen.insert(edge) {
                remapped.push(edge);
            }
        }
    }
    *edge_order = decimate_dissolve_active_edges(polygons, loose, &remapped);
    true
}

fn decimate_unsubdivide_dissolve_vertex(
    vertex: usize,
    method: DecimateUnsubdivideMethod,
    polygons: &mut Vec<Polygon>,
    loose: &mut Vec<[usize; 2]>,
    edge_order: &mut Vec<DecimateEdge>,
    adjacency: &[Vec<usize>],
    uses: &DecimateEdgeUses,
) -> bool {
    match method {
        DecimateUnsubdivideMethod::InteriorFan | DecimateUnsubdivideMethod::BoundaryFan => {
            decimate_unsubdivide_apply_fan(vertex, polygons, edge_order, method, adjacency, uses)
        }
        DecimateUnsubdivideMethod::InteriorChain | DecimateUnsubdivideMethod::WireChain => {
            decimate_unsubdivide_apply_chain(
                vertex,
                polygons,
                loose,
                edge_order,
                &adjacency[vertex],
            )
        }
    }
}

pub(super) fn decimate_unsubdivide(mesh: &Mesh, iterations: usize) -> Result<Mesh> {
    if iterations == 0 || mesh.faces.len() <= 3 {
        return Ok(mesh.clone());
    }
    let points = positions(mesh);
    let mut polygons = polygons(mesh)?;
    let mut loose = loose_edges(mesh)?;
    let indices = id_indices(mesh);
    let mut edge_order = decimate_dissolve_edge_order(mesh, &indices)?;
    for _ in 0..iterations {
        edge_order = decimate_dissolve_active_edges(&polygons, &loose, &edge_order);
        let uses = decimate_dissolve_edge_uses(&polygons);
        let adjacency = decimate_dissolve_adjacency(points.len(), &edge_order);
        let mut tags = vec![0_i8; points.len()];
        for (vertex, tag) in tags.iter_mut().enumerate() {
            if decimate_unsubdivide_method(vertex, &adjacency, &uses).is_none() {
                *tag = 1;
            }
        }
        let mut marked = false;
        for start in 0..points.len() {
            if tags[start] != 0 {
                continue;
            }
            tags[start] = 1;
            let mut ignore = vec![start];
            loop {
                let collapse =
                    decimate_unsubdivide_tag_neighbors(&ignore, -1, &mut tags, &adjacency);
                if collapse.is_empty() {
                    break;
                }
                marked = true;
                let next_ignore =
                    decimate_unsubdivide_tag_neighbors(&collapse, 1, &mut tags, &adjacency);
                if next_ignore.is_empty() {
                    break;
                }
                ignore = next_ignore;
            }
        }
        if !marked {
            break;
        }
        for (vertex, tag) in tags.iter().enumerate() {
            if *tag != -1 {
                continue;
            }
            let current_uses = decimate_dissolve_edge_uses(&polygons);
            let current_adjacency = decimate_dissolve_adjacency(points.len(), &edge_order);
            let Some(method) =
                decimate_unsubdivide_method(vertex, &current_adjacency, &current_uses)
            else {
                continue;
            };
            decimate_unsubdivide_dissolve_vertex(
                vertex,
                method,
                &mut polygons,
                &mut loose,
                &mut edge_order,
                &current_adjacency,
                &current_uses,
            );
        }
    }
    decimate_compact(points, polygons, loose, &mesh.attributes)
}
