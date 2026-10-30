use std::collections::{HashMap, HashSet};

use glam::Vec3;
use num_traits::ToPrimitive;
use serde_json::{Value, json};

use crate::{
    error::Result,
    geom::{Face, Mesh, edge_key},
};

use super::{Field, MeshFieldCache, domain_element_count, evaluation, field_json};

#[cfg(test)]
std::thread_local! {
    static ADAPTATION_MAP_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn cached_mesh_field_value(
    cache: &MeshFieldCache,
    mesh: &Mesh,
    domain: &str,
    index: usize,
    compute: impl FnOnce(&Mesh, &str) -> Result<Vec<Value>>,
) -> Result<Value> {
    let domain = canonical_domain(domain)?;
    let context_id = super::mesh_field_context_id(mesh);
    if context_id == 0 {
        return compute(mesh, domain)?
            .get(index)
            .cloned()
            .ok_or_else(|| evaluation("mesh topology field index is out of range"));
    }
    let key = (context_id, domain);
    let cached = cache
        .borrow()
        .get(&key)
        .and_then(|values| values.get(index))
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let values = compute(mesh, domain)?;
    let mut cache = cache.borrow_mut();
    let values = cache.entry(key).or_insert(values);
    values
        .get(index)
        .cloned()
        .ok_or_else(|| evaluation("mesh topology field index is out of range"))
}

pub(super) fn mesh_input_field_values(
    mesh: &Mesh,
    node_type: &str,
    output: &str,
    requested_domain: &str,
) -> Result<Vec<Value>> {
    let source_domain = input_domain(node_type)?;
    let requested_domain = canonical_domain(requested_domain)?;
    let source_count = domain_element_count(mesh, source_domain)?;
    let native = mesh_input_native_values(mesh, node_type, output)?;
    if native.len() != source_count {
        return Err(evaluation(
            "mesh topology input field has an invalid length",
        ));
    }
    if requested_domain == source_domain {
        return Ok(native);
    }
    adapt_mesh_field_values(
        &native,
        source_domain,
        requested_domain,
        mesh,
        &mesh_input_zero(node_type, output),
    )
}

pub(super) fn mesh_topology_field_values(
    mesh: &Mesh,
    node_type: &str,
    output: &str,
    inputs: &[Field],
    domain: &str,
) -> Result<Vec<Value>> {
    let domain = canonical_domain(domain)?;
    let maps = TopologyMaps::new(mesh)?;
    (0..domain_element_count(mesh, domain)?)
        .map(|index| topology_value(mesh, &maps, node_type, output, inputs, index, domain))
        .collect()
}

struct TopologyMaps {
    edges: HashMap<(u32, u32), usize>,
    edge_corners: Vec<Vec<usize>>,
    face_corners: Vec<Vec<usize>>,
    vertex_corners: Vec<Vec<usize>>,
    vertex_edges: Vec<Vec<usize>>,
    corner_context: Vec<(usize, usize, usize, usize)>,
}

impl TopologyMaps {
    fn new(mesh: &Mesh) -> Result<Self> {
        let edges = edge_indices(mesh);
        let vertex_indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<HashMap<_, _>>();
        let mut edge_corners = vec![Vec::new(); mesh.edges.len()];
        let mut face_corners = Vec::with_capacity(mesh.faces.len());
        let mut vertex_corners = vec![Vec::new(); mesh.vertices.len()];
        let mut vertex_edges = vec![Vec::new(); mesh.vertices.len()];
        let mut corner_context = Vec::new();
        for (edge_index, edge) in mesh.edges.iter().enumerate() {
            for vertex_id in edge.vertices {
                let vertex_index = *vertex_indices
                    .get(&vertex_id)
                    .ok_or_else(|| evaluation("mesh edge references a missing vertex"))?;
                vertex_edges[vertex_index].push(edge_index);
            }
        }
        let mut corner_index = 0;
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let mut corners = Vec::with_capacity(face.vertices.len());
            for local_index in 0..face.vertices.len() {
                let vertex_id = face.vertices[local_index];
                let vertex_index = *vertex_indices
                    .get(&vertex_id)
                    .ok_or_else(|| evaluation("mesh face references a missing vertex"))?;
                let edge_index = *edges
                    .get(&edge_key(
                        vertex_id,
                        face.vertices[(local_index + 1) % face.vertices.len()],
                    ))
                    .ok_or_else(|| evaluation("mesh face boundary edge is missing"))?;
                corners.push(corner_index);
                vertex_corners[vertex_index].push(corner_index);
                edge_corners[edge_index].push(corner_index);
                corner_context.push((face_index, local_index, vertex_index, edge_index));
                corner_index += 1;
            }
            face_corners.push(corners);
        }
        Ok(Self {
            edges,
            edge_corners,
            face_corners,
            vertex_corners,
            vertex_edges,
            corner_context,
        })
    }
}

fn input_domain(node_type: &str) -> Result<&'static str> {
    match node_type {
        "GeometryNodeInputMeshEdgeAngle"
        | "GeometryNodeInputMeshEdgeNeighbors"
        | "GeometryNodeInputMeshEdgeVertices" => Ok("EDGE"),
        "GeometryNodeInputMeshFaceArea" | "GeometryNodeInputMeshFaceNeighbors" => Ok("FACE"),
        "GeometryNodeInputMeshVertexNeighbors" => Ok("POINT"),
        _ => Err(evaluation("unknown mesh topology input node")),
    }
}

fn mesh_input_zero(node_type: &str, output: &str) -> Value {
    match (node_type, output) {
        ("GeometryNodeInputMeshEdgeVertices", "Position 1" | "Position 2") => {
            json!([0.0, 0.0, 0.0])
        }
        ("GeometryNodeInputMeshEdgeAngle" | "GeometryNodeInputMeshFaceArea", _) => json!(0.0),
        _ => json!(0),
    }
}

struct VertexPositions {
    indices: HashMap<u32, usize>,
    values: Vec<Vec3>,
}

impl VertexPositions {
    fn new(mesh: &Mesh, with_positions: bool) -> Self {
        Self {
            indices: mesh
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect(),
            values: if with_positions {
                mesh.vertices
                    .iter()
                    .map(|vertex| vertex.co.as_vec3())
                    .collect()
            } else {
                Vec::new()
            },
        }
    }

    fn get(&self, vertex_id: u32) -> Result<Vec3> {
        self.indices
            .get(&vertex_id)
            .and_then(|index| self.values.get(*index))
            .copied()
            .ok_or_else(|| evaluation("mesh topology field references a missing vertex"))
    }
}

fn mesh_input_native_values(mesh: &Mesh, node_type: &str, output: &str) -> Result<Vec<Value>> {
    let edge_faces = edge_face_uses(mesh);
    let with_positions = matches!(
        node_type,
        "GeometryNodeInputMeshEdgeAngle" | "GeometryNodeInputMeshFaceArea"
    ) || (node_type == "GeometryNodeInputMeshEdgeVertices"
        && matches!(output, "Position 1" | "Position 2"));
    let positions = VertexPositions::new(mesh, with_positions);
    let vertex_indices = &positions.indices;
    let edge_indices = edge_indices(mesh);
    let face_edges = mesh_face_edges(mesh, &edge_indices);
    let triangle_others =
        if node_type == "GeometryNodeInputMeshEdgeAngle" && output == "Signed Angle" {
            let mut triangle_others = vec![HashMap::<(u32, u32), u32>::new(); mesh.faces.len()];
            mesh.for_each_triangle_with_face(|face, [first, second, third]| {
                let edges = [
                    (first, second, third),
                    (second, third, first),
                    (third, first, second),
                ];
                for (edge_first, edge_second, other) in edges {
                    triangle_others[face]
                        .entry(edge_key(edge_first, edge_second))
                        .or_insert(other);
                }
            })
            .map_err(|error| evaluation(format!("mesh tessellation failed: {error}")))?;
            Some(triangle_others)
        } else {
            None
        };
    match node_type {
        "GeometryNodeInputMeshEdgeAngle" => {
            let signed = output == "Signed Angle";
            if !signed && output != "Unsigned Angle" {
                return Err(evaluation("unknown mesh edge angle output"));
            }
            mesh.edges
                .iter()
                .enumerate()
                .map(|(edge_index, _)| {
                    let uses = &edge_faces[edge_index];
                    if uses.len() != 2 {
                        return Ok(json!(0.0_f32));
                    }
                    let first = face_normal_with_positions(&positions, &mesh.faces[uses[0]])?;
                    let second = face_normal_with_positions(&positions, &mesh.faces[uses[1]])?;
                    let dot = first.dot(second);
                    let angle = if dot >= 0.0 {
                        2.0 * ((first - second).length() * 0.5).clamp(-1.0, 1.0).asin()
                    } else {
                        std::f32::consts::PI
                            - 2.0 * ((first + second).length() * 0.5).clamp(-1.0, 1.0).asin()
                    };
                    let value = if !signed || angle == 0.0 {
                        angle
                    } else {
                        let edge = &mesh.edges[edge_index];
                        let midpoint = (positions.get(edge.vertices[0])?
                            + positions.get(edge.vertices[1])?)
                            * 0.5;
                        let other = triangle_others
                            .as_ref()
                            .and_then(|faces| faces.get(uses[1]))
                            .and_then(|face| {
                                face.get(&edge_key(edge.vertices[0], edge.vertices[1]))
                            })
                            .copied()
                            .ok_or_else(|| {
                                evaluation("mesh edge angle triangle is missing a third vertex")
                            })?;
                        let tangent = (positions.get(other)? - midpoint).normalize_or_zero();
                        if first.dot(tangent) < 0.0 {
                            angle
                        } else {
                            -angle
                        }
                    };
                    Ok(json!(value))
                })
                .collect()
        }
        "GeometryNodeInputMeshEdgeNeighbors" => {
            if output != "Face Count" {
                return Err(evaluation("unknown mesh edge neighbor output"));
            }
            Ok(edge_faces.iter().map(|uses| json!(uses.len())).collect())
        }
        "GeometryNodeInputMeshEdgeVertices" => mesh
            .edges
            .iter()
            .map(|edge| {
                let first = *vertex_indices
                    .get(&edge.vertices[0])
                    .ok_or_else(|| evaluation("mesh edge references a missing first vertex"))?;
                let second = *vertex_indices
                    .get(&edge.vertices[1])
                    .ok_or_else(|| evaluation("mesh edge references a missing second vertex"))?;
                Ok(match output {
                    "Vertex Index 1" => json!(first),
                    "Vertex Index 2" => json!(second),
                    "Position 1" => vector_value(positions.get(edge.vertices[0])?),
                    "Position 2" => vector_value(positions.get(edge.vertices[1])?),
                    _ => return Err(evaluation("unknown mesh edge vertices output")),
                })
            })
            .collect(),
        "GeometryNodeInputMeshFaceArea" => {
            if output != "Area" {
                return Err(evaluation("unknown mesh face area output"));
            }
            mesh.faces
                .iter()
                .map(|face| Ok(json!(face_area(&positions, face)?)))
                .collect()
        }
        "GeometryNodeInputMeshFaceNeighbors" => {
            let neighbors = face_neighbors(mesh, &face_edges, &edge_faces);
            mesh.faces
                .iter()
                .enumerate()
                .map(|(index, face)| match output {
                    "Vertex Count" => Ok(json!(face.vertices.len())),
                    "Face Count" => Ok(json!(neighbors[index].len())),
                    _ => Err(evaluation("unknown mesh face neighbor output")),
                })
                .collect()
        }
        "GeometryNodeInputMeshVertexNeighbors" => {
            let mut edge_count = vec![0_usize; mesh.vertices.len()];
            let mut face_count = vec![0_usize; mesh.vertices.len()];
            for edge in &mesh.edges {
                for id in edge.vertices {
                    let index = *vertex_indices
                        .get(&id)
                        .ok_or_else(|| evaluation("mesh edge references a missing vertex"))?;
                    edge_count[index] += 1;
                }
            }
            for face in &mesh.faces {
                for id in &face.vertices {
                    let index = *vertex_indices
                        .get(id)
                        .ok_or_else(|| evaluation("mesh face references a missing vertex"))?;
                    face_count[index] += 1;
                }
            }
            let values = match output {
                "Vertex Count" => edge_count,
                "Face Count" => face_count,
                _ => return Err(evaluation("unknown mesh vertex neighbor output")),
            };
            Ok(values.into_iter().map(|count| json!(count)).collect())
        }
        _ => Err(evaluation("unknown mesh topology input node")),
    }
}

fn topology_value(
    mesh: &Mesh,
    maps: &TopologyMaps,
    node_type: &str,
    output: &str,
    inputs: &[Field],
    index: usize,
    domain: &str,
) -> Result<Value> {
    match node_type {
        "GeometryNodeCornersOfEdge" => {
            let edge_index = field_index(inputs, 0, mesh, index, domain)?;
            let sort_index = field_index(inputs, 2, mesh, index, domain)?;
            let Some(edge_index) = usize::try_from(edge_index)
                .ok()
                .filter(|index| *index < maps.edge_corners.len())
            else {
                return Ok(json!(0));
            };
            topology_multi_output(
                output,
                &maps.edge_corners[edge_index],
                sort_index,
                inputs.get(1),
                "CORNER",
                mesh,
            )
        }
        "GeometryNodeCornersOfFace" => {
            let face_index = field_index(inputs, 0, mesh, index, domain)?;
            let sort_index = field_index(inputs, 2, mesh, index, domain)?;
            let Some(face_index) = usize::try_from(face_index)
                .ok()
                .filter(|index| *index < maps.face_corners.len())
            else {
                return Ok(json!(0));
            };
            topology_multi_output(
                output,
                &maps.face_corners[face_index],
                sort_index,
                inputs.get(1),
                "CORNER",
                mesh,
            )
        }
        "GeometryNodeCornersOfVertex" | "GeometryNodeEdgesOfVertex" => {
            let vertex_index = field_index(inputs, 0, mesh, index, domain)?;
            let sort_index = field_index(inputs, 2, mesh, index, domain)?;
            let Some(vertex_index) = usize::try_from(vertex_index)
                .ok()
                .filter(|index| *index < mesh.vertices.len())
            else {
                return Ok(json!(0));
            };
            let (candidates, weights_domain) = if node_type == "GeometryNodeCornersOfVertex" {
                (&maps.vertex_corners[vertex_index], "CORNER")
            } else {
                (&maps.vertex_edges[vertex_index], "EDGE")
            };
            topology_multi_output(
                output,
                candidates,
                sort_index,
                inputs.get(1),
                weights_domain,
                mesh,
            )
        }
        "GeometryNodeEdgesOfCorner" => {
            let corner_index = field_index(inputs, 0, mesh, index, domain)?;
            let Some((face_index, local_index, _, edge_index)) = usize::try_from(corner_index)
                .ok()
                .and_then(|index| maps.corner_context.get(index))
                .copied()
            else {
                return Ok(json!(0));
            };
            let face = &mesh.faces[face_index];
            let previous = maps
                .edges
                .get(&edge_key(
                    face.vertices[(local_index + face.vertices.len() - 1) % face.vertices.len()],
                    face.vertices[local_index],
                ))
                .ok_or_else(|| evaluation("corner references a missing previous edge"))?;
            Ok(match output {
                "Next Edge Index" => json!(edge_index),
                "Previous Edge Index" => json!(previous),
                _ => return Err(evaluation("unknown Edges of Corner output")),
            })
        }
        "GeometryNodeFaceOfCorner" => {
            let corner_index = field_index(inputs, 0, mesh, index, domain)?;
            let Some((face_index, local_index, _, _)) = usize::try_from(corner_index)
                .ok()
                .and_then(|index| maps.corner_context.get(index))
                .copied()
            else {
                return Ok(json!(0));
            };
            Ok(match output {
                "Face Index" => json!(face_index),
                "Index in Face" => json!(local_index),
                _ => return Err(evaluation("unknown Face of Corner output")),
            })
        }
        "GeometryNodeOffsetCornerInFace" => {
            let corner_index = field_index(inputs, 0, mesh, index, domain)?;
            let offset = field_index(inputs, 1, mesh, index, domain)?;
            let Some((face_index, local_index, _, _)) = usize::try_from(corner_index)
                .ok()
                .and_then(|index| maps.corner_context.get(index))
                .copied()
            else {
                return Ok(json!(0));
            };
            let corners = &maps.face_corners[face_index];
            let length = i64::try_from(corners.len())
                .map_err(|_| evaluation("face corner count exceeds the supported range"))?;
            let offset = (i64::try_from(local_index)
                .map_err(|_| evaluation("corner index exceeds the supported range"))?
                + offset)
                .rem_euclid(length);
            Ok(json!(
                corners[usize::try_from(offset)
                    .map_err(|_| evaluation("corner offset is out of range"))?]
            ))
        }
        "GeometryNodeVertexOfCorner" => {
            let corner_index = field_index(inputs, 0, mesh, index, domain)?;
            let Some((_, _, vertex_index, _)) = usize::try_from(corner_index)
                .ok()
                .and_then(|index| maps.corner_context.get(index))
                .copied()
            else {
                return Ok(json!(0));
            };
            Ok(json!(vertex_index))
        }
        _ => Err(evaluation("unknown mesh topology function")),
    }
}

fn topology_multi_output(
    output: &str,
    candidates: &[usize],
    sort_index: i64,
    weights: Option<&Field>,
    weights_domain: &str,
    mesh: &Mesh,
) -> Result<Value> {
    if output == "Total" {
        return Ok(json!(candidates.len()));
    }
    if output != "Corner Index" && output != "Edge Index" {
        return Err(evaluation("unknown topology list output"));
    }
    if candidates.is_empty() {
        return Ok(json!(0));
    }
    let mut ordered = candidates.to_vec();
    if let Some(weights) = weights.filter(|field| !matches!(field, Field::Constant(_))) {
        let mut weighted = Vec::with_capacity(ordered.len());
        for candidate in ordered {
            let value = field_json(weights, mesh, candidate, weights_domain)?
                .as_f64()
                .ok_or_else(|| evaluation("topology sorting weight is not numeric"))?;
            weighted.push((candidate, value));
        }
        weighted.sort_by(|left, right| left.1.total_cmp(&right.1));
        ordered = weighted
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect();
    }
    let count = i64::try_from(ordered.len())
        .map_err(|_| evaluation("topology candidate count exceeds the supported range"))?;
    let chosen = usize::try_from(sort_index.rem_euclid(count))
        .map_err(|_| evaluation("topology sort index is out of range"))?;
    Ok(json!(ordered[chosen]))
}

fn field_index(
    inputs: &[Field],
    input: usize,
    mesh: &Mesh,
    index: usize,
    domain: &str,
) -> Result<i64> {
    inputs
        .get(input)
        .ok_or_else(|| evaluation("topology input field is missing"))
        .and_then(|field| field_json(field, mesh, index, domain))?
        .as_i64()
        .ok_or_else(|| evaluation("topology index field is not an integer"))
}

fn edge_face_uses(mesh: &Mesh) -> Vec<Vec<usize>> {
    let edges = edge_indices(mesh);
    let mut uses = vec![Vec::new(); mesh.edges.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for (corner, first) in face.vertices.iter().copied().enumerate() {
            let second = face.vertices[(corner + 1) % face.vertices.len()];
            if let Some(&edge) = edges.get(&edge_key(first, second)) {
                uses[edge].push(face_index);
            }
        }
    }
    uses
}

fn mesh_face_edges(mesh: &Mesh, edge_indices: &HashMap<(u32, u32), usize>) -> Vec<Vec<usize>> {
    mesh.faces
        .iter()
        .map(|face| {
            (0..face.vertices.len())
                .filter_map(|corner| {
                    edge_indices
                        .get(&edge_key(
                            face.vertices[corner],
                            face.vertices[(corner + 1) % face.vertices.len()],
                        ))
                        .copied()
                })
                .collect()
        })
        .collect()
}

fn edge_indices(mesh: &Mesh) -> HashMap<(u32, u32), usize> {
    mesh.edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
        .collect()
}

fn face_neighbors(
    mesh: &Mesh,
    face_edges: &[Vec<usize>],
    edge_faces: &[Vec<usize>],
) -> Vec<HashSet<usize>> {
    let mut result = vec![HashSet::new(); mesh.faces.len()];
    for (face_index, edges) in face_edges.iter().enumerate() {
        for &edge in edges {
            for &other in &edge_faces[edge] {
                if other != face_index {
                    result[face_index].insert(other);
                }
            }
        }
    }
    result
}

fn domain_adaptation_map(mesh: &Mesh, source: &str, destination: &str) -> Result<Vec<Vec<usize>>> {
    #[cfg(test)]
    ADAPTATION_MAP_BUILDS.with(|builds| builds.set(builds.get() + 1));
    let source = canonical_domain(source)?;
    let destination = canonical_domain(destination)?;
    let mut mapping = vec![Vec::new(); domain_element_count(mesh, destination)?];
    if source == destination {
        for (index, sources) in mapping.iter_mut().enumerate() {
            sources.push(index);
        }
        return Ok(mapping);
    }
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let edges = edge_indices(mesh);
    let face_edges = mesh_face_edges(mesh, &edges);
    match (source, destination) {
        ("POINT", "EDGE") => {
            for (edge_index, edge) in mesh.edges.iter().enumerate() {
                mapping[edge_index].extend(
                    edge.vertices
                        .iter()
                        .filter_map(|id| vertex_indices.get(id).copied()),
                );
            }
        }
        ("POINT", "FACE") => {
            for (face_index, face) in mesh.faces.iter().enumerate() {
                mapping[face_index].extend(
                    face.vertices
                        .iter()
                        .filter_map(|id| vertex_indices.get(id).copied()),
                );
            }
        }
        ("POINT", "CORNER") => {
            for (corner, (_, _, vertex_id, _)) in
                corner_layout(mesh, &edges).into_iter().enumerate()
            {
                mapping[corner].push(
                    *vertex_indices
                        .get(&vertex_id)
                        .ok_or_else(|| evaluation("corner references a missing vertex"))?,
                );
            }
        }
        ("EDGE", "POINT") => {
            for (edge_index, edge) in mesh.edges.iter().enumerate() {
                for id in edge.vertices {
                    mapping[*vertex_indices
                        .get(&id)
                        .ok_or_else(|| evaluation("edge references a missing vertex"))?]
                    .push(edge_index);
                }
            }
        }
        ("EDGE", "FACE") => {
            for (face_index, face_edges) in face_edges.iter().enumerate() {
                mapping[face_index].extend(face_edges.iter().copied());
            }
        }
        ("EDGE", "CORNER") => {
            for (corner, (face_index, local, vertex_id, outgoing_edge)) in
                corner_layout(mesh, &edges).into_iter().enumerate()
            {
                let face = &mesh.faces[face_index];
                let previous =
                    face.vertices[(local + face.vertices.len() - 1) % face.vertices.len()];
                let incoming_edge = *edges
                    .get(&edge_key(previous, vertex_id))
                    .ok_or_else(|| evaluation("corner incoming edge is missing"))?;
                mapping[corner].extend([incoming_edge, outgoing_edge]);
            }
        }
        ("FACE", "POINT") => {
            for (face_index, face) in mesh.faces.iter().enumerate() {
                for id in &face.vertices {
                    mapping[*vertex_indices
                        .get(id)
                        .ok_or_else(|| evaluation("face references a missing vertex"))?]
                    .push(face_index);
                }
            }
        }
        ("FACE", "EDGE") => {
            for (edge_index, faces) in edge_face_uses(mesh).iter().enumerate() {
                mapping[edge_index].extend(faces.iter().copied());
            }
        }
        ("FACE", "CORNER") => {
            for (corner, (face_index, _, _, _)) in
                corner_layout(mesh, &edges).into_iter().enumerate()
            {
                mapping[corner].push(face_index);
            }
        }
        ("CORNER", "POINT" | "FACE") => {
            for (corner, (face_index, _, vertex_id, _)) in
                corner_layout(mesh, &edges).into_iter().enumerate()
            {
                let destination_index = match destination {
                    "POINT" => *vertex_indices
                        .get(&vertex_id)
                        .ok_or_else(|| evaluation("corner references a missing vertex"))?,
                    "FACE" => face_index,
                    _ => return Err(evaluation("unsupported corner field destination domain")),
                };
                mapping[destination_index].push(corner);
            }
        }
        ("CORNER", "EDGE") => {
            let mut first_corner = 0;
            for face in &mesh.faces {
                for local in 0..face.vertices.len() {
                    let next = (local + 1) % face.vertices.len();
                    if let Some(&edge_index) =
                        edges.get(&edge_key(face.vertices[local], face.vertices[next]))
                    {
                        mapping[edge_index].extend([first_corner + local, first_corner + next]);
                    }
                }
                first_corner += face.vertices.len();
            }
        }
        _ => return Err(evaluation("unsupported mesh topology domain adaptation")),
    }
    Ok(mapping)
}

pub(super) fn adapt_mesh_field_values(
    values: &[Value],
    source_domain: &str,
    destination_domain: &str,
    mesh: &Mesh,
    empty_value: &Value,
) -> Result<Vec<Value>> {
    let source_domain = canonical_domain(source_domain)?;
    let destination_domain = canonical_domain(destination_domain)?;
    if source_domain == destination_domain {
        let count = domain_element_count(mesh, destination_domain)?;
        if values.len() != count {
            return Err(evaluation("captured mesh field has an invalid length"));
        }
        return Ok(values.to_vec());
    }
    let mapping = domain_adaptation_map(mesh, source_domain, destination_domain)?;
    mapping
        .iter()
        .map(|indices| {
            mix_adapted_values(
                values,
                indices,
                source_domain,
                destination_domain,
                empty_value,
            )
        })
        .collect()
}

fn corner_layout(
    mesh: &Mesh,
    edges: &HashMap<(u32, u32), usize>,
) -> Vec<(usize, usize, u32, usize)> {
    let mut result = Vec::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for local in 0..face.vertices.len() {
            let first = face.vertices[local];
            let second = face.vertices[(local + 1) % face.vertices.len()];
            if let Some(&edge_index) = edges.get(&edge_key(first, second)) {
                result.push((face_index, local, first, edge_index));
            }
        }
    }
    result
}

fn mix_adapted_values(
    values: &[Value],
    indices: &[usize],
    source_domain: &str,
    destination_domain: &str,
    empty_value: &Value,
) -> Result<Value> {
    if indices.is_empty() {
        return Ok(typed_zero(values.first().unwrap_or(empty_value)));
    }
    let values = indices
        .iter()
        .map(|index| {
            values
                .get(*index)
                .ok_or_else(|| evaluation("adapted mesh field index is out of range"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values[0].is_boolean() {
        if values.iter().any(|value| !value.is_boolean()) {
            return Err(evaluation("adapted boolean field value is invalid"));
        }
        let requires_all = matches!(
            (source_domain, destination_domain),
            ("POINT", "EDGE" | "FACE")
                | ("EDGE", "CORNER" | "FACE")
                | ("CORNER", "POINT" | "EDGE" | "FACE")
        );
        return Ok(json!(if requires_all {
            values.iter().all(|value| value.as_bool() == Some(true))
        } else {
            values.iter().any(|value| value.as_bool() == Some(true))
        }));
    }

    let weight = 1.0 / count_f32(indices.len())?;
    if let Some(value) = values[0].as_array()
        && value.len() == 3
    {
        let mut average = Vec3::ZERO;
        for value in values {
            let vector = value
                .as_array()
                .filter(|parts| parts.len() == 3)
                .ok_or_else(|| evaluation("adapted vector field value is invalid"))?;
            average += Vec3::new(
                value_f32(&vector[0], "adapted vector component")?,
                value_f32(&vector[1], "adapted vector component")?,
                value_f32(&vector[2], "adapted vector component")?,
            ) * weight;
        }
        return Ok(vector_value(average));
    }
    if values[0].is_i64() || values[0].is_u64() {
        let weight = f64::from(weight);
        let average = values.iter().try_fold(0.0_f64, |sum, value| {
            Ok::<_, crate::error::PotError>(
                sum + value
                    .as_f64()
                    .ok_or_else(|| evaluation("adapted integer field value is invalid"))?
                    * weight,
            )
        })?;
        let value = average
            .round()
            .to_i64()
            .ok_or_else(|| evaluation("adapted integer field value is out of range"))?;
        Ok(json!(value))
    } else {
        let mut average = 0.0_f32;
        for value in values {
            average += value_f32(value, "adapted mesh field value")? * weight;
        }
        Ok(json!(average))
    }
}

fn typed_zero(value: &Value) -> Value {
    match value {
        Value::Null => json!(0),
        Value::Bool(_) => json!(false),
        Value::Number(number) if number.is_i64() || number.is_u64() => json!(0),
        Value::Number(_) => json!(0.0),
        Value::String(_) => json!(""),
        Value::Array(values) => Value::Array(values.iter().map(typed_zero).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), typed_zero(value)))
                .collect(),
        ),
    }
}

pub(super) fn face_normal(mesh: &Mesh, face: &Face) -> Result<Vec3> {
    face_normal_with_positions(&VertexPositions::new(mesh, true), face)
}

fn face_normal_with_positions(positions: &VertexPositions, face: &Face) -> Result<Vec3> {
    let normal = match face.vertices.as_slice() {
        [first, second, third] => (positions.get(*first)? - positions.get(*second)?)
            .cross(positions.get(*second)? - positions.get(*third)?),
        [first, second, third, fourth] => (positions.get(*first)? - positions.get(*third)?)
            .cross(positions.get(*second)? - positions.get(*fourth)?),
        vertices if vertices.len() > 4 => {
            let mut normal = Vec3::ZERO;
            let previous = vertices
                .last()
                .ok_or_else(|| evaluation("mesh face has fewer than three vertices"))?;
            let mut previous = positions.get(*previous)?;
            for vertex in vertices {
                let current = positions.get(*vertex)?;
                normal.x += (previous.y - current.y) * (previous.z + current.z);
                normal.y += (previous.z - current.z) * (previous.x + current.x);
                normal.z += (previous.x - current.x) * (previous.y + current.y);
                previous = current;
            }
            normal
        }
        _ => return Err(evaluation("mesh face has fewer than three vertices")),
    };
    Ok(if normal == Vec3::ZERO {
        Vec3::Z
    } else {
        normal.normalize()
    })
}

fn face_area(positions: &VertexPositions, face: &Face) -> Result<f32> {
    match face.vertices.as_slice() {
        [first, second, third] => {
            let first = positions.get(*first)? - positions.get(*second)?;
            let second = positions.get(*second)? - positions.get(*third)?;
            Ok(first.cross(second).length() * 0.5)
        }
        vertices if vertices.len() > 3 => {
            let mut normal = Vec3::ZERO;
            let previous = vertices
                .last()
                .ok_or_else(|| evaluation("mesh face has fewer than three vertices"))?;
            let mut previous = positions.get(*previous)?;
            for vertex in vertices {
                let current = positions.get(*vertex)?;
                normal.x += (previous.y - current.y) * (previous.z + current.z);
                normal.y += (previous.z - current.z) * (previous.x + current.x);
                normal.z += (previous.x - current.x) * (previous.y + current.y);
                previous = current;
            }
            Ok(normal.length() * 0.5)
        }
        _ => Ok(0.0),
    }
}

fn vector_value(value: Vec3) -> Value {
    json!([value.x, value.y, value.z])
}

fn canonical_domain(domain: &str) -> Result<&'static str> {
    match domain {
        "POINT" | "POINTS" => Ok("POINT"),
        "EDGE" => Ok("EDGE"),
        "FACE" => Ok("FACE"),
        "CORNER" => Ok("CORNER"),
        _ => Err(evaluation(
            "mesh topology fields require POINT, EDGE, FACE, or CORNER domain",
        )),
    }
}

fn value_f32(value: &Value, context: &str) -> Result<f32> {
    value
        .as_f64()
        .and_then(|number| number.to_f32())
        .ok_or_else(|| evaluation(format!("{context} is not representable as a Blender float")))
}

fn count_f32(count: usize) -> Result<f32> {
    count
        .to_f32()
        .ok_or_else(|| evaluation("mesh topology count exceeds Blender's float range"))
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };

    use glam::DVec3;
    use serde_json::json;

    use crate::geom::{Edge, Vertex};

    use super::*;

    fn make_mesh(points: &[[f64; 3]], loose_edges: &[[u32; 2]], faces: &[&[u32]]) -> Mesh {
        let mut edge_keys = HashSet::new();
        let mut edge_vertices = Vec::new();
        let mut add_edge = |vertices: [u32; 2]| {
            if edge_keys.insert(edge_key(vertices[0], vertices[1])) {
                edge_vertices.push(vertices);
            }
        };
        for face in faces {
            for corner in 0..face.len() {
                add_edge([face[corner], face[(corner + 1) % face.len()]]);
            }
        }
        for edge in loose_edges {
            add_edge(*edge);
        }
        Mesh {
            vertices: points
                .iter()
                .enumerate()
                .map(|(index, point)| Vertex {
                    id: index as u32,
                    co: DVec3::new(point[0], point[1], point[2]),
                })
                .collect(),
            edges: edge_vertices
                .into_iter()
                .enumerate()
                .map(|(index, vertices)| Edge {
                    id: index as u32,
                    vertices,
                })
                .collect(),
            faces: faces
                .iter()
                .enumerate()
                .map(|(index, vertices)| Face {
                    id: index as u32,
                    vertices: vertices.to_vec(),
                    material_index: 0,
                })
                .collect(),
            ..Mesh::default()
        }
    }

    fn set_context(mesh: &mut Mesh, context_id: u64) {
        mesh.attributes.insert(
            ".potter_gn_evaluation_context".to_owned(),
            json!(context_id),
        );
    }

    fn triangle() -> Mesh {
        make_mesh(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[],
            &[&[0, 1, 2]],
        )
    }

    #[test]
    fn empty_adaptation_returns_typed_vector_and_boolean_zeros() -> Result<()> {
        let mesh = make_mesh(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 4.0, 3.0]],
            &[[0, 1]],
            &[],
        );
        let positions = mesh_input_field_values(
            &mesh,
            "GeometryNodeInputMeshEdgeVertices",
            "Position 1",
            "POINT",
        )?;
        assert_eq!(positions, vec![json!([0.0, 0.0, 0.0]); 3]);

        let booleans =
            adapt_mesh_field_values(&[json!(true)], "EDGE", "POINT", &mesh, &json!(false))?;
        assert_eq!(booleans, vec![json!(true), json!(true), json!(false)]);
        Ok(())
    }

    #[test]
    fn boolean_domain_reductions_match_blender() -> Result<()> {
        let mesh = triangle();
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(false), json!(true)],
                "POINT",
                "EDGE",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false), json!(false), json!(true)]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(true), json!(false)],
                "POINT",
                "FACE",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false)]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(true), json!(false)],
                "EDGE",
                "CORNER",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false), json!(true), json!(false)]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(true), json!(false)],
                "EDGE",
                "FACE",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false)]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(false), json!(true)],
                "CORNER",
                "EDGE",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false), json!(false), json!(true)]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(false), json!(true)],
                "CORNER",
                "FACE",
                &mesh,
                &json!(false),
            )?,
            vec![json!(false)]
        );
        assert_eq!(
            adapt_mesh_field_values(&[json!(true)], "FACE", "POINT", &mesh, &json!(false),)?,
            vec![json!(true); 3]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(false), json!(false)],
                "EDGE",
                "POINT",
                &mesh,
                &json!(false),
            )?,
            vec![json!(true), json!(true), json!(false)]
        );
        assert_eq!(
            adapt_mesh_field_values(&[json!(true)], "FACE", "EDGE", &mesh, &json!(false),)?,
            vec![json!(true); 3]
        );
        assert_eq!(
            adapt_mesh_field_values(&[json!(true)], "FACE", "CORNER", &mesh, &json!(false),)?,
            vec![json!(true); 3]
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[json!(true), json!(false), json!(true)],
                "POINT",
                "CORNER",
                &mesh,
                &json!(false),
            )?,
            vec![json!(true), json!(false), json!(true)]
        );

        let fan = make_mesh(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            &[],
            &[&[0, 1, 2], &[0, 2, 3]],
        );
        assert_eq!(
            adapt_mesh_field_values(
                &[
                    json!(true),
                    json!(true),
                    json!(true),
                    json!(false),
                    json!(true),
                    json!(true),
                ],
                "CORNER",
                "POINT",
                &fan,
                &json!(false),
            )?[0],
            json!(false)
        );
        Ok(())
    }

    #[test]
    fn corner_to_edge_mixes_both_endpoint_corners() -> Result<()> {
        let mesh = triangle();
        let adapted = adapt_mesh_field_values(
            &[json!(0), json!(1), json!(2)],
            "CORNER",
            "EDGE",
            &mesh,
            &json!(0),
        )?;
        assert_eq!(adapted[2], json!(1));

        let averages = adapt_mesh_field_values(
            &[json!(0.0), json!(1.0), json!(2.0)],
            "CORNER",
            "EDGE",
            &mesh,
            &json!(0.0),
        )?;
        assert_eq!(averages, vec![json!(0.5), json!(1.5), json!(1.0)]);

        let fan = make_mesh(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            &[],
            &[&[0, 1, 2], &[0, 2, 3]],
        );
        let shared_edge_values = adapt_mesh_field_values(
            &[
                json!(0.0),
                json!(1.0),
                json!(2.0),
                json!(3.0),
                json!(4.0),
                json!(5.0),
            ],
            "CORNER",
            "EDGE",
            &fan,
            &json!(0.0),
        )?;
        assert_eq!(shared_edge_values[2], json!(2.25));
        Ok(())
    }

    #[test]
    fn face_area_and_zero_area_normals_match_blender() -> Result<()> {
        let translated = make_mesh(
            &[
                [1_048_576.0, 1_048_576.0, 0.0],
                [1_048_577.0, 1_048_576.0, 0.0],
                [1_048_576.0, 1_048_577.0, 0.0],
            ],
            &[],
            &[&[0, 1, 2]],
        );
        assert_eq!(
            mesh_input_field_values(&translated, "GeometryNodeInputMeshFaceArea", "Area", "FACE",)?,
            vec![json!(0.5)]
        );

        let polygon = make_mesh(
            &[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            &[],
            &[&[0, 1, 2, 3]],
        );
        assert_eq!(
            mesh_input_field_values(&polygon, "GeometryNodeInputMeshFaceArea", "Area", "FACE",)?,
            vec![json!(2.0)]
        );

        let collinear = make_mesh(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            &[],
            &[&[0, 1, 2], &[1, 0, 3]],
        );
        assert_eq!(face_normal(&collinear, &collinear.faces[0])?, Vec3::Z);
        assert_eq!(
            mesh_input_field_values(
                &collinear,
                "GeometryNodeInputMeshEdgeAngle",
                "Unsigned Angle",
                "EDGE",
            )?[0],
            json!(0.0)
        );
        Ok(())
    }

    #[test]
    fn topology_cache_distinguishes_reused_mesh_addresses_by_context() -> Result<()> {
        let cache = Rc::new(RefCell::new(HashMap::new()));
        let mut mesh = Box::new(triangle());
        set_context(&mut mesh, 41);
        let address = std::ptr::from_ref(&*mesh);
        let first = cached_mesh_field_value(&cache, &mesh, "FACE", 0, |mesh, domain| {
            mesh_input_field_values(mesh, "GeometryNodeInputMeshFaceArea", "Area", domain)
        })?;

        *mesh = self::make_mesh(
            &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[],
            &[&[0, 1, 2]],
        );
        set_context(&mut mesh, 42);
        assert_eq!(std::ptr::from_ref(&*mesh), address);
        let second = cached_mesh_field_value(&cache, &mesh, "FACE", 0, |mesh, domain| {
            mesh_input_field_values(mesh, "GeometryNodeInputMeshFaceArea", "Area", domain)
        })?;

        assert_eq!(first, json!(0.5));
        assert_eq!(second, json!(1.0));
        Ok(())
    }

    #[test]
    fn batch_adaptation_builds_one_map_for_all_faces() -> Result<()> {
        const FACE_COUNT: usize = 64;
        let points = (0..FACE_COUNT * 3)
            .map(|index| [index as f64, 0.0, 0.0])
            .collect::<Vec<_>>();
        let faces = (0..FACE_COUNT)
            .map(|index| {
                [
                    (index * 3) as u32,
                    (index * 3 + 1) as u32,
                    (index * 3 + 2) as u32,
                ]
            })
            .collect::<Vec<_>>();
        let face_refs = faces.iter().map(<[u32; 3]>::as_slice).collect::<Vec<_>>();
        let mesh = make_mesh(&points, &[], &face_refs);
        let values = (0..FACE_COUNT)
            .map(|index| json!(index))
            .collect::<Vec<_>>();
        ADAPTATION_MAP_BUILDS.with(|builds| builds.set(0));

        let adapted = adapt_mesh_field_values(&values, "FACE", "CORNER", &mesh, &json!(0))?;

        assert_eq!(adapted.len(), FACE_COUNT * 3);
        ADAPTATION_MAP_BUILDS.with(|builds| assert_eq!(builds.get(), 1));

        let mut mesh = make_mesh(&points, &[], &face_refs);
        set_context(&mut mesh, 73);
        let cache = Rc::new(RefCell::new(HashMap::new()));
        let computations = Cell::new(0);
        for index in 0..FACE_COUNT {
            let value = cached_mesh_field_value(&cache, &mesh, "FACE", index, |mesh, _| {
                computations.set(computations.get() + 1);
                Ok((0..mesh.faces.len()).map(|face| json!(face)).collect())
            })?;
            assert_eq!(value, json!(index));
        }
        assert_eq!(computations.get(), 1);
        Ok(())
    }
}
