use std::collections::{HashMap, HashSet};

use glam::{DVec3, Vec3};
use num_traits::ToPrimitive;

use crate::{
    error::Result,
    geom::{Mesh, edge_key},
};

use super::{evaluation, mesh_error};

#[path = "mesh_dual_attributes.rs"]
mod mesh_dual_attributes;

struct DualFace {
    vertices: Vec<u32>,
    source_corners: Vec<usize>,
    source_edges: Vec<usize>,
}

struct OrderedFaces {
    faces: Vec<usize>,
    corners: Vec<usize>,
    outgoing_edges: Vec<usize>,
    start_boundary: Option<usize>,
    end_boundary: Option<usize>,
}

pub(super) fn evaluate(source: &Mesh, keep_boundaries: bool) -> Result<Mesh> {
    let topology = DualTopology::new(source, keep_boundaries)?;
    let points = DualPoints::new(source, &topology, keep_boundaries)?;
    let faces = DualFaces::new(&topology, &points, keep_boundaries)?;
    build_dual_mesh(source, &topology, points, &faces)
}

struct DualTopology {
    vertex_indices: HashMap<u32, usize>,
    edge_faces: Vec<Vec<usize>>,
    vertex_kinds: Vec<VertexKind>,
    ordered: Vec<Option<OrderedFaces>>,
    dissolved_edges: Vec<(usize, usize, usize)>,
}

impl DualTopology {
    fn new(source: &Mesh, keep_boundaries: bool) -> Result<Self> {
        let vertex_indices = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<HashMap<_, _>>();
        let edge_indices = source
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
            .collect::<HashMap<_, _>>();
        let mut edge_faces = vec![Vec::<usize>::new(); source.edges.len()];
        let mut vertex_faces = vec![Vec::<(usize, usize)>::new(); source.vertices.len()];
        let mut corner_offsets = Vec::with_capacity(source.faces.len());
        let mut corner_offset = 0;
        for (face_index, face) in source.faces.iter().enumerate() {
            corner_offsets.push(corner_offset);
            for local in 0..face.vertices.len() {
                let corner = corner_offset + local;
                let vertex_index = *vertex_indices
                    .get(&face.vertices[local])
                    .ok_or_else(|| evaluation("Dual Mesh face references a missing vertex"))?;
                let edge_index = *edge_indices
                    .get(&edge_key(
                        face.vertices[local],
                        face.vertices[(local + 1) % face.vertices.len()],
                    ))
                    .ok_or_else(|| evaluation("Dual Mesh face boundary edge is missing"))?;
                edge_faces[edge_index].push(face_index);
                vertex_faces[vertex_index].push((face_index, corner));
            }
            corner_offset += face.vertices.len();
        }
        let mut boundary_count = vec![0_usize; source.vertices.len()];
        let mut nonmanifold = vec![false; source.vertices.len()];
        for (edge_index, edge) in source.edges.iter().enumerate() {
            for vertex_id in edge.vertices {
                let vertex_index = *vertex_indices
                    .get(&vertex_id)
                    .ok_or_else(|| evaluation("Dual Mesh edge references a missing vertex"))?;
                if edge_faces[edge_index].len() == 1 {
                    boundary_count[vertex_index] += 1;
                } else if edge_faces[edge_index].len() > 2 {
                    nonmanifold[vertex_index] = true;
                }
            }
        }
        let mut vertex_kinds = vec![VertexKind::Unsupported; source.vertices.len()];
        let mut ordered = (0..source.vertices.len())
            .map(|_| None)
            .collect::<Vec<Option<OrderedFaces>>>();
        for vertex_index in 0..source.vertices.len() {
            if nonmanifold[vertex_index]
                || vertex_faces[vertex_index].is_empty()
                || boundary_count[vertex_index] >= 3
            {
                continue;
            }
            let boundary = boundary_count[vertex_index] == 2;
            if boundary_count[vertex_index] == 1 {
                if vertex_faces[vertex_index].len() <= 2 {
                    vertex_kinds[vertex_index] = VertexKind::Normal;
                }
                continue;
            }
            if boundary && !keep_boundaries {
                continue;
            }
            let Some(faces) = ordered_faces(
                source,
                vertex_index,
                &vertex_faces[vertex_index],
                &edge_faces,
                &edge_indices,
                boundary,
                &corner_offsets,
            ) else {
                continue;
            };
            vertex_kinds[vertex_index] = if boundary {
                VertexKind::Boundary
            } else {
                VertexKind::Normal
            };
            ordered[vertex_index] = Some(faces);
        }
        let mut dissolved_edges = Vec::new();
        for (vertex_index, kind) in vertex_kinds.iter().enumerate() {
            if *kind != VertexKind::Normal || vertex_faces[vertex_index].len() != 2 {
                continue;
            }
            let first_face = vertex_faces[vertex_index][0].0;
            let second_face = vertex_faces[vertex_index][1].0;
            let vertex_id = source.vertices[vertex_index].id;
            let face = &source.faces[first_face];
            let source_edge = (0..face.vertices.len())
                .find_map(|local| {
                    let first = face.vertices[local];
                    let second = face.vertices[(local + 1) % face.vertices.len()];
                    (first == vertex_id || second == vertex_id)
                        .then(|| edge_indices.get(&edge_key(first, second)).copied())
                        .flatten()
                })
                .ok_or_else(|| evaluation("Dual Mesh redundant vertex edge is missing"))?;
            dissolved_edges.push((first_face, second_face, source_edge));
        }

        Ok(Self {
            vertex_indices,
            edge_faces,
            vertex_kinds,
            ordered,
            dissolved_edges,
        })
    }
}

struct DualPoints {
    mesh: Mesh,
    face_centers: Vec<u32>,
    boundary_midpoints: HashMap<usize, u32>,
    boundary_vertices: HashMap<usize, u32>,
    point_face_sources: Vec<Option<usize>>,
}

impl DualPoints {
    fn new(source: &Mesh, topology: &DualTopology, keep_boundaries: bool) -> Result<Self> {
        let mut mesh = Mesh::default();
        let mut point_face_sources = Vec::new();
        let mut face_centers = Vec::with_capacity(source.faces.len());
        for (face_index, face) in source.faces.iter().enumerate() {
            let count = face
                .vertices
                .len()
                .to_f32()
                .ok_or_else(|| evaluation("Dual Mesh face vertex count is too large"))?;
            let mut center = Vec3::ZERO;
            for vertex_id in &face.vertices {
                center += source.vertices[topology.vertex_indices[vertex_id]]
                    .co
                    .as_vec3();
            }
            face_centers.push(
                mesh.insert_vertex(DVec3::from(center / count))
                    .map_err(mesh_error)?,
            );
            point_face_sources.push(Some(face_index));
        }
        let mut boundary_midpoints = HashMap::<usize, u32>::new();
        if keep_boundaries {
            for (edge_index, edge) in source.edges.iter().enumerate() {
                if topology.edge_faces[edge_index].len() != 1 {
                    continue;
                }
                let first = topology.vertex_indices[&edge.vertices[0]];
                let second = topology.vertex_indices[&edge.vertices[1]];
                let midpoint = (source.vertices[first].co.as_vec3()
                    + source.vertices[second].co.as_vec3())
                    * 0.5;
                boundary_midpoints.insert(
                    edge_index,
                    mesh.insert_vertex(DVec3::from(midpoint))
                        .map_err(mesh_error)?,
                );
                point_face_sources.push(topology.edge_faces[edge_index].first().copied());
            }
        }
        let mut boundary_vertices = HashMap::<usize, u32>::new();
        if keep_boundaries {
            for (vertex_index, kind) in topology.vertex_kinds.iter().enumerate() {
                if *kind != VertexKind::Boundary {
                    continue;
                }
                boundary_vertices.insert(
                    vertex_index,
                    mesh.insert_vertex(DVec3::from(source.vertices[vertex_index].co.as_vec3()))
                        .map_err(mesh_error)?,
                );
                point_face_sources.push(
                    topology.ordered[vertex_index]
                        .as_ref()
                        .and_then(|faces| faces.faces.last())
                        .copied(),
                );
            }
        }
        Ok(Self {
            mesh,
            face_centers,
            boundary_midpoints,
            boundary_vertices,
            point_face_sources,
        })
    }
}

struct DualFaces {
    items: Vec<DualFace>,
    point_sources: Vec<usize>,
}

impl DualFaces {
    fn new(topology: &DualTopology, points: &DualPoints, keep_boundaries: bool) -> Result<Self> {
        let mut items = Vec::new();
        let mut point_sources = Vec::new();
        for (vertex_index, kind) in topology.vertex_kinds.iter().enumerate() {
            let Some(ordered) = topology.ordered[vertex_index].as_ref() else {
                continue;
            };
            if *kind == VertexKind::Normal {
                if ordered.faces.len() < 3 {
                    continue;
                }
                items.push(DualFace {
                    vertices: ordered
                        .faces
                        .iter()
                        .map(|face| points.face_centers[*face])
                        .collect(),
                    source_corners: ordered.corners.clone(),
                    source_edges: ordered.outgoing_edges.clone(),
                });
                point_sources.push(vertex_index);
            } else if *kind == VertexKind::Boundary && keep_boundaries {
                items.push(boundary_dual_face(vertex_index, ordered, points)?);
                point_sources.push(vertex_index);
            }
        }
        Ok(Self {
            items,
            point_sources,
        })
    }
}

fn boundary_dual_face(
    vertex_index: usize,
    ordered: &OrderedFaces,
    points: &DualPoints,
) -> Result<DualFace> {
    let start_edge = ordered
        .start_boundary
        .ok_or_else(|| evaluation("Dual Mesh boundary vertex has no start boundary edge"))?;
    let end_edge = ordered
        .end_boundary
        .ok_or_else(|| evaluation("Dual Mesh boundary vertex has no end boundary edge"))?;
    let start_midpoint = *points
        .boundary_midpoints
        .get(&start_edge)
        .ok_or_else(|| evaluation("Dual Mesh boundary start midpoint is missing"))?;
    let end_midpoint = *points
        .boundary_midpoints
        .get(&end_edge)
        .ok_or_else(|| evaluation("Dual Mesh boundary end midpoint is missing"))?;
    let boundary_vertex = *points
        .boundary_vertices
        .get(&vertex_index)
        .ok_or_else(|| evaluation("Dual Mesh boundary vertex is missing"))?;
    let first_corner = *ordered
        .corners
        .first()
        .ok_or_else(|| evaluation("Dual Mesh boundary face has no source corners"))?;
    let last_corner = *ordered
        .corners
        .last()
        .ok_or_else(|| evaluation("Dual Mesh boundary face has no source corners"))?;
    let mut vertices = ordered
        .faces
        .iter()
        .map(|face| points.face_centers[*face])
        .collect::<Vec<_>>();
    vertices.extend([end_midpoint, boundary_vertex, start_midpoint]);
    let mut source_corners = ordered.corners.clone();
    source_corners.extend([last_corner, first_corner, first_corner]);
    let mut source_edges = ordered.outgoing_edges[..ordered.outgoing_edges.len() - 1].to_vec();
    source_edges.extend([end_edge, end_edge, start_edge, start_edge]);
    Ok(DualFace {
        vertices,
        source_corners,
        source_edges,
    })
}

fn build_dual_mesh(
    source: &Mesh,
    topology: &DualTopology,
    points: DualPoints,
    faces: &DualFaces,
) -> Result<Mesh> {
    let mut result = points.mesh;
    let mut edge_source_by_key = HashMap::<(u32, u32), usize>::new();
    for &(first_face, second_face, source_edge) in &topology.dissolved_edges {
        add_dual_edge(
            &mut result,
            &mut edge_source_by_key,
            points.face_centers[first_face],
            points.face_centers[second_face],
            source_edge,
        )?;
    }
    for face in &faces.items {
        if face.vertices.len() != face.source_edges.len() {
            return Err(evaluation(
                "Dual Mesh face edge provenance has an invalid length",
            ));
        }
        for local in 0..face.vertices.len() {
            add_dual_edge(
                &mut result,
                &mut edge_source_by_key,
                face.vertices[local],
                face.vertices[(local + 1) % face.vertices.len()],
                face.source_edges[local],
            )?;
        }
    }
    for (edge_index, adjacent_faces) in topology.edge_faces.iter().enumerate() {
        let source_edge = &source.edges[edge_index];
        let first_vertex = topology.vertex_indices[&source_edge.vertices[0]];
        let second_vertex = topology.vertex_indices[&source_edge.vertices[1]];
        if adjacent_faces.len() == 2
            && (topology.vertex_kinds[first_vertex] != VertexKind::Unsupported
                || topology.vertex_kinds[second_vertex] != VertexKind::Unsupported)
        {
            add_dual_edge(
                &mut result,
                &mut edge_source_by_key,
                points.face_centers[adjacent_faces[0]],
                points.face_centers[adjacent_faces[1]],
                edge_index,
            )?;
        }
    }
    let mut polygons = Vec::with_capacity(faces.items.len());
    let mut corner_sources = Vec::new();
    for face in &faces.items {
        polygons.push((face.vertices.clone(), 0));
        corner_sources.extend(face.source_corners.iter().copied());
    }
    result
        .insert_faces(polygons.into_iter().map(Ok))
        .map_err(mesh_error)?;
    let edge_sources = result
        .edges
        .iter()
        .map(|edge| {
            edge_source_by_key
                .get(&edge_key(edge.vertices[0], edge.vertices[1]))
                .copied()
        })
        .collect::<Vec<_>>();
    mesh_dual_attributes::transfer(
        source,
        &mut result,
        &points.point_face_sources,
        &faces.point_sources,
        &edge_sources,
        &corner_sources,
    )?;
    Ok(result)
}
fn add_dual_edge(
    mesh: &mut Mesh,
    edge_sources: &mut HashMap<(u32, u32), usize>,
    first: u32,
    second: u32,
    source_edge: usize,
) -> Result<()> {
    let key = edge_key(first, second);
    if edge_sources.contains_key(&key) {
        edge_sources.entry(key).or_insert(source_edge);
        return Ok(());
    }
    mesh.insert_edge([first, second]).map_err(mesh_error)?;
    edge_sources.insert(key, source_edge);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VertexKind {
    Unsupported,
    Normal,
    Boundary,
}

fn ordered_faces(
    mesh: &Mesh,
    vertex_index: usize,
    incident: &[(usize, usize)],
    edge_faces: &[Vec<usize>],
    edge_indices: &HashMap<(u32, u32), usize>,
    boundary: bool,
    corner_offsets: &[usize],
) -> Option<OrderedFaces> {
    let vertex_id = mesh.vertices[vertex_index].id;
    let corner_for_face: HashMap<usize, usize> = incident
        .iter()
        .map(|(face, corner)| (*face, *corner - corner_offsets[*face]))
        .collect();
    let incoming_edge = |face_index: usize, corner: usize| {
        let face = &mesh.faces[face_index];
        edge_indices.get(&edge_key(
            face.vertices[(corner + face.vertices.len() - 1) % face.vertices.len()],
            vertex_id,
        ))
    };
    let outgoing_edge = |face_index: usize, corner: usize| {
        let face = &mesh.faces[face_index];
        edge_indices.get(&edge_key(
            vertex_id,
            face.vertices[(corner + 1) % face.vertices.len()],
        ))
    };
    let (start_face, start_global_corner) = if boundary {
        incident
            .iter()
            .find(|(face, global_corner)| {
                let local = *global_corner - corner_offsets[*face];
                outgoing_edge(*face, local).is_some_and(|edge| edge_faces[*edge].len() == 1)
            })
            .or_else(|| {
                incident.iter().find(|(face, global_corner)| {
                    let local = *global_corner - corner_offsets[*face];
                    incoming_edge(*face, local).is_some_and(|edge| edge_faces[*edge].len() == 1)
                })
            })
            .copied()?
    } else {
        *incident.first()?
    };
    let start = (start_face, start_global_corner - corner_offsets[start_face]);
    let mut faces = Vec::with_capacity(incident.len());
    let mut corners = Vec::with_capacity(incident.len());
    let mut outgoing_edges = Vec::with_capacity(incident.len());
    let mut visited = HashSet::with_capacity(incident.len());
    let mut current = start.0;
    let mut closed = false;
    let mut end_boundary = None;
    loop {
        if !visited.insert(current) {
            closed = current == start.0;
            break;
        }
        let corner = *corner_for_face.get(&current)?;
        faces.push(current);
        corners.push(corner_offsets[current] + corner);
        let edge = *incoming_edge(current, corner)?;
        outgoing_edges.push(edge);
        let uses = &edge_faces[edge];
        if uses.len() == 1 {
            end_boundary = Some(edge);
            break;
        }
        if uses.len() != 2 {
            return None;
        }
        let next = if uses[0] == current { uses[1] } else { uses[0] };
        if next == start.0 {
            closed = true;
            break;
        }
        current = next;
        if faces.len() > incident.len() {
            return None;
        }
    }
    if faces.len() != incident.len() || (!boundary && !closed) {
        return None;
    }
    let start_boundary = if boundary {
        outgoing_edge(start.0, start.1)
            .filter(|edge| edge_faces[**edge].len() == 1)
            .or_else(|| {
                incoming_edge(start.0, start.1).filter(|edge| edge_faces[**edge].len() == 1)
            })
            .copied()
    } else {
        None
    };
    if boundary && (start_boundary.is_none() || end_boundary.is_none()) {
        return None;
    }
    Some(OrderedFaces {
        faces,
        corners,
        outgoing_edges,
        start_boundary,
        end_boundary,
    })
}
