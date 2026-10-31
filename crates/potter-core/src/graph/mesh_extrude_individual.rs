use std::collections::HashMap;

use glam::{DVec3, Vec3};
use serde_json::json;

use crate::{
    error::Result,
    geom::{
        Mesh, clnor, color_layers, edge_key,
        provenance::{Origin, copy_direct_matrix_edge_attributes, remap_attributes_with_edge_mix2},
        uv_layers,
    },
};

use super::{evaluation, mesh_error};

pub(super) fn extrude_faces_individual(
    source: &Mesh,
    selection: &[bool],
    offsets: &[Vec3],
) -> Result<super::ExtrudeResult> {
    if selection.len() != source.faces.len() || offsets.len() != source.faces.len() {
        return Err(evaluation(
            "extrude face fields do not match the face domain",
        ));
    }
    if !selection.iter().any(|selected| *selected) {
        return Ok(super::ExtrudeResult {
            mesh: source.clone(),
            top_domain: "FACE",
            top: vec![json!(false); source.faces.len()],
            side_domain: "FACE",
            side: vec![json!(false); source.faces.len()],
        });
    }

    color_layers::validate(source)?;
    uv_layers::validate(source)?;
    clnor::validate(source)?;
    super::mesh_extrude_region::validate_edge_attribute_types(source)?;

    let mut corner_offsets = Vec::with_capacity(source.faces.len());
    let mut corner_count = 0usize;
    let mut extruded_corner_count = 0usize;
    for (face, selected) in source.faces.iter().zip(selection) {
        corner_offsets.push(corner_count);
        corner_count = corner_count
            .checked_add(face.vertices.len())
            .ok_or_else(|| evaluation("extrude face corner count exceeds the supported range"))?;
        if *selected {
            extruded_corner_count = extruded_corner_count
                .checked_add(face.vertices.len())
                .ok_or_else(|| {
                    evaluation("extrude face corner count exceeds the supported range")
                })?;
        }
    }

    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut edge_indices = HashMap::with_capacity(source.edges.len());
    for (index, edge) in source.edges.iter().enumerate() {
        if edge_indices
            .insert(edge_key(edge.vertices[0], edge.vertices[1]), index)
            .is_some()
        {
            return Err(evaluation("extrude mesh contains duplicate edges"));
        }
    }

    let mut output = source.clone();
    let mut duplicate_vertices = vec![None; corner_count];
    let mut source_edges_by_corner = vec![None; corner_count];
    let mut vertex_origins = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, _)| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut top = vec![json!(false); source.faces.len() + extruded_corner_count];
    for (face_index, face) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        if face.vertices.len() < 3 {
            return Err(evaluation("extrude face has fewer than three corners"));
        }
        top[face_index] = json!(true);
        for (corner, vertex_id) in face.vertices.iter().copied().enumerate() {
            let vertex_index = *vertex_indices
                .get(&vertex_id)
                .ok_or_else(|| evaluation("extrude face references a missing vertex"))?;
            let source_edge = *edge_indices
                .get(&edge_key(
                    vertex_id,
                    face.vertices[(corner + 1) % face.vertices.len()],
                ))
                .ok_or_else(|| evaluation("extrude face boundary edge is missing"))?;
            let position = source.vertices[vertex_index].co.as_vec3() + offsets[face_index];
            let duplicate = output
                .insert_vertex(DVec3::from(position))
                .map_err(mesh_error)?;
            let source_corner = corner_offsets[face_index] + corner;
            duplicate_vertices[source_corner] = Some(duplicate);
            source_edges_by_corner[source_corner] = Some(source_edge);
            vertex_origins.push(vec![(vertex_index, 1.0)]);
        }
    }

    let mut edge_origins = (0..source.edges.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    for (face_index, face) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        let start = corner_offsets[face_index];
        for corner in 0..face.vertices.len() {
            let source_corner = start + corner;
            let vertex_id = face.vertices[corner];
            let duplicate = duplicate_vertices[source_corner]
                .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))?;
            output
                .insert_edge([vertex_id, duplicate])
                .map_err(mesh_error)?;
            let source_edge = source_edges_by_corner[source_corner]
                .ok_or_else(|| evaluation("extrude face boundary edge is missing"))?;
            let previous = start
                + if corner == 0 {
                    face.vertices.len() - 1
                } else {
                    corner - 1
                };
            let previous_edge = source_edges_by_corner[previous]
                .ok_or_else(|| evaluation("extrude face boundary edge is missing"))?;
            edge_origins.push(vec![(source_edge, 0.5), (previous_edge, 0.5)]);
        }
    }

    for (face_index, face) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        let start = corner_offsets[face_index];
        for corner in 0..face.vertices.len() {
            let next = (corner + 1) % face.vertices.len();
            let first = duplicate_vertices[start + corner]
                .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))?;
            let second = duplicate_vertices[start + next]
                .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))?;
            output.insert_edge([first, second]).map_err(mesh_error)?;
            let source_edge = source_edges_by_corner[start + corner]
                .ok_or_else(|| evaluation("extrude face boundary edge is missing"))?;
            edge_origins.push(vec![(source_edge, 1.0)]);
        }
        output.faces[face_index].vertices = (0..face.vertices.len())
            .map(|corner| {
                duplicate_vertices[start + corner]
                    .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))
            })
            .collect::<Result<Vec<_>>>()?;
    }

    let mut face_origins = (0..source.faces.len()).map(Some).collect::<Vec<_>>();
    let mut corner_origins = Vec::with_capacity(corner_count + extruded_corner_count * 4);
    for face in &source.faces {
        for _ in &face.vertices {
            let corner = corner_origins.len();
            corner_origins.push(vec![(corner, 1.0)]);
        }
    }
    let mut side = vec![json!(false); source.faces.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        let start = corner_offsets[face_index];
        for corner in 0..face.vertices.len() {
            let next = (corner + 1) % face.vertices.len();
            let new_vertex = duplicate_vertices[start + corner]
                .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))?;
            let new_vertex_next = duplicate_vertices[start + next]
                .ok_or_else(|| evaluation("extrude face vertex has no duplicate"))?;
            output
                .insert_face(
                    vec![
                        new_vertex_next,
                        new_vertex,
                        face.vertices[corner],
                        face.vertices[next],
                    ],
                    face.material_index,
                )
                .map_err(mesh_error)?;
            face_origins.push(Some(face_index));
            side.push(json!(true));
            let source_corner = start + corner;
            let source_next_corner = start + next;
            for origin_corner in [
                source_next_corner,
                source_corner,
                source_corner,
                source_next_corner,
            ] {
                corner_origins.push(vec![(origin_corner, 1.0)]);
            }
        }
    }

    let origin = Origin {
        vertices: vertex_origins,
        faces: face_origins,
        corners: Some(corner_origins),
        ..Default::default()
    };
    remap_attributes_with_edge_mix2(source, &mut output, &origin, edge_origins)?;
    copy_direct_matrix_edge_attributes(source, &mut output, &origin)?;
    super::mesh_extrude_region::reset_untransferred_string_attributes(
        source,
        &mut output,
        corner_count,
    );
    for key in [
        "skin_radii",
        "skin_roots",
        "blender_skin_vertices",
        "multires_displacement",
    ] {
        output.attributes.remove(key);
    }

    Ok(super::ExtrudeResult {
        mesh: output,
        top_domain: "FACE",
        top,
        side_domain: "FACE",
        side,
    })
}
