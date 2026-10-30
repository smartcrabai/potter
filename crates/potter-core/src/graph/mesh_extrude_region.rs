use std::collections::HashMap;

use glam::{DVec3, Vec3};
use num_traits::ToPrimitive;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        clnor::{self, NormalCodeOptions},
        provenance::{Origin, copy_direct_matrix_edge_attributes, remap_attributes_with_edges},
    },
};

pub(super) fn extrude_faces_region(
    source: &Mesh,
    selection: &[bool],
    offsets: &[Vec3],
    offset_is_single: bool,
) -> Result<super::ExtrudeResult> {
    source.validate().map_err(super::mesh_error)?;
    if selection.len() != source.faces.len() || offsets.len() != source.faces.len() {
        return Err(super::evaluation(
            "face extrusion selection and offset slices must match the mesh face domain",
        ));
    }

    let mut top = selection
        .iter()
        .copied()
        .map(Value::Bool)
        .collect::<Vec<_>>();
    if !selection.iter().any(|selected| *selected) {
        return Ok(super::ExtrudeResult {
            mesh: source.clone(),
            top_domain: "FACE",
            top,
            side_domain: "FACE",
            side: vec![Value::Bool(false); source.faces.len()],
        });
    }

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
        .map(|(index, edge)| (edge_key(edge.vertices), index))
        .collect::<HashMap<_, _>>();
    let mut face_corners = Vec::with_capacity(source.faces.len().saturating_add(1));
    face_corners.push(0usize);
    for face in &source.faces {
        let corner_start = face_corners
            .last()
            .copied()
            .ok_or_else(|| super::evaluation("face corner offset is missing"))?;
        let next = corner_start
            .checked_add(face.vertices.len())
            .ok_or_else(|| super::evaluation("face extrusion corner count overflowed"))?;
        face_corners.push(next);
    }
    let source_corner_count = face_corners
        .last()
        .copied()
        .ok_or_else(|| super::evaluation("face corner offset is missing"))?;

    let mut edge_faces = vec![Vec::<usize>::new(); source.edges.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        for corner in 0..face.vertices.len() {
            let edge = edge_key([
                face.vertices[corner],
                face.vertices[(corner + 1) % face.vertices.len()],
            ]);
            let edge_index = edge_indices.get(&edge).copied().ok_or_else(|| {
                super::evaluation("a mesh face references an edge missing from the edge domain")
            })?;
            edge_faces[edge_index].push(face_index);
        }
    }

    let mut boundary_edges = Vec::<(usize, usize)>::new();
    let mut duplicated_inner_edges = Vec::<usize>::new();
    let mut inner_edges = Vec::<usize>::new();
    for (edge_index, faces) in edge_faces.iter().enumerate() {
        let mut selected_face = None;
        let mut selected_count = 0usize;
        let mut deselected_count = 0usize;
        for face_index in faces {
            if selection[*face_index] {
                selected_count += 1;
                selected_face = Some(*face_index);
            } else {
                deselected_count += 1;
            }
        }
        if selected_count == 1 {
            let selected_face = selected_face
                .ok_or_else(|| super::evaluation("selected boundary edge has no selected face"))?;
            boundary_edges.push((edge_index, selected_face));
        } else if selected_count > 1 && deselected_count > 0 {
            duplicated_inner_edges.push(edge_index);
        } else if selected_count > 1 {
            inner_edges.push(edge_index);
        }
    }

    if !boundary_edges.is_empty() {
        validate_edge_attribute_types(source)?;
    }

    let mut selected_vertices = vec![false; source.vertices.len()];
    let mut offset_sums = vec![Vec3::ZERO; source.vertices.len()];
    let mut offset_counts = vec![0usize; source.vertices.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        let offset = offsets[face_index];
        if !offset.is_finite() {
            return Err(super::evaluation(
                "selected face extrusion offsets must be finite",
            ));
        }
        for vertex_id in &face.vertices {
            let vertex_index = vertex_indices.get(vertex_id).copied().ok_or_else(|| {
                super::evaluation("a selected face references a missing mesh vertex")
            })?;
            selected_vertices[vertex_index] = true;
            if !offset_is_single {
                offset_sums[vertex_index] += offset;
                offset_counts[vertex_index] += 1;
            }
        }
    }
    let single_offset = if offset_is_single {
        offsets
            .iter()
            .zip(selection)
            .find_map(|(offset, selected)| (*selected).then_some(*offset))
            .ok_or_else(|| super::evaluation("selected face extrusion offset is missing"))?
    } else {
        Vec3::ZERO
    };
    let mut vertex_offsets = vec![Vec3::ZERO; source.vertices.len()];
    for vertex_index in 0..source.vertices.len() {
        if selected_vertices[vertex_index] {
            vertex_offsets[vertex_index] = if offset_is_single {
                single_offset
            } else {
                let count = offset_counts[vertex_index]
                    .to_f32()
                    .ok_or_else(|| super::evaluation("face region has too many offsets"))?;
                offset_sums[vertex_index] / count
            };
        }
    }

    let mut new_vertex_indices = Vec::<usize>::new();
    let mut new_vertex_lookup = vec![false; source.vertices.len()];
    for (edge_index, _) in &boundary_edges {
        for vertex_id in source.edges[*edge_index].vertices {
            let vertex_index = vertex_indices[&vertex_id];
            if !new_vertex_lookup[vertex_index] {
                new_vertex_lookup[vertex_index] = true;
                new_vertex_indices.push(vertex_index);
            }
        }
    }
    let extruded_vertex_count = new_vertex_indices.len();
    for edge_index in &duplicated_inner_edges {
        for vertex_id in source.edges[*edge_index].vertices {
            let vertex_index = vertex_indices[&vertex_id];
            if !new_vertex_lookup[vertex_index] {
                new_vertex_lookup[vertex_index] = true;
                new_vertex_indices.push(vertex_index);
            }
        }
    }

    let mut output = source.clone();
    for vertex_index in 0..source.vertices.len() {
        if selected_vertices[vertex_index] && !new_vertex_lookup[vertex_index] {
            output.vertices[vertex_index].co = DVec3::from(
                source.vertices[vertex_index].co.as_vec3() + vertex_offsets[vertex_index],
            );
            if !output.vertices[vertex_index].co.is_finite() {
                return Err(super::evaluation(
                    "face extrusion produced a non-finite vertex position",
                ));
            }
        }
    }
    let mut new_vertex_ids = vec![None::<u32>; source.vertices.len()];
    for vertex_index in &new_vertex_indices {
        let position = DVec3::from(
            source.vertices[*vertex_index].co.as_vec3() + vertex_offsets[*vertex_index],
        );
        let id = output.insert_vertex(position).map_err(super::mesh_error)?;
        new_vertex_ids[*vertex_index] = Some(id);
    }

    // Only boundary vertices receive connector edges; interior-only duplicates do not.
    for vertex_index in new_vertex_indices.iter().take(extruded_vertex_count) {
        let original_id = source.vertices[*vertex_index].id;
        let new_id = new_vertex_ids[*vertex_index]
            .ok_or_else(|| super::evaluation("extruded vertex provenance is missing"))?;
        output
            .insert_edge([original_id, new_id])
            .map_err(super::mesh_error)?;
    }
    for (edge_index, _) in &boundary_edges {
        let edge = &source.edges[*edge_index];
        let first = vertex_indices[&edge.vertices[0]];
        let second = vertex_indices[&edge.vertices[1]];
        output
            .insert_edge([
                new_vertex_ids[first]
                    .ok_or_else(|| super::evaluation("boundary vertex duplication is missing"))?,
                new_vertex_ids[second]
                    .ok_or_else(|| super::evaluation("boundary vertex duplication is missing"))?,
            ])
            .map_err(super::mesh_error)?;
    }
    for edge_index in &duplicated_inner_edges {
        let edge = &source.edges[*edge_index];
        let first = vertex_indices[&edge.vertices[0]];
        let second = vertex_indices[&edge.vertices[1]];
        output
            .insert_edge([
                new_vertex_ids[first]
                    .ok_or_else(|| super::evaluation("inner edge vertex duplication is missing"))?,
                new_vertex_ids[second]
                    .ok_or_else(|| super::evaluation("inner edge vertex duplication is missing"))?,
            ])
            .map_err(super::mesh_error)?;
    }

    for edge_index in &inner_edges {
        let source_edge = &source.edges[*edge_index];
        let first = vertex_indices[&source_edge.vertices[0]];
        let second = vertex_indices[&source_edge.vertices[1]];
        output.edges[*edge_index].vertices = [
            new_vertex_ids[first].unwrap_or(source_edge.vertices[0]),
            new_vertex_ids[second].unwrap_or(source_edge.vertices[1]),
        ];
    }
    for (face_index, _) in source.faces.iter().enumerate() {
        if !selection[face_index] {
            continue;
        }
        for vertex_id in &mut output.faces[face_index].vertices {
            let vertex_index = vertex_indices[vertex_id];
            if let Some(new_id) = new_vertex_ids[vertex_index] {
                *vertex_id = new_id;
            }
        }
    }

    let mut side_face_sources = Vec::with_capacity(boundary_edges.len());
    let mut side_corner_sources = Vec::<[usize; 4]>::with_capacity(boundary_edges.len());
    for (edge_index, selected_face_index) in &boundary_edges {
        let source_edge = &source.edges[*edge_index];
        let top_first = new_vertex_ids[vertex_indices[&source_edge.vertices[0]]]
            .ok_or_else(|| super::evaluation("boundary vertex duplication is missing"))?;
        let top_second = new_vertex_ids[vertex_indices[&source_edge.vertices[1]]]
            .ok_or_else(|| super::evaluation("boundary vertex duplication is missing"))?;
        let top_face = &output.faces[*selected_face_index];
        let first_to_second = top_face
            .vertices
            .iter()
            .enumerate()
            .any(|(corner, vertex)| {
                *vertex == top_first
                    && top_face.vertices[(corner + 1) % top_face.vertices.len()] == top_second
            });
        let second_to_first = top_face
            .vertices
            .iter()
            .enumerate()
            .any(|(corner, vertex)| {
                *vertex == top_second
                    && top_face.vertices[(corner + 1) % top_face.vertices.len()] == top_first
            });
        if !first_to_second && !second_to_first {
            return Err(super::evaluation(
                "a boundary edge is not present on its selected face",
            ));
        }
        let face = &source.faces[*selected_face_index];
        let corner_first = face
            .vertices
            .iter()
            .position(|vertex| *vertex == source_edge.vertices[0])
            .ok_or_else(|| super::evaluation("boundary edge endpoint is absent from its face"))?;
        let corner_second = face
            .vertices
            .iter()
            .position(|vertex| *vertex == source_edge.vertices[1])
            .ok_or_else(|| super::evaluation("boundary edge endpoint is absent from its face"))?;
        let corner_first = face_corners[*selected_face_index] + corner_first;
        let corner_second = face_corners[*selected_face_index] + corner_second;
        let side_vertices = if first_to_second {
            [
                top_first,
                source_edge.vertices[0],
                source_edge.vertices[1],
                top_second,
            ]
        } else {
            [
                top_first,
                top_second,
                source_edge.vertices[1],
                source_edge.vertices[0],
            ]
        };
        let side_corner_sources_for_face = side_vertices.map(|vertex| {
            if vertex == top_first || vertex == source_edge.vertices[0] {
                corner_first
            } else {
                corner_second
            }
        });
        output
            .insert_face(side_vertices.to_vec(), face.material_index)
            .map_err(super::mesh_error)?;
        side_face_sources.push(*selected_face_index);
        side_corner_sources.push(side_corner_sources_for_face);
    }

    let expected_edges = source
        .edges
        .len()
        .checked_add(extruded_vertex_count)
        .and_then(|count| count.checked_add(boundary_edges.len()))
        .and_then(|count| count.checked_add(duplicated_inner_edges.len()))
        .ok_or_else(|| super::evaluation("face extrusion edge count overflowed"))?;
    let expected_faces = source
        .faces
        .len()
        .checked_add(boundary_edges.len())
        .ok_or_else(|| super::evaluation("face extrusion face count overflowed"))?;
    if output.edges.len() != expected_edges || output.faces.len() != expected_faces {
        return Err(super::evaluation(
            "face extrusion generated topology outside its provenance mapping",
        ));
    }

    let mut vertex_origins = (0..source.vertices.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut source_vertices = (0..source.vertices.len()).map(Some).collect::<Vec<_>>();
    for vertex_index in &new_vertex_indices {
        vertex_origins.push(vec![(*vertex_index, 1.0)]);
        source_vertices.push(Some(*vertex_index));
    }

    let mut edge_origins = (0..source.edges.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut boundary_edges_by_vertex = vec![Vec::<usize>::new(); source.vertices.len()];
    for (edge_index, _) in &boundary_edges {
        let edge = &source.edges[*edge_index];
        boundary_edges_by_vertex[vertex_indices[&edge.vertices[0]]].push(*edge_index);
        boundary_edges_by_vertex[vertex_indices[&edge.vertices[1]]].push(*edge_index);
    }
    for vertex_index in new_vertex_indices.iter().take(extruded_vertex_count) {
        let sources = &boundary_edges_by_vertex[*vertex_index];
        let weight = if sources.is_empty() {
            0.0
        } else {
            1.0 / sources
                .len()
                .to_f64()
                .ok_or_else(|| super::evaluation("face region edge fan is too large"))?
        };
        edge_origins.push(
            sources
                .iter()
                .map(|edge_index| (*edge_index, weight))
                .collect(),
        );
    }
    for (edge_index, _) in &boundary_edges {
        edge_origins.push(vec![(*edge_index, 1.0)]);
    }
    for edge_index in &duplicated_inner_edges {
        edge_origins.push(vec![(*edge_index, 1.0)]);
    }

    let mut face_origins = (0..source.faces.len()).map(Some).collect::<Vec<_>>();
    face_origins.extend(side_face_sources.iter().copied().map(Some));
    let mut corner_origins = (0..source_corner_count)
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut custom_normal_sources = (0..source_corner_count).map(Some).collect::<Vec<_>>();
    for side_sources in &side_corner_sources {
        for source_corner in side_sources {
            corner_origins.push(vec![(*source_corner, 1.0)]);
            custom_normal_sources.push(Some(*source_corner));
        }
    }
    let origin = Origin {
        vertices: vertex_origins,
        source_vertices: Some(source_vertices),
        faces: face_origins,
        corners: Some(corner_origins),
        ..Origin::default()
    };
    remap_attributes_with_edges(source, &mut output, &origin, edge_origins)?;
    copy_direct_matrix_edge_attributes(source, &mut output, &origin)?;
    reset_untransferred_string_attributes(source, &mut output, source_corner_count);
    for key in [
        "skin_radii",
        "skin_roots",
        "blender_skin_vertices",
        "multires_displacement",
    ] {
        output.attributes.remove(key);
    }
    clnor::transfer_custom_normals(
        source,
        &mut output,
        &custom_normal_sources,
        NormalCodeOptions::default(),
    )?;
    output.validate().map_err(super::mesh_error)?;

    top.extend((0..boundary_edges.len()).map(|_| Value::Bool(false)));
    let mut side = vec![Value::Bool(false); source.faces.len()];
    side.extend((0..boundary_edges.len()).map(|_| Value::Bool(true)));
    Ok(super::ExtrudeResult {
        mesh: output,
        top_domain: "FACE",
        top,
        side_domain: "FACE",
        side,
    })
}

pub(super) fn reset_untransferred_string_attributes(
    source: &Mesh,
    output: &mut Mesh,
    source_corner_count: usize,
) {
    if let (Some(source_entries), Some(output_entries)) = (
        source
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array),
        output
            .attributes
            .get_mut("blender_attributes")
            .and_then(Value::as_array_mut),
    ) {
        for (source_entry, output_entry) in source_entries.iter().zip(output_entries) {
            if !source_entry
                .get("data_type")
                .and_then(Value::as_str)
                .is_some_and(|data_type| data_type.eq_ignore_ascii_case("STRING"))
            {
                continue;
            }
            let Some(values) = output_entry.get_mut("values").and_then(Value::as_array_mut) else {
                continue;
            };
            let start = match source_entry
                .get("domain")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_uppercase()
                .as_str()
            {
                "POINT" => source.vertices.len(),
                "EDGE" => source.edges.len(),
                "FACE" => source.faces.len(),
                "CORNER" => source_corner_count,
                _ => continue,
            };
            for value in values.iter_mut().skip(start) {
                *value = Value::String(String::new());
            }
        }
    }

    for (name, source_attribute) in &source.attributes {
        if !source_attribute
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|data_type| data_type.eq_ignore_ascii_case("STRING"))
        {
            continue;
        }
        let domain = source_attribute
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        let Some(values) = output
            .attributes
            .get_mut(name)
            .and_then(|attribute| attribute.get_mut("values"))
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        match domain.as_str() {
            "point" => {
                for vertex in output.vertices.iter().skip(source.vertices.len()) {
                    values.insert(format!("v{}", vertex.id), Value::String(String::new()));
                }
            }
            "edge" => {
                for edge in output.edges.iter().skip(source.edges.len()) {
                    values.insert(format!("e{}", edge.id), Value::String(String::new()));
                }
            }
            "face" => {
                for face in output.faces.iter().skip(source.faces.len()) {
                    values.insert(format!("f{}", face.id), Value::String(String::new()));
                }
            }
            "corner" => {
                for face in output.faces.iter().skip(source.faces.len()) {
                    values.insert(
                        format!("f{}", face.id),
                        Value::Array(vec![Value::String(String::new()); face.vertices.len()]),
                    );
                }
            }
            _ => {}
        }
    }
}

pub(super) fn validate_edge_attribute_types(source: &Mesh) -> Result<()> {
    if let Some(attributes) = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    {
        for attribute in attributes {
            if !attribute
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("EDGE"))
            {
                continue;
            }
            let name = attribute
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("<unnamed>");
            let data_type = attribute
                .get("data_type")
                .and_then(Value::as_str)
                .unwrap_or("UNKNOWN");
            if !supported_edge_data_type(data_type) {
                return Err(unsupported_store(name, data_type));
            }
        }
    }
    for (name, attribute) in &source.attributes {
        if attribute.get("values").and_then(Value::as_object).is_none()
            || !attribute
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("EDGE"))
        {
            continue;
        }
        let raw_type = attribute
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        let data_type = match raw_type.to_ascii_lowercase().as_str() {
            "int" => "INT",
            "bool" => "BOOLEAN",
            "float" => "FLOAT",
            "float2" => "FLOAT2",
            "float3" => "FLOAT_VECTOR",
            "float4" => "FLOAT4",
            "color" => "FLOAT_COLOR",
            "quaternion" => "QUATERNION",
            _ => raw_type,
        };
        if !supported_edge_data_type(data_type) {
            return Err(unsupported_store(name, data_type));
        }
    }
    Ok(())
}

fn supported_edge_data_type(data_type: &str) -> bool {
    matches!(
        data_type.to_ascii_uppercase().as_str(),
        "BOOLEAN"
            | "BOOL"
            | "INT"
            | "INT8"
            | "INT16_2D"
            | "SHORT2"
            | "FLOAT"
            | "FLOAT2"
            | "FLOAT3"
            | "FLOAT_VECTOR"
            | "FLOAT4"
            | "FLOAT_COLOR"
            | "BYTE_COLOR"
            | "INT32_2D"
            | "STRING"
            | "QUATERNION"
            | "FLOAT4X4"
            | "MATRIX"
    )
}

fn unsupported_store(store: &str, data_type: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("Extrude Mesh cannot exactly mix edge attribute type `{data_type}`"),
        json!({
            "feature_id": "graph.node.extrude_mesh.edge_attribute_type",
            "data_type": data_type,
            "attribute": store,
        }),
    )
}

fn edge_key(vertices: [u32; 2]) -> (u32, u32) {
    if vertices[0] < vertices[1] {
        (vertices[0], vertices[1])
    } else {
        (vertices[1], vertices[0])
    }
}
