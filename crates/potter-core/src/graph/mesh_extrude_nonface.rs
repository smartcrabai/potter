use std::collections::HashMap;

use glam::{DVec3, Vec3};
use num_traits::ToPrimitive;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, attribute_keys, color_layers, edge_key, uv_layers,
        provenance::{
            self, Origin, remap_attributes_with_default_edge_mixer,
        },
    },
};

use super::evaluation;

pub(super) struct ExtrudeResult {
    pub(super) mesh: Mesh,
    pub(super) top_domain: &'static str,
    pub(super) top: Vec<Value>,
    pub(super) side_domain: &'static str,
    pub(super) side: Vec<Value>,
}

pub(super) fn extrude_vertices(
    source: &Mesh,
    selection: &[bool],
    offsets: &[Vec3],
    _offset_is_single: bool,
) -> Result<ExtrudeResult> {
    if selection.len() != source.vertices.len() || offsets.len() != source.vertices.len() {
        return Err(evaluation(
            "extrude vertex fields do not match the point domain",
        ));
    }
    if !selection.iter().any(|selected| *selected) {
        return Ok(ExtrudeResult {
            mesh: source.clone(),
            top_domain: "POINT",
            top: vec![json!(false); source.vertices.len()],
            side_domain: "EDGE",
            side: vec![json!(false); source.edges.len()],
        });
    }

    let mut result = source.clone();
    let mut vertex_origins = (0..source.vertices.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut edge_sources = vec![Vec::<usize>::new(); source.vertices.len()];
    for (edge_index, edge) in source.edges.iter().enumerate() {
        for vertex_id in edge.vertices {
            let vertex_index = *vertex_indices
                .get(&vertex_id)
                .ok_or_else(|| evaluation("extrude edge references a missing vertex"))?;
            edge_sources[vertex_index].push(edge_index);
        }
    }
    if selection
        .iter()
        .enumerate()
        .any(|(index, selected)| *selected && edge_sources[index].len() > 1)
    {
        super::mesh_operations::validate_extrusion_edge_attribute_types(source)?;
    }
    let mut generated_edge_sources = Vec::new();
    let mut top = vec![json!(false); source.vertices.len()];
    let mut side = vec![json!(false); source.edges.len()];
    for (vertex_index, (vertex, selected)) in source.vertices.iter().zip(selection).enumerate() {
        if !selected {
            continue;
        }
        let position = vertex.co.as_vec3() + offsets[vertex_index];
        let duplicate = result
            .insert_vertex(DVec3::from(position))
            .map_err(mesh_error)?;
        vertex_origins.push(vec![(vertex_index, 1.0)]);
        result
            .insert_edge([vertex.id, duplicate])
            .map_err(mesh_error)?;
        let output_edge_index = result.edges.len() - 1;
        let sources = &edge_sources[vertex_index];
        let weight = if sources.is_empty() {
            0.0
        } else {
            1.0 / count_as_f64(sources.len(), "too many source edges for attribute mixing")?
        };
        generated_edge_sources.push(
            sources
                .iter()
                .map(|source| (*source, weight))
                .collect::<Vec<_>>(),
        );
        top.push(json!(true));
        side.resize(result.edges.len(), json!(false));
        side[output_edge_index] = json!(true);
    }

    let mut edge_origins = (0..source.edges.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    edge_origins.extend(generated_edge_sources);
    let origin = Origin {
        vertices: vertex_origins,
        faces: (0..source.faces.len()).map(Some).collect(),
        ..Default::default()
    };
    remap_attributes_with_default_edge_mixer(source, &mut result, &origin, edge_origins)?;
    provenance::copy_direct_matrix_edge_attributes(source, &mut result, &origin)?;
    super::mesh_operations::reset_untransferred_string_attributes(source, &mut result)?;
    remove_unsupported_point_data(&mut result);

    Ok(ExtrudeResult {
        mesh: result,
        top_domain: "POINT",
        top,
        side_domain: "EDGE",
        side,
    })
}

pub(super) fn extrude_edges(
    source: &Mesh,
    selection: &[bool],
    offsets: &[Vec3],
    offset_is_single: bool,
) -> Result<ExtrudeResult> {
    if selection.len() != source.edges.len() || offsets.len() != source.edges.len() {
        return Err(evaluation(
            "extrude edge fields do not match the edge domain",
        ));
    }
    let selected_edges = selection
        .iter()
        .enumerate()
        .filter_map(|(index, selected)| selected.then_some(index))
        .collect::<Vec<_>>();
    if selected_edges.is_empty() {
        return Ok(ExtrudeResult {
            mesh: source.clone(),
            top_domain: "EDGE",
            top: vec![json!(false); source.edges.len()],
            side_domain: "FACE",
            side: vec![json!(false); source.faces.len()],
        });
    }

    let edge_indices = source_edge_indices(source)?;
    let edge_faces = edge_face_map(source, &edge_indices)?;
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut shell = build_edge_shell(
        source,
        &selected_edges,
        offsets,
        offset_is_single,
        &vertex_indices,
    )?;
    let faces = append_edge_faces(
        source,
        &mut shell.mesh,
        &selected_edges,
        &edge_faces,
        &vertex_indices,
        &shell.duplicate_vertices,
    )?;
    let origin = Origin {
        vertices: shell.vertex_origins,
        faces: faces.origins,
        corners: Some(faces.corners),
        ..Default::default()
    };
    remap_attributes_with_default_edge_mixer(source, &mut shell.mesh, &origin, shell.edge_origins)?;
    provenance::copy_direct_matrix_edge_attributes(source, &mut shell.mesh, &origin)?;
    mix_generated_face_attributes(source, &mut shell.mesh, &faces.sources, &faces.indices)?;
    fill_empty_generated_face_uvs(&mut shell.mesh, &faces.indices)?;
    super::mesh_operations::reset_untransferred_string_attributes(source, &mut shell.mesh)?;
    remove_unsupported_point_data(&mut shell.mesh);
    shell.mesh.attributes.remove("multires_displacement");

    let mut top = vec![json!(false); shell.mesh.edges.len()];
    for index in shell.duplicate_edge_indices {
        top[index] = json!(true);
    }
    let mut side = vec![json!(false); shell.mesh.faces.len()];
    for index in faces.indices {
        side[index] = json!(true);
    }
    Ok(ExtrudeResult {
        mesh: shell.mesh,
        top_domain: "EDGE",
        top,
        side_domain: "FACE",
        side,
    })
}

fn fill_empty_generated_face_uvs(mesh: &mut Mesh, faces: &[usize]) -> Result<()> {
    let mut store = uv_layers::read(mesh);
    for layer in &mut store.layers {
        for &face_index in faces {
            let face = mesh
                .faces
                .get(face_index)
                .ok_or_else(|| evaluation("extrude UV face index is out of range"))?;
            let uvs = layer
                .faces
                .get_mut(face_index)
                .ok_or_else(|| evaluation("extrude UV layer has an invalid face count"))?;
            if uvs.is_none() {
                *uvs = Some(vec![[0.0, 1.0]; face.vertices.len()]);
            }
        }
    }
    uv_layers::write(mesh, &store);
    Ok(())
}

struct EdgeShell {
    mesh: Mesh,
    duplicate_vertices: Vec<Option<u32>>,
    vertex_origins: Vec<Vec<(usize, f64)>>,
    edge_origins: Vec<Vec<(usize, f64)>>,
    duplicate_edge_indices: Vec<usize>,
}

struct GeneratedFaces {
    origins: Vec<Option<usize>>,
    corners: Vec<Vec<(usize, f64)>>,
    sources: Vec<Vec<usize>>,
    indices: Vec<usize>,
}

fn build_edge_shell(
    source: &Mesh,
    selected_edges: &[usize],
    offsets: &[Vec3],
    offset_is_single: bool,
    vertex_indices: &HashMap<u32, usize>,
) -> Result<EdgeShell> {
    let mut mesh = source.clone();
    let mut selected_vertices = vec![false; source.vertices.len()];
    let mut incident_selected_edges = vec![Vec::<usize>::new(); source.vertices.len()];
    let mut offset_sums = vec![Vec3::ZERO; source.vertices.len()];
    let mut offset_counts = vec![0usize; source.vertices.len()];
    for &edge_index in selected_edges {
        for vertex_id in source.edges[edge_index].vertices {
            let vertex_index = *vertex_indices
                .get(&vertex_id)
                .ok_or_else(|| evaluation("extrude edge references a missing vertex"))?;
            selected_vertices[vertex_index] = true;
            incident_selected_edges[vertex_index].push(edge_index);
            offset_sums[vertex_index] += offsets[edge_index];
            offset_counts[vertex_index] += 1;
        }
    }
    if incident_selected_edges.iter().any(|edges| edges.len() > 1) {
        super::mesh_operations::validate_extrusion_edge_attribute_types(source)?;
    }

    let single_offset = selected_edges.first().map(|edge| offsets[*edge]);
    let mut duplicate_vertices = vec![None; source.vertices.len()];
    let mut vertex_origins = (0..source.vertices.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    for (vertex_index, vertex) in source.vertices.iter().enumerate() {
        if !selected_vertices[vertex_index] {
            continue;
        }
        let offset = if offset_is_single {
            single_offset.ok_or_else(|| evaluation("selected edge offset field is empty"))?
        } else {
            let count = count_as_f32(
                offset_counts[vertex_index],
                "extrude edge offset fan is too large",
            )?;
            offset_sums[vertex_index] * (1.0 / count)
        };
        let position = vertex.co.as_vec3() + offset;
        let duplicate = mesh
            .insert_vertex(DVec3::from(position))
            .map_err(mesh_error)?;
        duplicate_vertices[vertex_index] = Some(duplicate);
        vertex_origins.push(vec![(vertex_index, 1.0)]);
    }

    let mut edge_origins = (0..source.edges.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut duplicate_edge_indices = Vec::with_capacity(selected_edges.len());
    for (vertex_index, vertex) in source.vertices.iter().enumerate() {
        let Some(duplicate) = duplicate_vertices[vertex_index] else {
            continue;
        };
        mesh.insert_edge([vertex.id, duplicate])
            .map_err(mesh_error)?;
        let incident = &incident_selected_edges[vertex_index];
        let weight = 1.0 / count_as_f64(incident.len(), "extrude edge fan is too large")?;
        edge_origins.push(
            incident
                .iter()
                .map(|source| (*source, weight))
                .collect::<Vec<_>>(),
        );
    }
    for &edge_index in selected_edges {
        let edge = &source.edges[edge_index];
        let first_index = vertex_indices[&edge.vertices[0]];
        let second_index = vertex_indices[&edge.vertices[1]];
        let first = duplicate_vertices[first_index]
            .ok_or_else(|| evaluation("extrude selected edge has no duplicate vertex"))?;
        let second = duplicate_vertices[second_index]
            .ok_or_else(|| evaluation("extrude selected edge has no duplicate vertex"))?;
        mesh.insert_edge([first, second]).map_err(mesh_error)?;
        duplicate_edge_indices.push(mesh.edges.len() - 1);
        edge_origins.push(vec![(edge_index, 1.0)]);
    }

    Ok(EdgeShell {
        mesh,
        duplicate_vertices,
        vertex_origins,
        edge_origins,
        duplicate_edge_indices,
    })
}

fn append_edge_faces(
    source: &Mesh,
    mesh: &mut Mesh,
    selected_edges: &[usize],
    edge_faces: &[Vec<usize>],
    vertex_indices: &HashMap<u32, usize>,
    duplicate_vertices: &[Option<u32>],
) -> Result<GeneratedFaces> {
    let offsets = corner_offsets(source);
    let mut corners = Vec::new();
    for (face_index, face) in source.faces.iter().enumerate() {
        for corner in 0..face.vertices.len() {
            corners.push(vec![(offsets[face_index] + corner, 1.0)]);
        }
    }
    let mut origins = (0..source.faces.len()).map(Some).collect::<Vec<_>>();
    let mut sources = Vec::with_capacity(selected_edges.len());
    let mut indices = Vec::with_capacity(selected_edges.len());
    for &edge_index in selected_edges {
        let edge = &source.edges[edge_index];
        let connected_faces = &edge_faces[edge_index];
        let first_duplicate = duplicate_vertices[vertex_indices[&edge.vertices[0]]]
            .ok_or_else(|| evaluation("extrude selected edge has no duplicate vertex"))?;
        let second_duplicate = duplicate_vertices[vertex_indices[&edge.vertices[1]]]
            .ok_or_else(|| evaluation("extrude selected edge has no duplicate vertex"))?;
        let quad = edge_quad(
            source,
            edge,
            connected_faces,
            first_duplicate,
            second_duplicate,
        )?;
        mesh.insert_face(quad.to_vec(), 0).map_err(mesh_error)?;
        indices.push(mesh.faces.len() - 1);
        sources.push(connected_faces.clone());
        origins.push(connected_faces.first().copied());

        let endpoint_corners = [
            face_corner_sources(source, connected_faces, edge.vertices[0], &offsets)?,
            face_corner_sources(source, connected_faces, edge.vertices[1], &offsets)?,
        ];
        for vertex_id in quad {
            if vertex_id == edge.vertices[0] || vertex_id == first_duplicate {
                corners.push(endpoint_corners[0].clone());
            } else if vertex_id == edge.vertices[1] || vertex_id == second_duplicate {
                corners.push(endpoint_corners[1].clone());
            } else {
                return Err(evaluation("extrude side face has an unknown corner vertex"));
            }
        }
    }
    Ok(GeneratedFaces {
        origins,
        corners,
        sources,
        indices,
    })
}

fn source_edge_indices(source: &Mesh) -> Result<HashMap<(u32, u32), usize>> {
    let mut indices = HashMap::with_capacity(source.edges.len());
    for (index, edge) in source.edges.iter().enumerate() {
        if indices
            .insert(edge_key(edge.vertices[0], edge.vertices[1]), index)
            .is_some()
        {
            return Err(evaluation("mesh contains duplicate edges"));
        }
    }
    Ok(indices)
}

fn edge_face_map(
    source: &Mesh,
    edge_indices: &HashMap<(u32, u32), usize>,
) -> Result<Vec<Vec<usize>>> {
    let mut edge_faces = vec![Vec::new(); source.edges.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        for local in 0..face.vertices.len() {
            let edge_index = edge_indices
                .get(&edge_key(
                    face.vertices[local],
                    face.vertices[(local + 1) % face.vertices.len()],
                ))
                .ok_or_else(|| evaluation("extrude face boundary edge is missing"))?;
            edge_faces[*edge_index].push(face_index);
        }
    }
    Ok(edge_faces)
}

fn edge_quad(
    source: &Mesh,
    edge: &crate::geom::Edge,
    connected_faces: &[usize],
    first_duplicate: u32,
    second_duplicate: u32,
) -> Result<[u32; 4]> {
    let mut start_with_connecting_edge = true;
    if connected_faces.len() == 1 {
        let face = source
            .faces
            .get(connected_faces[0])
            .ok_or_else(|| evaluation("extrude edge-to-face index is out of range"))?;
        let local = (0..face.vertices.len())
            .find(|corner| {
                edge_key(
                    face.vertices[*corner],
                    face.vertices[(*corner + 1) % face.vertices.len()],
                ) == edge_key(edge.vertices[0], edge.vertices[1])
            })
            .ok_or_else(|| evaluation("extrude boundary edge has no face corner"))?;
        start_with_connecting_edge = face.vertices[local] == edge.vertices[0];
    }
    Ok(if start_with_connecting_edge {
        [
            edge.vertices[0],
            first_duplicate,
            second_duplicate,
            edge.vertices[1],
        ]
    } else {
        [
            edge.vertices[0],
            edge.vertices[1],
            second_duplicate,
            first_duplicate,
        ]
    })
}

fn corner_offsets(source: &Mesh) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(source.faces.len());
    let mut offset = 0;
    for face in &source.faces {
        offsets.push(offset);
        offset += face.vertices.len();
    }
    offsets
}

fn face_corner_sources(
    source: &Mesh,
    face_indices: &[usize],
    vertex_id: u32,
    offsets: &[usize],
) -> Result<Vec<(usize, f64)>> {
    let mut corners = Vec::new();
    for &face_index in face_indices {
        if let Some(face) = source.faces.get(face_index) {
            for (local, candidate) in face.vertices.iter().enumerate() {
                if *candidate == vertex_id {
                    corners.push(offsets[face_index] + local);
                }
            }
        }
    }
    let weight = if corners.is_empty() {
        0.0
    } else {
        1.0 / count_as_f64(corners.len(), "extrude corner fan is too large")?
    };
    Ok(corners.into_iter().map(|corner| (corner, weight)).collect())
}

fn mix_generated_face_attributes(
    source: &Mesh,
    output: &mut Mesh,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) -> Result<()> {
    if source_faces.len() != output_faces.len() {
        return Err(evaluation(
            "extrude face attribute map has an invalid length",
        ));
    }
    for (output_index, sources) in output_faces.iter().zip(source_faces) {
        if *output_index >= output.faces.len() {
            return Err(evaluation("extrude generated face index is out of range"));
        }
        let mut material_sum = 0.0_f64;
        for source_index in sources {
            let source_face = source
                .faces
                .get(*source_index)
                .ok_or_else(|| evaluation("extrude source face index is out of range"))?;
            material_sum += f64::from(source_face.material_index);
        }
        if !sources.is_empty() {
            let count = count_as_f32(sources.len(), "extrude face fan is too large")?;
            let inverse = 1.0_f32 / count;
            output.faces[*output_index].material_index = (material_sum * f64::from(inverse))
                .round()
                .to_u32()
                .ok_or_else(|| evaluation("extrude face material index is out of range"))?;
        }
    }
    for (key, source_value) in &source.attributes {
        if key == "blender_attributes" {
            let Some(source_entries) = source_value.as_array() else {
                continue;
            };
            let Some(output_entries) = output.attributes.get_mut(key).and_then(Value::as_array_mut)
            else {
                continue;
            };
            for (source_entry, output_entry) in source_entries.iter().zip(output_entries) {
                mix_array_face_entry(source_entry, output_entry, source_faces, output_faces)?;
            }
        } else if key == "blender_polygon_smooth" {
            mix_polygon_smooth(source_value, output, source_faces, output_faces);
        } else if key == "shade_smooth" {
            mix_shade_smooth(source, output, source_value, source_faces, output_faces)?;
        } else if let Some(domain) = source_value.get("domain").and_then(Value::as_str)
            && domain.eq_ignore_ascii_case("face")
        {
            if source_value.get("data").is_some() {
                mix_stored_face_entry(source, output, key, source_value, source_faces, output_faces)?;
            } else {
                mix_keyed_face_entry(
                    source,
                    output,
                    key,
                    source_value,
                    source_faces,
                    output_faces,
                )?;
            }
        }
    }
    Ok(())
}

fn mix_array_face_entry(
    source_entry: &Value,
    output_entry: &mut Value,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) -> Result<()> {
    if !source_entry
        .get("domain")
        .and_then(Value::as_str)
        .is_some_and(|domain| domain.eq_ignore_ascii_case("face"))
    {
        return Ok(());
    }
    let Some(source_values) = source_entry.get("values").and_then(Value::as_array) else {
        return Ok(());
    };
    let data_type = source_entry
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let default = face_default_like(data_type, source_values.first());
    let Some(output_values) = output_entry.get_mut("values").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for (sources, output_index) in source_faces.iter().zip(output_faces) {
        let values = sources
            .iter()
            .map(|index| source_values.get(*index).unwrap_or(&default))
            .collect::<Vec<_>>();
        let mixed = mix_face_values(&values, data_type)?;
        if let Some(output_value) = output_values.get_mut(*output_index) {
            *output_value = mixed;
        }
    }
    Ok(())
}

fn mix_stored_face_entry(
    source: &Mesh,
    output: &mut Mesh,
    key: &str,
    source_entry: &Value,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) -> Result<()> {
    let Some(source_values) = source_entry.get("data").and_then(Value::as_array) else {
        return Ok(());
    };
    let data_type = canonical_face_type(
        source,
        key,
        source_entry
            .get("data_type")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    let default = face_default_like(&data_type, source_values.first());
    let Some(output_values) = output
        .attributes
        .get_mut(key)
        .and_then(|entry| entry.get_mut("data"))
        .and_then(Value::as_array_mut)
    else {
        return Ok(());
    };
    for (sources, output_index) in source_faces.iter().zip(output_faces) {
        let values = sources
            .iter()
            .map(|index| source_values.get(*index).unwrap_or(&default))
            .collect::<Vec<_>>();
        let mixed = mix_face_values(&values, &data_type)?;
        if let Some(output_value) = output_values.get_mut(*output_index) {
            *output_value = mixed;
        }
    }
    Ok(())
}

fn mix_keyed_face_entry(
    source: &Mesh,
    output: &mut Mesh,
    key: &str,
    source_entry: &Value,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) -> Result<()> {
    let Some(source_values) = source_entry.get("values").and_then(Value::as_object) else {
        return Ok(());
    };
    let raw_type = source_entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let data_type = canonical_face_type(source, key, raw_type);
    let default = face_default_like(&data_type, source_values.values().next());
    let output_face_keys = output_faces
        .iter()
        .map(|index| {
            output
                .faces
                .get(*index)
                .map(|face| format!("f{}", face.id))
                .ok_or_else(|| evaluation("extrude generated face index is out of range"))
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(output_values) = output
        .attributes
        .get_mut(key)
        .and_then(|entry| entry.get_mut("values"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };
    for (sources, output_key) in source_faces.iter().zip(output_face_keys) {
        let source_values_for_face = sources
            .iter()
            .filter_map(|index| source.faces.get(*index))
            .map(|face| {
                source_values
                    .get(&format!("f{}", face.id))
                    .unwrap_or(&default)
            })
            .collect::<Vec<_>>();
        let has_value = sources.iter().any(|index| {
            source
                .faces
                .get(*index)
                .is_some_and(|face| source_values.contains_key(&format!("f{}", face.id)))
        });
        if has_value {
            output_values.insert(
                output_key,
                mix_face_values(&source_values_for_face, &data_type)?,
            );
        } else {
            output_values.remove(&output_key);
        }
    }
    Ok(())
}

fn mix_polygon_smooth(
    source_value: &Value,
    output: &mut Mesh,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) {
    let Some(source_values) = source_value.as_array() else {
        return;
    };
    let Some(output_values) = output
        .attributes
        .get_mut("blender_polygon_smooth")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for (sources, output_index) in source_faces.iter().zip(output_faces) {
        // Blender requires every source face to be smooth; an empty fan keeps the smooth default.
        let smooth = sources
            .iter()
            .all(|index| source_values.get(*index).and_then(Value::as_bool) == Some(true));
        if let Some(value) = output_values.get_mut(*output_index) {
            *value = json!(smooth);
        }
    }
}

fn mix_shade_smooth(
    source: &Mesh,
    output: &mut Mesh,
    source_value: &Value,
    source_faces: &[Vec<usize>],
    output_faces: &[usize],
) -> Result<()> {
    let Some(source_values) = source_value.get("values").and_then(Value::as_object) else {
        return Ok(());
    };
    let Some(output_values) = output
        .attributes
        .get_mut("shade_smooth")
        .and_then(|entry| entry.get_mut("values"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };
    for (sources, output_index) in source_faces.iter().zip(output_faces) {
        let output_face = output
            .faces
            .get(*output_index)
            .ok_or_else(|| evaluation("extrude generated face index is out of range"))?;
        let mut smooth = true;
        for source_index in sources {
            let source_face = source
                .faces
                .get(*source_index)
                .ok_or_else(|| evaluation("extrude source face index is out of range"))?;
            smooth &= source_values
                .get(&format!("f{}", source_face.id))
                .and_then(Value::as_bool)
                == Some(true);
        }
        output_values.insert(format!("f{}", output_face.id), json!(smooth));
    }
    Ok(())
}

fn canonical_face_type(source: &Mesh, key: &str, raw_type: &str) -> String {
    match raw_type.to_ascii_uppercase().as_str() {
        "FLOAT" => "FLOAT".to_owned(),
        "INT" | "INT32" => "INT".to_owned(),
        "INT8" => "INT8".to_owned(),
        "INT16_2D" | "SHORT2" => "INT16_2D".to_owned(),
        "BOOL" | "BOOLEAN" => "BOOLEAN".to_owned(),
        "FLOAT2" => "FLOAT2".to_owned(),
        "FLOAT3" | "FLOAT_VECTOR" => "FLOAT_VECTOR".to_owned(),
        "FLOAT4" => "FLOAT4".to_owned(),
        "INT32_2D" => "INT32_2D".to_owned(),
        "FLOAT_COLOR" => "FLOAT_COLOR".to_owned(),
        "BYTE_COLOR" => "BYTE_COLOR".to_owned(),
        "FLOAT4X4" | "MATRIX" => "FLOAT4X4".to_owned(),
        "QUATERNION" | "ROTATION" => "QUATERNION".to_owned(),
        "COLOR" => color_data_type(source, key).to_owned(),
        "STRING" => "STRING".to_owned(),
        other => other.to_owned(),
    }
}
fn color_data_type(source: &Mesh, key: &str) -> &'static str {
    let name = attribute_keys::user_name(key).unwrap_or(key);
    match source
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(|types| types.get(name))
        .and_then(Value::as_str)
    {
        Some("BYTE_COLOR") => "BYTE_COLOR",
        _ => "FLOAT_COLOR",
    }
}

fn face_default(data_type: &str) -> Value {
    match data_type {
        "STRING" => json!(""),
        "INT16_2D" => json!([0, 0]),
        "FLOAT4" => json!([0.0, 0.0, 0.0, 0.0]),
        "FLOAT4X4" => json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        _ => provenance::default_value(data_type),
    }
}
fn face_default_like(data_type: &str, template: Option<&Value>) -> Value {
    if data_type == "FLOAT4X4" {
        let nested = template.is_some_and(|value| {
            value
                .as_array()
                .is_some_and(|rows| rows.len() == 4 && rows[0].as_array().is_some())
        });
        return float_matrix_value(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            nested,
        );
    }
    face_default(data_type)
}

fn mix_face_values(values: &[&Value], data_type: &str) -> Result<Value> {
    if values.is_empty() {
        return Ok(face_default(data_type));
    }
    match data_type {
        "BOOLEAN" | "BOOL" => Ok(json!(
            values.iter().any(|value| value.as_bool() == Some(true))
        )),
        "INT" => mix_int(values),
        "INT8" => mix_int8(values),
        "INT16_2D" => mix_int16_2d(values),
        "INT32_2D" => mix_int32_2d(values),
        "FLOAT" => mix_float(values),
        "FLOAT2" => mix_float_array(values, 2),
        "FLOAT_VECTOR" => mix_float_array(values, 3),
        "FLOAT4" | "FLOAT_COLOR" => mix_float_array(values, 4),
        "BYTE_COLOR" => mix_byte_colors(values),
        "FLOAT4X4" => mix_float4x4(values),
        "QUATERNION" => mix_quaternions(values),
        "STRING" => Ok(json!("")),
        _ => Err(unsupported_face_attribute_type(data_type)),
    }
}

fn mix_int(values: &[&Value]) -> Result<Value> {
    let mut sum = 0.0_f64;
    for value in values {
        let value = value
            .as_i64()
            .ok_or_else(|| evaluation("extrude face integer attribute is malformed"))?;
        sum += value
            .to_f64()
            .ok_or_else(|| evaluation("extrude face integer attribute is out of range"))?;
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    (sum * f64::from(1.0_f32 / count))
        .round()
        .to_i64()
        .map(|value| json!(value))
        .ok_or_else(|| evaluation("extrude face integer attribute is out of range"))
}

fn mix_int8(values: &[&Value]) -> Result<Value> {
    let mut sum = 0.0_f32;
    for value in values {
        let value = value
            .as_i64()
            .ok_or_else(|| evaluation("extrude face integer attribute is malformed"))?;
        sum += value
            .to_f32()
            .ok_or_else(|| evaluation("extrude face integer attribute is out of range"))?;
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    (sum * (1.0_f32 / count))
        .round()
        .to_i64()
        .map(|value| json!(value))
        .ok_or_else(|| evaluation("extrude face integer attribute is out of range"))
}

fn mix_int16_2d(values: &[&Value]) -> Result<Value> {
    let mut sums = [0.0_f32; 2];
    for value in values {
        for (sum, component) in sums.iter_mut().zip(numeric_array(value, 2)?) {
            *sum += component;
        }
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    let inverse = 1.0_f32 / count;
    let x = (sums[0] * inverse)
        .round()
        .to_i64()
        .ok_or_else(|| evaluation("extrude face integer vector is out of range"))?;
    let y = (sums[1] * inverse)
        .round()
        .to_i64()
        .ok_or_else(|| evaluation("extrude face integer vector is out of range"))?;
    Ok(json!([x, y]))
}

fn mix_int32_2d(values: &[&Value]) -> Result<Value> {
    let mut sums = [0.0_f64; 2];
    for value in values {
        for (sum, component) in sums.iter_mut().zip(integer_array(value, 2)?) {
            *sum += component
                .to_f64()
                .ok_or_else(|| evaluation("extrude face integer vector is out of range"))?;
        }
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    let inverse = f64::from(1.0_f32 / count);
    let x = (sums[0] * inverse)
        .round()
        .to_i64()
        .ok_or_else(|| evaluation("extrude face integer vector is out of range"))?;
    let y = (sums[1] * inverse)
        .round()
        .to_i64()
        .ok_or_else(|| evaluation("extrude face integer vector is out of range"))?;
    Ok(json!([x, y]))
}

fn mix_float(values: &[&Value]) -> Result<Value> {
    let mut sum = 0.0_f32;
    for value in values {
        let value = value
            .as_f64()
            .ok_or_else(|| evaluation("extrude face float attribute is malformed"))?;
        sum += value
            .to_f32()
            .ok_or_else(|| evaluation("extrude face float attribute is out of range"))?;
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    Ok(json!(sum * (1.0_f32 / count)))
}

fn mix_byte_colors(values: &[&Value]) -> Result<Value> {
    let colors = values
        .iter()
        .map(|value| {
            let components = numeric_array(value, 4)?;
            Ok([
                f64::from(components[0]),
                f64::from(components[1]),
                f64::from(components[2]),
                f64::from(components[3]),
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let sources = (0..colors.len())
        .map(|index| (index, 1.0))
        .collect::<Vec<_>>();
    Ok(json!(color_layers::mix_byte_color(&colors, &sources)))
}

fn mix_quaternions(values: &[&Value]) -> Result<Value> {
    let mut sum = Vec3::ZERO;
    for value in values {
        let exponential = provenance::quaternion_expmap(value)
            .ok_or_else(|| evaluation("extrude face quaternion attribute is malformed"))?;
        sum += exponential;
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    let [x, y, z, w] = provenance::quaternion_from_expmap(sum * (1.0 / count));
    Ok(json!([x, y, z, w]))
}

fn mix_float_array(values: &[&Value], length: usize) -> Result<Value> {
    let mut sums = vec![0.0_f32; length];
    for value in values {
        for (sum, component) in sums.iter_mut().zip(numeric_array(value, length)?) {
            *sum += component;
        }
    }
    let count = count_as_f32(values.len(), "extrude face attribute group is too large")?;
    let inverse = 1.0_f32 / count;
    Ok(Value::Array(
        sums.into_iter().map(|sum| json!(sum * inverse)).collect(),
    ))
}

fn numeric_array(value: &Value, length: usize) -> Result<Vec<f32>> {
    let components = value
        .as_array()
        .filter(|components| components.len() == length)
        .ok_or_else(|| evaluation("extrude face vector attribute is malformed"))?;
    components
        .iter()
        .map(|component| {
            let value = component
                .as_f64()
                .ok_or_else(|| evaluation("extrude face vector attribute is malformed"))?;
            value
                .to_f32()
                .ok_or_else(|| evaluation("extrude face vector attribute is out of range"))
        })
        .collect()
}
fn integer_array(value: &Value, length: usize) -> Result<Vec<i64>> {
    let components = value
        .as_array()
        .filter(|components| components.len() == length)
        .ok_or_else(|| evaluation("extrude face integer vector attribute is malformed"))?;
    components
        .iter()
        .map(|component| {
            component
                .as_i64()
                .ok_or_else(|| evaluation("extrude face integer vector attribute is malformed"))
        })
        .collect()
}

type FloatMatrix = [[f32; 4]; 4];

fn mix_float4x4(values: &[&Value]) -> Result<Value> {
    let nested = values.first().is_some_and(|value| {
        value
            .as_array()
            .is_some_and(|rows| rows.len() == 4 && rows[0].as_array().is_some())
    });
    let mut location = Vec3::ZERO;
    let mut rotation = Vec3::ZERO;
    let mut scale = Vec3::ZERO;
    for value in values {
        let matrix = float_matrix(value)?;
        let columns = [
            Vec3::new(matrix[0][0], matrix[0][1], matrix[0][2]),
            Vec3::new(matrix[1][0], matrix[1][1], matrix[1][2]),
            Vec3::new(matrix[2][0], matrix[2][1], matrix[2][2]),
        ];
        let mut matrix_scale = Vec3::new(
            columns[0].length(),
            columns[1].length(),
            columns[2].length(),
        );
        let mut normalized = [
            normalize_column(columns[0], matrix_scale.x),
            normalize_column(columns[1], matrix_scale.y),
            normalize_column(columns[2], matrix_scale.z),
        ];
        let determinant = normalized[0].dot(normalized[1].cross(normalized[2]));
        if determinant < 0.0 {
            for column in &mut normalized {
                *column = -*column;
            }
            matrix_scale = -matrix_scale;
        }
        let euler = matrix_euler(normalized);
        let quaternion = euler_quaternion(euler);
        let exponential = provenance::quaternion_expmap(&json!(quaternion))
            .ok_or_else(|| evaluation("extrude face matrix rotation is malformed"))?;
        location += Vec3::new(matrix[3][0], matrix[3][1], matrix[3][2]);
        rotation += exponential;
        scale += matrix_scale;
    }
    let count = count_as_f32(values.len(), "extrude face matrix group is too large")?;
    let inverse = 1.0_f32 / count;
    let location = location * inverse;
    let rotation = rotation * inverse;
    let scale = scale * inverse;
    let quaternion = provenance::quaternion_from_expmap(rotation);
    Ok(float_matrix_value(
        matrix_from_transform(location, quaternion, scale),
        nested,
    ))
}

fn float_matrix(value: &Value) -> Result<FloatMatrix> {
    let values = value
        .as_array()
        .ok_or_else(|| evaluation("extrude face matrix attribute is malformed"))?;
    if values.len() == 16 {
        let mut matrix = [[0.0; 4]; 4];
        for (index, component) in values.iter().enumerate() {
            let value = component
                .as_f64()
                .ok_or_else(|| evaluation("extrude face matrix attribute is malformed"))?;
            matrix[index / 4][index % 4] = value
                .to_f32()
                .ok_or_else(|| evaluation("extrude face matrix attribute is out of range"))?;
        }
        return Ok(matrix);
    }
    if values.len() == 4 {
        let mut matrix = [[0.0; 4]; 4];
        for (row, values) in values.iter().enumerate() {
            let values = values
                .as_array()
                .filter(|values| values.len() == 4)
                .ok_or_else(|| evaluation("extrude face matrix attribute is malformed"))?;
            for (column, component) in values.iter().enumerate() {
                let value = component
                    .as_f64()
                    .ok_or_else(|| evaluation("extrude face matrix attribute is malformed"))?;
                matrix[column][row] = value
                    .to_f32()
                    .ok_or_else(|| evaluation("extrude face matrix attribute is out of range"))?;
            }
        }
        return Ok(matrix);
    }
    Err(evaluation("extrude face matrix attribute is malformed"))
}

fn float_matrix_value(matrix: FloatMatrix, nested: bool) -> Value {
    if nested {
        Value::Array(
            (0..4)
                .map(|row| Value::Array((0..4).map(|column| json!(matrix[column][row])).collect()))
                .collect(),
        )
    } else {
        Value::Array(
            (0..4)
                .flat_map(|column| (0..4).map(move |row| json!(matrix[column][row])))
                .collect(),
        )
    }
}

fn normalize_column(column: Vec3, length: f32) -> Vec3 {
    if length == 0.0 {
        Vec3::ZERO
    } else {
        column * (1.0 / length)
    }
}

fn matrix_euler(matrix: [Vec3; 3]) -> Vec3 {
    const EULER_HYPOT_EPSILON: f32 = 0.000_037_5;
    let cy = matrix[0].x.hypot(matrix[0].y);
    let first = if cy > EULER_HYPOT_EPSILON {
        Vec3::new(
            matrix[1].z.atan2(matrix[2].z),
            (-matrix[0].z).atan2(cy),
            matrix[0].y.atan2(matrix[0].x),
        )
    } else {
        Vec3::new(
            (-matrix[2].y).atan2(matrix[1].y),
            (-matrix[0].z).atan2(cy),
            0.0,
        )
    };
    let second = if cy > EULER_HYPOT_EPSILON {
        Vec3::new(
            (-matrix[1].z).atan2(-matrix[2].z),
            (-matrix[0].z).atan2(-cy),
            (-matrix[0].y).atan2(-matrix[0].x),
        )
    } else {
        first
    };
    if first.x.abs() + first.y.abs() + first.z.abs()
        > second.x.abs() + second.y.abs() + second.z.abs()
    {
        second
    } else {
        first
    }
}

fn euler_quaternion(euler: Vec3) -> [f32; 4] {
    let [x, y, z] = (euler * 0.5).to_array();
    let ci = x.cos();
    let cj = y.cos();
    let ch = z.cos();
    let si = x.sin();
    let sj = y.sin();
    let sh = z.sin();
    let cc = ci * ch;
    let cs = ci * sh;
    let sc = si * ch;
    let ss = si * sh;
    [
        cj * sc - sj * cs,
        cj * ss + sj * cc,
        cj * cs - sj * sc,
        cj * cc + sj * ss,
    ]
}

fn matrix_from_transform(location: Vec3, quaternion: [f32; 4], scale: Vec3) -> FloatMatrix {
    let [x, y, z, w] = quaternion;
    let qda = w * x;
    let qdb = w * y;
    let qdc = w * z;
    let qaa = x * x;
    let qab = x * y;
    let qac = x * z;
    let qbb = y * y;
    let qbc = y * z;
    let qcc = z * z;
    let mut matrix = [[0.0; 4]; 4];
    matrix[0][0] = 1.0 - qbb - qcc;
    matrix[0][1] = qdc + qab;
    matrix[0][2] = -qdb + qac;
    matrix[1][0] = -qdc + qab;
    matrix[1][1] = 1.0 - qaa - qcc;
    matrix[1][2] = qda + qbc;
    matrix[2][0] = qdb + qac;
    matrix[2][1] = -qda + qbc;
    matrix[2][2] = 1.0 - qaa - qbb;
    for (column, column_scale) in matrix.iter_mut().zip(scale.to_array()) {
        for value in column.iter_mut().take(3) {
            *value *= column_scale;
        }
    }
    matrix[3][0] = location.x;
    matrix[3][1] = location.y;
    matrix[3][2] = location.z;
    matrix[3][3] = 1.0;
    matrix
}

fn remove_unsupported_point_data(mesh: &mut Mesh) {
    for key in ["skin_radii", "skin_roots", "blender_skin_vertices"] {
        mesh.attributes.remove(key);
    }
}

fn count_as_f32(count: usize, message: &'static str) -> Result<f32> {
    count.to_f32().ok_or_else(|| evaluation(message))
}

fn unsupported_face_attribute_type(data_type: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("Extrude Mesh cannot mix face attribute type `{data_type}`"),
        json!({
            "feature_id": "graph.node.extrude_mesh.face_attribute_type",
            "data_type": data_type
        }),
    )
}

fn count_as_f64(count: usize, message: &'static str) -> Result<f64> {
    count.to_f64().ok_or_else(|| evaluation(message))
}

fn mesh_error(error: crate::geom::MeshError) -> PotError {
    PotError::new(crate::error::ErrorCode::EvaluationFailed, error.to_string())
}
