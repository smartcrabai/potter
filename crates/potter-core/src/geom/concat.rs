// SPDX-License-Identifier: GPL-3.0-or-later

//! Collection-instance joins. Each member keeps its own attributes, as Blender's per-object meshes
//! do: a member's mesh is placed under fresh element IDs, its ID-keyed attributes are rewritten onto
//! those IDs by [`remap_attributes`] (the same provenance rule modifiers use), and positional
//! attributes concatenate in storage order. Members that lack an attribute contribute its default.

use std::collections::HashMap;

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::normals::normal_matrix;
use super::provenance::{Origin, default_value, remap_attributes};
use super::uv_layers::uv_default;
use super::{IdCounters, Mesh};

/// One collection member: its evaluated mesh, its world matrix, and the material slots its faces
/// index.
pub(crate) struct Part<'a> {
    pub mesh: &'a Mesh,
    pub matrix: DMat4,
    /// Where this member's slot 0 sits in the combined slot table.
    pub material_offset: u32,
}

/// A placed member's attributes and element counts, the inputs of [`merge_attributes`].
struct Piece {
    attributes: Map<String, Value>,
    vertex_base: usize,
    vertices: usize,
    edges: usize,
    faces: usize,
    corners: usize,
}

/// Joins `parts` in order. Element IDs are allocated in part order, so IDs never collide between
/// members, and every ID-keyed attribute is rewritten onto the combined IDs.
///
/// # Errors
///
/// Returns `UNSUPPORTED_FEATURE` when members disagree on a mesh-level attribute that cannot be
/// merged, and `SCENE_INVALID` for a member whose attribute shape is malformed.
pub(crate) fn concatenate(parts: &[Part<'_>]) -> Result<Mesh> {
    let mut output = Mesh::new();
    let mut pieces = Vec::with_capacity(parts.len());
    for part in parts {
        let mut placed = place(part, output.next_id)?;
        pieces.push(Piece {
            attributes: std::mem::take(&mut placed.attributes),
            vertex_base: output.vertices.len(),
            vertices: placed.vertices.len(),
            edges: placed.edges.len(),
            faces: placed.faces.len(),
            corners: placed.faces.iter().map(|face| face.vertices.len()).sum(),
        });
        output.next_id = placed.next_id;
        output.vertices.append(&mut placed.vertices);
        output.edges.append(&mut placed.edges);
        output.faces.append(&mut placed.faces);
    }
    output.attributes = merge_attributes(&pieces)?;
    Ok(output)
}

/// Places one member's mesh under IDs counted from `first_ids`, with its attributes remapped and its
/// vertices moved to the instance space. Custom normals rotate with the member's normal matrix.
fn place(part: &Part<'_>, first_ids: IdCounters) -> Result<Mesh> {
    let source = part.mesh;
    let mut mesh = Mesh {
        next_id: first_ids,
        ..Mesh::new()
    };
    let mut vertex_ids = HashMap::with_capacity(source.vertices.len());
    for vertex in &source.vertices {
        let id = mesh
            .insert_vertex(vertex.co)
            .map_err(|error| join_error(&error))?;
        vertex_ids.insert(vertex.id, id);
    }
    for edge in &source.edges {
        let [first, second] = edge.vertices.map(|id| vertex_ids.get(&id).copied());
        let (Some(first), Some(second)) = (first, second) else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "instance edge references a missing vertex",
            ));
        };
        mesh.insert_edge([first, second])
            .map_err(|error| join_error(&error))?;
    }
    for face in &source.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                vertex_ids.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "instance face references a missing vertex",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let material_index = face
            .material_index
            .checked_add(part.material_offset)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "instance material slot table exceeds the u32 range",
                )
            })?;
        mesh.insert_face(vertices, material_index)
            .map_err(|error| join_error(&error))?;
    }
    // Flat fallbacks are measured in member space; the transform below carries them with the mesh.
    let origin = Origin::copies(
        (0..source.vertices.len()).map(Some),
        (0..source.faces.len()).map(Some).collect(),
    );
    remap_attributes(source, &mut mesh, &origin)?;
    let normals = normal_matrix(part.matrix);
    for vertex in &mut mesh.vertices {
        vertex.co = part.matrix.transform_point3(vertex.co);
    }
    if let Some(values) = mesh
        .attributes
        .get_mut("custom_normal")
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
    {
        for corners in values.values_mut() {
            *corners = rotate_corner_normals(corners, normals);
        }
    }
    Ok(mesh)
}

fn join_error(error: &crate::geom::MeshError) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, error.to_string())
}

/// Rotates `custom_normal` corner vectors into instance space. Malformed entries pass through
/// unchanged so the normal computation rejects them.
pub(crate) fn rotate_corner_normals(value: &Value, normal_matrix: DMat3) -> Value {
    let Some(corners) = value.as_array() else {
        return value.clone();
    };
    Value::Array(
        corners
            .iter()
            .map(|corner| {
                let vector = corner
                    .as_array()
                    .and_then(|components| match components.as_slice() {
                        [x, y, z] => Some(DVec3::new(x.as_f64()?, y.as_f64()?, z.as_f64()?)),
                        _ => None,
                    });
                match vector {
                    Some(vector) => json!((normal_matrix * vector).normalize_or_zero().to_array()),
                    None => corner.clone(),
                }
            })
            .collect(),
    )
}

/// One member's value of an attribute, if it has one, with the member's piece.
type Member<'a> = (Option<&'a Value>, &'a Piece);

/// Merges every member's mesh-level attributes. An attribute present on some members only is
/// filled with defaults for the others, so positional data stays aligned.
fn merge_attributes(pieces: &[Piece]) -> Result<Map<String, Value>> {
    let mut names: Vec<&str> = Vec::new();
    for piece in pieces {
        for name in piece.attributes.keys() {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
    }
    let mut merged = Map::new();
    for name in names {
        let members: Vec<Member<'_>> = pieces
            .iter()
            .map(|piece| (piece.attributes.get(name), piece))
            .collect();
        merged.insert(name.to_owned(), merge_attribute(name, &members)?);
    }
    Ok(merged)
}

fn merge_attribute(name: &str, members: &[Member<'_>]) -> Result<Value> {
    match name {
        "uv_map" => concat_arrays(name, members),
        "blender_uv_layers" => merge_entries(
            name,
            members,
            |entry| text(entry, "name").to_owned(),
            |piece, _| Some(piece.corners),
            |_| uv_default(),
        ),
        "blender_attributes" => merge_entries(
            name,
            members,
            |entry| format!("{}\u{0}{}", text(entry, "domain"), text(entry, "name")),
            |piece, entry| match text(entry, "domain") {
                "POINT" => Some(piece.vertices),
                "EDGE" => Some(piece.edges),
                "FACE" => Some(piece.faces),
                "CORNER" => Some(piece.corners),
                _ => None,
            },
            |entry| default_value(text(entry, "data_type")),
        ),
        "blender_edge_flags" => merge_positional(
            name,
            members,
            |piece| piece.edges,
            &json!({"sharp": false, "seam": false}),
        ),
        "blender_polygon_smooth" => {
            merge_positional(name, members, |piece| piece.faces, &json!(false))
        }
        "blender_skin_vertices" => {
            merge_positional(name, members, |piece| piece.vertices, &json!({}))
        }
        "blender_metadata" => merge_metadata(name, members),
        "skin_radii" | "skin_roots" => merge_vertex_objects(name, members),
        "vertex_groups" => merge_vertex_groups(name, members),
        _ if members
            .iter()
            .any(|(value, _)| value.is_some_and(|value| value.get("values").is_some())) =>
        {
            merge_named(name, members)
        }
        _ => merge_equal(name, members),
    }
}

fn shape_error(name: &str) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        format!("instance member attribute `{name}` has an invalid shape"),
        json!({ "attribute": name }),
    )
}

fn conflict_error(name: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("collection members disagree on attribute `{name}`, which cannot be combined"),
        json!({ "attribute": name }),
    )
}

fn duplicate_error(name: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InternalError,
        format!("instance attribute `{name}` repeats an element key"),
        json!({ "attribute": name }),
    )
}

fn text<'a>(entry: &'a Value, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// Arrays of entries, each member's entries in member order (`uv_map`).
fn concat_arrays(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut output = Vec::new();
    for (value, _) in members {
        if let Some(value) = value {
            output.extend(
                value
                    .as_array()
                    .ok_or_else(|| shape_error(name))?
                    .iter()
                    .cloned(),
            );
        }
    }
    Ok(Value::Array(output))
}

/// Positional values, one per element of the member's domain (`length`); a member without the
/// attribute contributes `default` for each of its elements.
fn merge_positional(
    name: &str,
    members: &[Member<'_>],
    length: impl Fn(&Piece) -> usize,
    default: &Value,
) -> Result<Value> {
    let mut output = Vec::new();
    for (value, piece) in members {
        let count = length(piece);
        match value {
            Some(value) => {
                let values = value
                    .as_array()
                    .filter(|values| values.len() == count)
                    .ok_or_else(|| shape_error(name))?;
                output.extend(values.iter().cloned());
            }
            None => output.extend(std::iter::repeat_n(default.clone(), count)),
        }
    }
    Ok(Value::Array(output))
}

/// Named entries (`blender_uv_layers` by `name`, `blender_attributes` by domain and name) keep the
/// first member's header. Each member's values concatenate under its own entry, and a member
/// without the entry contributes the entry's default values.
fn merge_entries(
    name: &str,
    members: &[Member<'_>],
    key: impl Fn(&Value) -> String,
    length: impl Fn(&Piece, &Value) -> Option<usize>,
    default: impl Fn(&Value) -> Value,
) -> Result<Value> {
    // First-seen order, with each key's header entry.
    let mut order: Vec<(String, &Value)> = Vec::new();
    for (value, _) in members {
        for entry in value.and_then(Value::as_array).into_iter().flatten() {
            let entry_key = key(entry);
            if !order.iter().any(|(known, _)| *known == entry_key) {
                order.push((entry_key, entry));
            }
        }
    }
    let mut output = Vec::with_capacity(order.len());
    for (entry_key, header) in order {
        let mut values = Vec::new();
        for (value, piece) in members {
            let found = value
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|entry| key(entry) == entry_key);
            let count = length(piece, header).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("instance attribute `{name}` uses a domain that cannot be combined"),
                    json!({ "attribute": name }),
                )
            })?;
            if let Some(entry) = found {
                if entry.get("data_type") != header.get("data_type") {
                    return Err(conflict_error(name));
                }
                let entry_values = entry
                    .get("values")
                    .and_then(Value::as_array)
                    .filter(|entry_values| entry_values.len() == count)
                    .ok_or_else(|| shape_error(name))?;
                values.extend(entry_values.iter().cloned());
            } else {
                let fill = default(header);
                values.extend(std::iter::repeat_n(fill, count));
            }
        }
        let mut merged = header.as_object().cloned().unwrap_or_default();
        merged.insert("values".to_owned(), Value::Array(values));
        output.push(Value::Object(merged));
    }
    Ok(Value::Array(output))
}

/// Vertex-indexed objects keyed by storage index (`skin_radii`, `skin_roots`). Each member's keys
/// move past the vertices of the members before it.
fn merge_vertex_objects(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut output = Map::new();
    for (value, piece) in members {
        let Some(value) = value else { continue };
        for (key, entry) in value.as_object().ok_or_else(|| shape_error(name))? {
            let index: usize = key.parse().map_err(|_| shape_error(name))?;
            if output
                .insert((piece.vertex_base + index).to_string(), entry.clone())
                .is_some()
            {
                return Err(duplicate_error(name));
            }
        }
    }
    Ok(Value::Object(output))
}

/// `vertex_groups`: group name to `v{id}` weights. Groups union across members; IDs never repeat.
fn merge_vertex_groups(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut groups = Map::new();
    for (value, _) in members {
        let Some(value) = value else { continue };
        for (group, weights) in value.as_object().ok_or_else(|| shape_error(name))? {
            let weights = weights.as_object().ok_or_else(|| shape_error(name))?;
            let target = groups
                .entry(group.clone())
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .ok_or_else(|| shape_error(name))?;
            for (id, weight) in weights {
                if target.insert(id.clone(), weight.clone()).is_some() {
                    return Err(duplicate_error(name));
                }
            }
        }
    }
    Ok(Value::Object(groups))
}

/// `{"domain","type","values":{prefix+id: value}}` attributes. Element IDs never repeat across
/// members, so the values union; the header fields must agree.
fn merge_named(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut header: Option<Map<String, Value>> = None;
    let mut values = Map::new();
    for (value, _) in members {
        let Some(value) = value else { continue };
        let object = value.as_object().ok_or_else(|| shape_error(name))?;
        let entries = object
            .get("values")
            .and_then(Value::as_object)
            .ok_or_else(|| shape_error(name))?;
        let fields: Map<String, Value> = object
            .iter()
            .filter(|(key, _)| key.as_str() != "values")
            .map(|(key, field)| (key.clone(), field.clone()))
            .collect();
        match &header {
            None => header = Some(fields),
            Some(first) if *first != fields => return Err(conflict_error(name)),
            Some(_) => {}
        }
        for (key, entry) in entries {
            if values.insert(key.clone(), entry.clone()).is_some() {
                return Err(duplicate_error(name));
            }
        }
    }
    let mut merged = header.unwrap_or_default();
    merged.insert("values".to_owned(), Value::Object(values));
    Ok(Value::Object(merged))
}

/// `blender_metadata` describes each member's data-block. The joined mesh keeps every member's custom
/// properties (a key set to different values on two members is refused); the data-block's users and
/// fake-user flags belong to no single joined mesh and are not carried.
fn merge_metadata(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut custom = Map::new();
    for (value, _) in members {
        let Some(value) = *value else { continue };
        let Some(properties) = value.get("custom_properties").and_then(Value::as_object) else {
            continue;
        };
        for (key, property) in properties {
            match custom.get(key) {
                Some(known) if known != property => return Err(conflict_error(name)),
                Some(_) => {}
                None => {
                    custom.insert(key.clone(), property.clone());
                }
            }
        }
    }
    Ok(json!({ "custom_properties": custom }))
}

/// Mesh-level data with no element keys must agree across members.
fn merge_equal(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut first: Option<&Value> = None;
    for (value, _) in members {
        let Some(value) = *value else { continue };
        match first {
            None => first = Some(value),
            Some(known) if known != value => return Err(conflict_error(name)),
            Some(_) => {}
        }
    }
    Ok(first.cloned().unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "small mesh fixtures")]

    use glam::{DMat4, DVec3};
    use serde_json::{Value, json};

    use super::{Part, concatenate};
    use crate::geom::{BoxParams, Mesh};

    fn cube() -> Mesh {
        Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap()
    }

    #[test]
    fn members_keep_their_own_ids_uvs_and_positional_flags() {
        let mut first = cube();
        let first_face = first.faces[0].id;
        let first_edge = first.edges[0].id;
        first.attributes.insert(
            "uv_map".to_owned(),
            json!([{"face_id": first_face, "uv": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}]),
        );
        first.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "values": {format!("e{first_edge}"): true}}),
        );
        first.attributes.insert(
            "blender_polygon_smooth".to_owned(),
            json!([true, false, false, false, false, false]),
        );
        let mut second = cube();
        let second_face = second.faces[5].id;
        second.attributes.insert(
            "uv_map".to_owned(),
            json!([{"face_id": second_face, "uv": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]}]),
        );
        let parts = [
            Part {
                mesh: &first,
                matrix: DMat4::IDENTITY,
                material_offset: 0,
            },
            Part {
                mesh: &second,
                matrix: DMat4::from_translation(DVec3::X * 4.0),
                material_offset: 1,
            },
        ];
        let joined = concatenate(&parts).unwrap();
        assert_eq!(joined.vertices.len(), 16);
        assert_eq!(joined.faces.len(), 12);
        assert_eq!(joined.faces[6].material_index, 1);

        // The second member's UV entry follows its faces to their combined IDs.
        let uv = joined.attributes["uv_map"].as_array().unwrap();
        assert_eq!(uv.len(), 2);
        assert_eq!(uv[0]["face_id"], json!(first_face));
        assert_eq!(uv[1]["face_id"], json!(joined.faces[11].id));

        // The first member's seam keeps its combined edge ID; the second member has none.
        let seam = joined.attributes["seam"]["values"].as_object().unwrap();
        assert_eq!(seam.len(), 1);
        assert_eq!(seam[&format!("e{first_edge}")], json!(true));

        // Positional flags of the member without them default to flat.
        let smooth = joined.attributes["blender_polygon_smooth"]
            .as_array()
            .unwrap();
        assert_eq!(smooth.len(), 12);
        assert_eq!(smooth[0], json!(true));
        assert!(smooth[6..].iter().all(|flag| *flag == json!(false)));

        // Each member's vertices move with its matrix.
        assert!(joined.vertices[8..].iter().all(|vertex| vertex.co.x > 2.9));
    }

    #[test]
    fn member_colors_and_vertex_groups_follow_their_source_elements() {
        let mut first = cube();
        let mut second = cube();
        first.attributes.insert(
            "color".to_owned(),
            json!({"domain":"point","type":"color","values":{"v0":[1.0,0.0,0.0,1.0]}}),
        );
        second.attributes.insert(
            "color".to_owned(),
            json!({"domain":"point","type":"color","values":{"v0":[0.0,0.0,1.0,1.0]}}),
        );
        first
            .attributes
            .insert("vertex_groups".to_owned(), json!({"Weight":{"v0":0.25}}));
        second
            .attributes
            .insert("vertex_groups".to_owned(), json!({"Weight":{"v0":0.75}}));
        first.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name":"LoopColor",
                "domain":"CORNER",
                "data_type":"FLOAT_COLOR",
                "values":std::iter::repeat_n(json!([1.0,0.0,0.0,1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        second.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name":"LoopColor",
                "domain":"CORNER",
                "data_type":"FLOAT_COLOR",
                "values":std::iter::repeat_n(json!([0.0,0.0,1.0,1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        let joined = concatenate(&[
            Part {
                mesh: &first,
                matrix: DMat4::IDENTITY,
                material_offset: 0,
            },
            Part {
                mesh: &second,
                matrix: DMat4::IDENTITY,
                material_offset: 1,
            },
        ])
        .unwrap();
        let point_colors = joined.attributes["color"]["values"].as_object().unwrap();
        assert_eq!(point_colors["v0"], json!([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(point_colors["v8"], json!([0.0, 0.0, 1.0, 1.0]));
        let group = &joined.attributes["vertex_groups"]["Weight"];
        assert_eq!(group["v0"], json!(0.25));
        assert_eq!(group["v8"], json!(0.75));
        let corner_colors = joined.attributes["blender_attributes"][0]["values"]
            .as_array()
            .unwrap();
        assert_eq!(corner_colors.len(), 48);
        assert_eq!(corner_colors[0], json!([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(corner_colors[24], json!([0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn members_join_custom_properties_and_refuse_conflicts() {
        let metadata = |custom: Value| json!({"custom_properties": custom, "users": 1});
        let mut first = cube();
        first
            .attributes
            .insert("blender_metadata".to_owned(), metadata(json!({"tag": 1})));
        let mut second = cube();
        second.attributes.insert(
            "blender_metadata".to_owned(),
            metadata(json!({"part": "hat"})),
        );
        let join = |first: &Mesh, second: &Mesh| {
            concatenate(&[
                Part {
                    mesh: first,
                    matrix: DMat4::IDENTITY,
                    material_offset: 0,
                },
                Part {
                    mesh: second,
                    matrix: DMat4::IDENTITY,
                    material_offset: 1,
                },
            ])
        };
        let joined = join(&first, &second).unwrap();
        assert_eq!(
            joined.attributes["blender_metadata"],
            json!({"custom_properties": {"tag": 1, "part": "hat"}})
        );

        second
            .attributes
            .insert("blender_metadata".to_owned(), metadata(json!({"tag": 2})));
        let error = join(&first, &second).unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
    }
}
