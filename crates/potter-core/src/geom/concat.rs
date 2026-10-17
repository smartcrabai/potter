// SPDX-License-Identifier: GPL-3.0-or-later

//! Collection-instance joins. Each member keeps its own attributes, as Blender's per-object meshes
//! do: a member's mesh is placed under fresh element IDs, its ID-keyed attributes are rewritten onto
//! those IDs by [`remap_attributes`] (the same provenance rule modifiers use), and positional
//! attributes concatenate in storage order. Members that lack an attribute contribute its default.
//!
//! UV layers form one canonical V-down store and join by name in first-seen order. A layer keeps
//! each member's values and pins by that name; a missing layer or face row contributes Blender's
//! default UV(0, 0), stored canonically as `[0, 1]`. The evaluator retains source-face coverage
//! separately so synthesized defaults do not become available member data.
//!
//! A member whose transform mirrors (negative determinant) reverses each face's corner order together
//! with every corner-keyed value of that face. Its normals rotate with its normal matrix. A face's
//! material index clamps to the member's last slot, as Blender clamps at export, so no slot table is
//! padded.
//!
//! Smooth and sharp flags are written to each member's native `shade_smooth` and `sharp` keys before
//! the join, so one member's native key never hides another member's flags.

use std::borrow::Cow;
use std::collections::HashMap;

use glam::DMat4;
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::clnor::{self, NormalCodeOptions, SetterInput};

use super::normals::{normal_matrix, sharp_edge_flags, smooth_face_flags};
use super::provenance::{Origin, default_value, remap_attributes_skipping};
use super::uv_layers::{UvLayer, read_layers, write_layers};
use super::{Edge, IdCounters, Mesh, MeshError, Vertex};

/// One collection member: its evaluated mesh (borrowed for plain members, owned for nested joins),
/// its world matrix, and the material slots its faces index.
pub(crate) struct Part<'a> {
    pub mesh: Cow<'a, Mesh>,
    /// Member world matrix.
    pub matrix: DMat4,
    /// Where the member's slot 0 sits in the combined slot table.
    pub material_offset: u32,
    /// Length of the member's slot table: at least one, since an empty material list is one empty slot.
    pub slot_count: usize,
}

/// A member placed under joined IDs: its attributes are remapped, and its canonical UV layers stay
/// in Potter's V-down convention. Legacy `uv_map` and bridge-only `blender_uv_layers` are excluded.
struct Placed {
    mesh: Mesh,
    uv_layers: Vec<UvLayer>,
}

/// A placed member's attributes and element counts, the inputs of [`merge_attributes`] and
/// [`join_uv_layers`].
struct Piece {
    attributes: Map<String, Value>,
    uv_layers: Vec<UvLayer>,
    /// The member's `active` and `render` color layer names as it stated them. Placement writes a
    /// null designation as the first layer, so the join reads these from the source mesh.
    color_active: Option<String>,
    color_render: Option<String>,
    vertex_base: usize,
    vertices: usize,
    edges: usize,
    face_base: usize,
    faces: usize,
    corners: usize,
}

/// Joins `parts` in order. Element IDs are allocated in part order, so IDs never collide between
/// members, and every ID-keyed attribute is rewritten onto the combined IDs. The joined mesh is
/// validated once, after every member is placed.
///
/// # Errors
///
/// Returns `UNSUPPORTED_FEATURE` when members disagree on an attribute that cannot be combined (its
/// details name the attribute), and `SCENE_INVALID` for a malformed member. Returns
/// `EVALUATION_FAILED` when the joined mesh is not valid.
pub(crate) fn concatenate(parts: Vec<Part<'_>>) -> Result<Mesh> {
    let mut output = Mesh::new();
    let mut pieces = Vec::with_capacity(parts.len());
    for part in parts {
        let Placed {
            mut mesh,
            uv_layers,
        } = place(&part, output.next_id)?;
        pieces.push(Piece {
            attributes: std::mem::take(&mut mesh.attributes),
            uv_layers,
            color_active: color_designation(&part.mesh, "active"),
            color_render: color_designation(&part.mesh, "render"),
            vertex_base: output.vertices.len(),
            vertices: mesh.vertices.len(),
            edges: mesh.edges.len(),
            face_base: output.faces.len(),
            faces: mesh.faces.len(),
            corners: mesh.faces.iter().map(|face| face.vertices.len()).sum(),
        });
        output.next_id = mesh.next_id;
        output.vertices.append(&mut mesh.vertices);
        output.edges.append(&mut mesh.edges);
        output.faces.append(&mut mesh.faces);
    }
    output.attributes = merge_attributes(&pieces, &output)?;
    let layers = join_uv_layers(&mut pieces, &output.faces)?;
    if !layers.is_empty() {
        write_layers(&mut output, &layers);
    }
    output.validate().map_err(|error| join_error(&error))?;
    Ok(output)
}

/// Places one member's mesh under IDs counted from `first_ids`. Elements are built in one pass over
/// the source, attributes are remapped and shading canonicalised, and UVs are read. A mirrored member
/// reverses its corners. Vertices move to the instance space last, and custom normals rotate with
/// the member's normal matrix.
fn place(part: &Part<'_>, first_ids: IdCounters) -> Result<Placed> {
    let source: &Mesh = &part.mesh;
    let mut mesh = Mesh {
        next_id: first_ids,
        ..Mesh::new()
    };
    let mut vertex_ids = HashMap::with_capacity(source.vertices.len());
    mesh.vertices.reserve_exact(source.vertices.len());
    for vertex in &source.vertices {
        let id = take_id(&mut mesh.next_id.vertex)?;
        vertex_ids.insert(vertex.id, id);
        mesh.vertices.push(Vertex { id, co: vertex.co });
    }
    mesh.edges.reserve_exact(source.edges.len());
    for edge in &source.edges {
        let [first, second] = edge.vertices.map(|id| vertex_ids.get(&id).copied());
        let (Some(first), Some(second)) = (first, second) else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "instance edge references a missing vertex",
            ));
        };
        let id = take_id(&mut mesh.next_id.edge)?;
        mesh.edges.push(Edge {
            id,
            vertices: [first, second],
        });
    }
    let last_slot = u32::try_from(part.slot_count.saturating_sub(1)).unwrap_or(u32::MAX);
    let faces = source
        .faces
        .iter()
        .map(|face| -> Result<(Vec<u32>, u32)> {
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
                .min(last_slot)
                .checked_add(part.material_offset)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "instance material slot table exceeds the u32 range",
                    )
                })?;
            Ok((vertices, material_index))
        })
        .collect::<Result<Vec<_>>>()?;
    mesh.insert_faces(faces.into_iter().map(Ok))
        .map_err(|error| join_error(&error))?;
    // Flat fallbacks are measured in member space; the transform below carries them with the mesh.
    let origin = Origin::copies(
        (0..source.vertices.len()).map(Some),
        (0..source.faces.len()).map(Some).collect(),
    );
    // UV stores are excluded from attribute remapping and rebuilt as one named union below.
    remap_attributes_skipping(
        source,
        &mut mesh,
        &origin,
        &["uv_map", "uv_layers", "blender_uv_layers"],
    )?;
    canonical_shading(source, &mut mesh)?;
    let mut uv_layers = read_layers(source);
    // Transform the evaluated fan normal for Auto corners as well as authored codes.
    let mut custom_normals = clnor::corner_codes(&mesh)?
        .map(|codes| {
            let source_positions = mesh
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>();
            clnor::evaluate_codes(
                &mesh,
                &codes,
                &source_positions,
                NormalCodeOptions::default(),
            )
            .map(|normals| normals.into_iter().map(Some).collect::<Vec<_>>())
        })
        .transpose()?;
    let transform_normals = normal_matrix(part.matrix);
    if let Some(normals) = &mut custom_normals {
        for normal in normals.iter_mut().flatten() {
            *normal = (transform_normals * *normal).normalize_or_zero();
        }
    }
    for vertex in &mut mesh.vertices {
        vertex.co = part.matrix.transform_point3(vertex.co);
    }
    if part.matrix.determinant() < 0.0 {
        let face_sizes = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .collect::<Vec<_>>();
        mirror_corners(&mut mesh, &mut uv_layers);
        if let Some(normals) = &mut custom_normals {
            reverse_face_segments(normals, &face_sizes);
        }
    }
    if let Some(normals) = custom_normals {
        let corner_order = clnor::corner_order(&mesh)?;
        clnor::encode_vectors(
            &mut mesh,
            Some(&normals),
            NormalCodeOptions {
                corner_order: corner_order.as_deref(),
                setter_input: SetterInput::NormalizeInput,
            },
        )?;
    }
    Ok(Placed { mesh, uv_layers })
}

/// Takes the next element ID from `counter`.
fn take_id(counter: &mut u32) -> Result<u32> {
    let id = *counter;
    *counter = id
        .checked_add(1)
        .ok_or_else(|| join_error(&MeshError::IdExhausted))?;
    Ok(id)
}

fn join_error(error: &MeshError) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, error.to_string())
}

/// Writes the member's flags as native `shade_smooth` (per face) and `sharp` (per edge) keys, which
/// the joined mesh reads. An imported `.blend` member keeps its flags only in `blender_*` keys.
fn canonical_shading(source: &Mesh, mesh: &mut Mesh) -> Result<()> {
    if ["shade_smooth", "blender_polygon_smooth"]
        .iter()
        .any(|key| source.attributes.contains_key(*key))
    {
        let flags = member_smooth_flags(source)?;
        let values: Map<String, Value> = mesh
            .faces
            .iter()
            .zip(flags)
            .map(|(face, flag)| (format!("f{}", face.id), json!(flag)))
            .collect();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": values}),
        );
    }
    if ["sharp", "blender_edge_flags"]
        .iter()
        .any(|key| source.attributes.contains_key(*key))
    {
        let flags = member_sharp_flags(source)?;
        let values: Map<String, Value> = mesh
            .edges
            .iter()
            .zip(flags)
            .map(|(edge, flag)| (format!("e{}", edge.id), json!(flag)))
            .collect();
        mesh.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": values}),
        );
    }
    Ok(())
}

/// A member's smooth flag per face, read as stored. A member without a `shade_smooth` key may carry
/// the legacy `blender_polygon_smooth` array (one bool per face) instead, which
/// [`smooth_face_flags`] does not read.
fn member_smooth_flags(source: &Mesh) -> Result<Vec<bool>> {
    if !source.attributes.contains_key("shade_smooth")
        && let Some(flags) = source
            .attributes
            .get("blender_polygon_smooth")
            .and_then(Value::as_array)
            .filter(|flags| flags.len() == source.faces.len())
            .and_then(|flags| flags.iter().map(Value::as_bool).collect::<Option<Vec<bool>>>())
    {
        return Ok(flags);
    }
    smooth_face_flags(source)
}

/// A member's sharp flag per edge, read as stored. A member without a `sharp` key may carry the
/// legacy `blender_edge_flags` array (one `{sharp, seam}` object per edge) instead, which
/// [`sharp_edge_flags`] does not read.
fn member_sharp_flags(source: &Mesh) -> Result<Vec<bool>> {
    if !source.attributes.contains_key("sharp")
        && let Some(edges) = source
            .attributes
            .get("blender_edge_flags")
            .and_then(Value::as_array)
            .filter(|edges| edges.len() == source.edges.len())
    {
        return Ok(edges
            .iter()
            .map(|edge| edge.get("sharp").and_then(Value::as_bool).unwrap_or(false))
            .collect());
    }
    sharp_edge_flags(source)
}

/// Reverses a mirrored member's faces with every corner-keyed value: the corner order, `custom_normal`
/// and named `corner` rows, `CORNER` `blender_attributes` segments, and UV rows and pin flags. A
/// corner keeps its vertex, UV, pin flag, and normal together.
fn mirror_corners(mesh: &mut Mesh, layers: &mut [UvLayer]) {
    let sizes: Vec<usize> = mesh.faces.iter().map(|face| face.vertices.len()).collect();
    for (key, attribute) in &mut mesh.attributes {
        let corner_keyed = key.as_str() == "custom_normal"
            || attribute
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"));
        if !corner_keyed {
            continue;
        }
        if let Some(rows) = attribute.get_mut("values").and_then(Value::as_object_mut) {
            for row in rows.values_mut().filter_map(Value::as_array_mut) {
                row.reverse();
            }
        }

        if key == "custom_normal"
            && let Some(rows) = attribute
                .get_mut("corner_order")
                .and_then(Value::as_object_mut)
        {
            for row in rows.values_mut().filter_map(Value::as_array_mut) {
                row.reverse();
            }
        }
    }
    if let Some(entries) = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
    {
        for entry in entries
            .iter_mut()
            .filter(|entry| entry.get("domain").and_then(Value::as_str) == Some("CORNER"))
        {
            if let Some(values) = entry.get_mut("values").and_then(Value::as_array_mut) {
                reverse_face_segments(values, &sizes);
            }
        }
    }
    for layer in layers.iter_mut() {
        for row in layer.faces.iter_mut().flatten() {
            row.reverse();
        }
        for row in layer.pinned.iter_mut().flatten() {
            row.reverse();
        }
    }
    for face in &mut mesh.faces {
        face.vertices.reverse();
    }
}

/// Reverses each face's run of `values`, where face `f` owns `sizes[f]` consecutive entries.
fn reverse_face_segments<T>(values: &mut [T], sizes: &[usize]) {
    let mut start = 0;
    for size in sizes {
        let end = start + size;
        if let Some(segment) = values.get_mut(start..end) {
            segment.reverse();
        }
        start = end;
    }
}

/// Joins canonical UV layers by first-seen name, preserving rows and pins from matching members.
/// Missing layers and face rows receive the canonical default; unions over the mesh limit fail.
fn join_uv_layers(pieces: &mut [Piece], faces: &[super::Face]) -> Result<Vec<UvLayer>> {
    let mut names = Vec::new();
    for piece in pieces.iter() {
        for layer in &piece.uv_layers {
            if !names.contains(&layer.name) {
                names.push(layer.name.clone());
            }
        }
    }
    if names.len() > super::uv_layers::MAX_LAYERS {
        return Err(PotError::with_details(
            ErrorCode::LimitExceeded,
            "collection instance UV layers exceed the mesh limit",
            json!({"layer_count":names.len(),"limit":super::uv_layers::MAX_LAYERS}),
        ));
    }
    let mut joined = Vec::with_capacity(names.len());
    for name in names {
        let mut layer_faces = Vec::new();
        let mut pinned = Vec::new();
        for piece in pieces.iter_mut() {
            let member_faces = &faces[piece.face_base..piece.face_base + piece.faces];
            if let Some(layer) = piece.uv_layers.iter_mut().find(|layer| layer.name == name) {
                for (face_index, face) in member_faces.iter().enumerate() {
                    let corners = layer
                        .faces
                        .get_mut(face_index)
                        .and_then(Option::take)
                        .unwrap_or_else(|| vec![[0.0, 1.0]; face.vertices.len()]);
                    layer_faces.push(Some(corners));
                    pinned.push(layer.pinned.get_mut(face_index).and_then(Option::take));
                }
            } else {
                for face in member_faces {
                    layer_faces.push(Some(vec![[0.0, 1.0]; face.vertices.len()]));
                    pinned.push(None);
                }
            }
        }
        joined.push(UvLayer {
            name,
            faces: layer_faces,
            pinned,
        });
    }
    Ok(joined)
}

/// One member's value of an attribute, if it has one, with the member's piece.
type Member<'a> = (Option<&'a Value>, &'a Piece);

/// Merges every member's mesh-level attributes. An attribute present on some members only is
/// filled with defaults for the others, so positional data stays aligned.
fn merge_attributes(pieces: &[Piece], mesh: &Mesh) -> Result<Map<String, Value>> {
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
        let value = merge_attribute(name, &members, mesh)?;
        if name == "custom_normal"
            && value.get("values").is_none()
            && value.get("corner_order").is_none()
        {
            continue;
        }
        merged.insert(name.to_owned(), value);
    }
    Ok(merged)
}

fn merge_attribute(name: &str, members: &[Member<'_>], mesh: &Mesh) -> Result<Value> {
    let attribute_name = super::attribute_keys::user_name(name).unwrap_or(name);
    match name {
        "custom_normal" => merge_custom_normal(members, mesh),
        "blender_attributes" => merge_blender_attributes(name, members),
        "color_layers" => merge_color_layers(name, members),
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
        _ if members.iter().any(|(value, _)| {
            value.is_some_and(|value| value.get("type").and_then(Value::as_str) == Some("color"))
        }) && members.iter().all(|(value, piece)| {
            let Some(value) = value else { return true };
            value.get("type").and_then(Value::as_str) == Some("color")
                && matches!(
                    value.get("domain").and_then(Value::as_str),
                    Some("point" | "corner")
                )
                && piece
                    .attributes
                    .get("color_layers")
                    .and_then(|layers| layers.get("layers"))
                    .and_then(Value::as_array)
                    .is_some_and(|layers| {
                        layers
                            .iter()
                            .any(|layer| layer.as_str() == Some(attribute_name))
                    })
        }) =>
        {
            merge_color_attribute(attribute_name, members, mesh)
        }
        _ if members
            .iter()
            .any(|(value, _)| value.is_some_and(|value| value.get("values").is_some())) =>
        {
            merge_named(name, members)
        }
        _ => merge_equal(name, members),
    }
}

/// Joins typed custom-normal codes and their member-local loop bases without decoding the codes.
/// Missing code layers contribute Auto; a non-identity basis is offset into the joined loop order.
fn merge_custom_normal(members: &[Member<'_>], mesh: &Mesh) -> Result<Value> {
    let has_values = members
        .iter()
        .any(|(value, _)| value.is_some_and(|value| value.get("values").is_some()));
    let has_order = members
        .iter()
        .any(|(value, _)| value.is_some_and(|value| value.get("corner_order").is_some()));
    let mut values = has_values.then(Map::new);
    let mut corner_order = has_order.then(Map::new);
    let mut has_nonidentity_order = false;
    let mut preceding_corners = 0;

    for (attribute, piece) in members {
        let attribute = *attribute;
        let member_values = attribute
            .and_then(|attribute| attribute.get("values"))
            .map(|rows| rows.as_object().ok_or_else(|| shape_error("custom_normal")))
            .transpose()?;
        let member_order = attribute
            .and_then(|attribute| attribute.get("corner_order"))
            .map(|rows| rows.as_object().ok_or_else(|| shape_error("custom_normal")))
            .transpose()?;
        let mut member_corner = 0;
        for face in &mesh.faces[piece.face_base..piece.face_base + piece.faces] {
            let key = format!("f{}", face.id);
            if let Some(values) = &mut values {
                let row = if let Some(rows) = member_values {
                    rows.get(&key)
                        .and_then(Value::as_array)
                        .filter(|row| row.len() == face.vertices.len())
                        .cloned()
                        .ok_or_else(|| shape_error("custom_normal"))?
                } else {
                    vec![Value::Null; face.vertices.len()]
                };
                values.insert(key.clone(), Value::Array(row));
            }
            if let Some(corner_order) = &mut corner_order {
                let natural_start = preceding_corners + member_corner;
                let row = if let Some(rows) = member_order {
                    rows.get(&key)
                        .and_then(Value::as_array)
                        .filter(|row| row.len() == face.vertices.len())
                        .ok_or_else(|| shape_error("custom_normal"))?
                        .iter()
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|ordinal| usize::try_from(ordinal).ok())
                                .filter(|ordinal| *ordinal < piece.corners)
                                .map(|ordinal| ordinal + preceding_corners)
                                .ok_or_else(|| shape_error("custom_normal"))
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    (0..face.vertices.len())
                        .map(|corner| natural_start + corner)
                        .collect()
                };
                has_nonidentity_order |= row
                    .iter()
                    .enumerate()
                    .any(|(corner, ordinal)| *ordinal != natural_start + corner);
                corner_order.insert(
                    key,
                    Value::Array(row.into_iter().map(Value::from).collect()),
                );
            }
            member_corner += face.vertices.len();
        }
        preceding_corners += piece.corners;
    }

    let mut merged = Map::new();
    merged.insert("domain".to_owned(), json!("corner"));
    merged.insert("type".to_owned(), json!("normal_code"));
    if let Some(values) = values {
        merged.insert("values".to_owned(), Value::Object(values));
    }
    if let Some(corner_order) = corner_order.filter(|_| has_nonidentity_order) {
        merged.insert("corner_order".to_owned(), Value::Object(corner_order));
    }
    Ok(Value::Object(merged))
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

/// `blender_attributes` entries, keyed by domain and name. Each member's values concatenate in
/// member order, and a member without the entry contributes the default of the joined data type.
/// UV-named mirrors are omitted because the canonical store already carries these corner values.
/// Keeping both would create a second live UV store on the joined mesh.
fn merge_blender_attributes(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let uv_names: Vec<&str> = members
        .iter()
        .flat_map(|(_, piece)| piece.uv_layers.iter().map(|layer| layer.name.as_str()))
        .collect();
    let kept: Vec<Vec<&Value>> = members
        .iter()
        .map(|(value, _)| {
            value
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|entry| !is_uv_mirror(entry, &uv_names))
                .collect()
        })
        .collect();
    // First-seen order, each key's header (every field but `values`) with its joined data type.
    let mut headers: Vec<(String, Value)> = Vec::new();
    for entry in kept.iter().flatten() {
        let key = attribute_key(entry);
        match headers.iter().position(|(known, _)| *known == key) {
            Some(index) => {
                let header = &mut headers[index].1;
                let known = text(header, "data_type");
                let next = text(entry, "data_type");
                if known != next {
                    let joined = promoted_data_type(text(entry, "name"), known, next)?;
                    header["data_type"] = json!(joined);
                }
            }
            None => headers.push((key, header_of(entry))),
        }
    }
    let mut output = Vec::with_capacity(headers.len());
    for (key, header) in headers {
        let mut values = Vec::new();
        for ((_, piece), member_entries) in members.iter().zip(&kept) {
            let count = entry_length(&header, piece).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("instance attribute `{name}` uses a domain that cannot be combined"),
                    json!({ "attribute": name }),
                )
            })?;
            let found = member_entries
                .iter()
                .find(|entry| attribute_key(entry) == key);
            match found {
                Some(entry) => {
                    let entry_values = entry
                        .get("values")
                        .and_then(Value::as_array)
                        .filter(|entry_values| entry_values.len() == count)
                        .ok_or_else(|| shape_error(name))?;
                    values.extend(entry_values.iter().cloned());
                }
                None => values.extend(std::iter::repeat_n(
                    default_value(text(&header, "data_type")),
                    count,
                )),
            }
        }
        let mut merged = header.as_object().cloned().unwrap_or_default();
        merged.insert("values".to_owned(), Value::Array(values));
        output.push(Value::Object(merged));
    }
    Ok(Value::Array(output))
}

/// A UV mirror: a `CORNER` `FLOAT2` `blender_attributes` entry named after a UV layer, which Blender
/// lists beside the layer (see [`write_layers`]).
fn is_uv_mirror(entry: &Value, uv_names: &[&str]) -> bool {
    text(entry, "domain") == "CORNER"
        && text(entry, "data_type") == "FLOAT2"
        && uv_names.contains(&text(entry, "name"))
}

/// The join key of a `blender_attributes` entry: its domain and name.
fn attribute_key(entry: &Value) -> String {
    format!("{}\u{0}{}", text(entry, "domain"), text(entry, "name"))
}

/// An entry's fields except its `values`.
fn header_of(entry: &Value) -> Value {
    Value::Object(
        entry
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(key, _)| key.as_str() != "values")
            .map(|(key, field)| (key.clone(), field.clone()))
            .collect(),
    )
}

/// Element count of a `blender_attributes` entry's domain in `piece`; `None` for an unknown domain.
fn entry_length(entry: &Value, piece: &Piece) -> Option<usize> {
    match text(entry, "domain") {
        "POINT" => Some(piece.vertices),
        "EDGE" => Some(piece.edges),
        "FACE" => Some(piece.faces),
        "CORNER" => Some(piece.corners),
        _ => None,
    }
}

/// The data type two same-named `blender_attributes` entries join as. Byte colors are float colors at
/// lower precision, so they join as `FLOAT_COLOR` with each value kept as it is. Any other pair of
/// types cannot be combined; the error names the entry (`attribute`) and both types.
fn promoted_data_type(attribute: &str, known: &str, next: &str) -> Result<&'static str> {
    match (known, next) {
        ("BYTE_COLOR" | "FLOAT_COLOR", "BYTE_COLOR" | "FLOAT_COLOR") => Ok("FLOAT_COLOR"),
        _ => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!(
                "collection members store attribute `{attribute}` as {known} and {next}, which cannot be combined"
            ),
            json!({ "attribute": attribute, "types": [known, next] }),
        )),
    }
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

/// A member's own `active` or `render` color layer name, read before placement rewrites it.
fn color_designation(mesh: &Mesh, key: &str) -> Option<String> {
    mesh.attributes
        .get(super::color_layers::ATTRIBUTE)?
        .get(key)?
        .as_str()
        .map(str::to_owned)
}

/// Joins canonical color layers in first-seen order. The first valid active/render designation is
/// retained; when none survives, the first joined layer is used. Each member's designation is the one
/// it stated before placement, so a null designation never claims the first layer.
fn merge_color_layers(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut layers = Vec::<String>::new();
    let mut active = None;
    let mut render = None;
    for (value, piece) in members {
        let Some(value) = value else { continue };
        let object = value.as_object().ok_or_else(|| shape_error(name))?;
        let names = object
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| shape_error(name))?;
        for layer in names {
            let layer = layer.as_str().ok_or_else(|| shape_error(name))?;
            if !layers.iter().any(|known| known == layer) {
                layers.push(layer.to_owned());
            }
        }
        if active.is_none() {
            active.clone_from(&piece.color_active);
        }
        if render.is_none() {
            render.clone_from(&piece.color_render);
        }
    }
    let fallback = layers.first().cloned();
    let designation = |name: Option<String>| {
        name.filter(|name| layers.contains(name))
            .or_else(|| fallback.clone())
    };
    Ok(json!({
        "layers": layers,
        "active": designation(active),
        "render": designation(render)
    }))
}

/// Joins canonical named color attributes, filling absent layers with transparent black. When point and
/// corner members share a layer name, the joined layer becomes corner-domain without losing point
/// colors: each source vertex value is expanded to every face corner that references it.
fn merge_color_attribute(name: &str, members: &[Member<'_>], mesh: &Mesh) -> Result<Value> {
    const DEFAULT: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut corner_domain = false;
    for (value, _) in members {
        let Some(value) = value else { continue };
        if text(value, "type") != "color" {
            return Err(conflict_error(name));
        }
        match text(value, "domain") {
            "point" => {}
            "corner" => corner_domain = true,
            _ => return Err(conflict_error(name)),
        }
    }
    let mut values = Map::new();
    if corner_domain {
        for (attribute, piece) in members {
            let fields = attribute
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object);
            if attribute.is_some() && fields.is_none() {
                return Err(shape_error(name));
            }
            for face in &mesh.faces[piece.face_base..piece.face_base + piece.faces] {
                let row = match (attribute, fields) {
                    (Some(attribute), Some(fields)) if text(attribute, "domain") == "corner" => {
                        let row = fields
                            .get(&format!("f{}", face.id))
                            .and_then(Value::as_array)
                            .filter(|row| row.len() == face.vertices.len())
                            .ok_or_else(|| shape_error(name))?;
                        row.iter()
                            .map(|value| color_value(value, name))
                            .collect::<Result<Vec<_>>>()?
                    }
                    (Some(_), Some(fields)) => face
                        .vertices
                        .iter()
                        .map(|vertex| {
                            fields
                                .get(&format!("v{vertex}"))
                                .map_or(Ok(json!(DEFAULT)), |value| color_value(value, name))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    (None, None) => vec![json!(DEFAULT); face.vertices.len()],
                    _ => return Err(shape_error(name)),
                };
                values.insert(format!("f{}", face.id), Value::Array(row));
            }
        }
    } else {
        for (attribute, piece) in members {
            let fields = attribute
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object);
            if attribute.is_some() && fields.is_none() {
                return Err(shape_error(name));
            }
            for vertex in &mesh.vertices[piece.vertex_base..piece.vertex_base + piece.vertices] {
                let color = fields
                    .and_then(|fields| fields.get(&format!("v{}", vertex.id)))
                    .map_or(Ok(json!(DEFAULT)), |value| color_value(value, name))?;
                values.insert(format!("v{}", vertex.id), color);
            }
        }
    }
    Ok(json!({
        "domain": if corner_domain { "corner" } else { "point" },
        "type": "color",
        "values": values
    }))
}

fn color_value(value: &Value, name: &str) -> Result<Value> {
    value
        .as_array()
        .filter(|values| values.len() == 4)
        .and_then(|values| {
            values
                .iter()
                .all(|component| component.as_f64().is_some_and(f64::is_finite))
                .then(|| value.clone())
        })
        .ok_or_else(|| shape_error(name))
}

/// `blender_metadata` describes each member's data-block. The joined mesh keeps every member's custom
/// properties; a key set to different values on two members is refused. `potter.*` keys are bridge
/// bookkeeping that differs per member, so they are not carried. The data-block's users and fake-user
/// flags belong to no single joined mesh and are not carried either.
fn merge_metadata(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut custom = Map::new();
    let mut color_types = Map::new();
    for (value, _) in members {
        let Some(value) = *value else { continue };
        if let Some(properties) = value.get("custom_properties").and_then(Value::as_object) {
            for (key, property) in properties {
                if key.starts_with("potter.") {
                    continue;
                }
                match custom.get(key) {
                    Some(known) if known != property => return Err(conflict_error(name)),
                    Some(_) => {}
                    None => {
                        custom.insert(key.clone(), property.clone());
                    }
                }
            }
        }
        if let Some(types) = value.get("color_data_types").and_then(Value::as_object) {
            for (layer, data_type) in types {
                let data_type = data_type.as_str().ok_or_else(|| shape_error(name))?;
                match color_types.get(layer).and_then(Value::as_str) {
                    None => {
                        color_types.insert(layer.clone(), json!(data_type));
                    }
                    Some(known @ ("BYTE_COLOR" | "FLOAT_COLOR")) => {
                        if !matches!(data_type, "BYTE_COLOR" | "FLOAT_COLOR") {
                            return Err(conflict_error(layer));
                        }
                        if known == "FLOAT_COLOR" || data_type == "FLOAT_COLOR" {
                            color_types.insert(layer.clone(), json!("FLOAT_COLOR"));
                        }
                    }
                    Some(known) if known == data_type => {}
                    Some(_) => return Err(conflict_error(layer)),
                }
            }
        }
    }
    let mut metadata = json!({ "custom_properties": custom });
    if !color_types.is_empty() {
        metadata["color_data_types"] = Value::Object(color_types);
    }
    Ok(metadata)
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

    use std::borrow::Cow;

    use glam::{DMat4, DVec3};
    use serde_json::{Map, Value, json};

    use super::{Part, concatenate};
    use crate::geom::normals::smooth_face_flags;
    use crate::geom::{BoxParams, Mesh};

    fn cube() -> Mesh {
        Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap()
    }

    /// A member borrowed from `mesh` with `slot_count` material slots.
    fn member(mesh: &Mesh, matrix: DMat4, material_offset: u32, slot_count: usize) -> Part<'_> {
        Part {
            mesh: Cow::Borrowed(mesh),
            matrix,
            material_offset,
            slot_count,
        }
    }

    type UvLayerFixture<'a> = (&'a str, [f64; 2], Option<[bool; 4]>);

    fn set_uv_layers(mesh: &mut Mesh, layers: &[UvLayerFixture<'_>]) {
        let layers = layers
            .iter()
            .map(|(name, uv, pins)| {
                let faces = mesh
                    .faces
                    .iter()
                    .map(|face| Some(vec![*uv; face.vertices.len()]))
                    .collect();
                let mut layer = crate::geom::uv_layers::UvLayer::new(*name, faces);
                layer.pinned = mesh
                    .faces
                    .iter()
                    .map(|_| pins.as_ref().map(|flags| flags.to_vec()))
                    .collect();
                layer
            })
            .collect::<Vec<_>>();
        let selected = layers.first().map(|layer| layer.name.clone());
        crate::geom::uv_layers::write(
            mesh,
            &crate::geom::uv_layers::UvStore {
                layers,
                active: selected.clone(),
                render: selected,
            },
        );
    }

    fn assert_uv_rows(
        layer: &crate::geom::uv_layers::UvLayer,
        face_range: std::ops::Range<usize>,
        expected: [f64; 2],
    ) {
        for face in &layer.faces[face_range] {
            let corners = face.as_ref().unwrap();
            assert_eq!(corners.len(), 4);
            for actual in corners {
                assert!(
                    (actual[0] - expected[0]).abs() <= 1.0e-12
                        && (actual[1] - expected[1]).abs() <= 1.0e-12
                );
            }
        }
    }

    #[test]
    fn uv_layers_join_by_name_with_defaults_pins_and_material_slots() {
        let mut first = cube();
        set_uv_layers(
            &mut first,
            &[
                ("A", [0.1, 0.2], Some([true, false, true, false])),
                ("Extra", [0.3, 0.4], Some([false, true, false, true])),
            ],
        );
        let mut second = cube();
        set_uv_layers(
            &mut second,
            &[("Extra", [0.5, 0.6], Some([true, true, false, false]))],
        );
        second.faces[0].material_index = 1;
        let mut third = cube();
        set_uv_layers(
            &mut third,
            &[
                ("Extra", [0.7, 0.8], Some([false, false, true, true])),
                ("A", [0.9, 0.25], Some([false, true, false, true])),
            ],
        );
        let fourth = cube();

        let parts = vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 1, 2),
            member(&third, DMat4::IDENTITY, 3, 1),
            member(&fourth, DMat4::IDENTITY, 4, 1),
        ];

        let joined = concatenate(parts).unwrap();
        assert_eq!(joined.faces[6].material_index, 2);
        assert!(!joined.attributes.contains_key("uv_map"));
        assert!(!joined.attributes.contains_key("blender_uv_layers"));
        let store = crate::geom::uv_layers::read(&joined);
        let names: Vec<_> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["A", "Extra"]);
        assert_eq!(store.active.as_deref(), Some("A"));
        assert_eq!(store.render.as_deref(), Some("A"));

        let a = &store.layers[0];
        assert_uv_rows(a, 0..6, [0.1, 0.2]);
        assert_uv_rows(a, 6..12, [0.0, 1.0]);
        assert_uv_rows(a, 12..18, [0.9, 0.25]);
        assert_uv_rows(a, 18..24, [0.0, 1.0]);
        assert_eq!(a.pinned[0], Some(vec![true, false, true, false]));
        assert!(a.pinned[6..12].iter().all(Option::is_none));
        assert_eq!(a.pinned[12], Some(vec![false, true, false, true]));
        assert!(a.pinned[18..24].iter().all(Option::is_none));

        let extra = &store.layers[1];
        assert_uv_rows(extra, 0..6, [0.3, 0.4]);
        assert_uv_rows(extra, 6..12, [0.5, 0.6]);
        assert_uv_rows(extra, 12..18, [0.7, 0.8]);
        assert_uv_rows(extra, 18..24, [0.0, 1.0]);
        assert_eq!(extra.pinned[0], Some(vec![false, true, false, true]));
        assert_eq!(extra.pinned[6], Some(vec![true, true, false, false]));
        assert_eq!(extra.pinned[12], Some(vec![false, false, true, true]));
        assert!(extra.pinned[18..24].iter().all(Option::is_none));
    }

    #[test]
    fn collection_uv_union_enforces_the_mesh_layer_limit() {
        let mut meshes = (0..=crate::geom::uv_layers::MAX_LAYERS)
            .map(|_| cube())
            .collect::<Vec<_>>();
        let mut parts = Vec::with_capacity(meshes.len());
        for (index, mesh) in meshes.iter_mut().enumerate() {
            let name = format!("Layer{index}");
            set_uv_layers(mesh, &[(name.as_str(), [0.25, 0.75], None)]);
            parts.push(member(
                mesh,
                DMat4::IDENTITY,
                u32::try_from(index).unwrap(),
                1,
            ));
        }
        let error = concatenate(parts).unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::LimitExceeded);
    }
    #[test]
    fn members_keep_their_own_ids_and_positional_flags() {
        let mut first = cube();
        let first_edge = first.edges[0].id;
        first.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "values": {format!("e{first_edge}"): true}}),
        );
        first.attributes.insert(
            "blender_polygon_smooth".to_owned(),
            json!([true, false, false, false, false, false]),
        );
        let second = cube();
        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::from_translation(DVec3::X * 4.0), 1, 1),
        ])
        .unwrap();
        assert_eq!(joined.vertices.len(), 16);
        assert_eq!(joined.faces.len(), 12);
        assert_eq!(joined.faces[6].material_index, 1);
        joined.validate().unwrap();

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
    fn each_members_smooth_flags_survive_the_join() {
        let mut first = cube();
        let first_face = first.faces[0].id;
        first.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": {format!("f{first_face}"): true}}),
        );
        let mut second = cube();
        second.attributes.insert(
            "blender_polygon_smooth".to_owned(),
            json!([false, true, false, false, false, false]),
        );
        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        assert_eq!(
            smooth_face_flags(&joined).unwrap(),
            [
                true, false, false, false, false, false, false, true, false, false, false, false
            ]
        );
    }

    #[test]
    fn mirrored_members_reverse_each_face_with_uvs_pins_and_normal_codes() {
        let mut source = cube();
        let smooth_values: Map<String, Value> = source
            .faces
            .iter()
            .map(|face| (format!("f{}", face.id), json!(true)))
            .collect();
        source.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": smooth_values}),
        );
        let mut uv_faces = Vec::new();
        let mut source_pins = Vec::new();
        let mut corner_order = Map::new();
        let mut corner_offset = 0_usize;
        for (index, face) in source.faces.iter().enumerate() {
            let corners = face.vertices.len();
            uv_faces.push(Some(
                (0..corners)
                    .map(|corner| [index as f64, corner as f64])
                    .collect(),
            ));
            let mut pins = vec![false; corners];
            pins[index % corners] = true;
            source_pins.push(pins);
            let order = (0..corners)
                .map(|corner| json!(corner_offset + (corner + 1) % corners))
                .collect::<Vec<_>>();
            corner_order.insert(format!("f{}", face.id), Value::Array(order));
            corner_offset += corners;
        }
        let mut uv_layer = crate::geom::uv_layers::UvLayer::new("UVMap", uv_faces);
        uv_layer.pinned = source_pins.iter().cloned().map(Some).collect();
        crate::geom::uv_layers::write(
            &mut source,
            &crate::geom::uv_layers::UvStore {
                layers: vec![uv_layer],
                active: Some("UVMap".to_owned()),
                render: Some("UVMap".to_owned()),
            },
        );
        source.attributes.insert(
            "custom_normal".to_owned(),
            json!({
                "domain": "corner",
                "type": "normal_code",
                "corner_order": corner_order
            }),
        );
        let source_positions = source
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let source_order = super::clnor::corner_order(&source).unwrap().unwrap();
        let fans = super::clnor::mesh_fans(&source, &source_positions, &source_order).unwrap();
        assert!(fans.corners.iter().all(|fan| fan.len() > 1));
        let mut code_values = vec![Value::Null; corner_offset];
        for fan in &fans.corners {
            for &corner in fan.iter().skip(1) {
                code_values[corner] = json!([1000 + corner * 100, 300 + corner * 50]);
            }
        }
        assert!(fans.corners.iter().all(|fan| {
            fan.iter().any(|&corner| code_values[corner].is_null())
                && fan.iter().any(|&corner| !code_values[corner].is_null())
        }));
        let mut normal_codes = Map::new();
        let mut code_offset = 0_usize;
        for face in &source.faces {
            let end = code_offset + face.vertices.len();
            normal_codes.insert(
                format!("f{}", face.id),
                Value::Array(code_values[code_offset..end].to_vec()),
            );
            code_offset = end;
        }
        source
            .attributes
            .get_mut("custom_normal")
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert("values".to_owned(), Value::Object(normal_codes));
        let source_normals =
            crate::geom::normals::corner_normals(&source, &source_positions).unwrap();
        let source_codes = super::clnor::corner_codes(&source).unwrap().unwrap();
        assert!(source_codes.contains(&super::clnor::CornerNormalCode::Auto));
        assert!(
            source_codes
                .iter()
                .any(|code| matches!(code, super::clnor::CornerNormalCode::Authored(_)))
        );
        assert!(
            source_order
                .iter()
                .enumerate()
                .any(|(index, ordinal)| index != *ordinal)
        );

        let identity = concatenate(vec![member(&source, DMat4::IDENTITY, 0, 1)]).unwrap();
        let identity_positions = identity
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let identity_normals =
            crate::geom::normals::corner_normals(&identity, &identity_positions).unwrap();
        for (corner, (source, actual)) in source_normals.iter().zip(&identity_normals).enumerate() {
            assert!(
                actual.abs_diff_eq(*source, 2.0e-4),
                "identity member changed mixed fan corner {corner}"
            );
        }

        let mirror = DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        let joined = concatenate(vec![member(&source, mirror, 0, 1)]).unwrap();
        joined.validate().unwrap();
        let output_positions = joined
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let output_normals =
            crate::geom::normals::corner_normals(&joined, &output_positions).unwrap();
        let output_codes = super::clnor::corner_codes(&joined).unwrap().unwrap();
        assert_eq!(output_codes.len(), source_codes.len());
        let output_order = super::clnor::corner_order(&joined).unwrap().unwrap();
        let transformed = crate::geom::normals::normal_matrix(mirror);
        let uv_store = crate::geom::uv_layers::read(&joined);
        let uv = &uv_store.layers[0];
        let mut corner_base = 0;
        for (index, face) in joined.faces.iter().enumerate() {
            let source_face = &source.faces[index];
            let uv_row = uv.faces[index].as_ref().unwrap();
            let pin_row = uv.pinned[index].as_ref().unwrap();
            let corners = face.vertices.len();
            for (corner, vertex) in face.vertices.iter().enumerate() {
                let original = corners - 1 - corner;
                let source_corner = corner_base + original;
                let output_corner = corner_base + corner;
                let expected = mirror
                    .transform_point3(source.vertex(source_face.vertices[original]).unwrap().co);
                assert!((joined.vertex(*vertex).unwrap().co - expected).length() < 1e-9);
                assert!(
                    (uv_row[corner][0] - index as f64).abs() <= 1.0e-12
                        && (uv_row[corner][1] - original as f64).abs() <= 1.0e-12
                );
                assert_eq!(pin_row[corner], source_pins[index][original]);
                assert_eq!(output_order[output_corner], source_order[source_corner]);
                assert!(
                    output_normals[output_corner].abs_diff_eq(
                        (transformed * source_normals[source_corner]).normalize_or_zero(),
                        2.0e-4,
                    ),
                    "source corner {source_corner} moved to output corner {output_corner}"
                );
            }
            corner_base += corners;
        }
    }

    #[test]
    fn mirrored_member_keeps_basis_only_normal_state() {
        let mut source = cube();
        let mut corner_order = Map::new();
        let mut corner_offset = 0_usize;
        for face in &source.faces {
            let corners = face.vertices.len();
            corner_order.insert(
                format!("f{}", face.id),
                Value::Array(
                    (0..corners)
                        .map(|corner| json!(corner_offset + (corner + 1) % corners))
                        .collect(),
                ),
            );
            corner_offset += corners;
        }
        source.attributes.insert(
            "custom_normal".to_owned(),
            json!({
                "domain": "corner",
                "type": "normal_code",
                "corner_order": corner_order
            }),
        );
        let source_order = super::clnor::corner_order(&source).unwrap().unwrap();
        assert!(super::clnor::corner_codes(&source).unwrap().is_none());

        let mirror = DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        let joined = concatenate(vec![member(&source, mirror, 0, 1)]).unwrap();
        joined.validate().unwrap();
        assert!(joined.attributes["custom_normal"].get("values").is_none());
        let output_order = super::clnor::corner_order(&joined).unwrap().unwrap();
        let mut corner_base = 0;
        for face in &source.faces {
            for corner in 0..face.vertices.len() {
                let source_corner = corner_base + face.vertices.len() - 1 - corner;
                assert_eq!(
                    output_order[corner_base + corner],
                    source_order[source_corner]
                );
            }
            corner_base += face.vertices.len();
        }
    }

    #[test]
    fn mirrored_member_preserves_explicit_zero_custom_normals() {
        let mut source = cube();
        let codes = vec![super::clnor::CornerNormalCode::ExplicitZero; corner_count(&source)];
        super::clnor::write_codes(
            &mut source,
            Some(&codes),
            super::clnor::NormalCodeOptions::default(),
        )
        .unwrap();
        let mirror = DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        let joined = concatenate(vec![member(&source, mirror, 0, 1)]).unwrap();
        joined.validate().unwrap();
        let output_codes = super::clnor::corner_codes(&joined).unwrap().unwrap();
        assert!(
            output_codes
                .iter()
                .all(|code| *code == super::clnor::CornerNormalCode::ExplicitZero)
        );
        let positions = joined
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let normals = crate::geom::normals::corner_normals(&joined, &positions).unwrap();
        assert!(normals.iter().all(|normal| *normal == DVec3::ZERO));
    }

    fn corner_count(mesh: &Mesh) -> usize {
        mesh.faces.iter().map(|face| face.vertices.len()).sum()
    }

    fn shifted_corner_order(corners: usize, shift: usize) -> Vec<usize> {
        (0..corners)
            .map(|corner| (corner + shift) % corners)
            .collect()
    }

    fn set_custom_normal(mesh: &mut Mesh, shift: usize, codes: Option<&[Value]>) {
        let order = shifted_corner_order(corner_count(mesh), shift);
        let mut corner_order = Map::new();
        let mut values = Map::new();
        let mut start = 0;
        for face in &mesh.faces {
            let end = start + face.vertices.len();
            corner_order.insert(
                format!("f{}", face.id),
                Value::Array(order[start..end].iter().copied().map(Value::from).collect()),
            );
            if let Some(codes) = codes {
                values.insert(
                    format!("f{}", face.id),
                    Value::Array(codes[start..end].to_vec()),
                );
            }
            start = end;
        }
        let mut attribute = Map::new();
        attribute.insert("domain".to_owned(), json!("corner"));
        attribute.insert("type".to_owned(), json!("normal_code"));
        attribute.insert("corner_order".to_owned(), Value::Object(corner_order));
        if codes.is_some() {
            attribute.insert("values".to_owned(), Value::Object(values));
        }
        mesh.attributes
            .insert("custom_normal".to_owned(), Value::Object(attribute));
    }

    #[test]
    fn basis_only_normal_state_joins_member_local_orders() {
        let mut first = cube();
        let mut second = cube();
        set_custom_normal(&mut first, 1, None);
        set_custom_normal(&mut second, 1, None);
        let first_corners = corner_count(&first);
        let second_corners = corner_count(&second);

        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        joined.validate().unwrap();
        assert!(joined.attributes["custom_normal"].get("values").is_none());
        let expected = shifted_corner_order(first_corners, 1)
            .into_iter()
            .chain(
                shifted_corner_order(second_corners, 1)
                    .into_iter()
                    .map(|ordinal| first_corners + ordinal),
            )
            .collect::<Vec<_>>();
        assert_eq!(super::clnor::corner_order(&joined).unwrap(), Some(expected));
    }

    #[test]
    fn basis_only_normal_state_fills_natural_order_for_state_less_members() {
        let mut first = cube();
        let second = cube();
        set_custom_normal(&mut first, 1, None);
        let first_corners = corner_count(&first);
        let second_corners = corner_count(&second);

        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        joined.validate().unwrap();
        assert!(joined.attributes["custom_normal"].get("values").is_none());
        let expected = shifted_corner_order(first_corners, 1)
            .into_iter()
            .chain(first_corners..first_corners + second_corners)
            .collect::<Vec<_>>();
        assert_eq!(super::clnor::corner_order(&joined).unwrap(), Some(expected));
    }

    #[test]
    fn allocated_auto_normal_layer_survives_join_with_state_less_member() {
        let mut first = cube();
        let second = cube();
        let first_corners = corner_count(&first);
        let second_corners = corner_count(&second);
        let auto_codes = vec![Value::Null; first_corners];
        set_custom_normal(&mut first, 1, Some(&auto_codes));

        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        joined.validate().unwrap();
        assert!(joined.attributes["custom_normal"].get("values").is_some());
        let codes = super::clnor::corner_codes(&joined).unwrap().unwrap();
        assert_eq!(codes.len(), first_corners + second_corners);
        assert!(
            codes
                .iter()
                .all(|code| *code == super::clnor::CornerNormalCode::Auto)
        );
        let expected = shifted_corner_order(first_corners, 1)
            .into_iter()
            .chain(first_corners..first_corners + second_corners)
            .collect::<Vec<_>>();
        assert_eq!(super::clnor::corner_order(&joined).unwrap(), Some(expected));
    }

    #[test]
    fn allocated_normal_codes_keep_member_local_bases_and_rows() {
        let mut first = cube();
        let mut second = cube();
        let first_corners = corner_count(&first);
        let second_corners = corner_count(&second);
        let first_codes = (0..first_corners)
            .map(|corner| {
                if corner % 4 == 0 {
                    Value::Null
                } else {
                    json!([1000 + corner as i64, 300])
                }
            })
            .collect::<Vec<_>>();
        let second_codes = (0..second_corners)
            .map(|corner| {
                if corner % 4 == 1 {
                    Value::Null
                } else {
                    json!([-1000 - corner as i64, 1500])
                }
            })
            .collect::<Vec<_>>();
        set_custom_normal(&mut first, 1, Some(&first_codes));
        set_custom_normal(&mut second, 3, Some(&second_codes));

        let first_placed = concatenate(vec![member(&first, DMat4::IDENTITY, 0, 1)]).unwrap();
        let second_placed = concatenate(vec![member(&second, DMat4::IDENTITY, 0, 1)]).unwrap();
        let expected_codes = super::clnor::corner_codes(&first_placed)
            .unwrap()
            .unwrap()
            .into_iter()
            .chain(super::clnor::corner_codes(&second_placed).unwrap().unwrap())
            .collect::<Vec<_>>();
        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        joined.validate().unwrap();
        let codes = super::clnor::corner_codes(&joined).unwrap().unwrap();
        assert_eq!(codes, expected_codes);
        assert!(codes.contains(&super::clnor::CornerNormalCode::Auto));
        assert!(
            codes
                .iter()
                .any(|code| matches!(code, super::clnor::CornerNormalCode::Authored(_)))
        );
        let expected_order = shifted_corner_order(first_corners, 1)
            .into_iter()
            .chain(
                shifted_corner_order(second_corners, 3)
                    .into_iter()
                    .map(|ordinal| first_corners + ordinal),
            )
            .collect::<Vec<_>>();
        assert_eq!(
            super::clnor::corner_order(&joined).unwrap(),
            Some(expected_order)
        );
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
        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 1, 1),
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
    fn generic_face_edge_and_sparse_corner_colors_use_named_merge() {
        let mut first = cube();
        let mut second = cube();
        let first_face = first.faces[0].id;
        let second_face = second.faces[0].id;
        let first_edge = first.edges[0].id;
        let second_edge = second.edges[0].id;
        let first_face_color = json!([1.0, 0.0, 0.0, 1.0]);
        let second_face_color = json!([0.0, 0.0, 1.0, 1.0]);
        let first_edge_color = json!([0.0, 1.0, 0.0, 1.0]);
        let second_edge_color = json!([1.0, 1.0, 0.0, 1.0]);
        let first_corner_color = json!([0.25, 0.5, 0.75, 1.0]);
        let second_corner_color = json!([0.75, 0.5, 0.25, 1.0]);

        first.attributes.insert(
            "generic_face_color".to_owned(),
            json!({"domain":"face","type":"color","values":{format!("f{first_face}"):first_face_color}}),
        );
        second.attributes.insert(
            "generic_face_color".to_owned(),
            json!({"domain":"face","type":"color","values":{format!("f{second_face}"):second_face_color}}),
        );
        first.attributes.insert(
            "generic_edge_color".to_owned(),
            json!({"domain":"edge","type":"color","values":{format!("e{first_edge}"):first_edge_color}}),
        );
        second.attributes.insert(
            "generic_edge_color".to_owned(),
            json!({"domain":"edge","type":"color","values":{format!("e{second_edge}"):second_edge_color}}),
        );
        first.attributes.insert(
            "sparse_corner_color".to_owned(),
            json!({"domain":"corner","type":"color","values":{format!("f{first_face}"):vec![first_corner_color.clone(); first.faces[0].vertices.len()]}}),
        );
        second.attributes.insert(
            "sparse_corner_color".to_owned(),
            json!({"domain":"corner","type":"color","values":{format!("f{second_face}"):vec![second_corner_color.clone(); second.faces[0].vertices.len()]}}),
        );

        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        let face_values = joined.attributes["generic_face_color"]["values"]
            .as_object()
            .unwrap();
        assert_eq!(face_values.len(), 2);
        assert!(face_values.values().any(|value| value == &first_face_color));
        assert!(
            face_values
                .values()
                .any(|value| value == &second_face_color)
        );

        let edge_values = joined.attributes["generic_edge_color"]["values"]
            .as_object()
            .unwrap();
        assert_eq!(edge_values.len(), 2);
        assert!(edge_values.values().any(|value| value == &first_edge_color));
        assert!(
            edge_values
                .values()
                .any(|value| value == &second_edge_color)
        );

        let corner_values = joined.attributes["sparse_corner_color"]["values"]
            .as_object()
            .unwrap();
        assert_eq!(corner_values.len(), 2);
        assert!(
            corner_values
                .values()
                .any(|value| { value == &json!(vec![first_corner_color.clone(); 4]) })
        );
        assert!(
            corner_values
                .values()
                .any(|value| { value == &json!(vec![second_corner_color.clone(); 4]) })
        );
    }

    #[test]
    fn generic_color_attributes_keep_named_conflict_errors() {
        let mut first = cube();
        let mut second = cube();
        first.attributes.insert(
            "generic_color".to_owned(),
            json!({"domain":"face","type":"color","values":{}}),
        );
        second.attributes.insert(
            "generic_color".to_owned(),
            json!({"domain":"edge","type":"color","values":{}}),
        );

        let error = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
        assert_eq!(error.details["attribute"], json!("generic_color"));
    }

    #[test]
    fn native_color_layers_union_and_promote_point_to_corner() {
        let mut point = cube();
        let point_values: Map<String, Value> = point
            .vertices
            .iter()
            .map(|vertex| {
                (
                    format!("v{}", vertex.id),
                    json!([f64::from(vertex.id), 0.0, 0.0, 1.0]),
                )
            })
            .collect();
        let first_color = point_values[&format!("v{}", point.faces[0].vertices[0])].clone();
        point.attributes.insert(
            "Body".to_owned(),
            json!({"domain":"point","type":"color","values":point_values}),
        );
        point.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":["Body"],"active":"Body","render":null}),
        );
        point.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types":{"Body":"BYTE_COLOR"}}),
        );

        let mut corner = cube();
        let body_values: Map<String, Value> = corner
            .faces
            .iter()
            .map(|face| {
                (
                    format!("f{}", face.id),
                    json!(vec![[0.0, 0.0, 1.0, 1.0]; face.vertices.len()]),
                )
            })
            .collect();
        let atlas_values: Map<String, Value> = corner
            .faces
            .iter()
            .map(|face| {
                (
                    format!("f{}", face.id),
                    json!(vec![[0.25, 0.5, 0.75, 1.0]; face.vertices.len()]),
                )
            })
            .collect();
        corner.attributes.insert(
            "Body".to_owned(),
            json!({"domain":"corner","type":"color","values":body_values}),
        );
        corner.attributes.insert(
            "Atlas".to_owned(),
            json!({"domain":"corner","type":"color","values":atlas_values}),
        );
        corner.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":["Body","Atlas"],"active":null,"render":"Atlas"}),
        );
        corner.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types":{"Body":"FLOAT_COLOR","Atlas":"FLOAT_COLOR"}}),
        );

        let joined = concatenate(vec![
            member(&point, DMat4::IDENTITY, 0, 1),
            member(&corner, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        let layers = &joined.attributes["color_layers"];
        assert_eq!(layers["layers"], json!(["Body", "Atlas"]));
        assert_eq!(layers["active"], json!("Body"));
        assert_eq!(layers["render"], json!("Atlas"));
        assert_eq!(joined.attributes["Body"]["domain"], json!("corner"));
        assert_eq!(
            joined.attributes["Body"]["values"][format!("f{}", joined.faces[0].id)][0],
            first_color
        );
        assert_eq!(
            joined.attributes["Body"]["values"][format!("f{}", joined.faces[6].id)][0],
            json!([0.0, 0.0, 1.0, 1.0])
        );
        assert_eq!(
            joined.attributes["Atlas"]["values"][format!("f{}", joined.faces[0].id)],
            json!(vec![[0.0, 0.0, 0.0, 0.0]; 4])
        );
        assert_eq!(
            joined.attributes["blender_metadata"]["color_data_types"]["Body"],
            json!("FLOAT_COLOR")
        );
    }
    #[test]
    fn reserved_color_layer_name_joins_point_and_corner_domains() {
        let key = crate::geom::attribute_keys::storage_key("color_layers").into_owned();
        let mut point = cube();
        let point_values: Map<String, Value> = point
            .vertices
            .iter()
            .map(|vertex| (format!("v{}", vertex.id), json!([0.25, 0.0, 0.0, 1.0])))
            .collect();
        point.attributes.insert(
            key.clone(),
            json!({"domain":"point","type":"color","values":point_values}),
        );
        point.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":["color_layers"],"active":"color_layers","render":"color_layers"}),
        );

        let mut corner = cube();
        let corner_values: Map<String, Value> = corner
            .faces
            .iter()
            .map(|face| {
                (
                    format!("f{}", face.id),
                    json!(vec![[0.0, 0.0, 0.75, 1.0]; face.vertices.len()]),
                )
            })
            .collect();
        corner.attributes.insert(
            key.clone(),
            json!({"domain":"corner","type":"color","values":corner_values}),
        );
        corner.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":["color_layers"],"active":"color_layers","render":"color_layers"}),
        );

        let joined = concatenate(vec![
            member(&point, DMat4::IDENTITY, 0, 1),
            member(&corner, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap();
        let layer = joined.attributes.get(&key).unwrap();
        assert_eq!(
            joined.attributes["color_layers"]["layers"],
            json!(["color_layers"])
        );
        assert_eq!(layer["domain"], json!("corner"));
        assert_eq!(
            layer["values"][format!("f{}", joined.faces[0].id)][0],
            json!([0.25, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            layer["values"][format!("f{}", joined.faces[6].id)][0],
            json!([0.0, 0.0, 0.75, 1.0])
        );
    }

    #[test]
    fn byte_and_float_colors_join_as_float_colors() {
        let mut first = cube();
        first.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name": "Col", "domain": "CORNER", "data_type": "BYTE_COLOR",
                "values": std::iter::repeat_n(json!([1.0, 0.0, 0.0, 1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        let mut second = cube();
        second.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name": "Col", "domain": "CORNER", "data_type": "FLOAT_COLOR",
                "values": std::iter::repeat_n(json!([0.0, 0.0, 1.0, 1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        let joined = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 1, 1),
        ])
        .unwrap();
        let entry = &joined.attributes["blender_attributes"][0];
        assert_eq!(entry["data_type"], json!("FLOAT_COLOR"));
        let values = entry["values"].as_array().unwrap();
        assert_eq!(values.len(), 48);
        assert_eq!(values[0], json!([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(values[24], json!([0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn unpromotable_data_types_are_refused_with_their_names() {
        let mut first = cube();
        first.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name": "Col", "domain": "CORNER", "data_type": "FLOAT_COLOR",
                "values": std::iter::repeat_n(json!([1.0, 0.0, 0.0, 1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        let mut second = cube();
        second.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{
                "name": "Col", "domain": "CORNER", "data_type": "FLOAT_VECTOR",
                "values": std::iter::repeat_n(json!([0.0, 0.0, 1.0]), 24).collect::<Vec<_>>()
            }]),
        );
        let error = concatenate(vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
        ])
        .unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
        assert_eq!(error.details["attribute"], json!("Col"));
        assert_eq!(
            error.details["types"],
            json!(["FLOAT_COLOR", "FLOAT_VECTOR"])
        );
    }

    #[test]
    fn members_join_custom_properties_and_refuse_conflicts() {
        let metadata = |custom: Value| json!({"custom_properties": custom, "users": 1});
        let mut first = cube();
        first.attributes.insert(
            "blender_metadata".to_owned(),
            metadata(json!({"tag": 1, "potter.id": 1})),
        );
        let mut second = cube();
        second.attributes.insert(
            "blender_metadata".to_owned(),
            metadata(json!({"part": "hat", "potter.id": 2})),
        );
        let join = |first: &Mesh, second: &Mesh| {
            concatenate(vec![
                member(first, DMat4::IDENTITY, 0, 1),
                member(second, DMat4::IDENTITY, 1, 1),
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

    #[test]
    fn out_of_range_material_indices_clamp_to_the_last_slot() {
        let mut first = cube();
        first.faces[0].material_index = 5;
        let joined = concatenate(vec![member(&first, DMat4::IDENTITY, 3, 2)]).unwrap();
        // Slot 5 clamps to the member's last slot (1), then joins at the member's offset (3).
        assert_eq!(joined.faces[0].material_index, 4);
        assert_eq!(joined.faces[1].material_index, 3);
    }

    #[test]
    fn uv_mirrors_of_member_layers_are_not_carried() {
        let mut first = cube();
        set_uv_layers(&mut first, &[("Body", [0.5, 0.5], None)]);
        first.attributes.insert(
            "blender_attributes".to_owned(),
            json!([
                {"name": "Body", "domain": "CORNER", "data_type": "FLOAT2", "values": vec![json!([0.5, 0.5]); 24]},
                {"name": "LoopColor", "domain": "CORNER", "data_type": "FLOAT_COLOR", "values": vec![json!([1.0, 0.0, 0.0, 1.0]); 24]}
            ]),
        );
        let joined = concatenate(vec![member(&first, DMat4::IDENTITY, 0, 1)]).unwrap();
        let names: Vec<Value> = joined.attributes["blender_attributes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["name"].clone())
            .collect();
        assert_eq!(names, vec![json!("LoopColor")]);
        let layers = crate::geom::uv_layers::read(&joined).layers;
        assert_eq!(layers[0].name, "Body");
        assert!(!joined.attributes.contains_key("blender_uv_layers"));
    }
}
