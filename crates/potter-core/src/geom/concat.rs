// SPDX-License-Identifier: GPL-3.0-or-later

//! Collection-instance joins. Each member keeps its own attributes, as Blender's per-object meshes
//! do: a member's mesh is placed under fresh element IDs, its ID-keyed attributes are rewritten onto
//! those IDs by [`remap_attributes`] (the same provenance rule modifiers use), and positional
//! attributes concatenate in storage order. Members that lack an attribute contribute its default.
//!
//! UV layers join by position. Joined layer `k` is every member's `k`-th [`read_layers`] layer, in
//! member order, in Blender's V-up convention. Potter's `uv_map` entries without a `layer` key hold
//! glTF V (the exporter writes them unflipped) and flip to Blender's; entries with a `layer` key and
//! `blender_uv_layers` already hold Blender's V. The joined layer keeps the name of the first member
//! with that layer at that position, and a Potter default reads as `UVMap`, so references to the
//! default layer still resolve. A corner with no value joins as `[0, 0]`; [`uv_covered_faces`] reports
//! which faces really have one. The joined mesh stores UVs only in `blender_uv_layers`.
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

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

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
    /// Transient local normals aligned with the member's vertex storage order.
    pub adaptive_normals: Option<Cow<'a, [Option<DVec3>]>>,
}

/// A member placed under the joined IDs: its element tables, its attributes remapped, and its UV
/// layers read in Blender's convention. The `uv_map` and `blender_uv_layers` stores are not kept.
struct Placed {
    mesh: Mesh,
    uv_layers: Vec<UvLayer>,
    adaptive_normals: Option<Vec<Option<DVec3>>>,
}

/// A placed member's attributes and element counts, the inputs of [`merge_attributes`] and
/// [`join_uv_layers`].
struct Piece {
    attributes: Map<String, Value>,
    uv_layers: Vec<UvLayer>,
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
    Ok(concatenate_inner(parts, false)?.0)
}

/// Like [`concatenate`], while returning member-ordered transformed transient normals when present.
pub(crate) fn concatenate_with_normals(
    parts: Vec<Part<'_>>,
) -> Result<(Mesh, Option<Vec<Option<DVec3>>>)> {
    concatenate_inner(parts, true)
}

fn concatenate_inner(
    parts: Vec<Part<'_>>,
    include_normals: bool,
) -> Result<(Mesh, Option<Vec<Option<DVec3>>>)> {
    let mut output = Mesh::new();
    let mut pieces = Vec::with_capacity(parts.len());
    let mut adaptive_normals = Vec::new();
    let mut has_adaptive_normals = false;
    for part in parts {
        let Placed {
            mut mesh,
            uv_layers,
            adaptive_normals: mut member_normals,
        } = place(&part, output.next_id, include_normals)?;
        if include_normals {
            if let Some(normals) = &mut member_normals {
                has_adaptive_normals = true;
                adaptive_normals.append(normals);
            } else {
                for _ in 0..mesh.vertices.len() {
                    adaptive_normals.push(None);
                }
            }
        }
        pieces.push(Piece {
            attributes: std::mem::take(&mut mesh.attributes),
            uv_layers,
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
    let layers = join_uv_layers(&mut pieces);
    if !layers.is_empty() {
        let stores: Vec<Value> = layers
            .iter()
            .map(|layer| json!({"name": layer.name, "values": []}))
            .collect();
        output
            .attributes
            .insert("blender_uv_layers".to_owned(), Value::Array(stores));
        // Fills the layer stores and the UV-named `blender_attributes` mirrors with the joined values.
        write_layers(&mut output, &layers);
    }
    output.validate().map_err(|error| join_error(&error))?;
    Ok((output, has_adaptive_normals.then_some(adaptive_normals)))
}

/// Per joined face, whether the face's member has a value on every corner of its first UV layer,
/// which is the joined `TEXCOORD_0`. The joined mesh cannot say this: a corner without a value joins
/// as `[0, 0]`, which reads as a real UV.
pub(crate) fn uv_covered_faces(parts: &[Part<'_>]) -> Vec<bool> {
    parts
        .iter()
        .flat_map(|part| {
            let first = read_layers(&part.mesh).into_iter().next();
            (0..part.mesh.faces.len()).map(move |face| {
                first
                    .as_ref()
                    .is_some_and(|layer| layer.faces[face].is_some())
            })
        })
        .collect()
}

/// Places one member's mesh under IDs counted from `first_ids`. Elements are built in one pass over
/// the source, attributes are remapped and shading canonicalised, and UVs are read. A mirrored member
/// reverses its corners. Vertices move to the instance space last, and custom normals rotate with
/// the member's normal matrix.
fn place(part: &Part<'_>, first_ids: IdCounters, include_normals: bool) -> Result<Placed> {
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
    // `uv_map` is rebuilt in Blender's convention by `take_uv_layers`, so the remap never runs on it.
    remap_attributes_skipping(source, &mut mesh, &origin, &["uv_map"])?;
    canonical_shading(source, &mut mesh)?;
    let mut uv_layers = take_uv_layers(source, &mut mesh);
    if part.matrix.determinant() < 0.0 {
        mirror_corners(&mut mesh, &mut uv_layers);
    }
    let normals = normal_matrix(part.matrix);
    for vertex in &mut mesh.vertices {
        vertex.co = part.matrix.transform_point3(vertex.co);
    }
    let adaptive_normals = if include_normals {
        part.adaptive_normals
            .as_deref()
            .map(|values| {
                if values.len() != source.vertices.len() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "instance adaptive normals do not match member vertices",
                    ));
                }
                Ok(values
                    .iter()
                    .map(|normal| normal.map(|normal| (normals * normal).normalize_or_zero()))
                    .collect())
            })
            .transpose()?
    } else {
        None
    };
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
    Ok(Placed {
        mesh,
        uv_layers,
        adaptive_normals,
    })
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
        let flags = smooth_face_flags(source)?;
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
        let flags = sharp_edge_flags(source)?;
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

/// Reads `mesh`'s UV layers in Blender's V-up convention, then removes the `uv_map` and
/// `blender_uv_layers` stores, which the joined mesh rebuilds from the layers.
fn take_uv_layers(source: &Mesh, mesh: &mut Mesh) -> Vec<UvLayer> {
    if source.attributes.contains_key("uv_map") {
        let entries = blender_uv_map(source, mesh);
        mesh.attributes
            .insert("uv_map".to_owned(), Value::Array(entries));
    }
    let layers = read_layers(mesh);
    mesh.attributes.remove("uv_map");
    mesh.attributes.remove("blender_uv_layers");
    layers
}

/// The member's `uv_map` entries on the placed faces, in Blender's V-up convention. Placement keeps
/// face order, so each source face maps to the placed face at the same index. The source entries are
/// the only copy converted here: the remap skips `uv_map`, so no discarded remapped copy is made.
/// Entries without a `layer` key are glTF V (Potter's default) and flip; the rest already hold
/// Blender's V.
fn blender_uv_map(source: &Mesh, placed: &Mesh) -> Vec<Value> {
    let placed_faces: HashMap<u32, u32> = source
        .faces
        .iter()
        .zip(&placed.faces)
        .map(|(from, to)| (from.id, to.id))
        .collect();
    let entries = source
        .attributes
        .get("uv_map")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    entries
        .iter()
        .filter_map(|entry| {
            let face = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok())
                .and_then(|id| placed_faces.get(&id))?;
            // Entries without a `layer` key are Potter's glTF-space default; the rest are Blender's.
            let flip = entry.get("layer").is_none();
            let layer = match entry.get("layer") {
                Some(Value::String(name)) if !name.is_empty() => name.as_str(),
                Some(_) => return None,
                None => "UVMap",
            };
            let uv: Vec<Value> = entry
                .get("uv")
                .and_then(Value::as_array)?
                .iter()
                .map(|corner| if flip { flip_v(corner) } else { corner.clone() })
                .collect();
            Some(json!({
                "face_id": *face,
                "layer": layer,
                "uv": uv,
            }))
        })
        .collect()
}

/// `[u, 1 - v]` for a `[u, v]` corner. A malformed corner passes through for [`read_layers`] to skip.
fn flip_v(corner: &Value) -> Value {
    match corner.as_array().map(Vec::as_slice) {
        Some([u, v]) => match (u.as_f64(), v.as_f64()) {
            (Some(u), Some(v)) => json!([u, 1.0 - v]),
            _ => corner.clone(),
        },
        _ => corner.clone(),
    }
}

/// Reverses a mirrored member's faces, with every corner-keyed value of each face: the corner order
/// of each face, `custom_normal` and named `corner` rows, `CORNER` `blender_attributes` segments, and
/// UV rows. A corner keeps its vertex, UV, and normal together.
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
fn reverse_face_segments(values: &mut [Value], sizes: &[usize]) {
    let mut start = 0;
    for size in sizes {
        let end = start + size;
        if let Some(segment) = values.get_mut(start..end) {
            segment.reverse();
        }
        start = end;
    }
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

/// Joins the members' UV layers by position: joined layer `k` is every member's layer `k`, in member
/// order. The name comes from the first member with that layer. A member without it contributes
/// faces with no value.
fn join_uv_layers(pieces: &mut [Piece]) -> Vec<UvLayer> {
    let count = pieces
        .iter()
        .map(|piece| piece.uv_layers.len())
        .max()
        .unwrap_or(0);
    (0..count)
        .map(|index| {
            let name = pieces
                .iter()
                .find_map(|piece| piece.uv_layers.get(index))
                .map(|layer| layer.name.clone())
                .unwrap_or_default();
            let mut faces = Vec::new();
            let mut pinned = Vec::new();
            for piece in pieces.iter_mut() {
                match piece.uv_layers.get_mut(index) {
                    Some(layer) => {
                        faces.append(&mut layer.faces);
                        pinned.append(&mut layer.pinned);
                    }
                    None => {
                        faces.extend(std::iter::repeat_n(None, piece.faces));
                        pinned.extend(std::iter::repeat_n(None, piece.faces));
                    }
                }
            }
            UvLayer {
                name,
                faces,
                pinned,
            }
        })
        .collect()
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
        merged.insert(name.to_owned(), merge_attribute(name, &members, mesh)?);
    }
    Ok(merged)
}

fn merge_attribute(name: &str, members: &[Member<'_>], mesh: &Mesh) -> Result<Value> {
    match name {
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
        }) =>
        {
            merge_color_attribute(name, members, mesh)
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
/// UV mirrors are left out: the joined UV layers are rebuilt from `blender_uv_layers`, and a mirror
/// named after a member's layer would otherwise name a layer the joined store does not carry.
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
/// Joins canonical color layers in first-seen order. The first valid active/render designation is
/// retained; when none survives, the first joined layer is used.
fn merge_color_layers(name: &str, members: &[Member<'_>]) -> Result<Value> {
    let mut layers = Vec::<String>::new();
    let mut active = None;
    let mut render = None;
    for (value, _) in members {
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
            active = object
                .get("active")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if render.is_none() {
            render = object
                .get("render")
                .and_then(Value::as_str)
                .map(str::to_owned);
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

    use super::{Part, concatenate, concatenate_with_normals, uv_covered_faces};
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
            adaptive_normals: None,
        }
    }

    #[test]
    fn adaptive_normals_follow_member_order_and_inverse_transpose() {
        let first = cube();
        let second = cube();
        let mut first_part = member(&first, DMat4::from_scale(DVec3::new(2.0, 1.0, 1.0)), 0, 1);
        first_part.adaptive_normals = Some(Cow::Owned(vec![
            Some(DVec3::new(1.0, 1.0, 0.0));
            first.vertices.len()
        ]));
        let (joined, normals) =
            concatenate_with_normals(vec![first_part, member(&second, DMat4::IDENTITY, 1, 1)])
                .unwrap();
        let normals = normals.unwrap();
        assert_eq!(normals.len(), joined.vertices.len());
        let expected = DVec3::new(0.5, 1.0, 0.0).normalize();
        assert!((normals[0].unwrap() - expected).length() < 1.0e-12);
        assert!(normals[first.vertices.len()..].iter().all(Option::is_none));
        assert!(!joined.attributes.contains_key("adaptive_normals"));
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
    fn uv_layers_join_by_position_and_convert_potter_v() {
        // First member: one Blender layer, named "Map", in V-up space.
        let mut first = cube();
        first.attributes.insert(
            "blender_uv_layers".to_owned(),
            json!([{"name": "Map", "values": vec![json!([0.1, 0.2]); 24]}]),
        );
        // Second member: Potter's default `uv_map` (no `layer` key, glTF V) on every face.
        let mut second = cube();
        let entries: Vec<Value> = second
            .faces
            .iter()
            .map(|face| {
                json!({"face_id": face.id, "uv": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]})
            })
            .collect();
        second
            .attributes
            .insert("uv_map".to_owned(), Value::Array(entries));
        // Third member: no UVs at all.
        let third = cube();

        let parts = vec![
            member(&first, DMat4::IDENTITY, 0, 1),
            member(&second, DMat4::IDENTITY, 0, 1),
            member(&third, DMat4::IDENTITY, 0, 1),
        ];
        let covered = uv_covered_faces(&parts);
        assert_eq!(covered, [vec![true; 12], vec![false; 6]].concat());

        let joined = concatenate(parts).unwrap();
        assert!(!joined.attributes.contains_key("uv_map"));
        let layers = joined.attributes["blender_uv_layers"].as_array().unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0]["name"], json!("Map"));
        let values = layers[0]["values"].as_array().unwrap();
        assert_eq!(values.len(), 72);
        assert_eq!(values[0], json!([0.1, 0.2]));
        // Potter's glTF-space corners become Blender's: [0, 0] -> [0, 1], [1, 0] -> [1, 1], ...
        assert_eq!(values[24], json!([0.0, 1.0]));
        assert_eq!(values[25], json!([1.0, 1.0]));
        assert_eq!(values[26], json!([1.0, 0.0]));
        assert_eq!(values[27], json!([0.0, 0.0]));
        // The member without UVs joins as zeros, which `uv_covered_faces` reports as uncovered.
        assert!(values[48..].iter().all(|value| *value == json!([0.0, 0.0])));
    }

    #[test]
    fn mirrored_members_reverse_each_face_with_its_corner_values() {
        let mut source = cube();
        let mut uvs = Vec::new();
        let mut normals = Map::new();
        for (index, face) in source.faces.iter().enumerate() {
            for corner in 0..face.vertices.len() {
                uvs.push(json!([index as f64, corner as f64]));
            }
            let face_normals: Vec<Value> = (0..face.vertices.len())
                .map(|corner| {
                    if corner == 0 {
                        json!([1.0, 0.0, 0.0])
                    } else {
                        json!([0.0, 0.0, 1.0])
                    }
                })
                .collect();
            normals.insert(format!("f{}", face.id), Value::Array(face_normals));
        }
        source.attributes.insert(
            "blender_uv_layers".to_owned(),
            json!([{"name": "UVMap", "values": uvs}]),
        );
        source.attributes.insert(
            "custom_normal".to_owned(),
            json!({"domain": "corner", "type": "FLOAT_VECTOR", "values": normals}),
        );
        let mirror = DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        let joined = concatenate(vec![member(&source, mirror, 0, 1)]).unwrap();
        joined.validate().unwrap();

        let uv = joined.attributes["blender_uv_layers"][0]["values"]
            .as_array()
            .unwrap();
        let custom = joined.attributes["custom_normal"]["values"]
            .as_object()
            .unwrap();
        for (index, face) in joined.faces.iter().enumerate() {
            let source_face = &source.faces[index];
            let rows = &custom[&format!("f{}", face.id)];
            for (corner, vertex) in face.vertices.iter().enumerate() {
                let original = source_face.vertices.len() - 1 - corner;
                let expected = mirror
                    .transform_point3(source.vertex(source_face.vertices[original]).unwrap().co);
                assert!((joined.vertex(*vertex).unwrap().co - expected).length() < 1e-9);
                assert_eq!(
                    uv[index * 4 + corner],
                    json!([index as f64, original as f64])
                );
                let normal = if original == 0 {
                    json!([-1.0, 0.0, 0.0])
                } else {
                    json!([0.0, 0.0, 1.0])
                };
                assert_eq!(rows[corner], normal);
            }
        }
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
        first.attributes.insert(
            "blender_uv_layers".to_owned(),
            json!([{"name": "Body", "values": vec![json!([0.5, 0.5]); 24]}]),
        );
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
        assert_eq!(
            joined.attributes["blender_uv_layers"][0]["name"],
            json!("Body")
        );
    }
}
