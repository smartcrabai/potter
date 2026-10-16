// SPDX-License-Identifier: GPL-3.0-or-later

//! Scene document upgrades. [`parse_scene`] is the one way to read a scene document (`scene.json`,
//! a library scene, or a history snapshot). It upgrades the raw JSON one schema version at a time,
//! then deserializes. Version 2 replaced the mesh attributes `uv_map` and `blender_uv_layers` with
//! `uv_layers`, and the texture reference field `uv_map` with `uv_layer`. Version 3 converted
//! Blender-keyed shading (including compressed custom normals), color, and generic element data to
//! native keys, and user attributes to their storage keys (see [`upgrade_mesh_v2`]).
use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use super::SceneDoc;
use crate::{
    error::{ErrorCode, PotError, Result},
    store::HistoryRecord,
};
/// Returns the kind of an inline resource; non-inline binary resources return `None`.
///
/// # Errors
///
/// Returns `SCENE_INVALID` when an inline resource has no string `kind`.
pub fn inline_resource_kind(value: &Value) -> Result<Option<&str>> {
    if value.get("storage").and_then(Value::as_str) != Some("inline") {
        return Ok(None);
    }
    value
        .get("kind")
        .and_then(Value::as_str)
        .map(Some)
        .ok_or_else(|| invalid("inline resource is missing its kind"))
}

/// Texture reference fields of a material.
const MATERIAL_TEXTURES: [&str; 4] = [
    "base_color_texture",
    "roughness_texture",
    "metallic_texture",
    "normal_texture",
];

/// Texture reference fields of an `MToon` avatar material.
const MTOON_TEXTURES: [&str; 5] = [
    "shade_multiply_texture",
    "matcap_texture",
    "rim_multiply_texture",
    "outline_width_multiply_texture",
    "uv_animation_mask_texture",
];

/// A scene document read from bytes, with what its upgrade produced.
pub struct ParsedScene {
    pub doc: SceneDoc,
    /// Diagnostics for version-1 UV data skipped or dropped during migration.
    pub warnings: Vec<Value>,
    /// Whether the bytes were schema version 1. Their original bytes must be kept before the first
    /// schema-version-3 write replaces them.
    pub legacy: bool,
}

/// Parses and upgrades a scene document, then migrates its loaded-only legacy resource sentinels.
pub fn parse_scene(bytes: &[u8]) -> Result<ParsedScene> {
    let mut value = parse_value(bytes)?;
    let legacy = value.get("schema_version").and_then(Value::as_u64) == Some(1);
    let warnings = upgrade_scene(&mut value)?;
    let mut doc: SceneDoc = serde_json::from_value(value).map_err(|error| json_error(&error))?;
    crate::geom::legacy_texture::migrate_legacy_volume_noise_sentinel(&mut doc)?;
    Ok(ParsedScene {
        doc,
        warnings,
        legacy,
    })
}

/// Parses a history record, upgrading its snapshot and migrating loaded-only texture sentinels.
pub(crate) fn parse_history_record(bytes: &[u8]) -> Result<(HistoryRecord, Vec<Value>)> {
    let mut value = parse_value(bytes)?;
    let warnings = match value.get_mut("snapshot") {
        Some(snapshot) => upgrade_scene(snapshot)?,
        None => Vec::new(),
    };
    let mut record: HistoryRecord =
        serde_json::from_value(value).map_err(|error| json_error(&error))?;
    crate::geom::legacy_texture::migrate_legacy_volume_noise_sentinel(&mut record.snapshot)?;
    Ok((record, warnings))
}

fn parse_value(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes).map_err(|error| json_error(&error))
}

fn json_error(error: &serde_json::Error) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        error.to_string(),
        json!({ "line": error.line(), "column": error.column() }),
    )
}

/// Upgrades a document from version 1 through 3, retaining diagnostics from legacy UV data.
pub fn upgrade_scene(value: &mut Value) -> Result<Vec<Value>> {
    let mut warnings = Vec::new();
    if value.get("schema_version").and_then(Value::as_u64) == Some(1) {
        warnings = upgrade_v1(value)?;
    }
    if value.get("schema_version").and_then(Value::as_u64) == Some(2) {
        upgrade_v2(value)?;
    }
    Ok(warnings)
}

/// Upgrades a schema-version-1 scene document to version 2.
fn upgrade_v1(value: &mut Value) -> Result<Vec<Value>> {
    let referenced: BTreeSet<String> = texture_refs(value)
        .iter()
        .filter_map(|texture| texture.get("uv_map")?.as_str())
        .filter(|name| !name.is_empty() && *name != "uv_map")
        .map(str::to_owned)
        .collect();
    let mut warnings = Vec::new();
    if let Some(blocks) = value.get_mut("data_blocks").and_then(Value::as_object_mut) {
        for (id, block) in blocks.iter_mut() {
            if let Some(mesh) = block.get_mut("mesh").filter(|mesh| !mesh.is_null()) {
                warnings.extend(crate::geom::uv_layers::upgrade_mesh_v1(
                    mesh,
                    id,
                    &referenced,
                ));
            }
        }
    }
    for texture in texture_refs(value) {
        upgrade_texture_ref(texture)?;
    }
    value["schema_version"] = json!(2);
    Ok(warnings)
}

/// Every material and MToon texture reference, for mutation.
fn texture_refs(value: &mut Value) -> Vec<&mut Value> {
    let mut refs = Vec::new();
    let Some(object) = value.as_object_mut() else {
        return refs;
    };
    for (key, section) in object.iter_mut() {
        match key.as_str() {
            "materials" => {
                if let Some(materials) = section.as_object_mut() {
                    for material in materials.values_mut() {
                        push_material_refs(material, &MATERIAL_TEXTURES, &mut refs);
                    }
                }
            }
            "avatar" => {
                if let Some(mtoon) = section.get_mut("mtoon").and_then(Value::as_object_mut) {
                    for material in mtoon.values_mut() {
                        push_mtoon_refs(material, &mut refs);
                    }
                }
            }
            _ => {}
        }
    }
    refs
}

fn push_material_refs<'a>(material: &'a mut Value, keys: &[&str], refs: &mut Vec<&'a mut Value>) {
    if let Some(fields) = material.as_object_mut() {
        refs.extend(
            fields
                .iter_mut()
                .filter(|(key, _)| keys.contains(&key.as_str()))
                .map(|(_, texture)| texture),
        );
    }
}

fn push_mtoon_refs<'a>(material: &'a mut Value, refs: &mut Vec<&'a mut Value>) {
    let Some(fields) = material.as_object_mut() else {
        return;
    };
    for (key, field) in fields.iter_mut() {
        if key.as_str() == "shading_shift_texture" {
            if let Some(texture) = field.get_mut("texture") {
                refs.push(texture);
            }
        } else if MTOON_TEXTURES.contains(&key.as_str()) {
            refs.push(field);
        }
    }
}

/// Converts one version-1 texture reference; empty and `uv_map` select the render layer.
fn upgrade_texture_ref(texture: &mut Value) -> Result<()> {
    let Some(object) = texture.as_object_mut() else {
        return Ok(());
    };
    let Some(uv_map) = object.remove("uv_map") else {
        return Ok(());
    };
    let uv_layer = match uv_map {
        Value::Null => Value::Null,
        Value::String(name) if name.is_empty() || name == "uv_map" => Value::Null,
        Value::String(name) => Value::String(name),
        _ => {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "version-1 texture uv_map must be a string or null",
            ));
        }
    };
    object.insert("uv_layer".to_owned(), uv_layer);
    Ok(())
}

/// Upgrades a schema-version-2 scene document to version 3 on every data-block mesh.
fn upgrade_v2(value: &mut Value) -> Result<()> {
    if let Some(blocks) = value.get_mut("data_blocks").and_then(Value::as_object_mut) {
        for block in blocks.values_mut() {
            if let Some(mesh) = block.get_mut("mesh").filter(|mesh| !mesh.is_null()) {
                upgrade_mesh_v2(mesh)?;
            }
        }
    }
    value["schema_version"] = json!(3);
    Ok(())
}

/// Blender attributes that mirror a native key or topology Potter recomputes. They are dropped.
const MIRROR_ATTRIBUTES: [&str; 5] = [
    "sharp_face",
    "sharp_edge",
    "uv_seam",
    "material_index",
    "position",
];

/// Blender internal attributes that hold selection, visibility, or sculpt state. They stay in
/// `blender_attributes`. Other dot-prefixed Blender attributes are topology or UV pins and are dropped.
const BLENDER_STATE_PREFIXES: [&str; 4] = [".select_", ".uv_select_", ".hide_", ".sculpt_"];

/// Element IDs of a version-2 mesh in storage order. Blender's element order is the storage order, so
/// the positional arrays of version 2 index the same elements.
struct Elements {
    vertices: Vec<u64>,
    edges: Vec<u64>,
    /// Face ID and corner vertex IDs of each face, in storage order.
    faces: Vec<(u64, Vec<u64>)>,
}

impl Elements {
    fn read(mesh: &Map<String, Value>) -> Result<Self> {
        let faces = mesh
            .get("faces")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|face| {
                let id = face
                    .get("id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| invalid("version-2 face has no id"))?;
                let corners = face
                    .get("v")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("version-2 face has no vertices"))?
                    .iter()
                    .map(|vertex| {
                        vertex
                            .as_u64()
                            .ok_or_else(|| invalid("version-2 face vertex ID is invalid"))
                    })
                    .collect::<Result<Vec<u64>>>()?;
                Ok((id, corners))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vertices: element_ids(mesh, "vertices")?,
            edges: element_ids(mesh, "edges")?,
            faces,
        })
    }

    fn corner_count(&self) -> usize {
        self.faces.iter().map(|(_, corners)| corners.len()).sum()
    }
}

fn element_ids(mesh: &Map<String, Value>, key: &str) -> Result<Vec<u64>> {
    mesh.get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|element| {
            element
                .get("id")
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("version-2 mesh element has no id"))
        })
        .collect()
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::SceneInvalid, message)
}

/// Blender custom-normal loop codes in Blender storage order. Invalid pairs are rejected rather than
/// retained as an unconverted compatibility store.
fn blender_custom_normal_codes(
    attributes: &Map<String, Value>,
    elements: &Elements,
) -> Result<Option<Vec<crate::geom::clnor::CornerNormalCode>>> {
    let Some(entries) = attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    let mut matches = entries.iter().filter(|entry| {
        text(entry, "name") == "custom_normal" && text(entry, "data_type") == "INT16_2D"
    });
    let Some(entry) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(invalid(
            "version-2 mesh has multiple custom_normal INT16_2D attributes",
        ));
    }
    let values = entry
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("version-2 custom_normal INT16_2D values must be an array"))?;
    if values.len() != elements.corner_count() {
        return Err(invalid(
            "version-2 custom_normal INT16_2D value count does not match mesh corners",
        ));
    }
    values
        .iter()
        .map(|value| {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| {
                    invalid("version-2 custom_normal INT16_2D values must be signed 16-bit pairs")
                })?;
            let component = |value: &Value| {
                value
                    .as_i64()
                    .and_then(|value| i16::try_from(value).ok())
                    .ok_or_else(|| {
                        invalid(
                            "version-2 custom_normal INT16_2D values must be signed 16-bit pairs",
                        )
                    })
            };
            Ok(crate::geom::clnor::CornerNormalCode::from_pair([
                component(&pair[0])?,
                component(&pair[1])?,
            ]))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Reads the former keyed float3 custom-normal store in face-corner order.
fn legacy_custom_normal_vectors(mesh: &crate::geom::Mesh) -> Result<Vec<Option<glam::DVec3>>> {
    let values = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("version-2 custom_normal float3 values must be an object"))?;
    let mut normals = Vec::new();
    for face in &mesh.faces {
        let Some(row) = values.get(&format!("f{}", face.id)) else {
            normals.resize(normals.len() + face.vertices.len(), None);
            continue;
        };
        if row.is_null() {
            normals.resize(normals.len() + face.vertices.len(), None);
            continue;
        }
        let row = row
            .as_array()
            .filter(|row| row.len() == face.vertices.len())
            .ok_or_else(|| invalid("version-2 custom_normal float3 row has the wrong length"))?;
        for value in row {
            if value.is_null() {
                normals.push(None);
                continue;
            }
            let vector = value
                .as_array()
                .filter(|vector| vector.len() == 3)
                .ok_or_else(|| invalid("version-2 custom_normal float3 value must be a vector"))?;
            let component = |value: &Value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid("version-2 custom_normal float3 value must be finite"))
            };
            normals.push(Some(glam::DVec3::new(
                component(&vector[0])?,
                component(&vector[1])?,
                component(&vector[2])?,
            )));
        }
    }
    Ok(normals)
}

/// Whether a version-2 `attributes` entry is a user attribute. A few v2-owned shading stores were
/// also domain/type attributes, so their native shape must be distinguished from generic attributes
/// that merely use a name internal to v3. `color_layers` was never a v2 store.
fn is_v2_user_attribute(key: &str, attribute: &Value) -> bool {
    let generic = attribute.get("domain").is_some() && attribute.get("type").is_some();
    generic && !is_v2_native_attribute(key, attribute)
}

fn is_v2_native_attribute(key: &str, attribute: &Value) -> bool {
    let domain = attribute.get("domain").and_then(Value::as_str);
    let kind = attribute.get("type").and_then(Value::as_str);
    matches!(
        (key, domain, kind),
        (
            "custom_normal",
            Some("corner"),
            Some("float3" | "normal_code")
        ) | ("sharp" | "seam", Some("edge"), Some("bool"))
            | ("shade_smooth", Some("face"), Some("bool"))
            | ("crease_edge", Some("edge"), Some("float"))
    )
}

/// Moves every version-2 user attribute to its storage key. Keys that version 2 wrote itself keep theirs.
fn rekey_user_attributes(attributes: &mut Map<String, Value>) {
    let keys: Vec<String> = attributes
        .iter()
        .filter(|(key, attribute)| is_v2_user_attribute(key, attribute))
        .map(|(key, _)| key.clone())
        .collect();
    let moved: Vec<(String, Value)> = keys
        .into_iter()
        .filter_map(|key| attributes.remove(&key).map(|attribute| (key, attribute)))
        .collect();
    for (key, attribute) in moved {
        attributes.insert(user_key(&key), attribute);
    }
}

/// The storage key of the user attribute `name`.
fn user_key(name: &str) -> String {
    crate::geom::attribute_keys::storage_key(name).into_owned()
}

/// Upgrades one version-2 mesh in place.
///
/// - `blender_polygon_smooth` becomes `shade_smooth`, and `blender_edge_flags` becomes `sharp` and
///   `seam`. A native key that already exists wins, as the version-2 readers preferred it. A
///   positional array whose length does not match its domain is ignored, as those readers ignored it.
/// - Blender `custom_normal` `INT16_2D` pairs become the native `normal_code` store through
///   `CornerNormalCode::from_pair` and `clnor::write_codes`. A zero pair is Auto. The layer is present
///   whenever Blender has it, even if every pair is zero. Malformed pairs are boundary errors. Legacy
///   native float3 values are encoded once at this boundary, preserving explicit zero separately from
///   Auto.
/// - User attributes move to their storage keys (`attribute_keys::storage_key`), so a user name that
///   collides with a Potter store (`color_layers`, `vertex_groups`, `sharp`, ...) keeps its data.
///   Version 2's own stores keep their keys.
/// - `blender_attributes` colors (`FLOAT_COLOR` or `BYTE_COLOR` on POINT or CORNER) become color
///   layers in Blender's order, with their source encoding in `blender_metadata.color_data_types`.
/// - Other entries whose type and domain Potter represents become native generic attributes.
/// - Mirrors and topology are dropped. Selection and sculpt state, and anything unrepresentable,
///   stay in `blender_attributes`.
/// - The paint `color` attribute (`ops/paint.rs`, `f<face>:v<vertex>` keys) becomes a corner color
///   layer named `color`. A Blender layer of that name keeps it, and the paint layer takes a `.NNN`
///   suffix.
/// - The color registry lists only the color layers above. The first layer becomes both the active and
///   the render layer.
fn upgrade_mesh_v2(mesh: &mut Value) -> Result<()> {
    let Some(object) = mesh.as_object_mut() else {
        return Ok(());
    };
    let elements = Elements::read(object)?;
    let (custom_normal_codes, legacy_custom_normal) = {
        let Some(attributes) = object.get_mut("attributes").and_then(Value::as_object_mut) else {
            return Ok(());
        };
        rekey_user_attributes(attributes);
        upgrade_shading(attributes, &elements);
        (
            blender_custom_normal_codes(attributes, &elements)?,
            attributes
                .get("custom_normal")
                .filter(|attribute| {
                    attribute.get("domain").and_then(Value::as_str) == Some("corner")
                        && attribute.get("type").and_then(Value::as_str) == Some("float3")
                })
                .is_some(),
        )
    };
    let custom_normal_attribute = if let Some(codes) = custom_normal_codes {
        let mut upgraded_mesh: crate::geom::Mesh =
            serde_json::from_value(Value::Object(object.clone())).map_err(|error| {
                invalid(format!(
                    "version-2 mesh is invalid for custom normals: {error}"
                ))
            })?;
        upgraded_mesh.attributes.remove("custom_normal");
        crate::geom::clnor::write_codes(
            &mut upgraded_mesh,
            Some(&codes),
            crate::geom::clnor::NormalCodeOptions {
                corner_order: None,
                setter_input: crate::geom::clnor::SetterInput::NormalizeInput,
            },
        )?;
        upgraded_mesh.attributes.get("custom_normal").cloned()
    } else if legacy_custom_normal {
        let mut upgraded_mesh: crate::geom::Mesh =
            serde_json::from_value(Value::Object(object.clone())).map_err(|error| {
                invalid(format!(
                    "version-2 mesh is invalid for custom normals: {error}"
                ))
            })?;
        let vectors = legacy_custom_normal_vectors(&upgraded_mesh)?;
        upgraded_mesh.attributes.remove("custom_normal");
        crate::geom::clnor::encode_vectors(
            &mut upgraded_mesh,
            Some(&vectors),
            crate::geom::clnor::NormalCodeOptions {
                corner_order: None,
                setter_input: crate::geom::clnor::SetterInput::NormalizeInput,
            },
        )?;
        upgraded_mesh.attributes.get("custom_normal").cloned()
    } else {
        None
    };
    let Some(attributes) = object.get_mut("attributes").and_then(Value::as_object_mut) else {
        return Ok(());
    };
    if let Some(attribute) = custom_normal_attribute {
        attributes.insert("custom_normal".to_owned(), attribute);
    }
    // Reserved-name user attributes were moved aside before native shading conversion.
    // The paint attribute is set aside so that a Blender layer of the same name keeps its name.
    let paint = attributes.remove("color");
    let uv_names = uv_layer_names(attributes);
    let mut layers = Vec::new();
    let mut color_types = Map::new();
    let mut kept = Vec::new();
    match attributes.remove("blender_attributes") {
        Some(Value::Array(entries)) => {
            for entry in entries {
                if text(&entry, "name") == "custom_normal"
                    && text(&entry, "data_type") == "INT16_2D"
                {
                    continue;
                }
                match classify(&entry, &elements, &uv_names, attributes) {
                    Entry::Dropped => {}
                    Entry::Kept => kept.push(entry),
                    Entry::Generic { name, attribute } => {
                        attributes.insert(user_key(&name), attribute);
                    }
                    Entry::Color {
                        name,
                        attribute,
                        data_type,
                    } => {
                        color_types.insert(name.clone(), json!(data_type));
                        attributes.insert(user_key(&name), attribute);
                        layers.push(name);
                    }
                }
            }
        }
        Some(other) => {
            attributes.insert("blender_attributes".to_owned(), other);
        }
        None => {}
    }
    if !kept.is_empty() {
        attributes.insert("blender_attributes".to_owned(), Value::Array(kept));
    }
    if let Some(paint) = paint {
        match legacy_paint_layer(&paint, &elements) {
            Some(attribute) => {
                let name = unique_name("color", attributes);
                attributes.insert(name.clone(), attribute);
                layers.push(name);
            }
            None => {
                attributes.insert("color".to_owned(), paint);
            }
        }
    }
    let typed: Vec<String> = attributes
        .iter()
        .filter(|(_, attribute)| is_typed_color(attribute))
        .map(|(key, _)| key.clone())
        .collect();
    for key in typed {
        let Some(name) = crate::geom::attribute_keys::user_name(&key).map(str::to_owned) else {
            continue;
        };
        if layers.iter().any(|layer| layer == &name) {
            continue;
        }
        let Some(attribute) = attributes.remove(&key) else {
            continue;
        };
        match typed_color_layer(&attribute, &elements) {
            Some((converted, byte)) => {
                if byte {
                    color_types.insert(name.clone(), json!("BYTE_COLOR"));
                }
                attributes.insert(key, converted);
                layers.push(name);
            }
            None => {
                attributes.insert(key, attribute);
            }
        }
    }
    if !layers.is_empty() {
        let first = layers[0].clone();
        attributes.insert(
            crate::geom::color_layers::ATTRIBUTE.to_owned(),
            json!({"layers": layers, "active": first.clone(), "render": first}),
        );
    }
    if !color_types.is_empty() {
        let metadata = attributes
            .entry("blender_metadata")
            .or_insert_with(|| json!({}));
        let metadata = metadata
            .as_object_mut()
            .ok_or_else(|| invalid("blender_metadata must be an object"))?;
        metadata.insert("color_data_types".to_owned(), Value::Object(color_types));
    }
    Ok(())
}

/// What a version-2 `blender_attributes` entry becomes.
enum Entry {
    /// A mirror of a native key, or topology that Potter recomputes.
    Dropped,
    /// Blender-only data, which stays in `blender_attributes`.
    Kept,
    /// A native generic attribute.
    Generic { name: String, attribute: Value },
    /// A color layer, with its Blender source encoding.
    Color {
        name: String,
        attribute: Value,
        data_type: String,
    },
}

fn classify(
    entry: &Value,
    elements: &Elements,
    uv_names: &[String],
    attributes: &Map<String, Value>,
) -> Entry {
    let name = text(entry, "name");
    let domain = text(entry, "domain");
    let data_type = text(entry, "data_type");
    if dropped_name(name)
        || (domain == "CORNER" && data_type == "FLOAT2" && uv_names.iter().any(|uv| uv == name))
    {
        return Entry::Dropped;
    }
    if name.is_empty() || name.starts_with('.') || attributes.contains_key(user_key(name).as_str())
    {
        return Entry::Kept;
    }
    let Some(values) = entry.get("values").and_then(Value::as_array) else {
        return Entry::Kept;
    };
    if matches!(data_type, "FLOAT_COLOR" | "BYTE_COLOR") {
        return match blender_color_attribute(values, domain, elements) {
            Some(attribute) => Entry::Color {
                name: name.to_owned(),
                attribute,
                data_type: data_type.to_owned(),
            },
            None => Entry::Kept,
        };
    }
    if let (Some(domain), Some(kind)) = (native_domain(domain), native_type(data_type))
        && let Some(attribute) = generic_attribute(values, domain, kind, elements)
    {
        return Entry::Generic {
            name: name.to_owned(),
            attribute,
        };
    }
    Entry::Kept
}

/// Blender attribute names that are dropped: mirrors, and dot-prefixed attributes that are not state.
fn dropped_name(name: &str) -> bool {
    MIRROR_ATTRIBUTES.contains(&name)
        || (name.starts_with('.')
            && !BLENDER_STATE_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix)))
}

fn native_domain(domain: &str) -> Option<&'static str> {
    match domain {
        "POINT" => Some("point"),
        "EDGE" => Some("edge"),
        "FACE" => Some("face"),
        "CORNER" => Some("corner"),
        _ => None,
    }
}

fn native_type(data_type: &str) -> Option<&'static str> {
    match data_type {
        "FLOAT" => Some("float"),
        "INT" => Some("int"),
        "FLOAT2" => Some("float2"),
        "FLOAT_VECTOR" => Some("float3"),
        "BOOLEAN" => Some("bool"),
        "QUATERNION" => Some("quaternion"),
        _ => None,
    }
}

/// A Blender color attribute as a color layer. Blender's values are already linear floats (the
/// bridge reads `item.color`), so they are stored unchanged. `None` when the count is wrong.
fn blender_color_attribute(values: &[Value], domain: &str, elements: &Elements) -> Option<Value> {
    let colors = values
        .iter()
        .map(|value| color_value(value, 1.0))
        .collect::<Option<Vec<Value>>>()?;
    match domain {
        "POINT" if colors.len() == elements.vertices.len() => {
            let keyed: Map<String, Value> = elements
                .vertices
                .iter()
                .zip(colors)
                .map(|(id, color)| (format!("v{id}"), color))
                .collect();
            Some(json!({"domain": "point", "type": "color", "values": keyed}))
        }
        "CORNER" if colors.len() == elements.corner_count() => Some(json!({
            "domain": "corner",
            "type": "color",
            "values": corner_rows(&elements.faces, colors),
        })),
        _ => None,
    }
}

/// A Blender generic attribute as a native attribute, or `None` when its count or value types do not
/// match its domain and type.
fn generic_attribute(
    values: &[Value],
    domain: &str,
    kind: &str,
    elements: &Elements,
) -> Option<Value> {
    let count = match domain {
        "point" => elements.vertices.len(),
        "edge" => elements.edges.len(),
        "face" => elements.faces.len(),
        _ => elements.corner_count(),
    };
    if values.len() != count || !values.iter().all(|value| value_has_type(value, kind)) {
        return None;
    }
    let keyed: Map<String, Value> = match domain {
        "point" => elements
            .vertices
            .iter()
            .zip(values)
            .map(|(id, value)| (format!("v{id}"), value.clone()))
            .collect(),
        "edge" => elements
            .edges
            .iter()
            .zip(values)
            .map(|(id, value)| (format!("e{id}"), value.clone()))
            .collect(),
        "face" => elements
            .faces
            .iter()
            .zip(values)
            .map(|((id, _), value)| (format!("f{id}"), value.clone()))
            .collect(),
        _ => corner_rows(&elements.faces, values.iter().cloned()),
    };
    Some(json!({"domain": domain, "type": kind, "values": keyed}))
}

fn value_has_type(value: &Value, kind: &str) -> bool {
    match kind {
        "float" => value.as_f64().is_some_and(f64::is_finite),
        "int" => value.as_i64().is_some(),
        "bool" => value.is_boolean(),
        "float2" => numbers(value, 2),
        "float3" => numbers(value, 3),
        "quaternion" => numbers(value, 4),
        _ => false,
    }
}

fn numbers(value: &Value, count: usize) -> bool {
    value.as_array().is_some_and(|items| {
        items.len() == count
            && items
                .iter()
                .all(|item| item.as_f64().is_some_and(f64::is_finite))
    })
}

/// Corner rows keyed `f<face id>`, split from `values` in face corner order.
fn corner_rows(
    faces: &[(u64, Vec<u64>)],
    values: impl IntoIterator<Item = Value>,
) -> Map<String, Value> {
    let mut values = values.into_iter();
    faces
        .iter()
        .map(|(id, corners)| {
            let row: Vec<Value> = values.by_ref().take(corners.len()).collect();
            (format!("f{id}"), Value::Array(row))
        })
        .collect()
}

/// One `[r, g, b, a]` value as JSON, each channel multiplied by `scale`. `None` for anything else.
fn color_value(value: &Value, scale: f64) -> Option<Value> {
    let [red, green, blue, alpha] = value.as_array()?.as_slice() else {
        return None;
    };
    let channel = |value: &Value| {
        value
            .as_f64()
            .map(|value| value * scale)
            .filter(|value| value.is_finite())
    };
    Some(json!([
        channel(red)?,
        channel(green)?,
        channel(blue)?,
        channel(alpha)?
    ]))
}

fn opaque_black() -> Value {
    json!([0.0, 0.0, 0.0, 1.0])
}

/// The paint `color` attribute of version 2 (`ops/paint.rs`): corner values keyed `f<face>:v<vertex>`,
/// or point values keyed `v<vertex>`. Elements without a value take Blender's default, opaque black.
/// `None` for a typed attribute, which [`typed_color_layer`] handles.
fn legacy_paint_layer(paint: &Value, elements: &Elements) -> Option<Value> {
    if paint.get("type").is_some() {
        return None;
    }
    let values = paint.get("values").and_then(Value::as_object)?;
    let color_at = |key: &str| {
        values
            .get(key)
            .and_then(|value| color_value(value, 1.0))
            .unwrap_or_else(opaque_black)
    };
    match paint.get("domain").and_then(Value::as_str)? {
        "corners" => {
            let rows: Map<String, Value> = elements
                .faces
                .iter()
                .map(|(face, corners)| {
                    let row: Vec<Value> = corners
                        .iter()
                        .map(|vertex| color_at(&format!("f{face}:v{vertex}")))
                        .collect();
                    (format!("f{face}"), Value::Array(row))
                })
                .collect();
            Some(json!({"domain": "corner", "type": "color", "values": rows}))
        }
        "vertices" => {
            let keyed: Map<String, Value> = elements
                .vertices
                .iter()
                .map(|vertex| (format!("v{vertex}"), color_at(&format!("v{vertex}"))))
                .collect();
            Some(json!({"domain": "point", "type": "color", "values": keyed}))
        }
        _ => None,
    }
}

fn is_typed_color(attribute: &Value) -> bool {
    matches!(
        attribute.get("type").and_then(Value::as_str),
        Some("color" | "byte_color")
    ) && matches!(
        attribute.get("domain").and_then(Value::as_str),
        Some("point" | "corner")
    )
}

/// A top-level `color` or `byte_color` attribute in the native keyed form (`v<id>` for points,
/// `f<id>` rows for corners), as a complete color layer. Returns the layer and whether its source was
/// byte colors, whose channels are scaled from `0..255` to `[0, 1]`.
fn typed_color_layer(attribute: &Value, elements: &Elements) -> Option<(Value, bool)> {
    let byte = attribute.get("type").and_then(Value::as_str) == Some("byte_color");
    let scale = if byte { 1.0 / 255.0 } else { 1.0 };
    let values = attribute.get("values").and_then(Value::as_object)?;
    let converted = match attribute.get("domain").and_then(Value::as_str)? {
        "point" => {
            let keyed: Map<String, Value> = elements
                .vertices
                .iter()
                .map(|vertex| {
                    let color = values
                        .get(&format!("v{vertex}"))
                        .and_then(|value| color_value(value, scale))
                        .unwrap_or_else(opaque_black);
                    (format!("v{vertex}"), color)
                })
                .collect();
            json!({"domain": "point", "type": "color", "values": keyed})
        }
        "corner" => {
            let rows: Map<String, Value> = elements
                .faces
                .iter()
                .map(|(face, corners)| {
                    let stored = values.get(&format!("f{face}")).and_then(Value::as_array);
                    let row: Vec<Value> = (0..corners.len())
                        .map(|corner| {
                            stored
                                .and_then(|row| row.get(corner))
                                .and_then(|value| color_value(value, scale))
                                .unwrap_or_else(opaque_black)
                        })
                        .collect();
                    (format!("f{face}"), Value::Array(row))
                })
                .collect();
            json!({"domain": "corner", "type": "color", "values": rows})
        }
        _ => return None,
    };
    Some((converted, byte))
}

/// Blender's `polygon.use_smooth` and `edge.use_edge_sharp`/`use_seam` as native keys. A native key
/// that already exists wins over the positional array.
fn upgrade_shading(attributes: &mut Map<String, Value>, elements: &Elements) {
    if let Some(flags) = attributes.remove("blender_polygon_smooth")
        && let Some(rows) = flags
            .as_array()
            .filter(|rows| rows.len() == elements.faces.len())
        && !attributes.contains_key("shade_smooth")
    {
        let values: Map<String, Value> = elements
            .faces
            .iter()
            .zip(rows)
            .map(|((id, _), row)| (format!("f{id}"), json!(row.as_bool() == Some(true))))
            .collect();
        attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": values}),
        );
    }
    if let Some(flags) = attributes.remove("blender_edge_flags")
        && let Some(rows) = flags
            .as_array()
            .filter(|rows| rows.len() == elements.edges.len())
    {
        for name in ["sharp", "seam"] {
            if attributes.contains_key(name) {
                continue;
            }
            let values: Map<String, Value> = elements
                .edges
                .iter()
                .zip(rows)
                .map(|(id, row)| {
                    let flag = row.get(name).and_then(Value::as_bool) == Some(true);
                    (format!("e{id}"), json!(flag))
                })
                .collect();
            attributes.insert(
                name.to_owned(),
                json!({"domain": "edge", "type": "bool", "values": values}),
            );
        }
    }
}

/// Names of the layers in the mesh's `uv_layers` store.
fn uv_layer_names(attributes: &Map<String, Value>) -> Vec<String> {
    attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| layer.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

/// `base`, or `base.001`, `base.002`, … when an attribute already has that name.
fn unique_name(base: &str, attributes: &Map<String, Value>) -> String {
    if !attributes.contains_key(crate::geom::attribute_keys::storage_key(base).as_ref()) {
        return base.to_owned();
    }
    (1_usize..)
        .map(|index| format!("{base}.{index:03}"))
        .find(|name| {
            !attributes.contains_key(crate::geom::attribute_keys::storage_key(name).as_ref())
        })
        .unwrap_or_else(|| base.to_owned())
}

/// Text field of a JSON entry, or `""`.
fn text<'a>(entry: &'a Value, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "scene migration unit fixtures")]
    use serde_json::json;

    use super::*;

    #[test]
    fn inline_resource_kind_classifies_inline_records_without_treating_binary_as_inline() {
        assert_eq!(
            inline_resource_kind(&json!({"storage":"inline","kind":"legacy_texture"}))
                .expect("inline kind"),
            Some("legacy_texture")
        );
        assert_eq!(
            inline_resource_kind(&json!({"kind":"alembic","uri":"cache.abc"}))
                .expect("binary record"),
            None
        );
        assert_eq!(
            inline_resource_kind(&json!({"storage":"linked","kind":"alembic"}))
                .expect("linked record"),
            None
        );
        assert_eq!(
            inline_resource_kind(&json!({"storage":"inline"}))
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
    }

    fn legacy_volume_noise_scene() -> SceneDoc {
        let mut scene = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node: crate::model::Node = serde_json::from_value(json!({
            "name": "Volume",
            "kind": "empty",
            "modifiers": [
                {"id":"volume_a","type":"volume_displace","name":"Volume A","enabled":true,"params":{"texture":"noise"}},
                {"id":"volume_b","type":"volume_displace","name":"Volume B","enabled":true,"params":{"texture":"noise"}}
            ]
        }))
        .expect("volume node");
        scene
            .nodes
            .insert(crate::model::Id::new("volume").expect("node ID"), node);
        scene
    }

    fn assert_native_value_noise(scene: &SceneDoc) {
        let node = &scene.nodes[&crate::model::Id::new("volume").expect("node ID")];
        let texture = node.modifiers[0].params["texture"]
            .as_str()
            .expect("migrated texture ID");
        assert_eq!(
            node.modifiers[1].params["texture"],
            json!(texture),
            "all legacy volume modifiers share one migrated resource"
        );
        assert_eq!(
            scene.resources[&crate::model::Id::new(texture).expect("resource ID")],
            json!({
                "storage":"inline",
                "kind":"legacy_texture",
                "texture_type":"NATIVE_VALUE_NOISE",
                "settings":{}
            })
        );
    }

    #[test]
    fn parse_scene_migrates_the_legacy_volume_noise_sentinel() {
        let bytes = serde_json::to_vec(&legacy_volume_noise_scene()).expect("scene JSON");
        let scene = parse_scene(&bytes).expect("scene migration");
        assert_native_value_noise(&scene.doc);
        assert_eq!(scene.doc.resources.len(), 1);
    }

    #[test]
    fn parse_history_record_migrates_after_reading_its_raw_snapshot() {
        let record = HistoryRecord {
            record_version: 1,
            parent: None,
            kind: "commit".to_owned(),
            revision: 0,
            scene_hash: "sha256:raw-scene".to_owned(),
            operations: json!([]),
            changes: json!({}),
            target: None,
            snapshot: legacy_volume_noise_scene(),
        };
        let bytes = serde_json::to_vec(&record).expect("history JSON");
        let (record, warnings) = parse_history_record(&bytes).expect("history migration");
        assert!(warnings.is_empty());
        assert_native_value_noise(&record.snapshot);
        assert_eq!(record.snapshot.resources.len(), 1);
    }

    #[test]
    fn version_one_texture_refs_become_layer_references() {
        let mut scene = json!({
            "schema_version": 1,
            "materials": {"m1": {
                "base_color_texture": {"image": "img", "uv_map": "uv_map", "interpolation": "linear"},
                "normal_texture": {"image": "img", "uv_map": "Light", "interpolation": "linear"},
                "metallic_texture": {"image": "img", "uv_map": null, "interpolation": "linear"}
            }},
            "avatar": {"mtoon": {"m1": {
                "matcap_texture": {"image": "img", "uv_map": "uv_map", "interpolation": "linear"},
                "shading_shift_texture": {
                    "texture": {"image": "img", "uv_map": "Light", "interpolation": "linear"},
                    "scale": 1.0
                }
            }}}
        });
        upgrade_scene(&mut scene).expect("upgrade");
        assert_eq!(scene["schema_version"], json!(3));
        let material = &scene["materials"]["m1"];
        assert_eq!(material["base_color_texture"]["uv_layer"], Value::Null);
        assert!(material["base_color_texture"].get("uv_map").is_none());
        assert_eq!(material["normal_texture"]["uv_layer"], json!("Light"));
        assert_eq!(material["metallic_texture"]["uv_layer"], Value::Null);
        let mtoon = &scene["avatar"]["mtoon"]["m1"];
        assert_eq!(mtoon["matcap_texture"]["uv_layer"], Value::Null);
        assert_eq!(
            mtoon["shading_shift_texture"]["texture"]["uv_layer"],
            json!("Light")
        );
    }

    #[test]
    fn current_scenes_pass_through_untouched() {
        let mut scene = json!({
            "schema_version": 3,
            "materials": {"m1": {"base_color_texture": {"image": "img", "uv_layer": "Light"}}}
        });
        let before = scene.clone();
        upgrade_scene(&mut scene).expect("no-op");
        assert_eq!(scene, before);
    }

    #[test]
    fn loading_version_one_scene_converts_both_uv_stores() -> TestResult {
        use glam::DVec3;

        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mesh = crate::geom::Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )?;
        let face_id = mesh.faces[0].id;
        let data_id = crate::model::Id::new("meshdata")?;
        doc.data_blocks.insert(
            data_id.clone(),
            crate::model::DataBlock {
                mesh: Some(mesh),
                ..crate::model::DataBlock::default()
            },
        );
        let mut source = serde_json::to_value(doc)?;
        source["schema_version"] = json!(1);
        source["data_blocks"]["meshdata"]["mesh"]["attributes"] = json!({
            "uv_map": [{"face_id": face_id, "uv": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}],
            "blender_uv_layers": [{
                "name": "Light",
                "values": [[0.0, 0.25], [0.5, 0.5], [1.0, 0.75]]
            }]
        });

        let migrated = parse_scene(&serde_json::to_vec(&source)?)?.doc;
        assert_eq!(migrated.schema_version, 3);
        let mesh = migrated.data_blocks[&data_id]
            .mesh
            .as_ref()
            .ok_or("migrated data block has no mesh")?;
        let store = crate::geom::uv_layers::read(mesh);
        assert_eq!(store.layers.len(), 2);
        assert_eq!(store.layers[0].name, "Light");
        assert_eq!(store.layers[1].name, "UVMap");
        assert_eq!(store.render.as_deref(), Some("UVMap"));
        assert_eq!(
            store.layers[0].faces[0],
            Some(vec![[0.0, 0.75], [0.5, 0.5], [1.0, 0.25]])
        );
        assert_eq!(
            store.layers[1].faces[0],
            Some(vec![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        );
        Ok(())
    }

    fn triangle_mesh() -> TestResult<crate::geom::Mesh> {
        use glam::DVec3;
        Ok(crate::geom::Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )?)
    }

    /// A schema-version-2 document holding `mesh` with the given mesh `attributes`.
    fn version_two_document(mesh: &crate::geom::Mesh, attributes: Value) -> TestResult<Value> {
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        doc.data_blocks.insert(
            crate::model::Id::new("meshdata")?,
            crate::model::DataBlock {
                mesh: Some(mesh.clone()),
                ..crate::model::DataBlock::default()
            },
        );
        let mut source = serde_json::to_value(doc)?;
        source["schema_version"] = json!(2);
        source["data_blocks"]["meshdata"]["mesh"]["attributes"] = attributes;
        Ok(source)
    }

    fn migrated_mesh(source: &Value) -> TestResult<crate::geom::Mesh> {
        let migrated = parse_scene(&serde_json::to_vec(source)?)?.doc;
        let mesh = migrated.data_blocks[&crate::model::Id::new("meshdata")?]
            .mesh
            .clone()
            .ok_or("migrated data block has no mesh")?;
        crate::geom::color_layers::validate(&mesh)?;
        Ok(mesh)
    }

    #[test]
    fn version_two_blender_keys_become_native_keys() -> TestResult {
        let mesh = triangle_mesh()?;
        let face = format!("f{}", mesh.faces[0].id);
        let edges: Vec<String> = mesh
            .edges
            .iter()
            .map(|edge| format!("e{}", edge.id))
            .collect();
        let vertices: Vec<String> = mesh
            .vertices
            .iter()
            .map(|vertex| format!("v{}", vertex.id))
            .collect();
        let source = version_two_document(
            &mesh,
            json!({
                "blender_polygon_smooth": [true],
                "blender_edge_flags": [
                    {"sharp": true, "seam": false},
                    {"sharp": false, "seam": true},
                    {"sharp": false, "seam": false}
                ],
                "blender_metadata": {"rna_properties": {"kept": true}},
                "blender_attributes": [
                    {"name": "Col", "domain": "POINT", "data_type": "FLOAT_COLOR", "values": [
                        [0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 0.5], [0.7, 0.8, 0.9, 1.0]
                    ]},
                    {"name": "Paint", "domain": "CORNER", "data_type": "BYTE_COLOR", "values": [
                        [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]
                    ]},
                    {"name": "EdgeColor", "domain": "EDGE", "data_type": "FLOAT_COLOR", "values": [
                        [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
                    ]},
                    {"name": "_wetness", "domain": "POINT", "data_type": "FLOAT", "values": [0.5, 0.25, 0.0]},
                    {"name": "face_group", "domain": "FACE", "data_type": "INT", "values": [7]},
                    {"name": "_marked", "domain": "FACE", "data_type": "BOOLEAN", "values": [true]},
                    {"name": "drift", "domain": "POINT", "data_type": "FLOAT_VECTOR", "values": [
                        [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]
                    ]},
                    {"name": "sharp_face", "domain": "FACE", "data_type": "BOOLEAN", "values": [false]},
                    {"name": ".select_vert", "domain": "POINT", "data_type": "BOOLEAN", "values": [true, false, false]},
                    {"name": ".edge_verts", "domain": "EDGE", "data_type": "INT32_2D", "values": [[0, 1], [1, 2], [2, 0]]},
                    {"name": "custom_normal", "domain": "CORNER", "data_type": "INT16_2D", "values": [[0, 0], [0, 0], [0, 0]]}
                ]
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        let attributes = &migrated.attributes;
        assert!(!attributes.contains_key("blender_polygon_smooth"));
        assert!(!attributes.contains_key("blender_edge_flags"));
        assert_eq!(
            attributes["shade_smooth"]["values"][face.as_str()],
            json!(true)
        );
        assert_eq!(
            attributes["sharp"]["values"][edges[0].as_str()],
            json!(true)
        );
        assert_eq!(
            attributes["sharp"]["values"][edges[1].as_str()],
            json!(false)
        );
        assert_eq!(attributes["seam"]["values"][edges[1].as_str()], json!(true));
        assert_eq!(
            attributes["seam"]["values"][edges[0].as_str()],
            json!(false)
        );
        assert_eq!(attributes["seam"]["type"], json!("bool"));

        let store = crate::geom::color_layers::read(&migrated);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Col", "Paint"]);
        assert_eq!(store.active.as_deref(), Some("Col"));
        assert_eq!(store.render.as_deref(), Some("Col"));
        assert_eq!(store.layers[0].values[1], [0.4, 0.5, 0.6, 0.5]);
        assert_eq!(
            store.layers[1].values,
            vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ]
        );
        assert_eq!(
            attributes["blender_metadata"]["color_data_types"],
            json!({"Col": "FLOAT_COLOR", "Paint": "BYTE_COLOR"})
        );
        assert_eq!(
            attributes["blender_metadata"]["rna_properties"],
            json!({"kept": true})
        );

        assert_eq!(attributes["_wetness"]["domain"], json!("point"));
        assert_eq!(attributes["_wetness"]["type"], json!("float"));
        assert_eq!(
            attributes["_wetness"]["values"][vertices[0].as_str()],
            json!(0.5)
        );
        assert_eq!(attributes["face_group"]["type"], json!("int"));
        assert_eq!(attributes["face_group"]["values"][face.as_str()], json!(7));
        assert_eq!(attributes["_marked"]["type"], json!("bool"));
        assert_eq!(attributes["_marked"]["values"][face.as_str()], json!(true));
        assert_eq!(attributes["drift"]["type"], json!("float3"));
        assert_eq!(
            attributes["drift"]["values"][vertices[2].as_str()],
            json!([1.0, 0.0, 0.0])
        );

        assert_eq!(attributes["custom_normal"]["domain"], json!("corner"));
        assert_eq!(attributes["custom_normal"]["type"], json!("normal_code"));
        assert_eq!(
            attributes["custom_normal"]["values"][face.as_str()],
            json!([null, null, null])
        );
        let kept: Vec<&str> = attributes["blender_attributes"]
            .as_array()
            .ok_or("blender_attributes is missing")?
            .iter()
            .filter_map(|entry| entry["name"].as_str())
            .collect();
        assert_eq!(kept, ["EdgeColor", ".select_vert"]);
        Ok(())
    }

    #[test]
    fn version_two_custom_normal_codes_become_native_code_and_auto_values() -> TestResult {
        let mesh = triangle_mesh()?;
        let face = format!("f{}", mesh.faces[0].id);
        let edges: Vec<String> = mesh
            .edges
            .iter()
            .map(|edge| format!("e{}", edge.id))
            .collect();
        let source = version_two_document(
            &mesh,
            json!({
                "blender_polygon_smooth": [true],
                "blender_edge_flags": [
                    {"sharp": true, "seam": true},
                    {"sharp": false, "seam": false},
                    {"sharp": false, "seam": false}
                ],
                "blender_attributes": [{
                    "name": "custom_normal",
                    "domain": "CORNER",
                    "data_type": "INT16_2D",
                    "values": [[16000, 0], [0, 0], [0, 0]]
                }]
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        let attributes = &migrated.attributes;
        assert_eq!(
            attributes["shade_smooth"]["values"][face.as_str()],
            json!(true)
        );
        assert_eq!(
            attributes["sharp"]["values"][edges[0].as_str()],
            json!(true)
        );
        assert_eq!(attributes["seam"]["values"][edges[0].as_str()], json!(true));

        assert_eq!(attributes["custom_normal"]["domain"], json!("corner"));
        assert_eq!(attributes["custom_normal"]["type"], json!("normal_code"));
        let corners = attributes["custom_normal"]["values"][face.as_str()]
            .as_array()
            .ok_or("migrated custom normal row is missing")?;
        assert_eq!(corners[0], json!([16000, 0]));
        assert_eq!(corners[1], Value::Null);
        assert_eq!(corners[2], Value::Null);
        assert!(
            !attributes
                .get("blender_attributes")
                .and_then(Value::as_array)
                .is_some_and(|entries| {
                    entries.iter().any(|entry| entry["name"] == "custom_normal")
                })
        );
        Ok(())
    }

    #[test]
    fn version_two_legacy_custom_normal_preserves_explicit_zero() -> TestResult {
        let mesh = triangle_mesh()?;
        let face_id = mesh.faces[0].id;
        let face = format!("f{face_id}");
        let mut values = Map::new();
        values.insert(
            face.clone(),
            json!([[0.0, 0.0, 0.0], null, [0.0, 0.0, 1.0]]),
        );
        let source = version_two_document(
            &mesh,
            json!({
                "blender_polygon_smooth": [true],
                "custom_normal": {
                    "domain": "corner",
                    "type": "float3",
                    "values": values
                }
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        assert_eq!(
            migrated.attributes["custom_normal"]["type"],
            json!("normal_code")
        );
        assert_eq!(
            migrated.attributes["custom_normal"]["values"][face.as_str()],
            json!(["explicit_zero", null, null])
        );
        Ok(())
    }

    #[test]
    fn malformed_version_two_custom_normal_codes_are_scene_invalid() -> TestResult {
        let mesh = triangle_mesh()?;
        for values in [json!([[0, 0], [32768, 0], [0, 0]]), json!([[0, 0], [0, 0]])] {
            let source = version_two_document(
                &mesh,
                json!({
                    "blender_attributes": [{
                        "name": "custom_normal",
                        "domain": "CORNER",
                        "data_type": "INT16_2D",
                        "values": values
                    }]
                }),
            )?;
            let error = parse_scene(&serde_json::to_vec(&source)?)
                .expect_err("malformed custom-normal codes must be rejected");
            assert_eq!(error.code, ErrorCode::SceneInvalid);
        }
        Ok(())
    }

    #[test]
    fn version_two_paint_color_becomes_a_color_layer_beside_blender_color() -> TestResult {
        let mesh = triangle_mesh()?;
        let face = mesh.faces[0].id;
        let mut paint = serde_json::Map::new();
        paint.insert(
            format!("f{face}:v{}", mesh.vertices[1].id),
            json!([0.5, 0.25, 0.125, 1.0]),
        );
        let source = version_two_document(
            &mesh,
            json!({
                "color": {"domain": "corners", "values": paint},
                "blender_attributes": [{
                    "name": "color", "domain": "POINT", "data_type": "FLOAT_COLOR",
                    "values": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
                }]
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        assert_eq!(migrated.attributes["color"]["domain"], json!("point"));
        let store = crate::geom::color_layers::read(&migrated);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["color", "color.001"]);
        assert_eq!(
            store.layers[1].domain,
            crate::geom::color_layers::ColorDomain::Corner
        );
        assert_eq!(
            store.layers[1].values,
            vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.25, 0.125, 1.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );
        Ok(())
    }

    #[test]
    fn version_two_paint_color_suffix_preserves_generic_float_color() -> TestResult {
        let mesh = triangle_mesh()?;
        let face = mesh.faces[0].id;
        let mut paint = Map::new();
        paint.insert(
            format!("f{face}:v{}", mesh.vertices[1].id),
            json!([0.5, 0.25, 0.125, 1.0]),
        );
        let source = version_two_document(
            &mesh,
            json!({
                "color": {"domain": "corners", "values": paint},
                "blender_attributes": [{
                    "name": "color", "domain": "POINT", "data_type": "FLOAT",
                    "values": [0.1, 0.2, 0.3]
                }]
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        let vertex = format!("v{}", mesh.vertices[1].id);
        assert_eq!(
            migrated.attributes["color"]["values"][vertex.as_str()],
            json!(0.2)
        );
        let store = crate::geom::color_layers::read(&migrated);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["color.001"]);
        assert_eq!(store.active.as_deref(), Some("color.001"));
        assert_eq!(store.render.as_deref(), Some("color.001"));
        Ok(())
    }

    #[test]
    fn version_two_user_attribute_named_color_layers_is_kept_beside_a_typed_color() -> TestResult {
        let mesh = triangle_mesh()?;
        let vertices: Vec<String> = mesh
            .vertices
            .iter()
            .map(|vertex| format!("v{}", vertex.id))
            .collect();
        let mut floats = Map::new();
        floats.insert(vertices[0].clone(), json!(0.5));
        floats.insert(vertices[1].clone(), json!(0.25));
        floats.insert(vertices[2].clone(), json!(0.0));
        let mut tint = Map::new();
        for vertex in &vertices {
            tint.insert(vertex.clone(), json!([1.0, 0.0, 0.0, 1.0]));
        }
        let source = version_two_document(
            &mesh,
            json!({
                "color_layers": {"domain": "point", "type": "float", "values": floats},
                "Tint": {"domain": "point", "type": "color", "values": tint}
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        let attributes = &migrated.attributes;
        assert_eq!(attributes["user:color_layers"]["type"], json!("float"));
        assert_eq!(
            attributes["user:color_layers"]["values"][vertices[1].as_str()],
            json!(0.25)
        );
        assert_eq!(attributes["color_layers"]["layers"], json!(["Tint"]));
        let store = crate::geom::color_layers::read(&migrated);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Tint"]);
        assert_eq!(store.active.as_deref(), Some("Tint"));
        assert_eq!(store.render.as_deref(), Some("Tint"));
        Ok(())
    }

    #[test]
    fn version_two_user_attribute_named_color_layers_without_a_color_leaves_no_registry()
    -> TestResult {
        let mesh = triangle_mesh()?;
        let vertex = format!("v{}", mesh.vertices[1].id);
        let mut floats = Map::new();
        floats.insert(vertex.clone(), json!(0.25));
        let source = version_two_document(
            &mesh,
            json!({"color_layers": {"domain": "point", "type": "float", "values": floats}}),
        )?;
        let migrated = migrated_mesh(&source)?;
        assert!(!migrated.attributes.contains_key("color_layers"));
        assert_eq!(
            migrated.attributes["user:color_layers"]["values"][vertex.as_str()],
            json!(0.25)
        );
        assert!(crate::geom::color_layers::read(&migrated).layers.is_empty());
        Ok(())
    }

    #[test]
    fn version_two_blender_names_are_stored_under_storage_keys() -> TestResult {
        let mesh = triangle_mesh()?;
        let vertex = format!("v{}", mesh.vertices[0].id);
        let source = version_two_document(
            &mesh,
            json!({"blender_attributes": [
                {"name": "skin_tone", "domain": "POINT", "data_type": "FLOAT", "values": [0.1, 0.2, 0.3]},
                {"name": "vertex_groups", "domain": "POINT", "data_type": "FLOAT", "values": [1.0, 2.0, 3.0]}
            ]}),
        )?;
        let migrated = migrated_mesh(&source)?;
        assert_eq!(
            migrated.attributes["skin_tone"]["values"][vertex.as_str()],
            json!(0.1)
        );
        assert_eq!(
            migrated.attributes["user:vertex_groups"]["values"][vertex.as_str()],
            json!(1.0)
        );
        assert!(!migrated.attributes.contains_key("vertex_groups"));
        Ok(())
    }

    #[test]
    fn version_two_generic_sharp_and_seam_names_are_rekeyed() -> TestResult {
        let mesh = triangle_mesh()?;
        let vertices: Vec<String> = mesh
            .vertices
            .iter()
            .map(|vertex| format!("v{}", vertex.id))
            .collect();
        let mut sharp_values = Map::new();
        sharp_values.insert(vertices[0].clone(), json!(0.75));
        let mut seam_values = Map::new();
        seam_values.insert(vertices[1].clone(), json!(0.25));
        let source = version_two_document(
            &mesh,
            json!({
                "sharp": {"domain": "point", "type": "float", "values": sharp_values},
                "seam": {"domain": "point", "type": "float", "values": seam_values}
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        assert_eq!(
            migrated.attributes["user:sharp"]["values"][vertices[0].as_str()],
            json!(0.75)
        );
        assert_eq!(
            migrated.attributes["user:seam"]["values"][vertices[1].as_str()],
            json!(0.25)
        );
        assert!(!migrated.attributes.contains_key("sharp"));
        assert!(!migrated.attributes.contains_key("seam"));
        Ok(())
    }

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
}
