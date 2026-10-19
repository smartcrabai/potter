// SPDX-License-Identifier: GPL-3.0-or-later

//! Scene document upgrades. [`parse_scene`] is the one way to read a scene document (`scene.json`,
//! a library scene, or a history snapshot). It upgrades the raw JSON one schema version at a time,
//! then deserializes. Version 2 replaced the mesh attributes `uv_map` and `blender_uv_layers` with
//! `uv_layers`, and the texture reference field `uv_map` with `uv_layer`. Version 3 converted
//! Blender-keyed shading (including compressed custom normals), color, and generic element data to
//! native keys, and user attributes to their storage keys (see [`upgrade_mesh_v2`]). Version 4 marks an EXR image's
//! alpha as premultiplied (see [`upgrade_v3`]) and adds optional captured Blender authoring matrices, which older
//! documents lack.
use std::{collections::BTreeSet, path::Path};

use serde_json::{Map, Value, json};

use super::{Image, ImageAlphaMode, SceneDoc};
use crate::{
    color::curve::{CurveMapping, CurvePoint, HandleType},
    error::{ErrorCode, PotError, Result},
    store::HistoryRecord,
};

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

/// A scene document read from bytes, with diagnostics from its schema upgrade.
#[derive(Debug)]
pub struct ParsedScene {
    pub doc: SceneDoc,
    /// Diagnostics for legacy data the upgrade skipped or dropped.
    pub warnings: Vec<Value>,
    /// Whether the original bytes used schema version 1.
    pub legacy: bool,
    /// Canonical hash of the original schema-1 scene, before migration changes its schema version.
    pub legacy_hash: Option<String>,
}

/// Parses a scene document, upgrading it through every schema version to the current one.
pub fn parse_scene(bytes: &[u8], root: &Path) -> Result<ParsedScene> {
    parse_scene_inner(bytes, root, false)
}

/// Parses a project scene while retaining its pre-migration hash for project metadata.
pub(crate) fn parse_scene_with_original_hash(bytes: &[u8], root: &Path) -> Result<ParsedScene> {
    parse_scene_inner(bytes, root, true)
}

fn parse_scene_inner(bytes: &[u8], root: &Path, retain_original_hash: bool) -> Result<ParsedScene> {
    let mut value = parse_value(bytes)?;
    let legacy = value.get("schema_version").and_then(Value::as_u64) == Some(1);
    let legacy_hash = if legacy && retain_original_hash {
        Some(crate::hash::sha256(&crate::hash::canonicalize(&value)?))
    } else {
        None
    };
    let warnings = upgrade_scene(&mut value, root)?;
    let doc = serde_json::from_value(value).map_err(|error| json_error(&error))?;
    Ok(ParsedScene {
        doc,
        warnings,
        legacy,
        legacy_hash,
    })
}

/// Parses a history record, upgrading its snapshot's schema version and retiring the removed operation
/// `seed` fields from its recorded operations.
pub(crate) fn parse_history_record(
    bytes: &[u8],
    root: &Path,
) -> Result<(HistoryRecord, Vec<Value>)> {
    let mut value = parse_value(bytes)?;
    let mut warnings = match value.get_mut("snapshot") {
        Some(snapshot) => upgrade_scene(snapshot, root)?,
        None => Vec::new(),
    };
    if let Some(operations) = value.get_mut("operations") {
        warnings.extend(retire_operation_seed(operations));
    }
    let record = serde_json::from_value(value).map_err(|error| json_error(&error))?;
    Ok((record, warnings))
}

/// Drops the `seed` key from recorded operations whose schema no longer accepts it: `sculpt.stroke`,
/// `sculpt.dyntopo`, `paint.vertex`, `paint.weight` and `paint.texture`. The parser rejects unknown
/// fields, so an old record would not replay. Each dropped seed is reported. The sculpt brushes read
/// the value in older builds, so a replay of such a record can differ from the one it was recorded with.
fn retire_operation_seed(operations: &mut Value) -> Vec<Value> {
    let mut warnings = Vec::new();
    let Some(operations) = operations.as_array_mut() else {
        return warnings;
    };
    for (index, operation) in operations.iter_mut().enumerate() {
        let Some(fields) = operation.as_object_mut() else {
            continue;
        };
        let Some(name) = fields
            .get("op")
            .and_then(Value::as_str)
            .filter(|name| {
                matches!(
                    *name,
                    "sculpt.stroke"
                        | "sculpt.dyntopo"
                        | "paint.vertex"
                        | "paint.weight"
                        | "paint.texture"
                )
            })
            .map(str::to_owned)
        else {
            continue;
        };
        if let Some(seed) = fields.remove("seed") {
            warnings.push(json!({
                "code": "operation.seed_retired",
                "message": "seed is no longer an operation field and was dropped",
                "operation_index": index,
                "op": name,
                "seed": seed,
            }));
        }
    }
    warnings
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

/// Upgrades a scene document in place through schema version 6. `root` is the project directory that holds the document's
/// image blobs and linked sources, which the version 3 to 4 step reads.
pub fn upgrade_scene(value: &mut Value, root: &Path) -> Result<Vec<Value>> {
    let warnings = if value.get("schema_version").and_then(Value::as_u64) == Some(1) {
        upgrade_v1(value)?
    } else {
        Vec::new()
    };
    if value.get("schema_version").and_then(Value::as_u64) == Some(2) {
        upgrade_v2(value)?;
    }
    if value.get("schema_version").and_then(Value::as_u64) == Some(3) {
        upgrade_v3(value, root)?;
    }
    if value.get("schema_version").and_then(Value::as_u64) == Some(4) {
        upgrade_v4(value)?;
    }
    if value.get("schema_version").and_then(Value::as_u64) == Some(5) {
        upgrade_v5(value);
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

/// Every material and `MToon` texture reference, including the nested shading-shift texture.
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

/// Converts one version-1 texture selector. The old default names the render layer.
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

/// Upgrades a schema-version-4 scene document to version 5. The released view curve, an array of
/// `[x, y]` points evaluated piecewise-linearly with horizontal extension, becomes Blender's view
/// `CurveMapping` on the combined curve with VECTOR handles and horizontal extension, so existing
/// renders keep their shape. The R/G/B identity curves cover the combined curve's full x-domain and
/// `[0, 1]`; a single legacy point becomes a constant two-point curve. Clip is `[0, 1]`, black and
/// white identity. An empty array means no curve.
fn upgrade_v4(value: &mut Value) -> Result<()> {
    if let Some(scenes) = value.get_mut("scenes").and_then(Value::as_object_mut) {
        for scene in scenes.values_mut() {
            if let Some(color) = scene.get_mut("color_management") {
                upgrade_color_curve(color)?;
            }
        }
    }
    value["schema_version"] = json!(5);
    Ok(())
}

/// Upgrades a schema-version-5 scene document to version 6. Version 5 drew a fisheye camera over a full turn in two forms:
/// the `fisheye` projection, and a `panorama` with `panorama_type` `fisheye_equidistant`. Both become a `panorama` with
/// `fisheye_equidistant` and a `fisheye_fov` of 2π, so each keeps its image and a fisheye has one representation.
fn upgrade_v5(value: &mut Value) {
    if let Some(blocks) = value.get_mut("data_blocks").and_then(Value::as_object_mut) {
        for camera in blocks
            .values_mut()
            .filter_map(|block| block.get_mut("camera")?.as_object_mut())
        {
            let full_turn_fisheye = match camera.get("projection").and_then(Value::as_str) {
                Some("fisheye") => true,
                Some("panorama") => {
                    camera.get("panorama_type").and_then(Value::as_str)
                        == Some("fisheye_equidistant")
                }
                _ => false,
            };
            if full_turn_fisheye {
                camera.insert("projection".to_owned(), json!("panorama"));
                camera.insert("panorama_type".to_owned(), json!("fisheye_equidistant"));
                camera.insert("fisheye_fov".to_owned(), json!(std::f64::consts::TAU));
            }
        }
    }
    value["schema_version"] = json!(6);
}

fn upgrade_color_curve(color: &mut Value) -> Result<()> {
    let Some(curve) = color.get_mut("curve") else {
        return Ok(());
    };
    if curve.is_null() {
        return Ok(());
    }
    let Value::Array(points) = curve else {
        return Err(invalid(
            "color curve must be an array of [x, y] points or null",
        ));
    };
    let mut points = points
        .iter()
        .map(legacy_curve_point)
        .collect::<Result<Vec<_>>>()?;
    let converted = if points.is_empty() {
        Value::Null
    } else {
        if points.len() == 1 {
            let y = points[0].location[1];
            points = vec![
                CurvePoint {
                    location: [0.0, y],
                    handle: HandleType::Vector,
                },
                CurvePoint {
                    location: [1.0, y],
                    handle: HandleType::Vector,
                },
            ];
        }
        let x_min = points[0].location[0].min(0.0);
        let x_max = points[points.len() - 1].location[0].max(1.0);
        let mut mapping = CurveMapping::default();
        for channel in &mut mapping.curves[..3] {
            channel.points = vec![
                CurvePoint {
                    location: [x_min, x_min],
                    handle: HandleType::Vector,
                },
                CurvePoint {
                    location: [x_max, x_max],
                    handle: HandleType::Vector,
                },
            ];
        }
        mapping.curves[3].points = points;
        mapping.extend_extrapolate = false;
        mapping.validate()?;
        serde_json::to_value(&mapping).map_err(|error| json_error(&error))?
    };
    *curve = converted;
    Ok(())
}

fn legacy_curve_point(point: &Value) -> Result<CurvePoint> {
    let [x, y] = point
        .as_array()
        .and_then(|pair| <&[Value; 2]>::try_from(pair.as_slice()).ok())
        .ok_or_else(|| invalid("color curve point must be [x, y]"))?;
    let coordinate = |value: &Value| {
        value
            .as_f64()
            .map(|number| number as f32)
            .ok_or_else(|| invalid("color curve coordinate must be a number"))
    };
    Ok(CurvePoint {
        location: [coordinate(x)?, coordinate(y)?],
        handle: HandleType::Vector,
    })
}

/// Upgrades a schema-version-3 scene document to version 4. Schema 3 labelled every image straight, EXR included, so a
/// saved EXR was read as straight. Blender reads an EXR as premultiplied, and so does the loader now. The old loader
/// unpremultiplied an EXR at load and premultiplied it again when sampling, so an EXR labelled straight becomes
/// premultiplied here, and then samples the colour the old loader produced. Only images whose bytes are an EXR change. An
/// image whose bytes cannot be read is left as it is, and the loader reports the missing or changed asset when it is used.
fn upgrade_v3(value: &mut Value, root: &Path) -> Result<()> {
    if let Some(images) = value.get_mut("images").and_then(Value::as_object_mut) {
        for image in images.values_mut() {
            // A missing alpha mode deserialises to straight, so it is straight here too.
            if image
                .get("alpha_mode")
                .and_then(Value::as_str)
                .is_some_and(|mode| mode != "straight")
            {
                continue;
            }
            let typed: Image =
                serde_json::from_value(image.clone()).map_err(|error| json_error(&error))?;
            let Ok(bytes) = crate::image::storage::image_source_bytes(&typed, root) else {
                continue;
            };
            if crate::image::storage::default_alpha_mode(&bytes) == ImageAlphaMode::Premultiplied {
                image["alpha_mode"] = json!("premultiplied");
            }
        }
    }
    value["schema_version"] = json!(4);
    Ok(())
}
/// Blender-owned mesh attributes are mirrors or topology rebuilt by Potter. Their exact signatures
/// match the `.blend` importer.
fn is_blender_managed_attribute(name: &str, domain: &str, data_type: &str) -> bool {
    matches!(
        (name, domain, data_type),
        ("position", "POINT", "FLOAT_VECTOR")
            | ("sharp_face", "FACE", "BOOLEAN")
            | ("sharp_edge" | "uv_seam", "EDGE", "BOOLEAN")
            | ("material_index", "FACE", "INT")
            | ("custom_normal", "CORNER", "INT16_2D")
    )
}

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
        text(entry, "name") == "custom_normal"
            && is_blender_managed_attribute(
                text(entry, "name"),
                text(entry, "domain"),
                text(entry, "data_type"),
            )
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
            attributes.get("custom_normal").is_some_and(|attribute| {
                attribute.get("domain").and_then(Value::as_str) == Some("corner")
                    && attribute.get("type").and_then(Value::as_str) == Some("float3")
            }),
        )
    };
    let (custom_normal_attribute, legacy_sharp_attribute) = if let Some(codes) = custom_normal_codes
    {
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
        (upgraded_mesh.attributes.get("custom_normal").cloned(), None)
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
        (
            upgraded_mesh.attributes.get("custom_normal").cloned(),
            upgraded_mesh.attributes.get("sharp").cloned(),
        )
    } else {
        (None, None)
    };
    let Some(attributes) = object.get_mut("attributes").and_then(Value::as_object_mut) else {
        return Ok(());
    };
    if let Some(attribute) = custom_normal_attribute {
        attributes.insert("custom_normal".to_owned(), attribute);
    }
    if let Some(sharp) = legacy_sharp_attribute {
        attributes.insert("sharp".to_owned(), sharp);
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
    if dropped_name(name, domain, data_type)
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

/// Blender-managed signatures and dot-prefixed attributes that are not state are dropped.
fn dropped_name(name: &str, domain: &str, data_type: &str) -> bool {
    is_blender_managed_attribute(name, domain, data_type)
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
    (1_usize..=usize::MAX)
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

    /// A project root with no assets, for documents that name no image blobs.
    fn no_assets() -> &'static Path {
        Path::new("no-assets")
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
        upgrade_scene(&mut scene, no_assets()).expect("upgrade");
        assert_eq!(scene["schema_version"], json!(6));
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
    fn version_three_scenes_only_advance_their_version_without_images() {
        let mut scene = json!({
            "schema_version": 3,
            "materials": {"m1": {"base_color_texture": {"image": "img", "uv_layer": "Light"}}}
        });
        let before = scene.clone();
        upgrade_scene(&mut scene, no_assets()).expect("upgrade");
        assert_eq!(scene["schema_version"], json!(6));
        assert_eq!(scene["materials"], before["materials"]);
    }

    #[test]
    fn version_five_full_turn_fisheyes_become_equidistant_panoramas() {
        let mut scene = json!({
            "schema_version": 5,
            "data_blocks": {
                "legacy": {"type": "camera", "camera": {"projection": "fisheye", "lens_mm": 50.0}},
                "panorama": {"type": "camera", "camera": {
                    "projection": "panorama", "panorama_type": "fisheye_equidistant", "lens_mm": 50.0
                }},
                "sphere": {"type": "camera", "camera": {
                    "projection": "panorama", "panorama_type": "equirectangular"
                }},
                "plain": {"type": "camera", "camera": {"projection": "perspective"}}
            }
        });
        upgrade_scene(&mut scene, no_assets()).expect("upgrade");
        assert_eq!(scene["schema_version"], json!(6));
        // Both forms drew a full turn in version 5, so both keep it as a 2π equidistant fisheye.
        for id in ["legacy", "panorama"] {
            let camera = &scene["data_blocks"][id]["camera"];
            assert_eq!(camera["projection"], json!("panorama"), "{id}");
            assert_eq!(
                camera["panorama_type"],
                json!("fisheye_equidistant"),
                "{id}"
            );
            assert_eq!(camera["fisheye_fov"], json!(std::f64::consts::TAU), "{id}");
            assert_eq!(camera["lens_mm"], json!(50.0), "{id}");
        }
        // An equirectangular panorama and a perspective camera gain no fisheye field.
        let sphere = &scene["data_blocks"]["sphere"]["camera"];
        assert_eq!(sphere["panorama_type"], json!("equirectangular"));
        assert!(sphere.get("fisheye_fov").is_none());
        assert_eq!(
            scene["data_blocks"]["plain"]["camera"]["projection"],
            json!("perspective")
        );
    }

    #[test]
    fn current_schema_six_scenes_pass_through_untouched() {
        let mut scene = json!({
            "schema_version": 6,
            "materials": {"m1": {"base_color_texture": {"image": "img", "uv_layer": "Light"}}}
        });
        let before = scene.clone();
        upgrade_scene(&mut scene, no_assets()).expect("upgrade");
        assert_eq!(scene, before);
    }

    /// A project saved at schema 3 labelled its EXR straight, but Blender and the loader read an EXR as premultiplied.
    /// The upgrade relabels it, so it samples as the old loader did. The old loader unpremultiplied the file at load and
    /// premultiplied it again when sampling: unlinked, the file's associated colour comes back; linked, the straight colour.
    #[test]
    fn version_three_exr_images_load_premultiplied_as_the_old_loader_sampled_them() -> TestResult {
        let root = tempfile::tempdir()?;
        // Blender writes an EXR's colour associated: this texel is straight (0.8, 0.4, 0.2) at alpha 0.5.
        let exr_path = root.path().join("texel.exr");
        exr::prelude::write_rgba_file(&exr_path, 1, 1, |_, _| {
            (0.4_f32, 0.2_f32, 0.1_f32, 0.5_f32)
        })?;
        let bytes = std::fs::read(&exr_path)?;
        let digest = crate::hash::sha256(&bytes);
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or("digest is not sha256")?;
        let blob_dir = root.path().join("assets").join("sha256").join(hex);
        std::fs::create_dir_all(&blob_dir)?;
        std::fs::write(blob_dir.join("blob"), &bytes)?;
        let image = Image {
            name: "texel".to_owned(),
            source: crate::model::ImageSource::Packed,
            colorspace: crate::model::ImageColorspace::Linear,
            width: 1,
            height: 1,
            tiles: Vec::new(),
            blob: Some(digest.clone()),
            source_path: None,
            source_hash: Some(digest),
            alpha_mode: ImageAlphaMode::Straight,
            tiled: false,
        };
        let mut scene = serde_json::to_value(SceneDoc::new(uuid::Uuid::new_v4().to_string()))?;
        scene["schema_version"] = json!(3);
        scene["images"] = json!({ "texel": serde_json::to_value(&image)? });
        let parsed = parse_scene(&serde_json::to_vec(&scene)?, root.path())?;
        assert_eq!(parsed.doc.schema_version, 6);
        let id = crate::model::Id::new("texel")?;
        let migrated = &parsed.doc.images[&id];
        assert_eq!(migrated.alpha_mode, ImageAlphaMode::Premultiplied);
        let data = crate::image::load_image_data(
            migrated,
            root.path(),
            crate::image::ImageInterpolation::Closest,
        )?;
        let sample = |linked: bool| {
            crate::image::sample_with_extension_alpha(
                &data,
                [0.5, 0.5],
                false,
                crate::image::ImageInterpolation::Closest,
                crate::image::ImageExtension::Repeat,
                linked,
            )
        };
        for (actual, expected) in sample(false).into_iter().zip([0.4, 0.2, 0.1, 0.5]) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "unlinked {actual} != {expected}"
            );
        }
        for (actual, expected) in sample(true).into_iter().zip([0.8, 0.4, 0.2, 0.5]) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "linked {actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn version_three_color_curve_array_becomes_view_curve() -> TestResult {
        let mut scene = json!({
            "schema_version": 3,
            "scenes": {
                "empty": {"color_management": {"curve": []}},
                "curved": {"color_management": {"curve": [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]]}},
                "hdr": {"color_management": {"curve": [[0.0, 0.0], [2.0, 1.0]]}},
                "single": {"color_management": {"curve": [[0.5, 0.25]]}}
            }
        });
        upgrade_scene(&mut scene, no_assets())?;
        assert_eq!(scene["schema_version"], json!(6));
        assert_eq!(
            scene["scenes"]["empty"]["color_management"]["curve"],
            Value::Null
        );
        let curve: CurveMapping =
            serde_json::from_value(scene["scenes"]["curved"]["color_management"]["curve"].clone())?;
        assert_eq!(curve.curves[3].points.len(), 3);
        assert_eq!(curve.curves[3].points[1].location, [0.5, 0.25]);
        assert_eq!(curve.curves[3].points[1].handle, HandleType::Vector);
        assert_eq!(curve.curves[0].points[0].location, [0.0, 0.0]);
        assert_eq!(curve.curves[0].points[1].location, [1.0, 1.0]);
        assert!(!curve.extend_extrapolate);

        let hdr: CurveMapping =
            serde_json::from_value(scene["scenes"]["hdr"]["color_management"]["curve"].clone())?;
        for channel in &hdr.curves[..3] {
            assert_eq!(channel.points[0].location, [0.0, 0.0]);
            assert_eq!(channel.points[1].location, [2.0, 2.0]);
        }
        assert!((hdr.prepare().evaluate_rgb([1.5; 3])[0] - 0.75).abs() < 1.0e-4);

        let single: CurveMapping =
            serde_json::from_value(scene["scenes"]["single"]["color_management"]["curve"].clone())?;
        assert_eq!(single.curves[3].points.len(), 2);
        assert_eq!(single.curves[3].points[0].location, [0.0, 0.25]);
        assert_eq!(single.curves[3].points[1].location, [1.0, 0.25]);
        assert!((single.prepare().evaluate_rgb([0.5; 3])[0] - 0.25).abs() < 1.0e-4);
        Ok(())
    }

    #[test]
    fn parsing_version_three_scene_migrates_and_validates() -> TestResult {
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        doc.nodes.insert(
            crate::model::Id::new("node")?,
            super::super::Node::default(),
        );
        doc.schema_version = 3;
        let mut expected = serde_json::to_value(&doc)?;
        let parsed = parse_scene(&serde_json::to_vec(&expected)?, no_assets())?;
        parsed.doc.validate()?;
        let migrated = serde_json::to_value(parsed.doc)?;
        expected["schema_version"] = json!(6);
        assert_eq!(migrated, expected);
        Ok(())
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

        let migrated = parse_scene(&serde_json::to_vec(&source)?, no_assets())?.doc;
        assert_eq!(migrated.schema_version, 6);
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

    fn history_record(snapshot: &Value) -> Value {
        json!({
            "record_version": 1,
            "parent": null,
            "kind": "apply",
            "revision": 0,
            "scene_hash": "sha256:scene",
            "operations": [],
            "changes": {},
            "target": null,
            "snapshot": snapshot
        })
    }

    fn migrated_mesh(source: &Value) -> TestResult<crate::geom::Mesh> {
        let migrated = parse_scene(&serde_json::to_vec(source)?, no_assets())?.doc;
        let mesh = migrated.data_blocks[&crate::model::Id::new("meshdata")?]
            .mesh
            .clone()
            .ok_or("migrated data block has no mesh")?;
        crate::geom::color_layers::validate(&mesh)?;
        Ok(mesh)
    }

    #[test]
    fn version_two_user_attribute_with_managed_name_is_preserved() -> TestResult<()> {
        let mesh = triangle_mesh()?;
        let vertex = format!("v{}", mesh.vertices[0].id);
        let source = version_two_document(
            &mesh,
            json!({
                "blender_attributes": [
                    {"name": "sharp_face", "domain": "POINT", "data_type": "FLOAT", "values": [0.25, 0.5, 0.75]},
                    {"name": "sharp_face", "domain": "FACE", "data_type": "BOOLEAN", "values": [true]}
                ]
            }),
        )?;
        let migrated = migrated_mesh(&source)?;
        let storage_key = user_key("sharp_face");
        let user_attribute = &migrated.attributes[&storage_key];

        assert_eq!(user_attribute["domain"], json!("point"));
        assert_eq!(user_attribute["type"], json!("float"));
        assert_eq!(user_attribute["values"][vertex], json!(0.25));
        assert!(!migrated.attributes.contains_key("blender_attributes"));
        Ok(())
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
    fn version_two_legacy_custom_normals_keep_setter_sharp_edges() -> TestResult<()> {
        use glam::DVec3;

        let mesh = crate::geom::Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z],
            vec![vec![0, 1, 2], vec![1, 0, 3]],
        )?;
        let mut smooth = Map::new();
        let mut normals = Map::new();
        for (index, face) in mesh.faces.iter().enumerate() {
            let face_key = format!("f{}", face.id);
            smooth.insert(face_key.clone(), json!(true));
            let normal = if index == 0 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            normals.insert(face_key, json!([normal, normal, normal]));
        }
        let source = version_two_document(
            &mesh,
            json!({
                "shade_smooth": {"domain":"face","type":"bool","values":smooth},
                "custom_normal": {"domain":"corner","type":"float3","values":normals}
            }),
        )?;
        let first = mesh.vertices[0].id;
        let second = mesh.vertices[1].id;
        let shared_edge = mesh
            .edges
            .iter()
            .find(|edge| edge.vertices.contains(&first) && edge.vertices.contains(&second))
            .ok_or("shared edge is missing")?;
        let migrated = migrated_mesh(&source)?;
        let edge_key = format!("e{}", shared_edge.id);
        assert_eq!(
            migrated.attributes["sharp"]["values"][edge_key.as_str()],
            json!(true)
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
            let error = parse_scene(&serde_json::to_vec(&source)?, no_assets())
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
        let (record, history_warnings) =
            parse_history_record(&serde_json::to_vec(&history_record(&source))?, no_assets())?;
        assert!(history_warnings.is_empty(), "{history_warnings:?}");
        assert_eq!(record.snapshot.schema_version, 6);
        let history_mesh = record
            .snapshot
            .data_blocks
            .values()
            .next()
            .and_then(|block| block.mesh.as_ref())
            .ok_or("history snapshot has no mesh")?;
        assert_eq!(&history_mesh.attributes, attributes);
        let current = parse_scene(&serde_json::to_vec(&source)?, no_assets())?.doc;
        let current = parse_scene(&serde_json::to_vec(&current)?, no_assets())?.doc;
        let current_mesh = current
            .data_blocks
            .values()
            .next()
            .and_then(|block| block.mesh.as_ref())
            .ok_or("current scene has no mesh")?;
        assert_eq!(current_mesh.attributes, *attributes);

        let mut selected = migrated.clone();
        crate::geom::edit::apply(
            &mut selected,
            "attribute_update",
            &json!({
                "name": "color_layers",
                "elements": {
                    "domain": "point",
                    "selector": {
                        "type": "attribute",
                        "name": "color_layers",
                        "value": 0.25
                    }
                },
                "value": 0.75
            }),
        )?;
        assert_eq!(
            selected.attributes["user:color_layers"]["values"][vertices[1].as_str()],
            json!(0.75)
        );
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
        assert_eq!(
            crate::geom::color_layers::read(&migrated).layers,
            Vec::new()
        );
        let (record, history_warnings) =
            parse_history_record(&serde_json::to_vec(&history_record(&source))?, no_assets())?;
        assert!(history_warnings.is_empty(), "{history_warnings:?}");
        assert_eq!(record.snapshot.schema_version, 6);
        let history_mesh = record
            .snapshot
            .data_blocks
            .values()
            .next()
            .and_then(|block| block.mesh.as_ref())
            .ok_or("history snapshot has no mesh")?;
        assert_eq!(
            history_mesh.attributes["user:color_layers"]["values"][vertex.as_str()],
            json!(0.25)
        );
        assert!(!history_mesh.attributes.contains_key("color_layers"));
        assert_eq!(
            crate::geom::color_layers::read(history_mesh).layers,
            Vec::new()
        );
        let current = parse_scene(&serde_json::to_vec(&source)?, no_assets())?.doc;
        let current = parse_scene(&serde_json::to_vec(&current)?, no_assets())?.doc;
        let current_mesh = current
            .data_blocks
            .values()
            .next()
            .and_then(|block| block.mesh.as_ref())
            .ok_or("current scene has no mesh")?;
        assert_eq!(
            current_mesh.attributes["user:color_layers"]["values"][vertex.as_str()],
            json!(0.25)
        );
        assert!(!current_mesh.attributes.contains_key("color_layers"));
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

    #[test]
    fn empty_and_repeated_legacy_selectors_become_the_render_reference() {
        let mut scene = json!({
            "schema_version": 1,
            "materials": {"m1": {
                "base_color_texture": {"image": "img", "uv_map": "", "interpolation": "linear"},
                "normal_texture": {"image": "img", "uv_map": "uv_map", "interpolation": "linear"}
            }}
        });
        let warnings = upgrade_scene(&mut scene, no_assets()).expect("upgrade");
        assert!(warnings.is_empty(), "{warnings:?}");
        let material = &scene["materials"]["m1"];
        assert_eq!(material["base_color_texture"]["uv_layer"], Value::Null);
        assert_eq!(material["normal_texture"]["uv_layer"], Value::Null);
    }

    #[test]
    fn non_string_legacy_selector_is_rejected() {
        let mut scene = json!({
            "schema_version": 1,
            "materials": {"m1": {"base_color_texture": {"image": "img", "uv_map": 3}}}
        });
        assert!(upgrade_scene(&mut scene, no_assets()).is_err());
    }

    #[test]
    fn parse_scene_reports_the_legacy_origin_and_skipped_layers() -> TestResult {
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
        let data_id = crate::model::Id::new("meshdata")?;
        doc.data_blocks.insert(
            data_id,
            crate::model::DataBlock {
                mesh: Some(mesh),
                ..crate::model::DataBlock::default()
            },
        );
        let mut source = serde_json::to_value(doc)?;
        source["schema_version"] = json!(1);
        source["data_blocks"]["meshdata"]["mesh"]["attributes"] = json!({
            "blender_uv_layers": [{"name": "Light", "values": [[0.0, 0.25], [0.5, 0.5]]}]
        });

        let parsed = parse_scene(&serde_json::to_vec(&source)?, no_assets())?;
        assert!(parsed.legacy);
        assert_eq!(parsed.warnings.len(), 1, "{:?}", parsed.warnings);
        assert_eq!(parsed.warnings[0]["code"], "uv.legacy_layer_skipped");
        assert_eq!(parsed.warnings[0]["layer"], "Light");
        assert_eq!(parsed.warnings[0]["mesh"], "meshdata");

        let current = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let parsed = parse_scene(
            &serde_json::to_vec(&serde_json::to_value(current)?)?,
            no_assets(),
        )?;
        assert!(!parsed.legacy);
        assert_eq!(parsed.warnings, Vec::<Value>::new());
        Ok(())
    }

    #[test]
    fn history_operation_seed_is_dropped_at_load_and_the_record_replays() -> TestResult {
        let snapshot = serde_json::to_value(SceneDoc::new(uuid::Uuid::new_v4().to_string()))?;
        let mut source = history_record(&snapshot);
        source["operations"] = json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
            {"op":"sculpt.stroke","target":{"id":"body"},"scope":"shared","brush":"draw",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":0.5,"time":0.0}],
             "falloff":"linear","seed":7},
            {"op":"sculpt.dyntopo","target":{"id":"body"},"scope":"shared",
             "samples":[{"position":[0.0,0.0,0.0],"pressure":1.0,"radius":10.0,"strength":1.0,"time":0.0}],
             "falloff":"constant","edge_length":0.5,"seed":7},
            {"op":"paint.vertex","target":{"id":"body"},"scope":"shared","samples":[],"color":[1.0,0.0,0.0,1.0],"seed":7}
        ]);
        let (record, warnings) = parse_history_record(&serde_json::to_vec(&source)?, no_assets())?;
        let operations = record
            .operations
            .as_array()
            .ok_or("history operations are not an array")?;
        assert_eq!(operations[0]["params"]["size"], 2.0);
        assert!(operations[1].get("seed").is_none());
        assert!(operations[2].get("seed").is_none());
        assert!(operations[3].get("seed").is_none());
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert_eq!(warnings[0]["code"], "operation.seed_retired");
        assert_eq!(warnings[0]["operation_index"], 1);
        assert_eq!(warnings[0]["op"], "sculpt.stroke");
        assert_eq!(warnings[0]["seed"], 7);
        assert_eq!(warnings[1]["op"], "sculpt.dyntopo");
        assert_eq!(warnings[2]["code"], "operation.seed_retired");
        assert_eq!(warnings[2]["op"], "paint.vertex");

        let replay = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [operations[0], operations[1]]
        });
        crate::ops::apply_batch(&record.snapshot, &replay)?;

        let mut rejected = operations[1].clone();
        rejected["seed"] = json!(7);
        let rejected_batch = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [operations[0], rejected]
        });
        let error = crate::ops::apply_batch(&record.snapshot, &rejected_batch)
            .err()
            .ok_or("sculpt.stroke accepted a seed")?;
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        Ok(())
    }

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
}
