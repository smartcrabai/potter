// SPDX-License-Identifier: GPL-3.0-or-later

//! Scene document upgrades. [`parse_scene`] is the one way to read a scene document (`scene.json`,
//! a library scene, or a history snapshot). It upgrades schema version 1 on the raw JSON, then
//! deserializes. Version 2 replaced the mesh attributes `uv_map` and `blender_uv_layers` with
//! `uv_layers`, and the texture reference field `uv_map` with `uv_layer`.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::SceneDoc;
use crate::{
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

/// A scene document read from bytes, with what its upgrade produced.
pub struct ParsedScene {
    pub doc: SceneDoc,
    /// Diagnostics for version-1 data the upgrade skipped or dropped. Empty for version 2.
    pub warnings: Vec<Value>,
    /// Whether the bytes were schema version 1. Their original bytes must be kept before the first
    /// version-2 write replaces them.
    pub legacy: bool,
}

/// Parses a scene document, upgrading schema version 1 first.
pub fn parse_scene(bytes: &[u8]) -> Result<ParsedScene> {
    let mut value = parse_value(bytes)?;
    let legacy = value.get("schema_version").and_then(Value::as_u64) == Some(1);
    let warnings = upgrade_scene(&mut value)?;
    let doc: SceneDoc = serde_json::from_value(value).map_err(|error| json_error(&error))?;
    Ok(ParsedScene {
        doc,
        warnings,
        legacy,
    })
}

/// Parses a history record, upgrading its snapshot's schema version first. The warnings describe
/// the snapshot.
pub(crate) fn parse_history_record(bytes: &[u8]) -> Result<(HistoryRecord, Vec<Value>)> {
    let mut value = parse_value(bytes)?;
    let warnings = match value.get_mut("snapshot") {
        Some(snapshot) => upgrade_scene(snapshot)?,
        None => Vec::new(),
    };
    let record: HistoryRecord =
        serde_json::from_value(value).map_err(|error| json_error(&error))?;
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

/// Upgrades a schema-version-1 scene document in place and returns the diagnostics for the legacy
/// data it skipped or dropped. Other versions are left for `SceneDoc::validate` to accept or reject.
pub fn upgrade_scene(value: &mut Value) -> Result<Vec<Value>> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Ok(Vec::new());
    }
    // Layers that a texture names are kept ahead of unnamed ones when the layer limit applies.
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

/// Every texture reference of a material and of an `MToon` avatar material, for mutation.
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

/// Adds the fields of one material whose key is in `keys`.
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

/// Adds the texture fields of one `MToon` material and the texture of its shading-shift field.
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

/// Converts one version-1 texture reference. `uv_map` named the single version-1 UV set, which is
/// the render layer, so `null`, `""`, and `"uv_map"` all become `uv_layer: null`. Any other name
/// stays a layer reference. A non-string value was never valid version-1 data, so it is rejected.
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

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "scene migration unit fixtures")]
    use serde_json::json;

    use super::*;

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
        assert_eq!(scene["schema_version"], json!(2));
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
            "schema_version": 2,
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
        assert_eq!(migrated.schema_version, 2);
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

    #[test]
    fn empty_and_repeated_legacy_selectors_become_the_render_reference() {
        let mut scene = json!({
            "schema_version": 1,
            "materials": {"m1": {
                "base_color_texture": {"image": "img", "uv_map": "", "interpolation": "linear"},
                "normal_texture": {"image": "img", "uv_map": "uv_map", "interpolation": "linear"}
            }}
        });
        let warnings = upgrade_scene(&mut scene).expect("upgrade");
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
        assert!(upgrade_scene(&mut scene).is_err());
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

        let parsed = parse_scene(&serde_json::to_vec(&source)?)?;
        assert!(parsed.legacy);
        assert_eq!(parsed.warnings.len(), 1, "{:?}", parsed.warnings);
        assert_eq!(parsed.warnings[0]["code"], "uv.legacy_layer_skipped");
        assert_eq!(parsed.warnings[0]["layer"], "Light");
        assert_eq!(parsed.warnings[0]["mesh"], "meshdata");

        let current = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let parsed = parse_scene(&serde_json::to_vec(&serde_json::to_value(current)?)?)?;
        assert!(!parsed.legacy);
        assert_eq!(parsed.warnings, Vec::<Value>::new());
        Ok(())
    }

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
}
