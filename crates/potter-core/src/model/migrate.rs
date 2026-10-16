// SPDX-License-Identifier: GPL-3.0-or-later

//! Scene document upgrades. [`parse_scene`] is the one way to read a scene document (`scene.json`,
//! a library scene, or a history snapshot). It upgrades schema version 1 on the raw JSON, then
//! deserializes. Version 2 replaced the mesh attributes `uv_map` and `blender_uv_layers` with
//! `uv_layers`, and the texture reference field `uv_map` with `uv_layer`.

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

/// Parses a scene document, upgrading schema version 1 first.
pub fn parse_scene(bytes: &[u8]) -> Result<SceneDoc> {
    let mut value = parse_value(bytes)?;
    upgrade_scene(&mut value)?;
    serde_json::from_value(value).map_err(|error| json_error(&error))
}

/// Parses a history record, upgrading its snapshot's schema version first.
pub(crate) fn parse_history_record(bytes: &[u8]) -> Result<HistoryRecord> {
    let mut value = parse_value(bytes)?;
    if let Some(snapshot) = value.get_mut("snapshot") {
        upgrade_scene(snapshot)?;
    }
    serde_json::from_value(value).map_err(|error| json_error(&error))
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

/// Upgrades a schema-version-1 scene document in place. Other versions are left for
/// `SceneDoc::validate` to accept or reject.
pub fn upgrade_scene(value: &mut Value) -> Result<()> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Ok(());
    }
    if let Some(blocks) = value.get_mut("data_blocks").and_then(Value::as_object_mut) {
        for block in blocks.values_mut() {
            if let Some(mesh) = block.get_mut("mesh").filter(|mesh| !mesh.is_null()) {
                crate::geom::uv_layers::upgrade_mesh_v1(mesh)?;
            }
        }
    }
    if let Some(materials) = value.get_mut("materials").and_then(Value::as_object_mut) {
        for material in materials.values_mut() {
            for key in MATERIAL_TEXTURES {
                if let Some(texture) = material.get_mut(key) {
                    upgrade_texture_ref(texture);
                }
            }
        }
    }
    if let Some(mtoon) = value
        .pointer_mut("/avatar/mtoon")
        .and_then(Value::as_object_mut)
    {
        for material in mtoon.values_mut() {
            for key in MTOON_TEXTURES {
                if let Some(texture) = material.get_mut(key) {
                    upgrade_texture_ref(texture);
                }
            }
            if let Some(texture) = material.pointer_mut("/shading_shift_texture/texture") {
                upgrade_texture_ref(texture);
            }
        }
    }
    value["schema_version"] = json!(2);
    Ok(())
}

/// `uv_map: "uv_map"` named the single version-1 UV set, which is the render layer, so it becomes
/// `uv_layer: null`. Any other name stays as a layer reference.
fn upgrade_texture_ref(texture: &mut Value) {
    let Some(object) = texture.as_object_mut() else {
        return;
    };
    let Some(uv_map) = object.remove("uv_map") else {
        return;
    };
    let uv_layer = match uv_map {
        Value::String(name) if name != "uv_map" => Value::String(name),
        _ => Value::Null,
    };
    object.insert("uv_layer".to_owned(), uv_layer);
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

        let migrated = parse_scene(&serde_json::to_vec(&source)?)?;
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

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
}
