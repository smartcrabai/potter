//! VRM 1.0 `MToon` texture transforms (`AvatarMtoon::texture_transforms`).
//!
//! - Round trip: a VRM 1.0 export carries `KHR_texture_transform` on the `MToon` texture info, and
//!   re-importing it without `--allow-lossy` restores the transform in the scene avatar.
//! - VRM 0.x export of the same avatar is refused with `vrm.v0.mtoon_texture_transform`.
//! - A VRM 1.0 file that carries `KHR_texture_transform` on `shadeMultiplyTexture` imports without a loss.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn cli_json(output: &Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init_project() -> TestResult<(TempDir, PathBuf)> {
    let directory = tempfile::tempdir()?;
    let project = directory.path().join("avatar");
    let output = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(
        output.status.success(),
        "pot init failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((directory, project))
}

fn apply_batch(project: &Path, base_revision: u64, operations: &[Value]) -> TestResult<Output> {
    let file = project.join("vrm-operations.json");
    fs::write(
        &file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations,
        }))?,
    )?;
    Ok(pot()
        .arg("apply")
        .arg(project)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()?)
}

fn scene_avatar(project: &Path) -> TestResult<Value> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    Ok(document["avatar"].clone())
}

fn export(project: &Path, format: &str, output_path: &Path) -> TestResult {
    let output = pot()
        .arg("export")
        .arg(project)
        .args(["--format", format, "--out"])
        .arg(output_path)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "`{format}` export failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn import_file(
    project: &Path,
    file: &Path,
    format: &str,
    base_revision: u64,
    allow_lossy: bool,
) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(file)
        .args(["--format", format, "--base-revision"])
        .arg(base_revision.to_string());
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

/// The glTF JSON chunk of a GLB file.
fn glb_json(path: &Path) -> TestResult<Value> {
    let bytes = fs::read(path)?;
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err(format!("{} is not a GLB file", path.display()).into());
    }
    let chunk_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let json = bytes
        .get(20..20 + chunk_length)
        .ok_or("GLB JSON chunk is truncated")?;
    Ok(serde_json::from_slice(json)?)
}

/// Rewrites the glTF JSON of a GLB file in place and keeps its binary chunk.
fn edit_glb(path: &Path, edit: impl FnOnce(&mut Value) -> TestResult) -> TestResult {
    let bytes = fs::read(path)?;
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let binary = &bytes[20 + json_length..];
    let mut document: Value = serde_json::from_slice(&bytes[20..20 + json_length])?;
    edit(&mut document)?;
    let mut json = serde_json::to_vec(&document)?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + binary.len();
    let mut output = Vec::with_capacity(total);
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(&u32::try_from(total)?.to_le_bytes());
    output.extend_from_slice(&u32::try_from(json.len())?.to_le_bytes());
    output.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    output.extend_from_slice(&json);
    output.extend_from_slice(binary);
    fs::write(path, output)?;
    Ok(())
}

/// The `VRMC_materials_mtoon` object of the first VRM 1.0 `MToon` material of a glTF document.
fn vrm1_mtoon(document: &Value) -> TestResult<Value> {
    document["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|material| material["extensions"]["VRMC_materials_mtoon"].clone())
        .find(Value::is_object)
        .ok_or_else(|| "the glTF has no VRMC_materials_mtoon material".into())
}

fn uses_extension(document: &Value, name: &str) -> bool {
    document["extensionsUsed"]
        .as_array()
        .is_some_and(|used| used.iter().any(|value| value == name))
}

// ---------------------------------------------------------------------------------------------
// Potter fixture. Copied from tests/vrm_import.rs with its round-trip options.
// ---------------------------------------------------------------------------------------------

fn image_operations() -> Vec<Value> {
    [
        ("tex_albedo", "Albedo", [0.8, 0.4, 0.2, 1.0]),
        ("tex_shade", "Shade", [0.2, 0.3, 0.4, 1.0]),
        ("tex_outline", "Outline", [0.6, 0.7, 0.8, 1.0]),
        ("tex_rim", "Rim", [0.5, 0.6, 0.7, 1.0]),
        ("tex_uv_mask", "UV Mask", [0.7, 0.8, 0.9, 1.0]),
        ("thumbnail", "Avatar Thumbnail", [0.9, 0.8, 0.7, 1.0]),
    ]
    .into_iter()
    .map(|(id, name, fill_color)| {
        json!({
            "op": "image.create",
            "id": id,
            "name": name,
            "width": 1,
            "height": 1,
            "fill_color": fill_color,
        })
    })
    .collect()
}

fn humanoid() -> Value {
    json!({
        "hips": "hips",
        "spine": "spine",
        "chest": "spine_001",
        "upperChest": "spine_002",
        "neck": "neck",
        "head": "head",
        "leftEye": "eye_l",
        "rightEye": "eye_r",
        "leftShoulder": "shoulder_l",
        "rightShoulder": "shoulder_r",
        "leftUpperArm": "upper_arm_l",
        "leftLowerArm": "forearm_l",
        "leftHand": "hand_l",
        "rightUpperArm": "upper_arm_r",
        "rightLowerArm": "forearm_r",
        "rightHand": "hand_r",
        "leftUpperLeg": "thigh_l",
        "leftLowerLeg": "shin_l",
        "leftFoot": "foot_l",
        "leftToes": "toe_l",
        "rightUpperLeg": "thigh_r",
        "rightLowerLeg": "shin_r",
        "rightFoot": "foot_r",
        "rightToes": "toe_r",
        "leftThumbMetacarpal": "thumb_meta_l",
        "leftThumbProximal": "thumb_prox_l",
        "leftThumbDistal": "thumb_distal_l",
        "rightThumbMetacarpal": "thumb_meta_r",
        "rightThumbProximal": "thumb_prox_r",
        "rightThumbDistal": "thumb_distal_r",
    })
}

fn fixture_meta() -> Value {
    json!({
        "name": "Native VRM Avatar",
        "authors": ["Potter VRM Test"],
        "license_url": "https://vrm.dev/licenses/1.0/",
        "thumbnail": "thumbnail",
        "avatar_permission": "only_author",
        "allow_excessively_violent_usage": false,
        "allow_excessively_sexual_usage": false,
        "commercial_usage": "personal_non_profit",
        "allow_political_or_religious_usage": false,
        "allow_antisocial_or_hate_usage": false,
        "credit_notation": "required",
        "allow_redistribution": false,
        "modification": "prohibited",
        "legacy": {
            "title": "Legacy Fixture Avatar",
            "author": "Potter VRM Test",
            "reference": "https://example.test/reference",
            "allowed_user": "only_author",
            "violent_usage": "disallow",
            "sexual_usage": "disallow",
            "commercial_usage": "disallow",
            "license_name": "redistribution_prohibited",
            "other_permission_url": "https://example.test/permissions",
        },
    })
}

fn fixture_spring_bone() -> Value {
    json!({
        "colliders": [{
            "node": {"node": "arm", "bone": "hair_root"},
            "shape": {"type": "sphere", "offset": [0.1, 0.2, 0.3], "radius": 0.08},
        }],
        "collider_groups": [{"name": "Hair", "colliders": [0]}],
        "springs": [{
            "name": "Hair",
            "center": {"node": "arm", "bone": "head"},
            "joints": [
                {
                    "node": {"node": "arm", "bone": "hair_root"},
                    "hit_radius": 0.02,
                    "stiffness": 0.8,
                    "gravity_power": 1.1,
                    "gravity_dir": [1.0, 2.0, 3.0],
                    "drag_force": 0.2,
                },
                {
                    "node": {"node": "arm", "bone": "hair_tip"},
                    "hit_radius": 0.02,
                    "stiffness": 0.8,
                    "gravity_power": 1.1,
                    "gravity_dir": [1.0, 2.0, 3.0],
                    "drag_force": 0.2,
                },
            ],
            "collider_groups": [0],
        }],
    })
}

fn fixture_mtoon() -> Value {
    json!({
        "transparent_with_z_write": false,
        "render_queue_offset_number": 0,
        "shade_color_factor": [0.3, 0.35, 0.4],
        "shade_multiply_texture": {"image": "tex_shade"},
        "shading_shift_factor": -0.15,
        "shading_toony_factor": 0.7,
        "gi_equalization_factor": 0.2,
        "matcap_factor": [1.0, 1.0, 1.0],
        "matcap_texture": {"image": "tex_outline"},
        "parametric_rim_color_factor": [0.1, 0.2, 0.3],
        "rim_multiply_texture": {"image": "tex_rim"},
        "rim_lighting_mix_factor": 0.6,
        "parametric_rim_fresnel_power_factor": 3.0,
        "parametric_rim_lift_factor": 0.1,
        "outline_width_mode": "world_coordinates",
        "outline_width_factor": 0.01,
        "outline_width_multiply_texture": {"image": "tex_outline"},
        "outline_color_factor": [0.15, 0.2, 0.25],
        "outline_lighting_mix_factor": 0.5,
        "uv_animation_mask_texture": {"image": "tex_uv_mask"},
        "uv_animation_scroll_x_speed_factor": 0.2,
        "uv_animation_scroll_y_speed_factor": -0.1,
        "uv_animation_rotation_speed_factor": 0.0,
    })
}

fn fixture_avatar() -> Value {
    let mut expressions = json!({
        "happy": {
            "is_binary": false,
            "override_blink": "none",
            "override_mouth": "none",
            "override_look_at": "none",
            "morph_target_binds": [{"node": "body", "shape_key": "smile", "weight": 0.4}],
            "material_color_binds": [{
                "material": "skin_material",
                "color_type": "color",
                "target_value": [0.2, 0.4, 0.6, 0.8],
            }],
        },
        "winkExtra": {
            "is_binary": true,
            "override_blink": "none",
            "override_mouth": "none",
            "override_look_at": "none",
            "morph_target_binds": [{"node": "body", "shape_key": "smile", "weight": 0.3}],
        },
    });
    for (preset, shape_key) in [
        ("lookLeft", "look_left"),
        ("lookRight", "look_right"),
        ("lookUp", "look_up"),
        ("lookDown", "look_down"),
    ] {
        expressions[preset] = json!({
            "is_binary": false,
            "morph_target_binds": [{"node": "body", "shape_key": shape_key, "weight": 0.65}],
        });
    }
    json!({
        "armature": "arm",
        "humanoid": humanoid(),
        "meta": fixture_meta(),
        "expressions": expressions,
        "first_person": {
            "mesh_annotations": [{"node": "body", "annotation_type": "first_person_only"}],
        },
        "look_at": {
            "look_at_type": "bone",
            "offset_from_head_bone": [0.1, 0.2, 0.3],
            "range_map_horizontal_inner": {"input_max_value": 30.0, "output_scale": 0.25},
            "range_map_horizontal_outer": {"input_max_value": 40.0, "output_scale": 0.5},
            "range_map_vertical_down": {"input_max_value": 50.0, "output_scale": 0.75},
            "range_map_vertical_up": {"input_max_value": 60.0, "output_scale": 1.0},
        },
        "spring_bone": fixture_spring_bone(),
        "mtoon": {"skin_material": fixture_mtoon()},
    })
}

fn fixture_operations() -> Vec<Value> {
    let mut operations = vec![
        // Shifts node indices away from mesh indices so a node/mesh index mix-up is detectable.
        json!({"op": "node.create", "id": "anchor", "kind": "empty"}),
        json!({"op": "node.create", "id": "arm", "kind": "armature"}),
        json!({"op": "rig.generate_basic_human", "target": {"id": "arm"}, "scale": 1.0}),
    ];
    operations.extend(image_operations());
    operations.extend([
        json!({
            "op": "material.create",
            "id": "skin_material",
            "name": "Avatar Skin",
            // Not the glTF default of 1.5: the MToon IOR fallback must restore it from extras.potter.ior.
            "base_color": [0.8, 0.65, 0.5, 1.0],
            "ior": 1.45,
            "base_color_texture": {"image": "tex_albedo", "uv_layer": null},
        }),
        // Non-uniformly scaled and skinned with shape keys: the exporter bakes the scale into the
        // geometry and morph NORMAL deltas must survive the bake.
        json!({"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1.0}, "material": "skin_material", "transform": {"scale": [1.0, 2.0, 0.5]}}),
        json!({"op": "uv.unwrap", "target": {"id": "body"}, "elements": {"domain": "face", "ids": ["f1"]}, "method": "smart"}),
        json!({"op": "vertex_group.create", "target": {"id": "body"}, "id": "hips", "name": "hips"}),
        json!({
            "op": "vertex_group.assign",
            "target": {"id": "body"},
            "group_id": "hips",
            "weights": (0..8).map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0})).collect::<Vec<_>>(),
        }),
        json!({"op": "modifier.create", "target": {"id": "body"}, "id": "skin_binding", "type": "armature", "params": {"object": "arm", "use_vertex_groups": true}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "smile", "name": "Smile", "positions": {"0": [2.5, 3.5, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_left", "name": "Look Left", "positions": {"1": [2.6, 3.5, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_right", "name": "Look Right", "positions": {"2": [2.5, 3.6, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_up", "name": "Look Up", "positions": {"3": [2.5, 3.5, 4.6]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_down", "name": "Look Down", "positions": {"4": [2.5, 3.5, 4.4]}}),
        json!({"op": "shape_key.update", "target": {"id": "body"}, "id": "smile", "set": {"value": 0.0}}),
    ]);
    operations.extend([
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_meta_l", "name": "thumb_metacarpal.L", "head": [0.2, 0.0, 1.5], "tail": [0.3, 0.0, 1.52], "parent": "hand_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_prox_l", "name": "thumb_proximal.L", "head": [0.3, 0.0, 1.52], "tail": [0.4, 0.0, 1.54], "parent": "thumb_meta_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_distal_l", "name": "thumb_distal.L", "head": [0.4, 0.0, 1.54], "tail": [0.5, 0.0, 1.56], "parent": "thumb_prox_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_meta_r", "name": "thumb_metacarpal.R", "head": [-0.2, 0.0, 1.5], "tail": [-0.3, 0.0, 1.52], "parent": "hand_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_prox_r", "name": "thumb_proximal.R", "head": [-0.3, 0.0, 1.52], "tail": [-0.4, 0.0, 1.54], "parent": "thumb_meta_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_distal_r", "name": "thumb_distal.R", "head": [-0.4, 0.0, 1.54], "tail": [-0.5, 0.0, 1.56], "parent": "thumb_prox_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "eye_l", "name": "Eye.L", "head": [0.04, 0.0, 1.78], "tail": [0.04, 0.03, 1.78], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "eye_r", "name": "Eye.R", "head": [-0.04, 0.0, 1.78], "tail": [-0.04, 0.03, 1.78], "parent": "head"}),
        // Tilted, like the harness hair chain: an axis-aligned rest frame has no f32 rounding in its axes.
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_root", "name": "Hair Root", "head": [0.0, 0.0, 1.9], "tail": [0.03, 0.02, 2.05], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_tip", "name": "Hair Tip", "head": [0.0, 0.0, 2.05], "tail": [0.0, 0.0, 2.2], "parent": "hair_root"}),
        json!({"op": "avatar.update", "set": fixture_avatar()}),
    ]);
    operations
}

/// The fixture with a non-identity texture transform on the shade multiply texture.
fn fixture_operations_with_texture_transform() -> Vec<Value> {
    let mut operations = fixture_operations();
    for operation in &mut operations {
        if operation["op"] == "avatar.update" {
            operation["set"]["mtoon"]["skin_material"]["texture_transforms"] = json!({
                "shadeMultiplyTexture": {"offset": [0.25, -0.5], "rotation": 0.5, "scale": [2.0, 1.5]},
            });
        }
    }
    operations
}

fn new_fixture(operations: &[Value]) -> TestResult<(TempDir, PathBuf)> {
    let (directory, project) = init_project()?;
    let output = apply_batch(&project, 0, operations)?;
    assert!(
        output.status.success(),
        "VRM fixture setup failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((directory, project))
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn mtoon_texture_transform_round_trips_through_vrm1_without_loss() -> TestResult {
    let (directory, source) = new_fixture(&fixture_operations_with_texture_transform())?;
    let exported = directory.path().join("transform-v1.vrm");
    export(&source, "vrm", &exported)?;

    let document = glb_json(&exported)?;
    let expected = json!({"offset": [0.25, -0.5], "rotation": 0.5, "scale": [2.0, 1.5]});
    assert_eq!(
        vrm1_mtoon(&document)?["shadeMultiplyTexture"]["extensions"]["KHR_texture_transform"],
        expected
    );
    assert!(uses_extension(&document, "KHR_texture_transform"));

    let (_fresh_directory, fresh) = init_project()?;
    // No --allow-lossy: the transform is stored, so a Potter export must import without losses.
    let output = import_file(&fresh, &exported, "vrm", 0, false)?;
    assert!(
        output.status.success(),
        "VRM 1.0 round-trip import failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        scene_avatar(&fresh)?["mtoon"]["skin_material"]["texture_transforms"],
        json!({"shadeMultiplyTexture": {"offset": [0.25, -0.5], "rotation": 0.5, "scale": [2.0, 1.5]}})
    );

    let reexported = directory.path().join("transform-reexport.vrm");
    export(&fresh, "vrm", &reexported)?;
    assert_eq!(
        vrm1_mtoon(&glb_json(&reexported)?)?["shadeMultiplyTexture"]["extensions"],
        vrm1_mtoon(&document)?["shadeMultiplyTexture"]["extensions"]
    );
    Ok(())
}

#[test]
fn mtoon_texture_transform_is_refused_by_vrm0_export() -> TestResult {
    let (directory, source) = new_fixture(&fixture_operations_with_texture_transform())?;
    let output = pot()
        .arg("export")
        .arg(&source)
        .args(["--format", "vrm0", "--out"])
        .arg(directory.path().join("transform-v0.vrm"))
        .arg("--json")
        .output()?;
    assert!(
        !output.status.success(),
        "VRM 0.x export unexpectedly succeeded: stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let envelope = cli_json(&output)?;
    assert_eq!(envelope["ok"], false);
    assert_eq!(envelope["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        envelope["error"]["details"]["feature_id"],
        "vrm.v0.mtoon_texture_transform"
    );
    Ok(())
}

#[test]
fn vrm1_khr_texture_transform_on_a_mtoon_slot_imports_without_loss() -> TestResult {
    let (directory, source) = new_fixture(&fixture_operations())?;
    let exported = directory.path().join("edited-v1.vrm");
    export(&source, "vrm", &exported)?;
    edit_glb(&exported, |document| {
        for material in document["materials"].as_array_mut().into_iter().flatten() {
            if material["extensions"]["VRMC_materials_mtoon"].is_object() {
                material["extensions"]["VRMC_materials_mtoon"]["shadeMultiplyTexture"]["extensions"] = json!({
                    "KHR_texture_transform": {"offset": [0.1, 0.2], "scale": [1.0, 2.0]},
                });
            }
        }
        document["extensionsUsed"]
            .as_array_mut()
            .ok_or("the glTF has no extensionsUsed")?
            .push(json!("KHR_texture_transform"));
        Ok(())
    })?;

    let (_fresh_directory, fresh) = init_project()?;
    let output = import_file(&fresh, &exported, "vrm", 0, false)?;
    assert!(
        output.status.success(),
        "edited VRM 1.0 import reported a loss: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        scene_avatar(&fresh)?["mtoon"]["skin_material"]["texture_transforms"],
        json!({"shadeMultiplyTexture": {"offset": [0.1, 0.2], "rotation": 0.0, "scale": [1.0, 2.0]}})
    );
    Ok(())
}
