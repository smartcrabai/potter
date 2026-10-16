//! `pot import --format vrm|vrm0` tests.
//!
//! - Blender-gated: VRM 1.0 and VRM 0.x files written by the pinned VRM Add-on for Blender 4.10.2 through its own
//!   exporter (`tests/vrm-import/make_third_party.py`), imported with `--allow-lossy`. The expected values are the ones
//!   the generator writes, after the add-on's own unit conversions (see the comments on each assertion).
//! - Round trip (not Blender-gated): a Potter fixture exported to VRM and re-imported must reproduce the avatar and
//!   re-export the same glTF JSON.
//! - Format and container errors: a VRM 0.x file imported as `vrm`, a VRM 1.0 file imported as `vrm0`, and a glTF
//!   without a VRM extension are rejected with `IMPORT_FAILED`.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::TempDir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;
#[path = "common/vrm_import_blender.rs"]
mod vrm_import_blender;

use vrm_import_blender::Gate;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Float fields pass through f32 glTF frames, so imported values agree with the source to about 1e-7 relative.
const TOLERANCE: f64 = 1e-6;
/// The first export carries the source rig's f32 IK solve (Blender parity), up to about 5e-6 in node frames;
/// the re-imported avatar has no IK and re-exports the exact rest pose. Structure stays exact.
const RE_EXPORT_TOLERANCE: f64 = 1e-5;

/// VRM role to Blender bone name (the Potter bone id) for the generated third-party avatar.
const GENERATED_HUMANOID: [(&str, &str); 17] = [
    ("hips", "hips"),
    ("spine", "spine"),
    ("chest", "chest"),
    ("neck", "neck"),
    ("head", "head"),
    ("leftUpperArm", "left_upper_arm"),
    ("leftLowerArm", "left_lower_arm"),
    ("leftHand", "left_hand"),
    ("rightUpperArm", "right_upper_arm"),
    ("rightLowerArm", "right_lower_arm"),
    ("rightHand", "right_hand"),
    ("leftUpperLeg", "left_upper_leg"),
    ("leftLowerLeg", "left_lower_leg"),
    ("leftFoot", "left_foot"),
    ("rightUpperLeg", "right_upper_leg"),
    ("rightLowerLeg", "right_lower_leg"),
    ("rightFoot", "right_foot"),
];

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

/// Imports `file` and returns the sorted `feature_id`s of the losses it reported.
fn import_losses(project: &Path, file: &Path, format: &str) -> TestResult<Vec<String>> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let revision = scene["revision"].as_u64().ok_or("scene has no revision")?;
    let output = import_file(project, file, format, revision, true)?;
    assert!(
        output.status.success(),
        "pot import --format {format} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope = cli_json(&output)?;
    assert_eq!(envelope["ok"], true);
    let mut feature_ids = envelope["result"]["losses"]
        .as_array()
        .ok_or("import result has no losses array")?
        .iter()
        .map(|loss| {
            loss["feature_id"]
                .as_str()
                .map(str::to_owned)
                .ok_or("loss has no feature_id")
        })
        .collect::<Result<Vec<_>, _>>()?;
    feature_ids.sort();
    Ok(feature_ids)
}

/// Asserts a failed import with `IMPORT_FAILED`, a message containing `message`, and the extension name in details.
fn assert_import_failed(output: &Output, message: &str, extension: &str) -> TestResult {
    assert!(
        !output.status.success(),
        "pot import unexpectedly succeeded: stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let envelope = cli_json(output)?;
    assert_eq!(envelope["ok"], false);
    assert_eq!(envelope["error"]["code"], "IMPORT_FAILED");
    let actual = envelope["error"]["message"].as_str().unwrap_or_default();
    assert!(
        actual.contains(message),
        "error message `{actual}` does not contain `{message}`"
    );
    assert_eq!(envelope["error"]["details"]["extension"], extension);
    Ok(())
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

fn without_asset(mut document: Value) -> Value {
    if let Some(object) = document.as_object_mut() {
        object.remove("asset");
    }
    document
}

/// Removes every `scene_hash`: it fingerprints the whole scene document, which the f32 frames change.
fn without_scene_hash(mut document: Value) -> Value {
    fn strip(value: &mut Value) {
        match value {
            Value::Object(object) => {
                object.remove("scene_hash");
                object.values_mut().for_each(strip);
            }
            Value::Array(items) => items.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut document);
    document
}

/// Removes `meta.legacy`: VRM 1.0 has no legacy fields, so the VRM 1.0 round trip cannot carry them.
fn without_legacy(mut avatar: Value) -> Value {
    if let Some(meta) = avatar.get_mut("meta").and_then(Value::as_object_mut) {
        meta.remove("legacy");
    }
    avatar
}

fn number(value: &Value, what: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("{what} is not a number: {value}"))
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual} != {expected}"
    );
}

fn vec3(value: &Value, what: &str) -> [f64; 3] {
    let array = value
        .as_array()
        .unwrap_or_else(|| panic!("{what} is not an array: {value}"));
    assert_eq!(array.len(), 3, "{what} has three components");
    [
        number(&array[0], what),
        number(&array[1], what),
        number(&array[2], what),
    ]
}

fn assert_vec3_close(value: &Value, expected: [f64; 3], what: &str) {
    for (index, (actual, expected)) in vec3(value, what).into_iter().zip(expected).enumerate() {
        assert_close(actual, expected, TOLERANCE, &format!("{what}[{index}]"));
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

/// Structural JSON equality with a float tolerance. Object key sets and array lengths must match exactly.
fn assert_json_close(path: &str, actual: &Value, expected: &Value, tolerance: f64) {
    match (actual, expected) {
        (Value::Number(_), Value::Number(_)) => assert_close(
            number(actual, path),
            number(expected, path),
            tolerance,
            path,
        ),
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{path}: array length");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_json_close(&format!("{path}/{index}"), actual, expected, tolerance);
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "{path}: object keys"
            );
            for (key, actual) in actual {
                assert_json_close(
                    &format!("{path}/{key}"),
                    actual,
                    &expected[key.as_str()],
                    tolerance,
                );
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Potter fixture for the round trip. Copied from tests/vrm.rs with its default options.
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
            "base_color_texture": {"image": "tex_albedo", "uv_map": "uv_map"},
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

fn new_fixture() -> TestResult<(TempDir, PathBuf)> {
    let (directory, project) = init_project()?;
    let output = apply_batch(&project, 0, &fixture_operations())?;
    assert!(
        output.status.success(),
        "VRM fixture setup failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((directory, project))
}

// ---------------------------------------------------------------------------------------------
// Round trip (not Blender-gated)
// ---------------------------------------------------------------------------------------------

fn round_trip(format: &str) -> TestResult {
    let (directory, source) = new_fixture()?;
    let exported = directory.path().join(format!("source-{format}.vrm"));
    export(&source, format, &exported)?;
    // VRM 0.x cannot hold every meta field, so only it keeps the typed copy; VRM 1.0 decodes its own meta.
    let has_saved_meta = !glb_json(&exported)?["extras"]["potter"]["avatar_meta"].is_null();
    assert_eq!(has_saved_meta, format == "vrm0");

    let (_fresh_directory, fresh) = init_project()?;
    // No --allow-lossy: a Potter export must import without losses.
    let output = import_file(&fresh, &exported, format, 0, false)?;
    assert!(
        output.status.success(),
        "`{format}` round-trip import failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_json_close(
        "avatar",
        &without_legacy(scene_avatar(&fresh)?),
        &without_legacy(scene_avatar(&source)?),
        TOLERANCE,
    );
    // The fixture has one rig: no skinned mesh may leave an unused armature Data-Block behind.
    let scene: Value = serde_json::from_slice(&fs::read(fresh.join("scene.json"))?)?;
    let armature_blocks = scene["data_blocks"].as_object().map_or(0, |blocks| {
        blocks
            .values()
            .filter(|block| block["type"] == "armature")
            .count()
    });
    assert_eq!(
        armature_blocks, 1,
        "the rig must import as one armature Data-Block"
    );

    let reexported = directory.path().join(format!("reexport-{format}.vrm"));
    export(&fresh, format, &reexported)?;
    assert_json_close(
        "re-export",
        &without_scene_hash(without_asset(glb_json(&reexported)?)),
        &without_scene_hash(without_asset(glb_json(&exported)?)),
        RE_EXPORT_TOLERANCE,
    );
    Ok(())
}

#[test]
fn potter_vrm1_export_reimports_to_the_same_avatar_and_reexports_identically() -> TestResult {
    round_trip("vrm")
}

#[test]
fn potter_vrm0_export_reimports_to_the_same_avatar_and_reexports_identically() -> TestResult {
    round_trip("vrm0")
}

#[test]
fn vrm_versions_preserve_bone_parented_accessory_transforms() -> TestResult {
    let (directory, source) = new_fixture()?;
    let source_scene: Value = serde_json::from_slice(&fs::read(source.join("scene.json"))?)?;
    for (format, filename) in [("vrm", "v1.vrm"), ("vrm0", "v0.vrm")] {
        let file = directory.path().join(filename);
        export(&source, format, &file)?;
        let (_fresh_directory, imported) = init_project()?;
        assert_eq!(import_losses(&imported, &file, format)?, loss_ids(&[]));
        let imported_scene: Value =
            serde_json::from_slice(&fs::read(imported.join("scene.json"))?)?;
        for id in ["hat", "ribbon", "ribbon_mirror"] {
            let original = &source_scene["nodes"][id];
            let reimported = &imported_scene["nodes"][id];
            assert_eq!(reimported["parent"], original["parent"], "{format} {id}");
            assert_eq!(
                reimported["parent_type"], original["parent_type"],
                "{format} {id}"
            );
            assert_eq!(
                reimported["parent_bone"], original["parent_bone"],
                "{format} {id}"
            );
            assert_json_close(
                &format!("{format}/{id}/transform"),
                &reimported["transform"],
                &original["transform"],
                TOLERANCE,
            );
        }
    }
    Ok(())
}

#[test]
fn vrm_import_rejects_the_other_version_and_a_missing_vrm_extension() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm1 = directory.path().join("v1.vrm");
    let vrm0 = directory.path().join("v0.vrm");
    export(&source, "vrm", &vrm1)?;
    export(&source, "vrm0", &vrm0)?;
    // A glTF export of the avatar project is refused (glb cannot keep avatar settings), so use a plain scene.
    let (_plain_directory, plain_source) = init_project()?;
    let output = apply_batch(
        &plain_source,
        0,
        &[json!({"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 1.0}})],
    )?;
    assert!(output.status.success(), "plain scene setup failed");
    let glb = directory.path().join("plain.glb");
    export(&plain_source, "glb", &glb)?;
    // The VRM importer checks the file extension first, so a glTF needs a `.vrm` name to reach the extension check.
    let plain = directory.path().join("plain-as-vrm.vrm");
    fs::copy(&glb, &plain)?;

    let (_fresh_directory, fresh) = init_project()?;
    let output = import_file(&fresh, &vrm1, "vrm0", 0, true)?;
    assert_import_failed(&output, "glTF file has no VRM extension", "VRM")?;
    let output = import_file(&fresh, &vrm0, "vrm", 0, true)?;
    assert_import_failed(&output, "glTF file has no VRMC_vrm extension", "VRMC_vrm")?;
    let output = import_file(&fresh, &plain, "vrm", 0, true)?;
    assert_import_failed(&output, "glTF file has no VRMC_vrm extension", "VRMC_vrm")?;
    let output = import_file(&fresh, &plain, "vrm0", 0, true)?;
    assert_import_failed(&output, "glTF file has no VRM extension", "VRM")?;
    // Rejected imports leave the project untouched.
    assert_eq!(scene_avatar(&fresh)?, Value::Null);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Blender-gated third-party files
// ---------------------------------------------------------------------------------------------

fn assert_generated_humanoid(avatar: &Value) {
    let expected: Value = GENERATED_HUMANOID
        .iter()
        .map(|(role, bone)| ((*role).to_owned(), json!(bone)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    assert_eq!(
        avatar["humanoid"], expected,
        "each humanoid role maps to the bone the generator named"
    );
}

fn assert_generated_meta(avatar: &Value) {
    assert_eq!(avatar["meta"]["name"], "Third Party Fixture");
    assert_eq!(avatar["meta"]["authors"], json!(["Third Party Author"]));
    assert_eq!(avatar["meta"]["version"], "1.2.3");
}

/// The single morph target bind of an expression: mesh node `node`, shape key `shape_key`, weight 1.0.
fn assert_single_morph_bind(expression: &Value, node: &str, shape_key: &str, what: &str) {
    let binds = expression["morph_target_binds"]
        .as_array()
        .unwrap_or_else(|| panic!("{what} has no morph target binds: {expression}"));
    assert_eq!(binds.len(), 1, "{what} binds one morph target");
    assert_eq!(binds[0]["node"], node, "{what} mesh node");
    assert_eq!(binds[0]["shape_key"], shape_key, "{what} shape key");
    assert_close(number(&binds[0]["weight"], what), 1.0, TOLERANCE, what);
}

/// Spring chain and collider as the add-on wrote them. `vrm1` selects the VRM 1.0 names and gravity frame.
///
/// - Collider radius: VRM 1.0 writes the Blender radius unchanged; VRM 0.x writes `empty_display_size * scale`,
///   which is 0.05 for the generated empty.
/// - Collider offset: the add-on's offset vector is rigid-transformed on import (VRM 0.x also re-orients the scene into
///   Unity space), so only its length (0.1) is frame-independent. The length is asserted, not the components.
/// - Gravity: VRM 1.0 writes `[x, z, -y]` of the Blender vector (0.6, 0, -0.8), so the import returns it unchanged.
///   VRM 0.x applies its own Unity axis map, so only the unit length is asserted.
fn assert_generated_spring(spring: &Value, armature: &str, vrm1: bool) {
    let colliders = spring["colliders"]
        .as_array()
        .unwrap_or_else(|| panic!("spring colliders are missing: {spring}"));
    assert_eq!(colliders.len(), 1, "one sphere collider");
    let collider = &colliders[0];
    assert_eq!(collider["node"], json!({"node": armature, "bone": "head"}));
    assert_eq!(collider["shape"]["type"], "sphere");
    assert_close(
        number(&collider["shape"]["radius"], "collider radius"),
        0.05,
        TOLERANCE,
        "collider radius",
    );
    assert_close(
        norm(vec3(&collider["shape"]["offset"], "collider offset")),
        0.1,
        TOLERANCE,
        "collider offset length",
    );

    let groups = spring["collider_groups"]
        .as_array()
        .unwrap_or_else(|| panic!("spring collider groups are missing: {spring}"));
    assert_eq!(groups.len(), 1, "one collider group");
    assert_eq!(groups[0]["colliders"], json!([0]));
    if vrm1 {
        assert_eq!(groups[0]["name"], "Head");
    }

    let springs = spring["springs"]
        .as_array()
        .unwrap_or_else(|| panic!("springs are missing: {spring}"));
    assert_eq!(springs.len(), 1, "one spring chain");
    let chain = &springs[0];
    if vrm1 {
        assert_eq!(chain["name"], "Hair");
    }
    assert!(
        chain["center"].is_null(),
        "the generator sets no spring center"
    );
    assert_eq!(chain["collider_groups"], json!([0]));

    let joints = chain["joints"]
        .as_array()
        .unwrap_or_else(|| panic!("spring joints are missing: {chain}"));
    let bones = joints
        .iter()
        .map(|joint| joint["node"]["bone"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(
        bones,
        ["hair_root", "hair_mid", "hair_tip"],
        "joints run from the root bone to the tip"
    );
    for joint in joints {
        assert_eq!(joint["node"]["node"], armature);
        assert_close(
            number(&joint["hit_radius"], "hit radius"),
            0.02,
            TOLERANCE,
            "hit radius",
        );
        assert_close(
            number(&joint["stiffness"], "stiffness"),
            0.8,
            TOLERANCE,
            "stiffness",
        );
        assert_close(
            number(&joint["gravity_power"], "gravity power"),
            1.1,
            TOLERANCE,
            "gravity power",
        );
        assert_close(
            number(&joint["drag_force"], "drag force"),
            0.3,
            TOLERANCE,
            "drag force",
        );
        if vrm1 {
            assert_vec3_close(&joint["gravity_dir"], [0.6, 0.0, -0.8], "gravity direction");
        } else {
            assert_close(
                norm(vec3(&joint["gravity_dir"], "gravity direction")),
                1.0,
                TOLERANCE,
                "gravity direction length",
            );
        }
    }
}

fn assert_generated_v1_avatar(avatar: &Value) {
    let armature = avatar["armature"]
        .as_str()
        .unwrap_or_else(|| panic!("avatar armature is not an id: {avatar}"));
    assert_generated_humanoid(avatar);
    assert_generated_meta(avatar);
    assert_single_morph_bind(&avatar["expressions"]["happy"], "body", "smile", "happy");
    assert_single_morph_bind(&avatar["expressions"]["blink"], "body", "blink", "blink");
    assert_single_morph_bind(
        &avatar["expressions"]["wink"],
        "body",
        "wink",
        "custom wink",
    );
    assert_generated_spring(&avatar["spring_bone"], armature, true);

    // The add-on writes its look-at defaults (bone type, zero offset, 90/10 ranges).
    let look_at = &avatar["look_at"];
    assert_eq!(look_at["look_at_type"], "bone");
    assert_vec3_close(
        &look_at["offset_from_head_bone"],
        [0.0; 3],
        "look-at offset",
    );
    for range in [
        "range_map_horizontal_inner",
        "range_map_horizontal_outer",
        "range_map_vertical_down",
        "range_map_vertical_up",
    ] {
        assert_close(
            number(&look_at[range]["input_max_value"], range),
            90.0,
            TOLERANCE,
            range,
        );
        assert_close(
            number(&look_at[range]["output_scale"], range),
            10.0,
            TOLERANCE,
            range,
        );
    }

    assert!(
        avatar["mtoon"]["skin"].is_object(),
        "MToon settings are imported for the add-on's MToon material `skin`"
    );
}

fn assert_generated_v0_avatar(avatar: &Value) {
    let armature = avatar["armature"]
        .as_str()
        .unwrap_or_else(|| panic!("avatar armature is not an id: {avatar}"));
    assert_generated_humanoid(avatar);
    assert_generated_meta(avatar);
    // VRM 0.x blend shape weights are percentages (100 = 1.0). Joy and Blink are VRM 0.x presets for happy and blink.
    assert_single_morph_bind(&avatar["expressions"]["happy"], "body", "smile", "Joy");
    assert_single_morph_bind(&avatar["expressions"]["blink"], "body", "blink", "Blink");
    assert_single_morph_bind(
        &avatar["expressions"]["wink"],
        "body",
        "wink",
        "unknown wink",
    );
    assert_generated_spring(&avatar["spring_bone"], armature, false);
    assert!(
        avatar["mtoon"]["skin"].is_object(),
        "MToon settings are imported for the add-on's MToon material `skin`"
    );
}

fn loss_ids(expected: &[&str]) -> Vec<String> {
    let mut ids = expected
        .iter()
        .map(|id| (*id).to_owned())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

#[test]
fn blender_addon_vrm1_import_restores_humanoid_expressions_springs_and_mtoon() -> TestResult {
    let generator = match vrm_import_blender::setup()? {
        Gate::Ready(generator) => generator,
        Gate::Skip(reason) => {
            eprintln!("skipping VRM 1.0 third-party import: {reason}");
            return Ok(());
        }
    };
    let directory = tempfile::tempdir()?;
    let vrm = directory.path().join("third-party-v1.vrm");
    generator.generate(&vrm, "1.0")?;

    let (_fresh_directory, project) = init_project()?;
    // The add-on writes VRMC_node_constraint for the generator's Copy Rotation, which Potter does not represent.
    // Its KHR_materials_unlit fallback sits on MToon materials and is not a loss.
    assert_eq!(
        import_losses(&project, &vrm, "vrm")?,
        loss_ids(&["vrm.extension.VRMC_node_constraint"])
    );
    assert_generated_v1_avatar(&scene_avatar(&project)?);
    Ok(())
}

#[test]
fn blender_addon_vrm0_import_restores_legacy_humanoid_expressions_and_springs() -> TestResult {
    let generator = match vrm_import_blender::setup()? {
        Gate::Ready(generator) => generator,
        Gate::Skip(reason) => {
            eprintln!("skipping VRM 0.x third-party import: {reason}");
            return Ok(());
        }
    };
    let directory = tempfile::tempdir()?;
    let vrm = directory.path().join("third-party-v0.vrm");
    generator.generate(&vrm, "0.0")?;

    let (_fresh_directory, project) = init_project()?;
    // The add-on's MToon materials carry KHR_materials_unlit as MToon's fallback, which Potter reads as MToon: no loss.
    assert_eq!(import_losses(&project, &vrm, "vrm0")?, loss_ids(&[]));
    assert_generated_v0_avatar(&scene_avatar(&project)?);
    Ok(())
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

/// Sets `key` to `value` on every `MToon` material of a VRM 1.0 file.
fn set_mtoon_v1(document: &mut Value, key: &str, value: &Value) {
    for material in document["materials"].as_array_mut().into_iter().flatten() {
        if let Some(mtoon) = material["extensions"]["VRMC_materials_mtoon"].as_object_mut() {
            mtoon.insert(key.to_owned(), value.clone());
        }
    }
}

#[test]
fn vrm_import_append_refuses_the_avatar_when_an_imported_id_collides() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v1.vrm");
    export(&source, "vrm", &vrm)?;
    let (_plain_directory, project) = init_project()?;
    let output = apply_batch(
        &project,
        0,
        &[json!({"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1.0}})],
    )?;
    assert!(
        output.status.success(),
        "setup failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The scene's `body` collides with the fixture's, so append renames the import and keeps no avatar.
    assert_eq!(
        import_losses(&project, &vrm, "vrm")?,
        loss_ids(&["vrm.avatar"])
    );
    assert!(scene_avatar(&project)?.is_null());
    Ok(())
}

#[test]
fn vrm0_spring_group_with_several_roots_imports_one_chain_per_root() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v0.vrm");
    export(&source, "vrm0", &vrm)?;
    // VRoid lists several root bones in one VRM 0.x bone group; each root starts its own chain.
    edit_glb(&vrm, |document| {
        let eye = document["nodes"]
            .as_array()
            .and_then(|nodes| nodes.iter().position(|node| node["name"] == "Eye.L"))
            .ok_or("the fixture has no Eye.L node")?;
        let groups = document["extensions"]["VRM"]["secondaryAnimation"]["boneGroups"]
            .as_array_mut()
            .ok_or("the fixture has no bone groups")?;
        let hair = groups
            .iter_mut()
            .find(|group| group["comment"] == "Hair")
            .ok_or("the fixture has no hair group")?;
        let root = hair["bones"][0].clone();
        hair["bones"] = json!([root, eye]);
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(import_losses(&project, &vrm, "vrm0")?, loss_ids(&[]));
    let avatar = scene_avatar(&project)?;
    let springs = avatar["spring_bone"]["springs"]
        .as_array()
        .ok_or("the imported avatar has no springs")?;
    let joint_bones = springs
        .iter()
        .map(|spring| {
            spring["joints"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|joint| {
                    joint["node"]["bone"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        joint_bones,
        vec![
            vec!["hair_root".to_owned(), "hair_tip".to_owned()],
            vec!["eye_l".to_owned()],
        ],
        "the hair root and the eye root each start a chain"
    );
    Ok(())
}

#[test]
fn out_of_range_mtoon_color_clamps_with_a_loss_and_keeps_the_avatar() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v1.vrm");
    export(&source, "vrm", &vrm)?;
    // Third-party files write HDR rim colors; one of them must not discard the whole avatar.
    edit_glb(&vrm, |document| {
        set_mtoon_v1(
            document,
            "parametricRimColorFactor",
            &json!([1.5, 0.5, 0.25]),
        );
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(
        import_losses(&project, &vrm, "vrm")?,
        loss_ids(&["vrm.mtoon.parametricRimColorFactor"])
    );
    let avatar = scene_avatar(&project)?;
    assert!(
        !avatar.is_null(),
        "the avatar survives an out-of-range color"
    );
    let rims = avatar["mtoon"]
        .as_object()
        .ok_or("the imported avatar has no MToon settings")?
        .values()
        .map(|mtoon| mtoon["parametric_rim_color_factor"].clone())
        .collect::<Vec<_>>();
    assert_eq!(rims.len(), 1);
    assert_vec3_close(&rims[0], [1.0, 0.5, 0.25], "clamped rim color");
    Ok(())
}

#[test]
fn vrm0_texture_named_vectors_are_identity_unless_transformed() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v0.vrm");
    export(&source, "vrm0", &vrm)?;
    // The VRM Add-on names a texture's (offset, scale) vector after the texture: `[0, 0, 1, 1]` is untransformed.
    let set_vectors = |document: &mut Value, main: Value| {
        for material in document["extensions"]["VRM"]["materialProperties"]
            .as_array_mut()
            .into_iter()
            .flatten()
        {
            let vectors = &mut material["vectorProperties"];
            vectors["_MainTex"] = main.clone();
            vectors["_UvAnimMaskTexture"] = json!([0.0, 0.0, 1.0, 1.0]);
        }
    };
    edit_glb(&vrm, |document| {
        set_vectors(document, json!([0.0, 0.0, 1.0, 1.0]));
        Ok(())
    })?;
    let (_identity_directory, identity_project) = init_project()?;
    assert_eq!(
        import_losses(&identity_project, &vrm, "vrm0")?,
        loss_ids(&[])
    );

    edit_glb(&vrm, |document| {
        set_vectors(document, json!([0.25, 0.0, 2.0, 2.0]));
        Ok(())
    })?;
    let (_moved_directory, moved_project) = init_project()?;
    assert_eq!(
        import_losses(&moved_project, &vrm, "vrm0")?,
        loss_ids(&[]),
        "a transformed main texture moves into the base color graph, so it is not a loss"
    );
    Ok(())
}

#[test]
fn vrm_stale_potter_vertex_ids_after_addon_round_trip_import_with_a_loss() -> TestResult {
    let generator = match vrm_import_blender::setup()? {
        Gate::Ready(generator) => generator,
        Gate::Skip(reason) => {
            eprintln!("skipping stale vertex ID import: {reason}");
            return Ok(());
        }
    };
    let (directory, source) = new_fixture()?;
    let exported = directory.path().join("potter.vrm");
    export(&source, "vrm", &exported)?;
    // The add-on keeps mesh extras but drops root extras, so `vertex_ids` no longer match the vertices.
    let round_tripped = directory.path().join("addon.vrm");
    generator.round_trip(&exported, &round_tripped)?;
    let (_fresh_directory, project) = init_project()?;
    let losses = import_losses(&project, &round_tripped, "vrm")?;
    assert!(
        losses.iter().any(|loss| loss == "mesh.vertex_ids"),
        "stale vertex IDs must import with a loss, got {losses:?}"
    );
    Ok(())
}

#[test]
fn vrm_import_drops_a_broken_optional_component_and_keeps_the_rest_of_the_avatar() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v1.vrm");
    export(&source, "vrm", &vrm)?;
    // A look-at input range above 180 degrees is not representable. Only look-at is optional here.
    edit_glb(&vrm, |document| {
        let range = &mut document["extensions"]["VRMC_vrm"]["lookAt"]["rangeMapHorizontalInner"];
        range["inputMaxValue"] = json!(200.0);
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(
        import_losses(&project, &vrm, "vrm")?,
        loss_ids(&["vrm.look_at"])
    );
    let avatar = scene_avatar(&project)?;
    assert!(avatar["look_at"].is_null(), "the broken look-at is dropped");
    assert!(!avatar["humanoid"].is_null(), "the humanoid is kept");
    assert!(
        !avatar["expressions"].is_null()
            && avatar["expressions"]
                .as_object()
                .is_some_and(|map| !map.is_empty()),
        "expressions are kept"
    );
    assert!(!avatar["spring_bone"].is_null(), "spring bones are kept");
    assert!(!avatar["mtoon"].is_null(), "MToon settings are kept");
    Ok(())
}

#[test]
fn vrm0_main_texture_transform_moves_the_base_color_texture_into_a_graph() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("v0.vrm");
    export(&source, "vrm0", &vrm)?;
    // The VRM Add-on writes `_MainTex` as (offset x, stored offset y, scale x, scale y).
    edit_glb(&vrm, |document| {
        for material in document["extensions"]["VRM"]["materialProperties"]
            .as_array_mut()
            .into_iter()
            .flatten()
        {
            if material["shader"] == "VRM/MToon" {
                material["vectorProperties"]["_MainTex"] = json!([0.25, 0.1, 2.0, 3.0]);
            }
        }
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(import_losses(&project, &vrm, "vrm0")?, loss_ids(&[]));
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let graph_materials = scene["materials"]
        .as_object()
        .into_iter()
        .flat_map(|materials| materials.values())
        .filter(|material| !material["node_tree"].is_null())
        .count();
    assert_eq!(
        graph_materials, 1,
        "the MToon base texture moved into one shader graph"
    );
    Ok(())
}

#[test]
fn vrm1_base_color_texture_transform_round_trips_through_vrm_export() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("transformed.vrm");
    export(&source, "vrm", &vrm)?;
    let transform = json!({ "offset": [0.2, 0.1], "rotation": 0.25, "scale": [1.5, 2.0] });
    edit_glb(&vrm, |document| {
        let material = document["materials"]
            .as_array_mut()
            .and_then(|materials| {
                materials.iter_mut().find(|material| {
                    material["pbrMetallicRoughness"]["baseColorTexture"].is_object()
                })
            })
            .ok_or("the fixture has no textured material")?;
        material["pbrMetallicRoughness"]["baseColorTexture"]["extensions"] =
            json!({ "KHR_texture_transform": transform.clone() });
        document["extensionsUsed"]
            .as_array_mut()
            .ok_or("no extensionsUsed")?
            .push(json!("KHR_texture_transform"));
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(import_losses(&project, &vrm, "vrm")?, loss_ids(&[]));
    let again = directory.path().join("again.vrm");
    export(&project, "vrm", &again)?;
    let document = glb_json(&again)?;
    let exported = document["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|material| {
            &material["pbrMetallicRoughness"]["baseColorTexture"]["extensions"]["KHR_texture_transform"]
        })
        .find(|value| value.is_object())
        .ok_or("no exported base color transform")?;
    for (key, expected) in [("offset", [0.2, 0.1]), ("scale", [1.5, 2.0])] {
        let values = exported[key].as_array().ok_or("no exported vector")?;
        for (value, expected) in values.iter().zip(expected) {
            let value = value.as_f64().ok_or("exported component is not a number")?;
            assert!(
                (value - expected).abs() < 1e-12,
                "{key}: {value} != {expected}"
            );
        }
    }
    let rotation = exported["rotation"].as_f64().ok_or("no rotation")?;
    assert!((rotation - 0.25).abs() < 1e-12);
    Ok(())
}

#[test]
fn vrm1_shade_color_expression_imports_after_its_mtoon_material() -> TestResult {
    let (directory, source) = new_fixture()?;
    let vrm = directory.path().join("shade.vrm");
    export(&source, "vrm", &vrm)?;
    edit_glb(&vrm, |document| {
        document["extensions"]["VRMC_vrm"]["expressions"]["preset"]["happy"]["materialColorBinds"]
            [0]["type"] = json!("shadeColor");
        Ok(())
    })?;
    let (_fresh_directory, project) = init_project()?;
    assert_eq!(import_losses(&project, &vrm, "vrm")?, loss_ids(&[]));
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let bind = &scene["avatar"]["expressions"]["happy"]["material_color_binds"][0];
    assert_eq!(
        bind["color_type"], "shadeColor",
        "the bind keeps its shade color"
    );
    Ok(())
}
