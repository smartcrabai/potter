#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test fixture assertions"
)]

use glam::{DMat4, DQuat, DVec3};
use image::load_from_memory;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::SceneDoc,
};
use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Default)]
struct FixtureOptions {
    capsule: bool,
    uneven_spring_joints: bool,
    expression_texture_transform: bool,
    modern_extra_permissions: bool,
    expression_look_at: bool,
    uv_rotation_only: bool,
    uv_scroll_and_rotation: bool,
    v1_extended_mtoon: bool,
}

struct Glb {
    json: Value,
    binary: Vec<u8>,
}

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn cli_json(output: &std::process::Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init_project() -> TestResult<(TempDir, std::path::PathBuf)> {
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

fn apply_batch(
    project: &Path,
    base_revision: u64,
    operations: &[Value],
) -> TestResult<std::process::Output> {
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

fn apply_success(project: &Path, base_revision: u64, operations: &[Value]) -> TestResult<Value> {
    let output = apply_batch(project, base_revision, operations)?;
    assert!(
        output.status.success(),
        "pot apply failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    cli_json(&output)
}

fn inspect(project: &Path) -> TestResult<Value> {
    let output = pot().arg("inspect").arg(project).arg("--json").output()?;
    assert!(
        output.status.success(),
        "pot inspect failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    cli_json(&output)
}

fn revision(project: &Path) -> TestResult<u64> {
    inspect(project)?["scene"]["revision"]
        .as_u64()
        .ok_or_else(|| "inspect did not report the scene revision".into())
}

fn assert_failure(output: &std::process::Output, code: &str) -> TestResult<Value> {
    assert!(
        !output.status.success(),
        "expected the CLI operation to fail"
    );
    let envelope = cli_json(output)?;
    assert_eq!(envelope["ok"], false);
    assert_eq!(envelope["error"]["code"], code);
    assert!(
        envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );
    assert!(
        envelope["error"]["details"]
            .as_object()
            .is_some_and(|details| !details.is_empty())
    );
    Ok(envelope)
}
fn assert_operation_failure(output: &std::process::Output) -> TestResult<()> {
    let envelope = assert_failure(output, "INVALID_OPERATION")?;
    assert!(
        envelope["error"]["details"]["pointer"]
            .as_str()
            .is_some_and(|pointer| !pointer.is_empty()),
        "invalid operation must report its JSON pointer"
    );
    Ok(())
}

fn image_operations() -> Vec<Value> {
    [
        ("tex_albedo", "Albedo", [0.8, 0.4, 0.2, 1.0]),
        ("tex_shade", "Shade", [0.2, 0.3, 0.4, 1.0]),
        ("tex_shift", "Shift", [0.3, 0.4, 0.5, 1.0]),
        ("tex_rim", "Rim", [0.5, 0.6, 0.7, 1.0]),
        ("tex_outline", "Outline", [0.6, 0.7, 0.8, 1.0]),
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

fn meta(options: FixtureOptions) -> Value {
    let mut legacy = json!({
        "title": "Legacy Fixture Avatar",
        "author": "Potter VRM Test",
        "reference": "https://example.test/reference",
        "allowed_user": "only_author",
        "violent_usage": "disallow",
        "sexual_usage": "disallow",
        "commercial_usage": "disallow",
        "license_name": "redistribution_prohibited",
    });
    if !options.modern_extra_permissions {
        legacy["other_permission_url"] = json!("https://example.test/permissions");
    }
    json!({
        "name": "Native VRM Avatar",
        "authors": ["Potter VRM Test"],
        "license_url": "https://vrm.dev/licenses/1.0/",
        "thumbnail": "thumbnail",
        "avatar_permission": "only_author",
        "allow_excessively_violent_usage": false,
        "allow_excessively_sexual_usage": false,
        "commercial_usage": "personal_non_profit",
        "allow_political_or_religious_usage": options.modern_extra_permissions,
        "allow_antisocial_or_hate_usage": options.modern_extra_permissions,
        "credit_notation": "required",
        "allow_redistribution": false,
        "modification": "prohibited",
        "legacy": legacy,
    })
}

fn spring_bone(options: FixtureOptions) -> Value {
    let shape = if options.capsule {
        json!({
            "type": "capsule",
            "offset": [0.1, 0.2, 0.3],
            "tail": [0.4, 0.5, 0.6],
            "radius": 0.08,
        })
    } else {
        json!({
            "type": "sphere",
            "offset": [0.1, 0.2, 0.3],
            "radius": 0.08,
        })
    };
    let second_stiffness = if options.uneven_spring_joints {
        0.4
    } else {
        0.8
    };
    json!({
        "colliders": [{
            "node": {"node": "arm", "bone": "hair_root"},
            "shape": shape,
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
                    "stiffness": second_stiffness,
                    "gravity_power": 1.1,
                    "gravity_dir": [1.0, 2.0, 3.0],
                    "drag_force": 0.2,
                },
            ],
            "collider_groups": [0],
        }],
    })
}

fn mtoon(options: FixtureOptions) -> Value {
    let (scroll_x, scroll_y, rotation_speed) = if options.uv_scroll_and_rotation {
        (0.2, -0.1, 0.3)
    } else if options.uv_rotation_only {
        (0.0, 0.0, 0.3)
    } else {
        (0.2, -0.1, 0.0)
    };
    let mut settings = json!({
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
        "uv_animation_scroll_x_speed_factor": scroll_x,
        "uv_animation_scroll_y_speed_factor": scroll_y,
        "uv_animation_rotation_speed_factor": rotation_speed,
    });
    if options.v1_extended_mtoon {
        settings["matcap_factor"] = json!([0.8, 0.7, 0.6]);
        settings["shading_shift_texture"] =
            json!({"texture": {"image": "tex_shift"}, "scale": 0.4});
    }
    settings
}

fn avatar_settings(options: FixtureOptions) -> Value {
    let mut happy = json!({
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
    });
    if options.expression_texture_transform {
        happy["override_blink"] = json!("block");
        happy["override_mouth"] = json!("blend");
        happy["texture_transform_binds"] = json!([{
            "material": "skin_material",
            "scale": [1.25, 0.75],
            "offset": [0.1, 0.2],
        }]);
    }
    let mut expressions = json!({
        "happy": happy,
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
        "meta": meta(options),
        "expressions": expressions,
        "first_person": {
            "mesh_annotations": [{"node": "body", "annotation_type": "first_person_only"}],
        },
        "look_at": {
            "look_at_type": if options.expression_look_at { "expression" } else { "bone" },
            "offset_from_head_bone": [0.1, 0.2, 0.3],
            "range_map_horizontal_inner": {"input_max_value": 30.0, "output_scale": 0.25},
            "range_map_horizontal_outer": {"input_max_value": 40.0, "output_scale": 0.5},
            "range_map_vertical_down": {"input_max_value": 50.0, "output_scale": 0.75},
            "range_map_vertical_up": {"input_max_value": 60.0, "output_scale": 1.0},
        },
        "spring_bone": spring_bone(options),
        "mtoon": {"skin_material": mtoon(options)},
    })
}

fn fixture_operations(options: FixtureOptions) -> Vec<Value> {
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
            "base_color": [0.8, 0.65, 0.5, 1.0],
            "base_color_texture": {"image": "tex_albedo"},
        }),
        json!({"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1.0}, "material": "skin_material"}),
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
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_root", "name": "Hair Root", "head": [0.0, 0.0, 1.9], "tail": [0.0, 0.0, 2.05], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_tip", "name": "Hair Tip", "head": [0.0, 0.0, 2.05], "tail": [0.0, 0.0, 2.2], "parent": "hair_root"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "skip_start", "name": "Skipped Start", "head": [0.2, 0.0, 1.85], "tail": [0.2, 0.0, 1.9], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "skip_middle_l", "name": "Skipped Middle L", "head": [0.2, 0.0, 1.9], "tail": [0.2, 0.0, 1.95], "parent": "skip_start"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "skip_middle_r", "name": "Skipped Middle R", "head": [0.2, 0.0, 1.95], "tail": [0.2, 0.0, 2.0], "parent": "skip_middle_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "skip_end", "name": "Skipped End", "head": [0.2, 0.0, 2.0], "tail": [0.2, 0.0, 2.05], "parent": "skip_middle_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "branch_shoulder_l", "name": "Branch Shoulder.L", "head": [0.0, 0.0, 1.35], "tail": [0.1, 0.0, 1.4], "parent": "spine"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "branch_upper_arm_l", "name": "Branch Upper Arm.L", "head": [0.1, 0.0, 1.4], "tail": [0.2, 0.0, 1.45], "parent": "branch_shoulder_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "branch_forearm_l", "name": "Branch Forearm.L", "head": [0.2, 0.0, 1.45], "tail": [0.3, 0.0, 1.5], "parent": "branch_upper_arm_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "branch_hand_l", "name": "Branch Hand.L", "head": [0.3, 0.0, 1.5], "tail": [0.4, 0.0, 1.5], "parent": "branch_forearm_l"}),
        json!({"op": "node.duplicate", "target": {"id": "body"}, "id": "body_copy", "mode": "linked"}),
        json!({"op": "node.update", "target": {"id": "body_copy"}, "set": {"transform": {"translation": [2.0, 0.0, 0.0]}}}),
        json!({"op": "avatar.update", "set": avatar_settings(options)}),
    ]);
    operations
}

fn new_fixture(options: FixtureOptions) -> TestResult<(TempDir, std::path::PathBuf)> {
    let (directory, project) = init_project()?;
    let output = apply_batch(&project, 0, &fixture_operations(options))?;
    assert!(
        output.status.success(),
        "VRM fixture setup failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((directory, project))
}

fn export(project: &Path, format: &str, output_path: &Path) -> TestResult<Value> {
    let output = pot()
        .arg("export")
        .arg(project)
        .args(["--format", format, "--out"])
        .arg(output_path)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "VRM export `{format}` failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output_path
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("vrm")
    );
    let envelope = cli_json(&output)?;
    assert_eq!(envelope["ok"], true);
    Ok(envelope)
}

fn export_failure(project: &Path, format: &str, output_path: &Path) -> TestResult<Value> {
    let output = pot()
        .arg("export")
        .arg(project)
        .args(["--format", format, "--out"])
        .arg(output_path)
        .arg("--json")
        .output()?;
    assert_failure(&output, "UNSUPPORTED_FEATURE")
}
fn assert_validate_export_preflight(
    project: &Path,
    format: &str,
    output_path: &Path,
    feature_id: &str,
) -> TestResult<Value> {
    let validation_output = pot()
        .arg("validate")
        .arg(project)
        .args(["--format", format, "--json"])
        .output()?;
    assert!(validation_output.status.success());
    let validation = cli_json(&validation_output)?;
    assert_eq!(validation["result"]["valid"], false);
    assert!(
        validation["result"]["checks"]
            .as_array()
            .is_some_and(|checks| checks.iter().any(|check| {
                check["check"] == "format_loss" && check["status"] == "not_supported"
            }))
    );
    let issue = validation["result"]["issues"]
        .as_array()
        .and_then(|issues| {
            issues
                .iter()
                .find(|issue| issue["code"] == "VRM_EXPORT_PREFLIGHT_FAILED")
        })
        .ok_or("format validation omitted the writer preflight result")?;
    assert_eq!(issue["details"]["format"], format);
    assert_eq!(issue["details"]["error_code"], "UNSUPPORTED_FEATURE");
    assert_eq!(issue["details"]["details"]["feature_id"], feature_id);

    let strict_output = pot()
        .arg("validate")
        .arg(project)
        .args(["--format", format, "--strict", "--json"])
        .output()?;
    assert_eq!(strict_output.status.code(), Some(4));
    let strict = cli_json(&strict_output)?;
    assert_eq!(strict["error"]["code"], "VALIDATION_FAILED");
    assert_eq!(strict["error"]["details"]["strict"], true);
    assert_eq!(strict["error"]["details"]["result"]["valid"], false);
    assert!(
        strict["error"]["details"]["result"]["issues"]
            .as_array()
            .is_some_and(|issues| issues.iter().any(|issue| {
                issue["code"] == "VRM_EXPORT_PREFLIGHT_FAILED"
                    && issue["details"]["details"]["feature_id"] == feature_id
            }))
    );

    let export_failure = export_failure(project, format, output_path)?;
    assert_eq!(export_failure["error"]["details"]["feature_id"], feature_id);
    assert_eq!(
        issue["details"]["details"]["reason"],
        export_failure["error"]["details"]["reason"]
    );
    Ok(export_failure)
}

fn assert_preflight_both_versions(
    directory: &TempDir,
    project: &Path,
    output_stem: &str,
    feature_id: &str,
) -> TestResult<()> {
    for format in ["vrm", "vrm0"] {
        assert_validate_export_preflight(
            project,
            format,
            &directory.path().join(format!("{output_stem}-{format}.vrm")),
            feature_id,
        )?;
    }
    Ok(())
}

fn assert_format_loss_both_versions(project: &Path, feature_id: &str) -> TestResult<()> {
    for format in ["vrm", "vrm0"] {
        let validation_output = pot()
            .arg("validate")
            .arg(project)
            .args(["--format", format, "--json"])
            .output()?;
        assert!(validation_output.status.success());
        let validation = cli_json(&validation_output)?;
        assert_eq!(validation["result"]["valid"], false);
        assert!(
            validation["result"]["checks"]
                .as_array()
                .is_some_and(|checks| checks.iter().any(|check| {
                    check["check"] == "format_loss" && check["status"] == "not_supported"
                }))
        );
        assert!(
            validation["result"]["issues"]
                .as_array()
                .is_some_and(|issues| issues.iter().any(|issue| {
                    issue["code"] == "UNREPRESENTABLE_FEATURE"
                        && issue["details"]["feature_id"] == feature_id
                }))
        );
    }
    Ok(())
}

fn assert_exports_both_versions(
    directory: &TempDir,
    project: &Path,
    output_stem: &str,
) -> TestResult<()> {
    for format in ["vrm", "vrm0"] {
        export(
            project,
            format,
            &directory.path().join(format!("{output_stem}-{format}.vrm")),
        )?;
    }
    Ok(())
}

fn parse_glb(path: &Path) -> TestResult<Glb> {
    let bytes = fs::read(path)?;
    assert!(bytes.len() >= 28, "GLB container is too short");
    assert_eq!(&bytes[0..4], b"glTF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into()?), 2);
    assert_eq!(
        usize::try_from(u32::from_le_bytes(bytes[8..12].try_into()?))?,
        bytes.len()
    );
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    assert_eq!(&bytes[16..20], b"JSON");
    let json_end = 20_usize
        .checked_add(json_length)
        .ok_or("JSON chunk overflow")?;
    assert!(
        json_end
            .checked_add(8)
            .is_some_and(|end| end <= bytes.len())
    );
    let json: Value = serde_json::from_slice(&bytes[20..json_end])?;
    let binary_length = usize::try_from(u32::from_le_bytes(
        bytes[json_end..json_end + 4].try_into()?,
    ))?;
    assert_eq!(&bytes[json_end + 4..json_end + 8], b"BIN\0");
    let binary_start = json_end + 8;
    let binary_end = binary_start
        .checked_add(binary_length)
        .ok_or("BIN chunk overflow")?;
    assert_eq!(binary_end, bytes.len());
    assert!(
        json["buffers"][0]["byteLength"]
            .as_u64()
            .unwrap_or_default()
            <= u64::try_from(binary_length)?
    );
    Ok(Glb {
        json,
        binary: bytes[binary_start..binary_end].to_vec(),
    })
}

fn usize_field(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("expected a nonnegative glTF index")).unwrap()
}

fn gltf_node_index(root: &Value, id: &str) -> usize {
    root["nodes"]
        .as_array()
        .expect("glTF nodes are present")
        .iter()
        .position(|node| node["extras"]["potter"]["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("exported scene node `{id}` is missing"))
}

fn gltf_bone_index(root: &Value, id: &str) -> usize {
    root["nodes"]
        .as_array()
        .expect("glTF nodes are present")
        .iter()
        .position(|node| node["extras"]["potter"]["bone_id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("exported armature bone `{id}` is missing"))
}

fn gltf_material_index(root: &Value, id: &str) -> usize {
    root["materials"]
        .as_array()
        .expect("glTF materials are present")
        .iter()
        .position(|material| material["extras"]["potter"]["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("exported material `{id}` is missing"))
}

fn gltf_image_index(root: &Value, id: &str) -> usize {
    root["images"]
        .as_array()
        .expect("glTF images are present")
        .iter()
        .position(|image| image["extras"]["potter"]["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("exported image `{id}` is missing"))
}

fn texture_image_index(root: &Value, texture_index: usize) -> usize {
    usize_field(&root["textures"][texture_index]["source"])
}

fn embedded_image_pixel(glb: &Glb, image_index: usize) -> TestResult<[u8; 4]> {
    let image = &glb.json["images"][image_index];
    assert_eq!(image["mimeType"], "image/png");
    let view = &glb.json["bufferViews"][usize_field(&image["bufferView"])];
    let offset = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?;
    let length = usize::try_from(
        view["byteLength"]
            .as_u64()
            .ok_or("image bufferView length is missing")?,
    )?;
    let end = offset
        .checked_add(length)
        .ok_or("image bufferView range overflow")?;
    let bytes = glb
        .binary
        .get(offset..end)
        .ok_or("image bufferView exceeds BIN chunk")?;
    let decoded = load_from_memory(bytes)?.to_rgba8();
    Ok(decoded.get_pixel(0, 0).0)
}
fn assert_pixel_channels_close(actual: [u8; 4], expected: [u8; 4]) {
    for channel in 0..4 {
        assert!(
            (i16::from(actual[channel]) - i16::from(expected[channel])).abs() <= 1,
            "embedded pixels differ in channel {channel}: expected {expected:?}, got {actual:?}"
        );
    }
}

fn texture_image_id(root: &Value, texture_index: usize) -> &str {
    let image_index = usize_field(&root["textures"][texture_index]["source"]);
    root["images"][image_index]["extras"]["potter"]["id"]
        .as_str()
        .expect("embedded image keeps its source identity")
}

fn assert_texture_ref(root: &Value, texture_info: &Value, image_id: &str) {
    let texture_index = usize_field(&texture_info["index"]);
    assert_eq!(texture_image_id(root, texture_index), image_id);
}
fn assert_v0_mtoon_channel_remaps(v1: &Glb, v0: &Glb) -> TestResult<()> {
    let v1_root = &v1.json;
    let v0_root = &v0.json;
    let v1_material = &v1_root["materials"][gltf_material_index(v1_root, "skin_material")];
    let v1_mtoon = &v1_material["extensions"]["VRMC_materials_mtoon"];
    let v0_properties = &v0_root["extensions"]["VRM"]["materialProperties"]
        [gltf_material_index(v0_root, "skin_material")]["textureProperties"];

    let outline_image = texture_image_index(
        v1_root,
        usize_field(&v1_mtoon["outlineWidthMultiplyTexture"]["index"]),
    );
    let outline_pixel = embedded_image_pixel(v1, outline_image)?;
    let remapped_outline_image =
        texture_image_index(v0_root, usize_field(&v0_properties["_OutlineWidthTexture"]));
    let remapped_outline_pixel = embedded_image_pixel(v0, remapped_outline_image)?;
    assert_ne!(outline_pixel[1], 0);
    assert_ne!(outline_pixel[1], outline_pixel[2]);
    assert_pixel_channels_close(
        remapped_outline_pixel,
        [
            outline_pixel[1],
            outline_pixel[1],
            outline_pixel[1],
            outline_pixel[3],
        ],
    );

    let mask_image = texture_image_index(
        v1_root,
        usize_field(&v1_mtoon["uvAnimationMaskTexture"]["index"]),
    );
    let mask_pixel = embedded_image_pixel(v1, mask_image)?;
    let remapped_mask_image =
        texture_image_index(v0_root, usize_field(&v0_properties["_UvAnimMaskTexture"]));
    let remapped_mask_pixel = embedded_image_pixel(v0, remapped_mask_image)?;
    assert_ne!(mask_pixel[2], 0);
    assert_ne!(mask_pixel[2], mask_pixel[1]);
    assert_pixel_channels_close(
        remapped_mask_pixel,
        [mask_pixel[2], mask_pixel[2], mask_pixel[2], mask_pixel[3]],
    );

    let original_matcap_image =
        texture_image_index(v1_root, usize_field(&v1_mtoon["matcapTexture"]["index"]));
    let original_matcap_pixel = embedded_image_pixel(v1, original_matcap_image)?;
    let legacy_matcap_image =
        texture_image_index(v0_root, usize_field(&v0_properties["_SphereAdd"]));
    assert_pixel_channels_close(
        embedded_image_pixel(v0, legacy_matcap_image)?,
        original_matcap_pixel,
    );
    Ok(())
}

fn legacy_bone_node(root: &Value, role: &str) -> usize {
    root["extensions"]["VRM"]["humanoid"]["humanBones"]
        .as_array()
        .expect("legacy humanoid bones are an array")
        .iter()
        .find(|bone| bone["bone"].as_str() == Some(role))
        .map_or_else(
            || panic!("legacy humanoid role `{role}` is missing"),
            |bone| usize_field(&bone["node"]),
        )
}

fn close(actual: &Value, expected: f64) {
    let actual = actual.as_f64().expect("expected a numeric VRM value");
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

fn assert_vector(value: &Value, expected: [f64; 3]) {
    let actual = value.as_array().expect("expected a vector");
    assert_eq!(actual.len(), 3);
    for (index, expected) in expected.into_iter().enumerate() {
        close(&actual[index], expected);
    }
}
fn json_numbers<const N: usize>(value: &Value) -> TestResult<[f64; N]> {
    let values = value.as_array().ok_or("expected numeric array")?;
    if values.len() != N {
        return Err("numeric array has the wrong length".into());
    }
    let mut result = [0.0; N];
    for (index, value) in values.iter().enumerate() {
        result[index] = value.as_f64().ok_or("array component is not numeric")?;
    }
    Ok(result)
}

fn gltf_local_matrix(root: &Value, index: usize) -> TestResult<DMat4> {
    let node = &root["nodes"][index];
    if let Some(matrix) = node.get("matrix") {
        return Ok(DMat4::from_cols_array(&json_numbers::<16>(matrix)?));
    }
    let translation = node
        .get("translation")
        .map_or(Ok([0.0, 0.0, 0.0]), json_numbers::<3>)?;
    let rotation = node
        .get("rotation")
        .map_or(Ok([0.0, 0.0, 0.0, 1.0]), json_numbers::<4>)?;
    let scale = node
        .get("scale")
        .map_or(Ok([1.0, 1.0, 1.0]), json_numbers::<3>)?;
    Ok(DMat4::from_scale_rotation_translation(
        DVec3::from_array(scale),
        DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        DVec3::from_array(translation),
    ))
}

fn gltf_node_parent(root: &Value, index: usize) -> Option<usize> {
    let child = json!(index);
    root["nodes"].as_array()?.iter().position(|node| {
        node["children"]
            .as_array()
            .is_some_and(|children| children.iter().any(|candidate| candidate == &child))
    })
}

fn gltf_node_world_matrix(root: &Value, index: usize) -> TestResult<DMat4> {
    let local = gltf_local_matrix(root, index)?;
    gltf_node_parent(root, index).map_or(Ok(local), |parent| {
        Ok(gltf_node_world_matrix(root, parent)? * local)
    })
}

fn gltf_node_world_position(root: &Value, index: usize) -> TestResult<DVec3> {
    Ok(gltf_node_world_matrix(root, index)?.transform_point3(DVec3::ZERO))
}

fn potter_point_to_vrm1(point: DVec3) -> DVec3 {
    DVec3::new(point.x, point.z, -point.y)
}

fn potter_point_to_vrm0(point: DVec3) -> DVec3 {
    DVec3::new(-point.x, point.z, point.y)
}

fn object_vector(value: &Value) -> TestResult<DVec3> {
    Ok(DVec3::new(
        value["x"].as_f64().ok_or("vector x is missing")?,
        value["y"].as_f64().ok_or("vector y is missing")?,
        value["z"].as_f64().ok_or("vector z is missing")?,
    ))
}

fn assert_point_close(actual: DVec3, expected: DVec3) {
    assert!(
        actual.distance(expected) <= 1.0e-4,
        "expected transformed point {expected:?}, got {actual:?}"
    );
}

fn assert_v1_local_frames(project: &Path, glb: &Glb, options: FixtureOptions) -> TestResult<()> {
    let root = &glb.json;
    let source_head = evaluated_bone_matrix(project, "arm", "head")?;
    let output_head = gltf_node_world_matrix(root, gltf_bone_index(root, "head"))?;
    let input_look_offset = DVec3::new(0.1, 0.2, 0.3);
    let output_look_offset = DVec3::from_array(json_numbers::<3>(
        &root["extensions"]["VRMC_vrm"]["lookAt"]["offsetFromHeadBone"],
    )?);
    assert_point_close(
        output_head.transform_point3(output_look_offset),
        potter_point_to_vrm1(source_head.transform_point3(input_look_offset)),
    );

    let source_hair = evaluated_bone_matrix(project, "arm", "hair_root")?;
    let output_hair = gltf_node_world_matrix(root, gltf_bone_index(root, "hair_root"))?;
    let spring_collider = &root["extensions"]["VRMC_springBone"]["colliders"][0];
    let input_offset = DVec3::new(0.1, 0.2, 0.3);
    let output_offset = if options.capsule {
        &spring_collider["shape"]["capsule"]["offset"]
    } else {
        &spring_collider["shape"]["sphere"]["offset"]
    };
    assert_point_close(
        output_hair.transform_point3(DVec3::from_array(json_numbers::<3>(output_offset)?)),
        potter_point_to_vrm1(source_hair.transform_point3(input_offset)),
    );
    if options.capsule {
        let input_tail = DVec3::new(0.4, 0.5, 0.6);
        let output_tail = DVec3::from_array(json_numbers::<3>(
            &spring_collider["shape"]["capsule"]["tail"],
        )?);
        assert_point_close(
            output_hair.transform_point3(output_tail),
            potter_point_to_vrm1(source_hair.transform_point3(input_tail)),
        );
    }
    Ok(())
}

fn assert_v0_local_frames(project: &Path, glb: &Glb, v1_glb: &Glb) -> TestResult<()> {
    let root = &glb.json;
    let serialized_look_offset =
        object_vector(&root["extensions"]["VRM"]["firstPerson"]["firstPersonBoneOffset"])?;
    let loaded_look_offset = DVec3::new(
        serialized_look_offset.x,
        serialized_look_offset.y,
        -serialized_look_offset.z,
    );
    let vrm0_head = gltf_node_world_matrix(root, gltf_bone_index(root, "head"))?;
    let vrm0_root_rotation =
        DMat4::from_rotation_translation(DQuat::from_rotation_y(std::f64::consts::PI), DVec3::ZERO);
    let v1_root = &v1_glb.json;
    let v1_head = gltf_node_world_matrix(v1_root, gltf_bone_index(v1_root, "head"))?;
    let v1_look_offset = DVec3::from_array(json_numbers::<3>(
        &v1_root["extensions"]["VRMC_vrm"]["lookAt"]["offsetFromHeadBone"],
    )?);
    assert_point_close(
        (vrm0_root_rotation * vrm0_head).transform_point3(loaded_look_offset),
        v1_head.transform_point3(v1_look_offset),
    );

    let source_hair = evaluated_bone_matrix(project, "arm", "hair_root")?;
    let spring = &root["extensions"]["VRM"]["secondaryAnimation"];
    let collider = &spring["colliderGroups"][0];
    let collider_node = usize_field(&collider["node"]);
    let output_hair = gltf_node_world_matrix(root, collider_node)?;
    let serialized_offset = object_vector(&collider["colliders"][0]["offset"])?;
    let consumer_offset = DVec3::new(
        serialized_offset.x,
        serialized_offset.y,
        -serialized_offset.z,
    );
    assert_point_close(
        output_hair.transform_point3(consumer_offset),
        potter_point_to_vrm0(source_hair.transform_point3(DVec3::new(0.1, 0.2, 0.3))),
    );
    Ok(())
}

fn assert_gltf_avatar_axes(root: &Value, facing_sign: f64) -> TestResult<()> {
    let left = gltf_node_world_position(root, gltf_bone_index(root, "foot_l"))?;
    let right = gltf_node_world_position(root, gltf_bone_index(root, "foot_r"))?;
    let toe = gltf_node_world_position(root, gltf_bone_index(root, "toe_l"))?;
    assert!((left.x - right.x) * facing_sign > 0.1);
    assert!((toe.z - left.z) * facing_sign > 0.1);
    Ok(())
}

fn assert_positive_uniform_node_scales(root: &Value) -> TestResult<()> {
    for index in 0..root["nodes"]
        .as_array()
        .ok_or("glTF nodes are missing")?
        .len()
    {
        let scale = gltf_local_matrix(root, index)?
            .to_scale_rotation_translation()
            .0;
        assert!(scale.x.is_finite() && scale.y.is_finite() && scale.z.is_finite());
        assert!(scale.x > 0.0 && scale.y > 0.0 && scale.z > 0.0);
        assert!((scale.x - scale.y).abs() <= 1.0e-5);
        assert!((scale.y - scale.z).abs() <= 1.0e-5);
    }
    Ok(())
}

fn assert_source_avatar_axes(project: &Path) -> TestResult<()> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let data_id = document["nodes"]["arm"]["data"]
        .as_str()
        .ok_or("generated armature DataBlock is missing")?;
    let bones = &document["data_blocks"][data_id]["armature"]["bones"];
    let left_foot = &bones["foot_l"];
    let right_foot = &bones["foot_r"];
    let left_toe = &bones["toe_l"];
    assert!(json_numbers::<3>(&left_foot["head"])?[0] > json_numbers::<3>(&right_foot["head"])?[0]);
    assert!(json_numbers::<3>(&left_toe["head"])?[1] < json_numbers::<3>(&left_foot["head"])?[1]);
    assert!(json_numbers::<3>(&left_toe["tail"])?[1] < json_numbers::<3>(&left_toe["head"])?[1]);
    Ok(())
}

fn evaluated_bone_matrix(project: &Path, armature_id: &str, bone_id: &str) -> TestResult<DMat4> {
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate(&document, &EvaluationContext::default())?;
    let (_, bones) = snapshot
        .bone_matrices
        .iter()
        .find(|(id, _)| id.as_str() == armature_id)
        .ok_or("evaluated armature is missing")?;
    let (_, matrix) = bones
        .iter()
        .find(|(id, _)| id.as_str() == bone_id)
        .ok_or("evaluated bone is missing")?;
    Ok(DMat4::from_cols_array(matrix))
}

fn evaluated_bone_endpoint(
    project: &Path,
    armature_id: &str,
    bone_id: &str,
    length: f64,
) -> TestResult<DVec3> {
    Ok(evaluated_bone_matrix(project, armature_id, bone_id)?
        .transform_point3(DVec3::new(0.0, length, 0.0)))
}

fn accessor_matrix(glb: &Glb, accessor_index: usize) -> TestResult<Vec<Vec<f64>>> {
    let accessor = &glb.json["accessors"][accessor_index];
    let view = &glb.json["bufferViews"][usize_field(&accessor["bufferView"])];
    let component_type = accessor["componentType"]
        .as_u64()
        .ok_or("accessor componentType is missing")?;
    let components = match accessor["type"].as_str() {
        Some("SCALAR") => 1_usize,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        Some("MAT4") => 16,
        _ => return Err("unsupported accessor shape in VRM test".into()),
    };
    let component_size = match component_type {
        5121 => 1_usize,
        5123 => 2,
        5125 | 5126 => 4,
        _ => return Err("unsupported accessor component type in VRM test".into()),
    };
    let count = usize::try_from(
        accessor["count"]
            .as_u64()
            .ok_or("accessor count is missing")?,
    )?;
    let view_offset = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?;
    let accessor_offset = usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))?;
    let element_size = component_size
        .checked_mul(components)
        .ok_or("accessor element size overflow")?;
    let stride = usize::try_from(
        view["byteStride"]
            .as_u64()
            .unwrap_or(u64::try_from(element_size)?),
    )?;
    let base = view_offset
        .checked_add(accessor_offset)
        .ok_or("accessor offset overflow")?;
    let normalized = accessor["normalized"].as_bool().unwrap_or(false);
    let mut rows = Vec::with_capacity(count);
    for row in 0..count {
        let row_start = base
            .checked_add(row.checked_mul(stride).ok_or("accessor stride overflow")?)
            .ok_or("accessor data offset overflow")?;
        let mut values = Vec::with_capacity(components);
        for component in 0..components {
            let start = row_start
                .checked_add(
                    component
                        .checked_mul(component_size)
                        .ok_or("component offset overflow")?,
                )
                .ok_or("component data offset overflow")?;
            let value = match component_type {
                5121 => {
                    let value =
                        f64::from(*glb.binary.get(start).ok_or("accessor exceeds BIN chunk")?);
                    if normalized { value / 255.0 } else { value }
                }
                5123 => {
                    let bytes: [u8; 2] = glb
                        .binary
                        .get(start..start + 2)
                        .ok_or("accessor exceeds BIN chunk")?
                        .try_into()?;
                    let value = f64::from(u16::from_le_bytes(bytes));
                    if normalized { value / 65535.0 } else { value }
                }
                5125 => {
                    let bytes: [u8; 4] = glb
                        .binary
                        .get(start..start + 4)
                        .ok_or("accessor exceeds BIN chunk")?
                        .try_into()?;
                    f64::from(u32::from_le_bytes(bytes))
                }
                5126 => {
                    let bytes: [u8; 4] = glb
                        .binary
                        .get(start..start + 4)
                        .ok_or("accessor exceeds BIN chunk")?
                        .try_into()?;
                    f64::from(f32::from_le_bytes(bytes))
                }
                _ => unreachable!(),
            };
            values.push(value);
        }
        rows.push(values);
    }
    Ok(rows)
}

fn accessor_matrix_value(row: &[f64]) -> TestResult<DMat4> {
    if row.len() != 16 {
        return Err("inverse-bind accessor row is not MAT4".into());
    }
    let mut values = [0.0; 16];
    values.copy_from_slice(row);
    Ok(DMat4::from_cols_array(&values))
}

fn assert_skin_and_morph(glb: &Glb) -> TestResult<(usize, usize, usize)> {
    let root = &glb.json;
    let body_node = gltf_node_index(root, "body");
    let node = &root["nodes"][body_node];
    let mesh_index = usize_field(&node["mesh"]);
    let material_index = gltf_material_index(root, "skin_material");
    let mesh = &root["meshes"][mesh_index];
    let primitive = &mesh["primitives"][0];
    assert_eq!(usize_field(&primitive["material"]), material_index);

    let skin_index = usize_field(&node["skin"]);
    let skin = &root["skins"][skin_index];
    let hips_node = gltf_bone_index(root, "hips");
    let joints = skin["joints"]
        .as_array()
        .ok_or("skin joint list is missing")?;
    let hips_joint = joints
        .iter()
        .position(|joint| usize_field(joint) == hips_node)
        .ok_or("weighted hips bone is absent from the exported skin")?;
    let attributes = &primitive["attributes"];
    assert!(attributes["JOINTS_0"].is_number());
    assert!(attributes["WEIGHTS_0"].is_number());
    let joint_rows = accessor_matrix(glb, usize_field(&attributes["JOINTS_0"]))?;
    let weight_rows = accessor_matrix(glb, usize_field(&attributes["WEIGHTS_0"]))?;
    // UV seams split the 8 source cube vertices; every emitted vertex must stay hips-weighted.
    let vertex_count =
        usize_field(&root["accessors"][usize_field(&attributes["POSITION"])]["count"]);
    assert!(vertex_count >= 8);
    assert_eq!(joint_rows.len(), vertex_count);
    assert_eq!(weight_rows.len(), vertex_count);
    for (joint_row, weight_row) in joint_rows.iter().zip(&weight_rows) {
        assert_eq!(joint_row[0] as usize, hips_joint);
        close(&json!(weight_row[0]), 1.0);
        assert!(weight_row[1..].iter().all(|weight| weight.abs() <= 1.0e-6));
    }

    let target_names = mesh["extras"]["targetNames"]
        .as_array()
        .ok_or("exported morph target names are missing")?;
    let morph_index = target_names
        .iter()
        .position(|name| name.as_str() == Some("Smile"))
        .ok_or("exported Smile morph target is missing")?;
    let position_accessor = usize_field(&primitive["attributes"]["POSITION"]);
    let position_rows = accessor_matrix(glb, position_accessor)?;
    let inverse_bind_rows = accessor_matrix(glb, usize_field(&skin["inverseBindMatrices"]))?;
    assert_eq!(inverse_bind_rows.len(), joints.len());
    let mesh_world = gltf_node_world_matrix(root, body_node)?;
    for (vertex_index, joint_row) in joint_rows.iter().enumerate() {
        let joint_index = usize_field(&joints[joint_row[0] as usize]);
        let joint_world = gltf_node_world_matrix(root, joint_index)?;
        let inverse_bind = accessor_matrix_value(&inverse_bind_rows[joint_row[0] as usize])?;
        let position = DVec3::new(
            position_rows[vertex_index][0],
            position_rows[vertex_index][1],
            position_rows[vertex_index][2],
        );
        let actual = joint_world.transform_point3(inverse_bind.transform_point3(position));
        let expected = mesh_world.transform_point3(position);
        assert!(actual.distance(expected) <= 1.0e-4);
    }
    let morph_accessor = usize_field(&primitive["targets"][morph_index]["POSITION"]);
    let morph_rows = accessor_matrix(glb, morph_accessor)?;
    assert_eq!(morph_rows.len(), position_rows.len());
    let vertex_ids = mesh["extras"]["potter"]["vertex_ids"]
        .as_array()
        .ok_or("exported mesh vertex identities are missing")?;
    let moved_vertex = vertex_ids
        .iter()
        .position(|vertex_id| vertex_id.as_u64() == Some(0))
        .ok_or("shape key vertex identity was not exported")?;
    assert!(
        morph_rows[moved_vertex]
            .iter()
            .any(|component| component.abs() > 0.1)
    );
    assert!(
        morph_rows
            .iter()
            .flatten()
            .any(|component| component.abs() > 0.1)
    );
    Ok((body_node, mesh_index, material_index))
}
fn assert_scaled_primitive_preserved(base: &Glb, scaled: &Glb, node_id: &str) -> TestResult<()> {
    let base_root = &base.json;
    let scaled_root = &scaled.json;
    let base_node = gltf_node_index(base_root, node_id);
    let scaled_node = gltf_node_index(scaled_root, node_id);
    let base_mesh = usize_field(&base_root["nodes"][base_node]["mesh"]);
    let scaled_mesh = usize_field(&scaled_root["nodes"][scaled_node]["mesh"]);
    let base_primitive = &base_root["meshes"][base_mesh]["primitives"][0];
    let scaled_primitive = &scaled_root["meshes"][scaled_mesh]["primitives"][0];
    let base_positions =
        accessor_matrix(base, usize_field(&base_primitive["attributes"]["POSITION"]))?;
    let scaled_positions = accessor_matrix(
        scaled,
        usize_field(&scaled_primitive["attributes"]["POSITION"]),
    )?;
    assert_eq!(base_positions.len(), scaled_positions.len());
    let base_world = gltf_node_world_matrix(base_root, base_node)?;
    let scaled_world = gltf_node_world_matrix(scaled_root, scaled_node)?;
    for (base_position, scaled_position) in base_positions.iter().zip(&scaled_positions) {
        let scaled_source = DVec3::new(
            base_position[0] * 2.0,
            base_position[1] * 0.5,
            base_position[2],
        );
        let expected = base_world.transform_point3(scaled_source);
        let actual = scaled_world.transform_point3(DVec3::new(
            scaled_position[0],
            scaled_position[1],
            scaled_position[2],
        ));
        assert!(actual.distance(expected) <= 1.0e-4);
    }
    Ok(())
}

fn assert_v1_semantics(glb: &Glb, options: FixtureOptions) -> TestResult<()> {
    let root = &glb.json;
    assert_gltf_avatar_axes(root, 1.0)?;
    assert_positive_uniform_node_scales(root)?;
    assert_eq!(root["asset"]["version"], "2.0");
    let extensions_used = root["extensionsUsed"]
        .as_array()
        .ok_or("extensionsUsed is missing")?;
    for extension in ["VRMC_vrm", "VRMC_springBone", "VRMC_materials_mtoon"] {
        assert!(
            extensions_used
                .iter()
                .any(|value| value.as_str() == Some(extension))
        );
    }
    let vrm = &root["extensions"]["VRMC_vrm"];
    assert_eq!(vrm["specVersion"], "1.0");
    let (body_node, _mesh_index, material_index) = assert_skin_and_morph(glb)?;
    let body_copy_node = gltf_node_index(root, "body_copy");
    assert_ne!(body_node, body_copy_node);
    assert_eq!(
        usize_field(&root["nodes"][body_copy_node]["mesh"]),
        usize_field(&root["nodes"][body_node]["mesh"])
    );
    let bone_roles = [
        ("hips", "hips"),
        ("spine", "spine"),
        ("chest", "spine_001"),
        ("upperChest", "spine_002"),
        ("neck", "neck"),
        ("head", "head"),
        ("leftUpperArm", "upper_arm_l"),
        ("leftLowerArm", "forearm_l"),
        ("leftHand", "hand_l"),
        ("leftEye", "eye_l"),
        ("rightEye", "eye_r"),
        ("rightUpperArm", "upper_arm_r"),
        ("rightLowerArm", "forearm_r"),
        ("rightHand", "hand_r"),
        ("leftUpperLeg", "thigh_l"),
        ("leftLowerLeg", "shin_l"),
        ("leftFoot", "foot_l"),
        ("rightUpperLeg", "thigh_r"),
        ("rightLowerLeg", "shin_r"),
        ("rightFoot", "foot_r"),
        ("leftThumbMetacarpal", "thumb_meta_l"),
        ("leftThumbProximal", "thumb_prox_l"),
        ("leftThumbDistal", "thumb_distal_l"),
        ("rightThumbMetacarpal", "thumb_meta_r"),
        ("rightThumbProximal", "thumb_prox_r"),
        ("rightThumbDistal", "thumb_distal_r"),
    ];
    for (role, bone_id) in bone_roles {
        assert_eq!(
            usize_field(&vrm["humanoid"]["humanBones"][role]["node"]),
            gltf_bone_index(root, bone_id)
        );
    }
    assert_eq!(vrm["meta"]["name"], "Native VRM Avatar");
    assert_eq!(vrm["meta"]["licenseUrl"], "https://vrm.dev/licenses/1.0/");
    let thumbnail_index = gltf_image_index(root, "thumbnail");
    assert_eq!(usize_field(&vrm["meta"]["thumbnailImage"]), thumbnail_index);
    assert_eq!(root["images"][thumbnail_index]["mimeType"], "image/png");
    assert!(root["images"][thumbnail_index]["bufferView"].is_number());

    let happy = &vrm["expressions"]["preset"]["happy"];
    assert_eq!(happy["isBinary"], false);
    if !options.expression_texture_transform {
        assert_eq!(happy["overrideBlink"], "none");
    }
    let morph_index =
        root["meshes"][usize_field(&root["nodes"][body_node]["mesh"])]["extras"]["targetNames"]
            .as_array()
            .unwrap()
            .iter()
            .position(|name| name == "Smile")
            .unwrap();
    let morph_bind = &happy["morphTargetBinds"][0];
    assert_eq!(usize_field(&morph_bind["node"]), body_node);
    assert_eq!(usize_field(&morph_bind["index"]), morph_index);
    close(&morph_bind["weight"], 0.4);
    let color_bind = &happy["materialColorBinds"][0];
    assert_eq!(usize_field(&color_bind["material"]), material_index);
    assert_eq!(color_bind["type"], "color");
    assert_eq!(color_bind["targetValue"], json!([0.2, 0.4, 0.6, 0.8]));
    let custom = &vrm["expressions"]["custom"]["winkExtra"];
    assert_eq!(custom["isBinary"], true);
    assert_eq!(
        usize_field(&custom["morphTargetBinds"][0]["node"]),
        body_node
    );
    if options.expression_look_at {
        let names =
            root["meshes"][usize_field(&root["nodes"][body_node]["mesh"])]["extras"]["targetNames"]
                .as_array()
                .ok_or("exported look-at morph names are missing")?;
        for (preset, target_name) in [
            ("lookLeft", "Look Left"),
            ("lookRight", "Look Right"),
            ("lookUp", "Look Up"),
            ("lookDown", "Look Down"),
        ] {
            let target_index = names
                .iter()
                .position(|name| name.as_str() == Some(target_name))
                .ok_or("a look-at morph target is missing")?;
            let bind = &vrm["expressions"]["preset"][preset]["morphTargetBinds"][0];
            assert_eq!(usize_field(&bind["node"]), body_node);
            assert_eq!(usize_field(&bind["index"]), target_index);
            close(&bind["weight"], 0.65);
        }
    }
    if options.expression_texture_transform {
        let transform = &happy["textureTransformBinds"][0];
        assert_eq!(usize_field(&transform["material"]), material_index);
        assert_eq!(transform["scale"], json!([1.25, 0.75]));
        assert_eq!(transform["offset"], json!([0.1, 0.2]));
        assert_eq!(happy["overrideBlink"], "block");
        assert_eq!(happy["overrideMouth"], "blend");
    }

    let annotation = &vrm["firstPerson"]["meshAnnotations"][0];
    assert_eq!(usize_field(&annotation["node"]), body_node);
    assert_eq!(annotation["type"], "firstPersonOnly");
    let look_at = &vrm["lookAt"];
    assert_eq!(
        look_at["type"],
        if options.expression_look_at {
            "expression"
        } else {
            "bone"
        }
    );
    close(&look_at["rangeMapHorizontalInner"]["inputMaxValue"], 30.0);
    close(&look_at["rangeMapHorizontalInner"]["outputScale"], 0.25);
    close(&look_at["rangeMapVerticalUp"]["inputMaxValue"], 60.0);
    close(&look_at["rangeMapVerticalUp"]["outputScale"], 1.0);
    close(&look_at["rangeMapHorizontalOuter"]["inputMaxValue"], 40.0);
    close(&look_at["rangeMapHorizontalOuter"]["outputScale"], 0.5);
    close(&look_at["rangeMapVerticalDown"]["inputMaxValue"], 50.0);
    close(&look_at["rangeMapVerticalDown"]["outputScale"], 0.75);

    let spring = &root["extensions"]["VRMC_springBone"];
    assert_eq!(spring["specVersion"], "1.0");
    let collider = &spring["colliders"][0];
    assert_eq!(
        usize_field(&collider["node"]),
        gltf_bone_index(root, "hair_root")
    );
    if options.capsule {
        close(&collider["shape"]["capsule"]["radius"], 0.08);
    } else {
        close(&collider["shape"]["sphere"]["radius"], 0.08);
    }
    assert_eq!(spring["colliderGroups"][0]["colliders"], json!([0]));
    let spring_chain = &spring["springs"][0];
    assert_eq!(
        usize_field(&spring_chain["center"]),
        gltf_bone_index(root, "head")
    );
    assert_eq!(
        usize_field(&spring_chain["joints"][0]["node"]),
        gltf_bone_index(root, "hair_root")
    );
    assert_eq!(
        usize_field(&spring_chain["joints"][1]["node"]),
        gltf_bone_index(root, "hair_tip")
    );
    assert_vector(&spring_chain["joints"][0]["gravityDir"], [1.0, 3.0, -2.0]);
    close(&spring_chain["joints"][0]["stiffness"], 0.8);
    close(
        &spring_chain["joints"][1]["stiffness"],
        if options.uneven_spring_joints {
            0.4
        } else {
            0.8
        },
    );

    let material = &root["materials"][material_index];
    assert_texture_ref(
        root,
        &material["pbrMetallicRoughness"]["baseColorTexture"],
        "tex_albedo",
    );
    let mtoon = &material["extensions"]["VRMC_materials_mtoon"];
    assert_eq!(mtoon["specVersion"], "1.0");
    close(&mtoon["shadingToonyFactor"], 0.7);
    close(&mtoon["shadingShiftFactor"], -0.15);
    if options.v1_extended_mtoon {
        close(&mtoon["shadingShiftTexture"]["scale"], 0.4);
        assert_texture_ref(root, &mtoon["shadingShiftTexture"], "tex_shift");
        assert_eq!(mtoon["matcapFactor"], json!([0.8, 0.7, 0.6]));
    } else {
        assert!(mtoon.get("shadingShiftTexture").is_none_or(Value::is_null));
        if !mtoon["matcapFactor"].is_null() {
            assert_eq!(mtoon["matcapFactor"], json!([1.0, 1.0, 1.0]));
        }
    }
    assert_eq!(mtoon["outlineWidthMode"], "worldCoordinates");
    close(&mtoon["outlineWidthFactor"], 0.01);
    close(
        &json!(
            mtoon["uvAnimationScrollXSpeedFactor"]
                .as_f64()
                .unwrap_or(0.0)
        ),
        if options.uv_rotation_only { 0.0 } else { 0.2 },
    );
    close(
        &json!(
            mtoon["uvAnimationScrollYSpeedFactor"]
                .as_f64()
                .unwrap_or(0.0)
        ),
        if options.uv_rotation_only { 0.0 } else { -0.1 },
    );
    close(
        &json!(
            mtoon["uvAnimationRotationSpeedFactor"]
                .as_f64()
                .unwrap_or(0.0)
        ),
        if options.uv_rotation_only || options.uv_scroll_and_rotation {
            0.3
        } else {
            0.0
        },
    );
    for (field, image_id) in [
        ("shadeMultiplyTexture", "tex_shade"),
        ("matcapTexture", "tex_outline"),
        ("rimMultiplyTexture", "tex_rim"),
        ("outlineWidthMultiplyTexture", "tex_outline"),
        ("uvAnimationMaskTexture", "tex_uv_mask"),
    ] {
        assert_texture_ref(root, &mtoon[field], image_id);
    }
    Ok(())
}

fn assert_v0_semantics(glb: &Glb, options: FixtureOptions) -> TestResult<()> {
    let root = &glb.json;
    assert_gltf_avatar_axes(root, -1.0)?;
    let extensions_used = root["extensionsUsed"]
        .as_array()
        .ok_or("extensionsUsed is missing")?;
    assert!(
        extensions_used
            .iter()
            .any(|value| value.as_str() == Some("VRM"))
    );
    let vrm = &root["extensions"]["VRM"];
    let (body_node, mesh_index, material_index) = assert_skin_and_morph(glb)?;
    assert_ne!(body_node, mesh_index);
    let body_copy_node = gltf_node_index(root, "body_copy");
    let body_copy_mesh = usize_field(&root["nodes"][body_copy_node]["mesh"]);
    assert_ne!(body_node, body_copy_node);
    assert_ne!(mesh_index, body_copy_mesh);
    let original_primitive = &root["meshes"][mesh_index]["primitives"][0];
    let copied_primitive = &root["meshes"][body_copy_mesh]["primitives"][0];
    assert_eq!(
        original_primitive["attributes"],
        copied_primitive["attributes"]
    );
    assert_eq!(original_primitive["indices"], copied_primitive["indices"]);
    assert_eq!(original_primitive["targets"], copied_primitive["targets"]);
    let bone_roles = [
        ("hips", "hips"),
        ("spine", "spine"),
        ("chest", "spine_001"),
        ("upperChest", "spine_002"),
        ("neck", "neck"),
        ("head", "head"),
        ("leftUpperArm", "upper_arm_l"),
        ("leftLowerArm", "forearm_l"),
        ("leftHand", "hand_l"),
        ("rightUpperArm", "upper_arm_r"),
        ("rightLowerArm", "forearm_r"),
        ("rightHand", "hand_r"),
        ("leftEye", "eye_l"),
        ("rightEye", "eye_r"),
        ("leftUpperLeg", "thigh_l"),
        ("leftLowerLeg", "shin_l"),
        ("leftFoot", "foot_l"),
        ("rightUpperLeg", "thigh_r"),
        ("rightLowerLeg", "shin_r"),
        ("rightFoot", "foot_r"),
    ];
    for (role, bone_id) in bone_roles {
        assert_eq!(legacy_bone_node(root, role), gltf_bone_index(root, bone_id));
    }
    assert_eq!(
        legacy_bone_node(root, "leftThumbProximal"),
        gltf_bone_index(root, "thumb_meta_l")
    );
    assert_eq!(
        legacy_bone_node(root, "leftThumbIntermediate"),
        gltf_bone_index(root, "thumb_prox_l")
    );
    assert_eq!(
        legacy_bone_node(root, "leftThumbDistal"),
        gltf_bone_index(root, "thumb_distal_l")
    );
    assert_eq!(
        legacy_bone_node(root, "rightThumbProximal"),
        gltf_bone_index(root, "thumb_meta_r")
    );
    assert_eq!(
        legacy_bone_node(root, "rightThumbIntermediate"),
        gltf_bone_index(root, "thumb_prox_r")
    );
    assert_eq!(
        legacy_bone_node(root, "rightThumbDistal"),
        gltf_bone_index(root, "thumb_distal_r")
    );

    let meta = &vrm["meta"];
    assert_eq!(meta["title"], "Legacy Fixture Avatar");
    assert_eq!(meta["author"], "Potter VRM Test");
    assert_eq!(meta["allowedUserName"], "OnlyAuthor");
    assert_eq!(meta["violentUssageName"], "Disallow");
    assert_eq!(meta["sexualUssageName"], "Disallow");
    assert_eq!(meta["commercialUssageName"], "Disallow");
    assert_eq!(meta["licenseName"], "Redistribution_Prohibited");
    let thumbnail_index = usize_field(&meta["texture"]);
    assert_eq!(texture_image_id(root, thumbnail_index), "thumbnail");

    let groups = vrm["blendShapeMaster"]["blendShapeGroups"]
        .as_array()
        .ok_or("legacy expression groups are missing")?;
    assert!(groups.iter().all(|group| {
        group["binds"].as_array().is_none_or(|binds| {
            binds
                .iter()
                .all(|bind| usize_field(&bind["mesh"]) != body_copy_mesh)
        })
    }));
    let happy = groups
        .iter()
        .find(|group| group["presetName"] == "joy")
        .ok_or("legacy joy preset is missing")?;
    let mesh = &root["meshes"][mesh_index];
    assert_eq!(original_primitive["material"], copied_primitive["material"]);
    let morph_index = mesh["extras"]["targetNames"]
        .as_array()
        .unwrap()
        .iter()
        .position(|name| name == "Smile")
        .unwrap();
    assert_eq!(usize_field(&happy["binds"][0]["mesh"]), mesh_index);
    assert_eq!(usize_field(&happy["binds"][0]["index"]), morph_index);
    close(&happy["binds"][0]["weight"], 40.0);
    let material_value = &happy["materialValues"][0];
    assert_eq!(
        material_value["materialName"],
        root["materials"][material_index]["name"]
    );
    assert_eq!(material_value["propertyName"], "_Color");
    assert_eq!(material_value["targetValue"], json!([0.2, 0.4, 0.6, 0.8]));
    let custom = groups
        .iter()
        .find(|group| group["name"] == "winkExtra")
        .ok_or("legacy custom expression is missing")?;
    assert_eq!(custom["presetName"], "unknown");
    assert_eq!(custom["isBinary"], true);
    assert_eq!(usize_field(&custom["binds"][0]["mesh"]), mesh_index);

    let first_person = &vrm["firstPerson"];
    assert_eq!(
        usize_field(&first_person["firstPersonBone"]),
        gltf_bone_index(root, "head")
    );
    let annotation = &first_person["meshAnnotations"][0];
    assert_eq!(usize_field(&annotation["mesh"]), mesh_index);
    assert_ne!(mesh_index, body_copy_mesh);
    assert!(
        first_person["meshAnnotations"]
            .as_array()
            .is_some_and(|annotations| annotations
                .iter()
                .all(|annotation| { usize_field(&annotation["mesh"]) != body_copy_mesh }))
    );
    assert_eq!(annotation["firstPersonFlag"], "FirstPersonOnly");
    assert_eq!(first_person["lookAtTypeName"], "Bone");
    close(&first_person["lookAtHorizontalInner"]["xRange"], 30.0);
    close(&first_person["lookAtHorizontalInner"]["yRange"], 0.25);
    close(&first_person["lookAtVerticalUp"]["xRange"], 60.0);
    close(&first_person["lookAtVerticalUp"]["yRange"], 1.0);
    close(&first_person["lookAtHorizontalOuter"]["xRange"], 40.0);
    close(&first_person["lookAtHorizontalOuter"]["yRange"], 0.5);
    close(&first_person["lookAtVerticalDown"]["xRange"], 50.0);
    close(&first_person["lookAtVerticalDown"]["yRange"], 0.75);

    let secondary = &vrm["secondaryAnimation"];
    let collider_group = &secondary["colliderGroups"][0];
    assert_eq!(
        usize_field(&collider_group["node"]),
        gltf_bone_index(root, "hair_root")
    );
    close(&collider_group["colliders"][0]["radius"], 0.08);
    let spring = &secondary["boneGroups"][0];
    assert_eq!(spring["bones"], json!([gltf_bone_index(root, "hair_root")]));
    assert_eq!(
        usize_field(&spring["center"]),
        gltf_bone_index(root, "head")
    );
    assert_eq!(spring["colliderGroups"], json!([0]));
    close(&spring["stiffiness"], 0.8);
    close(&spring["gravityPower"], 1.1);
    close(&spring["dragForce"], 0.2);
    close(&spring["hitRadius"], 0.02);
    // VRM 0.x gravityDir is z-reflect(Rot_Y180(glTF dir)): VRM 1.0 [1,3,-2] becomes [-1,3,-2].
    close(&spring["gravityDir"]["x"], -1.0);
    close(&spring["gravityDir"]["y"], 3.0);
    close(&spring["gravityDir"]["z"], -2.0);

    let material = &root["materials"][material_index];
    assert_texture_ref(
        root,
        &material["pbrMetallicRoughness"]["baseColorTexture"],
        "tex_albedo",
    );
    let properties = &vrm["materialProperties"][material_index];
    assert_eq!(properties["shader"], "VRM/MToon");
    let float_properties = &properties["floatProperties"];
    close(&float_properties["_OutlineWidth"], 1.0);
    close(&float_properties["_OutlineWidthMode"], 1.0);
    close(&float_properties["_OutlineColorMode"], 1.0);
    close(&float_properties["_OutlineLightingMix"], 0.5);
    close(&float_properties["_ShadeShift"], -0.15);
    close(&float_properties["_ShadeToony"], 11.0 / 23.0);
    close(
        &json!(float_properties["_UvAnimScrollX"].as_f64().unwrap_or(0.0)),
        if options.uv_rotation_only { 0.0 } else { 0.2 },
    );
    close(
        &json!(float_properties["_UvAnimScrollY"].as_f64().unwrap_or(0.0)),
        if options.uv_rotation_only { 0.0 } else { 0.1 },
    );
    close(
        &json!(float_properties["_UvAnimRotation"].as_f64().unwrap_or(0.0)),
        if options.uv_rotation_only || options.uv_scroll_and_rotation {
            0.3 / std::f64::consts::TAU
        } else {
            0.0
        },
    );
    let shade_color = properties["vectorProperties"]["_ShadeColor"]
        .as_array()
        .ok_or("legacy MToon shade color is missing")?;
    assert_eq!(shade_color.len(), 4);
    for (index, linear) in [0.3_f64, 0.35, 0.4].into_iter().enumerate() {
        close(&shade_color[index], linear.powf(1.0 / 2.2));
    }
    close(&shade_color[3], 1.0);
    let texture_properties = &properties["textureProperties"];
    for (field, image_id) in [
        ("_MainTex", "tex_albedo"),
        ("_ShadeTexture", "tex_shade"),
        ("_SphereAdd", "tex_outline"),
        ("_RimTexture", "tex_rim"),
    ] {
        assert_eq!(
            texture_image_id(root, usize_field(&texture_properties[field])),
            image_id
        );
    }
    Ok(())
}

#[test]
fn vrm0_and_vrm1_export_actual_scene_indices_and_preserve_avatar_semantics() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    assert_source_avatar_axes(&project)?;
    assert_eq!(revision(&project)?, 1);
    let vrm1_path = directory.path().join("avatar-v1.vrm");
    let vrm0_path = directory.path().join("avatar-v0.vrm");
    export(&project, "vrm", &vrm1_path)?;
    export(&project, "vrm0", &vrm0_path)?;
    let v1 = parse_glb(&vrm1_path)?;
    let v0 = parse_glb(&vrm0_path)?;
    assert_v1_semantics(&v1, FixtureOptions::default())?;
    assert_v0_semantics(&v0, FixtureOptions::default())?;
    assert_v1_local_frames(&project, &v1, FixtureOptions::default())?;
    assert_v0_local_frames(&project, &v0, &v1)?;
    assert_v0_mtoon_channel_remaps(&v1, &v0)?;
    Ok(())
}

#[test]
fn vrm1_preserves_capsules_per_joint_settings_and_texture_transform_expressions() -> TestResult {
    let options = FixtureOptions {
        capsule: true,
        uneven_spring_joints: true,
        expression_texture_transform: true,
        modern_extra_permissions: true,
        expression_look_at: true,
        v1_extended_mtoon: true,
        ..FixtureOptions::default()
    };
    let (directory, project) = new_fixture(options)?;
    let output = directory.path().join("full-vrm1.vrm");
    export(&project, "vrm", &output)?;
    let glb = parse_glb(&output)?;
    assert_v1_semantics(&glb, options)?;
    assert_v1_local_frames(&project, &glb, options)?;
    assert_eq!(
        glb.json["extensions"]["VRMC_vrm"]["meta"]["allowPoliticalOrReligiousUsage"],
        true
    );
    assert_eq!(
        glb.json["extensions"]["VRMC_vrm"]["meta"]["allowAntisocialOrHateUsage"],
        true
    );
    Ok(())
}

#[test]
fn vrm_expressions_target_every_material_uv_variant() -> TestResult<()> {
    let options = FixtureOptions {
        expression_texture_transform: true,
        ..FixtureOptions::default()
    };
    let (directory, project) = new_fixture(options)?;
    apply_success(
        &project,
        revision(&project)?,
        &[
            json!({
                "op": "node.create",
                "id": "uv_variant",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "skin_material"
            }),
            json!({"op": "uv.layer_create", "target": {"id": "uv_variant"}, "name": "BaseUV"}),
            json!({"op": "uv.layer_create", "target": {"id": "uv_variant"}, "name": "VariantUV"}),
            json!({"op": "uv.layer_set_render", "target": {"id": "uv_variant"}, "name": "VariantUV"}),
        ],
    )?;
    let material_for_node = |root: &Value, id: &str| -> TestResult<usize> {
        let node = gltf_node_index(root, id);
        let mesh = usize_field(&root["nodes"][node]["mesh"]);
        Ok(usize_field(
            &root["meshes"][mesh]["primitives"][0]["material"],
        ))
    };

    let v1_path = directory.path().join("material-variants.vrm");
    export(&project, "vrm", &v1_path)?;
    let v1 = parse_glb(&v1_path)?;
    let v1_base = material_for_node(&v1.json, "body")?;
    let v1_variant = material_for_node(&v1.json, "uv_variant")?;
    assert_ne!(v1_base, v1_variant);
    assert_eq!(
        v1.json["materials"][v1_variant]["extensions"]["VRMC_materials_mtoon"]["shadeMultiplyTexture"]
            ["texCoord"],
        1
    );
    let happy = &v1.json["extensions"]["VRMC_vrm"]["expressions"]["preset"]["happy"];
    let material_indices = [v1_base, v1_variant];
    let color_binds = happy["materialColorBinds"]
        .as_array()
        .ok_or("VRM 1 material color binds are missing")?;
    assert_eq!(color_binds.len(), 2);
    for (bind, index) in color_binds.iter().zip(material_indices) {
        assert_eq!(usize_field(&bind["material"]), index);
        assert_eq!(bind["targetValue"], json!([0.2, 0.4, 0.6, 0.8]));
    }
    let transform_binds = happy["textureTransformBinds"]
        .as_array()
        .ok_or("VRM 1 texture transform binds are missing")?;
    assert_eq!(transform_binds.len(), 2);
    for (bind, index) in transform_binds.iter().zip(material_indices) {
        assert_eq!(usize_field(&bind["material"]), index);
        assert_eq!(bind["scale"], json!([1.25, 0.75]));
        assert_eq!(bind["offset"], json!([0.1, 0.2]));
    }

    // VRM 0.x has no procedural expression overrides; retain the texture binds while making this export fixture valid.
    let mut v0_expressions = avatar_settings(options)["expressions"].clone();
    v0_expressions["happy"]["override_blink"] = json!("none");
    v0_expressions["happy"]["override_mouth"] = json!("none");
    apply_success(
        &project,
        revision(&project)?,
        &[json!({"op": "avatar.update", "set": {"expressions": v0_expressions}})],
    )?;

    let v0_path = directory.path().join("material-variants-v0.vrm");
    let v0_output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "vrm0", "--out"])
        .arg(&v0_path)
        .args(["--allow-lossy", "--overwrite", "--json"])
        .output()?;
    assert!(
        v0_output.status.success(),
        "lossy VRM0 UV variant export failed: stdout={} stderr={}",
        String::from_utf8_lossy(&v0_output.stdout),
        String::from_utf8_lossy(&v0_output.stderr)
    );
    let v0 = parse_glb(&v0_path)?;
    let v0_base = material_for_node(&v0.json, "body")?;
    let v0_variant = material_for_node(&v0.json, "uv_variant")?;
    assert_ne!(v0_base, v0_variant);
    let source_name = v0.json["materials"][v0_base]["name"]
        .as_str()
        .ok_or("VRM 0 source material name is missing")?;
    let variant_name = v0.json["materials"][v0_variant]["name"]
        .as_str()
        .ok_or("VRM 0 variant material name is missing")?;
    assert_ne!(source_name, variant_name);
    let properties = &v0.json["extensions"]["VRM"]["materialProperties"];
    assert_eq!(properties[v0_base]["name"].as_str(), Some(source_name));
    assert_eq!(properties[v0_variant]["name"].as_str(), Some(variant_name));
    let mut variant_properties = properties[v0_variant].clone();
    variant_properties["name"] = json!(source_name);
    assert_eq!(properties[v0_base], variant_properties);
    let groups = v0.json["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        .as_array()
        .ok_or("VRM 0 expressions are missing")?;
    let happy = groups
        .iter()
        .find(|group| group["presetName"] == "joy")
        .ok_or("VRM 0 joy expression is missing")?;
    let material_values = happy["materialValues"]
        .as_array()
        .ok_or("VRM 0 material values are missing")?;
    assert_eq!(material_values.len(), 4);
    let source_values = material_values
        .iter()
        .filter(|value| value["materialName"].as_str() == Some(source_name))
        .collect::<Vec<_>>();
    let variant_values = material_values
        .iter()
        .filter(|value| value["materialName"].as_str() == Some(variant_name))
        .collect::<Vec<_>>();
    assert_eq!(source_values.len(), 2);
    assert_eq!(variant_values.len(), 2);
    for (source, variant) in source_values.into_iter().zip(variant_values) {
        assert_eq!(source["propertyName"], variant["propertyName"]);
        assert_eq!(source["targetValue"], variant["targetValue"]);
    }
    let (_roundtrip_directory, imported_project) = init_project()?;
    let imported = pot()
        .arg("import")
        .arg(&imported_project)
        .arg("--file")
        .arg(&v0_path)
        .args([
            "--format",
            "vrm0",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ])
        .output()?;
    assert!(
        imported.status.success(),
        "VRM0 variant round-trip import failed: stdout={} stderr={}",
        String::from_utf8_lossy(&imported.stdout),
        String::from_utf8_lossy(&imported.stderr)
    );
    let imported_avatar = scene_avatar(&imported_project)?;
    let imported_happy = &imported_avatar["expressions"]["happy"];
    let imported_color_binds = imported_happy["material_color_binds"]
        .as_array()
        .ok_or("VRM0 round-trip material color binds are missing")?;
    assert_eq!(imported_color_binds.len(), 2);
    assert_ne!(
        imported_color_binds[0]["material"],
        imported_color_binds[1]["material"]
    );
    assert!(
        imported_color_binds
            .iter()
            .all(|bind| bind["target_value"] == json!([0.2, 0.4, 0.6, 0.8]))
    );
    let imported_transform_binds = imported_happy["texture_transform_binds"]
        .as_array()
        .ok_or("VRM0 round-trip texture transform binds are missing")?;
    assert_eq!(imported_transform_binds.len(), 2);
    assert_ne!(
        imported_transform_binds[0]["material"],
        imported_transform_binds[1]["material"]
    );
    let exported_st = material_values
        .iter()
        .filter(|value| value["propertyName"] == "_MainTex_ST")
        .map(|value| (&value["materialName"], &value["targetValue"]))
        .collect::<Vec<_>>();
    eprintln!(
        "VRM0 texture transform round trip: input={:?}, exported_st={exported_st:?}, imported={imported_transform_binds:?}",
        avatar_settings(options)["expressions"]["happy"]["texture_transform_binds"]
    );
    assert!(imported_transform_binds
        .iter()
        .all(|bind| bind["scale"] == json!([1.25, 0.75]) && bind["offset"] == json!([0.1, 0.2])));
    Ok(())
}

#[test]
fn vrm0_rejects_procedural_expression_overrides_even_when_lossy() -> TestResult<()> {
    let (directory, project) = new_fixture(FixtureOptions {
        expression_texture_transform: true,
        ..FixtureOptions::default()
    })?;
    let output_path = directory.path().join("expression-overrides-v0.vrm");
    let output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "vrm0", "--out"])
        .arg(&output_path)
        .args(["--allow-lossy", "--json"])
        .output()?;
    let failure = assert_failure(&output, "UNSUPPORTED_FEATURE")?;
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "vrm.v0.expression_override"
    );
    assert_eq!(failure["error"]["details"]["status"], "not_supported");
    assert!(!output_path.exists());
    Ok(())
}

#[test]
fn avatar_updates_preserve_omitted_components_and_clear_all_settings() -> TestResult {
    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    assert_eq!(revision(&project)?, 1);
    apply_success(
        &project,
        1,
        &[json!({"op": "avatar.update", "set": {"first_person": null}})],
    )?;
    assert_eq!(revision(&project)?, 2);
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_eq!(document["avatar"]["meta"]["name"], "Native VRM Avatar");
    assert!(document["avatar"]["expressions"]["happy"].is_object());
    assert!(document["avatar"]["look_at"].is_object());
    assert!(
        document["avatar"]
            .get("first_person")
            .is_none_or(Value::is_null)
    );

    apply_success(&project, 2, &[json!({"op": "avatar.clear"})])?;
    assert_eq!(revision(&project)?, 3);
    let cleared: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert!(cleared.get("avatar").is_none());
    Ok(())
}

#[test]
fn invalid_avatar_settings_and_reference_deletions_are_atomic_and_diagnostic() -> TestResult {
    let (empty_directory, empty_project) = init_project()?;
    let missing_required = json!({
        "op": "avatar.update",
        "set": {"meta": {"name": "Incomplete", "authors": ["Author"], "license_url": "https://example.test/license"}},
    });
    assert_operation_failure(&apply_batch(&empty_project, 0, &[missing_required])?)?;
    assert_eq!(revision(&empty_project)?, 0);
    drop(empty_directory);

    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    let base_revision = revision(&project)?;
    let settings = avatar_settings(FixtureOptions::default());

    let mut duplicate_map = humanoid();
    duplicate_map["rightUpperArm"] = json!("upper_arm_l");
    let duplicate = json!({"op": "avatar.update", "set": {"humanoid": duplicate_map}});

    let mut wrong_hierarchy = humanoid();
    wrong_hierarchy["leftLowerArm"] = json!("root");
    let hierarchy = json!({"op": "avatar.update", "set": {"humanoid": wrong_hierarchy}});
    let mut missing_optional_parent_map = humanoid();
    assert!(
        missing_optional_parent_map
            .as_object_mut()
            .unwrap()
            .remove("chest")
            .is_some()
    );
    let missing_optional_parent = json!({
        "op": "avatar.update",
        "set": {"humanoid": missing_optional_parent_map},
    });

    let mut branch_arm_map = humanoid();
    branch_arm_map["leftShoulder"] = json!("branch_shoulder_l");
    branch_arm_map["leftUpperArm"] = json!("branch_upper_arm_l");
    branch_arm_map["leftLowerArm"] = json!("branch_forearm_l");
    branch_arm_map["leftHand"] = json!("branch_hand_l");
    let branch_under_spine = json!({
        "op": "avatar.update",
        "set": {"humanoid": branch_arm_map},
    });

    let mut missing_thumb_parent_map = humanoid();
    assert!(
        missing_thumb_parent_map
            .as_object_mut()
            .unwrap()
            .remove("leftThumbProximal")
            .is_some()
    );
    let missing_thumb_parent = json!({
        "op": "avatar.update",
        "set": {"humanoid": missing_thumb_parent_map},
    });

    let mut shade_color_expressions = settings["expressions"].clone();
    shade_color_expressions["happy"]["material_color_binds"][0]["color_type"] = json!("shadeColor");
    let shade_color_without_mtoon = json!({
        "op": "avatar.update",
        "set": {
            "expressions": shade_color_expressions,
            "mtoon": null,
        },
    });

    let mut expressions = settings["expressions"].clone();
    expressions["happy"]["morph_target_binds"][0]["shape_key"] = json!("missing_shape");
    let missing_reference = json!({"op": "avatar.update", "set": {"expressions": expressions}});

    let mut look_at = settings["look_at"].clone();
    look_at["range_map_horizontal_inner"]["input_max_value"] = json!(-1.0);
    let range = json!({"op": "avatar.update", "set": {"look_at": look_at}});

    let mut repeated_chains = settings["spring_bone"].clone();
    let repeated_chain = repeated_chains["springs"][0].clone();
    repeated_chains["springs"]
        .as_array_mut()
        .unwrap()
        .push(repeated_chain);
    let duplicate_spring_chains = json!({
        "op": "avatar.update",
        "set": {"spring_bone": repeated_chains},
    });

    let mut invalid_center_settings = settings["spring_bone"].clone();
    invalid_center_settings["springs"][0]["center"] = json!({"node": "arm", "bone": "hair_tip"});
    let invalid_spring_center = json!({
        "op": "avatar.update",
        "set": {"spring_bone": invalid_center_settings},
    });
    let mut shared_skipped_chain_settings = settings["spring_bone"].clone();
    let mut inner_chain = shared_skipped_chain_settings["springs"][0].clone();
    inner_chain["name"] = json!("Hair Inner");
    inner_chain["center"] = json!({"node": "arm", "bone": "skip_middle_l"});
    inner_chain["joints"][0]["node"] = json!({"node": "arm", "bone": "skip_middle_l"});
    inner_chain["joints"][1]["node"] = json!({"node": "arm", "bone": "skip_middle_r"});
    let mut outer_chain = inner_chain.clone();
    outer_chain["name"] = json!("Hair Outer");
    outer_chain["center"] = json!({"node": "arm", "bone": "skip_start"});
    outer_chain["joints"][0]["node"] = json!({"node": "arm", "bone": "skip_start"});
    outer_chain["joints"][1]["node"] = json!({"node": "arm", "bone": "skip_end"});
    shared_skipped_chain_settings["springs"]
        .as_array_mut()
        .unwrap()
        .extend([inner_chain, outer_chain]);
    let duplicate_skipped_spring_nodes = json!({
        "op": "avatar.update",
        "set": {"spring_bone": shared_skipped_chain_settings},
    });
    for (name, operation) in [
        ("duplicate humanoid bone", duplicate),
        ("incorrect humanoid hierarchy", hierarchy),
        (
            "optional humanoid parent is missing",
            missing_optional_parent,
        ),
        ("arm chain branches below mapped spine", branch_under_spine),
        (
            "thumb distal bone lacks mapped proximal parent",
            missing_thumb_parent,
        ),
        ("duplicate spring chain", duplicate_spring_chains),
        (
            "chains share a skipped intermediate spring bone",
            duplicate_skipped_spring_nodes,
        ),
        (
            "spring center is below its first joint",
            invalid_spring_center,
        ),
        (
            "shadeColor bind targets a non-MToon material",
            shade_color_without_mtoon,
        ),
        ("missing shape-key reference", missing_reference),
        ("out-of-range look-at input", range),
    ] {
        let output = apply_batch(&project, base_revision, &[operation])?;
        assert_operation_failure(&output)?;
        assert_eq!(
            revision(&project)?,
            base_revision,
            "{name} advanced the scene revision"
        );
    }

    let scene_before_invalid_references = fs::read(project.join("scene.json"))?;
    let moved_humanoid_parent = apply_batch(
        &project,
        base_revision,
        &[json!({
            "op": "bone.update",
            "target": {"id": "arm"},
            "bone_id": "thigh_l",
            "set": {"parent": "spine"},
        })],
    )?;
    assert!(!moved_humanoid_parent.status.success());
    let moved_parent_error = cli_json(&moved_humanoid_parent)?;
    assert!(matches!(
        moved_parent_error["error"]["code"].as_str(),
        Some("INVALID_OPERATION" | "SCENE_INVALID")
    ));
    assert!(
        moved_parent_error["error"]["details"]
            .as_object()
            .is_some_and(|details| !details.is_empty())
    );
    assert_eq!(revision(&project)?, base_revision);
    assert_eq!(
        fs::read(project.join("scene.json"))?,
        scene_before_invalid_references
    );

    let deleted_reference = apply_batch(
        &project,
        base_revision,
        &[json!({"op": "node.delete", "target": {"id": "body"}})],
    )?;
    assert_operation_failure(&deleted_reference)?;
    assert_eq!(revision(&project)?, base_revision);
    assert_eq!(
        fs::read(project.join("scene.json"))?,
        scene_before_invalid_references
    );
    Ok(())
}

#[test]
fn vrm0_rejects_capsules_nonuniform_joint_settings_and_unmapped_permissions() -> TestResult {
    let (capsule_dir, capsule_project) = new_fixture(FixtureOptions::default())?;
    let capsule_spring = spring_bone(FixtureOptions {
        capsule: true,
        ..FixtureOptions::default()
    });
    apply_success(
        &capsule_project,
        1,
        &[json!({"op": "avatar.update", "set": {"spring_bone": capsule_spring}})],
    )?;
    let current_revision = revision(&capsule_project)?;
    assert_validate_export_preflight(
        &capsule_project,
        "vrm0",
        &capsule_dir.path().join("capsule-v0.vrm"),
        "vrm.v0.spring_capsule",
    )?;
    assert_eq!(revision(&capsule_project)?, current_revision);

    let (joint_dir, joint_project) = new_fixture(FixtureOptions::default())?;
    let unequal_spring = spring_bone(FixtureOptions {
        uneven_spring_joints: true,
        ..FixtureOptions::default()
    });
    apply_success(
        &joint_project,
        1,
        &[json!({"op": "avatar.update", "set": {"spring_bone": unequal_spring}})],
    )?;
    let current_revision = revision(&joint_project)?;
    assert_validate_export_preflight(
        &joint_project,
        "vrm0",
        &joint_dir.path().join("unequal-joints-v0.vrm"),
        "vrm.v0.spring_joint_settings",
    )?;
    assert_eq!(revision(&joint_project)?, current_revision);

    let (permission_dir, permission_project) = new_fixture(FixtureOptions {
        modern_extra_permissions: true,
        ..FixtureOptions::default()
    })?;
    let current_revision = revision(&permission_project)?;
    assert_validate_export_preflight(
        &permission_project,
        "vrm0",
        &permission_dir.path().join("unmapped-permissions-v0.vrm"),
        "vrm.v0.meta_permissions",
    )?;
    assert_eq!(revision(&permission_project)?, current_revision);
    Ok(())
}
#[test]
fn mtoon_texture_image_shorthand_conforms_to_published_schemas() -> TestResult {
    let options = FixtureOptions {
        v1_extended_mtoon: true,
        ..FixtureOptions::default()
    };
    let (_directory, project) = new_fixture(options)?;
    let operation_schema_output = pot()
        .args([
            "schema",
            "--kind",
            "operations",
            "--op",
            "avatar.update",
            "--json",
        ])
        .output()?;
    assert!(operation_schema_output.status.success());
    let operation_schema = cli_json(&operation_schema_output)?["result"]["schema"].clone();
    let operation_validator = jsonschema::validator_for(&operation_schema)?;
    let operation = fixture_operations(options)
        .pop()
        .ok_or("avatar fixture update operation is missing")?;
    assert!(
        operation_validator.is_valid(&operation),
        "the published avatar.update schema rejected valid TextureRef image shorthand"
    );

    let scene_schema_output = pot()
        .args(["schema", "--kind", "scene", "--json"])
        .output()?;
    assert!(scene_schema_output.status.success());
    let scene_schema = cli_json(&scene_schema_output)?["result"]["schema"].clone();
    let scene_validator = jsonschema::validator_for(&scene_schema)?;
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert!(
        scene_validator.is_valid(&document),
        "the published scene schema rejected serialized MToon TextureRefs"
    );
    Ok(())
}

#[test]
fn spring_bone_references_export_unskinned_accessory_armature_bones_in_both_versions() -> TestResult
{
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let spring = json!({
        "colliders": [{
            "node": {"node": "accessory_arm", "bone": "accessory_root"},
            "shape": {"type": "sphere", "offset": [0.0, 0.0, 0.0], "radius": 0.04},
        }],
        "collider_groups": [{"name": "Accessory", "colliders": [0]}],
        "springs": [{
            "name": "Accessory",
            "center": {"node": "accessory_arm", "bone": "accessory_root"},
            "joints": [
                {
                    "node": {"node": "accessory_arm", "bone": "accessory_root"},
                    "hit_radius": 0.02,
                    "stiffness": 0.7,
                    "gravity_power": 0.1,
                    "gravity_dir": [0.0, 0.0, -1.0],
                    "drag_force": 0.5,
                },
                {
                    "node": {"node": "accessory_arm", "bone": "accessory_tip"},
                    "hit_radius": 0.02,
                    "stiffness": 0.7,
                    "gravity_power": 0.1,
                    "gravity_dir": [0.0, 0.0, -1.0],
                    "drag_force": 0.5,
                },
            ],
            "collider_groups": [0],
        }],
    });
    apply_success(
        &project,
        1,
        &[
            json!({"op": "node.create", "id": "accessory_arm", "kind": "armature"}),
            json!({"op": "bone.create", "target": {"id": "accessory_arm"}, "id": "accessory_root", "name": "Accessory Root", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.1]}),
            json!({"op": "bone.create", "target": {"id": "accessory_arm"}, "id": "accessory_tip", "name": "Accessory Tip", "head": [0.0, 0.0, 0.1], "tail": [0.0, 0.0, 0.2], "parent": "accessory_root"}),
            json!({"op": "avatar.update", "set": {"spring_bone": spring}}),
        ],
    )?;

    let v1_path = directory.path().join("accessory-v1.vrm");
    let v0_path = directory.path().join("accessory-v0.vrm");
    export(&project, "vrm", &v1_path)?;
    export(&project, "vrm0", &v0_path)?;

    let v1 = parse_glb(&v1_path)?;
    let v1_root = &v1.json;
    let v1_accessory_root = gltf_bone_index(v1_root, "accessory_root");
    let v1_accessory_tip = gltf_bone_index(v1_root, "accessory_tip");
    assert_eq!(
        v1_root["nodes"][v1_accessory_root]["extras"]["potter"]["armature_id"],
        "accessory_arm"
    );
    let v1_skins = v1_root["skins"]
        .as_array()
        .ok_or("main humanoid skin is missing")?;
    assert!(v1_skins.iter().all(|skin| {
        skin["joints"].as_array().is_some_and(|joints| {
            joints.iter().all(|joint| {
                usize_field(joint) != v1_accessory_root && usize_field(joint) != v1_accessory_tip
            })
        })
    }));
    let v1_spring = &v1_root["extensions"]["VRMC_springBone"];
    assert_eq!(
        usize_field(&v1_spring["colliders"][0]["node"]),
        v1_accessory_root
    );
    assert_eq!(
        usize_field(&v1_spring["springs"][0]["center"]),
        v1_accessory_root
    );
    assert_eq!(
        usize_field(&v1_spring["springs"][0]["joints"][0]["node"]),
        v1_accessory_root
    );
    assert_eq!(
        usize_field(&v1_spring["springs"][0]["joints"][1]["node"]),
        v1_accessory_tip
    );

    let v0 = parse_glb(&v0_path)?;
    let v0_root = &v0.json;
    let v0_accessory_root = gltf_bone_index(v0_root, "accessory_root");
    let v0_accessory_tip = gltf_bone_index(v0_root, "accessory_tip");
    assert_eq!(
        v0_root["nodes"][v0_accessory_root]["extras"]["potter"]["armature_id"],
        "accessory_arm"
    );
    let v0_skins = v0_root["skins"]
        .as_array()
        .ok_or("main humanoid skin is missing")?;
    assert!(v0_skins.iter().all(|skin| {
        skin["joints"].as_array().is_some_and(|joints| {
            joints.iter().all(|joint| {
                usize_field(joint) != v0_accessory_root && usize_field(joint) != v0_accessory_tip
            })
        })
    }));
    let secondary = &v0_root["extensions"]["VRM"]["secondaryAnimation"];
    assert_eq!(
        usize_field(&secondary["colliderGroups"][0]["node"]),
        v0_accessory_root
    );
    assert_eq!(
        secondary["boneGroups"][0]["bones"],
        json!([v0_accessory_root])
    );
    assert_ne!(v0_accessory_root, v0_accessory_tip);
    Ok(())
}

#[test]
fn spring_chain_can_start_at_armature_object_before_its_root_bone() -> TestResult<()> {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let spring = json!({
        "colliders": [],
        "collider_groups": [],
        "springs": [{
            "name": "Armature Root",
            "center": {"node": "accessory_arm"},
            "joints": [
                {
                    "node": {"node": "accessory_arm"},
                    "hit_radius": 0.02,
                    "stiffness": 0.7,
                    "gravity_power": 0.1,
                    "gravity_dir": [0.0, 0.0, -1.0],
                    "drag_force": 0.5,
                },
                {
                    "node": {"node": "accessory_arm", "bone": "accessory_root"},
                    "hit_radius": 0.02,
                    "stiffness": 0.7,
                    "gravity_power": 0.1,
                    "gravity_dir": [0.0, 0.0, -1.0],
                    "drag_force": 0.5,
                },
            ],
        }],
    });
    apply_success(
        &project,
        1,
        &[
            json!({"op": "node.create", "id": "accessory_arm", "kind": "armature"}),
            json!({"op": "bone.create", "target": {"id": "accessory_arm"}, "id": "accessory_root", "name": "Accessory Root", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.1]}),
            json!({"op": "avatar.update", "set": {"spring_bone": spring}}),
        ],
    )?;
    let validation = pot()
        .arg("validate")
        .arg(&project)
        .args(["--format", "vrm", "--json"])
        .output()?;
    assert!(validation.status.success());
    assert_eq!(cli_json(&validation)?["result"]["valid"], true);

    let output = directory.path().join("armature-object-to-root-bone.vrm");
    export(&project, "vrm", &output)?;
    let root = parse_glb(&output)?.json;
    let armature_node = gltf_node_index(&root, "accessory_arm");
    let root_bone = gltf_bone_index(&root, "accessory_root");
    let joints = &root["extensions"]["VRMC_springBone"]["springs"][0]["joints"];
    assert_eq!(usize_field(&joints[0]["node"]), armature_node);
    assert_eq!(usize_field(&joints[1]["node"]), root_bone);
    assert_eq!(gltf_node_parent(&root, root_bone), Some(armature_node));
    Ok(())
}

#[test]
fn vrm_material_alpha_modes_have_consumer_semantics_in_both_versions() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let mut mtoon_settings = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    mtoon_settings["mask_material"] = json!({
        "shading_toony_factor": 0.7,
        "shading_shift_factor": -0.15,
    });
    mtoon_settings["blend_material"] = json!({
        "shading_toony_factor": 0.7,
        "shading_shift_factor": -0.15,
        "transparent_with_z_write": true,
        "render_queue_offset_number": 2,
    });
    mtoon_settings["plain_blend_material"] = json!({
        "shading_toony_factor": 0.7,
        "shading_shift_factor": -0.15,
    });
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "mask_material",
                "name": "Mask Surface",
                "alpha_mode": "clip",
                "alpha_threshold": 0.37,
                "base_color": [0.7, 0.8, 0.9, 1.0],
            }),
            json!({
                "op": "material.create",
                "id": "blend_material",
                "name": "Blend Surface",
                "alpha_mode": "blend",
                "base_color": [0.7, 0.8, 0.9, 0.5],
            }),
            json!({
                "op": "material.update",
                "target": {"id": "skin_material"},
                "set": {
                    "ior": 1.33,
                    "emission_color": [0.2, 0.3, 0.4],
                    "emission_strength": 2.5,
                },
            }),
            json!({
                "op": "material.create",
                "id": "ordinary_physical",
                "name": "Ordinary Physical PBR",
                "ior": 1.25,
                "emission_color": [0.4, 0.5, 0.6],
                "emission_strength": 3.5,
                "transmission": 0.5,
            }),
            json!({
                "op": "node.create",
                "id": "ordinary_physical_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "ordinary_physical",
            }),
            json!({"op": "node.create", "id": "mask_surface", "kind": "box", "params": {"size": 0.5}, "material": "mask_material"}),
            json!({"op": "node.create", "id": "blend_surface", "kind": "box", "params": {"size": 0.5}, "material": "blend_material"}),
            json!({
                "op": "material.create",
                "id": "plain_blend_material",
                "name": "Plain Blend Surface",
                "alpha_mode": "blend",
                "base_color": [0.7, 0.8, 0.9, 0.5],
            }),
            json!({"op": "node.create", "id": "plain_blend_surface", "kind": "box", "params": {"size": 0.5}, "material": "plain_blend_material"}),
            json!({"op": "avatar.update", "set": {"mtoon": mtoon_settings}}),
        ],
    )?;

    let v1_path = directory.path().join("alpha-v1.vrm");
    let v0_path = directory.path().join("alpha-v0.vrm");
    export(&project, "vrm", &v1_path)?;
    export(&project, "vrm0", &v0_path)?;
    let v1 = parse_glb(&v1_path)?;
    let v1_root = &v1.json;
    let opaque = &v1_root["materials"][gltf_material_index(v1_root, "skin_material")];
    let mask_index = gltf_material_index(v1_root, "mask_material");
    let blend_index = gltf_material_index(v1_root, "blend_material");
    assert_eq!(opaque["alphaMode"].as_str().unwrap_or("OPAQUE"), "OPAQUE");
    let ordinary = &v1_root["materials"][gltf_material_index(v1_root, "ordinary_physical")];
    assert!(opaque["extensions"]["KHR_materials_ior"].is_null());
    close(&opaque["emissiveFactor"][0], 0.5);
    close(&opaque["emissiveFactor"][1], 0.75);
    close(&opaque["emissiveFactor"][2], 1.0);
    // The peak channel is exactly 1, so no emissive strength extension is written.
    assert!(opaque["extensions"]["KHR_materials_emissive_strength"].is_null());
    close(&ordinary["extensions"]["KHR_materials_ior"]["ior"], 1.25);
    close(
        &ordinary["extensions"]["KHR_materials_transmission"]["transmissionFactor"],
        0.5,
    );
    close(&ordinary["emissiveFactor"][0], 2.0 / 3.0);
    close(&ordinary["emissiveFactor"][1], 5.0 / 6.0);
    close(&ordinary["emissiveFactor"][2], 1.0);
    close(
        &ordinary["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
        2.1,
    );
    assert!(
        v1_root["extensionsUsed"]
            .as_array()
            .is_some_and(|extensions| extensions
                .iter()
                .any(|value| { value.as_str() == Some("KHR_materials_ior") }))
    );
    assert_eq!(v1_root["materials"][mask_index]["alphaMode"], "MASK");
    close(&v1_root["materials"][mask_index]["alphaCutoff"], 0.37);
    assert!(
        !v1_root["materials"][mask_index]["extensions"]["VRMC_materials_mtoon"]
            ["transparentWithZWrite"]
            .as_bool()
            .unwrap_or(false)
    );
    assert_eq!(v1_root["materials"][blend_index]["alphaMode"], "BLEND");
    assert_eq!(
        v1_root["materials"][blend_index]["extensions"]["VRMC_materials_mtoon"]["transparentWithZWrite"],
        true
    );
    close(
        &v1_root["materials"][blend_index]["extensions"]["VRMC_materials_mtoon"]["renderQueueOffsetNumber"],
        2.0,
    );

    let v0 = parse_glb(&v0_path)?;
    let v0_root = &v0.json;
    let legacy_properties = &v0_root["extensions"]["VRM"]["materialProperties"];
    let opaque_properties = &legacy_properties[gltf_material_index(v0_root, "skin_material")];
    let mask_properties = &legacy_properties[gltf_material_index(v0_root, "mask_material")];
    let blend_properties = &legacy_properties[gltf_material_index(v0_root, "blend_material")];
    close(&opaque_properties["floatProperties"]["_BlendMode"], 0.0);
    close(&opaque_properties["floatProperties"]["_ZWrite"], 1.0);
    close(&opaque_properties["floatProperties"]["_SrcBlend"], 1.0);
    close(&opaque_properties["floatProperties"]["_DstBlend"], 0.0);
    assert_eq!(opaque_properties["renderQueue"], 2000);
    assert_eq!(mask_properties["keywordMap"]["_ALPHATEST_ON"], true);
    close(&mask_properties["floatProperties"]["_Cutoff"], 0.37);
    close(&mask_properties["floatProperties"]["_BlendMode"], 1.0);
    close(&mask_properties["floatProperties"]["_ZWrite"], 1.0);
    close(&mask_properties["floatProperties"]["_SrcBlend"], 1.0);
    close(&mask_properties["floatProperties"]["_DstBlend"], 0.0);
    assert_eq!(mask_properties["renderQueue"], 2450);
    assert_eq!(blend_properties["keywordMap"]["_ALPHABLEND_ON"], true);
    close(&blend_properties["floatProperties"]["_BlendMode"], 3.0);
    close(&blend_properties["floatProperties"]["_ZWrite"], 1.0);
    close(&blend_properties["floatProperties"]["_SrcBlend"], 5.0);
    close(&blend_properties["floatProperties"]["_DstBlend"], 10.0);
    assert_eq!(blend_properties["renderQueue"], 2503);
    let plain_blend_properties =
        &legacy_properties[gltf_material_index(v0_root, "plain_blend_material")];
    assert_eq!(plain_blend_properties["keywordMap"]["_ALPHABLEND_ON"], true);
    close(
        &plain_blend_properties["floatProperties"]["_BlendMode"],
        2.0,
    );
    close(&plain_blend_properties["floatProperties"]["_ZWrite"], 0.0);
    assert_eq!(plain_blend_properties["renderQueue"], 3000);

    let (transmission_directory, transmission_project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &transmission_project,
        1,
        &[json!({
            "op": "material.update",
            "target": {"id": "skin_material"},
            "set": {"transmission": 0.25},
        })],
    )?;
    assert_preflight_both_versions(
        &transmission_directory,
        &transmission_project,
        "mtoon-transmission",
        "vrm.mtoon_transmission",
    )?;
    Ok(())
}

#[test]
fn vrm_preflight_rejects_hidden_first_person_and_morph_mesh_targets() -> TestResult {
    let (first_person_dir, first_person_project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &first_person_project,
        1,
        &[
            json!({"op": "avatar.update", "set": {"expressions": {}}}),
            json!({"op": "node.update", "target": {"id": "body"}, "set": {"visible": false}}),
        ],
    )?;
    let current_revision = revision(&first_person_project)?;
    assert_validate_export_preflight(
        &first_person_project,
        "vrm",
        &first_person_dir.path().join("hidden-first-person.vrm"),
        "vrm.first_person_mesh_export",
    )?;
    assert_eq!(revision(&first_person_project)?, current_revision);

    let (morph_dir, morph_project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &morph_project,
        1,
        &[
            json!({"op": "avatar.update", "set": {"first_person": null}}),
            json!({"op": "node.update", "target": {"id": "body"}, "set": {"visible": false}}),
        ],
    )?;
    let current_revision = revision(&morph_project)?;
    assert_validate_export_preflight(
        &morph_project,
        "vrm",
        &morph_dir.path().join("hidden-morph.vrm"),
        "vrm.expression_morph_mesh_export",
    )?;
    assert_eq!(revision(&morph_project)?, current_revision);
    Ok(())
}

#[test]
fn vrm0_rejects_unmapped_branched_spring_terminal_bones() -> TestResult<()> {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[json!({
            "op": "bone.create",
            "target": {"id": "arm"},
            "id": "a_inert_child",
            "name": "Inert Unmapped Child",
            "head": [0.15, 0.0, 2.05],
            "tail": [0.25, 0.0, 2.15],
            "parent": "hair_root",
        })],
    )?;
    export(&project, "vrm", &directory.path().join("branched-vrm1.vrm"))?;
    assert_validate_export_preflight(
        &project,
        "vrm0",
        &directory.path().join("branched-vrm0.vrm"),
        "vrm.v0.spring_hierarchy",
    )?;
    Ok(())
}

#[test]
fn ordinary_textured_relative_shape_key_exports_in_both_vrm_versions() -> TestResult<()> {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "ordinary_relative_material",
                "name": "Ordinary Relative Material",
                "base_color_texture": {"image": "tex_albedo"},
            }),
            json!({
                "op": "node.create",
                "id": "ordinary_relative_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "ordinary_relative_material",
            }),
            json!({
                "op": "uv.unwrap",
                "target": {"id": "ordinary_relative_surface"},
                "method": "smart",
            }),
            json!({
                "op": "shape_key.create",
                "target": {"id": "ordinary_relative_surface"},
                "id": "relative_key",
                "name": "Relative Key",
                "positions": {"0": [0.3, 0.4, 0.6]},
            }),
        ],
    )?;
    for format in ["vrm", "vrm0"] {
        let validation = pot()
            .arg("validate")
            .arg(&project)
            .args(["--format", format, "--json"])
            .output()?;
        assert!(validation.status.success());
        assert_eq!(cli_json(&validation)?["result"]["valid"], true);

        let output = directory
            .path()
            .join(format!("ordinary-relative-{format}.vrm"));
        export(&project, format, &output)?;
        let glb = parse_glb(&output)?;
        let node = gltf_node_index(&glb.json, "ordinary_relative_surface");
        let mesh_index = usize_field(&glb.json["nodes"][node]["mesh"]);
        assert!(
            glb.json["meshes"][mesh_index]["primitives"][0]["targets"][0]["POSITION"].is_number()
        );
    }
    Ok(())
}

#[test]
fn vrm_output_preflight_matches_emitter_for_ordinary_materials_and_shape_keys() -> TestResult<()> {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "uvless_ordinary_material",
                "name": "UV-less Ordinary",
                "base_color_texture": {"image": "tex_albedo"},
            }),
            json!({
                "op": "node.create",
                "id": "uvless_ordinary_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "uvless_ordinary_material",
            }),
        ],
    )?;
    assert_format_loss_both_versions(&project, "uv.layer_missing")?;

    let lossy_path = directory.path().join("uvless-ordinary.vrm");
    let lossy_output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "vrm", "--out"])
        .arg(&lossy_path)
        .args(["--allow-lossy", "--overwrite", "--json"])
        .output()?;
    assert!(
        lossy_output.status.success(),
        "lossy UV-less VRM export failed: stdout={} stderr={}",
        String::from_utf8_lossy(&lossy_output.stdout),
        String::from_utf8_lossy(&lossy_output.stderr)
    );
    let lossy_glb = parse_glb(&lossy_path)?;
    let uvless_node = gltf_node_index(&lossy_glb.json, "uvless_ordinary_surface");
    let uvless_mesh = usize_field(&lossy_glb.json["nodes"][uvless_node]["mesh"]);
    let attributes = &lossy_glb.json["meshes"][uvless_mesh]["primitives"][0]["attributes"];
    assert!(attributes["TEXCOORD_0"].is_null());
    assert!(
        attributes
            .as_object()
            .is_some_and(|attributes| !attributes.keys().any(|name| name.starts_with("TEXCOORD_")))
    );

    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "shader_graph_material",
                "name": "Shader Graph Surface",
                "graph": {
                    "kind": "shader",
                    "nodes": {
                        "out": {"type": "OutputMaterial"},
                        "principled": {"type": "ShaderNodeBsdfPrincipled"},
                        "noise": {"type": "ShaderNodeTexNoise"}
                    },
                    "links": [
                        {
                            "from_node": "principled",
                            "from_socket": "BSDF",
                            "to_node": "out",
                            "to_socket": "Surface"
                        },
                        {
                            "from_node": "noise",
                            "from_socket": "Color",
                            "to_node": "principled",
                            "to_socket": "Base Color"
                        }
                    ]
                },
            }),
            json!({
                "op": "node.create",
                "id": "shader_graph_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "shader_graph_material",
            }),
        ],
    )?;
    assert_format_loss_both_versions(&project, "material.shader_graph.unsupported_node")?;

    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "plain_graph_material",
                "name": "Plain Principled Graph",
                "graph": {
                    "kind": "shader",
                    "nodes": {
                        "out": {"type": "OutputMaterial"},
                        "principled": {
                            "type": "ShaderNodeBsdfPrincipled",
                            "inputs": {"Base Color": [0.2, 0.4, 0.6, 1.0]}
                        }
                    },
                    "links": [{
                        "from_node": "principled",
                        "from_socket": "BSDF",
                        "to_node": "out",
                        "to_socket": "Surface"
                    }]
                },
            }),
            json!({
                "op": "node.create",
                "id": "plain_graph_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "plain_graph_material",
            }),
        ],
    )?;
    assert_exports_both_versions(&directory, &project, "plain-principled-graph")?;

    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "split_map_material",
                "name": "Separate Metallic and Roughness",
                "metallic_texture": {"image": "tex_albedo"},
                "roughness_texture": {"image": "tex_shade"},
            }),
            json!({
                "op": "node.create",
                "id": "split_map_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "split_map_material",
            }),
            json!({"op": "uv.unwrap", "target": {"id": "split_map_surface"}, "method": "smart"}),
        ],
    )?;
    assert_exports_both_versions(&directory, &project, "separate-metallic-roughness-maps")?;
    let packed = parse_glb(
        &directory
            .path()
            .join("separate-metallic-roughness-maps-vrm.vrm"),
    )?;
    let material = gltf_material_index(&packed.json, "split_map_material");
    let texture = usize_field(
        &packed.json["materials"][material]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"],
    );
    let image = usize_field(&packed.json["textures"][texture]["source"]);
    assert_eq!(
        packed.json["images"][image]["extras"]["potter"]["packed"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );

    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[json!({
            "op": "shape_key.update",
            "target": {"id": "body"},
            "id": "smile",
            "set": {"absolute": true},
        })],
    )?;
    assert_preflight_both_versions(
        &directory,
        &project,
        "absolute-shape-key",
        "mesh.shape_key_absolute",
    )?;
    Ok(())
}

#[test]
fn vrm_reports_mtoon_textures_without_uv_layer_as_loss() -> TestResult {
    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    let mut mtoon_settings = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    mtoon_settings["uvless_mtoon_material"] = json!({
        "shade_multiply_texture": {"image": "tex_shade"},
        "shading_toony_factor": 0.7,
        "shading_shift_factor": -0.15,
    });
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "uvless_mtoon_material",
                "name": "UV-less MToon Surface",
            }),
            json!({
                "op": "node.create",
                "id": "uvless_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "uvless_mtoon_material",
            }),
            json!({"op": "avatar.update", "set": {"mtoon": mtoon_settings}}),
        ],
    )?;
    let current_revision = revision(&project)?;
    assert_format_loss_both_versions(&project, "uv.layer_missing")?;
    assert_eq!(revision(&project)?, current_revision);
    Ok(())
}

#[test]
fn changed_constrained_skin_pose_is_rejected_explicitly() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({"op": "node.create", "id": "skin_arm", "kind": "armature"}),
            json!({"op": "bone.create", "target": {"id": "skin_arm"}, "id": "skin_root", "name": "skin_root", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}),
            json!({"op": "bone.create", "target": {"id": "skin_arm"}, "id": "skin_tip", "name": "skin_tip", "head": [0.0, 0.0, 1.0], "tail": [0.0, 0.0, 2.0], "parent": "skin_root"}),
            json!({"op": "node.create", "id": "skin_goal", "kind": "empty", "transform": {"translation": [1.0, 0.0, 1.0]}}),
            json!({"op": "constraint.create", "target": {"id": "skin_arm"}, "id": "move_skin_tip", "type": "ik", "constraint_target": "skin_goal", "owner_bone": "skin_tip", "params": {"chain_count": 2}}),
            json!({"op": "node.create", "id": "constrained_skin", "kind": "box", "params": {"size": 0.2}, "material": "skin_material"}),
            json!({"op": "shape_key.create", "target": {"id": "constrained_skin"}, "id": "constrained_shape", "name": "Constrained Pose", "positions": {"0": [0.8, 0.6, 0.4]}}),
            json!({"op": "vertex_group.create", "target": {"id": "constrained_skin"}, "id": "skin_tip", "name": "skin_tip"}),
            json!({
                "op": "vertex_group.assign",
                "target": {"id": "constrained_skin"},
                "group_id": "skin_tip",
                "weights": (0..8).map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0})).collect::<Vec<_>>(),
            }),
            json!({"op": "modifier.create", "target": {"id": "constrained_skin"}, "id": "constrained_skin_binding", "type": "armature", "params": {"object": "skin_arm", "use_vertex_groups": true}}),
        ],
    )?;
    let current_revision = revision(&project)?;
    let evaluated_end = evaluated_bone_endpoint(&project, "skin_arm", "skin_tip", 1.0)?;
    assert!(evaluated_end.distance(DVec3::new(0.0, 0.0, 2.0)) > 0.2);

    let output_path = directory.path().join("changed-constrained-pose.vrm");
    let output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "vrm", "--out"])
        .arg(&output_path)
        .arg("--allow-lossy")
        .arg("--json")
        .output()?;
    let failure = assert_failure(&output, "UNSUPPORTED_FEATURE")?;
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "rig.skin_shape_key_pose"
    );
    assert_eq!(failure["error"]["details"]["status"], "not_supported");
    for field in ["armature", "bone"] {
        assert!(
            failure["error"]["details"][field]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "pose rejection omitted `{field}` context"
        );
    }
    assert!(
        !output_path.exists(),
        "failed export created an output file"
    );
    assert_eq!(revision(&project)?, current_revision);
    Ok(())
}

#[test]
fn vrm0_uv_rotation_uses_turns_per_second_and_scroll_rotation_order_is_explicit() -> TestResult {
    let rotation_only = FixtureOptions {
        uv_rotation_only: true,
        ..FixtureOptions::default()
    };
    let (rotation_dir, rotation_project) = new_fixture(rotation_only)?;
    let rotation_v1_path = rotation_dir.path().join("rotation-v1.vrm");
    let rotation_v0_path = rotation_dir.path().join("rotation-v0.vrm");
    export(&rotation_project, "vrm", &rotation_v1_path)?;
    export(&rotation_project, "vrm0", &rotation_v0_path)?;
    let rotation_v1 = parse_glb(&rotation_v1_path)?;
    let rotation_v0 = parse_glb(&rotation_v0_path)?;
    assert_v1_semantics(&rotation_v1, rotation_only)?;
    assert_v1_local_frames(&rotation_project, &rotation_v1, rotation_only)?;
    assert_v0_local_frames(&rotation_project, &rotation_v0, &rotation_v1)?;
    assert_v0_semantics(&rotation_v0, rotation_only)?;
    assert_v0_mtoon_channel_remaps(&rotation_v1, &rotation_v0)?;
    let legacy_material_index = gltf_material_index(&rotation_v0.json, "skin_material");
    close(
        &rotation_v0.json["extensions"]["VRM"]["materialProperties"][legacy_material_index]["floatProperties"]
            ["_UvAnimRotation"],
        0.3 / std::f64::consts::TAU,
    );

    let combined = FixtureOptions {
        uv_scroll_and_rotation: true,
        ..FixtureOptions::default()
    };
    let (combined_dir, combined_project) = new_fixture(combined)?;
    let combined_v1_path = combined_dir.path().join("scroll-rotation-v1.vrm");
    let combined_v0_path = combined_dir.path().join("scroll-rotation-v0.vrm");
    export(&combined_project, "vrm", &combined_v1_path)?;
    assert_v1_semantics(&parse_glb(&combined_v1_path)?, combined)?;
    assert_validate_export_preflight(
        &combined_project,
        "vrm0",
        &combined_v0_path,
        "vrm.v0.mtoon_uv_animation_order",
    )?;
    Ok(())
}

#[test]
fn matcap_only_mtoon_material_does_not_require_mesh_uv0() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let mut mtoon_settings = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    mtoon_settings["uvless_matcap_material"] = json!({
        "matcap_texture": {"image": "tex_outline"},
        "matcap_factor": [1.0, 1.0, 1.0],
    });
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "material.create",
                "id": "uvless_matcap_material",
                "name": "UV-less MatCap Surface",
            }),
            json!({
                "op": "node.create",
                "id": "uvless_matcap_surface",
                "kind": "box",
                "params": {"size": 0.5},
                "material": "uvless_matcap_material",
            }),
            json!({"op": "avatar.update", "set": {"mtoon": mtoon_settings}}),
        ],
    )?;

    let validation = pot()
        .arg("validate")
        .arg(&project)
        .args(["--format", "vrm", "--json"])
        .output()?;
    assert!(validation.status.success());
    let validation = cli_json(&validation)?;
    assert_eq!(validation["result"]["valid"], true);
    let legacy_validation = pot()
        .arg("validate")
        .arg(&project)
        .args(["--format", "vrm0", "--json"])
        .output()?;
    assert!(legacy_validation.status.success());
    assert_eq!(cli_json(&legacy_validation)?["result"]["valid"], true);

    let v1_path = directory.path().join("matcap-uvless-v1.vrm");
    let v0_path = directory.path().join("matcap-uvless-v0.vrm");
    export(&project, "vrm", &v1_path)?;
    export(&project, "vrm0", &v0_path)?;
    let v1 = parse_glb(&v1_path)?;
    let v1_root = &v1.json;
    let matcap_material = gltf_material_index(v1_root, "uvless_matcap_material");
    assert_texture_ref(
        v1_root,
        &v1_root["materials"][matcap_material]["extensions"]["VRMC_materials_mtoon"]["matcapTexture"],
        "tex_outline",
    );
    let v0 = parse_glb(&v0_path)?;
    let v0_root = &v0.json;
    let v0_material = gltf_material_index(v0_root, "uvless_matcap_material");
    let matcap_texture = usize_field(
        &v0_root["extensions"]["VRM"]["materialProperties"][v0_material]["textureProperties"]["_SphereAdd"],
    );
    assert_eq!(texture_image_id(v0_root, matcap_texture), "tex_outline");
    Ok(())
}

#[test]
fn vrm0_rejects_weighted_terminal_spring_rotation() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({"op": "vertex_group.create", "target": {"id": "body"}, "id": "hair_tip_weights", "name": "Hair Tip"}),
            json!({
                "op": "vertex_group.assign",
                "target": {"id": "body"},
                "group_id": "hair_tip_weights",
                "weights": [{"vertex_id": 0, "weight": 1.0}],
            }),
        ],
    )?;
    let current_revision = revision(&project)?;
    assert_validate_export_preflight(
        &project,
        "vrm0",
        &directory.path().join("weighted-terminal-v0.vrm"),
        "vrm.v0.spring_terminal_rotation",
    )?;
    assert_eq!(revision(&project)?, current_revision);
    Ok(())
}

#[test]
fn vrm0_license_matrix_validate_and_export_agree() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    // (legacy license_name, allow_redistribution, modification, commercial_usage, rejecting feature id)
    let rows = [
        (
            "redistribution_prohibited",
            false,
            "prohibited",
            "personal_non_profit",
            None,
        ),
        (
            "redistribution_prohibited",
            false,
            "allow_modification",
            "personal_non_profit",
            None,
        ),
        (
            "redistribution_prohibited",
            true,
            "prohibited",
            "personal_non_profit",
            Some("vrm.v0.meta_license"),
        ),
        (
            "cc_by",
            false,
            "allow_modification_redistribution",
            "corporation",
            Some("vrm.v0.meta_license"),
        ),
        (
            "cc_by",
            true,
            "prohibited",
            "corporation",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by",
            true,
            "allow_modification",
            "corporation",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by",
            true,
            "allow_modification_redistribution",
            "personal_profit",
            Some("vrm.v0.meta_license_commercial"),
        ),
        (
            "cc_by",
            true,
            "allow_modification_redistribution",
            "corporation",
            None,
        ),
        (
            "cc_by_sa",
            true,
            "prohibited",
            "corporation",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_sa",
            true,
            "allow_modification",
            "corporation",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_sa",
            true,
            "allow_modification_redistribution",
            "personal_non_profit",
            Some("vrm.v0.meta_license_commercial"),
        ),
        (
            "cc_by_sa",
            true,
            "allow_modification_redistribution",
            "corporation",
            None,
        ),
        (
            "cc_by_nc",
            true,
            "prohibited",
            "personal_non_profit",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_nc",
            true,
            "allow_modification",
            "personal_non_profit",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_nc",
            true,
            "allow_modification_redistribution",
            "personal_non_profit",
            None,
        ),
        (
            "cc_by_nc",
            true,
            "allow_modification_redistribution",
            "corporation",
            None,
        ),
        (
            "cc_by_nc_sa",
            true,
            "allow_modification",
            "personal_non_profit",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_nc_sa",
            true,
            "allow_modification_redistribution",
            "personal_non_profit",
            None,
        ),
        (
            "cc_by_nd",
            true,
            "prohibited",
            "corporation",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_nd",
            true,
            "allow_modification",
            "personal_non_profit",
            Some("vrm.v0.meta_license_commercial"),
        ),
        ("cc_by_nd", true, "allow_modification", "corporation", None),
        (
            "cc_by_nd",
            true,
            "allow_modification_redistribution",
            "corporation",
            None,
        ),
        (
            "cc_by_nc_nd",
            true,
            "prohibited",
            "personal_non_profit",
            Some("vrm.v0.meta_license_modification"),
        ),
        (
            "cc_by_nc_nd",
            true,
            "allow_modification",
            "personal_non_profit",
            None,
        ),
        (
            "cc_by_nc_nd",
            true,
            "allow_modification_redistribution",
            "personal_non_profit",
            None,
        ),
        // CC0 waives credit, which VRM 0.x cannot express as `required`, so it is rejected first.
        (
            "cc0",
            true,
            "prohibited",
            "corporation",
            Some("vrm.v0.meta_credit_license"),
        ),
    ];
    for (index, (license, allow_redistribution, modification, commercial, rejected_by)) in
        rows.into_iter().enumerate()
    {
        let row = format!(
            "license_name={license} allow_redistribution={allow_redistribution} modification={modification} commercial_usage={commercial}"
        );
        let mut updated = meta(FixtureOptions::default());
        updated["legacy"]["license_name"] = json!(license);
        updated["allow_redistribution"] = json!(allow_redistribution);
        updated["modification"] = json!(modification);
        updated["commercial_usage"] = json!(commercial);
        apply_success(
            &project,
            revision(&project)?,
            &[json!({"op": "avatar.update", "set": {"meta": updated}})],
        )?;

        let validation_output = pot()
            .arg("validate")
            .arg(&project)
            .args(["--format", "vrm0", "--json"])
            .output()?;
        assert!(validation_output.status.success(), "{row}: validate failed");
        let validation = cli_json(&validation_output)?;
        assert_eq!(
            validation["result"]["valid"],
            rejected_by.is_none(),
            "{row}: valid flag"
        );
        let reported = validation["result"]["issues"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|issue| issue["code"] == "VRM_EXPORT_PREFLIGHT_FAILED")
            .and_then(|issue| issue["details"]["details"]["feature_id"].as_str());
        assert_eq!(reported, rejected_by, "{row}: validate feature id");

        let output_path = directory.path().join(format!("matrix-{index}.vrm"));
        if let Some(feature_id) = rejected_by {
            let failure = export_failure(&project, "vrm0", &output_path)?;
            assert_eq!(
                failure["error"]["details"]["feature_id"], feature_id,
                "{row}: export feature id"
            );
            assert!(!output_path.exists(), "{row}: rejected export wrote output");
        } else {
            export(&project, "vrm0", &output_path)?;
        }
    }
    Ok(())
}

#[test]
fn vrm0_license_settings_survive_fresh_processes_and_reexport_identically() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let mut updated = meta(FixtureOptions::default());
    updated["legacy"]["license_name"] = json!("cc_by_sa");
    updated["allow_redistribution"] = json!(true);
    updated["modification"] = json!("allow_modification_redistribution");
    updated["commercial_usage"] = json!("corporation");
    apply_success(
        &project,
        1,
        &[json!({"op": "avatar.update", "set": {"meta": updated}})],
    )?;
    let applied_avatar = scene_avatar(&project)?;
    let applied_scene = inspect(&project)?["scene"].clone();

    // Each call below is a separate `pot` process: inspect, validate, then export.
    assert_eq!(inspect(&project)?["scene"], applied_scene);
    let validation = pot()
        .arg("validate")
        .arg(&project)
        .args(["--format", "vrm0", "--json"])
        .output()?;
    assert!(validation.status.success());
    assert_eq!(cli_json(&validation)?["result"]["valid"], true);
    let first_path = directory.path().join("persist-first.vrm");
    export(&project, "vrm0", &first_path)?;
    let second_path = directory.path().join("persist-second.vrm");
    export(&project, "vrm0", &second_path)?;

    assert_eq!(fs::read(&first_path)?, fs::read(&second_path)?);
    assert_eq!(scene_avatar(&project)?, applied_avatar);
    assert_eq!(inspect(&project)?["scene"], applied_scene);
    let glb = parse_glb(&second_path)?;
    assert_eq!(
        glb.json["extensions"]["VRM"]["meta"]["licenseName"],
        "CC_BY_SA"
    );
    assert_eq!(
        glb.json["extensions"]["VRM"]["meta"]["commercialUssageName"],
        "Disallow"
    );
    Ok(())
}

#[test]
fn vrm0_failed_overwrite_export_leaves_output_directory_and_scene_untouched() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let output_path = directory.path().join("published.vrm");
    export(&project, "vrm0", &output_path)?;
    let published = fs::read(&output_path)?;
    let entries_before = directory_entries(directory.path())?;

    apply_success(
        &project,
        1,
        &[
            json!({"op": "avatar.update", "set": {"spring_bone": spring_bone(FixtureOptions {
                capsule: true,
                ..FixtureOptions::default()
            })}}),
        ],
    )?;
    let scene_before = fs::read(project.join("scene.json"))?;
    let scene_info_before = inspect(&project)?["scene"].clone();

    let output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "vrm0", "--overwrite", "--out"])
        .arg(&output_path)
        .arg("--json")
        .output()?;
    let failure = assert_failure(&output, "UNSUPPORTED_FEATURE")?;
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "vrm.v0.spring_capsule"
    );

    assert_eq!(fs::read(&output_path)?, published);
    assert_eq!(directory_entries(directory.path())?, entries_before);
    assert_eq!(fs::read(project.join("scene.json"))?, scene_before);
    assert_eq!(inspect(&project)?["scene"], scene_info_before);
    Ok(())
}

fn scene_avatar(project: &Path) -> TestResult<Value> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    Ok(document["avatar"].clone())
}

fn directory_entries(path: &Path) -> TestResult<Vec<String>> {
    let mut names = fs::read_dir(path)?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<TestResult<Vec<_>>>()?;
    names.sort();
    Ok(names)
}

#[test]
fn nonuniform_primitive_scale_preserves_world_geometry_in_both_versions() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[
            json!({
                "op": "node.create",
                "id": "scaled_primitive",
                "kind": "box",
                "params": {"size": 1.0},
                "material": "skin_material",
            }),
            json!({
                "op": "uv.unwrap",
                "target": {"id": "scaled_primitive"},
                "elements": {"domain": "face", "ids": ["f1"]},
                "method": "smart",
            }),
        ],
    )?;
    let base_v1_path = directory.path().join("scaled-base-v1.vrm");
    let base_v0_path = directory.path().join("scaled-base-v0.vrm");
    export(&project, "vrm", &base_v1_path)?;
    export(&project, "vrm0", &base_v0_path)?;
    let base_v1 = parse_glb(&base_v1_path)?;
    let base_v0 = parse_glb(&base_v0_path)?;

    apply_success(
        &project,
        2,
        &[json!({
            "op": "node.update",
            "target": {"id": "scaled_primitive"},
            "set": {"transform": {"scale": [2.0, 1.0, 0.5]}},
        })],
    )?;
    let scaled_v1_path = directory.path().join("scaled-v1.vrm");
    let scaled_v0_path = directory.path().join("scaled-v0.vrm");
    export(&project, "vrm", &scaled_v1_path)?;
    export(&project, "vrm0", &scaled_v0_path)?;
    let scaled_v1 = parse_glb(&scaled_v1_path)?;
    let scaled_v0 = parse_glb(&scaled_v0_path)?;
    assert_positive_uniform_node_scales(&scaled_v1.json)?;
    assert_scaled_primitive_preserved(&base_v1, &scaled_v1, "scaled_primitive")?;
    assert_scaled_primitive_preserved(&base_v0, &scaled_v0, "scaled_primitive")?;
    Ok(())
}

#[test]
fn vrm0_rejects_mtoon_fields_without_faithful_legacy_equivalents() -> TestResult {
    let (shift_dir, shift_project) = new_fixture(FixtureOptions::default())?;
    let mut shift_settings = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    shift_settings["skin_material"]["shading_shift_texture"] =
        json!({"texture": {"image": "tex_shift"}, "scale": 0.4});
    apply_success(
        &shift_project,
        1,
        &[json!({"op": "avatar.update", "set": {"mtoon": shift_settings}})],
    )?;
    let shift_revision = revision(&shift_project)?;
    assert_validate_export_preflight(
        &shift_project,
        "vrm0",
        &shift_dir.path().join("shading-shift-texture-v0.vrm"),
        "vrm.v0.mtoon_shading_shift_texture",
    )?;
    assert_eq!(revision(&shift_project)?, shift_revision);

    let (matcap_dir, matcap_project) = new_fixture(FixtureOptions::default())?;
    let mut matcap_settings = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    matcap_settings["skin_material"]["matcap_factor"] = json!([0.8, 0.7, 0.6]);
    apply_success(
        &matcap_project,
        1,
        &[json!({"op": "avatar.update", "set": {"mtoon": matcap_settings}})],
    )?;
    let matcap_revision = revision(&matcap_project)?;
    assert_validate_export_preflight(
        &matcap_project,
        "vrm0",
        &matcap_dir.path().join("tinted-matcap-v0.vrm"),
        "vrm.v0.mtoon_matcap_factor",
    )?;
    assert_eq!(revision(&matcap_project)?, matcap_revision);
    Ok(())
}

fn plane_normals(glb: &Glb, node_id: &str) -> TestResult<(Vec<DVec3>, Vec<DVec3>)> {
    let root = &glb.json;
    let node = gltf_node_index(root, node_id);
    let mesh = usize_field(&root["nodes"][node]["mesh"]);
    let primitive = &root["meshes"][mesh]["primitives"][0];
    let positions = accessor_matrix(glb, usize_field(&primitive["attributes"]["POSITION"]))?;
    let normals = accessor_matrix(glb, usize_field(&primitive["attributes"]["NORMAL"]))?;
    let indices = accessor_matrix(glb, usize_field(&primitive["indices"]))?;
    let world = gltf_node_world_matrix(root, node)?;
    let world_normal = world.inverse().transpose();
    // A negative world determinant flips the rasterized front face, so the effective normal is sign * cross.
    let sign = if world.determinant() < 0.0 { -1.0 } else { 1.0 };
    let point = |index: usize| {
        let row = &positions[index];
        world.transform_point3(DVec3::new(row[0], row[1], row[2]))
    };
    let mut front = Vec::new();
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = [0, 1, 2].map(|corner| triangle[corner][0] as usize);
        front.push((point(b) - point(a)).cross(point(c) - point(a)).normalize() * sign);
    }
    let vertex_normals = normals
        .iter()
        .map(|row| {
            world_normal
                .transform_vector3(DVec3::new(row[0], row[1], row[2]))
                .normalize()
        })
        .collect();
    Ok((front, vertex_normals))
}

#[test]
fn mirrored_multi_material_and_smooth_meshes_export_consistent_normals_in_both_versions()
-> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let base_revision = revision(&project)?;
    apply_success(
        &project,
        base_revision,
        &[
            json!({"op":"node.create","id":"identity_plane","name":"identity_plane","kind":"plane","params":{"size":2.0},"transform":{"translation":[-3.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"mirror_plane","name":"mirror_plane","kind":"plane","params":{"size":2.0},"transform":{"translation":[3.0,0.0,0.0],"scale":[-1.0,1.0,1.0]}}),
            json!({"op":"material.create","id":"slot_jacket"}),
            json!({"op":"material.create","id":"slot_skin"}),
            json!({"op":"node.create","id":"slot_box","name":"slot_box","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,-2.0]}}),
            json!({"op":"node.update","target":{"id":"slot_box"},"set":{"materials":["slot_jacket","slot_skin"]}}),
            json!({"op":"mesh.set_attribute","target":{"id":"slot_box"},"elements":{"domain":"face","ids":["f1"]},"material_index":1}),
            json!({"op":"node.create","id":"smooth_box","name":"smooth_box","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,2.0]}}),
            json!({"op":"mesh.shade_smooth","target":{"id":"smooth_box"},"elements":{"domain":"face","ids":["f0","f1","f2","f3","f4","f5"]}}),
        ],
    )?;
    let v1_path = directory.path().join("mixed.vrm");
    export(&project, "vrm", &v1_path)?;
    let v0_path = directory.path().join("mixed-v0.vrm");
    export(&project, "vrm0", &v0_path)?;
    for glb in [parse_glb(&v1_path)?, parse_glb(&v0_path)?] {
        let root = &glb.json;
        for node_id in ["identity_plane", "mirror_plane"] {
            let (front, vertex_normals) = plane_normals(&glb, node_id)?;
            assert!(!front.is_empty(), "{node_id} has no triangles");
            for normal in front {
                assert!(
                    (normal - DVec3::Y).length() < 1.0e-6,
                    "{node_id} front face {normal:?} must match the identity plane's +Y"
                );
            }
            for normal in vertex_normals {
                assert!(
                    (normal - DVec3::Y).length() < 1.0e-6,
                    "{node_id} NORMAL {normal:?} must agree with its winding"
                );
            }
        }
        let slot_node = gltf_node_index(root, "slot_box");
        let slot_mesh = usize_field(&root["nodes"][slot_node]["mesh"]);
        let primitives = root["meshes"][slot_mesh]["primitives"]
            .as_array()
            .ok_or("slot_box primitives are missing")?;
        assert_eq!(primitives.len(), 2, "one primitive per used material slot");
        assert_eq!(
            primitives[0]["attributes"]["NORMAL"],
            primitives[1]["attributes"]["NORMAL"]
        );
        assert_eq!(
            primitives[0]["attributes"]["POSITION"],
            primitives[1]["attributes"]["POSITION"]
        );
        assert_eq!(
            usize_field(&primitives[0]["material"]),
            gltf_material_index(root, "slot_jacket")
        );
        assert_eq!(
            usize_field(&primitives[1]["material"]),
            gltf_material_index(root, "slot_skin")
        );
        let smooth_node = gltf_node_index(root, "smooth_box");
        let smooth_mesh = usize_field(&root["nodes"][smooth_node]["mesh"]);
        let smooth_normals = accessor_matrix(
            &glb,
            usize_field(&root["meshes"][smooth_mesh]["primitives"][0]["attributes"]["NORMAL"]),
        )?;
        let diagonal = 3.0_f64.sqrt().recip();
        for row in smooth_normals {
            for component in row {
                assert!(
                    (component.abs() - diagonal).abs() < 1.0e-6,
                    "smooth box NORMAL component {component} must be a vertex diagonal"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn spring_collider_and_look_at_world_centres_agree_between_vrm0_and_vrm1() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let v1_path = directory.path().join("frames-v1.vrm");
    let v0_path = directory.path().join("frames-v0.vrm");
    export(&project, "vrm", &v1_path)?;
    export(&project, "vrm0", &v0_path)?;
    let v1 = parse_glb(&v1_path)?;
    let v0 = parse_glb(&v0_path)?;
    // Consumer semantics: VRM 1.0 offsets are node-local. three-vrm negates the serialized z of
    // VRM 0.x offsets, and VRMUtils.rotateVRM0 rotates the whole scene 180 degrees about Y.
    let rotate_y = DMat4::from_rotation_y(std::f64::consts::PI);

    let v1_collider = &v1.json["extensions"]["VRMC_springBone"]["colliders"][0];
    let v1_centre = gltf_node_world_matrix(&v1.json, usize_field(&v1_collider["node"]))?
        .transform_point3(DVec3::from_array(json_numbers::<3>(
            &v1_collider["shape"]["sphere"]["offset"],
        )?));
    let v0_group = &v0.json["extensions"]["VRM"]["secondaryAnimation"]["colliderGroups"][0];
    let serialized_collider = object_vector(&v0_group["colliders"][0]["offset"])?;
    let v0_centre = (rotate_y * gltf_node_world_matrix(&v0.json, usize_field(&v0_group["node"]))?)
        .transform_point3(DVec3::new(
            serialized_collider.x,
            serialized_collider.y,
            -serialized_collider.z,
        ));
    assert_point_close(v0_centre, v1_centre);

    let v1_head = gltf_node_world_matrix(&v1.json, gltf_bone_index(&v1.json, "head"))?;
    let v1_look_at = v1_head.transform_point3(DVec3::from_array(json_numbers::<3>(
        &v1.json["extensions"]["VRMC_vrm"]["lookAt"]["offsetFromHeadBone"],
    )?));
    let v0_head = gltf_node_world_matrix(&v0.json, gltf_bone_index(&v0.json, "head"))?;
    let serialized_look_at =
        object_vector(&v0.json["extensions"]["VRM"]["firstPerson"]["firstPersonBoneOffset"])?;
    let v0_look_at = (rotate_y * v0_head).transform_point3(DVec3::new(
        serialized_look_at.x,
        serialized_look_at.y,
        -serialized_look_at.z,
    ));
    assert_point_close(v0_look_at, v1_look_at);
    Ok(())
}

#[test]
fn vrm_emissive_strength_extension_is_present_only_when_it_changes_shading() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        1,
        &[json!({
            "op": "material.create",
            "id": "inert_glow",
            "name": "Inert Glow",
            "emission_strength": 2.5,
        })],
    )?;
    let v1_path = directory.path().join("emissive-v1.vrm");
    let v0_path = directory.path().join("emissive-v0.vrm");
    export(&project, "vrm", &v1_path)?;
    export(&project, "vrm0", &v0_path)?;
    for glb in [parse_glb(&v1_path)?, parse_glb(&v0_path)?] {
        let materials = glb.json["materials"]
            .as_array()
            .ok_or("glTF materials are missing")?;
        let inert = materials
            .iter()
            .find(|material| material["name"] == "Inert Glow")
            .ok_or("Inert Glow material is missing")?;
        assert!(inert["extensions"]["KHR_materials_emissive_strength"].is_null());
        close(&inert["extras"]["potter"]["emission_strength"], 2.5);
    }
    Ok(())
}

/// Four extra deform bones, each weighting vertex 0 of `body`; with its base group that is five influences.
fn extra_influence_operations() -> Vec<Value> {
    let mut operations = Vec::new();
    for (name, x) in [
        ("extra_1", 0.5),
        ("extra_2", 0.6),
        ("extra_3", 0.7),
        ("extra_4", 0.8),
    ] {
        operations.push(json!({"op": "bone.create", "target": {"id": "arm"}, "id": name, "name": name, "head": [x, 0.0, 0.0], "tail": [x, 0.0, 0.2]}));
        operations.push(json!({"op": "vertex_group.create", "target": {"id": "body"}, "id": name, "name": name}));
        operations.push(json!({"op": "vertex_group.assign", "target": {"id": "body"}, "group_id": name, "weights": [{"vertex_id": 0, "weight": 0.25}]}));
    }
    operations
}

/// Skinned box without avatar settings, so glb export reaches the skin writer.
fn plain_skinned_box(
    extra_influences: bool,
    unweighted_vertex: bool,
) -> TestResult<(TempDir, std::path::PathBuf)> {
    let (directory, project) = init_project()?;
    let mut operations = vec![
        json!({"op": "node.create", "id": "arm", "kind": "armature"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "root", "name": "root", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}),
        json!({"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1.0}}),
        json!({"op": "vertex_group.create", "target": {"id": "body"}, "id": "root", "name": "root"}),
        json!({"op": "vertex_group.assign", "target": {"id": "body"}, "group_id": "root", "weights": (0..8).map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0})).collect::<Vec<_>>()}),
        json!({"op": "modifier.create", "target": {"id": "body"}, "id": "skin", "type": "armature", "params": {"object": "arm", "use_vertex_groups": true}}),
    ];
    if extra_influences {
        operations.extend(extra_influence_operations());
    }
    if unweighted_vertex {
        operations.push(json!({"op": "vertex_group.remove", "target": {"id": "body"}, "group_id": "root", "vertex_ids": [0]}));
    }
    apply_success(&project, revision(&project)?, &operations)?;
    Ok((directory, project))
}

#[test]
fn skinned_vertex_without_deform_weight_is_rejected_in_every_export_format() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(
        &project,
        revision(&project)?,
        &[
            json!({"op": "vertex_group.remove", "target": {"id": "body"}, "group_id": "hips", "vertex_ids": [0]}),
        ],
    )?;
    for format in ["vrm", "vrm0"] {
        let output_path = directory.path().join(format!("unweighted-{format}.vrm"));
        let failure = assert_validate_export_preflight(
            &project,
            format,
            &output_path,
            "mesh.skin_unweighted_vertex",
        )?;
        assert_eq!(failure["error"]["details"]["node"], "body");
        assert_eq!(failure["error"]["details"]["count"], 1);
        assert_eq!(failure["error"]["details"]["first_vertex"], 0);
        assert!(
            !output_path.exists(),
            "failed VRM export created an output file"
        );
    }
    let (_plain_directory, plain) = plain_skinned_box(false, true)?;
    let glb_path = directory.path().join("unweighted.glb");
    let failure = export_failure(&plain, "glb", &glb_path)?;
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "mesh.skin_unweighted_vertex"
    );
    assert_eq!(failure["error"]["details"]["node"], "body");
    assert_eq!(failure["error"]["details"]["count"], 1);
    assert!(
        !glb_path.exists(),
        "failed glb export created an output file"
    );
    Ok(())
}

#[test]
fn vrm_caps_skin_influences_at_four_while_glb_keeps_second_joint_set() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    apply_success(&project, revision(&project)?, &extra_influence_operations())?;
    for format in ["vrm", "vrm0"] {
        let output_path = directory
            .path()
            .join(format!("five-influences-{format}.vrm"));
        let failure = assert_validate_export_preflight(
            &project,
            format,
            &output_path,
            "vrm.skin_influence_limit",
        )?;
        assert_eq!(failure["error"]["details"]["node"], "body");
        assert_eq!(failure["error"]["details"]["count"], 1);
        assert_eq!(failure["error"]["details"]["first_vertex"], 0);
        assert_eq!(failure["error"]["details"]["max_influences"], 5);
    }
    let (_plain_directory, plain) = plain_skinned_box(true, false)?;
    let glb_path = directory.path().join("five-influences.glb");
    let output = pot()
        .arg("export")
        .arg(&plain)
        .args(["--format", "glb", "--out"])
        .arg(&glb_path)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "glb export rejected five influences: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let glb = parse_glb(&glb_path)?;
    let root = &glb.json;
    let body = gltf_node_index(root, "body");
    let primitive = &root["meshes"][usize_field(&root["nodes"][body]["mesh"])]["primitives"][0];
    assert!(primitive["attributes"]["JOINTS_1"].is_number());
    assert!(primitive["attributes"]["WEIGHTS_1"].is_number());
    let weights0 = accessor_matrix(&glb, usize_field(&primitive["attributes"]["WEIGHTS_0"]))?;
    let weights1 = accessor_matrix(&glb, usize_field(&primitive["attributes"]["WEIGHTS_1"]))?;
    assert_eq!(weights0.len(), weights1.len());
    for (first, second) in weights0.iter().zip(&weights1) {
        let total: f64 = first.iter().chain(second).sum();
        assert!((total - 1.0).abs() <= 1.0e-6, "skin weights sum to {total}");
    }
    assert!(
        weights1.iter().flatten().any(|weight| *weight > 0.0),
        "five-influence vertex lost its second joint set"
    );
    Ok(())
}

#[test]
fn append_import_keeps_base_avatar_and_reports_it_unchanged() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let base_avatar = scene_avatar(&project)?;
    let (_source_directory, source) = plain_skinned_box(false, false)?;
    let source_glb = directory.path().join("hat.glb");
    let exported = pot()
        .arg("export")
        .arg(&source)
        .args(["--format", "glb", "--out"])
        .arg(&source_glb)
        .arg("--json")
        .output()?;
    assert!(exported.status.success());
    let base_revision = revision(&project)?;
    let imported = pot()
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(&source_glb)
        .args(["--format", "glb", "--base-revision"])
        .arg(base_revision.to_string())
        .args(["--mode", "append", "--json"])
        .output()?;
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stdout)
    );
    let result = cli_json(&imported)?;
    assert_eq!(result["result"]["changes"]["avatar"]["updated"], false);
    assert_eq!(scene_avatar(&project)?, base_avatar);
    export(&project, "vrm", &directory.path().join("after-append.vrm"))?;
    Ok(())
}

#[test]
fn undo_reports_avatar_changes_in_changes() -> TestResult {
    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    let base_revision = revision(&project)?;
    apply_success(&project, base_revision, &[json!({"op": "avatar.clear"})])?;
    let output = pot()
        .arg("undo")
        .arg(&project)
        .args([
            "--base-revision",
            &(base_revision + 1).to_string(),
            "--json",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        cli_json(&output)?["result"]["changes"]["avatar"]["updated"],
        true
    );
    assert!(scene_avatar(&project)?.is_object());
    Ok(())
}

#[test]
fn mtoon_validation_pointers_name_the_material_and_field_through_apply() -> TestResult {
    let (_directory, project) = new_fixture(FixtureOptions::default())?;
    let base_revision = revision(&project)?;

    let mut range = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    range["skin_material"]["shading_toony_factor"] = json!(2.0);
    let output = apply_batch(
        &project,
        base_revision,
        &[json!({"op": "avatar.update", "set": {"mtoon": range}})],
    )?;
    let envelope = assert_failure(&output, "INVALID_OPERATION")?;
    assert_eq!(
        envelope["error"]["details"]["pointer"],
        "/operations/0/set/mtoon/skin_material/shading_toony_factor"
    );

    let mut texture = avatar_settings(FixtureOptions::default())["mtoon"].clone();
    texture["skin_material"]["matcap_texture"] = json!({"image": "missing_matcap"});
    let output = apply_batch(
        &project,
        base_revision,
        &[json!({"op": "avatar.update", "set": {"mtoon": texture}})],
    )?;
    let envelope = assert_failure(&output, "INVALID_OPERATION")?;
    assert_eq!(
        envelope["error"]["details"]["pointer"],
        "/operations/0/set/mtoon/skin_material/matcap_texture"
    );
    assert_eq!(revision(&project)?, base_revision);
    Ok(())
}

#[test]
fn mirrored_vrm1_export_reimports_with_the_winding_its_normals_describe() -> TestResult {
    let (directory, project) = new_fixture(FixtureOptions::default())?;
    let base_revision = revision(&project)?;
    apply_success(
        &project,
        base_revision,
        &[
            json!({"op":"node.create","id":"mirror_plane","name":"mirror_plane","kind":"plane","params":{"size":2.0},"transform":{"translation":[3.0,0.0,0.0],"scale":[-1.0,1.0,1.0]}}),
        ],
    )?;
    let vrm_path = directory.path().join("mirror.vrm");
    export(&project, "vrm", &vrm_path)?;
    // The GLB importer accepts the VRM container once its extension says `.glb`.
    let glb_path = directory.path().join("mirror.glb");
    fs::copy(&vrm_path, &glb_path)?;
    let (_fresh_directory, fresh) = init_project()?;
    let fresh_revision = revision(&fresh)?;
    let output = pot()
        .arg("import")
        .arg(&fresh)
        .arg("--file")
        .arg(&glb_path)
        .args(["--format", "glb", "--base-revision"])
        .arg(fresh_revision.to_string())
        .arg("--allow-lossy")
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot import failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let reexport_path = directory.path().join("reimported.glb");
    let reexport = pot()
        .arg("export")
        .arg(&fresh)
        .args(["--format", "glb", "--out"])
        .arg(&reexport_path)
        .arg("--json")
        .output()?;
    assert!(
        reexport.status.success(),
        "glb re-export failed: stdout={} stderr={}",
        String::from_utf8_lossy(&reexport.stdout),
        String::from_utf8_lossy(&reexport.stderr)
    );
    let glb = parse_glb(&reexport_path)?;
    let (front, vertex_normals) = plane_normals(&glb, "mirror_plane")?;
    assert!(!front.is_empty(), "mirror_plane has no triangles");
    for normal in front.into_iter().chain(vertex_normals) {
        assert!(
            (normal - DVec3::Y).length() < 1.0e-6,
            "reimported mirror_plane normal {normal:?} must agree with its winding"
        );
    }
    Ok(())
}
