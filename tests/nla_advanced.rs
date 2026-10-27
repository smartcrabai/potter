use std::{
    error::Error,
    fs, io,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::{Value, json};
use tempfile::tempdir;

const EXPECTED: &str = include_str!("fixtures/nla_advanced/blender_expected.json");
const MATRIX_TOLERANCE: f64 = 1.0e-5;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked_output(mut command: Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn init(project: &Path) -> Result<(), Box<dyn Error>> {
    let mut command = pot();
    command.arg("init").arg(project).arg("--json");
    checked_output(command, "pot init")?;
    Ok(())
}

fn import_command(project: &Path, blend: &Path, blender: &Path) -> Command {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .args(["--file"])
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .arg("--json");
    command
}

fn node_by_name<'a>(scene: &'a Value, name: &str) -> Result<&'a Value, Box<dyn Error>> {
    scene["nodes"]
        .as_object()
        .and_then(|nodes| nodes.values().find(|node| node["name"] == name))
        .ok_or_else(|| io::Error::other(format!("imported node `{name}` is missing")).into())
}

fn first_strip(node: &Value) -> Result<&Value, Box<dyn Error>> {
    node["nla_tracks"][0]["strips"]
        .as_array()
        .and_then(|strips| strips.first())
        .ok_or_else(|| io::Error::other("imported NLA strip is missing").into())
}

fn assert_matrix(actual: &Value, expected: &Value, context: &str) -> Result<(), Box<dyn Error>> {
    let actual = actual
        .as_array()
        .filter(|components| components.len() == 16)
        .ok_or_else(|| io::Error::other(format!("{context}: Potter matrix is malformed")))?;
    let expected = expected
        .as_array()
        .filter(|components| components.len() == 16)
        .ok_or_else(|| io::Error::other(format!("{context}: Blender matrix is malformed")))?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_f64().ok_or_else(|| {
            io::Error::other(format!("{context}: Potter matrix value is not numeric"))
        })?;
        let expected = expected.as_f64().ok_or_else(|| {
            io::Error::other(format!("{context}: Blender matrix value is not numeric"))
        })?;
        assert!(
            (actual - expected).abs() <= MATRIX_TOLERANCE,
            "{context}: matrix element {index} differs: Potter={actual}, Blender={expected}"
        );
    }
    Ok(())
}

fn assert_matrix_bits(
    actual: &Value,
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let actual = actual
        .as_array()
        .filter(|components| components.len() == 16)
        .ok_or_else(|| io::Error::other(format!("{context}: Potter matrix is malformed")))?;
    let expected = expected
        .as_array()
        .filter(|components| components.len() == 16)
        .ok_or_else(|| io::Error::other(format!("{context}: Blender matrix is malformed")))?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.as_f64().ok_or_else(|| {
            io::Error::other(format!("{context}: Potter matrix value is not numeric"))
        })?;
        let expected = expected.as_f64().ok_or_else(|| {
            io::Error::other(format!("{context}: Blender matrix value is not numeric"))
        })?;
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: matrix element {index} differs: Potter={actual:?}, Blender={expected:?}"
        );
    }
    Ok(())
}

fn assert_number_close(value: &Value, expected: f64, context: &str) -> Result<(), Box<dyn Error>> {
    let actual = value
        .as_f64()
        .ok_or_else(|| io::Error::other(format!("{context}: value is not numeric")))?;
    assert!(
        (actual - expected).abs() <= MATRIX_TOLERANCE,
        "{context}: Potter={actual}, expected={expected}"
    );
    Ok(())
}

fn assert_imported_controls(scene: &Value) -> Result<(), Box<dyn Error>> {
    let timing = first_strip(node_by_name(scene, "NLA_TimingRangeReverse")?)?;
    assert!(timing["reverse"].as_bool().unwrap_or(false));
    assert!(timing["use_sync_length"].as_bool().unwrap_or(false));
    assert_eq!(timing["action_frame_start"], 2.0);
    assert_eq!(timing["action_frame_end"], 7.0);
    assert_eq!(timing["scale"], 1.25);
    assert_eq!(timing["repeat"], 1.5);

    for (name, blend) in [
        ("NLA_BlendReplace", "replace"),
        ("NLA_BlendAdd", "add"),
        ("NLA_BlendCombine", "combine"),
        ("NLA_BlendSubtract", "subtract"),
        ("NLA_BlendMultiply", "multiply"),
    ] {
        assert_eq!(
            first_strip(node_by_name(scene, name)?)?["blend_type"],
            blend,
            "{name}"
        );
    }

    for (name, blend) in [
        ("NLA_Quaternion270", "combine"),
        ("NLA_Quaternion270Replace", "replace"),
    ] {
        let strip = first_strip(node_by_name(scene, name)?)?;
        assert_eq!(strip["blend_type"], blend);
        assert_number_close(&strip["influence"], 0.25, name)?;
        assert!(
            strip["use_animated_influence"].as_bool().unwrap_or(false),
            "{name}"
        );
    }

    let influence_blend = first_strip(node_by_name(scene, "NLA_InfluenceBlend")?)?;
    assert_number_close(&influence_blend["influence"], 0.65, "NLA strip influence")?;
    assert!(
        influence_blend["use_animated_influence"]
            .as_bool()
            .unwrap_or(false)
    );
    assert_eq!(influence_blend["blend_in"], 2.0);
    assert_eq!(influence_blend["blend_out"], 2.0);
    let fade = first_strip(node_by_name(scene, "NLA_BlendFade")?)?;
    assert_eq!(fade["blend_in"], 2.0);
    assert_eq!(fade["blend_out"], 2.0);
    let animated_influence = first_strip(node_by_name(scene, "NLA_AnimatedInfluence")?)?;
    let influence_curve = animated_influence["influence_curve"]["keyframes"]
        .as_array()
        .ok_or_else(|| io::Error::other("animated influence F-curve keyframes are missing"))?;
    assert_eq!(influence_curve.len(), 2);
    assert_eq!(influence_curve[0]["value"], 0.25);
    assert_eq!(influence_curve[1]["value"], 0.75);
    assert!(
        influence_curve
            .iter()
            .all(|keyframe| keyframe["interpolation"] == "bezier")
    );

    let muted = node_by_name(scene, "NLA_TrackMute")?;
    assert!(muted["nla_tracks"][1]["mute"].as_bool().unwrap_or(false));
    let strip_muted = first_strip(node_by_name(scene, "NLA_StripMute")?)?;
    assert!(strip_muted["mute"].as_bool().unwrap_or(false));
    let solo = node_by_name(scene, "NLA_TrackSolo")?;
    assert!(solo["nla_tracks"][1]["solo"].as_bool().unwrap_or(false));
    let muted_domain = first_strip(node_by_name(scene, "NLA_MutedStripDomainReset")?)?;
    assert!(muted_domain["mute"].as_bool().unwrap_or(false));
    let solo_action = node_by_name(scene, "NLA_SoloWithActiveAction")?;
    assert!(
        solo_action["nla_tracks"][1]["solo"]
            .as_bool()
            .unwrap_or(false)
    );
    assert_eq!(solo_action["action_blend_type"], "add");
    assert_number_close(
        &solo_action["action_influence"],
        0.5,
        "solo active Action influence",
    )?;

    let solo_domain = node_by_name(scene, "NLA_SoloStripDomainReset")?;
    assert_eq!(
        solo_domain["transform"]["translation"][0], 2.0,
        "solo reset-domain fixture must start with location.x=2"
    );
    let solo_domain_tracks = solo_domain["nla_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("solo reset-domain track is missing"))?;
    assert_eq!(solo_domain_tracks.len(), 1);
    assert!(solo_domain_tracks[0]["solo"].as_bool().unwrap_or(false));
    assert_eq!(solo_domain["action_blend_type"], "replace");
    let action_id = solo_domain["action"]
        .as_str()
        .ok_or_else(|| io::Error::other("solo reset-domain active Action is missing"))?;
    let action_curves = scene["actions"][action_id]["fcurves"]
        .as_array()
        .ok_or_else(|| io::Error::other("solo reset-domain Action curves are missing"))?;
    let active_x = action_curves
        .iter()
        .find(|curve| curve["path"] == "transform.translation" && curve["index"] == 0)
        .ok_or_else(|| io::Error::other("solo reset-domain Action location.x curve is missing"))?;
    assert!(
        active_x["keyframes"]
            .as_array()
            .is_some_and(|keyframes| keyframes.len() == 2
                && keyframes.iter().all(|keyframe| keyframe["value"] == 4.0)),
        "solo reset-domain Action location.x keys were not imported"
    );
    let rotation_strip = first_strip(solo_domain)?;
    let strip_action_id = rotation_strip["action"]
        .as_str()
        .ok_or_else(|| io::Error::other("solo reset-domain strip Action is missing"))?;
    let strip_curves = scene["actions"][strip_action_id]["fcurves"]
        .as_array()
        .ok_or_else(|| io::Error::other("solo reset-domain strip curves are missing"))?;
    assert_eq!(strip_curves.len(), 1);
    assert_eq!(strip_curves[0]["path"], "transform.rotation_euler");
    assert_eq!(strip_curves[0]["index"], 2);

    let static_euler = node_by_name(scene, "NLA_StaticEulerMatrixBasis")?;
    assert!(static_euler["action"].is_null());
    assert!(
        static_euler["nla_tracks"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "static Euler basis object must not have NLA tracks"
    );
    let matrix_basis = static_euler["blender_matrix_basis"]["matrix_basis"]
        .as_array()
        .ok_or_else(|| io::Error::other("static Euler matrix_basis capture is missing"))?;
    assert_eq!(matrix_basis.len(), 16);
    for (name, extrapolation) in [
        ("NLA_ActiveActionHold", "hold"),
        ("NLA_ActiveActionHoldForward", "hold_forward"),
        ("NLA_ActiveActionNothing", "nothing"),
    ] {
        let action = node_by_name(scene, name)?;
        let tracks = action["nla_tracks"]
            .as_array()
            .ok_or_else(|| io::Error::other(format!("{name}: baseline NLA track is missing")))?;
        assert_eq!(
            tracks.len(),
            1,
            "{name} must sample its active Action without a strip time map"
        );
        assert_eq!(
            tracks[0]["strips"].as_array().map(Vec::len),
            Some(1),
            "{name} must only use its baseline strip"
        );
        assert_eq!(action["action_extrapolation"], extrapolation, "{name}");
    }

    let auto = node_by_name(scene, "NLA_AutoBlend")?;
    let auto_tracks = auto["nla_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("automatic blend tracks are missing"))?;
    assert_eq!(auto_tracks.len(), 2);
    for track in auto_tracks {
        let strips = track["strips"]
            .as_array()
            .ok_or_else(|| io::Error::other("automatic blend strips are missing"))?;
        assert_eq!(strips.len(), 1);
        assert!(strips[0]["use_auto_blend"].as_bool().unwrap_or(false));
    }
    let auto_three = node_by_name(scene, "NLA_AutoBlendThreeTrack")?;
    let auto_three_tracks = auto_three["nla_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("three-track automatic blend tracks are missing"))?;
    assert_eq!(auto_three_tracks.len(), 3);
    for track in auto_three_tracks {
        let strips = track["strips"]
            .as_array()
            .ok_or_else(|| io::Error::other("three-track automatic blend strips are missing"))?;
        assert_eq!(strips.len(), 1);
        assert!(strips[0]["use_auto_blend"].as_bool().unwrap_or(false));
    }

    let time = first_strip(node_by_name(scene, "NLA_AnimatedStripTime")?)?;
    assert!(time["use_animated_time"].as_bool().unwrap_or(false));
    assert!(time["use_animated_time_cyclic"].as_bool().unwrap_or(false));
    let time_curve = time["strip_time_curve"]["keyframes"]
        .as_array()
        .ok_or_else(|| io::Error::other("constant strip-time F-curve keyframes are missing"))?;
    assert_eq!(time_curve.len(), 3);
    assert!(
        time_curve
            .iter()
            .all(|keyframe| keyframe["interpolation"] == "constant")
    );

    let active = node_by_name(scene, "NLA_ActiveActionControls")?;
    assert_eq!(active["action_blend_type"], "add");
    assert_number_close(&active["action_influence"], 0.4, "active Action influence")?;
    assert_eq!(active["action_extrapolation"], "nothing");
    Ok(())
}

#[test]
fn blender_nla_advanced_cli_inspection_matches_the_committed_oracle() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender NLA advanced parity: Blender is unavailable");
        return Ok(());
    };
    let expected: Value = serde_json::from_str(EXPECTED)?;
    assert_eq!(expected["blender_version"], "5.2.2 LTS");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nla_advanced");
    let directory = tempdir()?;
    let project = directory.path().join("project");
    init(&project)?;
    let imported = checked_output(
        import_command(&project, &fixtures.join("nla_advanced.blend"), &blender),
        "pot import of the Blender NLA oracle",
    )?;
    let imported: Value = serde_json::from_slice(&imported.stdout)?;
    assert_eq!(imported["result"]["losses"], Value::Array(Vec::new()));

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_imported_controls(&scene)?;
    let pushdown_input = node_by_name(&scene, "NLA_PushDownActiveAction")?;
    assert_eq!(pushdown_input["action_blend_type"], "add");
    assert_number_close(
        &pushdown_input["action_influence"],
        0.5,
        "push-down input Action influence",
    )?;
    assert_eq!(pushdown_input["action_extrapolation"], "nothing");
    assert!(
        pushdown_input["nla_tracks"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "push-down input must have no NLA tracks"
    );
    let pushdown_action_id = pushdown_input["action"]
        .as_str()
        .ok_or_else(|| io::Error::other("push-down input active Action is missing"))?
        .to_owned();
    let pushdown_node_id = scene["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"] == "NLA_PushDownActiveAction")
                .map(|(id, _)| id.to_owned())
        })
        .ok_or_else(|| io::Error::other("push-down input node ID is missing"))?;
    let pushdown_slot_id = scene["actions"]
        .get(pushdown_action_id.as_str())
        .and_then(|action| action["active_slots"].get(pushdown_node_id.as_str()))
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other("push-down input Action slot binding is missing"))?
        .to_owned();
    let objects = expected["objects"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender oracle object list is missing"))?;
    let frames = expected["frames"]
        .as_object()
        .ok_or_else(|| io::Error::other("Blender oracle frames are missing"))?;
    for (frame, frame_expected) in frames {
        let mut command = pot();
        command
            .arg("inspect")
            .arg(&project)
            .args(["--frame", frame.as_str(), "--json"]);
        let inspected = checked_output(command, &format!("pot inspect at frame {frame}"))?;
        let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
        let items = inspected["result"]["items"]
            .as_array()
            .ok_or_else(|| io::Error::other(format!("frame {frame}: inspect items are missing")))?;
        for name in objects {
            let name = name
                .as_str()
                .ok_or_else(|| io::Error::other("Blender oracle object name is invalid"))?;
            if name == "NLA_PushDownActiveAction" {
                continue;
            }
            let item = items
                .iter()
                .find(|item| item["name"] == name)
                .ok_or_else(|| {
                    io::Error::other(format!("frame {frame}: inspect omitted `{name}`"))
                })?;
            let world_matrix = &item["transform"]["world"]["matrix"];
            if name == "NLA_StaticEulerMatrixBasis" {
                assert_matrix_bits(
                    world_matrix,
                    &frame_expected[name]["matrix_world"],
                    &format!("{name} exact world transform at frame {frame}"),
                )?;
                assert_matrix_bits(
                    &frame_expected[name]["matrix_basis"],
                    &frame_expected[name]["matrix_world"],
                    &format!("{name} basis/evaluation at frame {frame}"),
                )?;
                let imported_static = node_by_name(&scene, name)?;
                assert_matrix_bits(
                    &imported_static["blender_matrix_basis"]["matrix_basis"],
                    &frame_expected[name]["matrix_basis"],
                    &format!("{name} imported matrix_basis at frame {frame}"),
                )?;
            } else {
                assert_matrix(
                    world_matrix,
                    &frame_expected[name]["matrix_world"],
                    &format!("{name} world transform at frame {frame}"),
                )?;
            }
            if name == "NLA_SoloStripDomainReset" {
                let expected_matrix =
                    frame_expected[name]["matrix_world"]
                        .as_array()
                        .ok_or_else(|| {
                            io::Error::other("Blender solo reset-domain matrix is missing")
                        })?;
                assert_eq!(
                    expected_matrix[12].as_f64(),
                    Some(0.0),
                    "Blender solo reset-domain must reset location.x at frame {frame}"
                );
                let matrix = world_matrix
                    .as_array()
                    .ok_or_else(|| io::Error::other("solo reset-domain world matrix is missing"))?;
                assert_eq!(
                    matrix[12].as_f64(),
                    Some(0.0),
                    "solo reset-domain must reset location.x to its RNA default at frame {frame}"
                );
                assert!(
                    matrix[1].as_f64().is_some_and(|value| value > 0.3),
                    "solo reset-domain rotation.z was not applied at frame {frame}"
                );
            }
            if !frame_expected[name]["bones"].is_null() {
                let evaluated_bones = item["evaluated_bones"].as_object().ok_or_else(|| {
                    io::Error::other(format!("{name}: inspect omitted evaluated bones"))
                })?;
                let bone_matrix = evaluated_bones.values().next().ok_or_else(|| {
                    io::Error::other(format!("{name}: evaluated bone matrix is missing"))
                })?;
                assert_matrix(
                    bone_matrix,
                    &frame_expected[name]["bones"]["OracleBone"],
                    &format!("{name} OracleBone pose at frame {frame}"),
                )?;
            }
        }
    }

    let pushdown_expected = &expected["push_down"];
    assert_eq!(pushdown_expected["action"], Value::Null);
    assert_eq!(pushdown_expected["action_slot"], Value::Null);
    assert_eq!(pushdown_expected["action_blend_type"], "add");
    assert_eq!(pushdown_expected["action_extrapolation"], "nothing");
    assert_number_close(
        &pushdown_expected["action_influence"],
        0.5,
        "Blender push-down Action influence",
    )?;
    let pushdown_operations = directory.path().join("push-down-operations.json");
    fs::write(
        &pushdown_operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": scene["revision"],
            "operations": [{
                "op": "nla.push_down",
                "target": {"id": pushdown_node_id},
            }],
        }))?,
    )?;
    let mut apply_pushdown = pot();
    apply_pushdown
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&pushdown_operations)
        .arg("--json");
    checked_output(apply_pushdown, "pot apply NLA push-down")?;

    let pushed_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let pushed_node = node_by_name(&pushed_scene, "NLA_PushDownActiveAction")?;
    assert!(
        pushed_node["action"].is_null(),
        "active Action was not cleared"
    );
    assert!(
        pushed_scene["actions"][pushdown_action_id.as_str()]["active_slots"]
            .get(pushdown_node_id.as_str())
            .is_none(),
        "active Action slot association was not cleared"
    );
    assert_eq!(
        pushed_node["action_blend_type"],
        pushdown_expected["action_blend_type"]
    );
    assert_eq!(
        pushed_node["action_extrapolation"],
        pushdown_expected["action_extrapolation"]
    );
    assert_number_close(
        &pushed_node["action_influence"],
        pushdown_expected["action_influence"]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender push-down influence capture is missing"))?,
        "push-down active Action influence after unassignment",
    )?;
    let tracks = pushed_node["nla_tracks"]
        .as_array()
        .ok_or_else(|| io::Error::other("push-down NLA track is missing"))?;
    assert_eq!(tracks.len(), 1);
    let strips = tracks[0]["strips"]
        .as_array()
        .ok_or_else(|| io::Error::other("push-down NLA strip is missing"))?;
    assert_eq!(strips.len(), 1);
    let pushed_strip = &strips[0];
    assert_eq!(pushed_strip["action"], pushdown_action_id);
    assert_eq!(pushed_strip["action_slot"], pushdown_slot_id);
    let strip_expected = &pushdown_expected["strip"];
    for field in ["blend_type", "extrapolation", "use_animated_influence"] {
        assert_eq!(pushed_strip[field], strip_expected[field], "strip {field}");
    }
    for field in [
        "frame_start",
        "frame_end",
        "action_frame_start",
        "action_frame_end",
        "scale",
        "repeat",
        "influence",
    ] {
        assert_number_close(
            &pushed_strip[field],
            strip_expected[field].as_f64().ok_or_else(|| {
                io::Error::other(format!("Blender push-down strip {field} is missing"))
            })?,
            &format!("push-down strip {field}"),
        )?;
    }
    assert_eq!(pushed_strip["blend_type"], "add");
    assert_number_close(&pushed_strip["influence"], 0.5, "push-down strip influence")?;
    assert_eq!(pushed_strip["extrapolation"], "nothing");
    assert!(
        pushed_strip["use_animated_influence"]
            .as_bool()
            .unwrap_or(false)
    );
    let influence_keys = pushed_strip["influence_curve"]["keyframes"]
        .as_array()
        .ok_or_else(|| io::Error::other("push-down animated influence keys are missing"))?;
    let expected_influence_keys = strip_expected["influence_curve"]["keyframes"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender push-down influence capture is missing"))?;
    assert_eq!(
        influence_keys.len(),
        1,
        "push-down influence must be a constant one-key curve"
    );
    assert_number_close(
        &influence_keys[0]["frame"],
        expected_influence_keys[0]["frame"]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender push-down key frame is missing"))?,
        "push-down influence key frame",
    )?;
    assert_number_close(
        &influence_keys[0]["value"],
        expected_influence_keys[0]["value"]
            .as_f64()
            .ok_or_else(|| io::Error::other("Blender push-down key value is missing"))?,
        "push-down influence key value",
    )?;
    assert_eq!(
        influence_keys[0]["interpolation"],
        expected_influence_keys[0]["interpolation"]
    );
    assert_eq!(influence_keys[0]["value"], 0.5);

    for (frame, frame_expected) in frames {
        let mut command = pot();
        command
            .arg("inspect")
            .arg(&project)
            .args(["--frame", frame.as_str(), "--json"]);
        let inspected = checked_output(
            command,
            &format!("pot inspect after push-down at frame {frame}"),
        )?;
        let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
        let items = inspected["result"]["items"].as_array().ok_or_else(|| {
            io::Error::other(format!(
                "post-push-down frame {frame}: inspect items are missing"
            ))
        })?;
        let pushed_item = items
            .iter()
            .find(|item| item["name"] == "NLA_PushDownActiveAction")
            .ok_or_else(|| {
                io::Error::other(format!("frame {frame}: inspect omitted pushed-down Action"))
            })?;
        assert_matrix(
            &pushed_item["transform"]["world"]["matrix"],
            &frame_expected["NLA_PushDownActiveAction"]["matrix_world"],
            &format!("NLA_PushDownActiveAction world transform after push-down at frame {frame}"),
        )?;
    }

    let rejected_project = directory.path().join("unsupported-project");
    init(&rejected_project)?;
    let rejected = import_command(
        &rejected_project,
        &fixtures.join("unsupported_transition.blend"),
        &blender,
    )
    .output()?;
    assert!(
        !rejected.status.success(),
        "unsupported NLA transition imported successfully"
    );
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "blender.animation.nla_transition_strip"
    );
    assert_eq!(rejected["error"]["details"]["strip_type"], "TRANSITION");
    let tweak_project = directory.path().join("unsupported-tweak-project");
    init(&tweak_project)?;
    let tweak = import_command(
        &tweak_project,
        &fixtures.join("unsupported_tweak_mode.blend"),
        &blender,
    )
    .output()?;
    assert!(
        !tweak.status.success(),
        "NLA tweak mode imported successfully"
    );
    let tweak: Value = serde_json::from_slice(&tweak.stdout)?;
    assert_eq!(tweak["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        tweak["error"]["details"]["feature_id"],
        "blender.animation.nla_tweak_mode"
    );
    Ok(())
}
