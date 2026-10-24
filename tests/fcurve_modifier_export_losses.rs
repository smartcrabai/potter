use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply_modifier_scene(scene: &Path) -> Result<(), Box<dyn Error>> {
    let init = pot().args(["init"]).arg(scene).arg("--json").output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );

    let operations = scene
        .parent()
        .ok_or("scene directory has no parent")?
        .join("fcurve-modifier-operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create", "id":"animated", "kind":"box", "params":{"size":2.0}},
                {"op":"action.create", "id":"motion", "name":"Motion"},
                {"op":"action.update", "target":{"id":"motion"}, "set":{"fcurves":[{
                    "path":"transform.translation", "index":0,
                    "keyframes":[{"frame":1.0,"value":0.0},{"frame":24.0,"value":1.0}],
                    "modifiers":[{"type":"NOISE","properties":{}}]
                }]}},
                {"op":"nla.track_create","target":{"id":"animated"},"id":"motion_track","name":"Motion"},
                {"op":"nla.strip_create","target":{"id":"animated"},"track":"motion_track",
                 "id":"motion_strip","action":"motion","frame_start":1.0,"frame_end":24.0,
                 "action_frame_start":1.0,"action_frame_end":24.0,"scale":1.0,"repeat":1.0,
                 "blend_type":"replace","influence":1.0,"extrapolation":"hold",
                 "blend_in":0.0,"blend_out":0.0}
            ]
        }))?,
    )?;
    let applied = pot()
        .args(["apply"])
        .arg(scene)
        .args(["--file"])
        .arg(&operations)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    Ok(())
}

fn export(
    scene: &Path,
    format: &str,
    output: &Path,
    allow_lossy: bool,
) -> Result<std::process::Output, Box<dyn Error>> {
    let mut command = pot();
    command
        .args(["export"])
        .arg(scene)
        .args(["--format", format, "--out"])
        .arg(output);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

fn modifier_loss(losses: &[Value]) -> Option<&Value> {
    losses
        .iter()
        .find(|loss| loss["feature_id"] == "animation.fcurve_modifier")
}

#[test]
fn fcurve_modifier_losses_refuse_strict_and_report_lossy_exports() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    apply_modifier_scene(&scene)?;

    for (format, extension) in [("fbx", "fbx"), ("glb", "glb"), ("usda", "usda")] {
        let output = directory.path().join(format!("scene.{extension}"));
        let strict = export(&scene, format, &output, false)?;
        assert!(
            !strict.status.success(),
            "strict {format} export accepted an F-Curve modifier"
        );
        assert!(
            !output.exists(),
            "strict {format} export created an output file"
        );
        let strict_report: Value = serde_json::from_slice(&strict.stdout)?;
        assert_eq!(strict_report["error"]["code"], "UNREPRESENTABLE_FEATURE");
        let strict_losses = strict_report["result"]["losses"]
            .as_array()
            .ok_or("strict export losses are missing")?;
        let strict_loss = modifier_loss(strict_losses).ok_or_else(|| {
            format!("strict {format} export did not report the modifier: {strict_report}")
        })?;
        assert_eq!(strict_loss["data_id"], "motion");
        for context in ["Motion", "transform.translation", "NOISE"] {
            assert!(
                strict_loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains(context)),
                "strict {format} loss is missing {context} context: {strict_loss}"
            );
        }

        let lossy = export(&scene, format, &output, true)?;
        assert!(
            lossy.status.success(),
            "lossy {format} export failed: {}",
            String::from_utf8_lossy(&lossy.stdout)
        );
        let lossy_report: Value = serde_json::from_slice(&lossy.stdout)?;
        let lossy_losses = lossy_report["result"]["losses"]
            .as_array()
            .ok_or("lossy export losses are missing")?;
        let lossy_loss = modifier_loss(lossy_losses).ok_or_else(|| {
            format!("lossy {format} export did not report the modifier: {lossy_report}")
        })?;
        assert_eq!(lossy_loss, strict_loss);
        assert!(
            output.is_file(),
            "lossy {format} export did not create output"
        );
    }
    Ok(())
}
