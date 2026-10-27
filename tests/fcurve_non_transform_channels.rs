use std::{
    env,
    error::Error,
    io,
    path::Path,
    path::PathBuf,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

const TOLERANCE: f64 = 1.0e-4;
const FRAMES: [&str; 8] = ["2", "3", "4", "5", "6", "7", "8", "10"];

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

fn blender_522() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping non-transform F-Curve channels: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = checked_output(command, "checking Blender version")?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
        "POTTER_REQUIRE_BLENDER is set but this test requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping non-transform F-Curve channels: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn pot_json(command: Command, label: &str) -> Result<Value, Box<dyn Error>> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn import_blend_command(
    project: &Path,
    blend: &Path,
    blender: &Path,
    allow_lossy: bool,
) -> Command {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .arg("--file")
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
        .arg(blender);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    command.arg("--json");
    command
}

fn import_blend(
    project: &Path,
    blend: &Path,
    blender: &Path,
    allow_lossy: bool,
    label: &str,
) -> Result<Value, Box<dyn Error>> {
    pot_json(
        import_blend_command(project, blend, blender, allow_lossy),
        label,
    )
}

fn inspect_channel(
    project: &Path,
    frame: &str,
    object: &str,
    group: &str,
    field: &str,
) -> Result<f64, Box<dyn Error>> {
    let mut command = pot();
    command
        .arg("inspect")
        .arg(project)
        .args(["--frame", frame, "--json"]);
    let inspected = pot_json(command, &format!("inspecting {object} at frame {frame}"))?;
    let item = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["name"] == object))
        .ok_or_else(|| io::Error::other(format!("inspect omitted `{object}` at frame {frame}")))?;
    item[group][field].as_f64().ok_or_else(|| {
        io::Error::other(format!(
            "inspect omitted `{object}.{group}.{field}` at frame {frame}"
        ))
        .into()
    })
}

fn assert_channels_match(
    project: &Path,
    expected: &Value,
    base: &Value,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    for frame in FRAMES {
        for (object, group, field, key) in [
            ("F-Curve Camera", "camera", "lens_mm", "camera_lens_mm"),
            ("F-Curve Light", "light", "energy", "light_energy"),
            (
                "Driven F-Curve Camera",
                "camera",
                "lens_mm",
                "driven_camera_lens_mm",
            ),
        ] {
            let expected_value = expected["frames"][frame][key]
                .as_f64()
                .ok_or_else(|| io::Error::other(format!("oracle lacks {key} at frame {frame}")))?;
            let base_value = base[key]
                .as_f64()
                .ok_or_else(|| io::Error::other(format!("oracle lacks base {key}")))?;
            let actual = inspect_channel(project, frame, object, group, field)?;
            assert!(
                (actual - expected_value).abs() <= TOLERANCE,
                "{label}: {object}.{key} at frame {frame}: expected {expected_value}, got {actual}"
            );
            assert!(
                (actual - base_value).abs() > TOLERANCE,
                "{label}: {object}.{key} at frame {frame} stayed at its unkeyed base {base_value}"
            );
        }
    }
    Ok(())
}
fn assert_nonobject_driver_targets_dropped(
    project: &Path,
    base: &Value,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    for frame in FRAMES {
        for (object, group, field, key) in [
            (
                "Scene Driver Light",
                "light",
                "energy",
                "scene_driver_light_energy",
            ),
            (
                "Data Target Camera",
                "camera",
                "lens_mm",
                "data_target_camera_lens_mm",
            ),
        ] {
            let expected_base = base[key]
                .as_f64()
                .ok_or_else(|| io::Error::other(format!("oracle lacks base {key}")))?;
            let actual = inspect_channel(project, frame, object, group, field)?;
            assert!(
                (actual - expected_base).abs() <= TOLERANCE,
                "{label}: unsupported {object} driver at frame {frame} was not dropped: expected base {expected_base}, got {actual}"
            );
        }
    }
    Ok(())
}

#[test]
fn camera_lens_and_light_energy_fcurves_import_inspect_and_round_trip() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fcurve_non_transform_channels");
    let expected: Value = serde_json::from_slice(&std::fs::read(fixture.join("expected.json"))?)?;
    assert!(
        expected["blender_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2")),
        "oracle must be generated with Blender 5.2.2, found {}",
        expected["blender_version"]
    );

    let directory = tempdir()?;
    let root = directory.path();
    let strict_project = root.join("strict-project");
    let mut init = pot();
    init.arg("init").arg(&strict_project).arg("--json");
    checked_output(init, "initializing strict channel project")?;
    let strict = import_blend_command(
        &strict_project,
        &fixture.join("fcurve_non_transform_channels.blend"),
        &blender,
        false,
    )
    .output()?;
    assert!(
        !strict.status.success(),
        "non-Object driver targets must require an explicit lossy import"
    );
    let strict_report = format!(
        "{}{}",
        String::from_utf8_lossy(&strict.stdout),
        String::from_utf8_lossy(&strict.stderr)
    );
    assert!(
        strict_report.contains("blender.animation.driver_target_id_type"),
        "strict import must name the unsupported driver target: {strict_report}"
    );
    let project = root.join("project");
    let mut init = pot();
    init.arg("init").arg(&project).arg("--json");
    checked_output(init, "initializing channel project")?;
    let imported = import_blend(
        &project,
        &fixture.join("fcurve_non_transform_channels.blend"),
        &blender,
        true,
        "importing native Blender F-Curve fixture with expected driver losses",
    )?;
    let losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import losses are missing")?;
    assert_eq!(
        losses.len(),
        2,
        "the two non-Object driver targets must be losses"
    );
    assert_eq!(
        losses
            .iter()
            .filter(|loss| loss["feature_id"] == "blender.animation.driver_target_id_type")
            .count(),
        2,
        "Scene and Camera data-block targets must each be reported: {losses:?}"
    );
    assert_channels_match(&project, &expected, &expected["base"], "source import")?;
    assert_nonobject_driver_targets_dropped(&project, &expected["base"], "source import")?;

    let round_trip_blend = root.join("round-trip.blend");
    let mut export = pot();
    export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--allow-lossy", "--out"])
        .arg(&round_trip_blend)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    let exported = pot_json(export, "exporting imported F-Curve fixture")?;
    assert_eq!(
        exported["result"]["losses"]
            .as_array()
            .ok_or("export losses are missing")?
            .iter()
            .filter(|loss| loss["feature_id"] == "blender.animation.driver_target_id_type")
            .count(),
        2,
        "the lossy export must report both dropped data-block drivers"
    );

    let round_trip_project = root.join("round-trip-project");
    let mut init = pot();
    init.arg("init").arg(&round_trip_project).arg("--json");
    checked_output(init, "initializing round-trip channel project")?;
    let reimported = import_blend(
        &round_trip_project,
        &round_trip_blend,
        &blender,
        true,
        "reimporting exported F-Curve fixture",
    )?;
    assert_eq!(reimported["result"]["losses"], json!([]));
    assert_nonobject_driver_targets_dropped(
        &round_trip_project,
        &expected["base"],
        "round-trip import",
    )?;
    assert_channels_match(
        &round_trip_project,
        &expected,
        &expected["base"],
        "round-trip import",
    )?;
    Ok(())
}
