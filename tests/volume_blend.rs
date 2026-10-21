//! Volume settings through the `.blend` bridge. A material's `cycles.volume_sampling` and `cycles.volume_interpolation`
//! are Blender's values: Blender 5.2.2 writes them into a `.blend`, Potter imports them, exports them, and Blender reads
//! them back. A scene with `cycles.volume_biased` on is a reported loss, since Potter renders Cycles' unbiased result
//! and Cycles' biased ray marching (which reads `volume_step_rate`) is not implemented.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use blender_checked::blender_executable;
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const LOSS: &str = "blender.scene.volume_biased";

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn blender_path() -> TestResult<PathBuf> {
    blender_executable().ok_or_else(|| "a Blender executable is required (POTTER_BLENDER)".into())
}

/// Runs a Blender Python expression headless. A raised exception exits non-zero (`--python-exit-code 1`).
fn blender_python(blender: &Path, expression: &str) -> TestResult<Output> {
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
        ])
        .arg(expression)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "Blender failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}

/// Runs `pot` with `args` and `--json`, returning the envelope of a successful run.
fn pot_json(args: &[&str]) -> TestResult<Value> {
    let output = pot().args(args).arg("--json").output()?;
    if !output.status.success() {
        return Err(format!(
            "pot {args:?} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// The feature ids of the losses in an import or export envelope.
fn loss_features(report: &Value) -> Vec<String> {
    report["result"]["losses"]
        .as_array()
        .map_or_else(Vec::new, |losses| {
            losses
                .iter()
                .filter_map(|loss| loss["feature_id"].as_str().map(str::to_owned))
                .collect()
        })
}

/// A fresh project at `root/name`, with no content.
fn init_project(root: &Path, name: &str) -> TestResult<PathBuf> {
    let project = root.join(name);
    pot_json(&["init", &project.to_string_lossy()])?;
    Ok(project)
}

/// The import arguments for `source` into `project` at revision 0.
fn import_args(project: &Path, source: &Path, blender: &Path) -> Vec<String> {
    vec![
        "import".to_owned(),
        project.to_string_lossy().into_owned(),
        "--file".to_owned(),
        source.to_string_lossy().into_owned(),
        "--format".to_owned(),
        "blend".to_owned(),
        "--mode".to_owned(),
        "replace".to_owned(),
        "--base-revision".to_owned(),
        "0".to_owned(),
        "--blender".to_owned(),
        blender.to_string_lossy().into_owned(),
    ]
}

/// The material named `name` in a project's `scene.json`.
fn material_named<'a>(scene: &'a Value, name: &str) -> TestResult<&'a Value> {
    scene["materials"]
        .as_object()
        .ok_or("scene materials are not an object")?
        .values()
        .find(|material| material["name"].as_str() == Some(name))
        .ok_or_else(|| format!("no material named {name}").into())
}

fn read_scene(project: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

#[test]
fn material_volume_settings_round_trip_through_blend() -> TestResult {
    let blender = blender_path()?;
    let root = tempdir()?;
    let source = root.path().join("volume_material.blend");
    // Non-default values, so a dropped field would read back as the default and fail the check.
    blender_python(
        &blender,
        &format!(
            "import bpy; bpy.ops.mesh.primitive_cube_add(); m = bpy.data.materials.new('VolumeMat'); \
             m.cycles.volume_interpolation = 'CUBIC'; m.cycles.volume_sampling = 'EQUIANGULAR'; \
             bpy.context.object.data.materials.append(m); bpy.ops.wm.save_as_mainfile(filepath={:?})",
            source.to_string_lossy()
        ),
    )?;

    let project = init_project(root.path(), "imported")?;
    let args = import_args(&project, &source, &blender);
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    pot_json(&refs)?;
    let scene = read_scene(&project)?;
    let material = material_named(&scene, "VolumeMat")?;
    assert_eq!(material["volume_interpolation"], "cubic");
    assert_eq!(material["volume_sampling"], "equiangular");

    // Potter's bridge writes both fields back into a `.blend`, which Blender reads with its own properties.
    let exported = root.path().join("exported.blend");
    pot_json(&[
        "export",
        &project.to_string_lossy(),
        "--format",
        "blend",
        "--out",
        &exported.to_string_lossy(),
        "--blender",
        &blender.to_string_lossy(),
    ])?;
    blender_python(
        &blender,
        &format!(
            "import bpy; bpy.ops.wm.open_mainfile(filepath={:?}); m = bpy.data.materials['VolumeMat']; \
             assert m.cycles.volume_interpolation == 'CUBIC', m.cycles.volume_interpolation; \
             assert m.cycles.volume_sampling == 'EQUIANGULAR', m.cycles.volume_sampling",
            exported.to_string_lossy()
        ),
    )?;
    Ok(())
}

#[test]
fn volume_biased_scene_is_a_reported_loss_and_the_default_is_not() -> TestResult {
    let blender = blender_path()?;
    let root = tempdir()?;
    let biased = root.path().join("biased.blend");
    blender_python(
        &blender,
        &format!(
            "import bpy; bpy.context.scene.cycles.volume_biased = True; \
             bpy.ops.wm.save_as_mainfile(filepath={:?})",
            biased.to_string_lossy()
        ),
    )?;
    let plain = root.path().join("plain.blend");
    blender_python(
        &blender,
        &format!(
            "import bpy; bpy.context.scene.cycles.volume_biased = False; \
             bpy.ops.wm.save_as_mainfile(filepath={:?})",
            plain.to_string_lossy()
        ),
    )?;

    // A strict import of the biased scene must refuse, and name the loss.
    let strict_project = init_project(root.path(), "strict")?;
    let strict = pot()
        .args(import_args(&strict_project, &biased, &blender))
        .arg("--json")
        .output()?;
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&strict.stdout),
        String::from_utf8_lossy(&strict.stderr)
    );
    assert!(
        !strict.status.success() && report.contains(LOSS),
        "a biased scene must not import silently: {report}"
    );

    // With --allow-lossy, the biased scene reports the loss and the default scene does not.
    let lossy_project = init_project(root.path(), "lossy")?;
    let mut args = import_args(&lossy_project, &biased, &blender);
    args.push("--allow-lossy".to_owned());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let lossy = pot_json(&refs)?;
    assert!(
        loss_features(&lossy).contains(&LOSS.to_owned()),
        "the biased scene must report {LOSS}: {:?}",
        loss_features(&lossy)
    );

    let control_project = init_project(root.path(), "control")?;
    let mut args = import_args(&control_project, &plain, &blender);
    args.push("--allow-lossy".to_owned());
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let control = pot_json(&refs)?;
    assert!(
        !loss_features(&control).contains(&LOSS.to_owned()),
        "a default scene must not report {LOSS}: {:?}",
        loss_features(&control)
    );
    Ok(())
}
