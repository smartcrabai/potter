//! Blender-generated disk caches must survive Potter import while the memory-only
//! baked cloth cache is reported as lossy; disk caches evaluate at baked, interpolated,
//! fractional, and out-of-range frames.

use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;
use serde_json::Value;
use tempfile::tempdir;

const GENERATOR: &str = include_str!("fixtures/point_cache_baked/generate.py");
const POSITION_TOLERANCE: f64 = 1.0e-5;
const SAMPLE_FRAMES: &[f64] = &[0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 9.0, 10.0];

type TestResult<T> = Result<T, Box<dyn Error>>;

fn blender_required() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

fn blender_522() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        assert!(
            !blender_required(),
            "POTTER_REQUIRE_BLENDER is set but Blender 5.2.2 is unavailable"
        );
        eprintln!("skipping baked point-cache parity: Blender is unavailable");
        return Ok(None);
    };
    let output = Command::new(&blender).arg("--version").output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "Blender version check failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        !blender_required(),
        "POTTER_REQUIRE_BLENDER is set but baked point-cache parity requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping baked point-cache parity: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn checked_output(mut command: Command, label: &str) -> TestResult<Output> {
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

fn checked_json(command: Command, label: &str) -> TestResult<Value> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn generate_fixture(blender: &Path, root: &Path) -> TestResult<()> {
    let script = root.join("generate.py");
    fs::write(&script, GENERATOR)?;
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(root);
    checked_output(command, "Blender baked point-cache fixture")?;
    Ok(())
}

fn inspect(project: &Path, id: &str, frame: f64) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .args(["inspect"])
        .arg(project)
        .args(["--id", id, "--frame"])
        .arg(frame.to_string())
        .arg("--json");
    checked_json(command, &format!("pot inspect {id} at frame {frame}"))
}

fn copy_directory(source: &Path, destination: &Path) -> TestResult<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_directory(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn assert_positions_close(expected: &[[f64; 3]], actual: &[[f64; 3]], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: vertex count differs"
    );
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        for axis in 0..3 {
            assert!(
                (expected[axis] - actual[axis]).abs() <= POSITION_TOLERANCE,
                "{context}: vertex {index} axis {axis}: Blender={}, Potter={}, tolerance={POSITION_TOLERANCE}",
                expected[axis],
                actual[axis],
            );
        }
    }
}

#[test]
fn mixed_disk_and_memory_baked_caches_match_blender_playback() -> TestResult<()> {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    generate_fixture(&blender, root)?;
    let oracle: Value = serde_json::from_slice(&fs::read(root.join("oracle.json"))?)?;
    let objects = oracle["objects"]
        .as_object()
        .ok_or("Blender oracle objects are missing")?;

    let project = root.join("project");
    let mut initialize = Command::new(env!("CARGO_BIN_EXE_pot"));
    initialize.args(["init"]).arg(&project).arg("--json");
    checked_output(initialize, "pot init")?;

    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .args(["import"])
        .arg(&project)
        .arg("--file")
        .arg(root.join("source.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .args(["--allow-lossy", "--json"]);
    let imported = checked_json(import, "pot import mixed baked point caches")?;
    let losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import losses are missing")?;
    let baked_cache_losses: Vec<_> = losses
        .iter()
        .filter(|loss| loss["feature_id"] == "blender.physics.point_cache.is_baked")
        .collect();
    assert_eq!(
        baked_cache_losses.len(),
        1,
        "lossy import must report only the memory cache's baked-cache loss: {losses:?}"
    );
    assert_eq!(
        baked_cache_losses[0]["data_id"], "BakedMemoryCloth",
        "the baked-cache loss must belong to the memory-baked cloth: {baked_cache_losses:?}"
    );
    let exported_blend = root.join("exported-point-caches.blend");
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args(["export"])
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported_blend)
        .args(["--blender"])
        .arg(&blender)
        .args(["--allow-lossy", "--json"]);
    let export_result = checked_json(export, "pot export baked point caches")?;
    let export_losses = export_result["result"]["losses"]
        .as_array()
        .ok_or("export losses are missing")?;
    assert!(
        export_losses
            .iter()
            .any(|loss| loss["feature_id"] == "blender.physics.point_cache.is_baked"),
        "lossy .blend export must report the baked-cache loss: {export_losses:?}"
    );

    let reimport_project = root.join("exported-blend-reimport");
    let mut initialize_reimport = Command::new(env!("CARGO_BIN_EXE_pot"));
    initialize_reimport
        .args(["init"])
        .arg(&reimport_project)
        .arg("--json");
    checked_output(initialize_reimport, "pot init exported blend reimport")?;

    let mut reimport = Command::new(env!("CARGO_BIN_EXE_pot"));
    reimport
        .args(["import"])
        .arg(&reimport_project)
        .args(["--file"])
        .arg(&exported_blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let reimported = checked_json(reimport, "pot import exported baked point caches")?;
    let reimport_mappings = reimported["result"]["id_mappings"]
        .as_object()
        .ok_or("reimport ID mappings are missing")?;
    let reimport_scene: Value =
        serde_json::from_slice(&fs::read(reimport_project.join("scene.json"))?)?;
    for (name, property) in [
        ("ClothBaked", "physics_cloth"),
        ("SoftBodyBaked", "physics_soft_body"),
    ] {
        let id = reimport_mappings
            .get(&format!("Object:{name}"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("reimported object mapping for {name} is missing"))?;
        assert_eq!(
            reimport_scene["nodes"][id]["properties"][property]["point_cache"]["is_baked"], false,
            "{name}: reimported .blend cache must no longer be baked"
        );
    }

    let mappings = imported["result"]["id_mappings"]
        .as_object()
        .ok_or("import ID mappings are missing")?;
    let imported_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    for (name, property) in [
        ("ClothBaked", "physics_cloth"),
        ("SoftBodyBaked", "physics_soft_body"),
    ] {
        let id = mappings
            .get(&format!("Object:{name}"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("imported object mapping for {name} is missing"))?;
        let point_cache = &imported_scene["nodes"][id]["properties"][property]["point_cache"];
        assert_eq!(
            point_cache["is_baked"], true,
            "{name}: baked flag is retained"
        );
        assert!(
            point_cache["_potter_cache_locator"]["frames"]
                .as_array()
                .is_some_and(|frames| !frames.is_empty()),
            "{name}: supported cache must retain project frame resources"
        );
    }
    let memory_id = mappings
        .get("Object:MemoryClothBaked")
        .and_then(Value::as_str)
        .ok_or("memory-baked object mapping is missing")?;
    let memory_cache =
        &imported_scene["nodes"][memory_id]["properties"]["physics_cloth"]["point_cache"];
    assert_eq!(memory_cache["is_baked"], true);
    assert!(
        memory_cache.get("_potter_cache_locator").is_none(),
        "unsupported memory cache must not retain a disk-cache locator"
    );
    let memory_inspection = inspect(&project, memory_id, 2.0)?;
    assert!(
        memory_inspection["result"]["items"]
            .as_array()
            .is_some_and(|items| items
                .iter()
                .any(|item| item["id"].as_str() == Some(memory_id))),
        "the accepted memory-cache loss should evaluate using the unbaked solver"
    );

    let relocated_project = root.join("relocated-project");
    copy_directory(&project, &relocated_project)?;
    fs::remove_file(root.join("source.blend"))?;
    let source_backup = root.join("source.blend1");
    if source_backup.exists() {
        fs::remove_file(source_backup)?;
    }
    let mut removed_internal_cache = false;
    let mut removed_external_cache = false;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_dir() {
            if name == "softbody_external_cache" {
                removed_external_cache = true;
                fs::remove_dir_all(entry.path())?;
            } else if name.starts_with("blendcache_") {
                removed_internal_cache = true;
                fs::remove_dir_all(entry.path())?;
            }
        }
    }
    assert!(
        removed_internal_cache,
        "fixture internal cache directory was missing"
    );
    assert!(
        removed_external_cache,
        "fixture external cache directory was missing"
    );
    assert!(!root.join("source.blend").exists());
    for name in ["ClothBaked", "SoftBodyBaked"] {
        let id = mappings
            .get(&format!("Object:{name}"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("imported object mapping for {name} is missing"))?;
        let oracle_object = objects
            .get(name)
            .ok_or_else(|| format!("Blender oracle for {name} is missing"))?;
        let rest: Vec<[f64; 3]> = serde_json::from_value(oracle_object["rest_positions"].clone())?;
        let samples = oracle_object["samples"]
            .as_array()
            .ok_or_else(|| format!("Blender samples for {name} are missing"))?;
        assert_eq!(
            samples.len(),
            SAMPLE_FRAMES.len(),
            "{name}: unexpected Blender sample count"
        );
        let mut final_snapshot_differs_from_rest = false;
        for (sample, expected_frame) in samples.iter().zip(SAMPLE_FRAMES) {
            let frame = sample["frame"]
                .as_f64()
                .ok_or_else(|| format!("{name}: sample frame is not numeric"))?;
            assert_eq!(frame, *expected_frame, "{name}: sample frame order");
            let expected: Vec<[f64; 3]> = serde_json::from_value(sample["positions"].clone())?;
            let inspected = inspect(&project, id, frame)?;
            let items = inspected["result"]["items"]
                .as_array()
                .ok_or("pot inspect items are missing")?;
            let item = items
                .iter()
                .find(|item| item["id"].as_str() == Some(id))
                .ok_or_else(|| format!("pot inspect omitted {name} at frame {frame}"))?;
            let actual: Vec<[f64; 3]> =
                serde_json::from_value(item["evaluated_geometry"]["positions"].clone())?;
            assert_positions_close(&expected, &actual, &format!("{name} frame {frame}"));
            if frame.to_bits() == 9.0_f64.to_bits() {
                eprintln!(
                    "CLI smoke: pot inspect <project> --id {id} --frame 9 --json: first={:?}, last={:?}",
                    actual.first(),
                    actual.last()
                );
                final_snapshot_differs_from_rest =
                    actual.iter().zip(&rest).any(|(current, rest)| {
                        current
                            .iter()
                            .zip(rest)
                            .any(|(current, rest)| (current - rest).abs() > POSITION_TOLERANCE)
                    });
                let relocated = inspect(&relocated_project, id, frame)?;
                let relocated_item = relocated["result"]["items"]
                    .as_array()
                    .ok_or("relocated pot inspect items are missing")?
                    .iter()
                    .find(|item| item["id"].as_str() == Some(id))
                    .ok_or_else(|| {
                        format!("relocated pot inspect omitted {name} at frame {frame}")
                    })?;
                let relocated_positions: Vec<[f64; 3]> = serde_json::from_value(
                    relocated_item["evaluated_geometry"]["positions"].clone(),
                )?;
                assert_positions_close(
                    &expected,
                    &relocated_positions,
                    &format!("{name} relocated frame {frame}"),
                );
            }
        }
        assert!(
            final_snapshot_differs_from_rest,
            "{name}: baked frame 9 must differ from the unbaked rest mesh"
        );
    }
    Ok(())
}
