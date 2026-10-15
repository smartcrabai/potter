#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{
    fs,
    process::{Command, Output},
};

use exchange_normals::{
    NORMAL_TOLERANCE, TestResult, assert_matches_reference, export, export_strict,
    usd_world_corners, write_fixture_scene,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> TestResult<Output> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "pot command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn export_fixture_abc(scene: &std::path::Path, out: &std::path::Path) -> TestResult {
    let output = export_strict(scene, "alembic", out)?;
    assert!(
        output.status.success(),
        "alembic export of shading attributes must not require --allow-lossy: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

#[test]
fn alembic_export_accepts_shading_attributes_without_allow_lossy() -> TestResult {
    let dir = tempdir()?;
    let scene = dir.path().join("scene");
    write_fixture_scene(&scene)?;
    let out = dir.path().join("fixture.abc");
    export_fixture_abc(&scene, &out)?;
    assert!(out.is_file(), "alembic file was not written");
    Ok(())
}

#[test]
fn alembic_round_trip_reproduces_blender_corner_normals() -> TestResult {
    let dir = tempdir()?;
    let scene = dir.path().join("scene");
    write_fixture_scene(&scene)?;
    let abc = dir.path().join("fixture.abc");
    export_fixture_abc(&scene, &abc)?;

    let project = dir.path().join("imported");
    run_ok(pot().arg("init").arg(&project).arg("--json"))?;
    let imported = run_ok(
        pot()
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(&abc)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ]),
    )?;
    let report: Value = serde_json::from_slice(&imported.stdout)?;
    assert_eq!(report["result"]["losses"], json!([]));

    let usda = export(&project, "usda", &dir.path().join("roundtrip.usda"))?;
    let observed = usd_world_corners(&fs::read_to_string(usda)?)?;
    assert_matches_reference(&observed)?;
    Ok(())
}

#[test]
fn alembic_import_restores_smooth_and_custom_shading() -> TestResult {
    let dir = tempdir()?;
    let scene = dir.path().join("scene");
    write_fixture_scene(&scene)?;
    let abc = dir.path().join("fixture.abc");
    export_fixture_abc(&scene, &abc)?;

    let project = dir.path().join("imported");
    run_ok(pot().arg("init").arg(&project).arg("--json"))?;
    run_ok(
        pot()
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(&abc)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ]),
    )?;

    // Each of the three boxes has f0 flat and f1..f5 smooth, and f2 carries custom normals.
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let blocks = document["data_blocks"]
        .as_object()
        .ok_or("scene.json has no data_blocks")?;
    let (mut smooth, mut flat, mut custom) = (0_usize, 0_usize, 0_usize);
    for block in blocks.values() {
        let Some(attributes) = block["mesh"]["attributes"].as_object() else {
            continue;
        };
        if let Some(values) = attributes
            .get("shade_smooth")
            .and_then(|attribute| attribute["values"].as_object())
        {
            for value in values.values() {
                if value == &json!(true) {
                    smooth += 1;
                } else {
                    flat += 1;
                }
            }
        }
        if attributes.contains_key("custom_normal") {
            custom += 1;
        }
    }
    assert_eq!((smooth, flat, custom), (15, 3, 3));
    Ok(())
}

fn apply_batch(scene: &std::path::Path, batch: &Value) -> TestResult {
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    run_ok(pot().arg("apply").arg(scene).arg("--file").arg(&batch_file))?;
    Ok(())
}

/// Imports `abc` into a fresh project under `dir` and returns the project path.
fn import_alembic(dir: &std::path::Path, abc: &std::path::Path) -> TestResult<std::path::PathBuf> {
    let project = dir.join("imported");
    run_ok(pot().arg("init").arg(&project).arg("--json"))?;
    run_ok(
        pot()
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(abc)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--allow-lossy",
                "--json",
            ]),
    )?;
    Ok(project)
}

/// Exports the scene's usda at `frame`, so its corner normals are Potter's for that sample.
fn export_usda_at_frame(scene: &std::path::Path, frame: f64, out: &std::path::Path) -> TestResult {
    run_ok(
        pot()
            .arg("export")
            .arg(scene)
            .args(["--format", "usda", "--frame"])
            .arg(frame.to_string())
            .arg("--out")
            .arg(out)
            .args(["--overwrite", "--allow-lossy", "--json"]),
    )?;
    Ok(())
}

/// A plane with a wave modifier and translation keys at frames 1 and 3. The keys hold the transform
/// at identity, so the object stays at the origin while its geometry changes between frames.
#[test]
fn alembic_animated_mesh_writes_each_frame_normals() -> TestResult {
    let dir = tempdir()?;
    let scene = dir.path().join("scene");
    run_ok(pot().arg("init").arg(&scene).arg("--json"))?;
    apply_batch(
        &scene,
        &json!({"schema_version": 1, "base_revision": 0, "operations": [
            {"op":"node.create","id":"wave","kind":"grid","params":{"size":2.0,"x_subdivisions":4,"y_subdivisions":4}},
            {"op":"modifier.create","target":{"id":"wave"},"id":"ripple","type":"wave","params":{"speed":0.25,"height":0.5,"width":1.5,"narrowness":1.5}},
            {"op":"action.create","id":"samples","name":"Samples"},
            {"op":"node.update","target":{"id":"wave"},"set":{"action":"samples"}},
            {"op":"keyframe.insert","target":{"id":"wave"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0},
            {"op":"keyframe.insert","target":{"id":"wave"},"path":"transform.translation","index":0,"frame":3.0,"value":0.0}
        ]}),
    )?;
    let abc = dir.path().join("animated.abc");
    export_fixture_abc(&scene, &abc)?;

    let project = import_alembic(dir.path(), &abc)?;
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let time_samples = document["data_blocks"]
        .as_object()
        .ok_or("scene.json has no data_blocks")?
        .values()
        .find_map(|block| block["mesh"]["attributes"].get("alembic_time_samples"))
        .ok_or("imported animated mesh has no alembic_time_samples")?;
    assert_eq!(time_samples["frames"], json!([1.0, 3.0]));
    let stored = time_samples["normals"]
        .as_array()
        .ok_or("per-frame normals are missing")?;
    assert_eq!(stored.len(), 2, "one normal set per frame");

    let mut per_frame = Vec::new();
    for (index, frame) in [1.0, 3.0].into_iter().enumerate() {
        let usda = dir.path().join(format!("frame-{index}.usda"));
        export_usda_at_frame(&scene, frame, &usda)?;
        let expected = usd_world_corners(&fs::read_to_string(&usda)?)?;
        let observed: Vec<[f64; 3]> = serde_json::from_value(stored[index].clone())?;
        assert_eq!(
            observed.len(),
            expected.len(),
            "frame {frame}: corner count"
        );
        for (corner, (normal, world)) in observed.iter().zip(&expected).enumerate() {
            let error = (0..3)
                .map(|axis| (normal[axis] - world.normal[axis]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error <= NORMAL_TOLERANCE,
                "frame {frame} corner {corner}: stored {normal:?} differs from {:?} (error {error})",
                world.normal
            );
        }
        per_frame.push(observed);
    }
    let change = per_frame[0]
        .iter()
        .zip(&per_frame[1])
        .flat_map(|(first, second)| (0..3).map(move |axis| (first[axis] - second[axis]).abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        change > 1.0e-3,
        "the fixture's normals must change between frames for this test to mean anything"
    );
    Ok(())
}

/// A zero-scale parent with a child must export, and the parent's scale must import back as zero.
#[test]
fn alembic_export_accepts_zero_scale_nodes() -> TestResult {
    let dir = tempdir()?;
    let scene = dir.path().join("scene");
    run_ok(pot().arg("init").arg(&scene).arg("--json"))?;
    apply_batch(
        &scene,
        &json!({"schema_version": 1, "base_revision": 0, "operations": [
            {"op":"node.create","id":"flat","kind":"box","params":{"size":2.0},"transform":{"translation":[0.0,0.0,0.0],"scale":[0.0,1.0,1.0]}},
            {"op":"node.create","id":"flat_child","kind":"box","params":{"size":2.0},"transform":{"translation":[0.0,3.0,0.0],"scale":[1.0,1.0,1.0]}},
            {"op":"node.parent","target":{"id":"flat_child"},"parent":"flat","keep_world":false}
        ]}),
    )?;
    let abc = dir.path().join("flat.abc");
    export_fixture_abc(&scene, &abc)?;

    let project = import_alembic(dir.path(), &abc)?;
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_eq!(
        document["nodes"]["flat"]["transform"]["scale"],
        json!([0.0, 1.0, 1.0])
    );
    assert_eq!(document["nodes"]["flat_child"]["parent"], json!("flat"));
    Ok(())
}
