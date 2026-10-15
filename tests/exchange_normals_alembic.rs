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
    TestResult, assert_matches_reference, export, export_strict, usd_world_corners,
    write_fixture_scene,
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
