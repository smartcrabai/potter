#![expect(
    clippy::unwrap_used,
    reason = "integration-test setup uses fixed valid data"
)]

use std::{error::Error, fs, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn create_scene() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(initialized.status.success());
    Ok((directory, scene))
}

fn apply(scene: &std::path::Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let file = scene.with_extension("operations.json");
    fs::write(
        &file,
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
        )?,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn scene_document(scene: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

/// The record's `total_levels` is canonical. A multires modifier created later at a lower
/// viewport level must keep the stored total in its params, not the clamped viewport level.
#[test]
fn new_multires_modifier_writes_the_stored_total_levels() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_scene()?;
    apply(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
            {"op":"sculpt.multires","target":{"id":"body"},"id":"multi_top","levels":3,"render_levels":3},
            {"op":"modifier.delete","target":{"id":"body"},"id":"multi_top"},
            {"op":"sculpt.multires","target":{"id":"body"},"id":"multi_low","levels":1,"render_levels":1},
        ]),
    )?;
    let document = scene_document(&scene)?;
    let body = &document["nodes"]["body"];
    let modifier = body["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers
                .iter()
                .find(|modifier| modifier["id"] == "multi_low")
        })
        .ok_or("multi_low modifier was not created")?;
    assert_eq!(
        modifier["params"]["total_levels"],
        json!(3),
        "params must carry the stored total, not the viewport level"
    );
    assert_eq!(modifier["params"]["levels"], json!(1));
    let data_id = body["data"].as_str().ok_or("body has no mesh data")?;
    assert_eq!(
        document["data_blocks"][data_id]["mesh"]["attributes"]["multires_displacement"]["total_levels"],
        json!(3),
        "the mesh record keeps its stored total"
    );
    Ok(())
}
