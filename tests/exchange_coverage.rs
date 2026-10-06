use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(scene: &Path) -> TestResult {
    let output = pot()
        .args([
            "init",
            scene.to_str().ok_or("scene path is not UTF-8")?,
            "--json",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn import(scene: &Path, file: &Path, allow_lossy: bool) -> TestResult<std::process::Output> {
    let mut command = pot();
    command
        .args([
            "import",
            scene.to_str().ok_or("scene path is not UTF-8")?,
            "--file",
        ])
        .arg(file)
        .args([
            "--format",
            "gltf",
            "--base-revision",
            "0",
            "--mode",
            "replace",
        ]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

fn output_json(output: &std::process::Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn gltf_import_requires_loss_permission_then_commits_when_allowed() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let source = directory.path().join("unknown-extension.gltf");
    init(&scene)?;
    fs::write(
        &source,
        serde_json::to_vec(&json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [] }],
            "nodes": [],
            "extensionsUsed": ["VENDOR_unknown_feature"]
        }))?,
    )?;

    let rejected = import(&scene, &source, false)?;
    assert_eq!(rejected.status.code(), Some(4));
    let rejection = output_json(&rejected)?;
    assert_eq!(rejection["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert_eq!(
        rejection["error"]["details"]["result"]["losses"][0]["feature_id"],
        "gltf.extension"
    );

    let accepted = import(&scene, &source, true)?;
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stdout)
    );
    let result = output_json(&accepted)?["result"].clone();
    assert_eq!(result["committed"], true);
    assert_eq!(result["base_revision"], 0);
    assert_eq!(result["candidate_revision"], 1);
    assert_eq!(result["losses"][0]["feature_id"], "gltf.extension");
    Ok(())
}
