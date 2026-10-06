use std::{error::Error, fs, path::Path};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn successful(output: &std::process::Output) -> Result<(), Box<dyn Error>> {
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).into_owned()).into(),
        );
    }
    Ok(())
}

fn render(scene: &Path, frames: &str, output: &Path) -> Result<(), Box<dyn Error>> {
    let rendered = pot()
        .arg("render")
        .arg(scene)
        .args([
            "--engine", "realtime", "--frames", frames, "--format", "png", "--out",
        ])
        .arg(output)
        .arg("--json")
        .output()?;
    successful(&rendered)
}

#[test]
fn sequence_filenames_identify_requested_frames_and_subframes() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    successful(&pot().arg("init").arg(&scene).arg("--json").output()?)?;

    let operations = directory.path().join("operations.json");
    fs::write(
        &operations,
        r#"{"schema_version":1,"base_revision":0,"operations":[
            {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":1,"resolution_y":1,"samples":1}}
        ]}"#,
    )?;
    successful(
        &pot()
            .arg("apply")
            .arg(&scene)
            .args(["--file"])
            .arg(&operations)
            .arg("--json")
            .output()?,
    )?;

    let integer_output = directory.path().join("integer-frames");
    render(&scene, "1:24:23", &integer_output)?;
    assert!(integer_output.join("frame_0001.png").is_file());
    assert!(integer_output.join("frame_0024.png").is_file());
    assert!(!integer_output.join("frame_0002.png").exists());
    let manifest: Value =
        serde_json::from_slice(&fs::read(integer_output.join("render.manifest.json"))?)?;
    assert_eq!(manifest["frames"][0]["frame"], 1.0);
    assert_eq!(manifest["frames"][0]["path"], "frame_0001.png");
    assert_eq!(manifest["frames"][1]["frame"], 24.0);
    assert_eq!(manifest["frames"][1]["path"], "frame_0024.png");

    let subframe_output = directory.path().join("subframe");
    render(&scene, "12.5:12.5", &subframe_output)?;
    assert!(subframe_output.join("frame_0012.5.png").is_file());
    Ok(())
}
