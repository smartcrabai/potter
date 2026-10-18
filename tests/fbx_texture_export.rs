//! The FBX export CLI resolves texture assets under the project, so it works from any working directory, and an
//! EXTEND texture field survives the FBX round trip.

use std::{error::Error, fs, process::Command};

use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn assert_success(output: &std::process::Output, what: &str) {
    assert!(
        output.status.success(),
        "{what} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn fbx_export_resolves_texture_assets_from_the_project_not_the_working_directory() -> TestResult {
    let directory = tempfile::tempdir()?;
    let png = directory.path().join("texels.png");
    ::image::RgbaImage::from_pixel(2, 2, ::image::Rgba([10, 20, 30, 255])).save(&png)?;

    let project = directory.path().join("textured");
    let init = pot().arg("init").arg(&project).arg("--json").output()?;
    assert_success(&init, "pot init");
    let operations = directory.path().join("operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op": "image.load", "id": "tex", "path": png, "colorspace": "srgb"},
                {"op": "material.create", "id": "wrapped", "base_color": [1.0, 1.0, 1.0, 1.0],
                 "base_color_texture": {"image": "tex", "extension": "extend"}},
                {"op": "node.create", "id": "box", "kind": "box", "params": {"size": 1.0}, "material": "wrapped"},
                {"op": "uv.unwrap", "target": {"id": "box"}, "method": "box"},
            ],
        }))?,
    )?;
    let apply = pot()
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&operations)
        .arg("--json")
        .output()?;
    assert_success(&apply, "pot apply");

    // The working directory is the parent, not the project, so the export must resolve the texture itself.
    let fbx = directory.path().join("textured.fbx");
    let export = pot()
        .current_dir(directory.path())
        .arg("export")
        .arg(&project)
        .args(["--format", "fbx", "--out"])
        .arg(&fbx)
        .arg("--json")
        .output()?;
    assert_success(&export, "pot export --format fbx from outside the project");

    let imported_project = directory.path().join("imported");
    let init = pot()
        .arg("init")
        .arg(&imported_project)
        .arg("--json")
        .output()?;
    assert_success(&init, "pot init (import)");
    let import = pot()
        .arg("import")
        .arg(&imported_project)
        .arg("--file")
        .arg(&fbx)
        .args(["--format", "fbx", "--base-revision", "0", "--json"])
        .output()?;
    assert_success(&import, "pot import --format fbx");

    let document: Value = serde_json::from_slice(&fs::read(imported_project.join("scene.json"))?)?;
    let extensions: Vec<Value> = document["materials"]
        .as_object()
        .into_iter()
        .flat_map(|materials| materials.values())
        .map(|material| material["base_color_texture"]["extension"].clone())
        .filter(|extension| !extension.is_null())
        .collect();
    assert_eq!(
        extensions,
        vec![json!("extend")],
        "the imported FBX texture should keep the EXTEND extension"
    );
    Ok(())
}
