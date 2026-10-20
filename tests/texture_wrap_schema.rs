//! Saved scenes with non-REPEAT texture fields conform to the published scene schema, and the schema rejects
//! invalid texture references.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn cli_json(output: &Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Creates a project through the CLI whose one material has EXTEND and MIRROR texture fields, and returns it saved.
fn project_with_wrapped_textures() -> TestResult<(TempDir, PathBuf)> {
    let directory = tempfile::tempdir()?;
    let project = directory.path().join("wrapped");
    let init = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(
        init.status.success(),
        "pot init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    let file = project.join("texture-operations.json");
    fs::write(
        &file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {
                    "op": "image.create",
                    "id": "tex",
                    "name": "Tex",
                    "width": 1,
                    "height": 1,
                    "fill_color": [1.0, 1.0, 1.0, 1.0],
                },
                {
                    "op": "material.create",
                    "id": "wrapped",
                    "base_color": [1.0, 1.0, 1.0, 1.0],
                    "base_color_texture": {"image": "tex", "extension": "extend"},
                    "roughness_texture": {"image": "tex", "extension": "mirror"},
                },
            ],
        }))?,
    )?;
    let apply = pot()
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&file)
        .arg("--json")
        .output()?;
    assert!(
        apply.status.success(),
        "pot apply failed: stdout={} stderr={}",
        String::from_utf8_lossy(&apply.stdout),
        String::from_utf8_lossy(&apply.stderr)
    );
    Ok((directory, project))
}

fn saved_scene(project: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

fn published_scene_validator() -> TestResult<jsonschema::Validator> {
    let output = pot()
        .args(["schema", "--kind", "scene", "--json"])
        .output()?;
    assert!(output.status.success());
    let schema = cli_json(&output)?["result"]["schema"].clone();
    Ok(jsonschema::validator_for(&schema)?)
}

#[test]
fn saved_scene_with_non_repeat_texture_fields_conforms_to_the_published_scene_schema() -> TestResult
{
    let (_directory, project) = project_with_wrapped_textures()?;
    let document = saved_scene(&project)?;
    let material = &document["materials"]["wrapped"];
    assert_eq!(material["base_color_texture"]["extension"], "extend");
    assert_eq!(material["roughness_texture"]["extension"], "mirror");
    assert!(
        published_scene_validator()?.is_valid(&document),
        "the published scene schema rejected a saved scene with EXTEND and MIRROR texture fields"
    );
    Ok(())
}

#[test]
fn published_scene_schema_rejects_an_unknown_extension_and_a_missing_interpolation() -> TestResult {
    let (_directory, project) = project_with_wrapped_textures()?;
    let document = saved_scene(&project)?;
    let validator = published_scene_validator()?;

    let mut unknown_extension = document.clone();
    unknown_extension["materials"]["wrapped"]["base_color_texture"]["extension"] = json!("tile");
    assert!(
        !validator.is_valid(&unknown_extension),
        "the scene schema accepted an unknown texture extension"
    );

    let mut missing_interpolation = document;
    missing_interpolation["materials"]["wrapped"]["base_color_texture"]
        .as_object_mut()
        .ok_or("the base colour texture is not an object")?
        .remove("interpolation");
    assert!(
        !validator.is_valid(&missing_interpolation),
        "the scene schema accepted a texture reference without interpolation"
    );
    Ok(())
}
