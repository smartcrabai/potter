//! Export modifier purpose (PR18L).
//!
//! Each export evaluates the modifier stack that Blender 5.2.2's default exporter for the same
//! format evaluates. Probes ran in Blender 5.2.2 on a cube carrying a viewport-only and a
//! render-only Array (count 3):
//!
//! - OBJ (`export_eval_mode` = `DAG_EVAL_VIEWPORT`): viewport stack.
//! - glTF / GLB: viewport stack when modifiers are applied (`export_apply` defaults to false, which
//!   exports the base cage). FBX: viewport stack; `use_mesh_modifiers_render` is not read in 5.2.
//! - PLY (no evaluation option; render measured), STL (`evaluation_mode` = `DAG_EVAL_RENDER`), USD
//!   and Alembic (`evaluation_mode` = `RENDER`): render stack.
//! - `object.bake` (AO) and `render.render` evaluate the render stack, so bake and render stay Render.
//!
//! The table below assigns each `pot export` format name its purpose. SVG and PDF have no Blender
//! writer and keep the render stack. BLEND and BVH evaluate no modifiers; their requested purpose is
//! still the viewport stack, and BVH exports armature data only. The VRM rows use the viewport stack
//! like the glTF writer they share. Potter's own `usda`/`usdc`/`usdz` writers are mapped to the
//! render stack with the USD default; that is a policy choice, not a Blender measurement.
//!
//! Every row asserts the purpose the export requested. Mesh rows also assert the evaluated vertex
//! counts. VRM rows use an avatar whose body carries a render-only subdivision that fails above five
//! levels: the viewport stack must export it, and validation must agree with export.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;
#[path = "common/vrm_avatar.rs"]
mod vrm_avatar;

use blender_checked::blender_executable;
use vrm_avatar::avatar_operations;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const PREVIEW: &str = "preview";
const RENDER: &str = "render";

/// The scene an export row is built from.
#[derive(Clone, Copy)]
enum Scene {
    /// One box with a viewport-only, render-only, applied, or absent Array.
    Cube,
    /// One rig with no mesh, for BVH.
    Armature,
    /// The VRM avatar, for VRM 1.0 and VRM 0.x.
    Avatar,
}

/// CLI format name, output extension, purpose the export requests, and the scene it exports.
const FORMATS: [(&str, &str, &str, Scene); 18] = [
    ("blend", "blend", PREVIEW, Scene::Cube),
    ("vrm", "vrm", PREVIEW, Scene::Avatar),
    ("vrm0", "vrm", PREVIEW, Scene::Avatar),
    ("glb", "glb", PREVIEW, Scene::Cube),
    ("gltf", "gltf", PREVIEW, Scene::Cube),
    ("usda", "usda", RENDER, Scene::Cube),
    ("usdc", "usdc", RENDER, Scene::Cube),
    ("usd", "usd", RENDER, Scene::Cube),
    ("usdz", "usdz", RENDER, Scene::Cube),
    ("alembic", "abc", RENDER, Scene::Cube),
    ("fbx", "fbx", PREVIEW, Scene::Cube),
    ("fbx-binary", "fbx", PREVIEW, Scene::Cube),
    ("obj", "obj", PREVIEW, Scene::Cube),
    ("ply", "ply", RENDER, Scene::Cube),
    ("stl", "stl", RENDER, Scene::Cube),
    ("bvh", "bvh", PREVIEW, Scene::Armature),
    ("svg", "svg", RENDER, Scene::Cube),
    ("pdf", "pdf", RENDER, Scene::Cube),
];

/// Formats whose validation preflight must agree with export under a render-only failing modifier.
const AGREEMENT: [&str; 6] = ["obj", "glb", "stl", "usd", "vrm", "vrm0"];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn success(output: &Output) -> TestResult<Value> {
    if !output.status.success() {
        return Err(format!(
            "pot failed ({}): stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn cube(modifiers: &[Value]) -> Vec<Value> {
    let mut operations =
        vec![json!({"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 2.0}})];
    operations.extend(modifiers.iter().cloned());
    operations
}

fn array(target: &str, show_viewport: bool, show_render: bool) -> Value {
    json!({
        "op": "modifier.create",
        "target": {"id": target},
        "id": "array",
        "type": "array",
        "params": {"count": 3},
        "show_viewport": show_viewport,
        "show_render": show_render,
    })
}

/// A subdivision visible only for render, with a render level above the evaluator's limit of five.
/// Evaluating the render stack fails; the viewport stack never sees it.
fn render_limit(target: &str) -> Value {
    json!({
        "op": "modifier.create",
        "target": {"id": target},
        "id": "render_subdivision",
        "type": "subdivision",
        "params": {"levels": 1, "render_levels": 6},
        "show_viewport": false,
        "show_render": true,
    })
}

fn new_project(root: &Path, name: &str, operations: &[Value]) -> TestResult<PathBuf> {
    let project = root.join(name);
    success(&pot().arg("init").arg(&project).arg("--json").output()?)?;
    let batch = root.join(format!("{name}-operations.json"));
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    success(
        &pot()
            .arg("apply")
            .arg(&project)
            .arg("--file")
            .arg(&batch)
            .arg("--json")
            .output()?,
    )?;
    Ok(project)
}

/// Every scene the matrix exports, built once.
struct Projects {
    applied: PathBuf,
    viewport_only: PathBuf,
    render_only: PathBuf,
    base: PathBuf,
    render_limit: PathBuf,
    armature: PathBuf,
    avatar: PathBuf,
}

impl Projects {
    fn build(root: &Path) -> TestResult<Self> {
        let mut avatar = avatar_operations();
        avatar.push(render_limit("body"));
        Ok(Self {
            applied: new_project(root, "applied", &cube(&[array("cube", true, true)]))?,
            viewport_only: new_project(
                root,
                "viewport-only",
                &cube(&[array("cube", true, false)]),
            )?,
            render_only: new_project(root, "render-only", &cube(&[array("cube", false, true)]))?,
            base: new_project(root, "base", &cube(&[]))?,
            render_limit: new_project(root, "render-limit", &cube(&[render_limit("cube")]))?,
            armature: new_project(
                root,
                "armature",
                &[
                    json!({"op": "node.create", "id": "arm", "kind": "armature"}),
                    json!({"op": "rig.generate_basic_human", "target": {"id": "arm"}, "scale": 1.0}),
                ],
            )?,
            avatar: new_project(root, "avatar", &avatar)?,
        })
    }

    fn for_scene(&self, scene: Scene) -> &Path {
        match scene {
            Scene::Cube => &self.applied,
            Scene::Armature => &self.armature,
            Scene::Avatar => &self.avatar,
        }
    }
}

/// Runs `pot export` and returns the raw process output, so failing exports can be inspected too.
fn run_export(
    project: &Path,
    format: &str,
    extension: &str,
    blender: Option<&Path>,
    out_dir: &Path,
    name: &str,
) -> TestResult<Output> {
    let out = out_dir.join(format!("{name}-{format}.{extension}"));
    let mut command = pot();
    command
        .arg("export")
        .arg(project)
        .args(["--format", format, "--out"])
        .arg(&out)
        .arg("--json");
    if let Some(blender) = blender {
        command.arg("--blender").arg(blender);
    }
    Ok(command.output()?)
}

/// `result` of a successful export.
fn export_result(
    project: &Path,
    format: &str,
    extension: &str,
    blender: Option<&Path>,
    out_dir: &Path,
    name: &str,
) -> TestResult<Value> {
    let output = run_export(project, format, extension, blender, out_dir, name)?;
    let envelope = success(&output)?;
    Ok(envelope["result"].clone())
}

fn requested_purpose(result: &Value) -> TestResult<&str> {
    result["context"]["requested"]["purpose"]
        .as_str()
        .ok_or_else(|| "export reported no requested purpose".into())
}

fn evaluated_vertices(result: &Value) -> TestResult<u64> {
    result["counts"]["vertices"]
        .as_u64()
        .ok_or_else(|| "export reported no vertex count".into())
}

fn validate_valid(project: &Path, format: &str) -> TestResult<bool> {
    let output = pot()
        .arg("validate")
        .arg(project)
        .args(["--format", format, "--json"])
        .output()?;
    success(&output)?["result"]["valid"]
        .as_bool()
        .ok_or_else(|| format!("validate {format} reported no valid flag").into())
}

#[test]
fn every_export_format_requests_its_blender_default_purpose() -> TestResult {
    let directory = tempdir()?;
    let blender = blender_executable();
    let projects = Projects::build(directory.path())?;
    for (format, extension, expected, scene) in FORMATS {
        if format == "blend" && blender.is_none() {
            eprintln!("skipping blend purpose row: Blender is unavailable");
            continue;
        }
        let result = export_result(
            projects.for_scene(scene),
            format,
            extension,
            blender.as_deref(),
            directory.path(),
            "purpose",
        )?;
        assert_eq!(
            requested_purpose(&result)?,
            expected,
            "{format} must request the {expected} stack"
        );
    }
    Ok(())
}

#[test]
fn mesh_exports_evaluate_the_stack_their_purpose_names() -> TestResult {
    let directory = tempdir()?;
    let projects = Projects::build(directory.path())?;
    for (format, extension, expected, scene) in FORMATS {
        // Blend stores the stack and its flags rather than an evaluated mesh, so it has no vertex row.
        if !matches!(scene, Scene::Cube) || format == "blend" {
            continue;
        }
        let vertices = |project: &Path, name: &str| -> TestResult<u64> {
            let result = export_result(project, format, extension, None, directory.path(), name)?;
            evaluated_vertices(&result)
        };
        let applied = vertices(&projects.applied, "applied")?;
        let base = vertices(&projects.base, "base")?;
        let viewport_only = vertices(&projects.viewport_only, "viewport-only")?;
        let render_only = vertices(&projects.render_only, "render-only")?;
        assert_ne!(
            applied, base,
            "{format}: the Array must change the exported mesh"
        );
        let (viewport_expected, render_expected) = if expected == PREVIEW {
            (applied, base)
        } else {
            (base, applied)
        };
        assert_eq!(
            viewport_only, viewport_expected,
            "{format}: a viewport-only Array must follow the {expected} stack"
        );
        assert_eq!(
            render_only, render_expected,
            "{format}: a render-only Array must follow the {expected} stack"
        );
    }
    Ok(())
}

#[test]
fn validate_preflight_matches_export_for_the_same_scene() -> TestResult {
    let directory = tempdir()?;
    let projects = Projects::build(directory.path())?;
    for (format, extension, expected, scene) in FORMATS {
        if !AGREEMENT.contains(&format) {
            continue;
        }
        let project = match scene {
            Scene::Cube => &projects.render_limit,
            Scene::Avatar => &projects.avatar,
            Scene::Armature => return Err("no agreement project for armature formats".into()),
        };
        let exported = run_export(project, format, extension, None, directory.path(), "agree")?;
        let exported_ok = exported.status.success();
        let valid = validate_valid(project, format)?;
        assert_eq!(
            exported_ok, valid,
            "{format}: validate and export disagree on the same scene"
        );
        assert_eq!(
            exported_ok,
            expected == PREVIEW,
            "{format} must evaluate its {expected} stack"
        );
    }
    Ok(())
}

#[test]
fn saved_render_manifest_records_render_visibility() -> TestResult {
    let directory = tempdir()?;
    let operations = [
        json!({
            "op": "node.create",
            "id": "body",
            "kind": "box",
            "params": {"size": 1.0},
            "transform": {"translation": [0.0, 0.0, 0.5]},
        }),
        json!({
            "op": "camera.create",
            "id": "camera_main",
            "transform": {"translation": [0.0, 0.0, 4.0]},
            "projection": "orthographic",
            "ortho_scale": 3.0,
        }),
        json!({"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_main"}}),
        json!({
            "op": "render.update",
            "target": {"id": "scene_main"},
            "set": {
                "resolution_x": 32,
                "resolution_y": 24,
                "resolution_percentage": 100,
                "samples": 1,
                "seed": 7,
            },
        }),
    ];
    let project = new_project(directory.path(), "render", &operations)?;
    let out = directory.path().join("png");
    success(
        &pot()
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "png", "--out"])
            .arg(&out)
            .arg("--json")
            .output()?,
    )?;
    let manifest: Value = serde_json::from_slice(&fs::read(out.join("render.manifest.json"))?)?;
    assert_eq!(manifest["visibility"], RENDER);
    Ok(())
}
