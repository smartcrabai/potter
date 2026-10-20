//! Zero-vertex, zero-face evaluated geometry is valid empty geometry. These tests drive the `pot`
//! CLI on a box whose vertices were all deleted and check that inspect, persistence, preview,
//! pick, render, modifier stacks, and every exporter accept it, and that exports re-read as legal
//! empty content without NaN or fabricated triangles.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Output,
};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const VERTEX_IDS: &[&str] = &["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"];
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const MODIFIER_TYPES: &[&str] = &[
    "subdivision",
    "mirror",
    "solidify",
    "bevel",
    "array",
    "decimate",
    "weld",
    "triangulate",
    "wireframe",
    "displace",
    "smooth",
    "edge_split",
    "simple_deform",
    "build",
    "laplacian_smooth",
    "cast",
    "warp",
    "wave",
    "surface_deform",
    "cloth",
    "ocean",
    "weighted_normal",
];
const EXPORT_FORMATS: &[(&str, &str)] = &[
    ("glb", "out.glb"),
    ("gltf", "out.gltf"),
    ("obj", "out.obj"),
    ("ply", "out.ply"),
    ("stl", "out.stl"),
    ("fbx", "out.fbx"),
    ("usda", "out.usda"),
    ("usdc", "out.usdc"),
    ("usdz", "out.usdz"),
    ("alembic", "out.abc"),
];

fn pot() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn text(path: &Path) -> TestResult<&str> {
    path.to_str().ok_or_else(|| "test path is not UTF-8".into())
}

/// Fails on any token that prints a non-finite number (`NaN`, `inf`, `infinity`).
fn assert_no_nan(text: &str, context: &str) {
    let lower = text.to_ascii_lowercase();
    for token in lower.split(|character: char| !character.is_ascii_alphanumeric()) {
        assert!(
            !matches!(token, "nan" | "inf" | "infinity"),
            "{context} contains the non-finite token `{token}`"
        );
    }
}

/// Runs `pot ... --json`, requires success, and returns the response envelope.
fn run_ok(arguments: &[&str]) -> TestResult<Value> {
    let output: Output = pot().args(arguments).arg("--json").output()?;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        output.status.success(),
        "pot {arguments:?} failed: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_no_nan(&stdout, &format!("pot {arguments:?}"));
    Ok(serde_json::from_str(&stdout)?)
}

fn init_scene(scene: &Path) -> TestResult {
    run_ok(&["init", text(scene)?])?;
    Ok(())
}

fn apply(
    scene: &Path,
    directory: &Path,
    base_revision: u64,
    operations: &Value,
) -> TestResult<Value> {
    let file = directory.join(format!("operations-{base_revision}.json"));
    fs::write(
        &file,
        json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": operations,
        })
        .to_string(),
    )?;
    run_ok(&["apply", text(scene)?, "--file", text(&file)?])
}

/// Scene whose `body` is a box with every vertex deleted: zero vertices and zero faces.
fn empty_body_scene(directory: &Path) -> TestResult<PathBuf> {
    let scene = directory.join("scene");
    init_scene(&scene)?;
    apply(
        &scene,
        directory,
        0,
        &json!([{"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1}}]),
    )?;
    apply(
        &scene,
        directory,
        1,
        &json!([{
            "op": "mesh.delete",
            "target": {"id": "body"},
            "elements": {"domain": "vertex", "ids": VERTEX_IDS},
        }]),
    )?;
    Ok(scene)
}

fn add_camera(scene: &Path, directory: &Path, base_revision: u64) -> TestResult {
    apply(
        scene,
        directory,
        base_revision,
        &json!([{
            "op": "node.create",
            "id": "cam",
            "kind": "camera",
            "params": {},
            "transform": {"translation": [0, -5, 1]},
        }]),
    )?;
    Ok(())
}

fn inspect_items(scene: &Path) -> TestResult<Value> {
    Ok(run_ok(&["inspect", text(scene)?])?["result"]["items"].clone())
}

fn find_item(items: &Value, id: &str) -> TestResult<Value> {
    items
        .as_array()
        .into_iter()
        .flatten()
        .find(|item| item["id"] == id)
        .cloned()
        .ok_or_else(|| format!("inspect has no item `{id}`").into())
}

/// Every evaluated mesh reports zero vertices and zero faces; nodes without a mesh are skipped.
fn assert_no_geometry(items: &Value) {
    for item in items.as_array().into_iter().flatten() {
        let geometry = &item["evaluated_geometry"];
        if geometry.is_null() {
            continue;
        }
        assert_eq!(geometry["vertex_count"], 0, "item {}", item["id"]);
        assert_eq!(geometry["face_count"], 0, "item {}", item["id"]);
    }
}

fn assert_png(bytes: &[u8], context: &str) {
    assert!(bytes.starts_with(PNG_SIGNATURE), "{context} is not a PNG");
}

fn check_gltf_json(root: &Value) {
    for key in ["buffers", "accessors", "bufferViews", "meshes", "materials"] {
        assert!(root.get(key).is_none(), "empty glTF must omit `{key}`");
    }
    let nodes = root["nodes"].as_array();
    assert!(
        nodes.is_some_and(|nodes| !nodes.is_empty()),
        "glTF keeps the object node"
    );
    assert!(
        nodes.is_some_and(|nodes| nodes.iter().all(|node| node.get("mesh").is_none())),
        "an empty object is a node without a mesh"
    );
}

fn check_glb(bytes: &[u8]) -> TestResult {
    assert_eq!(&bytes[0..4], b"glTF");
    let total = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    assert_eq!(total, bytes.len(), "GLB length must cover the whole file");
    let json_length = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
    assert_eq!(&bytes[16..20], b"JSON", "first GLB chunk must be JSON");
    check_gltf_json(&serde_json::from_slice(&bytes[20..20 + json_length])?);
    Ok(())
}

fn check_ply(bytes: &[u8]) -> TestResult {
    let marker = b"end_header\n";
    let position = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or("PLY header has no end_header line")?;
    let header = std::str::from_utf8(&bytes[..position])?;
    assert!(
        header.contains("element vertex 0\n"),
        "PLY must declare zero vertices"
    );
    assert!(
        header.contains("element face 0\n"),
        "PLY must declare zero faces"
    );
    assert_eq!(
        position + marker.len(),
        bytes.len(),
        "PLY must have no body data"
    );
    Ok(())
}

/// Checks the container of an empty export of `format` without requiring a particular writer.
fn check_empty_export(format: &str, bytes: &[u8]) -> TestResult {
    match format {
        "glb" => check_glb(bytes)?,
        "gltf" => check_gltf_json(&serde_json::from_slice(bytes)?),
        "obj" => {
            let text = std::str::from_utf8(bytes)?;
            assert!(
                text.lines().any(|line| line == "o body"),
                "OBJ keeps the object name"
            );
            assert!(
                !text
                    .lines()
                    .any(|line| line.starts_with("v ") || line.starts_with("f ")),
                "OBJ must have no vertex or face lines"
            );
        }
        "ply" => check_ply(bytes)?,
        "stl" => {
            assert_eq!(
                bytes.len(),
                84,
                "binary STL with zero triangles is header only"
            );
            assert_eq!(u32::from_le_bytes(bytes[80..84].try_into()?), 0);
        }
        "fbx" => {
            let text = std::str::from_utf8(bytes)?;
            assert!(
                text.contains("Vertices: *0"),
                "FBX must declare zero vertices"
            );
            assert!(
                text.contains("PolygonVertexIndex: *0"),
                "FBX must declare zero polygons"
            );
        }
        "usda" => {
            let text = std::str::from_utf8(bytes)?;
            assert!(text.contains("point3f[] points = []"));
            assert!(text.contains("int[] faceVertexCounts = []"));
        }
        "usdc" => assert!(bytes.starts_with(b"PXR-USDC"), "USDC magic is missing"),
        "usdz" => assert!(
            bytes.starts_with(b"PK\x03\x04"),
            "USDZ archive magic is missing"
        ),
        "alembic" => assert!(
            bytes.starts_with(b"Ogawa"),
            "Alembic Ogawa magic is missing"
        ),
        other => return Err(format!("no empty-export check for `{other}`").into()),
    }
    Ok(())
}

#[test]
fn empty_geometry_inspect_reports_zero_counts_and_no_bounds() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    let items = inspect_items(&scene)?;
    let body = find_item(&items, "body")?;
    let geometry = &body["evaluated_geometry"];
    assert_eq!(geometry["vertex_count"], 0);
    assert_eq!(geometry["face_count"], 0);
    assert_eq!(geometry["triangle_count"], 0);
    assert_eq!(geometry["positions"], json!([]));
    assert_eq!(geometry["faces"], json!([]));
    assert!(body["bounds"].is_null(), "empty geometry has no bounds");
    assert!(
        body["dimensions"].is_null(),
        "empty geometry has no dimensions"
    );
    Ok(())
}

#[test]
fn empty_geometry_persists_across_reload_undo_and_redo() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    let vertex_count = |items: &Value| -> TestResult<Value> {
        Ok(find_item(items, "body")?["evaluated_geometry"]["vertex_count"].clone())
    };
    // Each command reopens the project from disk, so these reads cover persistence.
    assert_eq!(vertex_count(&inspect_items(&scene)?)?, 0);
    run_ok(&["undo", text(&scene)?, "--base-revision", "2"])?;
    assert_eq!(
        vertex_count(&inspect_items(&scene)?)?,
        8,
        "undo restores the box"
    );
    run_ok(&["redo", text(&scene)?, "--base-revision", "3"])?;
    assert_eq!(
        vertex_count(&inspect_items(&scene)?)?,
        0,
        "redo empties it again"
    );
    Ok(())
}

#[test]
fn empty_geometry_modifier_apply_keeps_zero_geometry() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    apply(
        &scene,
        directory.path(),
        2,
        &json!([{
            "op": "modifier.create",
            "target": {"id": "body"},
            "id": "subdivide",
            "type": "subdivision",
            "params": {},
        }]),
    )?;
    apply(
        &scene,
        directory.path(),
        3,
        &json!([{"op": "modifier.apply", "target": {"id": "body"}, "modifier_id": "subdivide"}]),
    )?;
    let items = inspect_items(&scene)?;
    let body = find_item(&items, "body")?;
    assert_eq!(body["evaluated_geometry"]["vertex_count"], 0);
    assert_eq!(body["evaluated_geometry"]["face_count"], 0);
    assert!(
        body["modifiers"].as_array().is_some_and(Vec::is_empty),
        "applied modifier is removed from the stack"
    );
    Ok(())
}

#[test]
fn empty_geometry_preview_and_pick_succeed_in_every_mode() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    for mode in ["solid", "wire", "normal", "depth", "id", "beauty"] {
        let out = directory.path().join(format!("preview-{mode}"));
        let envelope = run_ok(&[
            "preview",
            text(&scene)?,
            "--views",
            "front,top",
            "--mode",
            mode,
            "--size",
            "64",
            "--out",
            text(&out)?,
        ])?;
        let previews = envelope["result"]["previews"]
            .as_array()
            .ok_or("preview result has no previews")?;
        assert_eq!(previews.len(), 2, "one preview per requested view ({mode})");
        for file in previews
            .iter()
            .flat_map(|preview| preview["files"].as_array().into_iter().flatten())
        {
            if file["kind"] == "image" {
                let path = Path::new(file["path"].as_str().ok_or("image path is missing")?);
                assert_png(&fs::read(path)?, &format!("{mode} preview"));
            }
        }
    }
    let manifest = directory
        .path()
        .join("preview-solid")
        .join("front.manifest.json");
    for domain in ["object", "face", "vertex"] {
        let envelope = run_ok(&[
            "pick",
            text(&scene)?,
            "--render",
            text(&manifest)?,
            "--pixel",
            "32,32",
            "--domain",
            domain,
        ])?;
        assert_eq!(
            envelope["result"]["hit"], false,
            "nothing is picked in empty geometry"
        );
    }
    Ok(())
}

#[test]
fn empty_geometry_realtime_render_writes_a_frame_without_triangles() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    add_camera(&scene, directory.path(), 2)?;
    let out = directory.path().join("render");
    let envelope = run_ok(&[
        "render",
        text(&scene)?,
        "--format",
        "png",
        "--out",
        text(&out)?,
        "--camera",
        "cam",
        "--engine",
        "realtime",
        "--frame",
        "1",
    ])?;
    let frames = envelope["result"]["frames"]
        .as_array()
        .ok_or("render result has no frames")?;
    assert_eq!(frames.len(), 1);
    let path = Path::new(frames[0]["path"].as_str().ok_or("frame path is missing")?);
    assert_png(&fs::read(path)?, "rendered frame");
    Ok(())
}

#[test]
fn empty_geometry_modifier_stack_evaluates_and_previews() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let mut operations = Vec::new();
    for (index, modifier_type) in MODIFIER_TYPES.iter().enumerate() {
        let node = format!("n{index}");
        operations.push(
            json!({"op": "node.create", "id": node.as_str(), "kind": "box", "params": {"size": 1}}),
        );
        operations.push(json!({
            "op": "mesh.delete",
            "target": {"id": node.as_str()},
            "elements": {"domain": "vertex", "ids": VERTEX_IDS},
        }));
        operations.push(json!({
            "op": "modifier.create",
            "target": {"id": node.as_str()},
            "id": format!("m{index}"),
            "type": modifier_type,
            "params": {},
        }));
    }
    apply(&scene, directory.path(), 0, &Value::Array(operations))?;
    let items = inspect_items(&scene)?;
    for (index, modifier_type) in MODIFIER_TYPES.iter().enumerate() {
        let item = find_item(&items, &format!("n{index}"))?;
        let geometry = &item["evaluated_geometry"];
        // Blender-style modifiers may create geometry from nothing; every number must be finite.
        let positions = geometry["positions"]
            .as_array()
            .ok_or("evaluated positions are missing")?;
        assert_eq!(
            geometry["vertex_count"].as_u64(),
            Some(positions.len() as u64)
        );
        for position in positions {
            let coordinates = position.as_array().ok_or("position is not an array")?;
            assert!(
                coordinates.len() == 3 && coordinates.iter().all(Value::is_f64),
                "{modifier_type} produced a non-finite position {position}",
            );
        }
    }
    let out = directory.path().join("stack-preview");
    run_ok(&[
        "preview",
        text(&scene)?,
        "--views",
        "front",
        "--size",
        "64",
        "--out",
        text(&out)?,
    ])?;
    Ok(())
}

#[test]
fn empty_geometry_exports_are_legal_and_reread() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    for &(format, file_name) in EXPORT_FORMATS {
        let out_directory = directory.path().join(format);
        fs::create_dir_all(&out_directory)?;
        let out = out_directory.join(file_name);
        let mut arguments = vec![
            "export",
            text(&scene)?,
            "--format",
            format,
            "--out",
            text(&out)?,
        ];
        // The box's default `UVMap` is a layer STL cannot store, so STL reports it as lost and needs
        // `--allow-lossy` (the STL writer in Blender drops UVs the same way).
        if format == "stl" {
            arguments.push("--allow-lossy");
        }
        run_ok(&arguments)?;
        let bytes = fs::read(&out)?;
        check_empty_export(format, &bytes)?;
        if let Ok(text) = std::str::from_utf8(&bytes) {
            assert_no_nan(text, &format!("{format} export"));
        }
        if format == "gltf" {
            let sidecars = fs::read_dir(&out_directory)?
                .filter_map(Result::ok)
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "bin")
                })
                .count();
            assert_eq!(
                sidecars, 0,
                "an empty glTF writes no zero-length .bin sidecar"
            );
        }
        run_ok(&["validate", text(&scene)?, "--format", format])?;

        let fresh = directory.path().join(format!("reread-{format}"));
        init_scene(&fresh)?;
        run_ok(&[
            "import",
            text(&fresh)?,
            "--file",
            text(&out)?,
            "--format",
            format,
            "--base-revision",
            "0",
        ])?;
        let items = inspect_items(&fresh)?;
        assert!(
            items.as_array().is_some_and(|items| !items.is_empty()),
            "{format} re-read keeps at least one object"
        );
        assert_no_geometry(&items);
    }
    Ok(())
}

#[test]
fn empty_geometry_blend_export_round_trips_through_blender() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping empty geometry blend round trip: Blender is not installed");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    let out = directory.path().join("empty.blend");
    run_ok(&[
        "export",
        text(&scene)?,
        "--format",
        "blend",
        "--out",
        text(&out)?,
        "--blender",
        text(&blender)?,
    ])?;
    let fresh = directory.path().join("reread");
    init_scene(&fresh)?;
    run_ok(&[
        "import",
        text(&fresh)?,
        "--file",
        text(&out)?,
        "--format",
        "blend",
        "--base-revision",
        "0",
        "--blender",
        text(&blender)?,
    ])?;
    let items = inspect_items(&fresh)?;
    assert!(items.as_array().is_some_and(|items| !items.is_empty()));
    assert_no_geometry(&items);
    Ok(())
}

#[test]
fn empty_geometry_bake_writes_zero_vertex_frames() -> TestResult {
    let directory = tempdir()?;
    let scene = empty_body_scene(directory.path())?;
    let out = directory.path().join("bake");
    run_ok(&[
        "bake",
        text(&scene)?,
        "--kind",
        "geometry",
        "--target",
        "body",
        "--out",
        text(&out)?,
    ])?;
    check_ply(&fs::read(out.join("frame_00000000_body.ply"))?)?;
    Ok(())
}

#[test]
fn empty_geometry_screw_skin_and_shrinkwrap_pass_empty_input_through() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    // An empty target: shrinkwrap has no tree to wrap onto, so the chain's empty input passes through.
    // Persist each empty object before adding its modifiers so the input geometry is unambiguously
    // zero-vertex, rather than a transient same-batch primitive.
    apply(
        &scene,
        directory.path(),
        0,
        &json!([{"op": "node.create", "id": "empty_target", "kind": "box", "params": {"size": 1}}]),
    )?;
    apply(
        &scene,
        directory.path(),
        1,
        &json!([{
            "op": "mesh.delete",
            "target": {"id": "empty_target"},
            "elements": {"domain": "vertex", "ids": VERTEX_IDS},
        }]),
    )?;
    apply(
        &scene,
        directory.path(),
        2,
        &json!([{"op": "node.create", "id": "chain", "kind": "box", "params": {"size": 1}}]),
    )?;
    apply(
        &scene,
        directory.path(),
        3,
        &json!([{
            "op": "mesh.delete",
            "target": {"id": "chain"},
            "elements": {"domain": "vertex", "ids": VERTEX_IDS},
        }]),
    )?;
    let input_items = inspect_items(&scene)?;
    for id in ["empty_target", "chain"] {
        let input = find_item(&input_items, id)?;
        assert_eq!(input["evaluated_geometry"]["vertex_count"], 0, "{id}");
        assert_eq!(input["evaluated_geometry"]["face_count"], 0, "{id}");
    }
    apply(
        &scene,
        directory.path(),
        4,
        &json!([
            {"op": "modifier.create", "target": {"id": "chain"}, "id": "screw", "type": "screw", "params": {}},
            {"op": "modifier.create", "target": {"id": "chain"}, "id": "skin", "type": "skin", "params": {}},
            {
                "op": "modifier.create",
                "target": {"id": "chain"},
                "id": "wrap",
                "type": "shrinkwrap",
                "params": {"target": "empty_target"},
            },
        ]),
    )?;
    let items = inspect_items(&scene)?;
    let chain = find_item(&items, "chain")?;
    assert_eq!(chain["evaluated_geometry"]["vertex_count"], 0);
    assert_eq!(chain["evaluated_geometry"]["face_count"], 0);
    assert_eq!(chain["evaluated_geometry"]["triangle_count"], 0);
    assert!(chain["bounds"].is_null(), "empty geometry has no bounds");
    assert_no_geometry(&items);
    Ok(())
}
