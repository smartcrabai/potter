//! Graph transparency against Blender's Cycles and EEVEE renders, and the OPAQUE glTF case.
//!
//! The DITHERED case is `Mix Shader(0.5, Transparent, Principled)` in front of an emitter plane. Cycles ignores the
//! render method and honours the graph, so the emitter shows through half the rays; EEVEE dithers the same graph.
//! Both renders reduce to the mean RGB of a central patch, and Potter's path tracer and realtime Beauty must match.
//!
//! The viewport case is a DITHERED Diffuse graph whose `diffuse_color` alpha is 0.5: Cycles renders it opaque, so the
//! closures must not inherit the viewport alpha. The tinted case is a Transparent BSDF with a colour over a lit floor:
//! Cycles weights the camera pass and the shadow ray by that colour. The OPAQUE glTF case gives a textured material a
//! base colour alpha of 0.5 with OPAQUE: its graph must stay opaque.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{
    fs,
    path::{Path, PathBuf},
};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

const SCRIPT: &str = include_str!("fixtures/native_oracle/graph_transparent.py");
const CASES: &str = include_str!("fixtures/native_oracle/graph_alpha_cases.py");
/// Patch grid of the Blender oracles: fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// The patch mean the DITHERED script's constants give: half the emitter (0.1, 0.6, 0.2) and half the diffuse albedo
/// (0.8, 0.3, 0.1) lit by the uniform world (0.5).
const EXPECTED: [f64; 3] = [0.25, 0.375, 0.125];
/// The viewport case: a diffuse of albedo 0.8 under a unit uniform world, opaque.
const VIEWPORT_OPAQUE: f64 = 0.8;
/// Linear render tolerance. Both estimators average 81 patch pixels, so this covers the sampling noise of 64
/// samples per pixel with margin, and it is far below the 0.25 a missing graph transparency causes.
const PIXEL_TOLERANCE: f64 = 0.01;

/// Writes `source` as `script_name` in `root/label`, runs it in Blender with `arguments` after the scene, EXR and
/// JSON paths, and returns the scene it wrote with the patch mean it recorded.
fn blender_render(
    blender: &Path,
    root: &Path,
    label: &str,
    script_name: &str,
    source: &str,
    arguments: &[&str],
) -> TestResult<(PathBuf, Vec<f64>)> {
    let dir = root.join(label);
    fs::create_dir_all(&dir)?;
    let script = write_script(&dir, script_name, source)?;
    let blend = dir.join("scene.blend");
    let image = dir.join("blender.exr");
    let json_path = dir.join("blender.json");
    let mut paths: Vec<&Path> = vec![blend.as_path(), image.as_path(), json_path.as_path()];
    paths.extend(arguments.iter().map(Path::new));
    run_blender_arguments(blender, &script, None, &paths)?;
    let reference: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let mean = reference["patch_mean"]
        .as_array()
        .ok_or("the Blender patch mean is missing")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "the Blender patch mean is not a number".into())
        })
        .collect::<TestResult<Vec<f64>>>()?;
    Ok((blend, mean))
}

/// Imports the Blender scene, sets its render size and samples, and returns the project and its active scene.
fn import_scene(blender: &Path, root: &Path, blend: &Path) -> TestResult<(PathBuf, String)> {
    let (project, _) = import_blend(blender, root, "project", blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": 64, "resolution_y": 64, "samples": 64, "seed": 0, "max_bounces": 4}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok((project, active))
}

/// Renders `project` with the `pot render` arguments `render_args` to a linear EXR, and loads the frame.
fn potter_frame(
    project: &Path,
    root: &Path,
    name: &str,
    render_args: &[&str],
) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(render_args)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": name, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][name].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

/// The patch mean over RGB of a rendered frame.
fn patch_mean(data: &ImageData) -> Vec<f64> {
    let mut total = [0.0_f64; 3];
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(data, [u, v], false);
            for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                *sum += value;
            }
        }
    }
    let count = (PATCH.len() * PATCH.len()) as f64;
    total.iter().map(|sum| sum / count).collect()
}

fn assert_patch_matches(actual: &[f64], wanted: &[f64], renderer: &str) {
    for (channel, (actual, wanted)) in actual.iter().zip(wanted).enumerate() {
        assert!(
            (actual - wanted).abs() <= PIXEL_TOLERANCE,
            "channel {channel}: Potter {actual} differs from {renderer} {wanted}"
        );
    }
}

#[test]
fn dithered_graph_transparency_matches_cycles_in_the_path_tracer() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the graph transparency oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(
        &blender,
        root.path(),
        "CYCLES",
        "graph_transparent.py",
        SCRIPT,
        &["CYCLES"],
    )?;
    assert_patch_matches(&wanted, &EXPECTED, "the analytic Cycles mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let frame = potter_frame(&project, root.path(), "path", &[])?;
    assert_patch_matches(&patch_mean(&frame), &wanted, "Cycles");
    Ok(())
}

#[test]
fn dithered_graph_transparency_matches_eevee_in_realtime() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the graph transparency realtime oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(
        &blender,
        root.path(),
        "BLENDER_EEVEE",
        "graph_transparent.py",
        SCRIPT,
        &["BLENDER_EEVEE"],
    )?;
    assert_patch_matches(&wanted, &EXPECTED, "the analytic EEVEE mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let frame = potter_frame(&project, root.path(), "realtime", &["--engine", "realtime"])?;
    assert_patch_matches(&patch_mean(&frame), &wanted, "EEVEE");
    Ok(())
}

#[test]
fn a_graph_without_transparency_ignores_the_viewport_alpha_as_cycles_does() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the viewport alpha oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(
        &blender,
        root.path(),
        "viewport",
        "graph_alpha_cases.py",
        CASES,
        &["CYCLES", "viewport_alpha"],
    )?;
    assert_patch_matches(&wanted, &[VIEWPORT_OPAQUE; 3], "the analytic Cycles mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let frame = potter_frame(&project, root.path(), "path", &[])?;
    assert_patch_matches(&patch_mean(&frame), &wanted, "Cycles");
    Ok(())
}

#[test]
fn a_tinted_transparent_bsdf_tints_camera_and_shadow_rays_as_cycles_does() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the tinted transparent oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, wanted) = blender_render(
        &blender,
        root.path(),
        "tinted",
        "graph_alpha_cases.py",
        CASES,
        &["CYCLES", "tinted_transparent"],
    )?;
    // The floor's albedo over pi, lit by a unit sun, crosses the tinted plane twice: once from the camera, once in
    // the shadow ray. The tint is (0.9, 0.4, 0.2).
    let lit = 0.8 / std::f64::consts::PI;
    let expected = [0.9 * 0.9 * lit, 0.4 * 0.4 * lit, 0.2 * 0.2 * lit];
    assert_patch_matches(&wanted, &expected, "the analytic Cycles mean");

    let (project, _) = import_scene(&blender, root.path(), &blend)?;
    let frame = potter_frame(&project, root.path(), "path", &[])?;
    assert_patch_matches(&patch_mean(&frame), &wanted, "Cycles");
    Ok(())
}

/// A 1x1 RGB PNG (200, 100, 50).
const PNG: [u8; 69] = [
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0,
    0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 56, 145, 98, 4, 0, 3, 86,
    1, 95, 232, 23, 132, 82, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// Positions (3 vertices) then `TEXCOORD_0` (3 vertices), little-endian f32: a unit triangle in glTF's XY plane.
const BUFFER: [u8; 60] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
    63,
];

/// A glTF triangle with `material`, written into `directory`.
fn write_gltf(directory: &Path, material: &Value) -> TestResult<PathBuf> {
    fs::write(directory.join("image.png"), PNG)?;
    fs::write(directory.join("buffer.bin"), BUFFER)?;
    let document = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [{
            "attributes": { "POSITION": 0, "TEXCOORD_0": 1 },
            "material": 0
        }] }],
        "materials": [material],
        "textures": [{ "source": 0 }],
        "images": [{ "uri": "image.png", "mimeType": "image/png" }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 24 }
        ],
        "buffers": [{ "uri": "buffer.bin", "byteLength": 60 }],
        "extensionsUsed": ["KHR_texture_transform"]
    });
    let path = directory.join("textured.gltf");
    fs::write(&path, serde_json::to_vec(&document)?)?;
    Ok(path)
}

#[test]
fn an_opaque_textured_gltf_material_keeps_its_base_alpha_out_of_its_graph() -> TestResult {
    // A transformed texture makes the importer build a graph (`exchange/material_graph.rs`), which the base colour
    // alpha of an OPAQUE material must not reach. No Blender is needed.
    let directory = TempDir::new()?;
    let material = json!({
        "name": "Opaque Textured",
        "alphaMode": "OPAQUE",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
            "baseColorTexture": {
                "index": 0,
                "extensions": { "KHR_texture_transform": { "offset": [0.1, 0.0] } }
            }
        }
    });
    let gltf = write_gltf(directory.path(), &material)?;
    let project = directory.path().join("scene");
    let mut init = pot();
    init.arg("init").arg(&project).arg("--json");
    checked(&mut init, "pot init")?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(&gltf)
        .args(["--format", "gltf", "--base-revision", "0", "--json"]);
    let output = checked(&mut import, "pot import")?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["ok"], true, "import failed: {envelope}");
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let imported = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    assert!(
        imported["node_tree"].is_string(),
        "the transformed texture must import as a graph: {imported}"
    );

    // glTF's XY plane is Potter's XZ plane (Z-up), so a camera on -Y looking along +Y sees the triangle face on.
    let quarter = std::f64::consts::FRAC_1_SQRT_2;
    let applied = apply_operations(
        &project,
        directory.path(),
        json!([
            {"op": "camera.create", "id": "front", "projection": "orthographic", "ortho_scale": 2.2,
             "transform": {"translation": [0.0, -5.0, 0.0], "rotation": [quarter, 0.0, 0.0, quarter]}},
            {"op": "scene.update", "target": {"id": active}, "set": {"camera": "front"}},
            {"op": "render.update", "target": {"id": active},
             "set": {"resolution_x": 64, "resolution_y": 64, "resolution_percentage": 100,
                     "samples": 16, "seed": 0, "max_bounces": 2, "film_transparent": true}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    // Film alpha is 1 where the triangle is hit and 0 elsewhere, so every covered pixel must be opaque.
    let frame = potter_frame(&project, directory.path(), "opaque", &[])?;
    let mut covered = Vec::new();
    for row in 0..64 {
        for column in 0..64 {
            let u = (f64::from(column) + 0.5) / 64.0;
            let v = (f64::from(row) + 0.5) / 64.0;
            let alpha = sample(&frame, [u, v], false)[3];
            if alpha > 0.01 {
                covered.push(alpha);
            }
        }
    }
    assert!(
        !covered.is_empty(),
        "the triangle must cover part of the frame"
    );
    assert!(
        covered.iter().all(|alpha| *alpha > 0.99),
        "an OPAQUE material must render opaque; covered alphas are {covered:?}"
    );
    Ok(())
}
