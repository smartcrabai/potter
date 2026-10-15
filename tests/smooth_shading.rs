use std::{collections::BTreeSet, error::Error, fs, io::Cursor, path::Path, process::Output};

use assert_cmd::Command;
use exr::prelude::{FlatSamples, read_all_flat_layers_from_file};
use serde_json::{Value, json};
use tempfile::tempdir;

const SEGMENTS: u32 = 24;
const RINGS: u32 = 12;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).into_owned()).into(),
        );
    }
    Ok(output)
}

fn apply(scene: &Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    run_ok(pot().arg("apply").arg(scene).arg("--file").arg(&batch_file))?;
    Ok(())
}

/// Builds a UV sphere scene with an orthographic camera looking along +Y at the origin.
/// `smooth` adds `shade_smooth` to every face; `scale` sets the object scale.
fn sphere_scene(scene: &Path, smooth: bool, scale: [f64; 3]) -> Result<(), Box<dyn Error>> {
    run_ok(pot().arg("init").arg(scene).arg("--json"))?;
    let half_root_two = std::f64::consts::FRAC_1_SQRT_2;
    let mut operations = vec![
        json!({
            "op": "node.create",
            "id": "ball",
            "kind": "sphere",
            "params": {"radius": 1.0, "segments": SEGMENTS, "ring_count": RINGS},
            "transform": {"scale": scale}
        }),
        json!({
            "op": "camera.create",
            "id": "camera_main",
            "transform": {"translation": [0.0, -6.0, 0.0], "rotation": [half_root_two, 0.0, 0.0, half_root_two]},
            "projection": "orthographic",
            "ortho_scale": 2.4
        }),
        json!({"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "camera_main"}}),
        json!({
            "op": "render.update",
            "target": {"id": "scene_main"},
            "set": {"resolution_x": 64, "resolution_y": 64, "resolution_percentage": 100, "samples": 1, "seed": 0}
        }),
        json!({"op": "render.passes_update", "target": {"id": "scene_main"}, "set": {"passes": ["normal"]}}),
    ];
    if smooth {
        let face_ids: Vec<String> = (0..SEGMENTS * RINGS).map(|id| format!("f{id}")).collect();
        operations.push(json!({
            "op": "mesh.shade_smooth",
            "target": {"id": "ball"},
            "elements": {"domain": "face", "ids": face_ids}
        }));
    }
    apply(scene, &Value::Array(operations))
}

/// Renders the isometric preview in `mode` and returns the number of distinct colours.
fn preview_colour_count(scene: &Path, out: &Path, mode: &str) -> Result<usize, Box<dyn Error>> {
    run_ok(
        pot()
            .arg("preview")
            .arg(scene)
            .args(["--views", "iso", "--mode", mode, "--size", "192", "--out"])
            .arg(out)
            .args(["--overwrite", "--json"]),
    )?;
    let bytes = fs::read(out.join("iso.png"))?;
    let mut reader = png::Decoder::new(Cursor::new(bytes.as_slice())).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let channels = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        _ => return Err("preview must be an RGB or RGBA PNG".into()),
    };
    let colours: BTreeSet<[u8; 3]> = pixels
        .chunks_exact(channels)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    Ok(colours.len())
}

/// Counts the distinct unit normals (to 1e-3) in the `Normal` layer of a multilayer EXR.
fn distinct_normal_count(path: &Path) -> Result<usize, Box<dyn Error>> {
    let image = read_all_flat_layers_from_file(path)?;
    let layer = image
        .layer_data
        .iter()
        .find(|layer| {
            layer
                .attributes
                .layer_name
                .as_ref()
                .is_some_and(|name| name.eq("Normal"))
        })
        .ok_or("EXR has no Normal layer")?;
    let channel = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        let samples = layer
            .channel_data
            .list
            .iter()
            .find(|channel| channel.name.eq(name))
            .map(|channel| &channel.sample_data)
            .ok_or_else(|| format!("Normal layer has no {name} channel"))?;
        match samples {
            FlatSamples::F32(values) => Ok(values.clone()),
            FlatSamples::F16(values) => Ok(values.iter().map(|value| value.to_f32()).collect()),
            FlatSamples::U32(_) => Err("Normal channels must be floating point".into()),
        }
    };
    let (x, y, z) = (channel("X")?, channel("Y")?, channel("Z")?);
    let normals: BTreeSet<[i32; 3]> = x
        .iter()
        .zip(&y)
        .zip(&z)
        .map(|((x, y), z)| {
            [x, y, z].map(|component| (f64::from(*component) * 1000.0).round() as i32)
        })
        .collect();
    Ok(normals.len())
}

#[test]
fn smooth_sphere_normals_vary_within_faces_while_flat_sphere_stays_faceted()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let flat_scene = directory.path().join("flat");
    let smooth_scene = directory.path().join("smooth");
    sphere_scene(&flat_scene, false, [1.0; 3])?;
    sphere_scene(&smooth_scene, true, [1.0; 3])?;

    let flat = preview_colour_count(&flat_scene, &directory.path().join("flat-normal"), "normal")?;
    let smooth = preview_colour_count(
        &smooth_scene,
        &directory.path().join("smooth-normal"),
        "normal",
    )?;
    assert!(
        smooth > 2 * flat,
        "smooth normals should take many more colours than {flat} flat facets; got {smooth}"
    );
    Ok(())
}

#[test]
fn solid_preview_lights_smooth_faces_per_pixel() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let flat_scene = directory.path().join("flat");
    let smooth_scene = directory.path().join("smooth");
    sphere_scene(&flat_scene, false, [1.0; 3])?;
    sphere_scene(&smooth_scene, true, [1.0; 3])?;

    let flat = preview_colour_count(&flat_scene, &directory.path().join("flat-solid"), "solid")?;
    let smooth = preview_colour_count(
        &smooth_scene,
        &directory.path().join("smooth-solid"),
        "solid",
    )?;
    // An unlit smooth sphere would be one flat colour, far fewer than the flat facets.
    assert!(
        smooth > 2 * flat,
        "smooth Solid shading should vary per pixel; flat had {flat} colours, smooth {smooth}"
    );
    Ok(())
}

#[test]
fn path_traced_normal_pass_interpolates_smooth_normals() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let flat_scene = directory.path().join("flat");
    let smooth_scene = directory.path().join("smooth");
    sphere_scene(&flat_scene, false, [1.0; 3])?;
    sphere_scene(&smooth_scene, true, [1.0; 3])?;

    let mut counts = Vec::new();
    for (name, scene) in [("flat", &flat_scene), ("smooth", &smooth_scene)] {
        let output = directory.path().join(format!("{name}-exr"));
        run_ok(
            pot()
                .arg("render")
                .arg(scene)
                .args(["--engine", "path", "--format", "exr", "--out"])
                .arg(&output)
                .arg("--json"),
        )?;
        counts.push(distinct_normal_count(&output.join("frame_0001.exr"))?);
    }
    let (flat, smooth) = (counts[0], counts[1]);
    assert!(
        smooth > 2 * flat,
        "path-traced Normal pass should interpolate smooth normals; flat {flat}, smooth {smooth}"
    );
    Ok(())
}

#[test]
fn render_reports_failed_smooth_normals_as_a_warning() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("singular");
    // A zero scale axis makes the object transform singular, so smooth normals cannot be mapped.
    sphere_scene(&scene, true, [1.0, 1.0, 0.0])?;
    let output = run_ok(
        pot()
            .arg("render")
            .arg(&scene)
            .args(["--engine", "path", "--format", "png", "--out"])
            .arg(directory.path().join("out"))
            .arg("--json"),
    )?;
    let response: Value = serde_json::from_slice(&output.stdout)?;
    let warnings = response["warnings"]
        .as_array()
        .ok_or("render response has no warnings array")?;
    assert!(
        warnings
            .iter()
            .any(|warning| warning["code"] == "shading.corner_normals_failed"),
        "expected shading.corner_normals_failed in {warnings:?}"
    );
    Ok(())
}

#[test]
fn flat_sphere_preview_has_no_warnings() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("flat");
    sphere_scene(&scene, false, [1.0; 3])?;
    let output = run_ok(
        pot()
            .arg("preview")
            .arg(&scene)
            .args(["--views", "iso", "--size", "64", "--out"])
            .arg(directory.path().join("out"))
            .args(["--overwrite", "--json"]),
    )?;
    let response: Value = serde_json::from_slice(&output.stdout)?;
    let warnings = response["result"]["previews"][0]["warnings"]
        .as_array()
        .ok_or("preview result has no warnings array")?;
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    Ok(())
}
