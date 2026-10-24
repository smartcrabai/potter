//! Blender IMAGE strip sources stay bound through import and realtime RGBA rendering.

#[path = "common/blender_discovery.rs"]
mod blender_discovery;

#[path = "common/process.rs"]
mod process;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use potter_core::{image::decode_pixels, model::ImageColorspace};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const RESOLUTION: u32 = 64;

const IMAGE_SEQUENCE_SCRIPT: &str = r#"
import bpy
import os
import sys

blend_path, first_path, second_path = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
def configure_scene(scene):
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.use_sequencer = True
    scene.sequencer_colorspace_settings.name = "Linear Rec.709"

scene = bpy.context.scene
scene.name = "ActualImageSequence"
configure_scene(scene)
strip = scene.sequence_editor_create().strips.new_image(
    "ArbitraryImageElements", first_path, 1, 1)
strip.directory = "//inputs/"
strip.elements[0].filename = os.path.basename(first_path)
strip.elements.append(os.path.basename(second_path))
strip.blend_type = "REPLACE"

single_scene = bpy.data.scenes.new("ActualSingleImage")
configure_scene(single_scene)
single_strip = single_scene.sequence_editor_create().strips.new_image(
    "SingleImageElement", first_path, 1, 1)
single_strip.directory = "//inputs/"
single_strip.elements[0].filename = os.path.basename(first_path)
single_strip.blend_type = "REPLACE"
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
"#;

fn blender_executable() -> Option<PathBuf> {
    blender_discovery::discover(|path| {
        if !path.is_file() {
            return None;
        }
        let mut command = Command::new(&path);
        command.arg("--version");
        process::run_guarded(command)
            .ok()
            .filter(|output| output.status.success())
            .map(|_| path)
    })
}

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked(command: Command, label: &str) -> TestResult<Output> {
    let output = process::run_guarded(command)?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn import_blend(blender: &Path, root: &Path, blend: &Path) -> TestResult<(PathBuf, Value)> {
    let project = root.join("image_sequence_project");
    let mut command = pot();
    command.arg("init").arg(&project).arg("--json");
    checked(command, "pot init")?;

    let mut command = pot();
    command
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .arg("--json");
    let report = serde_json::from_slice(&checked(command, "pot import")?.stdout)?;
    Ok((project, report))
}

fn scene_id_named(document: &Value, name: &str) -> TestResult<String> {
    document["scenes"]
        .as_object()
        .ok_or("imported scene registry is missing")?
        .iter()
        .find(|(_, scene)| scene["name"].as_str() == Some(name))
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("imported Blender scene `{name}` is missing").into())
}

fn decode_exr(bytes: &[u8]) -> TestResult<Vec<[f64; 4]>> {
    let (width, height, pixels) = decode_pixels(bytes, ImageColorspace::Linear)?;
    assert_eq!((width, height), (RESOLUTION, RESOLUTION));
    Ok(pixels)
}

fn render_frame(
    project: &Path,
    scene_id: &str,
    root: &Path,
    frame: u32,
) -> TestResult<Vec<[f64; 4]>> {
    let output_dir = root.join(format!("frames_{scene_id}_{frame}"));
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .arg("--frame")
        .arg(frame.to_string())
        .args([
            "--scene-id",
            scene_id,
            "--engine",
            "realtime",
            "--format",
            "exr",
            "--out",
        ])
        .arg(&output_dir)
        .arg("--json");
    checked(command, "Potter image-sequence render")?;
    decode_exr(&fs::read(output_dir.join(format!("frame_{frame:04}.exr")))?)
}

fn assert_float_sequence_frame(label: &str, pixels: &[[f64; 4]], channel: usize) {
    assert!(
        pixels
            .iter()
            .any(|pixel| pixel[3] == 0.0 && pixel[channel] > 0.1),
        "{label} lost its nonzero RGB at alpha zero"
    );
    assert!(
        pixels.iter().any(|pixel| {
            pixel[3].total_cmp(&0.5).is_eq()
                && (0.1..0.15).contains(&pixel[0])
                && (0.04..0.06).contains(&pixel[1])
                && (0.02..0.03).contains(&pixel[2])
        }),
        "{label} lost its fractional-alpha base pixel"
    );
}

fn assert_project_owned_asset(project: &Path, reference: &str) {
    let path = Path::new(reference);
    assert!(
        path.starts_with("assets/sha256"),
        "asset reference is not project-owned: {reference}"
    );
    assert!(
        project.join(path).is_file(),
        "project asset is missing: {reference}"
    );
}

#[test]
fn native_image_sequence_elements_render_in_order_with_float_rgba_intact() -> TestResult {
    const EMITTER_SAMPLE: f32 = f32::from_bits(0x3e01_a000);
    const BASE_SAMPLE: (f32, f32, f32, f32) = (
        0.125,
        f32::from_bits(0x3d4c_c000),
        f32::from_bits(0x3ccc_c000),
        0.5,
    );
    let Some(blender) = blender_executable() else {
        eprintln!("skipping native image-sequence consumer; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let input_dir = root.join("inputs");
    fs::create_dir(&input_dir)?;
    let first_path = input_dir.join("source-red.exr");
    let second_path = input_dir.join("source-green.exr");
    let image_size = usize::try_from(RESOLUTION)?;
    exr::prelude::write_rgba_file(&first_path, image_size, image_size, |x, y| {
        if (x, y) == (32, 32) {
            (EMITTER_SAMPLE, 0.0, 0.0, 0.0)
        } else if (x, y) == (52, 51) {
            BASE_SAMPLE
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    })?;
    exr::prelude::write_rgba_file(&second_path, image_size, image_size, |x, y| {
        if (x, y) == (32, 32) {
            (0.0, EMITTER_SAMPLE, 0.0, 0.0)
        } else if (x, y) == (52, 51) {
            BASE_SAMPLE
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    })?;

    let blend = root.join("image_sequence.blend");
    let script = root.join("image_sequence.py");
    fs::write(&script, IMAGE_SEQUENCE_SCRIPT)?;
    let mut command = Command::new(&blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
        ])
        .arg("--python")
        .arg(&script)
        .arg("--")
        .arg(&blend)
        .arg(&first_path)
        .arg(&second_path);
    checked(command, "Blender image-sequence fixture generation")?;

    let (project, imported) = import_blend(&blender, root, &blend)?;
    assert_eq!(imported["result"]["committed"], true);
    fs::remove_file(&first_path)?;
    fs::remove_file(&second_path)?;
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let single_scene_id = scene_id_named(&scene, "ActualSingleImage")?;
    let single_strip = &scene["scenes"][&single_scene_id]["sequencer"]["strips"][0];
    assert_eq!(single_strip["type"], "image");
    let single_source = single_strip["source"]
        .as_str()
        .ok_or("single IMAGE source is missing")?;
    assert_eq!(
        Path::new(single_source)
            .file_name()
            .and_then(|name| name.to_str()),
        Some("source-red.exr")
    );
    assert_project_owned_asset(&project, single_source);
    assert_float_sequence_frame(
        "single IMAGE",
        &render_frame(&project, &single_scene_id, root, 1)?,
        0,
    );

    let sequence_id = scene_id_named(&scene, "ActualImageSequence")?;
    let sequence_strip = &scene["scenes"][&sequence_id]["sequencer"]["strips"][0];
    assert_eq!(sequence_strip["source"], Value::Null);
    assert_eq!(
        sequence_strip["source_files"].as_array().map(Vec::len),
        Some(2)
    );
    for source_file in sequence_strip["source_files"]
        .as_array()
        .ok_or("image sequence source files are missing")?
    {
        assert_project_owned_asset(
            &project,
            source_file
                .as_str()
                .ok_or("image sequence source file is not a string")?,
        );
    }
    assert_float_sequence_frame(
        "first image element",
        &render_frame(&project, &sequence_id, root, 1)?,
        0,
    );
    assert_float_sequence_frame(
        "second image element",
        &render_frame(&project, &sequence_id, root, 2)?,
        1,
    );
    Ok(())
}
