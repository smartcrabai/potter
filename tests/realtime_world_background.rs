//! Realtime Beauty's background against Blender 5.2.2's EEVEE, for a world alone.
//!
//! `fixtures/native_oracle/eevee_world_background.py` builds two worlds whose background comes from a Background node:
//! a black node over a viewport colour of 0.05 (`node_black`), and a node colour of (0.2, 0.4, 0.1) at strength 0.5 over
//! a viewport colour of 0.9 (`node_colour`). EEVEE's renders are the committed macOS arm64 captures in
//! `fixtures/native_oracle/eevee_macos/` (see its `MANIFEST.json`). Potter must show the node's colour, not the viewport
//! colour, at every pixel within 1e-3.
//!
//! Every pixel is the same world value, so the row order of EEVEE's bottom-up captures does not matter here.
//!
//! Blender-gated: with `POTTER_REQUIRE_BLENDER` set, a missing Blender fails the test.

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

use std::{error::Error, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, loss_features, pot, pot_json, read_scene,
    run_blender_arguments, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::json;
use tempfile::tempdir;

/// The scene and EEVEE render (`eevee_world_background.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_world_background.py");

/// Builds a generic Mix node connected to the world background, then saves the `.blend`.
const MIX_WORLD_SCRIPT: &str = r#"
import sys
import bpy

data_type, blend_out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
tree = world.node_tree
tree.nodes.clear()
mix = tree.nodes.new("ShaderNodeMix")
mix.data_type = data_type
mix.blend_type = "MIX"
background = tree.nodes.new("ShaderNodeBackground")
output = tree.nodes.new("ShaderNodeOutputWorld")
result = next(socket for socket in mix.outputs if socket.name == "Result")
target = background.inputs["Strength"] if data_type == "FLOAT" else background.inputs["Color"]
tree.links.new(result, target)
tree.links.new(background.outputs["Background"], output.inputs["Surface"])
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
"#;

/// The committed macOS arm64 EEVEE renders, as little-endian f32 RGB rows bottom-up.
const EEVEE_NODE_BLACK: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/world_background_node_black.f32le");
const EEVEE_NODE_COLOUR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/world_background_node_colour.f32le");

const SIZE: u32 = 64;
const SAMPLES: u32 = 16;
/// Linear tolerance on each channel against EEVEE's world value.
const TOLERANCE: f64 = 1.0e-3;

/// The RGB values of an EEVEE capture, one entry per pixel.
fn eevee_rgb(bytes: &[u8]) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let expected = (SIZE as usize) * (SIZE as usize) * 12;
    if bytes.len() != expected {
        return Err(format!(
            "EEVEE capture has {} bytes, expected {expected}",
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .as_chunks::<12>()
        .0
        .iter()
        .map(|pixel| {
            let channel = |offset: usize| {
                f64::from(f32::from_le_bytes([
                    pixel[offset],
                    pixel[offset + 1],
                    pixel[offset + 2],
                    pixel[offset + 3],
                ]))
            };
            [channel(0), channel(4), channel(8)]
        })
        .collect())
}

/// The RGB values of the image `id`, sampled at each texel centre with Potter's own EXR loader.
fn potter_rgb(project: &Path, id: &str) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let mut pixels = Vec::with_capacity((data.width * data.height) as usize);
    for row in 0..data.height {
        for column in 0..data.width {
            let uv = [
                (f64::from(column) + 0.5) / f64::from(data.width),
                (f64::from(row) + 0.5) / f64::from(data.height),
            ];
            let value = sample(&data, uv, false);
            pixels.push([value[0], value[1], value[2]]);
        }
    }
    Ok(pixels)
}

#[test]
fn realtime_world_background_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_world_background.py", SCRIPT)?;
    let cases = [
        ("node_black", EEVEE_NODE_BLACK),
        ("node_colour", EEVEE_NODE_COLOUR),
    ];
    let mut failures = Vec::new();
    for (case, capture) in cases {
        let blend = root.path().join(format!("{case}.blend"));
        // The render path `-` makes the script save the .blend and stop before rendering: EEVEE's pixels are committed.
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[Path::new(case), blend.as_path(), Path::new("-")],
        )?;
        let (project, _) = import_blend(&blender, root.path(), case, &blend)?;
        let scene = read_scene(&project)?;
        let active = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let applied = apply_operations(
            &project,
            root.path(),
            json!([{"op": "render.update", "target": {"id": active}, "set": {"resolution_x": SIZE, "resolution_y": SIZE, "samples": SAMPLES, "seed": 0}}]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{case}"));
        let rendered = pot()
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "realtime render failed: {}",
            String::from_utf8_lossy(&rendered.stdout)
        );
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": "render", "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let potter = potter_rgb(&project, "render")?;
        let eevee = eevee_rgb(capture)?;
        assert_eq!(potter.len(), eevee.len(), "{case} pixel count");
        let worst = potter
            .iter()
            .zip(&eevee)
            .flat_map(|(actual, expected)| (0..3).map(move |c| (actual[c] - expected[c]).abs()))
            .fold(0.0_f64, f64::max);
        if worst > TOLERANCE {
            failures.push(format!(
                "{case}: Potter deviates from EEVEE by {worst} (tolerance {TOLERANCE}); Potter {:?}, EEVEE {:?}",
                potter[0], eevee[0]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn blender_float_world_mix_into_strength_is_reported_as_a_loss() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping generic world Mix import; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "generic_world_mix.py", MIX_WORLD_SCRIPT)?;

    for (case, data_type, should_report_loss) in [
        ("float_strength", "FLOAT", true),
        ("rgba_color", "RGBA", false),
    ] {
        let blend = root.path().join(format!("{case}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[Path::new(data_type), blend.as_path()],
        )?;

        let project = root.path().join(format!("{case}_project"));
        pot_json(|command| {
            command.arg("init").arg(&project);
        })?;
        let report = pot_json(|command| {
            command
                .arg("import")
                .arg(&project)
                .arg("--file")
                .arg(&blend)
                .args([
                    "--format",
                    "blend",
                    "--mode",
                    "replace",
                    "--base-revision",
                    "0",
                    "--blender",
                ])
                .arg(&blender)
                .arg("--allow-lossy");
        })?;
        let features = loss_features(&report);
        assert_eq!(
            features.contains(&"blender.world.node_graph".to_owned()),
            should_report_loss,
            "{case}: unexpected world graph losses: {features:?}"
        );
    }
    Ok(())
}
