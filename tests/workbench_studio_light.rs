//! Regression: the Solid preview lights surfaces as Blender 5.2.2's Workbench does with the
//! factory `Default` studio light. The oracle is `fixtures/native_oracle/workbench_studio_light.py`:
//! a smooth sphere over a plane, rendered by Workbench from a perspective and an orthographic
//! camera. Blender's Workbench shading is compared pixel by pixel, away from edges.
//! A `.blend` with a Workbench display option Potter's Solid preview does not model imports with
//! a reported loss for it.

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

use std::{fs, io::BufReader, path::Path, process::Command};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, loss_features, pot, pot_json, run_blender_arguments,
    write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

/// Blender 5.2.2 oracle: Workbench STUDIO lighting of the sphere and plane, per camera.
const STUDIO_SCRIPT: &str = include_str!("fixtures/native_oracle/workbench_studio_light.py");

/// Per-channel difference, in 8-bit levels, allowed away from edges. The oracle's own values are
/// 8-bit, and the Solid preview quantizes roughness and metallic as Workbench's G-buffer does, so
/// the two agree within rounding.
const INTERIOR_TOLERANCE: u8 = 2;

/// A pixel is an edge of the reference when a neighbour differs from it by more than this many
/// levels. Silhouettes and material boundaries sit there, and a sub-pixel difference in
/// tessellation can flip them, so edge pixels get a frame budget instead of the interior tolerance.
const EDGE_STEP: u8 = 4;

/// Percent of the frame that may differ beyond `INTERIOR_TOLERANCE` on edges.
const EDGE_BUDGET_PERCENT: usize = 5;

struct Image {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}

impl Image {
    fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        self.pixels[y * self.width + x]
    }
}

fn read_image(path: &Path) -> TestResult<Image> {
    let mut reader = png::Decoder::new(BufReader::new(fs::File::open(path)?)).read_info()?;
    let mut buffer = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut buffer)?;
    let channels = info.color_type.samples();
    let pixels = buffer
        .chunks_exact(channels)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    Ok(Image {
        width: usize::try_from(info.width)?,
        height: usize::try_from(info.height)?,
        pixels,
    })
}

/// Largest channel difference between two colours, in 8-bit levels.
fn channel_difference(a: [u8; 3], b: [u8; 3]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.abs_diff(b))
        .max()
        .unwrap_or(0)
}

/// Whether a pixel of the reference is on an edge: a pixel of its 3 x 3 neighbourhood differs from
/// it by more than `EDGE_STEP`.
fn is_edge(reference: &Image, x: usize, y: usize) -> bool {
    let centre = reference.pixel(x, y);
    let rows = y.saturating_sub(1)..=(y + 1).min(reference.height - 1);
    let columns = x.saturating_sub(1)..=(x + 1).min(reference.width - 1);
    rows.flat_map(|row| columns.clone().map(move |column| (column, row)))
        .any(|(column, row)| channel_difference(centre, reference.pixel(column, row)) > EDGE_STEP)
}

fn assert_studio_lighting(camera: &str, reference: &Image, solid: &Image) {
    assert_eq!(
        (reference.width, reference.height),
        (solid.width, solid.height),
        "{camera}: the preview and the Workbench render differ in size"
    );
    let mut interior_worst: u8 = 0;
    let mut edge_outside = 0_usize;
    for y in 0..reference.height {
        for x in 0..reference.width {
            let difference = channel_difference(reference.pixel(x, y), solid.pixel(x, y));
            if !is_edge(reference, x, y) {
                interior_worst = interior_worst.max(difference);
            } else if difference > INTERIOR_TOLERANCE {
                edge_outside += 1;
            }
        }
    }
    assert!(
        interior_worst <= INTERIOR_TOLERANCE,
        "{camera}: away from edges the Solid preview differs from Blender's Workbench by up to \
         {interior_worst} levels; the tolerance is {INTERIOR_TOLERANCE}"
    );
    let frame = reference.width * reference.height;
    assert!(
        edge_outside * 100 <= frame * EDGE_BUDGET_PERCENT,
        "{camera}: {edge_outside} of {frame} pixels are off by more than {INTERIOR_TOLERANCE} \
         levels on edges; the budget is {EDGE_BUDGET_PERCENT} percent of the frame"
    );
}

#[test]
fn solid_preview_lights_surfaces_as_workbench_studio_lighting_does() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the Workbench studio lighting oracle; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let oracle = root.path().join("oracle");
    fs::create_dir(&oracle)?;
    let script = write_script(root.path(), "workbench_studio_light.py", STUDIO_SCRIPT)?;
    run_blender_arguments(&blender, &script, None, &[oracle.as_path()])?;

    let scene = root.path().join("scene");
    pot_json(|command| {
        command.arg("init").arg(&scene);
    })?;
    // Quaternions are [x,y,z,w] (spec.md). These are the oracle script's camera rotations, which
    // Blender takes as (w, x, y, z). Both cameras look at the origin with +Z up.
    let sphere_faces: Vec<String> = (0..32 * 16).map(|id| format!("f{id}")).collect();
    let applied = apply_operations(
        &scene,
        root.path(),
        json!([
            {"op": "material.create", "id": "ball_mat", "base_color": [0.8, 0.3, 0.2, 1.0],
             "roughness": 0.25, "metallic": 0.5},
            {"op": "material.create", "id": "floor_mat", "base_color": [0.8, 0.8, 0.8, 1.0],
             "roughness": 0.4, "metallic": 0.0},
            {"op": "node.create", "id": "ball", "kind": "sphere",
             "params": {"radius": 1.0, "segments": 32, "ring_count": 16},
             "material": "ball_mat"},
            {"op": "mesh.shade_smooth", "target": {"id": "ball"},
             "elements": {"domain": "face", "ids": sphere_faces}},
            {"op": "node.create", "id": "floor", "kind": "plane", "params": {"size": 4.0},
             "material": "floor_mat", "transform": {"translation": [0.0, 0.0, -1.0]}},
            {"op": "camera.create", "id": "perspective",
             "transform": {"translation": [3.2, -4.6, 2.6],
             "rotation": [0.513445087, 0.161024448, 0.252227132, 0.804255397]}},
            {"op": "camera.create", "id": "orthographic", "projection": "orthographic",
             "ortho_scale": 4.4,
             "transform": {"translation": [-2.6, -3.4, 3.9],
             "rotation": [0.382697454, -0.129555881, -0.293319548, 0.866441905]}}
        ]),
    )?;
    assert_eq!(applied["ok"], json!(true), "scene apply failed: {applied}");

    for camera in ["perspective", "orthographic"] {
        let out = root.path().join(format!("preview-{camera}"));
        let preview = pot_json(|command| {
            command
                .arg("preview")
                .arg(&scene)
                .args(["--camera", camera, "--mode", "solid", "--size", "128"])
                .arg("--out")
                .arg(&out);
        })?;
        let image = preview["result"]["previews"]
            .as_array()
            .and_then(|previews| previews.first())
            .and_then(|item| item["image"].as_str())
            .ok_or("preview result has no image path")?;
        let solid = read_image(Path::new(image))?;
        let reference = read_image(&oracle.join(format!("{camera}.png")))?;
        assert_studio_lighting(camera, &reference, &solid);
    }
    Ok(())
}

#[test]
fn blender_workbench_display_options_import_as_reported_losses() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the Workbench display-loss import; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("shading.blend");
    // Blender-side: a studio light other than Default, and cavity on, saved to a .blend.
    let expression = format!(
        "import bpy; shading = bpy.context.scene.display.shading; shading.studio_light = 'studio.sl'; shading.show_cavity = True; bpy.ops.wm.save_as_mainfile(filepath={:?})",
        blend.to_string_lossy()
    );
    checked(
        Command::new(&blender)
            .args([
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python-expr",
            ])
            .arg(expression),
        "saving the shading scene",
    )?;

    let project = root.path().join("project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    let import = |extra: &[&str]| {
        let mut command = pot();
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
            .args(extra);
        command
    };
    let strict = import(&["--json"]).output()?;
    assert!(
        !strict.status.success(),
        "a scene with Workbench options Potter does not model must not import silently"
    );
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&strict.stdout),
        String::from_utf8_lossy(&strict.stderr)
    );
    for feature in [
        "blender.scene.display_studio_light",
        "blender.scene.display_cavity",
    ] {
        assert!(
            report.contains(feature),
            "the strict import must name {feature}: {report}"
        );
    }

    let lossy: Value = pot_json(|command| {
        *command = import(&["--allow-lossy"]);
    })?;
    let features = loss_features(&lossy);
    for feature in [
        "blender.scene.display_studio_light",
        "blender.scene.display_cavity",
    ] {
        assert!(
            features.contains(&feature.to_owned()),
            "the lossy import must record {feature}; got {features:?}"
        );
    }
    Ok(())
}
