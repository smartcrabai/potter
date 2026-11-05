//! The path tracer's bounce budget follows Cycles d13: the kernel's `max_bounce` is the user's `max_bounces` plus one
//! (`intern/cycles/scene/integrator.cpp`). A refraction therefore reaches the backdrop behind its second crossing at
//! `max_bounces = 1`, and a hair transmission keeps the emitter its transmitted ray reaches at `max_bounces = 0`. The
//! committed Blender captures in `path_seeds.json` (eight bounces) give the reference for these low budgets. A ray that
//! leaves the last allowed bounce attenuates through volumes without scattering (`PATH_RAY_TERMINATE_AFTER_TRANSPARENT`).

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::path::Path;

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURES: &str = "tests/fixtures/native_oracle/shader_bsdf_wave12";
const PATH_SEEDS_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/path_seeds.json");
const SIZE: u32 = 192;
const SAMPLES: u32 = 256;
const TRANSPARENT_MAX_BOUNCES: u32 = 8;
const SEEDS: [u64; 2] = [17, 29];
const RELATIVE_TOLERANCE: f64 = 0.1;
const ABSOLUTE_TOLERANCE: f64 = 0.002;

/// A black world, a diffuse floor at z = 0, and a point light at (-5, 0, 5). The orthographic camera sees
/// x, y in [-0.4, 0.4]. The optional fog box spans x in [0.5, 1.5], so camera rays and the floor's shadow rays to the
/// light never enter it; only the floor's diffuse bounce rays do.
const FOG_SCENE_SCRIPT: &str = r#"
import bpy
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
out_path, with_fog = argv[0], argv[1] == "fog"

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
world = bpy.data.worlds.new("Black")
world.use_nodes = True
background = world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
background.inputs["Strength"].default_value = 0.0
scene.world = world

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.object
floor_material = bpy.data.materials.new("Floor")
floor_material.use_nodes = True
floor.data.materials.append(floor_material)

light_data = bpy.data.lights.new("Key", type="POINT")
light_data.energy = 400.0
light = bpy.data.objects.new("Key", light_data)
scene.collection.objects.link(light)
light.location = (-5.0, 0.0, 5.0)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 0.8
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

if with_fog:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.0, 0.0, 1.0))
    box = bpy.context.object
    fog = bpy.data.materials.new("Fog")
    fog.use_nodes = True
    tree = fog.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    for node in list(tree.nodes):
        if node.type == "BSDF_PRINCIPLED":
            tree.nodes.remove(node)
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 4.0
    volume.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    volume.inputs["Anisotropy"].default_value = 0.0
    tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
    box.data.materials.append(fog)

bpy.ops.wm.save_as_mainfile(filepath=out_path)
"#;

#[test]
fn transmission_events_follow_cycles_bounce_budget() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping transmission bounce budget; no executable Blender was found");
        return Ok(());
    };
    let captures: Value = serde_json::from_str(PATH_SEEDS_CAPTURE)?;
    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "transmission_budget_project",
        &fixture_dir.join("path.blend"),
    )?;
    // The backdrop behind the glass sphere is reached through its back crossing, one bounce beyond the front one.
    check_budget(&project, root.path(), &captures, "refraction_ggx", 1)?;
    // The direct light of the hair transmission, including the emitter its transmitted ray reaches, needs no bounce.
    check_budget(&project, root.path(), &captures, "hair_transmission", 0)?;
    Ok(())
}

/// At `max_bounces = 0` the floor's bounce ray leaves its last allowed bounce as a terminal ray. It may cross the
/// fog only as attenuation, so the render with the fog box equals the render without it (a black world gives no
/// other contribution). The fog must not change any pixel that the floor's own direct light reaches.
#[test]
fn terminal_bounce_ray_attenuates_through_fog_without_in_scatter() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping terminal fog regression; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "fog_scene.py", FOG_SCENE_SCRIPT)?;
    let plain_blend = root.path().join("plain.blend");
    let fog_blend = root.path().join("fog.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[plain_blend.as_path(), Path::new("plain")],
    )?;
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[fog_blend.as_path(), Path::new("fog")],
    )?;
    let (plain_project, _) = import_blend(&blender, root.path(), "plain_project", &plain_blend)?;
    let (fog_project, _) = import_blend(&blender, root.path(), "fog_project", &fog_blend)?;

    configure_render(&plain_project, root.path(), 17, 0, 128)?;
    configure_render(&fog_project, root.path(), 17, 0, 128)?;
    let plain = render_frame(&plain_project, root.path(), "plain_mb0")?;
    let fog = render_frame(&fog_project, root.path(), "fog_mb0")?;

    let mut plain_total = 0.0;
    let mut largest_difference: f64 = 0.0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let a = sample_pixel(&plain, [x, y]);
            let b = sample_pixel(&fog, [x, y]);
            plain_total += a[0] + a[1] + a[2];
            for channel in 0..4 {
                largest_difference = largest_difference.max((a[channel] - b[channel]).abs());
            }
        }
    }
    assert!(
        plain_total > 1.0e-3,
        "the floor's direct light should be visible; plain render total {plain_total}"
    );
    assert!(
        largest_difference <= 1.0e-6,
        "fog changed a max_bounces = 0 render by {largest_difference}: the floor's bounce ray scatters in fog"
    );
    Ok(())
}

fn check_budget(
    project: &Path,
    root: &Path,
    captures: &Value,
    case: &str,
    max_bounces: u32,
) -> TestResult {
    let entry = captures["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|candidate| candidate["name"] == case))
        .ok_or_else(|| format!("path_seeds.json has no case `{case}`"))?;
    let probe: [u32; 3] = serde_json::from_value(entry["probe"].clone())?;
    let mut reference = [0.0; 4];
    for seed in SEEDS {
        let candidate = entry["seed_captures"]
            .as_array()
            .and_then(|seeds| seeds.iter().find(|candidate| candidate["seed"] == seed))
            .ok_or_else(|| format!("`{case}` has no Blender capture for seed {seed}"))?;
        let linear: [f64; 4] = serde_json::from_value(candidate["linear_rgba"].clone())?;
        for (sum, value) in reference.iter_mut().zip(linear) {
            *sum += value / SEEDS.len() as f64;
        }
    }
    let mut potter = [0.0; 4];
    for seed in SEEDS {
        let name = format!("{case}_mb{max_bounces}_seed{seed}");
        configure_render(project, root, seed as u32, max_bounces, SAMPLES)?;
        let frame = render_frame(project, root, &name)?;
        for (sum, value) in potter.iter_mut().zip(patch_mean(&frame, probe)) {
            *sum += value / SEEDS.len() as f64;
        }
    }
    for channel in 0..3 {
        let tolerance = RELATIVE_TOLERANCE * reference[channel] + ABSOLUTE_TOLERANCE;
        assert!(
            (potter[channel] - reference[channel]).abs() <= tolerance,
            "{case} at max_bounces {max_bounces}: Potter {potter:?} differs from Blender {reference:?}"
        );
    }
    Ok(())
}

fn configure_render(
    project: &Path,
    root: &Path,
    seed: u32,
    max_bounces: u32,
    samples: u32,
) -> TestResult {
    let scene = read_scene(project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": SIZE,
                "resolution_y": SIZE,
                "samples": samples,
                "seed": seed,
                "max_bounces": max_bounces,
                "transparent_max_bounces": TRANSPARENT_MAX_BOUNCES
            }
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok(())
}

fn render_frame(project: &Path, root: &Path, name: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": name, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image = serde_json::from_value(read_scene(project)?["images"][name].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

fn sample_pixel(frame: &ImageData, pixel: [u32; 2]) -> [f64; 4] {
    sample(
        frame,
        [
            (f64::from(pixel[0]) + 0.5) / f64::from(frame.width),
            (f64::from(pixel[1]) + 0.5) / f64::from(frame.height),
        ],
        false,
    )
}

fn patch_mean(frame: &ImageData, probe: [u32; 3]) -> [f64; 4] {
    let [x, y, radius] = probe;
    assert!(x >= radius && y >= radius && x + radius < frame.width && y + radius < frame.height);
    let mut total = [0.0; 4];
    let mut count = 0.0;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let pixel = sample_pixel(frame, [px, py]);
            for (sum, value) in total.iter_mut().zip(pixel) {
                *sum += value;
            }
            count += 1.0;
        }
    }
    total.map(|sum| sum / count)
}
