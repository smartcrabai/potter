//! Principled BSDF lobes against Cycles v5.2.2 (Blender 5.2.2 oracle), ten material variants.
//!
//! Each variant is one radius-1 UV sphere lit either by the three disk area lights (`area_*`) or by a
//! uniform world of colour `0.05087608844041824` at strength 1 (`world_*`). The Principled material
//! takes base colour, metallic and specular IOR level from the fixture, with roughness 0.24 and IOR
//! 1.5. Blender renders seeds 0..8 at 512 spp, 128x128; the central 12x12 patch of each frame is
//! averaged. Potter renders the same seeds and patch.
//!
//! The committed oracle is `tests/fixtures/native_oracle/principled/blender_variant_means.json`,
//! distilled from the generator `tests/fixtures/native_oracle/principled/blender_variants.py`. The
//! generator's EXR outputs are NOT copied into the repository because of their size; the JSON is the
//! committed oracle.
//!
//! Tolerance, per channel: `4 * sqrt(stderr_blender^2 + sigma_potter^2)`. `stderr_blender` is the seed
//! standard error stored in the fixture. `sigma_potter` is Potter's own seed standard error: the sample
//! standard deviation (n - 1) of its eight patch means divided by `sqrt(8)`. The band is derived only
//! from those two measured standard errors. The factor 4 is a fixed band width, not a fitted constant.
//!
//! Scene authoring: the Potter scene is imported from a `.blend` that Blender writes from the same
//! setup as `make_scene()` in the generator. `light.create` cannot set an area light's shape or size,
//! and the three disk lights need both. This is the import path that `point_spot_light.rs` uses.
//! Potter exposes one `max_bounces` (12); Blender's diffuse and glossy caps of 4 have no Potter
//! counterpart. They are immaterial here because a convex sphere sees only the environment and the
//! lights after its first bounce.
//!
//! Gated like the other native tests: skipped when Blender is absent, and `POTTER_REQUIRE_BLENDER`
//! turns that absence into a failure.

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

use std::fs;

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments, vec3,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// Committed Blender oracle: per-variant patch means, seed standard errors and variant parameters.
const ORACLE: &str = include_str!("fixtures/native_oracle/principled/blender_variant_means.json");

/// Blender scene author for one variant. It mirrors `make_scene()` in `blender_variants.py` and
/// saves the `.blend` without rendering. The variant parameters arrive as a JSON file.
const SCENE_SCRIPT: &str = r"import json
import sys

import bpy
from mathutils import Vector

out_blend, params_path = sys.argv[sys.argv.index('--') + 1:][:2]
with open(params_path, encoding='utf-8') as handle:
    params = json.load(handle)

WORLD = 0.05087608844041824
LIGHTS = [
    ('Key_Softbox', (0.0, -7.0, 13.0), 1800.0, 9.0, (0.80, 0.88, 1.0)),
    ('Warm_Rim', (7.0, 8.0, 10.0), 1300.0, 7.0, (1.0, 0.58, 0.31)),
    ('Cool_Fill', (-8.0, 1.0, 6.0), 900.0, 6.0, (0.34, 0.58, 1.0)),
]

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 512
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 12
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 12
scene.cycles.caustics_reflective = True
scene.cycles.caustics_refractive = True
scene.render.resolution_x = 128
scene.render.resolution_y = 128
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.film_transparent = False

world = bpy.data.worlds.new('ProbeWorld')
world.use_nodes = True
background = world.node_tree.nodes['Background']
background.inputs['Color'].default_value = (WORLD, WORLD, WORLD, 1.0)
background.inputs['Strength'].default_value = 1.0 if params['lighting'] == 'world' else 0.0
scene.world = world

bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0, location=(0, 0, 0))
sphere = bpy.context.object
for polygon in sphere.data.polygons:
    polygon.use_smooth = True
material = bpy.data.materials.new('Ceramic')
material.use_nodes = True
bsdf = material.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = tuple(params['base_color'])
bsdf.inputs['Metallic'].default_value = params['metallic']
bsdf.inputs['Roughness'].default_value = params['roughness']
bsdf.inputs['IOR'].default_value = params['ior']
bsdf.inputs['Specular IOR Level'].default_value = params['specular_ior_level']
sphere.data.materials.append(material)

if params['lighting'] == 'area':
    for name, location, energy, size, color in LIGHTS:
        data = bpy.data.lights.new(name, 'AREA')
        data.shape = 'DISK'
        data.size = size
        data.energy = energy
        data.color = color
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        aim(light, (0.0, 0.0, 0.0))

camera_data = bpy.data.cameras.new('ProbeCamera')
camera_data.type = 'ORTHO'
camera_data.ortho_scale = 3.5
camera = bpy.data.objects.new('ProbeCamera', camera_data)
scene.collection.objects.link(camera)
camera.location = (3.0, -5.0, 2.5)
aim(camera, (0.0, 0.0, 0.0))
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=out_blend)
";

/// Seeds rendered by both Blender and Potter: `0..SEEDS`.
const SEEDS: u64 = 8;
/// Path-tracing samples per pixel.
const SAMPLES: u64 = 512;
/// Path-tracing bounce limit (Potter's only bounce control).
const MAX_BOUNCES: u64 = 12;
/// Square frame edge in pixels.
const RESOLUTION: usize = 128;
/// Half-width of the centred patch. The patch is `2 * PATCH_HALF` pixels square.
const PATCH_HALF: usize = 6;
/// Band width in standard errors. A fixed width, not fitted to either renderer.
const BAND_SIGMAS: f64 = 4.0;

/// Checks one variant of the oracle against a Potter render of the same scene.
fn check_variant(name: &str) -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let oracle: Value = serde_json::from_str(ORACLE)?;
    let wanted = &oracle["variants"][name];
    if wanted.is_null() {
        return Err(format!("the oracle has no variant `{name}`").into());
    }
    let blender_mean = vec3(&wanted["mean"])?;
    let blender_stderr = vec3(&wanted["stderr"])?;

    let root = tempdir()?;
    let params_path = root.path().join("variant.json");
    fs::write(
        &params_path,
        serde_json::to_vec(&json!({
            "lighting": wanted["lighting"],
            "base_color": wanted["base_color"],
            "metallic": wanted["metallic"],
            "specular_ior_level": wanted["specular_ior_level"],
            "roughness": wanted["roughness"],
            "ior": wanted["ior"],
        }))?,
    )?;
    let script = write_script(root.path(), "principled_variant.py", SCENE_SCRIPT)?;
    let blend = root.path().join("variant.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), params_path.as_path()],
    )?;

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();

    let first = RESOLUTION / 2 - PATCH_HALF;
    let patch_count = (4 * PATCH_HALF * PATCH_HALF) as f64;
    let mut seed_means: Vec<[f64; 3]> = Vec::new();
    for seed in 0..SEEDS {
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": SAMPLES, "seed": seed, "max_bounces": MAX_BOUNCES}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");

        let frames = root.path().join(format!("frames_{seed}"));
        let rendered = pot()
            .arg("render")
            .arg(&project)
            .args(["--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "{name} seed {seed}: render failed: {}",
            String::from_utf8_lossy(&rendered.stdout)
        );
        let image_id = format!("render_{seed}");
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op": "image.load", "id": image_id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");

        let scene = read_scene(&project)?;
        let image: Image = serde_json::from_value(scene["images"][image_id.as_str()].clone())?;
        let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        let mut total = [0.0_f64; 3];
        for y in first..first + 2 * PATCH_HALF {
            for x in first..first + 2 * PATCH_HALF {
                let uv = [
                    (x as f64 + 0.5) / RESOLUTION as f64,
                    (y as f64 + 0.5) / RESOLUTION as f64,
                ];
                let pixel = sample(&data, uv, 0);
                for (sum, value) in total.iter_mut().zip(pixel.iter().take(3)) {
                    *sum += value;
                }
            }
        }
        seed_means.push(total.map(|sum| sum / patch_count));
    }

    let seeds = seed_means.len() as f64;
    let mut failures = Vec::new();
    for channel in 0..3 {
        let values: Vec<f64> = seed_means.iter().map(|mean| mean[channel]).collect();
        let potter_mean = values.iter().sum::<f64>() / seeds;
        let variance = values
            .iter()
            .map(|value| (value - potter_mean).powi(2))
            .sum::<f64>()
            / (seeds - 1.0);
        let sigma_potter = (variance / seeds).sqrt();
        let stderr_blender = blender_stderr[channel];
        let blender_value = blender_mean[channel];
        let tolerance = BAND_SIGMAS * (stderr_blender.powi(2) + sigma_potter.powi(2)).sqrt();
        let difference = potter_mean - blender_value;
        let relative_percent = 100.0 * difference / blender_value;
        if difference.abs() > tolerance {
            failures.push(format!(
                "channel {channel}: Potter {potter_mean:.6e} vs Blender {blender_value:.6e}; \
                 error {difference:+.3e} ({relative_percent:+.2}% relative); tolerance \
                 {tolerance:.3e} = 4*sqrt(stderr_blender {stderr_blender:.3e}^2 + \
                 sigma_potter {sigma_potter:.3e}^2)"
            ));
        }
    }
    assert!(failures.is_empty(), "{name}: {}", failures.join("; "));
    Ok(())
}

#[test]
fn principled_area_default_matches_cycles() -> TestResult {
    check_variant("area_default")
}

#[test]
fn principled_area_spec0_pure_diffuse_matches_cycles() -> TestResult {
    check_variant("area_spec0_pure_diffuse")
}

#[test]
fn principled_area_spec_only_black_base_matches_cycles() -> TestResult {
    check_variant("area_spec_only_black_base")
}

#[test]
fn principled_area_metal024_matches_cycles() -> TestResult {
    check_variant("area_metal024")
}

#[test]
fn principled_area_metal1_matches_cycles() -> TestResult {
    check_variant("area_metal1")
}

#[test]
fn principled_world_default_matches_cycles() -> TestResult {
    check_variant("world_default")
}

#[test]
fn principled_world_spec0_pure_diffuse_matches_cycles() -> TestResult {
    check_variant("world_spec0_pure_diffuse")
}

#[test]
fn principled_world_spec_only_black_base_matches_cycles() -> TestResult {
    check_variant("world_spec_only_black_base")
}

#[test]
fn principled_world_metal024_matches_cycles() -> TestResult {
    check_variant("world_metal024")
}

#[test]
fn principled_world_metal1_matches_cycles() -> TestResult {
    check_variant("world_metal1")
}

/// The committed oracle covers the ten variants, each with a 3-channel mean and seed stderr, and
/// records the eight seeds. This runs without Blender, so a broken fixture fails on every machine.
#[test]
fn principled_oracle_fixture_covers_ten_variants_over_eight_seeds() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(
        oracle["provenance"]["seeds"],
        json!([0, 1, 2, 3, 4, 5, 6, 7])
    );
    let variants = oracle["variants"]
        .as_object()
        .ok_or("the oracle has no variants object")?;
    assert_eq!(variants.len(), 10, "{variants:?}");
    for (name, variant) in variants {
        vec3(&variant["mean"]).map_err(|error| format!("{name} mean: {error}"))?;
        let stderr = vec3(&variant["stderr"]).map_err(|error| format!("{name} stderr: {error}"))?;
        assert!(
            stderr.iter().all(|value| *value > 0.0),
            "{name}: seed stderr must be positive, got {stderr:?}"
        );
        assert!(
            matches!(variant["lighting"].as_str(), Some("area" | "world")),
            "{name}: lighting must be area or world"
        );
    }
    Ok(())
}
