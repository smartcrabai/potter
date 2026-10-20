//! Closure-list parity against Blender 5.2.2's Cycles renders.
//!
//! `tests/fixtures/closure_list/generate_closure_oracle.py` renders three scenes with Cycles and commits each scene's
//! `.blend`, its linear EXR, and a JSON of per-pixel RGB values over a fixed region. The scenes exercise closures that
//! Cycles keeps as a list rather than blending one parameter set: a Mix of two Principled BSDFs with different roughness
//! and normals (a Bump drives one arm), an Add of Diffuse and Glossy, and a Mix of Glass and Diffuse.
//!
//! Each `.blend` is imported into Potter and rendered with two seeds. Every Potter render is compared with the committed
//! Cycles values as a mean absolute error over the region. The bound is measured, not chosen: Potter's seed-to-seed
//! error `d` is its noise floor, and each Potter render must lie within `NOISE_FACTOR * d` of Cycles. The Cycles
//! references are fixed; they are never regenerated from Potter's output.

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

use std::{fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{TestResult, apply_operations, checked, import_blend, pot, read_scene, vec3};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

/// The frame edge of the committed Cycles renders (`generate_closure_oracle.py`).
const RESOLUTION: u32 = 32;
/// Samples per pixel of the Potter renders. The Cycles references use 16384, so Potter's noise is the larger one.
const POTTER_SAMPLES: u32 = 1024;
/// The `max_bounces` of the Potter renders, the same budget as the Cycles references.
const MAX_BOUNCES: u32 = 8;
/// The two seeds of each Potter render. Their difference is Potter's noise floor.
const SEEDS: [u32; 2] = [1, 2];
/// The bound multiplier `k`: each Potter render must lie within `k * d` of Cycles, where `d` is the mean absolute error
/// between the two Potter seeds.
///
/// Let `s_P` be the per-pixel noise of Potter and `s_C` that of Cycles. Without bias, one Potter render differs from
/// Cycles by about `sqrt(s_P^2 + s_C^2)` per pixel, and the two Potter seeds differ by `sqrt(2) * s_P`. The expected
/// ratio of their mean absolute errors is therefore `sqrt((s_P^2 + s_C^2) / (2 * s_P^2))`. The Cycles seed-to-seed
/// error recorded in each JSON is below 3e-5, so `s_C` is negligible and the ratio is about 0.71. `k = 2` leaves a
/// factor of 2.8 over that, which absorbs the sampling spread of `d` over 256 pixels and three channels. It still
/// rejects a per-pixel bias `b` above about 2.6 `s_P` (`sqrt(7) * s_P`), where `sqrt(s_P^2 + b^2) <= 2 * sqrt(2) * s_P`
/// stops holding.
const NOISE_FACTOR: f64 = 2.0;
/// The committed scenes, their `.blend` files, and their Cycles references.
const FIXTURES: &str = "tests/fixtures/closure_list";

/// A rectangle of pixels from a top-left origin.
#[derive(Clone, Copy)]
struct Region {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// The committed Cycles values of one scene: the region and its per-pixel RGB, row-major from the top-left.
struct Reference {
    region: Region,
    pixels: Vec<[f64; 3]>,
}

#[test]
fn mix_of_two_principled_bsdfs_with_roughness_and_bump_normals_matches_cycles() -> TestResult {
    check_scene("mix_principled")
}

#[test]
fn add_of_diffuse_and_glossy_matches_cycles() -> TestResult {
    check_scene("add_diffuse_glossy")
}

#[test]
fn mix_of_glass_and_diffuse_matches_cycles() -> TestResult {
    check_scene("mix_glass_diffuse")
}

/// Imports the committed `.blend` of `name`, renders it with each seed, and checks every Potter frame against the
/// committed Cycles reference. Skips when no Blender executable exists, and fails when `POTTER_REQUIRE_BLENDER` is set.
fn check_scene(name: &str) -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the closure-list oracle `{name}`; no executable Blender was found");
        return Ok(());
    };
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let reference = read_reference(&fixtures.join(format!("{name}.json")))?;
    let root = tempdir()?;
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "project",
        &fixtures.join(format!("{name}.blend")),
    )?;

    let mut renders = Vec::with_capacity(SEEDS.len());
    for seed in SEEDS {
        configure_render(&project, root.path(), seed)?;
        let frame = render_frame(&project, root.path(), &format!("{name}_seed{seed}"))?;
        renders.push(region_pixels(&frame, reference.region)?);
    }

    let noise = mean_abs_error(&renders[0], &renders[1]);
    assert!(
        noise > 0.0,
        "{name}: the Potter seeds {SEEDS:?} render identically, so the noise floor is zero"
    );
    let bound = NOISE_FACTOR * noise;
    for (seed, pixels) in SEEDS.iter().zip(&renders) {
        let error = mean_abs_error(pixels, &reference.pixels);
        eprintln!(
            "closure-list {name}: Potter seed {seed} vs Cycles {error:.6}; seed-to-seed floor {noise:.6}; bound {bound:.6}"
        );
        assert!(
            error <= bound,
            "{name}: Potter seed {seed} differs from Cycles by {error:.6}, above {NOISE_FACTOR} x the seed-to-seed floor {noise:.6} = {bound:.6}"
        );
    }
    Ok(())
}

/// Reads the committed Cycles reference at `path`, checking its resolution and region size.
fn read_reference(path: &Path) -> TestResult<Reference> {
    let document: Value = serde_json::from_slice(&fs::read(path)?)?;
    if document["resolution"] != json!([RESOLUTION, RESOLUTION]) {
        return Err(format!(
            "{}: the resolution is not {RESOLUTION}x{RESOLUTION}",
            path.display()
        )
        .into());
    }
    let region_json = &document["region"];
    let field = |key: &str| -> TestResult<u32> {
        let value = region_json[key]
            .as_u64()
            .ok_or_else(|| format!("region.{key} is missing"))?;
        Ok(u32::try_from(value)?)
    };
    let region = Region {
        x: field("x")?,
        y: field("y")?,
        width: field("width")?,
        height: field("height")?,
    };
    let pixels = document["pixels"]
        .as_array()
        .ok_or("the reference pixels are missing")?
        .iter()
        .map(vec3)
        .collect::<TestResult<Vec<_>>>()?;
    if pixels.len() != (region.width * region.height) as usize {
        return Err(format!(
            "{}: {} pixels for a {}x{} region",
            path.display(),
            pixels.len(),
            region.width,
            region.height
        )
        .into());
    }
    Ok(Reference { region, pixels })
}

/// Sets the active scene's render size, samples, seed, and bounce budget for one Potter render.
fn configure_render(project: &Path, root: &Path, seed: u32) -> TestResult {
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
                "resolution_x": RESOLUTION,
                "resolution_y": RESOLUTION,
                "samples": POTTER_SAMPLES,
                "seed": seed,
                "max_bounces": MAX_BOUNCES
            }
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok(())
}

/// Renders `project` to a linear EXR under `root/name`, loads it as the image `name`, and returns the frame.
fn render_frame(project: &Path, root: &Path, name: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
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

/// The Potter frame's RGB at each pixel centre of `region`, row-major from the top-left, in the Cycles reference's order.
fn region_pixels(frame: &ImageData, region: Region) -> TestResult<Vec<[f64; 3]>> {
    if frame.width != RESOLUTION || frame.height != RESOLUTION {
        return Err(format!(
            "the Potter frame is {}x{}, not {RESOLUTION}x{RESOLUTION}",
            frame.width, frame.height
        )
        .into());
    }
    let mut pixels = Vec::with_capacity((region.width * region.height) as usize);
    for y in region.y..region.y + region.height {
        for x in region.x..region.x + region.width {
            // `sample` takes fractions of the frame and reads Potter's top-down storage, so `y` is the storage row.
            let uv = [
                (f64::from(x) + 0.5) / f64::from(RESOLUTION),
                (f64::from(y) + 0.5) / f64::from(RESOLUTION),
            ];
            let [red, green, blue, _] = sample(frame, uv, false);
            pixels.push([red, green, blue]);
        }
    }
    Ok(pixels)
}

/// The mean absolute difference over every pixel and RGB channel.
fn mean_abs_error(actual: &[[f64; 3]], wanted: &[[f64; 3]]) -> f64 {
    let total: f64 = actual
        .iter()
        .zip(wanted)
        .flat_map(|(a, w)| a.iter().zip(w).map(|(x, y)| (x - y).abs()))
        .sum();
    total / (actual.len() * 3) as f64
}
