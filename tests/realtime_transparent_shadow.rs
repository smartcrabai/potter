//! Realtime shadows from transparent surfaces against Blender's EEVEE 5.2.2 renders.
//!
//! A transparent occluder (a BLENDED Principled with Alpha 0.5) casts a shadow half as dense as an opaque one, as
//! EEVEE's transparent shadow pass gives (`eevee_surf_shadow.bsl.hh`). With `use_transparent_shadow` off the occluder
//! casts an opaque shadow. Each render is reduced to the mean RGB of a 9 by 9 grid over the frame, and the test compares
//! the shadowed-to-lit ratio of each occluder with EEVEE's. EEVEE's ratios and their tolerances are committed in
//! `realtime_transparent_shadow_spread.json`, which `measure_realtime_transparent_shadow_spread.py` derives from its
//! renders, so this test writes each scene with Blender and renders only Potter.

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
use tempfile::tempdir;

const SCRIPT: &str = include_str!("fixtures/native_oracle/realtime_transparent_shadow.py");
/// Patch grid of the Blender oracle (`realtime_transparent_shadow.py`): fractions of the frame.
const PATCH: [f64; 9] = [
    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
];
/// EEVEE's ratios and the tolerances of each (`measure_realtime_transparent_shadow_spread.py` derives them): per light
/// and ratio, four times EEVEE's noise over independent realisations, plus EEVEE's bias against the exact ratio over the
/// pixel footprints, plus Potter's edge-placement bound. The point light's transparent bound is zero, since Potter traces
/// that occluder exactly; its opaque bound is one depth-map edge error, which can flip a pixel of the 81-pixel patch.
const SPREAD: &str = include_str!("fixtures/native_oracle/realtime_transparent_shadow_spread.json");

/// The committed EEVEE mean ratio and tolerance of `ratio` (`alpha_ratio` or `alpha_off_ratio`) for `light`.
fn eevee(light: &str, ratio: &str) -> TestResult<(f64, f64)> {
    let spread: Value = serde_json::from_str(SPREAD)?;
    let entry = &spread["lights"][light][ratio];
    let mean = entry["mean"]
        .as_f64()
        .ok_or_else(|| format!("no EEVEE mean for {light} {ratio}"))?;
    let tolerance = entry["tolerance"]
        .as_f64()
        .ok_or_else(|| format!("no tolerance for {light} {ratio}"))?;
    Ok((mean, tolerance))
}

/// The occluder states of the oracle: a transparent shadow, an opaque one, and no occluder.
#[derive(Clone, Copy)]
enum Occluder {
    Alpha,
    AlphaOff,
    Absent,
}

impl Occluder {
    fn name(self) -> &'static str {
        match self {
            Self::Alpha => "ALPHA",
            Self::AlphaOff => "ALPHA_OFF",
            Self::Absent => "NONE",
        }
    }
}

/// Writes the Blender scene of `light` with `occluder` as a .blend, without rendering it, and returns its path.
fn blender_scene(
    blender: &Path,
    root: &Path,
    light: &str,
    occluder: Occluder,
) -> TestResult<PathBuf> {
    let dir = root.join(format!("{light}_{}", occluder.name()));
    fs::create_dir_all(&dir)?;
    let script = write_script(&dir, "realtime_transparent_shadow.py", SCRIPT)?;
    let blend = dir.join("scene.blend");
    // The render flag `0` stops the generator after saving the .blend, so the image and JSON paths are not written.
    run_blender_arguments(
        blender,
        &script,
        None,
        &[
            blend.as_path(),
            dir.join("unused.exr").as_path(),
            dir.join("unused.json").as_path(),
            Path::new(light),
            Path::new(occluder.name()),
            Path::new("0"),
            Path::new("0"),
        ],
    )?;
    Ok(blend)
}

/// Imports the Blender scene into its own project, since `pot init` refuses a non-empty directory, and returns the
/// project and the imported scene JSON.
fn import_scene(blender: &Path, root: &Path, blend: &Path) -> TestResult<(PathBuf, Value)> {
    let name = blend
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or("the blend has no directory name")?;
    let (project, _) = import_blend(blender, root, &format!("project_{name}"), blend)?;
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
    Ok((project, read_scene(&project)?))
}

/// Renders `project` in the realtime engine to a linear EXR and loads the frame.
fn potter_realtime_frame(project: &Path, root: &Path, name: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
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

/// The mean over RGB of the 9 by 9 grid of a rendered frame, as the Blender oracle takes it.
fn patch_mean(data: &ImageData) -> f64 {
    let mut total = 0.0;
    for v in PATCH {
        for u in PATCH {
            let pixel = sample(data, [u, v], false);
            total += (pixel[0] + pixel[1] + pixel[2]) / 3.0;
        }
    }
    total / (PATCH.len() * PATCH.len()) as f64
}

/// Checks Potter's realtime shadowed-to-lit ratios of the transparent and the opaque occluder against EEVEE's, and that
/// each imported occluder carries its `use_transparent_shadow` setting.
fn assert_shadow_ratios_match(light: &str) -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the realtime transparent shadow oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let root = tempdir()?;

    let mut lit = 0.0;
    let mut potter = Vec::new();
    for occluder in [Occluder::Absent, Occluder::Alpha, Occluder::AlphaOff] {
        let blend = blender_scene(&blender, root.path(), light, occluder)?;
        let (project, scene) = import_scene(&blender, root.path(), &blend)?;
        let expected = match occluder {
            Occluder::Alpha => Some(true),
            Occluder::AlphaOff => Some(false),
            Occluder::Absent => None,
        };
        if let Some(expected) = expected {
            let imported = scene["materials"]
                .as_object()
                .and_then(|materials| {
                    materials
                        .values()
                        .find(|material| material["name"] == "Occluder")
                })
                .ok_or("the occluder material is missing")?;
            assert_eq!(
                imported["use_transparent_shadow"],
                expected,
                "{light} {}: use_transparent_shadow must import from the .blend: {imported}",
                occluder.name()
            );
        }
        let frame = potter_realtime_frame(&project, root.path(), &occluder.name().to_lowercase())?;
        let mean = patch_mean(&frame);
        match occluder {
            Occluder::Absent => lit = mean,
            Occluder::Alpha | Occluder::AlphaOff => potter.push(mean / lit),
        }
    }
    let (potter_alpha, potter_off) = (potter[0], potter[1]);

    let (eevee_alpha, alpha_tolerance) = eevee(light, "alpha_ratio")?;
    let (eevee_off, off_tolerance) = eevee(light, "alpha_off_ratio")?;
    assert!(
        (potter_alpha - eevee_alpha).abs() <= alpha_tolerance,
        "{light}: Potter's transparent shadow ratio {potter_alpha:.4} differs from EEVEE's {eevee_alpha:.4} by more \
         than {alpha_tolerance:.4} (lit: Potter {lit:.4})"
    );
    assert!(
        (potter_off - eevee_off).abs() <= off_tolerance,
        "{light}: Potter's opaque shadow ratio {potter_off:.4} differs from EEVEE's {eevee_off:.4} by more than \
         {off_tolerance:.4}"
    );
    Ok(())
}

#[test]
fn sun_transparent_shadow_matches_eevee_in_realtime() -> TestResult {
    assert_shadow_ratios_match("SUN")
}

#[test]
fn point_transparent_shadow_matches_eevee_in_realtime() -> TestResult {
    assert_shadow_ratios_match("POINT")
}
