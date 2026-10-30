//! Native standalone shader BSDF path/realtime parity against committed Blender 5.2.2 linear captures.
//!
//! `fixtures/native_oracle/shader_bsdf_wave12/generate.py` builds the committed scenes and captures their
//! untransformed RGBA EXRs. The path scene covers every supported standalone node, including all distribution,
//! component, falloff, Fresnel, hair model and parametrization choices. The EEVEE scene contains only native
//! EEVEE-supported standalone nodes. Both tests import the source `.blend` through Potter and exercise the real
//! `pot render` CLI. Path results are checked against Blender's captured linear patch means with a seed-to-seed
//! noise bound; realtime results are checked at f32 capture precision. The holdout probes are over emissive
//! backplates so their zero-alpha holes and surrounding opaque pixels are both observed. The ray-portal case has
//! nonuniform backplates and a nonzero Position/Direction, so the output distinguishes portal redirection from an
//! ordinary surface closure.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeSet, fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, checked, f32_ulp_tolerance, import_blend, pot, read_scene,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde::Deserialize;
use serde_json::json;
use tempfile::tempdir;

const FIXTURES: &str = "tests/fixtures/native_oracle/shader_bsdf_wave12";
const SIZE: u32 = 192;
const CYCLES_SAMPLES: u32 = 256;
const EEVEE_SAMPLES: u32 = 64;
const PATH_SEEDS: [u32; 2] = [17, 29];
const CYCLES_CAPTURE: &str = include_str!("fixtures/native_oracle/shader_bsdf_wave12/path.json");
const EEVEE_CAPTURE: &str = include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee.json");

const PATH_CASES: &[&str] = &[
    "sheen_ashikhmin",
    "sheen_microfiber",
    "subsurface_burley",
    "subsurface_random_walk",
    "subsurface_random_walk_skin",
    "subsurface_random_walk_legacy",
    "translucent",
    "refraction_beckmann",
    "refraction_ggx",
    "toon_diffuse",
    "toon_glossy",
    "metallic_beckmann_f82",
    "metallic_ggx_f82",
    "metallic_multi_ggx_f82",
    "metallic_beckmann_conductor",
    "metallic_ggx_conductor",
    "metallic_multi_ggx_conductor",
    "holdout",
    "hair_reflection",
    "hair_transmission",
    "principled_hair_chiang_reflectance",
    "principled_hair_chiang_melanin",
    "principled_hair_chiang_absorption",
    "principled_hair_huang_reflectance",
    "principled_hair_huang_melanin",
    "principled_hair_huang_absorption",
    "ray_portal",
];
const EEVEE_CASES: &[&str] = &[
    "subsurface_burley",
    "subsurface_random_walk",
    "subsurface_random_walk_skin",
    "subsurface_random_walk_legacy",
    "translucent",
    "refraction_beckmann",
    "refraction_ggx",
    "metallic_beckmann_f82",
    "metallic_ggx_f82",
    "metallic_multi_ggx_f82",
    "metallic_beckmann_conductor",
    "metallic_ggx_conductor",
    "metallic_multi_ggx_conductor",
    "holdout",
];

#[derive(Deserialize)]
struct Capture {
    blender_version: String,
    engine: String,
    resolution: [u32; 2],
    samples: u32,
    seed: u32,
    cases: Vec<CaptureCase>,
}

#[derive(Deserialize)]
struct CaptureCase {
    name: String,
    probe: [u32; 3],
    linear_rgba: [f64; 4],
    #[serde(default)]
    neighbor_rgba: Option<[f64; 4]>,
    mode: serde_json::Value,
    inputs: serde_json::Value,
}
#[test]
fn path_standalone_bsdfs_match_blender_linear_captures() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping native standalone BSDF path parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(CYCLES_CAPTURE)?;
    check_capture(&capture, "CYCLES", CYCLES_SAMPLES, PATH_CASES)?;
    check_modes(&capture)?;

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "path_project",
        &fixture_dir.join("path.blend"),
    )?;
    let reference = capture
        .cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect::<std::collections::HashMap<_, _>>();
    let mut actuals = Vec::with_capacity(PATH_SEEDS.len());
    let mut holdout_neighbor = None;
    for seed in PATH_SEEDS {
        configure_render(&project, root.path(), seed, CYCLES_SAMPLES)?;
        let frame = render_frame(&project, root.path(), &format!("path_seed_{seed}"), "path")?;
        actuals.push(
            PATH_CASES
                .iter()
                .map(|name| patch_mean(&frame, reference[name].probe))
                .collect::<Vec<_>>(),
        );
        if seed == PATH_SEEDS[0] {
            let [x, y, _] = reference["holdout"].probe;
            holdout_neighbor = Some(sample_pixel(&frame, [x + 13, y]));
        }
    }
    let blender_noise = capture_seed_noise(&fixture_dir.join("path_noise.json"))?;
    for (case_index, name) in PATH_CASES.iter().enumerate() {
        let noise = rgba_mae(&actuals[0][case_index], &actuals[1][case_index]);
        let source_noise = blender_noise[*name];
        let tolerance = 2.0 * (noise + source_noise);
        let expected = reference[name].linear_rgba;
        for (seed, render) in PATH_SEEDS.iter().zip(&actuals) {
            for channel in 0..4 {
                let actual = render[case_index][channel];
                assert!(
                    (actual - expected[channel]).abs() <= tolerance,
                    "{name} path seed {seed} channel {channel}: Potter {actual} differs from Blender linear capture {} by more than 2 x (Potter seed noise {noise} + Blender seed noise {source_noise}) = {tolerance}",
                    expected[channel]
                );
            }
        }
    }
    check_holdout_alpha(&actuals[0], PATH_CASES, &reference, holdout_neighbor)?;
    Ok(())
}

#[test]
fn realtime_standalone_bsdfs_match_eevee_linear_captures() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping native standalone BSDF EEVEE parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(EEVEE_CAPTURE)?;
    check_capture(&capture, "BLENDER_EEVEE", EEVEE_SAMPLES, EEVEE_CASES)?;
    check_modes(&capture)?;

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "eevee_project",
        &fixture_dir.join("eevee.blend"),
    )?;
    configure_render(&project, root.path(), 17, EEVEE_SAMPLES)?;
    let frame = render_frame(&project, root.path(), "eevee", "realtime")?;
    let reference = capture
        .cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect::<std::collections::HashMap<_, _>>();
    for name in EEVEE_CASES {
        let expected = reference[name].linear_rgba;
        let actual = patch_mean(&frame, reference[name].probe);
        for channel in 0..4 {
            let tolerance = f32_ulp_tolerance(expected[channel], 4.0);
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "{name} EEVEE channel {channel}: Potter {} differs from Blender linear capture {} by more than the 32-bit EXR round-trip bound {tolerance}",
                actual[channel],
                expected[channel]
            );
        }
    }
    let holdout = reference["holdout"];
    let [holdout_x, holdout_y, _] = holdout.probe;
    let holdout_alpha = patch_mean(&frame, holdout.probe)[3];
    let expected_alpha = holdout.linear_rgba[3];
    if holdout_alpha.abs() > 1.0e-6 || expected_alpha.abs() > 1.0e-6 {
        return Err(format!("EEVEE holdout must leave a transparent hole: Potter alpha {holdout_alpha}, Blender alpha {expected_alpha}").into());
    }
    let neighbor = sample_pixel(&frame, [holdout_x + 13, holdout_y]);
    let expected_neighbor = holdout
        .neighbor_rgba
        .ok_or("Blender EEVEE holdout backplate capture is missing")?;
    if neighbor[3] < 0.999 || expected_neighbor[3] < 0.999 {
        return Err(format!("EEVEE holdout must preserve the opaque backplate outside its hole: Potter alpha {}, Blender alpha {}", neighbor[3], expected_neighbor[3]).into());
    }
    for channel in 0..4 {
        let tolerance = f32_ulp_tolerance(expected_neighbor[channel], 4.0);
        if (neighbor[channel] - expected_neighbor[channel]).abs() > tolerance {
            return Err(format!("EEVEE holdout backplate channel {channel}: Potter {} differs from Blender {} by more than {tolerance}", neighbor[channel], expected_neighbor[channel]).into());
        }
    }
    Ok(())
}

fn check_modes(capture: &Capture) -> TestResult {
    let expected = [
        ("sheen_ashikhmin", "distribution", "ASHIKHMIN"),
        ("sheen_microfiber", "distribution", "MICROFIBER"),
        ("subsurface_burley", "falloff", "BURLEY"),
        ("subsurface_random_walk", "falloff", "RANDOM_WALK"),
        ("subsurface_random_walk_skin", "falloff", "RANDOM_WALK_SKIN"),
        (
            "subsurface_random_walk_legacy",
            "falloff",
            "RANDOM_WALK_LEGACY",
        ),
        ("refraction_beckmann", "distribution", "BECKMANN"),
        ("refraction_ggx", "distribution", "GGX"),
        ("toon_diffuse", "component", "DIFFUSE"),
        ("toon_glossy", "component", "GLOSSY"),
        ("metallic_beckmann_f82", "distribution", "BECKMANN"),
        ("metallic_ggx_f82", "distribution", "GGX"),
        ("metallic_multi_ggx_f82", "distribution", "MULTI_GGX"),
        ("metallic_beckmann_conductor", "distribution", "BECKMANN"),
        ("metallic_ggx_conductor", "distribution", "GGX"),
        ("metallic_multi_ggx_conductor", "distribution", "MULTI_GGX"),
        ("metallic_beckmann_f82", "fresnel_type", "F82"),
        ("metallic_ggx_f82", "fresnel_type", "F82"),
        ("metallic_multi_ggx_f82", "fresnel_type", "F82"),
        (
            "metallic_beckmann_conductor",
            "fresnel_type",
            "PHYSICAL_CONDUCTOR",
        ),
        (
            "metallic_ggx_conductor",
            "fresnel_type",
            "PHYSICAL_CONDUCTOR",
        ),
        (
            "metallic_multi_ggx_conductor",
            "fresnel_type",
            "PHYSICAL_CONDUCTOR",
        ),
        ("hair_reflection", "component", "Reflection"),
        ("hair_transmission", "component", "Transmission"),
        ("principled_hair_chiang_reflectance", "model", "CHIANG"),
        ("principled_hair_chiang_melanin", "model", "CHIANG"),
        ("principled_hair_chiang_absorption", "model", "CHIANG"),
        ("principled_hair_huang_reflectance", "model", "HUANG"),
        ("principled_hair_huang_melanin", "model", "HUANG"),
        ("principled_hair_huang_absorption", "model", "HUANG"),
        (
            "principled_hair_chiang_reflectance",
            "parametrization",
            "COLOR",
        ),
        (
            "principled_hair_chiang_melanin",
            "parametrization",
            "MELANIN",
        ),
        (
            "principled_hair_chiang_absorption",
            "parametrization",
            "ABSORPTION",
        ),
        (
            "principled_hair_huang_reflectance",
            "parametrization",
            "COLOR",
        ),
        (
            "principled_hair_huang_melanin",
            "parametrization",
            "MELANIN",
        ),
        (
            "principled_hair_huang_absorption",
            "parametrization",
            "ABSORPTION",
        ),
    ];
    for (name, property, value) in expected {
        if let Some(case) = capture.cases.iter().find(|case| case.name == name)
            && case.mode[property] != value
        {
            return Err(format!(
                "{name} {property}: expected {value}, found {}",
                case.mode[property]
            )
            .into());
        }
    }
    if let Some(portal) = capture.cases.iter().find(|case| case.name == "ray_portal")
        && (portal.inputs["Position"] != json!([0.0, 0.0, 2.0])
            || portal.inputs["Direction"] != json!([0.684, 0.0, -0.73]))
    {
        return Err(format!("Ray Portal must capture the non-default position/direction redirection inputs, found {}", portal.inputs).into());
    }
    Ok(())
}

fn check_capture(capture: &Capture, engine: &str, samples: u32, names: &[&str]) -> TestResult {
    if capture.blender_version != "5.2.2 LTS" {
        return Err(format!(
            "fixture was generated by Blender {}, not Blender 5.2.2 LTS",
            capture.blender_version
        )
        .into());
    }
    if capture.engine != engine || capture.resolution != [SIZE, SIZE] || capture.samples != samples
    {
        return Err(format!(
            "unexpected {engine} capture metadata: engine={}, resolution={:?}, samples={}",
            capture.engine, capture.resolution, capture.samples
        )
        .into());
    }
    if capture.seed != 17 {
        return Err(format!("fixture seed is {}, not 17", capture.seed).into());
    }
    let actual = capture
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect::<BTreeSet<_>>();
    if capture.cases.len() != names.len() {
        return Err(format!(
            "{engine} capture has {} cases, expected {}",
            capture.cases.len(),
            names.len()
        )
        .into());
    }
    let expected = names.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "{engine} case coverage differs: expected {expected:?}, found {actual:?}"
        )
        .into());
    }
    if capture.cases.iter().any(|case| case.probe[2] != 1) {
        return Err("the capture must hold a 3x3 patch for each BSDF".into());
    }
    Ok(())
}

fn configure_render(project: &Path, root: &Path, seed: u32, samples: u32) -> TestResult {
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
                "max_bounces": 8
            }
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok(())
}

fn render_frame(project: &Path, root: &Path, name: &str, engine: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", engine, "--format", "exr", "--out"])
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
            (f64::from(pixel[0]) + 0.5) / f64::from(SIZE),
            (f64::from(pixel[1]) + 0.5) / f64::from(SIZE),
        ],
        false,
    )
}
fn patch_mean(frame: &ImageData, probe: [u32; 3]) -> [f64; 4] {
    assert_eq!([frame.width, frame.height], [SIZE, SIZE]);
    let [x, y, radius] = probe;
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

fn capture_seed_noise(path: &Path) -> TestResult<std::collections::HashMap<String, f64>> {
    let json: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let cases = json["cases"]
        .as_array()
        .ok_or("path seed-noise cases are missing")?;
    let mut result = std::collections::HashMap::with_capacity(cases.len());
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("path seed-noise case is unnamed")?;
        let noise = case["mean_absolute_error"]
            .as_f64()
            .ok_or("path seed-noise value is missing")?;
        result.insert(name.to_owned(), noise);
    }
    Ok(result)
}

fn rgba_mae(first: &[f64; 4], second: &[f64; 4]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 4.0
}

fn check_holdout_alpha(
    actual: &[[f64; 4]],
    names: &[&str],
    reference: &std::collections::HashMap<&str, &CaptureCase>,
    neighbor_actual: Option<[f64; 4]>,
) -> TestResult {
    let index = names
        .iter()
        .position(|name| *name == "holdout")
        .ok_or("path holdout probe is missing")?;
    let alpha = actual[index][3];
    let expected_alpha = reference["holdout"].linear_rgba[3];
    if alpha.abs() > 1.0e-6 || expected_alpha.abs() > 1.0e-6 {
        return Err(format!("holdout must leave a transparent 3x3 hole: Potter alpha {alpha}, Blender alpha {expected_alpha}").into());
    }
    let expected_neighbor = reference["holdout"]
        .neighbor_rgba
        .ok_or("Blender holdout backplate capture is missing")?;
    let neighbor = neighbor_actual.ok_or("Potter holdout backplate sample is missing")?;
    if expected_neighbor[3] < 0.999 || neighbor[3] < 0.999 {
        return Err(format!("holdout compositing must preserve the opaque backplate outside its hole: Potter {:?}, Blender {:?}", neighbor[3], expected_neighbor[3]).into());
    }
    for channel in 0..4 {
        let tolerance = f32_ulp_tolerance(expected_neighbor[channel], 4.0);
        if (neighbor[channel] - expected_neighbor[channel]).abs() > tolerance {
            return Err(format!("holdout backplate channel {channel}: Potter {} differs from Blender {} by more than {tolerance}", neighbor[channel], expected_neighbor[channel]).into());
        }
    }
    Ok(())
}
