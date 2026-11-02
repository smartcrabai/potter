//! Native standalone shader BSDF path/realtime parity against committed Blender 5.2.2 linear captures.
//!
//! `fixtures/native_oracle/shader_bsdf_wave12/generate.py` builds the committed scenes and captures their
//! untransformed RGBA EXRs. The path scene covers every supported standalone node, including all distribution,
//! component, falloff, Fresnel, hair model and parametrization choices. The original EEVEE capture covers nodes with
//! dedicated realtime closures; separate previews cover nodes whose GPU shaders compile to Diffuse or Transparency.
//! All tests import source `.blend` files through Potter and exercise the real `pot render` CLI. Stochastic path RGB
//! results compare paired-seed means using per-channel factor-2 noise bounds and a source-derived Cycles float32
//! accumulation/output floor (256 samples for path captures, 4096 for the same-ray skin crop; at most 17 RGB film
//! additions per sample). Alpha uses its same-channel seed-noise bound and output precision only; the RGB
//! accumulation floor does not apply.
//! The original area-lit Ashikhmin capture remains diagnostic; its direct-SUN D/G control uses the established
//! area-noise bound. Area-lit Toon Diffuse reports seed-paired Blender/Potter channel errors and measured
//! Monte Carlo reference spread without gating that noisy comparison; this is not a model-bias claim. A
//! separate deterministic point-SUN test gates plateau, falloff, and outside response with the shared Cycles
//! float32 accumulation and output-precision floor, with no Monte Carlo allowance.
//! Realtime EEVEE comparisons aggregate per-pixel RGBA16F spacing over each captured patch; the
//! float32 EXR wrapper does not imply float32 film precision. The Ray Portal case uses nonuniform
//! backplates and nonzero Position/Direction, distinguishing redirection from an ordinary closure.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeSet, fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE, TestResult, apply_operations, checked,
    cycles_film_precision_floor_for_pixels, import_blend, pot, read_scene,
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
const CYCLES_MAX_BOUNCES: u32 = 8;
const CYCLES_TRANSPARENT_MAX_BOUNCES: u32 = 8;
// The holdout neighbor is an emission-only backdrop hit, so each sample has one terminal RGB film write.
const HOLDOUT_BACKPLATE_WRITES_PER_SAMPLE: u32 = 1;
const CYCLES_CAPTURE: &str = include_str!("fixtures/native_oracle/shader_bsdf_wave12/path.json");
const EEVEE_CAPTURE: &str = include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee.json");
const EEVEE_SSS_TRANSMISSION_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee_sss_transmission.json");
const EEVEE_SSS_TRANSMISSION_SIZE: [u32; 2] = [64, 64];
const EEVEE_REFRACTION_LAYERS_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee_refraction_layers.json");
const EEVEE_REFRACTION_LAYERS_SIZE: [u32; 2] = [128, 64];

const EEVEE_SHEEN_ASHIKHMIN_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee_sheen_ashikhmin.json");
const EEVEE_SHEEN_MICROFIBER_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee_sheen_microfiber.json");
const EEVEE_PREVIEW_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/eevee_preview.json");
const EEVEE_PREVIEW_CASES: &[&str] = &[
    "toon_diffuse_eevee",
    "toon_glossy_eevee",
    "hair_eevee",
    "principled_hair_eevee",
    "ray_portal_eevee",
];
const ASHIKHMIN_SUN_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/sheen_ashikhmin_sun.json");
const TOON_SUN_CAPTURE: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/toon_diffuse_sun.json");
const TOON_SUN_RESOLUTION: [u32; 2] = [4, 4];
const TOON_SUN_SAMPLES: u32 = 32;
const TOON_SUN_SEEDS: [u32; 2] = [31, 43];
const SKIN_CROP_CAPTURE_SEED31: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/path_skin_crop_seed31.json");
const SKIN_CROP_CAPTURE_SEED43: &str =
    include_str!("fixtures/native_oracle/shader_bsdf_wave12/path_skin_crop_seed43.json");
const SKIN_CROP_SIZE: [u32; 2] = [4, 4];
const SKIN_CROP_SAMPLES: u32 = 4096;
const SKIN_CROP_SEEDS: [u32; 2] = [31, 43];

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
    #[serde(default)]
    blender_build_hash: Option<String>,
    cases: Vec<CaptureCase>,
}

#[derive(Deserialize)]
struct CaptureCase {
    name: String,
    probe: [u32; 3],
    #[serde(default)]
    corner_probe: Option<[u32; 2]>,
    #[serde(default)]
    corner_rgba: Option<[f64; 4]>,
    #[serde(rename = "linear_rgba")]
    associated_rgba: [f64; 4],
    #[serde(default, rename = "neighbor_rgba")]
    associated_neighbor_rgba: Option<[f64; 4]>,
    mode: serde_json::Value,
    inputs: serde_json::Value,
    #[serde(default)]
    normal_linked: bool,
}

#[derive(Deserialize)]
struct EeveeRefractionLayersCapture {
    blender_version: String,
    blender_build_hash: String,
    blender_binary: String,
    blender_binary_sha256: String,
    engine: String,
    resolution: [u32; 2],
    samples: u32,
    seed: u32,
    linear_exr: String,
    pixel_format: String,
    raytracing: EeveeRefractionRaytracing,
    cases: Vec<EeveeRefractionLayersCase>,
}

#[derive(Deserialize)]
struct EeveeRefractionRaytracing {
    enabled: bool,
    method: String,
    resolution_scale: String,
    screen_trace_quality: f64,
    screen_trace_thickness: f64,
    trace_max_roughness: f64,
}

#[derive(Deserialize)]
struct EeveeRefractionLayersCase {
    name: String,
    probe: [u32; 3],
    #[serde(rename = "linear_rgba")]
    associated_rgba: [f64; 4],
}

fn eevee_refraction_layer_case<'a>(
    capture: &'a EeveeRefractionLayersCapture,
    name: &str,
) -> TestResult<&'a EeveeRefractionLayersCase> {
    capture
        .cases
        .iter()
        .find(|case| case.name == name)
        .ok_or_else(|| format!("Blender EEVEE refraction-layer case {name} is missing").into())
}

fn imported_material_named<'a>(
    scene: &'a serde_json::Value,
    name: &str,
) -> TestResult<(&'a str, &'a serde_json::Value)> {
    let materials = scene["materials"]
        .as_object()
        .ok_or("imported Blender materials are missing")?;
    let (id, material) = materials
        .iter()
        .find(|(_, material)| material["name"].as_str() == Some(name))
        .ok_or_else(|| format!("imported Blender material {name} is missing"))?;
    Ok((id, material))
}

#[derive(Deserialize)]
struct ToonSunCapture {
    blender_version: String,
    blender_build_hash: String,
    engine: String,
    resolution: [u32; 2],
    samples: u32,
    seeds: [u32; 2],
    pixel_filter: String,
    filter_width: f64,
    view_transform: String,
    cases: Vec<ToonSunCase>,
}

#[derive(Deserialize)]
struct ToonSunCase {
    name: String,
    regime: String,
    theta_radians: f64,
    theta_degrees: f64,
    probe: [u32; 3],
    blend: String,
    seed_captures: Vec<ToonSunSeedCapture>,
}

#[derive(Deserialize)]
struct ToonSunSeedCapture {
    seed: u32,
    exr: String,
    linear_rgba: [f64; 4],
    field_range_rgba: [f64; 4],
}

fn assert_zero_associated_rgba(label: &str, rgba: [f64; 4]) -> TestResult {
    if rgba.iter().any(|channel| channel.abs() > 1.0e-6) {
        return Err(format!("{label}: expected zero associated RGBA, got {rgba:?}").into());
    }
    Ok(())
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
    let blender_seed17 = load_image_file(
        &project,
        root.path(),
        "blender_seed17",
        &fixture_dir.join("path.exr"),
    )?;
    let blender_seed29 = load_image_file(
        &project,
        root.path(),
        "blender_seed29",
        &fixture_dir.join("path_seed29.exr"),
    )?;
    let mut actuals = Vec::with_capacity(PATH_SEEDS.len());
    let mut potter_frames = Vec::with_capacity(PATH_SEEDS.len());
    for seed in PATH_SEEDS {
        configure_render(&project, root.path(), seed, CYCLES_SAMPLES)?;
        let frame = render_frame(&project, root.path(), &format!("path_seed_{seed}"), "path")?;
        actuals.push(
            PATH_CASES
                .iter()
                .map(|name| patch_mean(&frame, reference[name].probe))
                .collect::<Vec<_>>(),
        );
        potter_frames.push(frame);
    }
    let blender_noise = capture_seed_noise(&fixture_dir.join("path_noise.json"))?;
    let blender_seed29_values = PATH_CASES
        .iter()
        .map(|name| patch_mean(&blender_seed29, reference[name].probe))
        .collect::<Vec<_>>();
    let blender_seed17_values = PATH_CASES
        .iter()
        .map(|name| patch_mean(&blender_seed17, reference[name].probe))
        .collect::<Vec<_>>();
    let ashikhmin_index = PATH_CASES
        .iter()
        .position(|name| *name == "sheen_ashikhmin")
        .ok_or("path Ashikhmin probe is missing")?;
    let ashikhmin_capture = reference["sheen_ashikhmin"].associated_rgba;
    let area_noise = rgba_mae(&actuals[0][ashikhmin_index], &actuals[1][ashikhmin_index]);
    let source_noise = blender_noise["sheen_ashikhmin"];
    let area_tolerance = 2.0 * (area_noise + source_noise);
    eprintln!(
        "Ashikhmin area-light diagnostic (not an acceptance assertion): Blender seeds17/29 {:?}/{:?}, pair mean {:?}; Potter seeds17/29 {:?}/{:?}, pair mean {:?}; Potter noise {area_noise}, Blender noise {source_noise}",
        ashikhmin_capture,
        blender_seed29_values[ashikhmin_index],
        mean_rgba(&ashikhmin_capture, &blender_seed29_values[ashikhmin_index],),
        actuals[0][ashikhmin_index],
        actuals[1][ashikhmin_index],
        mean_rgba(&actuals[0][ashikhmin_index], &actuals[1][ashikhmin_index]),
    );

    let sun_capture: Capture = serde_json::from_str(ASHIKHMIN_SUN_CAPTURE)?;
    check_capture(
        &sun_capture,
        "CYCLES",
        CYCLES_SAMPLES,
        &["sheen_ashikhmin_sun"],
    )?;
    let sun_reference = &sun_capture.cases[0];
    let sun_associated = sun_reference.associated_rgba;
    assert_eq!(sun_reference.mode["distribution"], json!("ASHIKHMIN"));
    assert_eq!(
        sun_reference.inputs["Color"],
        json!([0.72, 0.31, 0.12, 1.0])
    );
    assert_eq!(sun_reference.inputs["Roughness"], json!(0.43));
    assert_eq!(sun_reference.inputs["Normal"], json!([0.06, 0.02, 0.998]));
    let (sun_project, _) = import_blend(
        &blender,
        root.path(),
        "sheen_sun_project",
        &fixture_dir.join("sheen_ashikhmin_sun.blend"),
    )?;
    let mut sun_actuals = Vec::with_capacity(PATH_SEEDS.len());
    for seed in PATH_SEEDS {
        configure_render(&sun_project, root.path(), seed, CYCLES_SAMPLES)?;
        let frame = render_frame(
            &sun_project,
            root.path(),
            &format!("sheen_sun_seed_{seed}"),
            "path",
        )?;
        let actual = patch_mean(&frame, sun_reference.probe);
        for channel in 0..4 {
            let expected = sun_associated[channel];
            assert!(
                (actual[channel] - expected).abs() <= area_tolerance,
                "Ashikhmin direct SUN seed {seed} channel {channel}: Potter {actual:?} differs from Blender {expected} by more than the unchanged area-light noise bound {area_tolerance}"
            );
        }
        sun_actuals.push(actual);
    }
    eprintln!(
        "Ashikhmin direct SUN diagnostic: Blender seed17 {:?}, Potter seeds17/29 {:?}/{:?}, direct seed noise {}",
        sun_associated,
        sun_actuals[0],
        sun_actuals[1],
        rgba_mae(&sun_actuals[0], &sun_actuals[1]),
    );
    let film_additions = CYCLES_SAMPLES * PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE;
    let gamma_n = native_oracle::cycles_gamma_n(film_additions);
    eprintln!(
        "Cycles RGB float32 film precision budget: {CYCLES_SAMPLES} samples × {PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE} maximum additions per sample = {film_additions} additions/pixel, gamma_n={gamma_n}; alpha uses output precision only"
    );

    let mut parity_failures = Vec::new();
    for (case_index, name) in PATH_CASES.iter().enumerate() {
        let actual = mean_rgba(&actuals[0][case_index], &actuals[1][case_index]);
        let expected = mean_rgba(
            &reference[name].associated_rgba,
            &blender_seed29_values[case_index],
        );
        let potter_noise_rgba = rgba_mae(&actuals[0][case_index], &actuals[1][case_index]);
        let blender_noise_rgba = blender_noise[*name];
        if *name == "sheen_ashikhmin" {
            continue;
        }
        if *name == "toon_diffuse" {
            let blender_values = [
                blender_seed17_values[case_index],
                blender_seed29_values[case_index],
            ];
            let potter_values = [actuals[0][case_index], actuals[1][case_index]];
            let seed_paired_errors: [[f64; 4]; 2] = std::array::from_fn(|seed_index| {
                std::array::from_fn(|channel| {
                    (potter_values[seed_index][channel] - blender_values[seed_index][channel]).abs()
                })
            });
            let potter_seed_spread: [f64; 4] = std::array::from_fn(|channel| {
                (potter_values[0][channel] - potter_values[1][channel]).abs()
            });
            let blender_seed_spread: [f64; 4] = std::array::from_fn(|channel| {
                (blender_values[0][channel] - blender_values[1][channel]).abs()
            });
            eprintln!(
                "Toon Diffuse area-light stochastic diagnostic (not an acceptance assertion; measured Monte Carlo spread): Blender seeds17/29 {:?}/{:?}, Potter seeds17/29 {:?}/{:?}; seed-paired per-channel absolute errors (Potter minus Blender) {seed_paired_errors:?}; per-channel seed spread Potter={potter_seed_spread:?}, Blender={blender_seed_spread:?}",
                blender_values[0], blender_values[1], potter_values[0], potter_values[1],
            );
            continue;
        }
        if *name == "subsurface_random_walk_skin" {
            eprintln!(
                "Original 192x192 random-walk-skin diagnostic (not an acceptance assertion): Blender path.exr/path_seed29.exr seed17/29 {:?}/{:?}, pair mean {:?}, captured seed noise MAE {blender_noise_rgba}; Potter seed17/29 {:?}/{:?}, pair mean {:?}, measured seed noise RGBA MAE {potter_noise_rgba}",
                reference[name].associated_rgba,
                blender_seed29_values[case_index],
                expected,
                actuals[0][case_index],
                actuals[1][case_index],
                actual,
            );
            continue;
        }
        let mut potter_noise_by_channel = [0.0; 4];
        let mut blender_noise_by_channel = [0.0; 4];
        let mut noise_bounds = [0.0; 4];
        let mut precision_floors = [0.0; 4];
        for channel in 0..4 {
            potter_noise_by_channel[channel] =
                (actuals[0][case_index][channel] - actuals[1][case_index][channel]).abs();
            blender_noise_by_channel[channel] = (blender_seed17_values[case_index][channel]
                - blender_seed29_values[case_index][channel])
                .abs();
            noise_bounds[channel] =
                2.0 * (potter_noise_by_channel[channel] + blender_noise_by_channel[channel]);
            precision_floors[channel] = cycles_film_precision_floor(
                [&potter_frames[0], &potter_frames[1]],
                [&blender_seed17, &blender_seed29],
                reference[name].probe,
                channel,
                gamma_n,
            );
            let tolerance = noise_bounds[channel].max(precision_floors[channel]);
            let error = (actual[channel] - expected[channel]).abs();
            if error > tolerance {
                parity_failures.push(format!(
                    "{name} channel {channel}: error {error} exceeds max(noise {}, RGB accumulation/output precision {}) = {tolerance}; Potter {actual:?}, Blender {expected:?}",
                    noise_bounds[channel],
                    precision_floors[channel],
                ));
            }
        }
        eprintln!(
            "{name} path pair means: Potter {actual:?}, Blender {expected:?}; historical RGBA-MAE noise Potter={potter_noise_rgba}, Blender={blender_noise_rgba}; per-channel Potter noise {potter_noise_by_channel:?}, Blender noise {blender_noise_by_channel:?}, bounds {noise_bounds:?}, Cycles f32 floors {precision_floors:?}"
        );
    }
    if let Err(error) = check_holdout_associated_rgba(
        &actuals[0],
        PATH_CASES,
        &reference,
        [&potter_frames[0], &potter_frames[1]],
        [&blender_seed17, &blender_seed29],
    ) {
        parity_failures.push(format!("holdout assertion: {error}"));
    }
    if !parity_failures.is_empty() {
        return Err(format!(
            "path BSDF per-channel parity failures:\n{}",
            parity_failures.join("\n")
        )
        .into());
    }
    Ok(())
}

#[test]
fn toon_diffuse_direct_sun_matches_deterministic_blender_captures() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping deterministic Toon Diffuse SUN parity; no executable Blender was found"
        );
        return Ok(());
    };
    let capture: ToonSunCapture = serde_json::from_str(TOON_SUN_CAPTURE)?;
    assert_eq!(capture.blender_version, "5.2.2 LTS");
    assert_eq!(capture.blender_build_hash, "d13f752e3b9c");
    assert_eq!(capture.engine, "CYCLES");
    assert_eq!(capture.resolution, TOON_SUN_RESOLUTION);
    assert_eq!(capture.samples, TOON_SUN_SAMPLES);
    assert_eq!(capture.seeds, TOON_SUN_SEEDS);
    assert_eq!(capture.pixel_filter, "BOX");
    assert_eq!(capture.filter_width, 1.0);
    assert_eq!(capture.view_transform, "Standard");
    assert_eq!(capture.cases.len(), 3);

    let expected_cases = [
        ("toon_diffuse_sun_plateau", "plateau", 0.0),
        (
            "toon_diffuse_sun_mid_smooth",
            "mid_smooth",
            (0.38 + 0.17 * 0.5) * std::f64::consts::PI / 2.0,
        ),
        (
            "toon_diffuse_sun_outside",
            "outside",
            (0.38 + 0.17 + 0.1) * std::f64::consts::PI / 2.0,
        ),
    ];

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let mut parity_failures = Vec::new();
    for (case, (name, regime, expected_theta)) in capture.cases.iter().zip(expected_cases) {
        assert_eq!(case.name, name);
        assert_eq!(case.regime, regime);
        assert!((case.theta_radians - expected_theta).abs() <= 1e-15);
        assert!((case.theta_degrees - expected_theta.to_degrees()).abs() <= 1e-12);
        assert_eq!(case.probe, [1, 1, 1]);
        assert_eq!(case.seed_captures.len(), TOON_SUN_SEEDS.len());

        let (project, _) = import_blend(
            &blender,
            root.path(),
            &format!("toon_sun_{}", case.regime),
            &fixture_dir.join(&case.blend),
        )?;

        let mut source_frames = Vec::with_capacity(TOON_SUN_SEEDS.len());
        let mut potter_frames = Vec::with_capacity(TOON_SUN_SEEDS.len());
        let mut source_values = Vec::with_capacity(TOON_SUN_SEEDS.len());
        let mut potter_values = Vec::with_capacity(TOON_SUN_SEEDS.len());
        for (seed, source) in TOON_SUN_SEEDS.into_iter().zip(&case.seed_captures) {
            assert_eq!(source.seed, seed);
            assert_eq!(source.field_range_rgba, [0.0; 4]);
            let source_frame = load_image_file(
                &project,
                root.path(),
                &format!("blender_{}_seed{seed}", case.regime),
                &fixture_dir.join(&source.exr),
            )?;
            assert_eq!(
                [source_frame.width, source_frame.height],
                TOON_SUN_RESOLUTION
            );
            let expected = patch_mean(&source_frame, case.probe);
            assert_eq!(expected, source.linear_rgba);

            configure_render_size(
                &project,
                root.path(),
                seed,
                TOON_SUN_SAMPLES,
                TOON_SUN_RESOLUTION,
            )?;
            let frame = render_frame(
                &project,
                root.path(),
                &format!("toon_sun_{}_seed{seed}", case.regime),
                "path",
            )?;
            assert_eq!([frame.width, frame.height], TOON_SUN_RESOLUTION);
            let actual = patch_mean(&frame, case.probe);
            if actual[3].to_bits() != 1.0_f64.to_bits() {
                parity_failures.push(format!(
                    "{regime}, seed {seed}: opaque SUN control alpha {} is not exactly 1",
                    actual[3]
                ));
            }
            if regime == "outside"
                && actual[..3]
                    .iter()
                    .any(|value| value.abs().to_bits() != 0.0_f64.to_bits())
            {
                parity_failures.push(format!(
                    "{regime}, seed {seed}: outside-control RGB {:?} is not exactly zero",
                    &actual[..3]
                ));
            }
            source_values.push(expected);
            potter_values.push(actual);
            source_frames.push(source_frame);
            potter_frames.push(frame);
        }
        if source_values[0].map(f64::to_bits) != source_values[1].map(f64::to_bits) {
            parity_failures.push(format!(
                "{regime}: Blender SUN seeds31/43 differ {:?} vs {:?}",
                source_values[0], source_values[1]
            ));
        }
        if potter_values[0].map(f64::to_bits) != potter_values[1].map(f64::to_bits) {
            parity_failures.push(format!(
                "{regime}: Potter SUN seeds31/43 differ {:?} vs {:?}",
                potter_values[0], potter_values[1]
            ));
        }
        let film_additions = TOON_SUN_SAMPLES * PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE;
        let gamma_n = native_oracle::cycles_gamma_n(film_additions);
        let mut precision_floors = [0.0; 4];
        for channel in 0..4 {
            let tolerance = cycles_film_precision_floor(
                [&potter_frames[0], &potter_frames[1]],
                [&source_frames[0], &source_frames[1]],
                case.probe,
                channel,
                gamma_n,
            );
            precision_floors[channel] = tolerance;
            for seed_index in 0..TOON_SUN_SEEDS.len() {
                let error =
                    (potter_values[seed_index][channel] - source_values[seed_index][channel]).abs();
                if error > tolerance {
                    parity_failures.push(format!(
                        "{regime}, seed {}, channel {channel}: error {error} exceeds source-derived RGB gamma_n/output precision bound {tolerance}; Potter {:?}, Blender {:?}",
                        TOON_SUN_SEEDS[seed_index],
                        potter_values[seed_index],
                        source_values[seed_index],
                    ));
                }
            }
        }
        eprintln!(
            "Toon Diffuse direct-SUN {regime} at {:.2}°: Blender seeds31/43 {:?}/{:?}, Potter seeds31/43 {:?}/{:?}; {TOON_SUN_SAMPLES} spp × {PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE} maximum additions = {film_additions}, gamma_n={gamma_n}, per-channel RGB/output floors {precision_floors:?}",
            case.theta_degrees,
            source_values[0],
            source_values[1],
            potter_values[0],
            potter_values[1],
        );
    }
    if !parity_failures.is_empty() {
        return Err(format!(
            "Toon Diffuse direct-SUN parity failures:\n{}",
            parity_failures.join("\n")
        )
        .into());
    }
    Ok(())
}

#[test]
fn path_random_walk_skin_crop_matches_blender_linear_captures() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping native random-walk-skin crop parity; no executable Blender was found");
        return Ok(());
    };
    let captures = [
        serde_json::from_str::<Capture>(SKIN_CROP_CAPTURE_SEED31)?,
        serde_json::from_str::<Capture>(SKIN_CROP_CAPTURE_SEED43)?,
    ];
    let original: Capture = serde_json::from_str(CYCLES_CAPTURE)?;
    let original_skin = original
        .cases
        .iter()
        .find(|case| case.name == "subsurface_random_walk_skin")
        .ok_or("original random-walk-skin capture is missing")?;
    let probe = [1, 1, 1];
    for (capture, seed) in captures.iter().zip(SKIN_CROP_SEEDS) {
        assert_eq!(capture.blender_version, "5.2.2 LTS");
        assert_eq!(capture.engine, "CYCLES");
        assert_eq!(capture.resolution, SKIN_CROP_SIZE);
        assert_eq!(capture.samples, SKIN_CROP_SAMPLES);
        assert_eq!(capture.seed, seed);
        assert_eq!(capture.cases.len(), 1);
        let case = &capture.cases[0];
        assert_eq!(case.name, "subsurface_random_walk_skin");
        assert_eq!(case.probe, probe);
        assert_eq!(case.mode, original_skin.mode);
        assert_eq!(case.inputs, original_skin.inputs);
    }

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "path_skin_crop_project",
        &fixture_dir.join("path_skin_crop.blend"),
    )?;
    let mut blender_frames = Vec::with_capacity(SKIN_CROP_SEEDS.len());
    for seed in SKIN_CROP_SEEDS {
        blender_frames.push(load_image_file(
            &project,
            root.path(),
            &format!("blender_skin_seed_{seed}"),
            &fixture_dir.join(format!("path_skin_crop_seed{seed}.exr")),
        )?);
    }

    let mut potter_frames = Vec::with_capacity(SKIN_CROP_SEEDS.len());
    for seed in SKIN_CROP_SEEDS {
        configure_render_size(
            &project,
            root.path(),
            seed,
            SKIN_CROP_SAMPLES,
            SKIN_CROP_SIZE,
        )?;
        potter_frames.push(render_frame(
            &project,
            root.path(),
            &format!("path_skin_crop_seed_{seed}"),
            "path",
        )?);
    }
    let blender_values = blender_frames
        .iter()
        .map(|frame| patch_mean(frame, probe))
        .collect::<Vec<_>>();
    let potter_values = potter_frames
        .iter()
        .map(|frame| patch_mean(frame, probe))
        .collect::<Vec<_>>();
    let expected = mean_rgba(&blender_values[0], &blender_values[1]);
    let actual = mean_rgba(&potter_values[0], &potter_values[1]);
    let gamma_n =
        native_oracle::cycles_gamma_n(SKIN_CROP_SAMPLES * PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE);
    let mut noise_bounds = [0.0; 4];
    let mut precision_floors = [0.0; 4];
    for channel in 0..4 {
        noise_bounds[channel] = 2.0
            * ((potter_values[0][channel] - potter_values[1][channel]).abs()
                + (blender_values[0][channel] - blender_values[1][channel]).abs());
        precision_floors[channel] = cycles_film_precision_floor(
            [&potter_frames[0], &potter_frames[1]],
            [&blender_frames[0], &blender_frames[1]],
            probe,
            channel,
            gamma_n,
        );
        let tolerance = noise_bounds[channel].max(precision_floors[channel]);
        assert!(
            (actual[channel] - expected[channel]).abs() <= tolerance,
            "subsurface_random_walk_skin 4096-spp crop pair mean channel {channel}: Potter {actual:?} differs from Blender {expected:?} by more than max(channel noise bound {}, source-derived Cycles f32 floor {}) = {tolerance}",
            noise_bounds[channel],
            precision_floors[channel],
        );
    }
    eprintln!(
        "4096-spp same-ray random-walk-skin crop pair means: Potter {actual:?}, Blender {expected:?}; per-channel seed-noise bounds {noise_bounds:?}, RGB Cycles gamma_n floor {gamma_n} plus f32 output rounding, alpha output-only floors {precision_floors:?}"
    );
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
    let blender_eevee = load_image_file(
        &project,
        root.path(),
        "blender_eevee",
        &fixture_dir.join("eevee.exr"),
    )?;
    configure_render(&project, root.path(), 17, EEVEE_SAMPLES)?;
    let frame = render_frame(&project, root.path(), "eevee", "realtime")?;
    let reference = capture
        .cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect::<std::collections::HashMap<_, _>>();
    let mut parity_failures = Vec::new();
    for name in EEVEE_CASES {
        let case = reference[name];
        let actual = patch_mean(&frame, case.probe);
        let expected = case.associated_rgba;
        for channel in 0..4 {
            let tolerance = patch_ulp_tolerance(&blender_eevee, case.probe, channel, |value| {
                f16_ulp_tolerance(value, 4.0)
            });
            let error = (actual[channel] - expected[channel]).abs();
            if error > tolerance {
                parity_failures.push(format!(
                    "{name} EEVEE channel {channel}: Potter {} differs from Blender associated capture {} by more than the averaged four-per-pixel RGBA16F spacing {tolerance}",
                    actual[channel],
                    expected[channel]
                ));
            }
        }
    }
    let holdout = reference["holdout"];
    let [holdout_x, holdout_y, _] = holdout.probe;
    let holdout_actual = patch_mean(&frame, holdout.probe);
    let holdout_expected = holdout.associated_rgba;
    assert_zero_associated_rgba("Potter EEVEE Holdout", holdout_actual)?;
    assert_zero_associated_rgba("Blender EEVEE Holdout capture", holdout_expected)?;
    let neighbor = sample_pixel(&frame, [holdout_x + 13, holdout_y]);
    let expected_neighbor_associated = holdout
        .associated_neighbor_rgba
        .ok_or("Blender EEVEE holdout backplate capture is missing")?;
    let expected_neighbor = expected_neighbor_associated;
    if neighbor[3] < 0.999 || expected_neighbor[3] < 0.999 {
        return Err(format!("EEVEE holdout must preserve the opaque backplate outside its hole: Potter alpha {}, Blender alpha {}", neighbor[3], expected_neighbor[3]).into());
    }
    let blender_neighbor = sample_pixel(&blender_eevee, [holdout_x + 13, holdout_y]);
    for channel in 0..4 {
        let tolerance = f16_ulp_tolerance(blender_neighbor[channel], 4.0);
        if (neighbor[channel] - expected_neighbor[channel]).abs() > tolerance {
            return Err(format!("EEVEE holdout backplate channel {channel}: Potter {} differs from Blender {} by more than four RGBA16F ULPs {tolerance}", neighbor[channel], expected_neighbor[channel]).into());
        }
    }
    if !parity_failures.is_empty() {
        return Err(format!(
            "EEVEE standalone BSDF per-channel parity failures:\n{}",
            parity_failures.join("\n")
        )
        .into());
    }
    Ok(())
}

#[test]
fn realtime_eevee_screen_refraction_layers_match_source() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping EEVEE refraction-layer parity; no executable Blender was found");
        return Ok(());
    };
    let capture: EeveeRefractionLayersCapture =
        serde_json::from_str(EEVEE_REFRACTION_LAYERS_CAPTURE)?;
    if capture.blender_version != "5.2.2 LTS"
        || capture.blender_build_hash != "d13f752e3b9c"
        || capture.engine != "BLENDER_EEVEE"
        || capture.resolution != EEVEE_REFRACTION_LAYERS_SIZE
        || capture.samples != EEVEE_SAMPLES
        || capture.seed != 17
        || capture.linear_exr != "eevee_refraction_layers.exr"
        || capture.pixel_format != "32-bit RGBA OpenEXR ZIP"
        || capture.blender_binary.is_empty()
        || capture.blender_binary_sha256.len() != 64
        || capture.cases.len() != 8
    {
        return Err("unexpected Blender EEVEE refraction-layer source metadata".into());
    }
    assert!(capture.raytracing.enabled);
    assert_eq!(capture.raytracing.method, "SCREEN");
    assert_eq!(capture.raytracing.resolution_scale, "1");
    assert_eq!(capture.raytracing.screen_trace_quality, 1.0);
    assert!((capture.raytracing.screen_trace_thickness - 0.1).abs() < 1.0e-6);
    assert_eq!(capture.raytracing.trace_max_roughness, 0.5);

    let enabled_emission =
        eevee_refraction_layer_case(&capture, "enabled_glass_over_flagged_emission")?;
    let enabled_principled =
        eevee_refraction_layer_case(&capture, "enabled_glass_over_principled_transmission_zero")?;
    let enabled_sss = eevee_refraction_layer_case(&capture, "enabled_glass_over_resolved_sss")?;
    let disabled = eevee_refraction_layer_case(&capture, "disabled_glass_uses_world")?;
    let visible_emission = eevee_refraction_layer_case(&capture, "flagged_emission_visible")?;
    let visible_principled = eevee_refraction_layer_case(&capture, "flagged_principled_visible")?;
    let visible_sss = eevee_refraction_layer_case(&capture, "opaque_sss_visible")?;
    let red_backplate = eevee_refraction_layer_case(&capture, "red_backplate_visible")?;
    for case in [enabled_emission, enabled_principled] {
        if case.associated_rgba[0] <= case.associated_rgba[1].max(case.associated_rgba[2]) {
            return Err(format!(
                "enabled source refraction must see the red opaque BACK, got {:?}",
                case.associated_rgba
            )
            .into());
        }
    }
    let black_tolerance = f16_ulp_tolerance(0.0, 4.0);
    if disabled.associated_rgba[..3]
        .iter()
        .any(|channel| channel.abs() > black_tolerance)
        || visible_sss.associated_rgba[2] <= visible_sss.associated_rgba[0]
        || enabled_sss.associated_rgba[2] <= enabled_sss.associated_rgba[0]
        || red_backplate.associated_rgba[0]
            <= red_backplate.associated_rgba[1].max(red_backplate.associated_rgba[2])
        || !(0..3).any(|channel| {
            (enabled_emission.associated_rgba[channel] - disabled.associated_rgba[channel]).abs()
                > f16_ulp_tolerance(enabled_emission.associated_rgba[channel], 4.0)
        })
    {
        return Err(
            "Blender source refraction-layer captures do not distinguish the required signals"
                .into(),
        );
    }

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "eevee_refraction_layers_project",
        &fixture_dir.join("eevee_refraction_layers.blend"),
    )?;
    let scene = read_scene(&project)?;
    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let (enabled_id, enabled_material) =
        imported_material_named(&scene, "Enabled screen-refraction glass")?;
    let (disabled_id, disabled_material) =
        imported_material_named(&scene, "Disabled screen-refraction glass")?;
    let (emission_id, emission_material) =
        imported_material_named(&scene, "Flagged emission without transmission")?;
    let (principled_id, principled_material) =
        imported_material_named(&scene, "Flagged Principled with zero transmission")?;
    assert_eq!(enabled_material["use_raytrace_refraction"], true);
    assert_eq!(disabled_material["use_raytrace_refraction"], false);
    assert_eq!(emission_material["use_raytrace_refraction"], true);
    assert_eq!(principled_material["use_raytrace_refraction"], true);
    assert_eq!(principled_material["transmission"], 0.0);

    let updated = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "render.update",
            "target": {"id": active_scene},
            "set": {
                "eevee_use_raytracing": true,
                "eevee_raytracing_method": "screen",
                "eevee_raytracing_resolution_scale": 1,
                "eevee_raytracing_screen_trace_quality": 1.0,
                "eevee_raytracing_screen_trace_thickness": 0.1,
                "eevee_raytracing_trace_max_roughness": 0.5
            }
        }, {
            "op": "material.update",
            "target": {"id": enabled_id},
            "set": {"use_raytrace_refraction": true}
        }, {
            "op": "material.update",
            "target": {"id": disabled_id},
            "set": {"use_raytrace_refraction": false}
        }, {
            "op": "material.update",
            "target": {"id": emission_id},
            "set": {"use_raytrace_refraction": true}
        }, {
            "op": "material.update",
            "target": {"id": principled_id},
            "set": {"use_raytrace_refraction": true}
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");
    configure_render_size(
        &project,
        root.path(),
        capture.seed,
        capture.samples,
        capture.resolution,
    )?;
    let blender_reference = load_image_file(
        &project,
        root.path(),
        "blender_eevee_refraction_layers",
        &fixture_dir.join(&capture.linear_exr),
    )?;
    let frame = render_frame(&project, root.path(), "eevee_refraction_layers", "realtime")?;
    if blender_reference.width != capture.resolution[0]
        || blender_reference.height != capture.resolution[1]
    {
        return Err(
            "Blender source refraction-layer EXR dimensions disagree with its metadata".into(),
        );
    }
    let cases = [
        enabled_emission,
        enabled_principled,
        enabled_sss,
        disabled,
        visible_emission,
        visible_principled,
        visible_sss,
        red_backplate,
    ];
    let mut parity_failures = Vec::new();
    for case in cases {
        let source_patch = patch_mean(&blender_reference, case.probe);
        for (channel, source) in source_patch.iter().copied().enumerate() {
            if (source - case.associated_rgba[channel]).abs() > 1.0e-6 {
                return Err(format!(
                    "Blender EEVEE refraction case {} channel {channel} {} differs from its raw EXR patch {}",
                    case.name, case.associated_rgba[channel], source
                )
                .into());
            }
        }
        let actual = patch_mean(&frame, case.probe);
        for (channel, (actual, source)) in actual.iter().zip(&source_patch).enumerate() {
            let tolerance = patch_ulp_tolerance(&blender_reference, case.probe, channel, |value| {
                f16_ulp_tolerance(value, 4.0)
            });
            if (actual - source).abs() > tolerance {
                parity_failures.push(format!(
                    "{} channel {channel}: Potter {} differs from Blender {} by more than the averaged four-per-pixel RGBA16F spacing {tolerance}",
                    case.name, actual, source
                ));
            }
        }
    }
    if !parity_failures.is_empty() {
        return Err(format!(
            "EEVEE refraction-layer source parity failures:\n{}",
            parity_failures.join("\n")
        )
        .into());
    }
    let unknown = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "material.update",
            "target": {"id": emission_id},
            "set": {"use_raytrace_refraction": null}
        }]),
    )?;
    assert_eq!(unknown["ok"], true, "{unknown}");
    let mut render = pot();
    render
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(root.path().join("unknown_screen_back"))
        .arg("--json");
    let rejected = render.output()?;
    assert!(
        !rejected.status.success(),
        "screen refraction accepted an unknown visible opaque-layer refraction setting: {}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    let rejected: serde_json::Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "render.eevee.ray_tracing.refraction_material"
    );

    let probe = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "render.update",
            "target": {"id": active_scene},
            "set": {"eevee_raytracing_method": "probe"}
        }]),
    )?;
    assert_eq!(probe["ok"], true, "{probe}");
    render_frame(
        &project,
        root.path(),
        "unknown_refraction_probe",
        "realtime",
    )?;
    let no_back = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "render.update",
            "target": {"id": active_scene},
            "set": {"eevee_raytracing_method": "screen"}
        }, {
            "op": "material.update",
            "target": {"id": enabled_id},
            "set": {"use_raytrace_refraction": false}
        }, {
            "op": "material.update",
            "target": {"id": principled_id},
            "set": {"use_raytrace_refraction": false}
        }]),
    )?;
    assert_eq!(no_back["ok"], true, "{no_back}");
    render_frame(
        &project,
        root.path(),
        "unknown_refraction_without_back",
        "realtime",
    )?;

    Ok(())
}

#[test]
fn realtime_eevee_burley_sss_backface_direct_matches_source() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping EEVEE Burley SSS transmission parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(EEVEE_SSS_TRANSMISSION_CAPTURE)?;
    if capture.blender_version != "5.2.2 LTS"
        || capture.blender_build_hash.as_deref() != Some("d13f752e3b9c")
        || capture.engine != "BLENDER_EEVEE"
        || capture.resolution != EEVEE_SSS_TRANSMISSION_SIZE
        || capture.samples != EEVEE_SAMPLES
    {
        return Err("unexpected EEVEE SSS source capture metadata".into());
    }
    let case = capture
        .cases
        .iter()
        .find(|case| case.name == "burley_sss_backface_direct")
        .ok_or("Blender EEVEE Burley SSS transmission capture is missing")?;
    if case.associated_rgba[..3].iter().any(|value| *value <= 0.1) {
        return Err(format!(
            "Blender source backface SSS signal must be nonzero in RGB, got {:?}",
            case.associated_rgba
        )
        .into());
    }
    let corner_probe = case
        .corner_probe
        .ok_or("Blender EEVEE SSS black-corner probe is missing")?;
    let corner_expected = case
        .corner_rgba
        .ok_or("Blender EEVEE SSS black-corner capture is missing")?;
    if corner_expected[..3]
        .iter()
        .any(|value| value.abs() > 1.0e-6)
    {
        return Err(
            format!("Blender EEVEE SSS corner must be black, got {corner_expected:?}").into(),
        );
    }

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "eevee_sss_transmission_project",
        &fixture_dir.join("eevee_sss_transmission.blend"),
    )?;
    let blender_reference = load_image_file(
        &project,
        root.path(),
        "blender_eevee_sss_transmission",
        &fixture_dir.join("eevee_sss_transmission.exr"),
    )?;
    configure_render_size(
        &project,
        root.path(),
        capture.seed,
        capture.samples,
        capture.resolution,
    )?;
    let frame = render_frame(&project, root.path(), "eevee_sss_transmission", "realtime")?;
    let expected_patch = patch_mean(&blender_reference, case.probe);
    let actual_patch = patch_mean(&frame, case.probe);
    for channel in 0..4 {
        if (expected_patch[channel] - case.associated_rgba[channel]).abs() > 1.0e-6 {
            return Err(format!(
                "Blender EEVEE SSS metadata channel {channel} {} differs from its raw EXR patch {}",
                case.associated_rgba[channel], expected_patch[channel]
            )
            .into());
        }
        let actual = actual_patch[channel];
        let tolerance = patch_ulp_tolerance(&blender_reference, case.probe, channel, |value| {
            f16_ulp_tolerance(value, 4.0)
        });
        if (actual - expected_patch[channel]).abs() > tolerance {
            return Err(format!(
                "EEVEE Burley SSS backface-direct channel {channel}: Potter {actual} differs from Blender {} by more than the averaged four-per-pixel RGBA16F spacing {tolerance}",
                expected_patch[channel]
            )
            .into());
        }
    }
    let expected_corner = sample_pixel(&blender_reference, corner_probe);
    if (0..4).any(|channel| (expected_corner[channel] - corner_expected[channel]).abs() > 1.0e-6) {
        return Err(format!(
            "Blender EEVEE SSS corner metadata {corner_expected:?} differs from raw EXR {expected_corner:?}"
        )
        .into());
    }
    let actual_corner = sample_pixel(&frame, corner_probe);
    let corner_tolerance = patch_ulp_tolerance(
        &blender_reference,
        [corner_probe[0], corner_probe[1], 0],
        0,
        |value| f16_ulp_tolerance(value, 4.0),
    );
    for channel in 0..4 {
        let tolerance = if channel == 0 {
            corner_tolerance
        } else {
            patch_ulp_tolerance(
                &blender_reference,
                [corner_probe[0], corner_probe[1], 0],
                channel,
                |value| f16_ulp_tolerance(value, 4.0),
            )
        };
        if (actual_corner[channel] - expected_corner[channel]).abs() > tolerance {
            return Err(format!(
                "EEVEE Burley SSS black-corner channel {channel}: Potter {} differs from Blender {} by more than four RGBA16F ULPs {tolerance}",
                actual_corner[channel], expected_corner[channel]
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn realtime_eevee_preview_nodes_match_source_closures() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping EEVEE shader preview parity; no executable Blender was found");
        return Ok(());
    };
    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let captures = [
        (
            "ASHIKHMIN",
            "eevee_sheen_ashikhmin",
            EEVEE_SHEEN_ASHIKHMIN_CAPTURE,
        ),
        (
            "MICROFIBER",
            "eevee_sheen_microfiber",
            EEVEE_SHEEN_MICROFIBER_CAPTURE,
        ),
    ];
    let mut blender_values = Vec::with_capacity(captures.len());
    let mut potter_values = Vec::with_capacity(captures.len());
    let mut parity_failures = Vec::new();
    for (distribution, prefix, capture_json) in captures {
        let capture: Capture = serde_json::from_str(capture_json)?;
        check_capture(&capture, "BLENDER_EEVEE", EEVEE_SAMPLES, &["sheen_eevee"])?;
        let case = &capture.cases[0];
        if case.mode["distribution"] != json!(distribution)
            || case.inputs["Color"] != json!([0.72, 0.31, 0.12, 1.0])
            || case.inputs["Normal"] != json!([0.06, 0.02, 0.998])
            || !case.normal_linked
            || case.inputs["Roughness"] != json!(0.43)
        {
            return Err(format!(
                "EEVEE Sheen {distribution} capture lost its full-color/normal source probe: {:?}",
                case.inputs
            )
            .into());
        }
        let (project, _) = import_blend(
            &blender,
            root.path(),
            &format!("{prefix}_project"),
            &fixture_dir.join(format!("{prefix}.blend")),
        )?;
        let blender_image = load_image_file(
            &project,
            root.path(),
            &format!("{prefix}_blender"),
            &fixture_dir.join(format!("{prefix}.exr")),
        )?;
        configure_render(&project, root.path(), 17, EEVEE_SAMPLES)?;
        let frame = render_frame(
            &project,
            root.path(),
            &format!("{prefix}_potter"),
            "realtime",
        )?;
        let actual = patch_mean(&frame, case.probe);
        for (channel, (actual_channel, expected_channel)) in
            actual.iter().zip(case.associated_rgba).enumerate()
        {
            let tolerance = patch_ulp_tolerance(&blender_image, case.probe, channel, |value| {
                f16_ulp_tolerance(value, 4.0)
            });
            if (*actual_channel - expected_channel).abs() > tolerance {
                parity_failures.push(format!(
                    "EEVEE Sheen {distribution} channel {channel}: Potter {actual_channel} differs from Blender {expected_channel} by more than the existing averaged four-per-pixel RGBA16F spacing {tolerance}"
                ));
            }
        }
        blender_values.push(case.associated_rgba);
        potter_values.push(actual);
    }
    if blender_values[0].map(f64::to_bits) != blender_values[1].map(f64::to_bits) {
        parity_failures.push(format!(
            "Blender EEVEE Sheen distributions should share one diffuse GPU closure, but captured means differ: {:?} vs {:?}",
            blender_values[0], blender_values[1]
        ));
    }
    if potter_values[0].map(f64::to_bits) != potter_values[1].map(f64::to_bits) {
        parity_failures.push(format!(
            "Potter EEVEE Sheen distributions should share one diffuse GPU closure, but rendered means differ: {:?} vs {:?}",
            potter_values[0], potter_values[1]
        ));
    }

    let capture: Capture = serde_json::from_str(EEVEE_PREVIEW_CAPTURE)?;
    check_capture(
        &capture,
        "BLENDER_EEVEE",
        EEVEE_SAMPLES,
        EEVEE_PREVIEW_CASES,
    )?;
    let reference = capture
        .cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect::<std::collections::HashMap<_, _>>();
    if reference["toon_diffuse_eevee"].mode["component"] != json!("DIFFUSE")
        || reference["toon_glossy_eevee"].mode["component"] != json!("GLOSSY")
        || reference["hair_eevee"].mode["component"] != json!("Transmission")
        || reference["principled_hair_eevee"].mode["model"] != json!("HUANG")
        || reference["principled_hair_eevee"].mode["parametrization"] != json!("ABSORPTION")
        || reference["toon_diffuse_eevee"].inputs["Color"] != json!([0.72, 0.31, 0.12, 1.0])
        || reference["toon_diffuse_eevee"].inputs["Normal"] != json!([0.06, 0.02, 0.998])
        || reference["hair_eevee"].inputs["Color"] != json!([0.72, 0.31, 0.12, 1.0])
        || !reference["toon_diffuse_eevee"].normal_linked
        || !reference["toon_glossy_eevee"].normal_linked
        || reference["principled_hair_eevee"].inputs["Color"] != json!([0.34, 0.12, 0.07, 1.0])
        || reference["ray_portal_eevee"].inputs["Position"] != json!([0.0, 0.0, 2.0])
        || reference["ray_portal_eevee"].inputs["Direction"] != json!([0.684, 0.0, -0.73])
        || reference["ray_portal_eevee"].inputs["Color"] != json!([0.92, 0.87, 0.73, 1.0])
    {
        return Err("EEVEE preview oracle lost its active Color/Normal inputs or ignored closure properties".into());
    }
    let (project, _) = import_blend(
        &blender,
        root.path(),
        "eevee_preview_project",
        &fixture_dir.join("eevee_preview.blend"),
    )?;
    let blender_image = load_image_file(
        &project,
        root.path(),
        "eevee_preview_blender",
        &fixture_dir.join("eevee_preview.exr"),
    )?;
    configure_render(&project, root.path(), 17, EEVEE_SAMPLES)?;
    let frame = render_frame(&project, root.path(), "eevee_preview_potter", "realtime")?;
    for name in EEVEE_PREVIEW_CASES {
        let case = reference[name];
        let actual = patch_mean(&frame, case.probe);
        for (channel, (actual_channel, expected_channel)) in
            actual.iter().zip(case.associated_rgba).enumerate()
        {
            let tolerance = patch_ulp_tolerance(&blender_image, case.probe, channel, |value| {
                f16_ulp_tolerance(value, 4.0)
            });
            if (*actual_channel - expected_channel).abs() > tolerance {
                parity_failures.push(format!(
                    "{name} EEVEE preview channel {channel}: Potter {actual_channel} differs from Blender {expected_channel} by more than the existing averaged four-per-pixel RGBA16F spacing {tolerance}"
                ));
            }
        }
    }
    if !parity_failures.is_empty() {
        return Err(format!(
            "EEVEE standalone/preview BSDF per-channel parity failures:\n{}",
            parity_failures.join("\n")
        )
        .into());
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
    configure_render_size(project, root, seed, samples, [SIZE, SIZE])
}

fn configure_render_size(
    project: &Path,
    root: &Path,
    seed: u32,
    samples: u32,
    resolution: [u32; 2],
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
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "samples": samples,
                "seed": seed,
                "max_bounces": CYCLES_MAX_BOUNCES,
                "transparent_max_bounces": CYCLES_TRANSPARENT_MAX_BOUNCES
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
fn load_image_file(project: &Path, root: &Path, id: &str, path: &Path) -> TestResult<ImageData> {
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": id, "path": path.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image = serde_json::from_value(read_scene(project)?["images"][id].clone())?;
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
    assert!(frame.width > 0 && frame.height > 0);
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
fn mean_rgba(first: &[f64; 4], second: &[f64; 4]) -> [f64; 4] {
    std::array::from_fn(|channel| f64::midpoint(first[channel], second[channel]))
}
fn patch_ulp_tolerance(
    frame: &ImageData,
    probe: [u32; 3],
    channel: usize,
    tolerance: impl Fn(f64) -> f64,
) -> f64 {
    let [x, y, radius] = probe;
    let mut total = 0.0;
    let mut count = 0.0;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            total += tolerance(sample_pixel(frame, [px, py])[channel]);
            count += 1.0;
        }
    }
    total / count
}
fn cycles_film_precision_floor(
    potter_frames: [&ImageData; 2],
    blender_frames: [&ImageData; 2],
    probe: [u32; 3],
    channel: usize,
    gamma_n: f64,
) -> f64 {
    let [x, y, radius] = probe;
    let pixel_groups = (y - radius..=y + radius).flat_map(|py| {
        (x - radius..=x + radius).map(move |px| {
            [
                sample_pixel(potter_frames[0], [px, py]),
                sample_pixel(potter_frames[1], [px, py]),
                sample_pixel(blender_frames[0], [px, py]),
                sample_pixel(blender_frames[1], [px, py]),
            ]
        })
    });
    cycles_film_precision_floor_for_pixels(pixel_groups, channel, gamma_n)
}

fn f16_ulp_tolerance(value: f64, ulps: f64) -> f64 {
    let magnitude = value.abs();
    let ulp = if magnitude < 2.0_f64.powi(-14) {
        2.0_f64.powi(-24)
    } else {
        2.0_f64.powi(magnitude.log2().floor() as i32 - 10)
    };
    ulps * ulp
}

fn rgba_mae(first: &[f64; 4], second: &[f64; 4]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 4.0
}

fn check_holdout_associated_rgba(
    actual: &[[f64; 4]],
    names: &[&str],
    reference: &std::collections::HashMap<&str, &CaptureCase>,
    potter_frames: [&ImageData; 2],
    blender_frames: [&ImageData; 2],
) -> TestResult {
    let index = names
        .iter()
        .position(|name| *name == "holdout")
        .ok_or("path holdout probe is missing")?;
    let holdout = reference["holdout"];
    let holdout_probe = holdout.probe;
    assert_zero_associated_rgba("Blender path Holdout capture", holdout.associated_rgba)?;
    assert_zero_associated_rgba("Potter path Holdout seed17", actual[index])?;
    assert_zero_associated_rgba(
        "Potter path Holdout seed29",
        patch_mean(potter_frames[1], holdout_probe),
    )?;
    assert_zero_associated_rgba(
        "Blender path Holdout seed17",
        patch_mean(blender_frames[0], holdout_probe),
    )?;
    assert_zero_associated_rgba(
        "Blender path Holdout seed29",
        patch_mean(blender_frames[1], holdout_probe),
    )?;
    let expected_neighbor_seed17 = holdout
        .associated_neighbor_rgba
        .ok_or("Blender holdout backplate capture is missing")?;
    let [holdout_x, holdout_y, _] = reference["holdout"].probe;
    let neighbor_position = [holdout_x + 13, holdout_y];
    let expected_neighbor_seed29 = sample_pixel(blender_frames[1], neighbor_position);
    let potter_neighbor_seed17 = sample_pixel(potter_frames[0], neighbor_position);
    let potter_neighbor_seed29 = sample_pixel(potter_frames[1], neighbor_position);
    if [
        expected_neighbor_seed17[3],
        expected_neighbor_seed29[3],
        potter_neighbor_seed17[3],
        potter_neighbor_seed29[3],
    ]
    .into_iter()
    .any(|alpha| alpha < 0.999)
    {
        return Err(format!(
            "holdout compositing must preserve an opaque backplate: Potter {:?}/{:?}, Blender {:?}/{:?}",
            potter_neighbor_seed17[3],
            potter_neighbor_seed29[3],
            expected_neighbor_seed17[3],
            expected_neighbor_seed29[3],
        )
        .into());
    }
    let actual_neighbor = mean_rgba(&potter_neighbor_seed17, &potter_neighbor_seed29);
    let expected_neighbor = mean_rgba(&expected_neighbor_seed17, &expected_neighbor_seed29);
    let gamma_n =
        native_oracle::cycles_gamma_n(CYCLES_SAMPLES * HOLDOUT_BACKPLATE_WRITES_PER_SAMPLE);
    for channel in 0..3 {
        let noise_bound = 2.0
            * ((potter_neighbor_seed17[channel] - potter_neighbor_seed29[channel]).abs()
                + (expected_neighbor_seed17[channel] - expected_neighbor_seed29[channel]).abs());
        let precision_floor = cycles_film_precision_floor(
            potter_frames,
            blender_frames,
            [neighbor_position[0], neighbor_position[1], 0],
            channel,
            gamma_n,
        );
        let tolerance = noise_bound.max(precision_floor);
        if (actual_neighbor[channel] - expected_neighbor[channel]).abs() > tolerance {
            return Err(format!(
                "holdout backplate RGB channel {channel}: Potter pair mean {} differs from Blender {} by more than max(seed-noise bound {noise_bound}, source-derived single-terminal RGB precision floor {precision_floor}) = {tolerance}",
                actual_neighbor[channel],
                expected_neighbor[channel],
            )
            .into());
        }
    }
    Ok(())
}
mod realtime_holdout_mix {
    use std::{fs, path::Path};

    use super::{
        blender_checked,
        native_oracle::{
            TestResult, apply_operations, checked, import_blend, losses, pot, read_scene,
        },
    };
    use potter_core::image::{
        ImageData, ImageInterpolation, decode_pixels, load_image_data, sample,
    };
    use potter_core::model::{Image, ImageAlphaMode, ImageColorspace};
    use serde_json::{Value, json};
    use tempfile::tempdir;

    const FIXTURE: &str = "tests/fixtures/native_oracle/eevee_macos/holdout_mix.blend";
    const CAPTURE: &str = include_str!("fixtures/native_oracle/eevee_macos/holdout_mix.json");
    const EEVEE_EXR: &[u8] = include_bytes!("fixtures/native_oracle/eevee_macos/holdout_mix.exr");
    const RESOLUTION: u32 = 64;
    const SAMPLES: u32 = 16;
    /// Blender's half-float render storage rounds RGB values; this remains far below a missing 50% contribution.
    const RGB_TOLERANCE: f64 = 1.0e-3;
    /// EEVEE stores 0.498046875 for the half-alpha targets; Potter's closure coverage is 0.5.
    const ALPHA_TOLERANCE: f64 = 2.1e-3;
    const JSON_EXR_TOLERANCE: f64 = 1.0e-6;
    const METHOD_FIXTURE_DIR: &str = "tests/fixtures/native_oracle/holdout_surface_render_method";
    const METHOD_CAPTURE: &str = include_str!(
        "fixtures/native_oracle/holdout_surface_render_method/holdout_surface_render_method.json"
    );
    const DITHERED_METHOD_EXR: &[u8] =
        include_bytes!("fixtures/native_oracle/holdout_surface_render_method/holdout_dithered.exr");
    const BLENDED_METHOD_EXR: &[u8] =
        include_bytes!("fixtures/native_oracle/holdout_surface_render_method/holdout_blended.exr");
    const METHOD_RESOLUTION: u32 = 128;

    fn capture_xy(probe: &Value) -> TestResult<(u32, u32)> {
        let coordinates = probe["xy"]
            .as_array()
            .ok_or("capture pixel coordinates are missing")?;
        if coordinates.len() != 2 {
            return Err("capture pixel coordinates must have two entries".into());
        }
        Ok((
            u32::try_from(
                coordinates[0]
                    .as_u64()
                    .ok_or("capture x coordinate is not an integer")?,
            )?,
            u32::try_from(
                coordinates[1]
                    .as_u64()
                    .ok_or("capture y coordinate is not an integer")?,
            )?,
        ))
    }

    fn capture_rgba(probe: &Value) -> TestResult<[f64; 4]> {
        let channels = probe["rgba"]
            .as_array()
            .ok_or("capture RGBA values are missing")?;
        if channels.len() != 4 {
            return Err("capture pixel must have four channels".into());
        }
        Ok([
            channels[0].as_f64().ok_or("capture red is not numeric")?,
            channels[1].as_f64().ok_or("capture green is not numeric")?,
            channels[2].as_f64().ok_or("capture blue is not numeric")?,
            channels[3].as_f64().ok_or("capture alpha is not numeric")?,
        ])
    }

    fn pixel_at(pixels: &[[f64; 4]], x: u32, y: u32) -> TestResult<[f64; 4]> {
        let index = usize::try_from(y)? * RESOLUTION as usize + usize::try_from(x)?;
        pixels
            .get(index)
            .copied()
            .ok_or_else(|| format!("pixel ({x}, {y}) is outside the pinned capture").into())
    }

    fn rendered_pixel(frame: &ImageData, x: u32, y: u32) -> [f64; 4] {
        sample(
            frame,
            [
                (f64::from(x) + 0.5) / f64::from(RESOLUTION),
                (f64::from(y) + 0.5) / f64::from(RESOLUTION),
            ],
            false,
        )
    }

    fn assert_pixel_matches(
        label: &str,
        actual: [f64; 4],
        expected: [f64; 4],
        rgb_tolerance: f64,
        alpha_tolerance: f64,
    ) {
        for channel in 0..4 {
            let tolerance = if channel == 3 {
                alpha_tolerance
            } else {
                rgb_tolerance
            };
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "{label} channel {channel}: actual {} differs from expected {} (tolerance {tolerance})",
                actual[channel],
                expected[channel]
            );
        }
    }

    #[test]
    fn realtime_holdout_diffuse_and_transparent_mixes_match_eevee() -> TestResult {
        let Some(blender) = blender_checked::blender_executable() else {
            eprintln!("skipping the realtime Holdout mix oracle; no executable Blender was found");
            return Ok(());
        };

        let capture: Value = serde_json::from_str(CAPTURE)?;
        assert_eq!(capture["blender_version"], "5.2.2 LTS");
        assert_eq!(capture["resolution"], json!([RESOLUTION, RESOLUTION]));
        assert_eq!(capture["samples"], SAMPLES);
        assert_eq!(capture["film_transparent"], true);

        let (eevee_width, eevee_height, eevee_pixels) =
            decode_pixels(EEVEE_EXR, ImageColorspace::Linear)?;
        assert_eq!((eevee_width, eevee_height), (RESOLUTION, RESOLUTION));

        let root = tempdir()?;
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
        let (project, _) = import_blend(&blender, root.path(), "holdout_mix_project", &fixture)?;
        let scene = read_scene(&project)?;
        let active_scene = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let updated = apply_operations(
            &project,
            root.path(),
            json!([{
                "op": "render.update",
                "target": {"id": active_scene},
                "set": {
                    "resolution_x": RESOLUTION,
                    "resolution_y": RESOLUTION,
                    "samples": SAMPLES,
                    "seed": 0,
                    "max_bounces": 4
                }
            }]),
        )?;
        assert_eq!(updated["ok"], true, "{updated}");

        let frames = root.path().join("frames");
        let mut render = pot();
        render
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json");
        checked(&mut render, "realtime Holdout mix render")?;
        let frame_path = frames.join("frame_0001.exr");
        let (potter_width, potter_height, potter_pixels) =
            decode_pixels(&fs::read(&frame_path)?, ImageColorspace::Linear)?;
        assert_eq!((potter_width, potter_height), (RESOLUTION, RESOLUTION));
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{
                "op": "image.load",
                "id": "holdout_mix_render",
                "path": frame_path.to_string_lossy()
            }]),
        )?;
        assert_eq!(loaded["ok"], true, "{loaded}");
        let scene = read_scene(&project)?;
        let image: Image = serde_json::from_value(scene["images"]["holdout_mix_render"].clone())?;
        assert_eq!(
            image.alpha_mode,
            ImageAlphaMode::Premultiplied,
            "rendered EXR must retain the documented associated-image mode"
        );
        let frame = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        assert_eq!((frame.width, frame.height), (RESOLUTION, RESOLUTION));
        let cases = [
            ("holdout_diffuse", "EEVEE_HOLDOUT_DIFFUSE_50_50"),
            ("holdout_transparent", "EEVEE_HOLDOUT_TRANSPARENT_50_50"),
        ];
        let mut rendered = Vec::with_capacity(cases.len());
        for (key, contract_id) in cases {
            let probe = &capture["probes"][key];
            assert_eq!(probe["contract_id"], contract_id);
            let target = &probe["target"];
            let neighbor = &probe["backplate_neighbor"];
            let (target_x, target_y) = capture_xy(target)?;
            let (neighbor_x, neighbor_y) = capture_xy(neighbor)?;
            let target_reference = pixel_at(&eevee_pixels, target_x, target_y)?;
            let neighbor_reference = pixel_at(&eevee_pixels, neighbor_x, neighbor_y)?;
            assert_pixel_matches(
                &format!("{contract_id} JSON/EXR target"),
                capture_rgba(target)?,
                target_reference,
                JSON_EXR_TOLERANCE,
                JSON_EXR_TOLERANCE,
            );
            assert_pixel_matches(
                &format!("{contract_id} JSON/EXR backplate neighbor"),
                capture_rgba(neighbor)?,
                neighbor_reference,
                JSON_EXR_TOLERANCE,
                JSON_EXR_TOLERANCE,
            );

            let target_associated = pixel_at(&potter_pixels, target_x, target_y)?;
            let neighbor_associated = pixel_at(&potter_pixels, neighbor_x, neighbor_y)?;
            assert_pixel_matches(
                &format!("{contract_id} raw Potter EXR target"),
                target_associated,
                target_reference,
                RGB_TOLERANCE,
                ALPHA_TOLERANCE,
            );
            assert_pixel_matches(
                &format!("{contract_id} raw Potter EXR backplate neighbor"),
                neighbor_associated,
                neighbor_reference,
                RGB_TOLERANCE,
                ALPHA_TOLERANCE,
            );

            let target_loaded = rendered_pixel(&frame, target_x, target_y);
            let neighbor_loaded = rendered_pixel(&frame, neighbor_x, neighbor_y);
            assert_pixel_matches(
                &format!("{contract_id} loaded Potter image target"),
                target_loaded,
                target_reference,
                RGB_TOLERANCE,
                ALPHA_TOLERANCE,
            );
            assert_pixel_matches(
                &format!("{contract_id} loaded Potter image backplate neighbor"),
                neighbor_loaded,
                neighbor_reference,
                RGB_TOLERANCE,
                ALPHA_TOLERANCE,
            );
            assert!(
                target_associated[3] > 0.0 && target_associated[3] < 1.0,
                "{contract_id}: raw target alpha must retain a partial hole, got {}",
                target_associated[3]
            );
            assert!(
                (target_associated[3] - 0.5).abs() <= ALPHA_TOLERANCE,
                "{contract_id}: raw target alpha must be half coverage, got {}",
                target_associated[3]
            );
            assert!(
                (neighbor_associated[3] - 1.0).abs() <= ALPHA_TOLERANCE,
                "{contract_id}: nearby opaque backplate alpha is {}",
                neighbor_associated[3]
            );
            rendered.push((key, target_associated, neighbor_associated));
        }

        let (_, diffuse, diffuse_neighbor) = rendered[0];
        assert!(
            diffuse[2] < diffuse_neighbor[2] * 0.1,
            "Holdout + Diffuse leaked backplate blue into the diffuse-only share: target {diffuse:?}, neighbor {diffuse_neighbor:?}"
        );

        let (_, transparent, transparent_neighbor) = rendered[1];
        for channel in 0..3 {
            let passed_backplate = transparent_neighbor[channel] * 0.5;
            assert!(
                (transparent[channel] - passed_backplate).abs() <= RGB_TOLERANCE,
                "Holdout + Transparent channel {channel}: target {} does not equal the 50% backplate contribution {}",
                transparent[channel],
                passed_backplate
            );
        }
        Ok(())
    }
    fn method_capture_xy(probe: &Value) -> TestResult<(u32, u32)> {
        let coordinates = probe["pixel_xy"]
            .as_array()
            .ok_or("surface-method capture pixel coordinates are missing")?;
        if coordinates.len() != 2 {
            return Err("surface-method capture coordinates must have two entries".into());
        }
        Ok((
            u32::try_from(
                coordinates[0]
                    .as_u64()
                    .ok_or("surface-method x coordinate is not an integer")?,
            )?,
            u32::try_from(
                coordinates[1]
                    .as_u64()
                    .ok_or("surface-method y coordinate is not an integer")?,
            )?,
        ))
    }

    fn method_capture_rgba(probe: &Value) -> TestResult<[f64; 4]> {
        let values = probe["raw_associated_rgba"]
            .as_array()
            .ok_or("surface-method capture RGBA is missing")?;
        if values.len() != 4 {
            return Err("surface-method capture must have four channels".into());
        }
        Ok([
            values[0]
                .as_f64()
                .ok_or("surface-method red is not numeric")?,
            values[1]
                .as_f64()
                .ok_or("surface-method green is not numeric")?,
            values[2]
                .as_f64()
                .ok_or("surface-method blue is not numeric")?,
            values[3]
                .as_f64()
                .ok_or("surface-method alpha is not numeric")?,
        ])
    }

    fn method_pixel_at(pixels: &[[f64; 4]], x: u32, y: u32) -> TestResult<[f64; 4]> {
        if x >= METHOD_RESOLUTION || y >= METHOD_RESOLUTION {
            return Err(format!("surface-method pixel ({x}, {y}) is outside the capture").into());
        }
        // The manifest coordinates come from Blender's bottom-up `image.pixels`; EXR decoder rows are top-down.
        let row = usize::try_from(METHOD_RESOLUTION - 1 - y)?;
        let width = usize::try_from(METHOD_RESOLUTION)?;
        let index = row * width + usize::try_from(x)?;
        pixels
            .get(index)
            .copied()
            .ok_or_else(|| format!("surface-method pixel ({x}, {y}) is outside the capture").into())
    }
    fn assert_material_rays(
        probe: &Value,
        object_id: &str,
        material_id: &str,
        coordinates: [[u32; 2]; 2],
        label: &str,
    ) -> TestResult<()> {
        assert_eq!(probe["expected_object_id"], object_id);
        assert_eq!(probe["expected_evaluated_material_id"], material_id);
        for (location, expected_xy) in ["center", "interior_corner"].into_iter().zip(coordinates) {
            let ray = &probe[location];
            assert_eq!(method_capture_xy(ray)?, (expected_xy[0], expected_xy[1]));
            assert_eq!(ray["hit"], true, "{label} {location} ray missed");
            assert_eq!(ray["object_id"], object_id, "{label} {location} object");
            assert_eq!(ray["face_index"], 0, "{label} {location} face");
            assert_eq!(ray["material_index"], 0, "{label} {location} material slot");
            assert_eq!(
                ray["evaluated_material_id"], material_id,
                "{label} {location} evaluated material"
            );
        }
        Ok(())
    }

    fn assert_world_rays_miss(probe: &Value, coordinates: [[u32; 2]; 2]) -> TestResult<()> {
        assert_eq!(probe["expected_hit"], false);
        for (location, expected_xy) in ["center", "interior_corner"].into_iter().zip(coordinates) {
            let ray = &probe[location];
            assert_eq!(method_capture_xy(ray)?, (expected_xy[0], expected_xy[1]));
            assert_eq!(
                ray["hit"], false,
                "transparent-world {location} ray hit geometry"
            );
            assert!(ray["object_id"].is_null());
            assert!(ray["face_index"].is_null());
            assert!(ray["material_index"].is_null());
            assert!(ray["evaluated_material_id"].is_null());
        }
        Ok(())
    }

    #[test]
    fn holdout_surface_method_uses_source_forward_and_deferred_association() -> TestResult<()> {
        let Some(blender) = blender_checked::blender_executable() else {
            eprintln!(
                "skipping EEVEE Holdout surface-method parity; no executable Blender was found"
            );
            return Ok(());
        };
        let capture: Value = serde_json::from_str(METHOD_CAPTURE)?;
        assert_eq!(capture["blender_version"], "5.2.2 LTS");
        assert_eq!(capture["blender_build_hash"], "d13f752e3b9c");
        assert_eq!(capture["scene_inputs"]["engine"], "BLENDER_EEVEE");
        assert_eq!(
            capture["scene_inputs"]["resolution"],
            json!([METHOD_RESOLUTION, METHOD_RESOLUTION])
        );
        assert_eq!(capture["scene_inputs"]["render_samples"], 1);
        assert_eq!(capture["scene_inputs"]["film_transparent"], true);
        assert_eq!(capture["scene_inputs"]["sun"]["angle"], 0.0);

        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(METHOD_FIXTURE_DIR);
        let root = tempdir()?;
        let modes = [
            ("DITHERED", "holdout_dithered.blend", DITHERED_METHOD_EXR),
            ("BLENDED", "holdout_blended.blend", BLENDED_METHOD_EXR),
        ];
        let mut source_targets = [[0.0; 4]; 2];
        for (index, (method, blend_name, source_exr)) in modes.into_iter().enumerate() {
            let source_case = &capture["captures_by_surface_render_method"][method];
            assert_eq!(source_case["surface_render_method"], method);
            assert_eq!(source_case["source_alpha_mode"], "PREMUL");
            let rays = &source_case["raycast_provenance"];
            assert_material_rays(
                &rays["probe_target"],
                "Holdout Diffuse Probe Patch",
                "Holdout 0.5 Plus Opaque Diffuse 0.5",
                [[64, 64], [40, 40]],
                "probe",
            )?;
            assert_material_rays(
                &rays["opaque_backplate_control"],
                "Opaque Diffuse Backplate Control",
                "Opaque Diffuse Backplate",
                [[16, 64], [8, 56]],
                "backplate",
            )?;
            assert_world_rays_miss(&rays["transparent_world"], [[112, 64], [104, 56]])?;
            let (source_width, source_height, source_pixels) =
                decode_pixels(source_exr, ImageColorspace::Linear)?;
            assert_eq!(
                (source_width, source_height),
                (METHOD_RESOLUTION, METHOD_RESOLUTION)
            );

            let (project, imported) = import_blend(
                &blender,
                root.path(),
                &format!("holdout_surface_method_{}", method.to_ascii_lowercase()),
                &fixture_dir.join(blend_name),
            )?;
            assert!(
                losses(&imported).is_empty(),
                "{method} Holdout import reported losses: {imported}"
            );
            let document = read_scene(&project)?;
            let scene_id = document["active_scene"]
                .as_str()
                .ok_or("active scene is missing")?;
            let scene = &document["scenes"][scene_id];
            assert_eq!(scene["render"]["engine"], "realtime");
            assert_eq!(scene["render"]["resolution_x"], METHOD_RESOLUTION);
            assert_eq!(scene["render"]["resolution_y"], METHOD_RESOLUTION);
            assert_eq!(scene["render"]["samples"], 1);
            assert_eq!(scene["render"]["film_transparent"], true);

            let frames = root
                .path()
                .join(format!("frames_{}", method.to_ascii_lowercase()));
            let mut render = pot();
            render
                .arg("render")
                .arg(&project)
                .args(["--engine", "realtime", "--format", "exr", "--out"])
                .arg(&frames)
                .arg("--json");
            checked(&mut render, &format!("realtime {method} Holdout render"))?;
            let frame_path = frames.join("frame_0001.exr");
            let (potter_width, potter_height, potter_pixels) =
                decode_pixels(&fs::read(&frame_path)?, ImageColorspace::Linear)?;
            assert_eq!(
                (potter_width, potter_height),
                (METHOD_RESOLUTION, METHOD_RESOLUTION)
            );

            for name in [
                "target",
                "opaque_backplate_control",
                "transparent_world_neighbor",
            ] {
                let probe = &source_case[name];
                let (x, y) = method_capture_xy(probe)?;
                let expected = method_capture_rgba(probe)?;
                let source_pixel = method_pixel_at(&source_pixels, x, y)?;
                assert_pixel_matches(
                    &format!("{method} JSON/EXR {name}"),
                    source_pixel,
                    expected,
                    JSON_EXR_TOLERANCE,
                    JSON_EXR_TOLERANCE,
                );
                let actual = method_pixel_at(&potter_pixels, x, y)?;
                assert_pixel_matches(
                    &format!("{method} raw Potter EXR {name}"),
                    actual,
                    expected,
                    RGB_TOLERANCE,
                    ALPHA_TOLERANCE,
                );
                if name == "target" {
                    source_targets[index] = expected;
                    assert!(
                        expected[3] > 0.0 && expected[3] < 1.0,
                        "{method} target must preserve its partial source film alpha: {expected:?}"
                    );
                }
                if name == "transparent_world_neighbor" {
                    assert_eq!(expected, [0.0; 4]);
                }
            }
        }
        for channel in 0..3 {
            assert!(
                source_targets[0][channel] > source_targets[1][channel],
                "the pinned DITHERED deferred contribution must exceed BLENDED forward radiance in channel {channel}: {:?} vs {:?}",
                source_targets[0],
                source_targets[1]
            );
        }
        Ok(())
    }
}
