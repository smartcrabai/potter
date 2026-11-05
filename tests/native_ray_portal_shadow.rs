//! Cycles mesh-shadow opacity for pure and mixed Ray Portal closures against Blender 5.2.2.
//!
//! The imported source scenes and EXR means/standard errors are retained as capture diagnostics.
//! Potter's consumer gate uses the exact SVM extinction/gating protocol (pure Portal opaque; a
//! qualifying Transparent closure exposes the signed Transparent + Portal RGB sum) and the shared
//! Cycles float32 film accumulation/output precision bound.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeMap, fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    FLOAT_TOLERANCE, PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE, TestResult, apply_operations, checked,
    cycles_film_precision_floor_for_frames, cycles_gamma_n, import_blend, losses, pot, read_scene,
};
use potter_core::{image::decode_pixels, model::ImageColorspace};
use serde::Deserialize;
use serde_json::json;
use tempfile::tempdir;

const FIXTURE_DIR: &str = "tests/fixtures/native_oracle/ray_portal_shadow";
const RESOLUTION: [u32; 2] = [4, 4];
const SAMPLES: u32 = 32;
const SEEDS: [u32; 2] = [31, 43];
const CASES: [&str; 3] = ["baseline", "pure_portal", "mixed"];
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/ray_portal_shadow/ray_portal_shadow.json");
const SOURCE_PURE_RATIO: [f64; 3] = [0.0; 3];
const SOURCE_MIXED_RATIO: [f64; 3] = [0.899_297_284_2, 0.698_359_565_2, 0.598_125_369_5];
const ANALYTIC_MIXED_RATIO: [f64; 3] = [0.9, 0.7, 0.6];

#[derive(Deserialize)]
struct Capture {
    blender_version: String,
    blender_build_hash: String,
    engine: String,
    device: String,
    resolution: [u32; 2],
    samples: u32,
    seeds: Vec<u32>,
    denoising: bool,
    max_bounces: u32,
    pixel_filter: String,
    filter_width: f64,
    view_transform: String,
    world_color: [f64; 3],
    world_strength: f64,
    sun_energy: f64,
    sun_angle: f64,
    sun_rotation_euler: [f64; 3],
    floor_dimensions: [f64; 2],
    camera_type: String,
    camera_ortho_scale: f64,
    occluder: OccluderInputs,
    portal_color: [f64; 4],
    portal_direction: [f64; 3],
    mixed_factor: f64,
    transparent_color: [f64; 4],
    transparent_shadow_gate: bool,
    cases: BTreeMap<String, CaptureCase>,
    ratios: Vec<CaptureRatio>,
}

#[derive(Deserialize)]
struct OccluderInputs {
    name: String,
    visible_camera: bool,
    visible_shadow: bool,
}

#[derive(Deserialize)]
struct CaptureCase {
    blend: String,
    renders: Vec<CaptureRender>,
}

#[derive(Deserialize)]
struct CaptureRender {
    seed: u32,
    linear_exr: String,
    pixel_count: u32,
    mean_linear_associated_rgba: [f64; 4],
    stderr_linear_associated_rgba: [f64; 4],
}

#[derive(Deserialize)]
struct CaptureRatio {
    seed: u32,
    case: String,
    rgb: [f64; 3],
    stderr_rgb: [f64; 3],
    alpha_ratio: f64,
}

struct FrameStats {
    mean: [f64; 4],
    stderr: [f64; 4],
    pixels: Vec<[f64; 4]>,
}

#[test]
fn ray_portal_mesh_shadow_matches_cycles_capture() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Ray Portal mesh-shadow parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(CAPTURE)?;
    check_capture(&capture)?;

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR);
    let mut projects = BTreeMap::new();
    for name in CASES {
        let blend = fixture_dir.join(&capture.cases[name].blend);
        let (project, imported) = import_blend(&blender, root.path(), name, &blend)?;
        assert!(
            losses(&imported).is_empty(),
            "{name} source import reported losses: {imported}"
        );
        let scene = read_scene(&project)?;
        let active = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let updated = apply_operations(
            &project,
            root.path(),
            json!([{
                "op": "render.update",
                "target": {"id": active},
                "set": {
                    "resolution_x": RESOLUTION[0],
                    "resolution_y": RESOLUTION[1],
                    "resolution_percentage": 100,
                    "samples": SAMPLES,
                    "seed": SEEDS[0],
                    "max_bounces": 1
                }
            }]),
        )?;
        assert_eq!(updated["ok"], true, "{updated}");
        assert!(projects.insert(name, (project, active)).is_none());
    }

    let mut actual = BTreeMap::new();
    for seed in SEEDS {
        for name in CASES {
            let (project, active) = &projects[name];
            let updated = apply_operations(
                project,
                root.path(),
                json!([{
                    "op": "render.update",
                    "target": {"id": active},
                    "set": {"seed": seed}
                }]),
            )?;
            assert_eq!(updated["ok"], true, "{updated}");
            assert!(
                actual
                    .insert(
                        (name, seed),
                        render_potter(project, root.path(), name, seed)?
                    )
                    .is_none()
            );
        }
    }

    let mut source = BTreeMap::new();
    for name in CASES {
        for reference in &capture.cases[name].renders {
            assert_eq!(reference.pixel_count, RESOLUTION[0] * RESOLUTION[1]);
            let exr = fixture_dir.join(&reference.linear_exr);
            let bytes = fs::read(&exr)?;
            let (width, height, pixels) = decode_pixels(&bytes, ImageColorspace::Linear)?;
            assert_eq!([width, height], RESOLUTION, "Blender EXR {name}");
            let stats = frame_stats(&pixels);
            assert_stats_match(
                &stats,
                reference.mean_linear_associated_rgba,
                reference.stderr_linear_associated_rgba,
                &format!("Blender {name} seed {}", reference.seed),
            );
            assert!(source.insert((name, reference.seed), stats).is_none());
        }
    }

    for seed in SEEDS {
        let source_baseline = &source[&("baseline", seed)];
        let actual_baseline = &actual[&("baseline", seed)];
        assert_alpha_one(source_baseline, &format!("Blender baseline seed {seed}"));
        assert_alpha_one(actual_baseline, &format!("Potter baseline seed {seed}"));

        for name in ["pure_portal", "mixed"] {
            let expected_capture = capture
                .ratios
                .iter()
                .find(|ratio| ratio.seed == seed && ratio.case == name)
                .ok_or_else(|| format!("missing {name} seed {seed} source ratio"))?;
            let source_case = &source[&(name, seed)];
            let actual_case = &actual[&(name, seed)];
            let (source_rgb, source_stderr, source_alpha) =
                ratio_stats(source_case, source_baseline);
            let (actual_rgb, actual_stderr, actual_alpha) =
                ratio_stats(actual_case, actual_baseline);
            assert_vector_close(
                source_rgb,
                expected_capture.rgb,
                FLOAT_TOLERANCE,
                &format!("Blender {name} seed {seed} reference ratio"),
            );
            assert_vector_close(
                source_stderr,
                expected_capture.stderr_rgb,
                FLOAT_TOLERANCE,
                &format!("Blender {name} seed {seed} reference ratio error"),
            );
            native_oracle::assert_close(
                source_alpha,
                expected_capture.alpha_ratio,
                FLOAT_TOLERANCE,
                &format!("Blender {name} seed {seed} alpha ratio"),
            );
            assert_eq!(
                expected_capture.alpha_ratio, 1.0,
                "Cycles alpha remains opaque"
            );
            assert_eq!(
                actual_alpha, 1.0,
                "Potter alpha remains opaque for {name} seed {seed}"
            );
            eprintln!(
                "{name} seed {seed}: Blender diagnostic ratio {source_rgb:?} +/- {source_stderr:?}; Potter ratio {actual_rgb:?} +/- {actual_stderr:?}"
            );
        }
    }

    let baseline_first = &actual[&("baseline", SEEDS[0])];
    let baseline_second = &actual[&("baseline", SEEDS[1])];
    let baseline_mean = paired_mean(baseline_first, baseline_second);
    let baseline_potter_frames = [
        baseline_first.pixels.as_slice(),
        baseline_second.pixels.as_slice(),
    ];
    let baseline_source_frames = [
        source[&("baseline", SEEDS[0])].pixels.as_slice(),
        source[&("baseline", SEEDS[1])].pixels.as_slice(),
    ];
    let film_additions = SAMPLES * PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE;
    let gamma_n = cycles_gamma_n(film_additions);
    for name in ["pure_portal", "mixed"] {
        let first = &actual[&(name, SEEDS[0])];
        let second = &actual[&(name, SEEDS[1])];
        let mean = paired_mean(first, second);
        let analytic = if name == "pure_portal" {
            SOURCE_PURE_RATIO
        } else {
            ANALYTIC_MIXED_RATIO
        };
        let potter_frames = [first.pixels.as_slice(), second.pixels.as_slice()];
        let source_frames = [
            source[&(name, SEEDS[0])].pixels.as_slice(),
            source[&(name, SEEDS[1])].pixels.as_slice(),
        ];
        let ratio: [f64; 4] = std::array::from_fn(|channel| mean[channel] / baseline_mean[channel]);
        assert_eq!(
            ratio[3], 1.0,
            "Potter paired alpha ratio remains opaque for {name}"
        );
        for channel in 0..3 {
            let baseline_floor = cycles_film_precision_floor_for_frames(
                baseline_potter_frames,
                baseline_source_frames,
                channel,
                gamma_n,
            );
            let numerator_floor = cycles_film_precision_floor_for_frames(
                potter_frames,
                source_frames,
                channel,
                gamma_n,
            );
            let denominator = baseline_mean[channel].abs();
            let denominator_lower = denominator - baseline_floor;
            assert!(
                denominator_lower > 0.0,
                "baseline channel {channel} must exceed its source precision floor"
            );
            let ratio_floor = numerator_floor / denominator_lower
                + (mean[channel].abs() + numerator_floor) * baseline_floor
                    / (denominator * denominator_lower);
            let error = (ratio[channel] - analytic[channel]).abs();
            assert!(
                error <= ratio_floor,
                "Potter {name} channel {channel}: RGB ratio {} differs from source closure ratio {} by {error}, exceeding source-derived gamma_n/output budget {ratio_floor} ({} samples x {PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE} additions, gamma_n={gamma_n})",
                ratio[channel],
                analytic[channel],
                SAMPLES * PATH_MAX_RGB_FILM_ADDITIONS_PER_SAMPLE,
            );
        }
    }

    Ok(())
}

fn check_capture(capture: &Capture) -> TestResult {
    assert_eq!(capture.blender_version, "5.2.2 LTS");
    assert_eq!(capture.blender_build_hash, "d13f752e3b9c");
    assert_eq!(capture.engine, "CYCLES");
    assert_eq!(capture.device, "CPU");
    assert_eq!(capture.resolution, RESOLUTION);
    assert_eq!(capture.samples, SAMPLES);
    assert_eq!(capture.seeds.as_slice(), SEEDS.as_slice());
    assert!(!capture.denoising);
    assert_eq!(capture.max_bounces, 1);
    assert_eq!(capture.pixel_filter, "BLACKMAN_HARRIS");
    assert_eq!(capture.filter_width, 1.5);
    assert_eq!(capture.view_transform, "Standard");
    assert_eq!(capture.world_color, [0.0; 3]);
    assert_eq!(capture.sun_rotation_euler, [0.0; 3]);
    assert_eq!(capture.floor_dimensions, [8.0, 8.0]);
    assert_eq!(capture.world_strength, 0.0);
    assert_eq!(capture.sun_energy, 1.0);
    assert_eq!(capture.sun_angle, 0.0);
    assert_eq!(capture.camera_type, "ORTHO");
    assert_eq!(capture.camera_ortho_scale, 1.5);
    assert_eq!(capture.occluder.name, "RayPortalOccluder");
    assert!(!capture.occluder.visible_camera);
    assert!(capture.occluder.visible_shadow);
    assert_eq!(capture.portal_color, [0.8, 0.4, 0.2, 1.0]);
    assert_eq!(capture.portal_direction, [0.0, 0.0, -1.0]);
    assert_eq!(capture.mixed_factor, 0.5);
    assert_eq!(capture.transparent_color, [1.0; 4]);
    assert!(capture.transparent_shadow_gate);

    assert_eq!(capture.cases.len(), CASES.len());
    for name in CASES {
        let case = capture
            .cases
            .get(name)
            .ok_or_else(|| format!("missing {name} fixture"))?;
        assert_eq!(case.blend, format!("{name}.blend"));
        assert_eq!(case.renders.len(), SEEDS.len());
        for (reference, seed) in case.renders.iter().zip(SEEDS) {
            assert_eq!(reference.seed, seed);
            assert_eq!(reference.linear_exr, format!("{name}-seed{seed}.exr"));
        }
    }
    assert_eq!(capture.ratios.len(), 4);
    for ratio in &capture.ratios {
        let expected_rgb = if ratio.case == "pure_portal" {
            SOURCE_PURE_RATIO
        } else {
            assert_eq!(ratio.case, "mixed");
            SOURCE_MIXED_RATIO
        };
        assert_vector_close(
            ratio.rgb,
            expected_rgb,
            FLOAT_TOLERANCE,
            &format!("pinned Blender {} seed {} ratio", ratio.case, ratio.seed),
        );
        assert_eq!(ratio.alpha_ratio, 1.0);
    }
    Ok(())
}

fn render_potter(project: &Path, root: &Path, name: &str, seed: u32) -> TestResult<FrameStats> {
    let label = format!("ray_portal_{name}_seed{seed}");
    let frames = root.join(&label);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, &label)?;
    let bytes = fs::read(frames.join("frame_0001.exr"))?;
    let (width, height, pixels) = decode_pixels(&bytes, ImageColorspace::Linear)?;
    assert_eq!([width, height], RESOLUTION, "Potter {name} seed {seed}");
    Ok(frame_stats(&pixels))
}

fn frame_stats(pixels: &[[f64; 4]]) -> FrameStats {
    assert!(pixels.len() > 1);
    let mean = std::array::from_fn(|channel| {
        pixels.iter().map(|pixel| pixel[channel]).sum::<f64>() / pixels.len() as f64
    });
    let stderr = std::array::from_fn(|channel| {
        let variance = pixels
            .iter()
            .map(|pixel| (pixel[channel] - mean[channel]).powi(2))
            .sum::<f64>()
            / (pixels.len() - 1) as f64;
        (variance / pixels.len() as f64).sqrt()
    });
    FrameStats {
        mean,
        stderr,
        pixels: pixels.to_vec(),
    }
}

fn paired_mean(first: &FrameStats, second: &FrameStats) -> [f64; 4] {
    std::array::from_fn(|channel| f64::midpoint(first.mean[channel], second.mean[channel]))
}

fn ratio_stats(tested: &FrameStats, baseline: &FrameStats) -> ([f64; 3], [f64; 3], f64) {
    let rgb = std::array::from_fn(|channel| tested.mean[channel] / baseline.mean[channel]);
    let stderr = std::array::from_fn(|channel| {
        (tested.stderr[channel] / baseline.mean[channel])
            .hypot(tested.mean[channel] * baseline.stderr[channel] / baseline.mean[channel].powi(2))
    });
    (rgb, stderr, tested.mean[3] / baseline.mean[3])
}

fn assert_stats_match(actual: &FrameStats, mean: [f64; 4], stderr: [f64; 4], label: &str) {
    assert_vector_close(
        actual.mean,
        mean,
        FLOAT_TOLERANCE,
        &format!("{label} EXR mean"),
    );
    assert_vector_close(
        actual.stderr,
        stderr,
        FLOAT_TOLERANCE,
        &format!("{label} EXR standard error"),
    );
}

fn assert_alpha_one(frame: &FrameStats, label: &str) {
    native_oracle::assert_close(frame.mean[3], 1.0, FLOAT_TOLERANCE, label);
}

fn assert_vector_close<const N: usize>(
    actual: [f64; N],
    expected: [f64; N],
    tolerance: f64,
    label: &str,
) {
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        native_oracle::assert_close(
            actual,
            expected,
            tolerance,
            &format!("{label} channel {channel}"),
        );
    }
}
