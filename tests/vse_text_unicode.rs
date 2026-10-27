use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::Value;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const FRAME_CASES: [(u32, &str, &str); 7] = [
    (1, "noto_cjk_multiline_text", "日本語\n漢字中文"),
    (2, "lato_accented_multiline_text", "AV Café\nÀ bientôt"),
    (3, "liberation_av_kern_text", "AV"),
    (4, "cjk_no_space_overflow_text", "日本語漢字"),
    (5, "cjk_ideographic_space_wrap_text", "日本語　漢"),
    (6, "cjk_ascii_space_wrap_text", "日本語 漢"),
    (7, "lato_italic_final_negative_lsb_text", "AF"),
];
const HALF_RESOLUTION_CASES: [(u32, &str, &str); 1] = [(1, "scaled_size_50_percent_text", "M")];
const HALF_RESOLUTION_PIXEL_PARITY_LIMITS: [(u64, f64, u32); 1] = [(3_013, 0.400, 1)];
const WRAP_WIDTH_CASES: [(&str, f64); 3] = [
    ("cjk_no_space_overflow_text", 0.24),
    ("cjk_ideographic_space_wrap_text", 0.24),
    ("cjk_ascii_space_wrap_text", 0.24),
];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn successful_json(output: &Output, operation: &str) -> TestResult<Value> {
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{operation} failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init_project(project: &Path) -> TestResult {
    let output = pot().arg("init").arg(project).arg("--json").output()?;
    successful_json(&output, "pot init")?;
    Ok(())
}

fn import_blend(project: &Path, blend: &Path, blender: &Path) -> TestResult<Value> {
    let output = pot()
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
        ])
        .arg("--blender")
        .arg(blender)
        .arg("--json")
        .output()?;
    successful_json(&output, "pot import Blender text-strip fixture")
}

fn inspect_fonts(
    project: &Path,
    cases: &[(u32, &str, &str)],
    wrap_widths: &[(&str, f64)],
) -> TestResult {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let scene_id = scene["active_scene"]
        .as_str()
        .ok_or("active scene ID is missing")?;
    let output = pot().arg("inspect").arg(project).arg("--json").output()?;
    let inspected = successful_json(&output, "pot inspect text-strip state")?;
    let strips = inspected["result"]["sequencers"][scene_id]["strips"]
        .as_array()
        .ok_or("inspect sequencers map is missing the active scene")?;
    for &(frame, expected_name, expected_text) in cases {
        let strip = strips
            .iter()
            .find(|strip| strip["name"] == expected_name)
            .ok_or_else(|| format!("inspect has no `{expected_name}` text strip"))?;
        assert_eq!(strip["type"], "text");
        assert_eq!(strip["frame_start"].as_f64(), Some(f64::from(frame)));
        assert_eq!(strip["text"], expected_text);
        let font = strip["font"]
            .as_str()
            .ok_or_else(|| format!("{expected_name} font reference is missing from inspect"))?;
        assert!(!font.is_empty(), "{expected_name} font reference is empty");
    }
    for &(expected_name, expected_width) in wrap_widths {
        let strip = strips
            .iter()
            .find(|strip| strip["name"] == expected_name)
            .ok_or_else(|| format!("inspect has no `{expected_name}` text strip"))?;
        let actual_width = strip["wrap_width"]
            .as_f64()
            .ok_or_else(|| format!("inspect has no `{expected_name}` wrap width"))?;
        assert!(
            (actual_width - expected_width).abs() < 1.0e-6,
            "{expected_name} wrap width {actual_width} differs from {expected_width}"
        );
    }
    Ok(())
}

fn render_frames(project: &Path, output_directory: &Path, frames: &str) -> TestResult<Value> {
    let output = pot()
        .arg("render")
        .arg(project)
        .args(["--format", "png", "--frames", frames, "--out"])
        .arg(output_directory)
        .arg("--json")
        .output()?;
    successful_json(&output, "pot render Unicode VSE frames")
}

fn update_alpha_bounds(bounds: &mut [u32; 4], x: u32, y: u32, alpha: u8) {
    if alpha > 0 {
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x + 1);
        bounds[3] = bounds[3].max(y + 1);
    }
}

#[derive(Debug)]
struct PixelMetrics {
    raw_max: u8,
    raw_mean: f64,
    alpha_max: u8,
    alpha_mean: f64,
    alpha_changed_pixels: u64,
    actual_alpha_bounds: [u32; 4],
    expected_alpha_bounds: [u32; 4],
    premultiplied_max: u8,
    premultiplied_mean: f64,
}

fn compare_frame(
    actual_path: &Path,
    expected_path: &Path,
    label: &str,
) -> TestResult<PixelMetrics> {
    let actual = image::load_from_memory(&fs::read(actual_path)?)?.to_rgba8();
    let expected = image::load_from_memory(&fs::read(expected_path)?)?.to_rgba8();
    assert_eq!(
        actual.dimensions(),
        expected.dimensions(),
        "{label}: dimensions"
    );
    let (mut raw_max, mut raw_total) = (0_u8, 0_u64);
    let (mut alpha_max, mut alpha_total) = (0_u8, 0_u64);
    let (mut premultiplied_max, mut premultiplied_total) = (0_u8, 0_u64);
    let mut alpha_changed_pixels = 0_u64;
    let mut actual_alpha_bounds = [u32::MAX, u32::MAX, 0, 0];
    let mut expected_alpha_bounds = [u32::MAX, u32::MAX, 0, 0];
    let mut channel_count = 0_u64;
    for (x, y, actual_pixel) in actual.enumerate_pixels() {
        let expected_pixel = expected.get_pixel(x, y);
        let actual_alpha = actual_pixel.0[3];
        let expected_alpha = expected_pixel.0[3];
        update_alpha_bounds(&mut actual_alpha_bounds, x, y, actual_alpha);
        update_alpha_bounds(&mut expected_alpha_bounds, x, y, expected_alpha);
        let alpha_delta = actual_alpha.abs_diff(expected_alpha);
        alpha_max = alpha_max.max(alpha_delta);
        alpha_total += u64::from(alpha_delta);
        alpha_changed_pixels += u64::from(alpha_delta > 0);
        for channel in 0..4 {
            let actual_channel = actual_pixel.0[channel];
            let expected_channel = expected_pixel.0[channel];
            let raw_delta = actual_channel.abs_diff(expected_channel);
            raw_max = raw_max.max(raw_delta);
            raw_total += u64::from(raw_delta);
            let actual_premultiplied = if channel == 3 {
                actual_alpha
            } else {
                ((u16::from(actual_channel) * u16::from(actual_alpha) + 127) / 255) as u8
            };
            let expected_premultiplied = if channel == 3 {
                expected_alpha
            } else {
                ((u16::from(expected_channel) * u16::from(expected_alpha) + 127) / 255) as u8
            };
            let premultiplied_delta = actual_premultiplied.abs_diff(expected_premultiplied);
            premultiplied_max = premultiplied_max.max(premultiplied_delta);
            premultiplied_total += u64::from(premultiplied_delta);
            channel_count += 1;
        }
    }
    let pixel_count = u64::from(actual.width()) * u64::from(actual.height());
    let metrics = PixelMetrics {
        raw_max,
        raw_mean: raw_total as f64 / channel_count as f64,
        alpha_max,
        alpha_mean: alpha_total as f64 / pixel_count as f64,
        alpha_changed_pixels,
        actual_alpha_bounds,
        expected_alpha_bounds,
        premultiplied_max,
        premultiplied_mean: premultiplied_total as f64 / channel_count as f64,
    };
    println!(
        "VSE Unicode {label}: {}x{}, raw max/mean {}/{:.6}, alpha max/mean {}/{:.6} over {} changed pixels, alpha bounds actual {:?} expected {:?}, premultiplied max/mean {}/{:.6}",
        actual.width(),
        actual.height(),
        metrics.raw_max,
        metrics.raw_mean,
        metrics.alpha_max,
        metrics.alpha_mean,
        metrics.alpha_changed_pixels,
        metrics.actual_alpha_bounds,
        metrics.expected_alpha_bounds,
        metrics.premultiplied_max,
        metrics.premultiplied_mean,
    );
    Ok(metrics)
}

const PIXEL_PARITY_LIMITS: [(u64, f64, u32); 2] = [(5_514, 2.540, 2), (3_013, 0.400, 1)];
const ADDITIONAL_PIXEL_PARITY_LIMITS: [(u64, f64, u32); 5] = [
    PIXEL_PARITY_LIMITS[1],
    PIXEL_PARITY_LIMITS[0],
    PIXEL_PARITY_LIMITS[0],
    PIXEL_PARITY_LIMITS[0],
    PIXEL_PARITY_LIMITS[1],
];

fn measure_blender_parity_cases(
    render_dir: &Path,
    fixture_dir: &Path,
    cases: &[(u32, &str, &str)],
    limits: &[(u64, f64, u32)],
    capture_prefix: &str,
) -> TestResult {
    assert_eq!(cases.len(), limits.len(), "oracle case and tolerance count");
    for (&(frame, name, _), &(changed_pixel_limit, mean_error_limit, bounds_limit)) in
        cases.iter().zip(limits)
    {
        let actual = render_dir.join(format!("frame_{frame:04}.png"));
        let expected = fixture_dir.join(format!("{capture_prefix}{frame:04}.png"));
        let metrics = compare_frame(&actual, &expected, name)?;
        let bounds_delta = metrics
            .actual_alpha_bounds
            .iter()
            .zip(metrics.expected_alpha_bounds)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .fold(0_u32, u32::max);
        assert!(
            metrics.alpha_changed_pixels <= changed_pixel_limit,
            "{name}: changed alpha pixel count {} exceeds {changed_pixel_limit}",
            metrics.alpha_changed_pixels
        );
        assert!(
            metrics.alpha_mean <= mean_error_limit
                && metrics.premultiplied_mean <= mean_error_limit,
            "{name}: alpha/premultiplied MAE {}/{:.6} exceeds {mean_error_limit}/255",
            metrics.alpha_max,
            metrics.premultiplied_mean
        );
        assert!(
            bounds_delta <= bounds_limit,
            "{name}: alpha bounds differ by {bounds_delta} pixels, limit {bounds_limit}"
        );
    }
    Ok(())
}

fn measure_blender_parity(render_dir: &Path, fixture_dir: &Path) -> TestResult {
    let limits = [
        PIXEL_PARITY_LIMITS[0],
        PIXEL_PARITY_LIMITS[1],
        ADDITIONAL_PIXEL_PARITY_LIMITS[0],
        ADDITIONAL_PIXEL_PARITY_LIMITS[1],
        ADDITIONAL_PIXEL_PARITY_LIMITS[2],
        ADDITIONAL_PIXEL_PARITY_LIMITS[3],
        ADDITIONAL_PIXEL_PARITY_LIMITS[4],
    ];
    measure_blender_parity_cases(
        render_dir,
        fixture_dir,
        &FRAME_CASES,
        &limits,
        "supported_frame_",
    )
}

#[test]
fn unicode_text_strips_import_render_inspect_and_packed_round_trip_match_blender() -> TestResult {
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let root = tempdir()?;
    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_oracle/vse_text_unicode");
    let source_blend = fixture_dir.join("vse_text_unicode_supported.blend");

    let project = root.path().join("unicode_text_project");
    init_project(&project)?;
    let imported = import_blend(&project, &source_blend, &blender)?;
    assert_eq!(imported["result"]["losses"], serde_json::json!([]));
    inspect_fonts(&project, &FRAME_CASES, &WRAP_WIDTH_CASES)?;

    let render_dir = root.path().join("unicode_text_render");
    render_frames(&project, &render_dir, "1:7")?;
    measure_blender_parity(&render_dir, &fixture_dir)?;

    let half_source_blend = fixture_dir.join("vse_text_unicode_half_resolution.blend");
    let half_project = root.path().join("unicode_text_half_resolution_project");
    init_project(&half_project)?;
    let imported = import_blend(&half_project, &half_source_blend, &blender)?;
    assert_eq!(imported["result"]["losses"], serde_json::json!([]));
    inspect_fonts(&half_project, &HALF_RESOLUTION_CASES, &[])?;

    let half_render_dir = root.path().join("unicode_text_half_resolution_render");
    render_frames(&half_project, &half_render_dir, "1:1")?;
    measure_blender_parity_cases(
        &half_render_dir,
        &fixture_dir,
        &HALF_RESOLUTION_CASES,
        &HALF_RESOLUTION_PIXEL_PARITY_LIMITS,
        "half_resolution_frame_",
    )?;

    let exported_blend = root.path().join("unicode_text_round_trip.blend");
    let exported = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported_blend)
        .arg("--blender")
        .arg(&blender)
        .arg("--pack")
        .arg("--json")
        .output()?;
    successful_json(&exported, "pot export packed text-strip fonts")?;

    let round_trip = root.path().join("unicode_text_round_trip_project");
    init_project(&round_trip)?;
    let imported = import_blend(&round_trip, &exported_blend, &blender)?;
    assert_eq!(imported["result"]["losses"], serde_json::json!([]));
    inspect_fonts(&round_trip, &FRAME_CASES, &WRAP_WIDTH_CASES)?;

    let round_trip_render = root.path().join("unicode_text_round_trip_render");
    render_frames(&round_trip, &round_trip_render, "1:7")?;
    measure_blender_parity(&round_trip_render, &fixture_dir)?;
    Ok(())
}
