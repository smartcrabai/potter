//! Blender 5.2.2 Mask-node coverage compared through Potter's compositor CLI.
//!
//! `fixtures/native_oracle/mask_bezier_splines.py` renders each case in `mask_bezier_cases.json`
//! through a fixed-size compositor Mask node; the EXRs under `fixtures/native_oracle/mask_bezier/`
//! are its captures. This test applies the same cases to Potter's scene and renders them with
//! `pot render`.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this consumer uses a subset of the shared native oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{fs, path::Path};

use potter_core::mask::Mask;

use native_oracle::{TestResult, apply_operations, checked, pot, read_scene};
use potter_core::{image::decode_pixels, model::ImageColorspace};
use serde_json::{Value, json};
use tempfile::tempdir;

const CASES: &str = include_str!("fixtures/native_oracle/mask_bezier_cases.json");
/// Blender stores mask coverage as f32 and evaluates the spline in f32 (forward-differenced Bezier
/// segments, f32 mean-value weights); Potter evaluates in f64. The measured residual is 6e-6.
const COVERAGE_TOLERANCE: f64 = 1.0e-5;

const CAPTURES: [(&str, &[u8]); 10] = [
    (
        "cyclic_hard",
        include_bytes!("fixtures/native_oracle/mask_bezier/cyclic_hard.exr"),
    ),
    (
        "cyclic_weighted_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/cyclic_weighted_feather.exr"),
    ),
    (
        "cyclic_ccw_weighted_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/cyclic_ccw_weighted_feather.exr"),
    ),
    (
        "square_spline_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/square_spline_feather.exr"),
    ),
    (
        "feather_input_off",
        include_bytes!("fixtures/native_oracle/mask_bezier/feather_input_off.exr"),
    ),
    (
        "open_filled",
        include_bytes!("fixtures/native_oracle/mask_bezier/open_filled.exr"),
    ),
    (
        "keyframed_handles",
        include_bytes!("fixtures/native_oracle/mask_bezier/keyframed_handles.exr"),
    ),
    (
        "non_square_wide_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/non_square_wide_feather.exr"),
    ),
    (
        "non_square_tall_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/non_square_tall_feather.exr"),
    ),
    (
        "sixteen_nine_feather",
        include_bytes!("fixtures/native_oracle/mask_bezier/sixteen_nine_feather.exr"),
    ),
];

fn capture(name: &str) -> TestResult<&'static [u8]> {
    CAPTURES
        .iter()
        .find(|(case, _)| *case == name)
        .map(|(_, bytes)| *bytes)
        .ok_or_else(|| format!("no Blender capture for mask case {name}").into())
}

fn decode_exr(bytes: &[u8], label: &str, width: u32, height: u32) -> TestResult<Vec<[f64; 4]>> {
    assert_eq!(
        bytes.get(..4),
        Some(&[0x76, 0x2f, 0x31, 0x01][..]),
        "{label} is not an OpenEXR file"
    );
    let (decoded_width, decoded_height, pixels) = decode_pixels(bytes, ImageColorspace::Linear)?;
    assert_eq!(
        (decoded_width, decoded_height),
        (width, height),
        "{label} dimensions"
    );
    Ok(pixels)
}

/// The `mask.create` spline for one case, with the case's pixel points and handles filled in.
fn spline_value(case: &Value) -> TestResult<Value> {
    let mut points = Vec::new();
    for point in case["points"].as_array().ok_or("case points are missing")? {
        let mut point = point.clone();
        if point.get("keyframes").is_none() {
            point["keyframes"] = json!([]);
        }
        for handle in ["handle_left", "handle_right"] {
            if point.get(handle).is_none() {
                point[handle] = point["position"].clone();
            }
        }
        points.push(point);
    }
    Ok(json!({
        "points": points,
        "feather": case["feather"],
        "cyclic": case["cyclic"]
    }))
}

fn case_dimension(case: &Value, defaults: &Value, name: &str) -> TestResult<u32> {
    let value = case
        .get(name)
        .and_then(Value::as_u64)
        .or_else(|| defaults.get(name).and_then(Value::as_u64))
        .ok_or_else(|| format!("case {name} is missing"))?;
    Ok(u32::try_from(value)?)
}

fn render_case(
    project: &Path,
    root: &Path,
    name: &str,
    frame: u32,
    width: u32,
    height: u32,
) -> TestResult<Vec<[f64; 4]>> {
    let output_dir = root.join(format!("render_{name}"));
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&output_dir)
        .args(["--frames", &format!("{frame}:{frame}"), "--json"]);
    checked(&mut command, "Potter compositor mask EXR render")?;
    let bytes = fs::read(output_dir.join(format!("frame_{frame:04}.exr")))?;
    decode_exr(&bytes, "Potter compositor mask output", width, height)
}

fn apply_case(
    project: &Path,
    root: &Path,
    case: &Value,
    width: u32,
    height: u32,
) -> TestResult<()> {
    let use_feather = case["use_feather"]
        .as_bool()
        .ok_or("case use_feather is missing")?;
    let spline = spline_value(case)?;
    let operations = json!([
        {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
        {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
        {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":width,"resolution_y":height,"samples":1}},
        {"op":"mask.create","id":"mask_main","splines":[spline.clone()]},
        {"op":"graph.create","id":"graph_comp","kind":"compositor"},
        {"op":"compositor.node_add","graph":"graph_comp","id":"mask","type":"CompositorNodeMask","properties":{"mask_id":"mask_main"},"inputs":{"Size Source":"Fixed","Size X":width,"Size Y":height,"Feather":use_feather}},
        {"op":"compositor.node_add","graph":"graph_comp","id":"composite","type":"CompositorNodeComposite"},
        {"op":"compositor.link","graph":"graph_comp","from_node":"mask","from_socket":"Mask","to_node":"composite","to_socket":"Image"},
        {"op":"compositor.enable","target":{"id":"scene_main"},"set":{"enabled":true,"graph":"graph_comp"}}
    ]);
    let result = apply_operations(project, root, operations)?;
    assert_eq!(result["result"]["committed"], true, "mask scene apply");
    let persisted: Mask =
        serde_json::from_value(read_scene(project)?["masks"]["mask_main"].clone())?;
    let expected: Mask = serde_json::from_value(json!({"splines": [spline]}))?;
    assert_eq!(persisted, expected, "persisted Bezier mask");
    Ok(())
}

#[test]
fn mask_bezier_splines_match_blender_compositor_mask_node() -> TestResult {
    let Some(_blender) = blender_checked::blender_executable() else {
        if std::env::var_os("POTTER_REQUIRE_BLENDER").is_some() {
            return Err("Blender is required for the mask Bezier spline oracle".into());
        }
        eprintln!("skipping mask Bezier spline oracle; Blender is unavailable");
        return Ok(());
    };
    let cases: Value = serde_json::from_str(CASES)?;
    let root = tempdir()?;
    for case in cases["cases"].as_array().ok_or("cases array is missing")? {
        let name = case["name"].as_str().ok_or("case name is missing")?;
        let frame = u32::try_from(case["frame"].as_u64().ok_or("case frame is missing")?)?;
        let width = case_dimension(case, &cases, "width")?;
        let height = case_dimension(case, &cases, "height")?;
        let project = root.path().join(format!("scene_{name}"));
        let mut init = pot();
        init.arg("init").arg(&project).arg("--json");
        checked(&mut init, "Potter project init")?;
        apply_case(&project, root.path(), case, width, height)?;

        let actual = render_case(&project, root.path(), name, frame, width, height)?;
        let expected = decode_exr(capture(name)?, &format!("Blender {name}"), width, height)?;
        let mut maximum_error = 0.0_f64;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let error = (actual - expected).abs();
                maximum_error = maximum_error.max(error);
                assert!(
                    error <= COVERAGE_TOLERANCE,
                    "{name} pixel {index} channel {channel}: Potter={actual} Blender={expected} tolerance={COVERAGE_TOLERANCE}"
                );
            }
        }
        eprintln!("MASK_ORACLE {name} max_abs_diff={maximum_error:.8}");
    }
    Ok(())
}
