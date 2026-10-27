use std::{
    env,
    error::Error,
    fs,
    io::{self, BufReader},
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

const TOLERANCE: f64 = 1.0e-4;
const FRAMES: [f64; 11] = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
type ClampedChannel = (
    &'static str,
    &'static str,
    &'static [&'static str],
    Option<usize>,
    f64,
);
type Channel = (
    &'static str,
    &'static str,
    &'static [&'static str],
    Option<usize>,
);

// The paths name Blender F-Curves; inspect paths follow Potter's consumer-facing data model.
const CHANNELS: &[Channel] = &[
    (
        "More Channels Camera",
        "camera.sensor_width_mm",
        &["camera", "sensor_width_mm"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.sensor_height_mm",
        &["camera", "sensor_height_mm"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.ortho_scale",
        &["camera", "ortho_scale"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.clip_start",
        &["camera", "clip_start"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.clip_end",
        &["camera", "clip_end"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.shift_x",
        &["camera", "shift"],
        Some(0),
    ),
    (
        "More Channels Camera",
        "camera.shift_y",
        &["camera", "shift"],
        Some(1),
    ),
    (
        "More Channels Camera",
        "camera.focus_distance",
        &["camera", "focus_distance"],
        None,
    ),
    (
        "More Channels Camera",
        "camera.aperture_fstop",
        &["camera", "f_stop"],
        None,
    ),
    (
        "More Channels Spot",
        "light.color[0]",
        &["light", "color"],
        Some(0),
    ),
    (
        "More Channels Spot",
        "light.color[1]",
        &["light", "color"],
        Some(1),
    ),
    (
        "More Channels Spot",
        "light.color[2]",
        &["light", "color"],
        Some(2),
    ),
    (
        "More Channels Spot",
        "light.spot_size",
        &["light", "spot_size"],
        None,
    ),
    (
        "More Channels Driven Spot",
        "light.spot_size",
        &["light", "spot_size"],
        None,
    ),
    (
        "More Channels Spot",
        "light.spot_blend",
        &["light", "spot_blend"],
        None,
    ),
    (
        "More Channels Spot",
        "light.shadow_soft_size",
        &["light", "radius"],
        None,
    ),
    (
        "More Channels Area",
        "light.area_size",
        &["light", "area_size"],
        None,
    ),
    (
        "More Channels Area",
        "light.area_size_y",
        &["light", "area_size_y"],
        None,
    ),
    (
        "More Channels Sun",
        "light.sun_angle",
        &["light", "sun_angle"],
        None,
    ),
    (
        "More Channels Visibility",
        "hide_viewport",
        &["visible"],
        None,
    ),
    (
        "More Channels Visibility",
        "hide_render",
        &["render_visible"],
        None,
    ),
    (
        "More Channels Visibility",
        "[\"gain\"]",
        &["properties", "gain"],
        None,
    ),
    (
        "More Channels Visibility",
        "[\"vector\"][1]",
        &["properties", "vector"],
        Some(1),
    ),
    (
        "More Channels Visibility",
        "[\"steps\"]",
        &["properties", "steps"],
        None,
    ),
    (
        "More Channels Visibility",
        "[\"enabled\"]",
        &["properties", "enabled"],
        None,
    ),
    (
        "More Channels NLA Object",
        "[\"nla_steps\"]",
        &["properties", "nla_steps"],
        None,
    ),
    (
        "More Channels NLA Object",
        "[\"nla_default_flag\"]",
        &["properties", "nla_default_flag"],
        None,
    ),
    (
        "More Channels NLA Object",
        "[\"nla_default_steps\"]",
        &["properties", "nla_default_steps"],
        None,
    ),
    (
        "More Channels Gain Light",
        "light.energy",
        &["light", "energy"],
        None,
    ),
];

// Blender blends a Boolean NLA Replace strip from its False default, so influence 0.6 can only reach
// 0.6, which thresholds to False. Every frame must match that oracle, which catches a rounded
// Boolean. A changed-from-base check cannot apply here, so these channels are exempt from that guard.
const BOUNDARY_CHANNELS: &[Channel] = &[(
    "More Channels NLA Object",
    "[\"nla_flag\"]",
    &["properties", "nla_flag"],
    None,
)];

const CLAMPED_CHANNELS: &[ClampedChannel] = &[
    (
        "More Channels Camera",
        "camera.sensor_width_mm",
        &["camera", "sensor_width_mm"],
        None,
        1.0,
    ),
    (
        "More Channels Camera",
        "camera.ortho_scale",
        &["camera", "ortho_scale"],
        None,
        0.0,
    ),
    (
        "More Channels Camera",
        "camera.clip_start",
        &["camera", "clip_start"],
        None,
        9.999_999_974_752_427e-7,
    ),
    (
        "More Channels Camera",
        "camera.focus_distance",
        &["camera", "focus_distance"],
        None,
        0.0,
    ),
    (
        "More Channels Camera",
        "camera.aperture_fstop",
        &["camera", "f_stop"],
        None,
        0.0,
    ),
    (
        "More Channels Visibility",
        "[\"gain\"]",
        &["properties", "gain"],
        None,
        0.0,
    ),
    (
        "More Channels Visibility",
        "[\"vector\"][1]",
        &["properties", "vector"],
        Some(1),
        0.0,
    ),
    (
        "More Channels Visibility",
        "[\"steps\"]",
        &["properties", "steps"],
        None,
        -2.0,
    ),
    (
        "More Channels Spot",
        "light.color[0]",
        &["light", "color"],
        Some(0),
        0.0,
    ),
    (
        "More Channels Spot",
        "light.spot_size",
        &["light", "spot_size"],
        None,
        0.017_453_292_384_743_69,
    ),
    (
        "More Channels Spot",
        "light.spot_blend",
        &["light", "spot_blend"],
        None,
        0.0,
    ),
    (
        "More Channels Spot",
        "light.shadow_soft_size",
        &["light", "radius"],
        None,
        0.0,
    ),
    (
        "More Channels Area",
        "light.area_size",
        &["light", "area_size"],
        None,
        0.0,
    ),
    (
        "More Channels Area",
        "light.area_size_y",
        &["light", "area_size_y"],
        None,
        0.0,
    ),
    (
        "More Channels Sun",
        "light.sun_angle",
        &["light", "sun_angle"],
        None,
        0.0,
    ),
];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked_output(mut command: Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn blender_522() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping extended F-Curve channels: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = checked_output(command, "checking Blender version")?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
        "POTTER_REQUIRE_BLENDER is set but this test requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping extended F-Curve channels: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn pot_json(command: Command, label: &str) -> Result<Value, Box<dyn Error>> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn import_blend(
    project: &Path,
    blend: &Path,
    blender: &Path,
    label: &str,
) -> Result<Value, Box<dyn Error>> {
    let mut command = pot();
    command
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
            "--blender",
        ])
        .arg(blender)
        .arg("--allow-lossy")
        .arg("--json");
    pot_json(command, label)
}

fn inspect_frame(project: &Path, frame: f64) -> Result<Value, Box<dyn Error>> {
    let mut command = pot();
    let frame_text = frame.to_string();
    command
        .arg("inspect")
        .arg(project)
        .arg("--frame")
        .arg(frame_text)
        .arg("--json");
    pot_json(
        command,
        &format!("inspecting F-Curve channels at frame {frame}"),
    )
}

fn item<'a>(inspection: &'a Value, name: &str, frame: f64) -> Result<&'a Value, Box<dyn Error>> {
    inspection["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["name"] == name))
        .ok_or_else(|| {
            io::Error::other(format!("inspect omitted `{name}` at frame {frame}")).into()
        })
}

fn field<'a>(value: &'a Value, path: &[&str], component: Option<usize>) -> Option<&'a Value> {
    let mut current = value;
    for part in path {
        current = current.get(*part)?;
    }
    component.map_or(Some(current), |index| current.get(index))
}

fn matches_oracle(actual: &Value, expected: &Value, tolerance: f64) -> bool {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(actual), Some(expected)) => (actual - expected).abs() <= tolerance,
        _ => actual == expected,
    }
}

fn assert_exported_nla_matches_blender(
    blend: &Path,
    blender: &Path,
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let frames = serde_json::to_string(&FRAMES)?;
    let script = format!(
        r#"
import bpy, json, math, struct
frames = {frames}
scene = bpy.context.scene
obj = bpy.data.objects.get("More Channels NLA Object")
if obj is None:
    raise RuntimeError("exported NLA fixture object is missing")
expected_influence = struct.unpack("f", struct.pack("f", 0.6))[0]
for track_name in ("More Channels NLA Flag", "More Channels NLA Steps"):
    track = next((track for track in obj.animation_data.nla_tracks
                  if track.name == track_name), None)
    if track is None or len(track.strips) != 1:
        raise RuntimeError("exported NLA track is missing or has unexpected strips: " + track_name)
    strip = track.strips[0]
    if track_name == "More Channels NLA Steps" and float(strip.blend_in) != 3.0:
        raise RuntimeError("exported NLA steps strip must retain its three-frame blend-in")
    if not strip.use_animated_influence:
        raise RuntimeError("exported NLA strip does not enable animated influence: " + track_name)
    curves = [curve for curve in strip.fcurves
              if curve.data_path == "influence" and curve.array_index == 0]
    if len(curves) != 1:
        raise RuntimeError("exported NLA strip must have one influence curve: " + track_name)
    curve = curves[0]
    points = sorted(curve.keyframe_points, key=lambda point: point.co.x)
    if len(points) != 2:
        raise RuntimeError("exported NLA influence curve must have two range keys: " + track_name)
    if float(strip.influence) != expected_influence:
        raise RuntimeError("exported NLA strip influence is not Blender's f32 0.6: " + track_name)
    if any(float(point.co.y) != expected_influence or point.interpolation != "CONSTANT"
           for point in points):
        raise RuntimeError("exported NLA influence keys must be constant at 0.6: " + track_name)
    if [float(point.co.x) for point in points] != [
            float(strip.frame_start), float(strip.frame_end)]:
        raise RuntimeError("exported NLA influence keys must span the strip: " + track_name)
values = {{}}
for frame in frames:
    whole = math.floor(frame)
    scene.frame_set(whole, subframe=frame - whole)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    values[f"{{frame:g}}"] = {{
        "nla_flag": bool(evaluated["nla_flag"]),
        "nla_steps": int(evaluated["nla_steps"]),
    }}
print("POTTER_NLA_ROUNDTRIP=" + json.dumps(values, sort_keys=True))
"#
    );
    let mut command = Command::new(blender);
    command
        .arg("--background")
        .arg(blend)
        .arg("--python-expr")
        .arg(script);
    let output = checked_output(command, "evaluating exported NLA strips in Blender")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NLA_ROUNDTRIP="))
        .ok_or_else(|| io::Error::other("Blender omitted exported NLA round-trip values"))?;
    let actual: Value = serde_json::from_str(values)?;
    for frame in FRAMES {
        let key = frame.to_string();
        let expected_properties =
            &expected["frames"][&key]["More Channels NLA Object"]["properties"];
        for property in ["nla_flag", "nla_steps"] {
            assert!(
                matches_oracle(
                    &actual[&key][property],
                    &expected_properties[property],
                    tolerance
                ),
                "exported Blender NLA {property} at frame {frame}: oracle expected {}, exported file evaluated to {}",
                expected_properties[property],
                actual[&key][property],
            );
        }
    }
    Ok(())
}
fn assert_channels_match(
    project: &Path,
    expected: &Value,
    tolerance: f64,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let mut changed_from_base = vec![false; CHANNELS.len()];
    for frame in FRAMES {
        let inspection = inspect_frame(project, frame)?;
        for (index, channel) in CHANNELS.iter().enumerate() {
            changed_from_base[index] |=
                assert_channel_at_frame(&inspection, expected, frame, channel, tolerance, label)?;
        }
        for channel in BOUNDARY_CHANNELS {
            assert_channel_at_frame(&inspection, expected, frame, channel, tolerance, label)?;
        }
    }
    for (changed, (object, channel, _, _)) in changed_from_base.iter().zip(CHANNELS) {
        assert!(
            changed,
            "{label}: {object}.{channel} never differs from its base state; an absent or dropped F-Curve could pass"
        );
    }
    Ok(())
}

/// Checks one channel against Blender at one frame and reports whether Blender differs from base.
fn assert_channel_at_frame(
    inspection: &Value,
    expected: &Value,
    frame: f64,
    channel: &Channel,
    tolerance: f64,
    label: &str,
) -> Result<bool, Box<dyn Error>> {
    let (object, channel_name, path, component) = *channel;
    let frame_key = frame.to_string();
    let expected_frame = &expected["frames"][&frame_key];
    let actual_item = item(inspection, object, frame)?;
    let expected_item = &expected_frame[object];
    let base_item = &expected["base"][object];
    let actual = field(actual_item, path, component).ok_or_else(|| {
        io::Error::other(format!(
            "inspect omitted {object}.{channel_name} at frame {frame}"
        ))
    })?;
    let oracle = field(expected_item, path, component).ok_or_else(|| {
        io::Error::other(format!(
            "Blender oracle omitted {object}.{channel_name} at frame {frame}"
        ))
    })?;
    assert!(
        matches_oracle(actual, oracle, tolerance),
        "{label}: {object}.{channel_name} at frame {frame}: Blender expected {oracle}, Potter inspect returned {actual}"
    );
    let base = field(base_item, path, component).ok_or_else(|| {
        io::Error::other(format!(
            "Blender oracle omitted base {object}.{channel_name}"
        ))
    })?;
    Ok(!matches_oracle(oracle, base, tolerance))
}

fn assert_native_clamps(
    project: &Path,
    expected: &Value,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let frame = 7.0;
    let frame_key = frame.to_string();
    let inspection = inspect_frame(project, frame)?;
    for (object, channel, path, component, minimum) in CLAMPED_CHANNELS {
        let actual_item = item(&inspection, object, frame)?;
        let oracle = field(&expected["frames"][&frame_key][*object], path, *component)
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                io::Error::other(format!("Blender clamp oracle omitted {object}.{channel}"))
            })?;
        let actual = field(actual_item, path, *component)
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                io::Error::other(format!("inspect omitted clamped {object}.{channel}"))
            })?;
        assert_eq!(
            oracle, *minimum,
            "{label}: Blender oracle did not hit the {object}.{channel} lower bound"
        );
        assert!(
            actual >= *minimum && actual - *minimum <= 1.0e-9,
            "{label}: {object}.{channel} missed the Blender lower bound {minimum}: got {actual}"
        );
    }
    Ok(())
}

fn image_coverage(path: &Path) -> Result<f64, Box<dyn Error>> {
    let mut reader = png::Decoder::new(BufReader::new(fs::File::open(path)?)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let channels = info.color_type.samples();
    if channels < 3 {
        return Err(io::Error::other("PNG image has fewer than three color channels").into());
    }
    let pixels = &pixels[..info.buffer_size()];
    let background = &pixels[..3];
    let changed = pixels
        .chunks_exact(channels)
        .filter(|pixel| pixel[..3] != *background)
        .count();
    Ok(changed as f64 / (f64::from(info.width) * f64::from(info.height)))
}

fn png_rgb_at(path: &Path, x: usize, y: usize) -> Result<[u8; 3], Box<dyn Error>> {
    let mut reader = png::Decoder::new(BufReader::new(fs::File::open(path)?)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let channels = info.color_type.samples();
    if channels < 3 || x >= info.width as usize || y >= info.height as usize {
        return Err(
            io::Error::other("PNG sample is outside the image or has too few channels").into(),
        );
    }
    let offset = (y * info.width as usize + x) * channels;
    Ok([pixels[offset], pixels[offset + 1], pixels[offset + 2]])
}

fn preview_image(
    project: &Path,
    camera_id: &str,
    frame: u32,
    mode: &str,
    output_dir: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut command = pot();
    command
        .arg("preview")
        .arg(project)
        .args([
            "--camera", camera_id, "--mode", mode, "--size", "64", "--frame",
        ])
        .arg(frame.to_string())
        .arg("--out")
        .arg(output_dir)
        .arg("--json");
    let preview = pot_json(
        command,
        &format!("previewing animated channels at frame {frame}"),
    )?;
    let warnings = preview["result"]["warnings"]
        .as_array()
        .ok_or("preview warnings are missing")?;
    assert!(
        warnings.iter().any(|warning| {
            warning["feature_id"] == "modifier.uv_project.animated_camera_parameters"
        }),
        "preview did not report the animated-camera UV Project refusal: {warnings:?}"
    );
    preview["result"]["previews"]
        .as_array()
        .and_then(|previews| previews.first())
        .and_then(|preview| preview["image"].as_str())
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("preview omitted its rendered image").into())
}

fn preview_coverage(
    project: &Path,
    camera_id: &str,
    frame: u32,
    output_dir: &Path,
) -> Result<f64, Box<dyn Error>> {
    let image = preview_image(project, camera_id, frame, "solid", output_dir)?;
    image_coverage(&image)
}

fn assert_camera_visibility_preview(
    project: &Path,
    camera_id: &str,
    expected: &Value,
    output_root: &Path,
) -> Result<(), Box<dyn Error>> {
    for frame in [1, 4] {
        let frame_text = frame.to_string();
        let expected_coverage = expected["render_coverage"][&frame_text]
            .as_f64()
            .ok_or("Blender render coverage oracle is missing")?;
        let output_dir = output_root.join(format!("preview_{frame}"));
        let actual = preview_coverage(project, camera_id, frame, &output_dir)?;
        let tolerance = if frame == 1 { 0.06 } else { 0.01 };
        assert!(
            (actual - expected_coverage).abs() <= tolerance,
            "camera/visibility preview at frame {frame}: Blender coverage {expected_coverage}, Potter coverage {actual}, tolerance {tolerance}"
        );
        if frame == 1 {
            assert!(expected_coverage > 0.0, "Blender preview oracle is empty");
        }
    }
    Ok(())
}

fn rendered_coverage(project: &Path, frame: u32, output_dir: &Path) -> Result<f64, Box<dyn Error>> {
    let mut command = pot();
    command
        .arg("render")
        .arg(project)
        .args(["--engine", "realtime", "--format", "png", "--frames"])
        .arg(format!("{frame}:{frame}"))
        .arg("--out")
        .arg(output_dir)
        .arg("--json");
    pot_json(command, &format!("rendering visibility at frame {frame}"))?;
    image_coverage(&output_dir.join(format!("frame_{frame:04}.png")))
}

fn remove_uv_project_for_render(
    project: &Path,
    operations_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let revision = document["revision"]
        .as_u64()
        .ok_or("project revision is missing")?;
    let nodes = document["nodes"]
        .as_object()
        .ok_or("project nodes are missing")?;
    let (node_id, node) = nodes
        .iter()
        .find(|(_, node)| node["name"] == "Animated Camera UV Project Subject")
        .ok_or("UV Project subject is missing")?;
    let modifier_id = node["modifiers"]
        .as_array()
        .and_then(|modifiers| modifiers.first())
        .and_then(|modifier| modifier["id"].as_str())
        .ok_or("UV Project modifier ID is missing")?;
    let operations = json!({
        "schema_version": 1,
        "base_revision": revision,
        "operations": [{
            "op": "modifier.delete",
            "target": {"id": node_id},
            "id": modifier_id
        }]
    });
    fs::write(operations_path, serde_json::to_vec(&operations)?)?;
    let mut command = pot();
    command
        .arg("apply")
        .arg(project)
        .arg("--file")
        .arg(operations_path)
        .arg("--json");
    let applied = pot_json(command, "removing refused UV Project before render")?;
    assert_eq!(applied["result"]["committed"], true);
    Ok(())
}

fn assert_viewport_and_render_visibility(
    project: &Path,
    camera_id: &str,
    expected: &Value,
    output_root: &Path,
) -> Result<(), Box<dyn Error>> {
    let expected_frame = &expected["frames"]["2"]["More Channels Visibility"];
    assert_eq!(expected_frame["visible"].as_bool(), Some(true));
    assert_eq!(expected_frame["render_visible"].as_bool(), Some(false));
    let preview_oracle = expected["preview_coverage"]["2"]
        .as_f64()
        .ok_or("Blender viewport coverage oracle is missing")?;
    let render_oracle = expected["render_coverage"]["2"]
        .as_f64()
        .ok_or("Blender render coverage oracle is missing")?;
    let preview = preview_coverage(
        project,
        camera_id,
        2,
        &output_root.join("preview_viewport_2"),
    )?;
    assert!(
        (preview - preview_oracle).abs() <= 0.06,
        "viewport preview at frame 2: Blender coverage {preview_oracle}, Potter coverage {preview}"
    );
    assert!(
        preview > render_oracle + 0.05,
        "hide_render=true must not hide the viewport preview when hide_viewport=false: preview={preview}, render oracle={render_oracle}"
    );
    remove_uv_project_for_render(
        project,
        &output_root.join("remove_uv_project_before_render.json"),
    )?;
    let render = rendered_coverage(project, 2, &output_root.join("render_hidden_2"))?;
    assert!(
        (render - render_oracle).abs() <= 0.06,
        "render at frame 2: Blender coverage {render_oracle}, Potter coverage {render}"
    );
    assert!(
        render < preview - 0.05,
        "hide_render=true must hide the visibility mesh from rendering: preview={preview}, render={render}"
    );
    Ok(())
}

fn assert_light_beauty_preview(
    project: &Path,
    camera_id: &str,
    expected: &Value,
    output_root: &Path,
) -> Result<(), Box<dyn Error>> {
    let luminance = |rgb: [f64; 3]| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let mut expected_luminance = [0.0; 2];
    let mut actual_luminance = [0.0; 2];
    for (index, frame) in [1, 4].into_iter().enumerate() {
        let frame_key = frame.to_string();
        let oracle = &expected["light_samples"][&frame_key];
        let pixel = oracle["pixel"]
            .as_array()
            .ok_or("Blender light sample pixel coordinate is missing")?;
        let x = usize::try_from(pixel[0].as_u64().ok_or("Blender pixel X is missing")?)?;
        let y = usize::try_from(pixel[1].as_u64().ok_or("Blender pixel Y is missing")?)?;
        let rgb = oracle["rgb"]
            .as_array()
            .ok_or("Blender light sample RGB is missing")?;
        expected_luminance[index] = luminance([
            rgb[0].as_f64().ok_or("Blender red sample is missing")?,
            rgb[1].as_f64().ok_or("Blender green sample is missing")?,
            rgb[2].as_f64().ok_or("Blender blue sample is missing")?,
        ]);
        let image = preview_image(
            project,
            camera_id,
            frame,
            "beauty",
            &output_root.join(format!("beauty_light_{frame}")),
        )?;
        let pixel = png_rgb_at(&image, x, y)?;
        actual_luminance[index] = luminance(pixel.map(|component| f64::from(component) / 255.0));
    }
    assert!(
        expected_luminance[1] > expected_luminance[0] + 0.03,
        "Blender light oracle does not discriminate the spot animation: {expected_luminance:?}"
    );
    assert!(
        actual_luminance[1] > actual_luminance[0] + 0.02,
        "Potter Beauty pixels did not respond to the animated light color/spot channels: {actual_luminance:?}"
    );
    Ok(())
}

#[test]
fn extended_camera_light_visibility_and_custom_property_curves_match_blender_oracle()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fcurve_more_channels");
    let expected: Value = serde_json::from_slice(&std::fs::read(fixture.join("expected.json"))?)?;
    assert!(
        expected["blender_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2")),
        "oracle must be generated with Blender 5.2.2, found {}",
        expected["blender_version"]
    );
    let tolerance = expected["tolerance"].as_f64().unwrap_or(TOLERANCE);
    assert_eq!(
        CHANNELS.len() + BOUNDARY_CHANNELS.len(),
        30,
        "the complete new-channel corpus must be checked"
    );

    let directory = tempdir()?;
    let project = directory.path().join("project");
    let mut init = pot();
    init.arg("init").arg(&project).arg("--json");
    checked_output(init, "initializing F-Curve channel project")?;

    let source_blend = fixture.join("fcurve_more_channels.blend");
    let imported = import_blend(
        &project,
        &source_blend,
        &blender,
        "importing Blender F-Curve channel fixture",
    )?;

    let import_losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import losses are missing")?;
    assert_eq!(
        import_losses.len(),
        1,
        "unexpected Blender import losses: {import_losses:?}"
    );
    assert_eq!(
        import_losses[0]["feature_id"],
        "modifier.uv_project.animated_camera_parameters"
    );
    assert_channels_match(&project, &expected, tolerance, "source import")?;
    assert_native_clamps(&project, &expected, "source import")?;
    let source_scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (nla_node_id, nla_node) = source_scene["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"] == "More Channels NLA Object")
        })
        .map(|(id, node)| (id.clone(), node))
        .ok_or("imported NLA fixture object is missing")?;
    let steps_track = nla_node["nla_tracks"]
        .as_array()
        .and_then(|tracks| {
            tracks
                .iter()
                .find(|track| track["name"] == "More Channels NLA Steps")
        })
        .ok_or("imported NLA steps track is missing")?;
    let steps_strip = steps_track["strips"]
        .as_array()
        .and_then(|strips| (strips.len() == 1).then(|| &strips[0]))
        .ok_or("imported NLA steps strip is missing")?;
    assert_eq!(steps_strip["blend_in"].as_f64(), Some(3.0));
    assert_eq!(steps_strip["use_animated_influence"].as_bool(), Some(true));
    assert!(
        steps_strip["influence"]
            .as_f64()
            .is_some_and(|influence| (influence - 0.6).abs() <= tolerance)
    );
    let steps_track_id = steps_track["id"]
        .as_str()
        .ok_or("imported NLA steps track ID is missing")?
        .to_owned();
    let steps_strip_id = steps_strip["id"]
        .as_str()
        .ok_or("imported NLA steps strip ID is missing")?
        .to_owned();
    let blended_frame = inspect_frame(&project, 2.0)?;
    assert_eq!(
        field(
            item(&blended_frame, "More Channels NLA Object", 2.0)?,
            &["properties", "nla_steps"],
            None
        )
        .and_then(Value::as_i64),
        Some(1),
        "animated influence must not be multiplied by the blend-in fade"
    );

    let set_influence_operations = json!({
        "schema_version": 1,
        "base_revision": source_scene["revision"],
        "operations": [
            {
                "op": "nla.strip_update",
                "target": {"id": nla_node_id},
                "track": steps_track_id,
                "strip": steps_strip_id,
                "set": {"influence": 1.0}
            },
            {
                "op": "nla.strip_update",
                "target": {"id": nla_node_id},
                "track": steps_track_id,
                "strip": steps_strip_id,
                "set": {"influence": 0.6}
            }
        ]
    });
    let set_influence_path = directory.path().join("set_nla_influence.json");
    fs::write(
        &set_influence_path,
        serde_json::to_vec(&set_influence_operations)?,
    )?;
    let mut set_influence = pot();
    set_influence
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(set_influence_path)
        .arg("--json");
    let updated = pot_json(set_influence, "setting NLA strip influence")?;
    assert_eq!(updated["result"]["committed"], true);
    let updated_frame = inspect_frame(&project, 2.0)?;
    assert_eq!(
        field(
            item(&updated_frame, "More Channels NLA Object", 2.0)?,
            &["properties", "nla_steps"],
            None
        )
        .and_then(Value::as_i64),
        Some(1),
        "setting a non-unit influence must enable animated influence without a fade"
    );

    let source_inspection = inspect_frame(&project, 1.0)?;
    let camera_id = item(&source_inspection, "More Channels Camera", 1.0)?["id"]
        .as_str()
        .ok_or("imported camera ID is missing")?
        .to_owned();
    assert_camera_visibility_preview(&project, &camera_id, &expected, directory.path())?;
    assert_light_beauty_preview(&project, &camera_id, &expected, directory.path())?;

    let exported_blend = directory.path().join("round_trip.blend");
    let mut export = pot();
    export
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported_blend)
        .arg("--blender")
        .arg(&blender)
        .arg("--json");
    let exported = pot_json(export, "exporting F-Curve channel fixture")?;
    assert_eq!(exported["result"]["losses"], Value::Array(Vec::new()));
    assert_exported_nla_matches_blender(&exported_blend, &blender, &expected, tolerance)?;

    let round_trip_project = directory.path().join("round_trip_project");
    let mut init = pot();
    init.arg("init").arg(&round_trip_project).arg("--json");
    checked_output(init, "initializing F-Curve round-trip project")?;
    let reimported = import_blend(
        &round_trip_project,
        &exported_blend,
        &blender,
        "reimporting F-Curve channel fixture",
    )?;
    let round_trip_losses = reimported["result"]["losses"]
        .as_array()
        .ok_or("round-trip import losses are missing")?;
    assert_eq!(
        round_trip_losses.len(),
        1,
        "unexpected round-trip Blender import losses: {round_trip_losses:?}"
    );
    assert_eq!(
        round_trip_losses[0]["feature_id"],
        "modifier.uv_project.animated_camera_parameters"
    );
    assert_channels_match(
        &round_trip_project,
        &expected,
        tolerance,
        "round-trip import",
    )?;
    assert_native_clamps(&round_trip_project, &expected, "round-trip import")?;
    let round_trip_inspection = inspect_frame(&round_trip_project, 1.0)?;
    let round_trip_camera_id = item(&round_trip_inspection, "More Channels Camera", 1.0)?["id"]
        .as_str()
        .ok_or("round-trip camera ID is missing")?
        .to_owned();
    assert_viewport_and_render_visibility(
        &round_trip_project,
        &round_trip_camera_id,
        &expected,
        directory.path(),
    )?;
    let spot_id = item(&round_trip_inspection, "More Channels Spot", 1.0)?["id"]
        .as_str()
        .ok_or("round-trip spot light ID is missing")?
        .to_owned();
    let driven_spot_id = item(&round_trip_inspection, "More Channels Driven Spot", 1.0)?["id"]
        .as_str()
        .ok_or("round-trip driven spot light ID is missing")?
        .to_owned();
    let visibility_id = item(&round_trip_inspection, "More Channels Visibility", 1.0)?["id"]
        .as_str()
        .ok_or("round-trip visibility object ID is missing")?
        .to_owned();
    let scene_value: Value =
        serde_json::from_slice(&fs::read(round_trip_project.join("scene.json"))?)?;
    let revision = scene_value["revision"]
        .as_u64()
        .ok_or("round-trip project revision is missing")?;
    let delete_operations = json!({
        "schema_version": 1,
        "base_revision": revision,
        "operations": [{
            "op": "property.delete",
            "target": {"id": visibility_id},
            "name": "gain"
        }]
    });
    let delete_path = directory
        .path()
        .join("delete_animated_custom_property.json");
    fs::write(&delete_path, serde_json::to_vec(&delete_operations)?)?;
    let mut delete = pot();
    delete
        .arg("apply")
        .arg(&round_trip_project)
        .arg("--file")
        .arg(&delete_path)
        .arg("--json");
    let deleted = pot_json(delete, "deleting an animated custom property")?;
    assert_eq!(deleted["result"]["committed"], true);
    let deleted_inspection = inspect_frame(&round_trip_project, 1.0)?;
    let deleted_visibility = item(&deleted_inspection, "More Channels Visibility", 1.0)?;
    assert!(deleted_visibility["properties"].get("gain").is_none());
    let edited_export_path = directory
        .path()
        .join("deleted_custom_property_round_trip.blend");
    let mut export = pot();
    export
        .arg("export")
        .arg(&round_trip_project)
        .args(["--format", "blend", "--out"])
        .arg(&edited_export_path)
        .arg("--blender")
        .arg(&blender)
        .arg("--json");
    pot_json(
        export,
        "exporting after an animated custom property is deleted",
    )?;

    let light_scene_value: Value =
        serde_json::from_slice(&fs::read(round_trip_project.join("scene.json"))?)?;
    let light_revision = light_scene_value["revision"]
        .as_u64()
        .ok_or("round-trip project revision is missing after custom-property deletion")?;
    let light_operations = json!({
        "schema_version": 1,
        "base_revision": light_revision,
        "operations": [
            {
                "op": "light.update",
                "target": {"id": spot_id},
                "set": {"light_type": "point"}
            },
            {
                "op": "light.update",
                "target": {"id": driven_spot_id},
                "set": {"light_type": "point"}
            }
        ]
    });
    let light_path = directory.path().join("change_animated_light_types.json");
    fs::write(&light_path, serde_json::to_vec(&light_operations)?)?;
    let mut change_lights = pot();
    change_lights
        .arg("apply")
        .arg(&round_trip_project)
        .arg("--file")
        .arg(light_path)
        .arg("--json");
    let changed = pot_json(change_lights, "changing animated light types")?;
    assert_eq!(changed["result"]["committed"], true);
    let changed_inspection = inspect_frame(&round_trip_project, 1.0)?;
    let changed_spot = item(&changed_inspection, "More Channels Spot", 1.0)?;
    assert_eq!(changed_spot["light"]["light_type"], "point");
    assert!(
        matches_oracle(
            &changed_spot["light"]["spot_size"],
            &expected["frames"]["1"]["More Channels Spot"]["light"]["spot_size"],
            tolerance,
        ),
        "Blender continues evaluating the F-Curve stored on the SpotLight RNA wrapper"
    );
    let changed_driven_spot = item(&changed_inspection, "More Channels Driven Spot", 1.0)?;
    assert!(
        matches_oracle(
            &changed_driven_spot["light"]["spot_size"],
            &expected["frames"]["1"]["More Channels Driven Spot"]["light"]["spot_size"],
            tolerance,
        ),
        "Blender continues evaluating the driver stored on the SpotLight RNA wrapper"
    );

    Ok(())
}
