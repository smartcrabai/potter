//! Blender 5.2 `MESH_CACHE` oracle for PC2 and MDD caches, evaluated through the native `pot` CLI.
//!
//! The cache files are written here from the base mesh, so the oracle inputs are independent of the
//! Potter reader. Blender evaluates each case's modifier at its scene frames; `pot` applies the same
//! modifier in a persisted scene and inspects every frame. Positions must match per vertex.

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, process::Command};
use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const ORACLE_SCRIPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mesh_cache/blender_mesh_cache_oracle.py"
);
const TOLERANCE: f64 = 1.0e-5;
const FRAME_COUNT: usize = 6;
const MDD_TIMES: [f32; FRAME_COUNT] = [0.0, 0.1, 0.35, 0.4, 0.9, 1.5];
const AXES: [&str; 6] = ["POS_X", "NEG_X", "POS_Y", "NEG_Y", "POS_Z", "NEG_Z"];

/// Deterministic per-frame displacement with distinct X, Y and Z behavior per vertex.
fn cache_position(base: [f64; 3], frame: usize, index: usize) -> [f32; 3] {
    let frame_value = f64::from(u16::try_from(frame).unwrap_or(u16::MAX));
    let index_value = f64::from(u16::try_from(index).unwrap_or(u16::MAX));
    let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
    [
        base[0] + 0.12 * frame_value * (index_value + 1.0)
            - 0.03 * sign * frame_value * frame_value,
        base[1] + 0.08 * sign * (frame_value + index_value),
        base[2] + 0.25 * (frame_value * 0.5 + index_value * 0.3).sin() + 0.05 * index_value,
    ]
    .map(|value| value as f32)
}

fn cache_frames(base: &[[f64; 3]]) -> Vec<Vec<[f32; 3]>> {
    (0..FRAME_COUNT)
        .map(|frame| {
            base.iter()
                .enumerate()
                .map(|(index, point)| cache_position(*point, frame, index))
                .collect()
        })
        .collect()
}

fn write_pc2(path: &Path, frames: &[Vec<[f32; 3]>], start: f32, sampling: f32) -> TestResult {
    let count = frames.first().map_or(0, Vec::len);
    let mut bytes = b"POINTCACHE2\0".to_vec();
    bytes.extend_from_slice(&1_i32.to_le_bytes());
    bytes.extend_from_slice(&i32::try_from(count)?.to_le_bytes());
    bytes.extend_from_slice(&start.to_le_bytes());
    bytes.extend_from_slice(&sampling.to_le_bytes());
    bytes.extend_from_slice(&i32::try_from(frames.len())?.to_le_bytes());
    for frame in frames {
        for point in frame {
            for component in point {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn write_mdd(path: &Path, frames: &[Vec<[f32; 3]>], times: &[f32]) -> TestResult {
    let count = frames.first().map_or(0, Vec::len);
    let mut bytes = i32::try_from(frames.len())?.to_be_bytes().to_vec();
    bytes.extend_from_slice(&i32::try_from(count)?.to_be_bytes());
    for time in times {
        bytes.extend_from_slice(&time.to_be_bytes());
    }
    for frame in frames {
        for point in frame {
            for component in point {
                bytes.extend_from_slice(&component.to_be_bytes());
            }
        }
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value, file: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(file, serde_json::to_vec(operations)?)?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(scene)
        .args(["--file"])
        .arg(file)
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(format!("apply failed: {}", String::from_utf8_lossy(&output.stdout)).into());
    }
    Ok(())
}

fn inspect_positions(scene: &Path, id: &str, frame: f64) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(scene)
        .args(["--id", id, "--frame", &frame.to_string(), "--json"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "inspect {id} at {frame} failed: {}",
            String::from_utf8_lossy(&output.stdout)
        )
        .into());
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    value["result"]["items"][0]["evaluated_geometry"]["positions"]
        .as_array()
        .ok_or("evaluated positions missing from pot inspect")?
        .iter()
        .map(|point| {
            point
                .as_array()
                .ok_or("position is not a vector")?
                .iter()
                .map(|component| {
                    component
                        .as_f64()
                        .ok_or_else(|| "position component is not numeric".into())
                })
                .collect::<Result<Vec<f64>, Box<dyn Error>>>()
        })
        .collect()
}

fn case(id: &str, cache: &str, params: &Value, groups: &Value, frames: &[f64]) -> Value {
    json!({"id":id,"cache":cache,"params":params,"groups":groups,"frames":frames})
}

fn axis_cases() -> Vec<Value> {
    let mut cases = Vec::new();
    for forward in AXES {
        for up in AXES {
            if forward
                .trim_start_matches("NEG_")
                .trim_start_matches("POS_")
                == up.trim_start_matches("NEG_").trim_start_matches("POS_")
            {
                continue;
            }
            let name = format!("axis_{}_{}", forward.to_lowercase(), up.to_lowercase());
            let params = json!({
                "cache_format":"PC2","play_mode":"CUSTOM","time_mode":"FRAME","eval_frame":2.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0,
                "forward_axis":forward,"up_axis":up,"flip_axis":[false,false,false]
            });
            cases.push(case(&name, "pc2", &params, &json!({}), &[1.0]));
        }
    }
    cases
}

fn cases() -> Vec<Value> {
    let scene_frame = json!({
        "play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
        "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0
    });
    let mut cases = vec![
        case(
            "pc2_frame_linear",
            "pc2",
            &scene_frame,
            &json!({}),
            &[1.0, 2.25, 3.0, 4.75, 5.0, 7.0],
        ),
        case(
            "mdd_frame_linear",
            "mdd",
            &scene_frame,
            &json!({}),
            &[1.0, 2.25, 3.0, 4.75, 5.0, 7.0],
        ),
        case(
            "pc2_frame_none",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"NONE","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.4, 1.6, 2.5, 3.5],
        ),
        case(
            "mdd_frame_offset_scale",
            "mdd",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":2.0,"frame_scale":0.5,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[4.0, 6.5, 7.5, 9.0, 12.0, 2.0],
        ),
        case(
            "pc2_frame_offset_scale",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":2.0,"frame_scale":0.5,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[4.0, 6.5, 7.5, 9.0, 12.0, 2.0],
        ),
        case(
            "pc2_scene_seconds_quirk",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"TIME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[24.0, 36.0, 60.0, 96.0, 120.0, 200.0],
        ),
        case(
            "mdd_scene_seconds",
            "mdd",
            &json!({"play_mode":"SCENE","time_mode":"TIME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[0.0, 2.4, 4.0, 8.4, 12.0, 21.6, 24.0, 36.0],
        ),
        case(
            "mdd_scene_seconds_none",
            "mdd",
            &json!({"play_mode":"SCENE","time_mode":"TIME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"NONE","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[2.4, 4.0, 8.4, 12.0, 21.6, 24.0, 36.0],
        ),
        case(
            "mdd_scene_factor",
            "mdd",
            &json!({"play_mode":"SCENE","time_mode":"FACTOR","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[6.0, 12.0, 18.0, 24.0, 30.0],
        ),
        case(
            "pc2_scene_factor",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FACTOR","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[6.0, 12.0, 18.0, 24.0, 30.0],
        ),
        case(
            "mdd_custom_frame",
            "mdd",
            &json!({"play_mode":"CUSTOM","time_mode":"FRAME","eval_frame":3.25,
                "frame_start":5.0,"frame_scale":7.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0, 20.0],
        ),
        case(
            "pc2_custom_frame",
            "pc2",
            &json!({"play_mode":"CUSTOM","time_mode":"FRAME","eval_frame":2.6,
                "frame_start":5.0,"frame_scale":7.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0, 20.0],
        ),
        case(
            "mdd_custom_seconds",
            "mdd",
            &json!({"play_mode":"CUSTOM","time_mode":"TIME","eval_time":0.2,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0],
        ),
        case(
            "pc2_custom_seconds",
            "pc2",
            &json!({"play_mode":"CUSTOM","time_mode":"TIME","eval_time":1.1,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0],
        ),
        case(
            "mdd_custom_factor",
            "mdd",
            &json!({"play_mode":"CUSTOM","time_mode":"FACTOR","eval_factor":0.35,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0],
        ),
        case(
            "pc2_custom_factor_last",
            "pc2",
            &json!({"play_mode":"CUSTOM","time_mode":"FACTOR","eval_factor":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[1.0],
        ),
        case(
            "pc2_influence_half",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.5}),
            &json!({}),
            &[2.5],
        ),
        case(
            "pc2_integrate_frame",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"INTEGRATE","factor":0.8,
                "forward_axis":"POS_X","up_axis":"POS_Z"}),
            &json!({}),
            &[2.5],
        ),
        case(
            "mdd_integrate_factor_group",
            "mdd",
            &json!({"play_mode":"CUSTOM","time_mode":"FACTOR","eval_factor":0.35,
                "interpolation":"LINEAR","deform_mode":"INTEGRATE","factor":0.8,
                "vertex_group":"A","forward_axis":"POS_X","up_axis":"POS_Z"}),
            &json!({"A":[[0,0.1],[1,0.4],[2,0.7],[3,1.0]]}),
            &[1.0],
        ),
        case(
            "pc2_group_weights",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.9,
                "vertex_group":"A"}),
            &json!({"A":[[0,0.2],[2,0.6],[5,1.0]]}),
            &[2.5],
        ),
        case(
            "pc2_group_weights_inverted",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.9,
                "vertex_group":"A","invert_vertex_group":true}),
            &json!({"A":[[0,0.2],[2,0.6],[5,1.0]]}),
            &[2.5],
        ),
        case(
            "pc2_group_missing",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.6,
                "vertex_group":"Missing","invert_vertex_group":true}),
            &json!({"A":[[0,0.2]]}),
            &[2.5],
        ),
        case(
            "pc2_group_empty_inverted",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.6,
                "vertex_group":"Empty","invert_vertex_group":true}),
            &json!({"Empty":[],"Other":[[1,0.5]]}),
            &[2.5],
        ),
        case(
            "pc2_group_empty_only",
            "pc2",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.6,
                "vertex_group":"Empty"}),
            &json!({"Empty":[]}),
            &[2.5],
        ),
        case(
            "pc2_flip_x",
            "pc2",
            &json!({"play_mode":"CUSTOM","time_mode":"FRAME","eval_frame":2.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0,
                "flip_axis":[true,false,false]}),
            &json!({}),
            &[1.0],
        ),
        case(
            "pc2_flip_yz_axes",
            "pc2",
            &json!({"play_mode":"CUSTOM","time_mode":"FRAME","eval_frame":2.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0,
                "forward_axis":"NEG_X","up_axis":"POS_Y","flip_axis":[false,true,true]}),
            &json!({}),
            &[1.0],
        ),
        case(
            "mdd_disabled_bad_cache",
            "bad",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":0.0}),
            &json!({}),
            &[2.5],
        ),
        case(
            "mdd_disabled_missing_resource",
            "missing",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0}),
            &json!({}),
            &[2.5],
        ),
        case(
            "mdd_disabled_empty_resource",
            "empty",
            &json!({"play_mode":"SCENE","time_mode":"FRAME","frame_start":0.0,"frame_scale":1.0,
                "interpolation":"LINEAR","deform_mode":"OVERWRITE","factor":1.0,"resource":""}),
            &json!({}),
            &[2.5],
        ),
    ];
    cases.extend(axis_cases());
    cases
}

#[test]
fn blender_mesh_cache_pc2_and_mdd_match_native_pot() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping mesh cache Blender parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let operations_file = root.join("operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(initialized.status.success());
    apply(
        &scene,
        &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"base","kind":"box","params":{"size":2.0}}
        ]}),
        &operations_file,
    )?;
    fs::write(
        &operations_file,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[{"op":"modifier.create","target":{"id":"base"},"id":"unknown_cache","type":"mesh_cache","params":{"resource":"not_registered"}}]
        }))?,
    )?;
    let rejected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_file)
        .arg("--json")
        .output()?;
    assert!(
        !rejected.status.success(),
        "unregistered cache resource was accepted"
    );
    assert!(
        String::from_utf8_lossy(&rejected.stdout)
            .contains("resource `not_registered` was not found"),
        "unregistered cache identity error was missing: {}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    let base_positions = inspect_positions(&scene, "base", 1.0)?;
    let base = base_positions
        .iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect::<Vec<_>>();

    let frames = cache_frames(&base);
    let pc2_path = root.join("cache.pc2");
    let mdd_path = root.join("cache.mdd");
    let bad_path = root.join("bad.mdd");
    write_pc2(&pc2_path, &frames, 1.0 / 24.0, 1.0 / 24.0)?;
    write_mdd(&mdd_path, &frames, &MDD_TIMES)?;
    let short = frames
        .iter()
        .map(|frame| frame[..3].to_vec())
        .collect::<Vec<_>>();
    write_mdd(&bad_path, &short, &MDD_TIMES)?;

    let specs = cases();
    let mut operations = vec![
        json!({"op":"resource.pack","id":"pc2","uri":pc2_path.to_string_lossy(),"kind":"mesh_cache"}),
        json!({"op":"resource.pack","id":"mdd","uri":mdd_path.to_string_lossy(),"kind":"mesh_cache"}),
        json!({"op":"resource.pack","id":"bad","uri":bad_path.to_string_lossy(),"kind":"mesh_cache"}),
    ];
    for spec in &specs {
        let id = spec["id"].as_str().ok_or("case id")?;
        operations.push(json!({"op":"node.create","id":id,"kind":"box","params":{"size":2.0}}));
        if let Some(groups) = spec["groups"].as_object() {
            for (name, members) in groups {
                let group_id = format!("{id}_{}", name.to_lowercase());
                operations.push(json!({"op":"vertex_group.create","target":{"id":id},"id":group_id,"name":name}));
                let weights = members
                    .as_array()
                    .ok_or("group members")?
                    .iter()
                    .map(|member| json!({"vertex_id":member[0],"weight":member[1]}))
                    .collect::<Vec<_>>();
                if !weights.is_empty() {
                    operations.push(json!({"op":"vertex_group.assign","target":{"id":id},"group_id":group_id,"weights":weights}));
                }
            }
        }
        let cache = spec["cache"].as_str().ok_or("cache")?;
        let mut params = spec["params"].clone();
        if matches!(cache, "pc2" | "mdd" | "bad") {
            params["resource"] = json!(cache);
        }
        params["cache_format"] = json!(if cache == "pc2" { "PC2" } else { "MDD" });
        operations.push(json!({"op":"modifier.create","target":{"id":id},"id":"cache","type":"mesh_cache","params":params}));
    }
    apply(
        &scene,
        &json!({"schema_version":1,"base_revision":1,"operations":operations}),
        &operations_file,
    )?;

    let blender_cases = specs
        .iter()
        .map(|spec| {
            let cache = spec["cache"].as_str().unwrap_or("mdd");
            let file = match cache {
                "pc2" => pc2_path.to_string_lossy().into_owned(),
                "mdd" => mdd_path.to_string_lossy().into_owned(),
                "bad" => bad_path.to_string_lossy().into_owned(),
                _ => String::new(),
            };
            let mut modifier = spec["params"].clone();
            if let Some(params) = modifier.as_object_mut() {
                params.remove("resource");
            }
            modifier["cache_format"] = json!(if cache == "pc2" { "PC2" } else { "MDD" });
            json!({
                "id": spec["id"],
                "file": file,
                "groups": spec["groups"],
                "frames": spec["frames"],
            })
        })
        .collect::<Vec<_>>();
    let config = root.join("oracle.json");
    let expected_path = root.join("expected.json");
    fs::write(
        &config,
        serde_json::to_vec(&json!({
            "fps": 24, "fps_base": 1.0, "base": base, "faces": faces_of(&scene)?,
            "cases": blender_cases,
        }))?,
    )?;
    let output = Command::new(&blender)
        .args(["-b", "--factory-startup", "--python"])
        .arg(ORACLE_SCRIPT)
        .arg("--")
        .arg(&config)
        .arg(&expected_path)
        .output()?;
    assert!(
        output.status.success(),
        "Blender oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&fs::read(&expected_path)?)?;

    let mut checked = 0_usize;
    for spec in &specs {
        let id = spec["id"].as_str().ok_or("case id")?;
        let blender_frames = expected[id]
            .as_array()
            .ok_or_else(|| format!("Blender frames missing for {id}"))?;
        for (frame, expected_points) in spec["frames"]
            .as_array()
            .ok_or("frames")?
            .iter()
            .zip(blender_frames)
        {
            let frame = frame.as_f64().ok_or("frame")?;
            let actual = inspect_positions(&scene, id, frame)?;
            let expected_points = expected_points.as_array().ok_or("expected positions")?;
            assert_eq!(actual.len(), expected_points.len(), "{id} at {frame}");
            let mut max_error = 0.0_f64;
            for (actual, expected) in actual.iter().zip(expected_points) {
                for (actual, expected) in actual.iter().zip(
                    expected
                        .as_array()
                        .ok_or("expected vector")?
                        .iter()
                        .map(|value| value.as_f64().unwrap_or(f64::NAN)),
                ) {
                    max_error = max_error.max((actual - expected).abs());
                }
            }
            assert!(
                max_error <= TOLERANCE,
                "{id} at frame {frame}: max error {max_error} exceeds {TOLERANCE}"
            );
            checked += 1;
        }
    }
    eprintln!("mesh cache Blender parity checked {checked} frame evaluations");
    Ok(())
}

fn faces_of(scene: &Path) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(scene)
        .args(["--id", "base", "--json"])
        .output()?;
    let value: Value = serde_json::from_slice(&output.stdout)?;
    Ok(value["result"]["items"][0]["evaluated_geometry"]["faces"].clone())
}
