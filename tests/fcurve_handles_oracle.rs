#[path = "common/blender_file.rs"]
mod blender_file;

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

const TOLERANCE: f64 = 1.0e-5;
const GENERATOR: &str = "tests/fixtures/fcurve_handles/generate_oracle.py";

fn checked_output(command: &mut Command, label: &str) -> Result<Output, Box<dyn Error>> {
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

fn run_json(command: &mut Command, label: &str) -> Result<Value, Box<dyn Error>> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn run_blender_oracle(blender: &Path, output_directory: &Path) -> Result<Value, Box<dyn Error>> {
    let root = fs::canonicalize(output_directory)?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GENERATOR);
    checked_output(
        Command::new(blender)
            .args([
                "--background",
                "--factory-startup",
                "--threads",
                "1",
                "--python",
            ])
            .arg(script)
            .args(["--"])
            .arg(&root),
        "Blender 5.2.2 F-Curve handle oracle",
    )?;
    if !root.join("expected.blend").is_file() {
        return Err(io::Error::other("Blender did not save expected.blend").into());
    }
    let oracle: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(oracle["generator"], GENERATOR);
    assert_eq!(oracle["tolerance"].as_f64(), Some(TOLERANCE));
    let cases = oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?;
    let required = [
        ("auto", "auto", "none"),
        ("auto_clamped", "auto_clamped", "none"),
        ("vector", "vector", "none"),
        ("aligned", "aligned", "none"),
        ("free", "free", "none"),
        ("smoothing_none", "auto", "none"),
        ("smoothing_cont_accel", "auto", "cont_accel"),
        ("cyclic", "auto_clamped", "none"),
        (
            "cyclic_cont_accel_auto_clamped",
            "auto_clamped",
            "cont_accel",
        ),
        ("mixed_first_auto_vector_constant", "auto", "none"),
        ("subdivide_nonauto_handles", "free", "none"),
        ("subdivide_auto_free_between_auto", "auto", "none"),
        (
            "vector_auto_endpoint_cont_accel_linear",
            "auto",
            "cont_accel",
        ),
    ];
    assert_eq!(cases.len(), required.len(), "oracle case coverage changed");
    for (name, handle_type, auto_smoothing) in required {
        let case = cases
            .iter()
            .find(|case| case["name"] == name)
            .ok_or_else(|| io::Error::other(format!("Blender oracle omitted `{name}`")))?;
        assert_eq!(case["handle_type"], handle_type, "{name} handle type");
        assert_eq!(
            case["auto_smoothing"], auto_smoothing,
            "{name} auto smoothing"
        );
    }
    let subdivision = cases
        .iter()
        .find(|case| case["name"] == "subdivide_nonauto_handles")
        .ok_or("Blender oracle omitted the subdivision case")?;
    assert_eq!(subdivision["scenario"], "subdivide_nonauto_handles");
    assert_eq!(
        subdivision["seed_keys"].as_array().map(Vec::len),
        Some(2),
        "subdivision case must seed two endpoint keys"
    );
    assert_eq!(
        subdivision["keys"].as_array().map(Vec::len),
        Some(3),
        "subdivision case must contain the inserted middle key"
    );
    for key in subdivision["seed_keys"]
        .as_array()
        .ok_or("subdivision seed keys are missing")?
    {
        assert_eq!(key["handle_left_type"], "free");
        assert_eq!(key["handle_right_type"], "free");
    }
    let cyclic = cases
        .iter()
        .find(|case| case["name"] == "cyclic")
        .ok_or("Blender oracle omitted the cyclic case")?;
    assert_eq!(cyclic["modifiers"][0]["type"], "CYCLES");
    assert!(
        cyclic["samples"].as_array().is_some_and(|samples| samples
            .iter()
            .any(|sample| sample["frame"] == -3.5)
            && samples.iter().any(|sample| sample["frame"] == 24.5)),
        "cyclic oracle must evaluate before and after its key range"
    );
    let cyclic_accel = cases
        .iter()
        .find(|case| case["name"] == "cyclic_cont_accel_auto_clamped")
        .ok_or("Blender oracle omitted the cyclic CONT_ACCEL case")?;
    assert_eq!(
        cyclic_accel["initial_keys"],
        json!([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0], [4.0, 0.0]])
    );
    assert_eq!(cyclic_accel["modifiers"][0]["type"], "CYCLES");
    let constant_mixed = cases
        .iter()
        .find(|case| case["name"] == "mixed_first_auto_vector_constant")
        .ok_or("Blender oracle omitted the mixed constant-extrapolation case")?;
    assert_eq!(constant_mixed["extrapolation"], "constant");
    assert_eq!(constant_mixed["keys"][0]["handle_left_type"], "auto");
    assert_eq!(constant_mixed["keys"][0]["handle_right_type"], "vector");

    let auto_free = cases
        .iter()
        .find(|case| case["name"] == "subdivide_auto_free_between_auto")
        .ok_or("Blender oracle omitted the mixed subdivision case")?;
    assert_eq!(auto_free["scenario"], "subdivide_auto_free_between_auto");
    assert_eq!(auto_free["seed_keys"].as_array().map(Vec::len), Some(2));
    assert_eq!(auto_free["keys"].as_array().map(Vec::len), Some(3));
    for key in auto_free["seed_keys"]
        .as_array()
        .ok_or("mixed subdivision seed keys are missing")?
    {
        assert_eq!(key["handle_left_type"], "auto");
        assert_eq!(key["handle_right_type"], "auto");
    }
    assert_eq!(auto_free["inserted_key"]["handle_left_type"], "auto");
    assert_eq!(auto_free["inserted_key"]["handle_right_type"], "free");

    let vector_endpoint = cases
        .iter()
        .find(|case| case["name"] == "vector_auto_endpoint_cont_accel_linear")
        .ok_or("Blender oracle omitted the VECTOR/AUTO endpoint case")?;
    assert_eq!(vector_endpoint["extrapolation"], "linear");
    let first_key = vector_endpoint["keys"]
        .as_array()
        .and_then(|keys| keys.first())
        .ok_or("VECTOR/AUTO endpoint key is missing")?;
    assert_eq!(first_key["handle_left_type"], "vector");
    assert_eq!(first_key["handle_right_type"], "auto");
    assert!(
        vector_endpoint["samples"]
            .as_array()
            .is_some_and(|samples| samples
                .iter()
                .any(|sample| sample["frame"] == first_key["frame"])),
        "VECTOR/AUTO case must evaluate its linear-extrapolation endpoint"
    );
    Ok(oracle)
}

fn action_ids(case: &Value) -> Result<(String, String), Box<dyn Error>> {
    let name = case["name"].as_str().ok_or("oracle case name is missing")?;
    Ok((format!("{name}_node"), format!("{name}_action")))
}

fn operations(oracle: &Value, direct_mixed_insert: bool) -> Result<Value, Box<dyn Error>> {
    let mut operations = Vec::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
    {
        let (node_id, action_id) = action_ids(case)?;
        let handle_type = case["handle_type"]
            .as_str()
            .ok_or("oracle handle type is missing")?;
        operations.push(json!({"op":"node.create", "id":node_id, "kind":"empty"}));
        operations.push(json!({"op":"action.create", "id":action_id, "name":node_id}));
        operations.push(json!({
            "op":"node.update",
            "target":{"id":node_id},
            "set":{"action":action_id}
        }));
        if matches!(
            case["scenario"].as_str(),
            Some("subdivide_nonauto_handles" | "subdivide_auto_free_between_auto")
        ) {
            let seed_keys = case["seed_keys"]
                .as_array()
                .ok_or("subdivision oracle seed keys are missing")?
                .iter()
                .map(|key| {
                    json!({
                        "frame":key["frame"],
                        "value":key["value"],
                        "interpolation":key["interpolation"],
                        "handle_left_type":key["handle_left_type"],
                        "handle_right_type":key["handle_right_type"],
                        "handle_left":key["handle_left"],
                        "handle_right":key["handle_right"]
                    })
                })
                .collect::<Vec<_>>();
            operations.push(json!({
                "op":"action.update",
                "target":{"id":action_id},
                "set":{"fcurves":[{
                    "path":"transform.translation",
                    "index":0,
                    "auto_smoothing":case["auto_smoothing"],
                    "extrapolation":case["extrapolation"],
                    "keyframes":seed_keys,
                    "modifiers":case["modifiers"]
                }]}
            }));
            let inserted = &case["inserted_key"];
            let mixed_subdivide = case["scenario"] == "subdivide_auto_free_between_auto";
            let set_mixed_after_insert = mixed_subdivide && !direct_mixed_insert;
            let initial_left_type = if set_mixed_after_insert {
                json!("auto")
            } else {
                inserted["handle_left_type"].clone()
            };
            let initial_right_type = if set_mixed_after_insert {
                json!("auto")
            } else {
                inserted["handle_right_type"].clone()
            };
            operations.push(json!({
                "op":"keyframe.insert",
                "target":{"id":node_id},
                "path":"transform.translation",
                "index":0,
                "frame":inserted["frame"],
                "value":inserted["value"],
                "interpolation":inserted["interpolation"],
                "handle_left_type":initial_left_type,
                "handle_right_type":initial_right_type
            }));
            if set_mixed_after_insert {
                operations.push(json!({
                    "op":"keyframe.insert",
                    "target":{"id":node_id},
                    "path":"transform.translation",
                    "index":0,
                    "frame":inserted["frame"],
                    "value":inserted["value"],
                    "interpolation":inserted["interpolation"],
                    "handle_left_type":inserted["handle_left_type"],
                    "handle_right_type":inserted["handle_right_type"]
                }));
            }
            continue;
        }
        for key in case["initial_keys"]
            .as_array()
            .ok_or("oracle initial keys are missing")?
        {
            operations.push(json!({
                "op":"keyframe.insert",
                "target":{"id":node_id},
                "path":"transform.translation",
                "index":0,
                "frame":key[0],
                "value":key[1],
                "interpolation":"bezier",
                "handle_left_type":handle_type,
                "handle_right_type":handle_type
            }));
        }

        let mut keyframes = Vec::new();
        for key in case["keys"].as_array().ok_or("oracle keys are missing")? {
            let mut updated = json!({
                "frame":key["frame"],
                "value":key["value"],
                "interpolation":"bezier",
                "handle_left_type":key["handle_left_type"],
                "handle_right_type":key["handle_right_type"]
            });
            for (field, input_field) in [
                ("handle_left", "input_handle_left"),
                ("handle_right", "input_handle_right"),
            ] {
                if !key[input_field].is_null() {
                    updated[field] = key[input_field].clone();
                }
            }
            keyframes.push(updated);
        }
        operations.push(json!({
            "op":"action.update",
            "target":{"id":action_id},
            "set":{"fcurves":[{
                "path":"transform.translation",
                "index":0,
                "auto_smoothing":case["auto_smoothing"],
                "extrapolation":case["extrapolation"],
                "keyframes":keyframes,
                "modifiers":case["modifiers"]
            }]}
        }));
    }
    Ok(json!({"schema_version":1,"base_revision":0,"operations":operations}))
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{context}: Potter={actual:.9}, Blender={expected:.9}, tolerance={TOLERANCE}"
    );
}

fn run_pot_inspect(project: &Path, frame: f64) -> Result<Value, Box<dyn Error>> {
    let frame_text = frame.to_string();
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["inspect"])
            .arg(project)
            .arg(format!("--frame={frame_text}"))
            .arg("--json"),
        "pot inspect F-Curve frame",
    )
}

#[test]
fn fcurve_handle_recalculation_matches_blender_522_through_pot_cli() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping F-Curve handle oracle: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let oracle = run_blender_oracle(&blender, root)?;
    let original_cases = oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
        .iter()
        .filter(|case| {
            matches!(
                case["name"].as_str(),
                Some(
                    "auto"
                        | "auto_clamped"
                        | "vector"
                        | "aligned"
                        | "free"
                        | "smoothing_none"
                        | "smoothing_cont_accel"
                        | "cyclic"
                        | "subdivide_nonauto_handles"
                )
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(original_cases.len(), 9, "original all-types oracle changed");
    assert_cases_through_pot(&json!({"cases":original_cases}), root)
}

fn assert_cases_through_pot(oracle: &Value, root: &Path) -> Result<(), Box<dyn Error>> {
    let project = root.join("project");
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["init"])
            .arg(&project)
            .arg("--json"),
        "pot init",
    )?;
    let batch_path = root.join("handle-operations.json");
    fs::write(
        &batch_path,
        serde_json::to_vec(&operations(oracle, false)?)?,
    )?;
    let applied = run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["apply"])
            .arg(&project)
            .args(["--file"])
            .arg(&batch_path)
            .arg("--json"),
        "pot apply F-Curve handle operations",
    )?;
    assert_eq!(applied["result"]["committed"], true);

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let mut actual_inspections = HashMap::<u64, Value>::new();
    let mut seen_frames = HashSet::new();
    let mut frames = Vec::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
    {
        let (_, action_id) = action_ids(case)?;
        let expected_keys = case["keys"].as_array().ok_or("oracle keys are missing")?;
        let actual_curve = scene["actions"][&action_id]["fcurves"][0]
            .as_object()
            .ok_or_else(|| io::Error::other(format!("{action_id}: F-Curve is missing")))?;
        assert_eq!(actual_curve["auto_smoothing"], case["auto_smoothing"]);
        assert_eq!(actual_curve["extrapolation"], case["extrapolation"]);
        assert_eq!(actual_curve["modifiers"], case["modifiers"]);
        let actual_keys = actual_curve["keyframes"]
            .as_array()
            .ok_or_else(|| io::Error::other(format!("{action_id}: keyframes are missing")))?;
        assert_eq!(
            actual_keys.len(),
            expected_keys.len(),
            "{action_id}: key count"
        );
        for (index, (actual, expected)) in actual_keys.iter().zip(expected_keys).enumerate() {
            for scalar in ["frame", "value"] {
                assert_close(
                    actual[scalar]
                        .as_f64()
                        .ok_or("Potter key scalar is missing")?,
                    expected[scalar]
                        .as_f64()
                        .ok_or("Blender key scalar is missing")?,
                    &format!("{action_id} key {index} {scalar}"),
                );
            }
            for (coordinate, expected_coordinate) in [
                ("handle_left", &expected["handle_left"]),
                ("handle_right", &expected["handle_right"]),
            ] {
                assert_eq!(
                    actual[format!("{coordinate}_type")],
                    expected[format!("{coordinate}_type")],
                    "{action_id} key {index} handle type"
                );
                let actual_handle = actual[coordinate].as_array().ok_or_else(|| {
                    io::Error::other(format!("{action_id} key {index} {coordinate} is missing"))
                })?;
                let expected_handle = expected_coordinate.as_array().ok_or_else(|| {
                    io::Error::other(format!("Blender key {index} {coordinate} is missing"))
                })?;
                assert_eq!(actual_handle.len(), 2);
                assert_eq!(expected_handle.len(), 2);
                for axis in 0..2 {
                    assert_close(
                        actual_handle[axis]
                            .as_f64()
                            .ok_or("Potter handle coordinate is missing")?,
                        expected_handle[axis]
                            .as_f64()
                            .ok_or("Blender handle coordinate is missing")?,
                        &format!("{action_id} key {index} {coordinate}[{axis}]"),
                    );
                }
            }
        }
        for sample in case["samples"]
            .as_array()
            .ok_or("oracle samples are missing")?
        {
            let frame = sample["frame"].as_f64().ok_or("oracle frame is missing")?;
            if seen_frames.insert(frame.to_bits()) {
                frames.push(frame);
            }
        }
    }

    frames.sort_by(f64::total_cmp);
    for frame in frames {
        actual_inspections.insert(frame.to_bits(), run_pot_inspect(&project, frame)?);
    }
    for case in oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
    {
        let (node_id, _) = action_ids(case)?;
        for sample in case["samples"]
            .as_array()
            .ok_or("oracle samples are missing")?
        {
            let frame = sample["frame"].as_f64().ok_or("oracle frame is missing")?;
            let inspected = actual_inspections
                .get(&frame.to_bits())
                .ok_or("Potter frame inspection is missing")?;
            let items = inspected["result"]["items"]
                .as_array()
                .ok_or("Potter inspect items are missing")?;
            let node = items
                .iter()
                .find(|item| item["id"] == node_id)
                .ok_or_else(|| {
                    io::Error::other(format!("Potter inspect omitted `{node_id}` at {frame}"))
                })?;
            let actual = node["transform"]["evaluated"]["translation"][0]
                .as_f64()
                .ok_or("Potter evaluated translation is missing")?;
            assert_close(
                actual,
                sample["value"]
                    .as_f64()
                    .ok_or("Blender evaluation is missing")?,
                &format!("{node_id} evaluated at frame {frame}"),
            );
        }
    }
    Ok(())
}

fn run_regression_case(name: &str) -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping `{name}` F-Curve handle regression: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let oracle = run_blender_oracle(&blender, directory.path())?;
    let case = oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
        .iter()
        .find(|case| case["name"] == name)
        .cloned()
        .ok_or_else(|| io::Error::other(format!("Blender oracle omitted `{name}`")))?;
    assert_cases_through_pot(&json!({"cases":[case]}), directory.path())
}

#[test]
fn cyclic_cont_accel_auto_clamped_handles_match_blender() -> Result<(), Box<dyn Error>> {
    run_regression_case("cyclic_cont_accel_auto_clamped")
}

#[test]
fn mixed_first_auto_vector_constant_extrapolation_matches_blender() -> Result<(), Box<dyn Error>> {
    run_regression_case("mixed_first_auto_vector_constant")
}

#[test]
fn subdivide_auto_free_key_between_auto_neighbors_matches_blender() -> Result<(), Box<dyn Error>> {
    run_regression_case("subdivide_auto_free_between_auto")
}

type Point2 = [f64; 2];

fn point2(value: &Value) -> Result<Point2, Box<dyn Error>> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or("expected a 2D point")?;
    Ok([
        pair[0].as_f64().ok_or("point x is missing")?,
        pair[1].as_f64().ok_or("point y is missing")?,
    ])
}

fn bezier_point(points: [Point2; 4], t: f64) -> Point2 {
    let a = [
        points[0][0] * (1.0 - t) + points[1][0] * t,
        points[0][1] * (1.0 - t) + points[1][1] * t,
    ];
    let b = [
        points[1][0] * (1.0 - t) + points[2][0] * t,
        points[1][1] * (1.0 - t) + points[2][1] * t,
    ];
    let c = [
        points[2][0] * (1.0 - t) + points[3][0] * t,
        points[2][1] * (1.0 - t) + points[3][1] * t,
    ];
    let d = [a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t];
    let e = [b[0] * (1.0 - t) + c[0] * t, b[1] * (1.0 - t) + c[1] * t];
    [d[0] * (1.0 - t) + e[0] * t, d[1] * (1.0 - t) + e[1] * t]
}

fn bezier_parameter_for_frame(points: [Point2; 4], frame: f64) -> f64 {
    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..64 {
        let t = f64::midpoint(lower, upper);
        if bezier_point(points, t)[0] < frame {
            lower = t;
        } else {
            upper = t;
        }
    }
    f64::midpoint(lower, upper)
}

fn seed_curve_points(seed_keys: &[Value]) -> Result<[Point2; 4], Box<dyn Error>> {
    let start = seed_keys
        .first()
        .ok_or("Blender seed start key is missing")?;
    let end = seed_keys.last().ok_or("Blender seed end key is missing")?;
    Ok([
        [
            start["frame"]
                .as_f64()
                .ok_or("seed start frame is missing")?,
            start["value"]
                .as_f64()
                .ok_or("seed start value is missing")?,
        ],
        point2(&start["handle_right"])?,
        point2(&end["handle_left"])?,
        [
            end["frame"].as_f64().ok_or("seed end frame is missing")?,
            end["value"].as_f64().ok_or("seed end value is missing")?,
        ],
    ])
}

/// Derive Potter's De Casteljau insertion handles from Blender's pre-insert seed, not its later
/// AUTO update.
fn bezier_split_handles(points: [Point2; 4], frame: f64, value: f64) -> [Point2; 4] {
    let [start, mut start_right, mut end_left, end] = points;
    let end_x = end[0] - start[0];
    let first_x = (start[0] - start_right[0]).abs();
    let second_x = (end[0] - end_left[0]).abs();
    if first_x + second_x != 0.0 {
        if first_x > end_x {
            let factor = end_x / first_x;
            start_right = [
                start[0] - factor * (start[0] - start_right[0]),
                start[1] - factor * (start[1] - start_right[1]),
            ];
        }
        if second_x > end_x {
            let factor = end_x / second_x;
            end_left = [
                end[0] - factor * (end[0] - end_left[0]),
                end[1] - factor * (end[1] - end_left[1]),
            ];
        }
    }
    let t = bezier_parameter_for_frame([start, start_right, end_left, end], frame);
    let split_a = [
        start[0] * (1.0 - t) + start_right[0] * t,
        start[1] * (1.0 - t) + start_right[1] * t,
    ];
    let split_b = [
        start_right[0] * (1.0 - t) + end_left[0] * t,
        start_right[1] * (1.0 - t) + end_left[1] * t,
    ];
    let split_c = [
        end_left[0] * (1.0 - t) + end[0] * t,
        end_left[1] * (1.0 - t) + end[1] * t,
    ];
    let split_d = [
        split_a[0] * (1.0 - t) + split_b[0] * t,
        split_a[1] * (1.0 - t) + split_b[1] * t,
    ];
    let split_e = [
        split_b[0] * (1.0 - t) + split_c[0] * t,
        split_b[1] * (1.0 - t) + split_c[1] * t,
    ];
    let split_point = [
        split_d[0] * (1.0 - t) + split_e[0] * t,
        split_d[1] * (1.0 - t) + split_e[1] * t,
    ];
    let difference = [frame - split_point[0], value - split_point[1]];
    [
        split_a,
        [split_d[0] + difference[0], split_d[1] + difference[1]],
        [split_e[0] + difference[0], split_e[1] + difference[1]],
        split_c,
    ]
}

#[test]
fn mixed_auto_free_types_passed_to_keyframe_insert_use_de_casteljau_split()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping mixed-handle insert split regression: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let oracle = run_blender_oracle(&blender, root)?;
    let case = oracle["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
        .iter()
        .find(|case| case["name"] == "subdivide_auto_free_between_auto")
        .cloned()
        .ok_or("Blender oracle omitted the mixed subdivision seed")?;
    let seed_keys = case["seed_keys"]
        .as_array()
        .ok_or("Blender subdivision seed keys are missing")?;
    let inserted = &case["inserted_key"];
    let frame = inserted["frame"]
        .as_f64()
        .ok_or("insert frame is missing")?;
    let value = inserted["value"]
        .as_f64()
        .ok_or("insert value is missing")?;
    let source_points = seed_curve_points(seed_keys)?;
    let split = bezier_split_handles(source_points, frame, value);
    let one_case = json!({"cases":[case.clone()]});
    let project = root.join("mixed-insert-project");
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["init"])
            .arg(&project)
            .arg("--json"),
        "pot init mixed-insert project",
    )?;
    let batch_path = root.join("mixed-insert-operations.json");
    fs::write(
        &batch_path,
        serde_json::to_vec(&operations(&one_case, true)?)?,
    )?;
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["apply"])
            .arg(&project)
            .args(["--file"])
            .arg(&batch_path)
            .arg("--json"),
        "pot apply mixed AUTO/FREE keyframe insert",
    )?;

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (_, action_id) = action_ids(&one_case["cases"][0])?;
    let actual_keys = scene["actions"][&action_id]["fcurves"][0]["keyframes"]
        .as_array()
        .ok_or("mixed-insert Potter keys are missing")?;
    assert_eq!(actual_keys.len(), 3);
    let expected_keys = [&seed_keys[0], inserted, &seed_keys[1]];
    let expected_free_right = split[2];
    let expected_types = [
        [
            seed_keys[0]["handle_left_type"]
                .as_str()
                .ok_or("seed left type is missing")?,
            seed_keys[0]["handle_right_type"]
                .as_str()
                .ok_or("seed right type is missing")?,
        ],
        ["auto", "free"],
        [
            seed_keys[1]["handle_left_type"]
                .as_str()
                .ok_or("seed left type is missing")?,
            seed_keys[1]["handle_right_type"]
                .as_str()
                .ok_or("seed right type is missing")?,
        ],
    ];
    for (index, key) in actual_keys.iter().enumerate() {
        for scalar in ["frame", "value"] {
            assert_close(
                key[scalar].as_f64().ok_or("Potter key scalar is missing")?,
                expected_keys[index][scalar]
                    .as_f64()
                    .ok_or("expected key scalar is missing")?,
                &format!("mixed-insert key {index} {scalar}"),
            );
        }
        for (side, handle) in ["left", "right"].iter().enumerate() {
            let coordinate = format!("handle_{handle}");
            assert_eq!(
                key[format!("{coordinate}_type")],
                expected_types[index][side],
                "mixed-insert key {index} {coordinate} type"
            );
            if index == 1 && side == 1 {
                let actual_handle = point2(&key[coordinate.as_str()])?;
                assert_close(
                    actual_handle[0],
                    expected_free_right[0],
                    "mixed-insert FREE right handle x",
                );
                assert_close(
                    actual_handle[1],
                    expected_free_right[1],
                    "mixed-insert FREE right handle y",
                );
            }
        }
    }

    Ok(())
}
#[test]
fn vector_auto_endpoint_cont_accel_linear_extrapolation_matches_blender()
-> Result<(), Box<dyn Error>> {
    run_regression_case("vector_auto_endpoint_cont_accel_linear")
}

#[test]
fn legacy_fcurve_handles_survive_separate_channel_edit_and_blend_round_trip()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping legacy F-Curve handle round trip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("legacy-project");
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["init"])
            .arg(&project)
            .arg("--json"),
        "pot init legacy handle project",
    )?;

    let legacy_curve: Value =
        serde_json::from_str(include_str!("fixtures/fcurve_handles/legacy_scene.json"))?;
    assert!(legacy_curve.get("auto_smoothing").is_none());
    let legacy_keys = legacy_curve["keyframes"]
        .as_array()
        .ok_or("legacy fixture keyframes are missing")?;
    assert_eq!(legacy_keys.len(), 2);
    for key in legacy_keys {
        assert!(key.get("handle_left_type").is_none());
        assert!(key.get("handle_right_type").is_none());
    }
    let seed_keys = legacy_keys
        .iter()
        .map(|key| {
            json!({
                "frame":key["frame"],
                "value":key["value"],
                "interpolation":key["interpolation"],
                "handle_left_type":"free",
                "handle_right_type":"free",
                "handle_left":key["handle_left"],
                "handle_right":key["handle_right"]
            })
        })
        .collect::<Vec<_>>();
    let setup_batch = root.join("legacy-setup.json");
    fs::write(
        &setup_batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"legacy_node","kind":"empty"},
                {"op":"action.create","id":"legacy_action","name":"Legacy Action"},
                {"op":"node.update","target":{"id":"legacy_node"},"set":{"action":"legacy_action"}},
                {"op":"action.update","target":{"id":"legacy_action"},"set":{"fcurves":[{
                    "path":"transform.translation",
                    "index":0,
                    "auto_smoothing":"none",
                    "extrapolation":"constant",
                    "keyframes":seed_keys,
                    "modifiers":[]
                }]}}
            ]
        }))?,
    )?;
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["apply"])
            .arg(&project)
            .args(["--file"])
            .arg(&setup_batch)
            .arg("--json"),
        "pot apply legacy handle fixture setup",
    )?;

    let scene_path = project.join("scene.json");
    let mut scene: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let fcurves = scene["actions"]["legacy_action"]["fcurves"]
        .as_array_mut()
        .ok_or("legacy setup F-Curve is missing")?;
    assert_eq!(fcurves.len(), 1);
    fcurves[0] = legacy_curve.clone();
    fs::write(&scene_path, serde_json::to_vec_pretty(&scene)?)?;

    let insert_batch = root.join("legacy-channel-insert.json");
    fs::write(
        &insert_batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[{
                "op":"keyframe.insert",
                "target":{"id":"legacy_node"},
                "path":"transform.translation",
                "index":1,
                "frame":1.0,
                "value":0.0
            }, {
                "op":"fcurve.update",
                "target":{"id":"legacy_node"},
                "path":"transform.translation",
                "index":1,
                "set":{"extrapolation":"linear"}
            }]
        }))?,
    )?;
    let applied = run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["apply"])
            .arg(&project)
            .args(["--file"])
            .arg(&insert_batch)
            .arg("--json"),
        "pot apply keyframe on separate legacy channel",
    )?;
    assert_eq!(applied["result"]["committed"], true);

    let blend = root.join("legacy-handles.blend");
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["export"])
            .arg(&project)
            .args(["--format", "blend", "--out"])
            .arg(&blend)
            .args(["--blender"])
            .arg(&blender)
            .arg("--json"),
        "pot export legacy F-Curve handles",
    )?;
    let reopened = root.join("reopened-project");
    run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["init"])
            .arg(&reopened)
            .arg("--json"),
        "pot init legacy round-trip project",
    )?;
    let imported = run_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import"])
            .arg(&reopened)
            .args(["--file"])
            .arg(&blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "pot import and reopen legacy F-Curve handles",
    )?;
    assert_eq!(imported["result"]["losses"], json!([]));

    let reopened_scene: Value = serde_json::from_slice(&fs::read(reopened.join("scene.json"))?)?;
    let action = reopened_scene["actions"]
        .as_object()
        .and_then(|actions| {
            actions
                .values()
                .find(|action| action["name"] == "Legacy Action")
        })
        .ok_or("round-tripped Legacy Action is missing")?;
    let curves = action["fcurves"]
        .as_array()
        .ok_or("round-tripped Legacy Action F-Curves are missing")?;
    let restored_legacy_curve = curves
        .iter()
        .find(|curve| curve["path"] == "transform.translation" && curve["index"] == 0)
        .ok_or("round-tripped legacy F-Curve is missing")?;
    let restored_keys = restored_legacy_curve["keyframes"]
        .as_array()
        .ok_or("round-tripped legacy F-Curve keys are missing")?;
    assert_eq!(restored_keys.len(), legacy_keys.len());
    for (index, (actual, expected)) in restored_keys.iter().zip(legacy_keys).enumerate() {
        for handle in ["handle_left", "handle_right"] {
            for axis in 0..2 {
                assert_close(
                    actual[handle][axis]
                        .as_f64()
                        .ok_or("round-tripped legacy handle coordinate is missing")?,
                    expected[handle][axis]
                        .as_f64()
                        .ok_or("legacy fixture handle coordinate is missing")?,
                    &format!("legacy key {index} {handle}[{axis}]"),
                );
            }
        }
    }
    let edited_curve = curves
        .iter()
        .find(|curve| curve["path"] == "transform.translation" && curve["index"] == 1)
        .ok_or("separate translation channel was not round-tripped")?;
    assert_eq!(
        edited_curve["extrapolation"], "linear",
        "edited F-Curve extrapolation must survive .blend export"
    );
    Ok(())
}
