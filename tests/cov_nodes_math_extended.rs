#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test setup"
)]

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/eevee_film.rs"]
mod eevee_film;
use eevee_film::assert_eevee_film_close;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};

use potter_core::{
    graph::{self, GraphKind, GraphLink, GraphNode, NodeGroup},
    model::Id,
};
use serde_json::{Map, Value, json};
use tempfile::{TempDir, tempdir};

struct Oracle {
    directory: TempDir,
    cases: Vec<Value>,
    geometry: Map<String, Value>,
    shader: Map<String, Value>,
    eevee: Map<String, Value>,
    eevee_engine: String,
    eevee_resolution: Value,
    blend_path: PathBuf,
}

fn run_oracle() -> Oracle {
    let blender = blender_file::blender_executable().expect("Blender 5.2.2 is available");
    let directory = tempdir().unwrap();
    let blend_path = directory.path().join("node_math_extended_probes.blend");
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/node_math/generate_extended.py");
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(&blend_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Blender extended node oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NODE_MATH_EXTENDED="))
        .expect("Blender extended oracle emitted a result line");
    let result: Value = serde_json::from_str(line).unwrap();
    Oracle {
        directory,
        cases: result["cases"].as_array().unwrap().clone(),
        geometry: result["geometry"].as_object().unwrap().clone(),
        shader: result["shader"].as_object().unwrap().clone(),
        eevee: result["eevee"].as_object().unwrap().clone(),
        eevee_engine: result["eevee_engine"].as_str().unwrap().to_owned(),
        eevee_resolution: result["eevee_resolution"].clone(),
        blend_path,
    }
}

fn add_node(group: &mut NodeGroup, id: &str, node_type: &str, inputs: Value, properties: Value) {
    let mut node = GraphNode::new(node_type);
    let Value::Object(inputs) = inputs else {
        unreachable!("node input fixtures are JSON objects")
    };
    let Value::Object(properties) = properties else {
        unreachable!("node property fixtures are JSON objects")
    };
    node.inputs = inputs.into_iter().collect();
    node.properties = properties;
    group.nodes.insert(Id::new(id).unwrap(), node);
}

fn link(group: &mut NodeGroup, from: &str, from_socket: &str, to: &str, to_socket: &str) {
    group.links.push(GraphLink {
        from_node: Id::new(from).unwrap(),
        from_socket: from_socket.to_owned(),
        to_node: Id::new(to).unwrap(),
        to_socket: to_socket.to_owned(),
    });
}

fn blender_geometry(oracle: &Oracle) -> Map<String, Value> {
    let mut group = NodeGroup::new("Extended node math parity", GraphKind::Geometry);
    add_node(
        &mut group,
        "line",
        "GeometryNodeMeshLine",
        json!({"Count": 1}),
        json!({}),
    );
    let mut geometry_source = "line".to_owned();
    for (index, case) in oracle.cases.iter().enumerate() {
        let operation_id = format!("operation_{index}");
        let store_id = format!("store_{index}");
        add_node(
            &mut group,
            &operation_id,
            case["node_type"].as_str().unwrap(),
            case["inputs"].clone(),
            case["properties"].clone(),
        );
        add_node(
            &mut group,
            &store_id,
            "GeometryNodeStoreNamedAttribute",
            json!({"Name": case["id"]}),
            json!({"domain": "POINT", "data_type": case["attribute_type"]}),
        );
        let geometry_socket = if index == 0 { "Mesh" } else { "Geometry" };
        link(
            &mut group,
            &geometry_source,
            geometry_socket,
            &store_id,
            "Geometry",
        );
        link(
            &mut group,
            &operation_id,
            case["output"].as_str().unwrap(),
            &store_id,
            "Value",
        );
        geometry_source = store_id;
    }
    add_node(
        &mut group,
        "output",
        "NodeGroupOutput",
        json!({}),
        json!({}),
    );
    link(
        &mut group,
        &geometry_source,
        "Geometry",
        "output",
        "Geometry",
    );

    let evaluation = graph::evaluate(&group, None, &Map::new(), "Extended node math parity")
        .expect("Potter evaluates generated geometry node cases");
    let mut actual = BTreeMap::new();
    for case in &oracle.cases {
        let id = case["id"].as_str().unwrap();
        let attribute = &evaluation.mesh.attributes[id];
        let data = attribute["data"].as_array().unwrap();
        assert_eq!(data.len(), 1, "{id}: expected one evaluated mesh point");
        actual.insert(id.to_owned(), data[0].clone());
    }
    actual.into_iter().collect()
}

fn json_mismatch(actual: &Value, expected: &Value, label: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            if let (Some(actual), Some(expected)) = (actual.as_i64(), expected.as_i64()) {
                return (actual != expected)
                    .then(|| format!("{label}: Blender {expected}, Potter {actual}"));
            }
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            ((actual - expected).abs() > tolerance).then(|| {
                format!("{label}: Blender {expected:?}, Potter {actual:?} (tolerance {tolerance})")
            })
        }
        (Value::Array(actual), Value::Array(expected)) if actual.len() == expected.len() => actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (actual, expected))| {
                json_mismatch(actual, expected, &format!("{label}[{index}]"))
            }),
        (Value::Bool(actual), Value::Bool(expected)) => {
            (actual != expected).then(|| format!("{label}: Blender {expected}, Potter {actual}"))
        }
        (Value::String(actual), Value::String(expected)) => (actual != expected)
            .then(|| format!("{label}: Blender {expected:?}, Potter {actual:?}")),
        _ => Some(format!(
            "{label}: Blender and Potter returned different value shapes"
        )),
    }
}

struct ProbeImage {
    width: usize,
    height: usize,
    pixels: Vec<[f64; 3]>,
}

fn read_probe_image(path: &Path) -> ProbeImage {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| ProbeImage {
            width: resolution.width(),
            height: resolution.height(),
            pixels: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |image: &mut ProbeImage, position, (red, green, blue, _): (f32, f32, f32, f32)| {
            image.pixels[position.y() * image.width + position.x()] =
                [f64::from(red), f64::from(green), f64::from(blue)];
        },
    )
    .unwrap();
    image.layer_data.channel_data.pixels
}
fn realtime_film_precision(value: f64) -> f64 {
    f64::from(exr::prelude::f16::from_f32(value as f32).to_f32())
}

fn pot_json(command: &mut Command, label: &str) -> Value {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "Potter {label} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn compare_potter_shader_render(oracle: &Oracle) {
    let blender = blender_file::blender_executable().expect("Blender 5.2.2 is available");
    let captured: Value =
        serde_json::from_str(include_str!("fixtures/node_math/extended_eevee_eevee.json")).unwrap();
    assert_eq!(oracle.eevee_engine, captured["engine"].as_str().unwrap());
    assert_eq!(oracle.eevee_resolution, captured["resolution"]);
    let realtime_cases: Vec<&Value> = oracle
        .cases
        .iter()
        .filter(|case| case["realtime_probe"].as_bool() == Some(true))
        .collect();
    let captured_cases = captured["cases"].as_array().unwrap();
    assert_eq!(realtime_cases.len(), captured_cases.len());
    for (case, expected_case) in realtime_cases.iter().zip(captured_cases) {
        assert_eq!(case["id"], expected_case["id"]);
        assert_eq!(case["render_kind"], expected_case["render_kind"]);
    }
    assert_eq!(
        oracle.eevee.keys().collect::<Vec<_>>(),
        captured["eevee"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
    );
    let project = oracle.directory.path().join("extended_shader_project");
    let project_text = project.to_str().unwrap();
    let initialized = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", project_text, "--json"]),
        "extended shader project initialization",
    );
    assert_eq!(initialized["ok"], true);
    let imported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import", project_text, "--file"])
            .arg(&oracle.blend_path)
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
            .arg("--json"),
        "extended shader probe import",
    );
    assert_eq!(imported["ok"], true);

    let output_directory = oracle.directory.path().join("extended_shader_render");
    let rendered = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                project_text,
                "--engine",
                "path",
                "--format",
                "exr",
                "--out",
            ])
            .arg(&output_directory)
            .arg("--json"),
        "extended shader render",
    );
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .expect("CLI render returned a frame path");
    let image = read_probe_image(Path::new(frame_path));
    let shader_cases: Vec<&Value> = oracle
        .cases
        .iter()
        .filter(|case| case["shader_probe"].as_bool() == Some(true))
        .collect();
    assert_eq!(image.width, shader_cases.len() * 8);
    assert_eq!(image.height, 8);

    let mut mismatches = Vec::new();
    for (column, case) in shader_cases.iter().enumerate() {
        let pixel = image.pixels[4 * image.width + column * 8 + 4];
        let actual = if case["render_kind"] == "scalar" {
            json!(pixel[0])
        } else {
            json!(pixel)
        };
        let id = case["id"].as_str().unwrap();
        if let Some(mismatch) = json_mismatch(&actual, &oracle.shader[id], &format!("shader_{id}"))
        {
            mismatches.push(mismatch);
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} exact-pixel Potter material/render probes diverge from Blender 5.2.2 Cycles:\n{}",
        mismatches.len(),
        shader_cases.len(),
        mismatches.join("\n")
    );
    let realtime_cases: Vec<&Value> = realtime_cases
        .into_iter()
        .filter(|case| !captured["eevee"][case["id"].as_str().unwrap()].is_string())
        .collect();
    let realtime_directory = oracle
        .directory
        .path()
        .join("extended_shader_realtime_render");
    let rendered = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                project_text,
                "--engine",
                "realtime",
                "--format",
                "exr",
                "--out",
            ])
            .arg(&realtime_directory)
            .arg("--json"),
        "extended realtime shader render",
    );
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .expect("CLI realtime render returned a frame path");
    let image = read_probe_image(Path::new(frame_path));
    assert_eq!(image.width, shader_cases.len() * 8);
    assert_eq!(image.height, 8);

    let mut mismatches = Vec::new();
    for case in &realtime_cases {
        let column = shader_cases
            .iter()
            .position(|probe| probe["id"] == case["id"])
            .unwrap();
        let pixel = image.pixels[4 * image.width + column * 8 + 4];
        let actual = if case["render_kind"] == "scalar" {
            json!(realtime_film_precision(pixel[0]))
        } else {
            json!(pixel.map(realtime_film_precision))
        };
        let id = case["id"].as_str().unwrap();
        let expected = &captured["eevee"][id];
        match (&actual, expected) {
            (Value::Number(actual), Value::Number(expected)) => assert_eevee_film_close(
                actual.as_f64().unwrap(),
                expected.as_f64().unwrap(),
                &format!("realtime_shader_{id}"),
            ),
            (Value::Array(actual), Value::Array(expected)) if actual.len() == expected.len() => {
                for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eevee_film_close(
                        actual.as_f64().unwrap(),
                        expected.as_f64().unwrap(),
                        &format!("realtime_shader_{id}[{channel}]"),
                    );
                }
            }
            _ => mismatches.push(format!(
                "realtime_shader_{id}: committed Blender and Potter returned different value shapes"
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} Potter realtime probes diverge from the committed Blender 5.2.2 EEVEE capture:\n{}",
        mismatches.len(),
        realtime_cases.len(),
        mismatches.join("\n")
    );
}

#[test]
fn remaining_node_math_families_match_blender_522_geometry_cycles_and_eevee() {
    if blender_file::blender_executable().is_none() {
        eprintln!("Skipping extended node math parity: no Blender executable was found");
        return;
    }
    let oracle = run_oracle();
    let actual_geometry = blender_geometry(&oracle);
    let mut geometry_mismatches = Vec::new();
    for case in &oracle.cases {
        let id = case["id"].as_str().unwrap();
        if let Some(mismatch) = json_mismatch(
            &actual_geometry[id],
            &oracle.geometry[id],
            &format!("geometry_{id}"),
        ) {
            geometry_mismatches.push(mismatch);
        }
    }
    assert!(
        geometry_mismatches.is_empty(),
        "{} of {} Geometry Nodes evaluated-attribute cases diverge from Blender 5.2.2:\n{}",
        geometry_mismatches.len(),
        oracle.cases.len(),
        geometry_mismatches.join("\n")
    );
    compare_potter_shader_render(&oracle);
}
