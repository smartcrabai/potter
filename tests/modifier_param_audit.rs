#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test fixtures use fixed valid inputs"
)]

//! F-specific schema truth, Blender-default parity, and behavior guards (PR18 slice A).
//!
//! Defaults checks compare F-owned modifier rows with Blender 5.2.2. Behavior scenarios are
//! filtered to F-owned modifiers; unexercised parameters remain outside this regression test.

use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const PARAM_TABLE: &str = include_str!("../crates/potter-core/src/param_table.json");
const BLENDER_DEFAULTS: &str = include_str!("fixtures/param_audit/blender_defaults.json");
const BEHAVIOR_SCENARIOS: &str = include_str!("fixtures/param_audit/behavior_scenarios.json");

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

/// Owners of the measured Blender divergences in `behavior_scenarios.json`.
const KNOWN_GAP_OWNERS: &[&str] = &[];

const F_MODIFIERS: &[&str] = &[
    "armature",
    "cast",
    "corrective_smooth",
    "displace",
    "hook",
    "laplacian_deform",
    "laplacian_smooth",
    "lattice",
    "mesh_deform",
    "simple_deform",
    "skin",
    "smooth",
    "surface_deform",
    "volume_displace",
    "warp",
    "wave",
];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn param_table() -> Value {
    serde_json::from_str(PARAM_TABLE).unwrap()
}

fn init_scene(scene: &Path) {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn apply_operations(scene: &Path, operations: &[Value]) -> Output {
    let batch = scene.with_extension("json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))
        .unwrap(),
    )
    .unwrap();
    pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()
        .unwrap()
}

/// Blender stores floats as float32; the table keeps the shortest decimal of the same value.
#[expect(
    clippy::float_cmp,
    reason = "exact equality after float32 conversion matches Blender's stored defaults"
)]
fn runtime_values_equal(table: &Value, blender: &Value) -> bool {
    match (table, blender) {
        (Value::Number(a), Value::Number(b)) => match (a.as_i64(), b.as_i64()) {
            (Some(a), Some(b)) => a == b,
            _ => (a.as_f64().unwrap() as f32) == (b.as_f64().unwrap() as f32),
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| runtime_values_equal(a, b))
        }
        _ => table == blender,
    }
}

#[test]
fn param_table_defaults_match_blender_fixture() {
    let table = param_table();
    let fixture: Value = serde_json::from_str(BLENDER_DEFAULTS).unwrap();
    assert_eq!(fixture["blender_version"], "5.2.2 LTS");
    let exceptions = fixture["exceptions"].as_object().unwrap();
    let mut mismatches = Vec::new();
    for modifier in F_MODIFIERS {
        let Some(spec) = table["modifiers"].get(*modifier) else {
            mismatches.push(format!("{modifier}: missing from parameter table"));
            continue;
        };
        let Some(blender_modifier) = fixture["modifiers"].get(*modifier) else {
            mismatches.push(format!("{modifier}: missing from Blender runtime fixture"));
            continue;
        };
        for (name, schema) in spec["properties"].as_object().unwrap() {
            let Some(default) = schema.get("default") else {
                continue;
            };
            let key = format!("{modifier}.{name}");
            if let Some(exception) = exceptions.get(&key) {
                if exception["reason"].as_str().is_none_or(str::is_empty) {
                    mismatches.push(format!("{key}: exception needs a reason"));
                }
                continue;
            }
            let Some(blender) = blender_modifier.get(name) else {
                mismatches.push(format!(
                    "{key}: missing from Blender runtime fixture and exception list"
                ));
                continue;
            };
            if !runtime_values_equal(default, blender) {
                mismatches.push(format!("{key}: table={default} blender={blender}"));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "F modifier parameter defaults differ from Blender runtime values:\n{}",
        mismatches.join("\n")
    );
}

const BLENDER_SCENARIO_SCRIPT: &str = r#"
import bpy
import json
import sys

if bpy.app.version != (5, 2, 2):
    print("POTTER_BLENDER_VERSION_SKIP=" + str(bpy.app.version))
    sys.exit(0)

cases_path = sys.argv[sys.argv.index("--") + 1]
cases = json.load(open(cases_path, encoding="utf-8"))
results = []
for case in cases:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if case["primitive"] == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0)
    else:
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object
    modifier = obj.modifiers.new(name=case["id"], type=case["modifier"].upper())
    for name, value in case["params"].items():
        setattr(modifier, name, tuple(value) if isinstance(value, list) else value)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    results.append({
        "id": case["id"],
        "state": case["state"],
        "geometry": {
            "positions": [[float(component) for component in vertex.co] for vertex in mesh.vertices],
            "faces": [list(face.vertices) for face in mesh.polygons],
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
        },
    })
    evaluated.to_mesh_clear()
print("POTTER_PARAM_AUDIT=" + json.dumps({"results": results}, sort_keys=True))
"#;

fn blender_scenario_results(scenarios: &Value) -> Option<BTreeMap<String, String>> {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping modifier parameter Blender scenarios: Blender is unavailable");
        return None;
    };
    let directory = tempdir().unwrap();
    let cases_path = directory.path().join("cases.json");
    let mut cases = Vec::new();
    for scenario in scenarios["scenarios"].as_array().unwrap() {
        for (state, key) in [("baseline", "baseline_params"), ("case", "case_params")] {
            cases.push(json!({
                "id": scenario["id"],
                "state": state,
                "primitive": scenario["primitive"],
                "modifier": scenario["modifier"],
                "params": scenario[key],
            }));
        }
    }
    fs::write(&cases_path, serde_json::to_vec(&cases).unwrap()).unwrap();
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
        "--python-expr",
        BLENDER_SCENARIO_SCRIPT,
        "--",
    ]);
    command.arg(&cases_path);
    let output = process::run_guarded(command).unwrap();
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "Blender modifier scenarios failed:\n{text}"
    );
    if text.contains("POTTER_BLENDER_VERSION_SKIP=") {
        assert!(
            env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
            "POTTER_REQUIRE_BLENDER is set but Blender is not 5.2.2:\n{text}"
        );
        eprintln!("skipping modifier parameter scenarios: this fixture requires Blender 5.2.2");
        return None;
    }
    let payload = text
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_PARAM_AUDIT="))
        .expect("Blender did not report modifier scenarios");
    let report: Value = serde_json::from_str(payload).unwrap();
    let mut results = BTreeMap::new();
    for item in report["results"].as_array().unwrap() {
        let key = format!(
            "{}:{}",
            item["id"].as_str().unwrap(),
            item["state"].as_str().unwrap()
        );
        results.insert(key, geometry_signature(&item["geometry"]));
    }
    Some(results)
}

fn rounded_point(point: &Value) -> Vec<String> {
    point
        .as_array()
        .unwrap()
        .iter()
        .map(|component| {
            let value = component.as_f64().unwrap();
            let rounded = (value * 100_000.0).round() / 100_000.0;
            format!("{:.5}", if rounded == 0.0 { 0.0 } else { rounded })
        })
        .collect()
}

/// Canonical position and face-set signatures ignore vertex/face storage order, as the Blender
/// audit oracle does, while retaining geometry and topology changes.
fn geometry_signature(geometry: &Value) -> String {
    let positions: Vec<Vec<String>> = geometry["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(rounded_point)
        .collect();
    let mut sorted_positions = positions.clone();
    sorted_positions.sort();
    let mut face_sets: Vec<Vec<Vec<String>>> = geometry["faces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|face| {
            let mut vertices: Vec<Vec<String>> = face
                .as_array()
                .unwrap()
                .iter()
                .map(|index| positions[index.as_u64().unwrap() as usize].clone())
                .collect();
            vertices.sort();
            vertices
        })
        .collect();
    face_sets.sort();
    let canonical = json!({
        "vertex_count": geometry["vertex_count"].as_u64().unwrap(),
        "edge_count": geometry["edge_count"].as_u64().unwrap(),
        "face_count": geometry["face_count"].as_u64().unwrap(),
        "positions": sorted_positions,
        "faces": face_sets,
    });
    hex::encode(Sha256::digest(serde_json::to_vec(&canonical).unwrap()))
}

fn potter_scenario_geometry(primitive: &str, modifier: &str, params: &Value) -> String {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init_scene(&scene);
    let operations = [
        json!({"op":"node.create","id":"body","kind":primitive,"params":{"size":2.0}}),
        json!({"op":"modifier.create","target":{"id":"body"},"id":"under_test","type":modifier,"params":params}),
    ];
    let applied = apply_operations(&scene, &operations);
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspected = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let envelope: Value = serde_json::from_slice(&inspected.stdout).unwrap();
    geometry_signature(&envelope["result"]["items"][0]["evaluated_geometry"])
}

fn scenario_signature<'a>(results: &'a BTreeMap<String, String>, id: &str, state: &str) -> &'a str {
    results
        .get(&format!("{id}:{state}"))
        .map(String::as_str)
        .expect("Blender scenario result is missing")
}

#[test]
fn f_modifier_parameter_scenarios_match_blender() {
    let mut scenarios: Value = serde_json::from_str(BEHAVIOR_SCENARIOS).unwrap();
    // The shared inventory also contains non-F modifiers; this audit covers only F-owned rows.
    scenarios["scenarios"]
        .as_array_mut()
        .unwrap()
        .retain(|scenario| F_MODIFIERS.contains(&scenario["modifier"].as_str().unwrap()));
    assert_ne!(
        scenarios["scenarios"].as_array().unwrap().as_slice(),
        &[] as &[Value]
    );
    let Some(blender) = blender_scenario_results(&scenarios) else {
        return;
    };
    for scenario in scenarios["scenarios"].as_array().unwrap() {
        let id = scenario["id"].as_str().unwrap();
        let primitive = scenario["primitive"].as_str().unwrap();
        let modifier = scenario["modifier"].as_str().unwrap();
        let blender_baseline = scenario_signature(&blender, id, "baseline");
        let blender_case = scenario_signature(&blender, id, "case");
        let potter_baseline =
            potter_scenario_geometry(primitive, modifier, &scenario["baseline_params"]);
        let potter_case = potter_scenario_geometry(primitive, modifier, &scenario["case_params"]);
        match scenario["expectation"].as_str().unwrap() {
            "equal_active" => {
                assert_ne!(
                    blender_baseline, blender_case,
                    "{id}: Blender fixture is inactive"
                );
                assert_ne!(
                    potter_baseline, potter_case,
                    "{id}: Potter fixture is inactive"
                );
                assert_eq!(
                    potter_case, blender_case,
                    "{id}: Potter differs from Blender"
                );
            }
            "inert" => {
                assert_eq!(
                    blender_baseline, blender_case,
                    "{id}: expected Blender no-op"
                );
                assert_eq!(potter_baseline, potter_case, "{id}: expected Potter no-op");
                assert_eq!(
                    potter_baseline, blender_baseline,
                    "{id}: no-op baseline differs"
                );
            }
            expectation => panic!("{id}: unknown scenario expectation {expectation}"),
        }
    }
}

#[test]
fn modifier_parameter_scenarios_match_blender_or_record_real_gaps() {
    let scenarios: Value = serde_json::from_str(BEHAVIOR_SCENARIOS).unwrap();
    let Some(blender) = blender_scenario_results(&scenarios) else {
        return;
    };
    for scenario in scenarios["scenarios"].as_array().unwrap() {
        let id = scenario["id"].as_str().unwrap();
        let primitive = scenario["primitive"].as_str().unwrap();
        let modifier = scenario["modifier"].as_str().unwrap();
        let blender_baseline = scenario_signature(&blender, id, "baseline");
        let blender_case = scenario_signature(&blender, id, "case");
        let potter_baseline =
            potter_scenario_geometry(primitive, modifier, &scenario["baseline_params"]);
        let potter_case = potter_scenario_geometry(primitive, modifier, &scenario["case_params"]);
        match scenario["expectation"].as_str().unwrap() {
            "equal_active" => {
                assert_ne!(
                    blender_baseline, blender_case,
                    "{id}: Blender fixture is inactive"
                );
                assert_ne!(
                    potter_baseline, potter_case,
                    "{id}: Potter fixture is inactive"
                );
                assert_eq!(
                    potter_case, blender_case,
                    "{id}: Potter differs from Blender"
                );
            }
            "inert" => {
                assert_eq!(
                    blender_baseline, blender_case,
                    "{id}: expected Blender no-op"
                );
                assert_eq!(potter_baseline, potter_case, "{id}: expected Potter no-op");
                assert_eq!(
                    potter_baseline, blender_baseline,
                    "{id}: no-op baseline differs"
                );
            }
            "known_gap" => {
                let owner = scenario["owner"]
                    .as_str()
                    .expect("known gap needs an owner");
                assert!(
                    KNOWN_GAP_OWNERS.contains(&owner),
                    "{id}: unknown parity-gap owner {owner}"
                );
                assert_ne!(
                    blender_baseline, blender_case,
                    "{id}: the tested Blender parameter is inactive"
                );
                assert_ne!(
                    potter_case, blender_case,
                    "{id}: recorded parity gap has been fixed; update the scenario inventory"
                );
                assert_eq!(
                    potter_baseline != potter_case,
                    scenario["potter_changes"].as_bool().unwrap(),
                    "{id}: Potter behavior changed; update the scenario inventory"
                );
            }
            expectation => panic!("{id}: unknown scenario expectation {expectation}"),
        }
    }
}
