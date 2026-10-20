#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test fixtures use fixed valid inputs"
)]

//! Schema truth, Blender-default parity, and a small differential behavior guard for modifier
//! parameters (PR18 slice A).
//!
//! The defaults test compares every recorded default with Blender 5.2.2. The behavior test uses
//! only explicit Blender 5.2.2 scenarios with activating inputs or confirmed inert conditions;
//! every other parameter remains unexercised and is tracked outside the regression test.

use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::{Map, Value, json};
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

const GUARD_TARGET: &str = "guard_target";
const GUARD_TARGET_ALT: &str = "guard_target_alt";
const GUARD_GROUP: &str = "guard_group";
const GUARD_GROUP_ALT: &str = "guard_group_alt";
const GUARD_RIG: &str = "guard_rig";
const GUARD_RIG_ALT: &str = "guard_rig_alt";
const GUARD_CAMERA: &str = "guard_camera";
const GUARD_CAMERA_ALT: &str = "guard_camera_alt";
const GUARD_LATTICE: &str = "guard_lattice";
const GUARD_GRAPH: &str = "guard_graph";
const GUARD_GRAPH_ALT: &str = "guard_graph_alt";

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn param_table() -> Value {
    serde_json::from_str(PARAM_TABLE).unwrap()
}

fn write_mesh_cache(directory: &Path, format: &str) -> std::path::PathBuf {
    const POINTS: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let (path, bytes) = if format == "PC2" {
        let mut bytes = b"POINTCACHE2\0".to_vec();
        bytes.extend_from_slice(&1_i32.to_le_bytes());
        bytes.extend_from_slice(&8_i32.to_le_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2_i32.to_le_bytes());
        for offset in [0.0_f32, 2.0] {
            for [x, y, z] in POINTS {
                bytes.extend_from_slice(&(x + offset).to_le_bytes());
                bytes.extend_from_slice(&y.to_le_bytes());
                bytes.extend_from_slice(&z.to_le_bytes());
            }
        }
        (directory.join("guard.pc2"), bytes)
    } else {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2_i32.to_be_bytes());
        bytes.extend_from_slice(&8_i32.to_be_bytes());
        bytes.extend_from_slice(&0.0_f32.to_be_bytes());
        bytes.extend_from_slice(&1.0_f32.to_be_bytes());
        for offset in [0.0_f32, 2.0] {
            for [x, y, z] in POINTS {
                bytes.extend_from_slice(&(x + offset).to_be_bytes());
                bytes.extend_from_slice(&y.to_be_bytes());
                bytes.extend_from_slice(&z.to_be_bytes());
            }
        }
        (directory.join("guard.mdd"), bytes)
    };
    fs::write(&path, bytes).unwrap();
    path
}

fn write_sequence_cache(directory: &Path) -> std::path::PathBuf {
    let source = directory.join("sequence_source");
    init_scene(&source);
    let operations = [json!({"op":"node.create","id":"body","kind":"box","params":{"size":2}})];
    let applied = apply_operations(&source, &operations);
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let archive = directory.join("guard.abc");
    let exported = pot()
        .args(["export"])
        .arg(&source)
        .args(["--format", "alembic", "--allow-lossy", "--out"])
        .arg(&archive)
        .args(["--overwrite", "--json"])
        .output()
        .unwrap();
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stdout)
    );
    archive
}

fn scene_operations(modifier: &str, params: &Map<String, Value>, directory: &Path) -> Vec<Value> {
    let weights = |weight: f64| -> Vec<Value> {
        (0..8)
            .map(|vertex| json!({"vertex_id": vertex, "weight": weight}))
            .collect()
    };
    let varied: Vec<Value> = (0..8)
        .map(|vertex| {
            let weight = if vertex % 2 == 0 { 1.0 } else { 0.25 };
            json!({"vertex_id": vertex, "weight": weight})
        })
        .collect();
    let mut operations = vec![
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":2}}),
        json!({"op":"node.create","id":GUARD_TARGET,"kind":"box","params":{"size":1}}),
        json!({"op":"node.create","id":GUARD_TARGET_ALT,"kind":"box","params":{"size":3}}),
        json!({"op":"node.create","id":GUARD_RIG,"kind":"armature"}),
        json!({"op":"bone.create","target":{"id":GUARD_RIG},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]}),
        json!({"op":"bone.create","target":{"id":GUARD_RIG},"id":"guard_group_bone","name":GUARD_GROUP,"head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root"}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"root_group","name":"Root"}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"root_group","weights":weights(1.0)}),
        json!({"op":"node.create","id":GUARD_RIG_ALT,"kind":"armature"}),
        json!({"op":"bone.create","target":{"id":GUARD_RIG_ALT},"id":"root_alt","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]}),
        json!({"op":"bone.create","target":{"id":GUARD_RIG_ALT},"id":"guard_group_bone_alt","name":GUARD_GROUP,"head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root_alt"}),
        json!({"op":"bone.create","target":{"id":GUARD_RIG_ALT},"id":"guard_group_alt_bone","name":GUARD_GROUP_ALT,"head":[0.0,1.0,0.0],"tail":[0.0,2.0,0.0],"parent":"root_alt"}),
        json!({"op":"node.create","id":GUARD_CAMERA,"kind":"camera"}),
        json!({"op":"node.create","id":GUARD_CAMERA_ALT,"kind":"camera","transform":{"translation":[2.0,1.0,3.0]}}),
        json!({"op":"lattice.create","id":GUARD_LATTICE}),
        json!({"op":"lattice.update","target":{"id":GUARD_LATTICE},"set":{"points":[[0.0,-1.0,-1.0],[2.0,-1.0,-1.0],[0.0,1.0,-1.0],[2.0,1.0,-1.0],[0.0,-1.0,1.0],[2.0,-1.0,1.0],[0.0,1.0,1.0],[2.0,1.0,1.0]]}}),
        json!({"op":"graph.create","id":GUARD_GRAPH,"kind":"geometry"}),
        json!({"op":"graph.node_add","graph":GUARD_GRAPH,"id":"input","type":"NodeGroupInput"}),
        json!({"op":"graph.node_add","graph":GUARD_GRAPH,"id":"output","type":"NodeGroupOutput"}),
        json!({"op":"graph.interface_update","graph":GUARD_GRAPH,"set":{"inputs":[{"id":"geometry_input","name":"Geometry","socket_type":"geometry","default":null}],"outputs":[{"id":"geometry_output","name":"Geometry","socket_type":"geometry","default":null}]}}),
        json!({"op":"graph.link","graph":GUARD_GRAPH,"from_node":"input","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        json!({"op":"graph.create","id":GUARD_GRAPH_ALT,"kind":"geometry"}),
        json!({"op":"graph.node_add","graph":GUARD_GRAPH_ALT,"id":"input","type":"NodeGroupInput"}),
        json!({"op":"graph.node_add","graph":GUARD_GRAPH_ALT,"id":"output","type":"NodeGroupOutput"}),
        json!({"op":"graph.interface_update","graph":GUARD_GRAPH_ALT,"set":{"inputs":[{"id":"geometry_input","name":"Geometry","socket_type":"geometry","default":null}],"outputs":[{"id":"geometry_output","name":"Geometry","socket_type":"geometry","default":null}]}}),
        json!({"op":"graph.link","graph":GUARD_GRAPH_ALT,"from_node":"input","from_socket":"Geometry","to_node":"output","to_socket":"Geometry"}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":GUARD_GROUP,"name":GUARD_GROUP}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":GUARD_GROUP,"weights":varied}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":GUARD_GROUP_ALT,"name":GUARD_GROUP_ALT}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":GUARD_GROUP_ALT,"weights":weights(0.5)}),
        json!({"op":"physics.particle_emitter.create","target":{"id":GUARD_TARGET},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":0.8,"seed":3}}),
    ];
    if modifier == "mesh_cache" {
        let format = params
            .get("cache_format")
            .and_then(Value::as_str)
            .unwrap_or("MDD");
        let cache = write_mesh_cache(directory, format);
        operations.push(json!({"op":"resource.pack","id":"guard_cache","uri":cache.to_string_lossy(),"kind":"mesh_cache"}));
    } else if modifier == "mesh_sequence_cache" {
        let archive = write_sequence_cache(directory);
        operations.push(json!({"op":"resource.pack","id":"guard_sequence_cache","uri":archive.to_string_lossy(),"kind":"alembic"}));
    }
    if modifier == "explode" {
        operations.push(json!({"op":"physics.particle_emitter.create","target":{"id":"body"},"settings":{"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":0.8,"seed":3}}));
    }
    let physics = match modifier {
        "cloth" => Some((
            "physics.cloth.create",
            "physics_cloth",
            json!({"tension_stiffness":12.0}),
        )),
        "soft_body" => Some((
            "physics.soft_body.create",
            "physics_soft_body",
            json!({"goal_strength":0.7}),
        )),
        "collision" => Some((
            "physics.collision.create",
            "physics_collision",
            json!({"thickness":0.01}),
        )),
        "dynamic_paint" => Some((
            "physics.dynamic_paint.create",
            "physics_dynamic_paint",
            json!({"role":"canvas"}),
        )),
        "fluid" => Some((
            "physics.fluid.create",
            "physics_fluid",
            json!({"type":"liquid","resolution":4}),
        )),
        "particle_system" => Some((
            "physics.particle_emitter.create",
            "physics_particle_emitter",
            json!({"emit_from":"VERT","count":2,"frame_start":1.0,"frame_end":1.0,"lifetime":12.0,"physics_type":"NEWTON","normal_factor":0.8,"seed":3}),
        )),
        _ => None,
    };
    if let Some((op, modifier_id, settings)) = physics {
        operations.push(json!({"op":op,"target":{"id":"body"},"settings":settings}));
        if !params.is_empty() {
            operations.push(json!({"op":"modifier.update","target":{"id":"body"},"modifier_id":modifier_id,"set":{"params":params}}));
        }
    } else {
        operations.push(json!({"op":"modifier.create","target":{"id":"body"},"id":"under_test","type":modifier,"params":params}));
    }
    operations
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

/// The operation layer's rejection message for these params, or `None` when it accepts them.
fn operation_rejection(modifier: &str, params: &Map<String, Value>) -> Option<String> {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    init_scene(&scene);
    let apply = apply_operations(
        &scene,
        &scene_operations(modifier, params, directory.path()),
    );
    if apply.status.success() {
        return None;
    }
    let envelope: Value = serde_json::from_slice(&apply.stdout).unwrap_or(Value::Null);
    Some(
        envelope["error"]["message"]
            .as_str()
            .unwrap_or("<no message>")
            .to_owned(),
    )
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
    let actual_exceptions: Vec<_> = exceptions.keys().map(String::as_str).collect();
    assert_eq!(
        actual_exceptions,
        vec![
            "bevel.custom_profile",
            "mesh_sequence_cache.frame_offset",
            "mesh_sequence_cache.override_frame",
            "mesh_sequence_cache.scale",
            "nodes.inputs",
            "ocean.filepath",
            "uv_project.projectors",
        ],
        "Blender runtime exceptions must stay explicit and narrowly scoped"
    );
    let mut mismatches = Vec::new();
    for (modifier, spec) in table["modifiers"].as_object().unwrap() {
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
            let Some(blender) = fixture["modifiers"][modifier].get(name) else {
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
        "param_table defaults differ from Blender runtime values:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn unknown_parameter_keys_are_refused_by_the_operation_layer() {
    let unknown = Map::from_iter([("__not_a_blender_property__".to_owned(), json!(1))]);
    assert!(
        operation_rejection("array", &unknown).is_some_and(|message| message.contains("unknown")),
        "unknown parameter key was not refused"
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
