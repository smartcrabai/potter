use std::{error::Error, fs, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use glam::DVec3;
use potter::{eval::Snapshot, model::SceneDoc};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const GEAR_STACK_FIXTURE: &str = r#"
import bpy
import json
import math
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def make_gear(name):
    teeth, root_radius, tip_radius, bore_radius, half_depth = 12, 0.66, 0.86, 0.2, 0.13
    outer = []
    for tooth in range(teeth):
        center = 2.0 * math.pi * tooth / teeth
        for offset, radius in ((-0.43, root_radius), (-0.24, tip_radius), (0.24, tip_radius), (0.43, root_radius)):
            angle = center + offset * (2.0 * math.pi / teeth)
            outer.append((radius * math.cos(angle), radius * math.sin(angle)))
    count = len(outer)
    vertices = []
    for z in (-half_depth, half_depth):
        vertices.extend((x, y, z) for x, y in outer)
        vertices.extend((bore_radius * math.cos(2.0 * math.pi * i / count),
                         bore_radius * math.sin(2.0 * math.pi * i / count), z)
                        for i in range(count))
    faces = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + next_index, count + index))
        faces.append((2 * count + index, 3 * count + index,
                      3 * count + next_index, 2 * count + next_index))
        faces.append((index, 2 * count + index, 2 * count + next_index, next_index))
        faces.append((count + index, count + next_index,
                      3 * count + next_index, 3 * count + index))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh

gear = bpy.data.objects.new("Gear_Array_Bevel_Subdivision_Normal", make_gear("DriveGearProfile"))
scene.collection.objects.link(gear)
gear.location = (-5.0, 0.0, 1.0)
bevel = gear.modifiers.new("Two segment bevel", "BEVEL")
bevel.width = 0.028
bevel.segments = 2
bevel.limit_method = "ANGLE"
array = gear.modifiers.new("Three copies with end cap", "ARRAY")
array.count = 3
array.use_relative_offset = True
array.relative_offset_displace = (1.55, 0.0, 0.0)
cap = bpy.data.objects.new("Gear_Array_EndCap_Source", make_gear("EndCapProfile"))
scene.collection.objects.link(cap)
cap.location = (-12.0, -8.0, -5.0)
cap.hide_render = True
cap.hide_set(True)
array.end_cap = cap
subdivision = gear.modifiers.new("Finishing subdivision", "SUBSURF")
subdivision.subdivision_type = "CATMULL_CLARK"
subdivision.levels = 1
subdivision.render_levels = 1
weighted_normal = gear.modifiers.new("Weighted face normals", "WEIGHTED_NORMAL")
weighted_normal.keep_sharp = True

def evaluated_positions():
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = gear.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)] for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()

prefixes = []
for prefix in range(0, len(gear.modifiers) + 1):
    for index, modifier in enumerate(gear.modifiers):
        modifier.show_viewport = index < prefix
    prefixes.append(evaluated_positions())
for modifier in gear.modifiers:
    modifier.show_viewport = True
bpy.context.view_layer.update()
with open(os.path.join(root, "blender_prefixes.json"), "w", encoding="utf-8") as output:
    json.dump(prefixes, output, separators=(",", ":"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

fn run_pot(args: &[&str]) -> TestResult<Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "pot {} failed: stdout={} stderr={}",
        args.first().copied().unwrap_or("command"),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "fixture coordinates are bounded to the finite gear profile"
)]
fn cell(point: DVec3, size: f64) -> [i64; 3] {
    [
        (point.x / size).floor() as i64,
        (point.y / size).floor() as i64,
        (point.z / size).floor() as i64,
    ]
}

fn directed_nearest_pair(points: &[DVec3], targets: &[DVec3]) -> (f64, usize, usize, DVec3, DVec3) {
    const CELL_SIZE: f64 = 0.5;
    let mut cells = std::collections::HashMap::<[i64; 3], Vec<(usize, DVec3)>>::new();
    for (index, target) in targets.iter().enumerate() {
        cells
            .entry(cell(*target, CELL_SIZE))
            .or_default()
            .push((index, *target));
    }
    let mut worst = (0.0_f64, 0, 0, DVec3::ZERO, DVec3::ZERO);
    for (point_index, point) in points.iter().enumerate() {
        let origin = cell(*point, CELL_SIZE);
        let mut nearest_squared = f64::INFINITY;
        let mut nearest = (0, DVec3::ZERO);
        for x in -1_i64..=1 {
            for y in -1_i64..=1 {
                for z in -1_i64..=1 {
                    let key = [origin[0] + x, origin[1] + y, origin[2] + z];
                    if let Some(candidates) = cells.get(&key) {
                        for (candidate_index, candidate) in candidates {
                            let distance_squared = point.distance_squared(*candidate);
                            if distance_squared < nearest_squared {
                                nearest_squared = distance_squared;
                                nearest = (*candidate_index, *candidate);
                            }
                        }
                    }
                }
            }
        }
        let error = nearest_squared.sqrt();
        if error > worst.0 {
            worst = (error, point_index, nearest.0, *point, nearest.1);
        }
    }
    worst
}

fn symmetric_nearest_pair(first: &[DVec3], second: &[DVec3]) -> (f64, usize, usize, DVec3, DVec3) {
    let forward = directed_nearest_pair(first, second);
    let reverse = directed_nearest_pair(second, first);
    if forward.0 >= reverse.0 {
        forward
    } else {
        (reverse.0, reverse.2, reverse.1, reverse.4, reverse.3)
    }
}

fn points_from_json(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("Blender prefix positions are missing")?
        .iter()
        .map(|point| {
            let coordinates = point
                .as_array()
                .filter(|coordinates| coordinates.len() == 3)
                .ok_or("Blender vertex must have three coordinates")?;
            let coordinate = |index: usize| {
                coordinates[index]
                    .as_f64()
                    .ok_or("Blender vertex coordinate is invalid")
            };
            Ok(DVec3::new(coordinate(0)?, coordinate(1)?, coordinate(2)?))
        })
        .collect()
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one fixture verifies the source mesh, bevel gate, and Blender prefix parity end to end"
)]
fn gear_bevel_array_subdivision_weighted_normal_stack_stops_at_unsupported_blender_corner()
-> TestResult<()> {
    const TOLERANCE: f64 = 1.0e-5;
    let Some(blender) = blender_executable() else {
        eprintln!("skipping gear stack Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("gear_stack.py");
    fs::write(&script, GEAR_STACK_FIXTURE)?;
    let canonical_root = fs::canonicalize(root)?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(&canonical_root)
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender gear fixture failed: stdout={} stderr={}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );
    let blender_prefixes: Value =
        serde_json::from_slice(&fs::read(root.join("blender_prefixes.json"))?)?;
    let prefixes = blender_prefixes
        .as_array()
        .ok_or("Blender did not return stack prefixes")?;
    assert_eq!(
        prefixes.len(),
        5,
        "the fixture must include the source plus four modifier prefixes"
    );

    let project = root.join("project");
    let project_path = project.to_str().ok_or("project path is not UTF-8")?;
    let source_path = root
        .join("source.blend")
        .to_str()
        .ok_or("blend path is not UTF-8")?
        .to_owned();
    let blender_path = blender
        .to_str()
        .ok_or("Blender path is not UTF-8")?
        .to_owned();
    run_pot(&["init", project_path])?;
    let imported = run_pot(&[
        "import",
        project_path,
        "--file",
        &source_path,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        &blender_path,
    ])?;
    assert_eq!(imported["result"]["losses"], serde_json::json!([]));
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let gear_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "Gear_Array_Bevel_Subdivision_Normal")
        .map(|(id, _)| id.clone())
        .ok_or("imported gear node is missing")?;
    assert_eq!(document.nodes[&gear_id].modifiers.len() + 1, prefixes.len());
    let cap_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "Gear_Array_EndCap_Source")
        .map(|(id, _)| id.clone())
        .ok_or("imported end-cap node is missing")?;
    assert_eq!(
        document.nodes[&gear_id].modifiers[1].params["end_cap"],
        serde_json::json!(cap_id.as_str())
    );
    let cap_only = Snapshot::evaluate_nodes_with_cache(
        &document,
        &potter::eval::EvaluationContext::default(),
        None,
        &std::collections::BTreeSet::from([cap_id.clone()]),
    )?;
    assert!(cap_only.meshes.contains_key(&cap_id));
    assert!(!cap_only.meshes.contains_key(&gear_id));
    let feature_catalog = potter::catalog::feature_catalog();
    let bevel_corner_feature = feature_catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "modifier.bevel.high_valence_vertex_mesh")
        })
        .ok_or("high-valence bevel corner is missing from the feature catalog")?;
    assert_eq!(bevel_corner_feature["status"], "not_supported");
    assert_eq!(bevel_corner_feature["capabilities"]["evaluate"], false);
    let preview = run_pot(&["preview", project_path, "--views", "iso", "--size", "64"])?;
    let warnings = preview["result"]["warnings"]
        .as_array()
        .ok_or("preview warnings are missing")?;
    assert!(
        warnings.iter().any(|warning| {
            warning["feature_id"] == "modifier.bevel.high_valence_vertex_mesh"
                && warning["node_id"] == gear_id.as_str()
        }),
        "preview must warn about the unsupported gear object: {warnings:?}"
    );

    let targeted_inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect", project_path, "--id", gear_id.as_str(), "--json"])
        .output()?;
    assert_eq!(targeted_inspect.status.code(), Some(4));
    let targeted_inspect: Value = serde_json::from_slice(&targeted_inspect.stdout)?;
    assert_eq!(targeted_inspect["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        targeted_inspect["error"]["details"]["feature_id"],
        "modifier.bevel.high_valence_vertex_mesh"
    );
    assert_eq!(
        targeted_inspect["error"]["details"]["node_id"],
        gear_id.as_str()
    );

    let whole_scene_inspect = run_pot(&["inspect", project_path])?;
    let gear_item = whole_scene_inspect["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["id"] == gear_id.as_str()))
        .ok_or("whole-scene inspect omitted the unsupported gear object")?;
    assert_eq!(gear_item["evaluation_error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        gear_item["evaluation_error"]["details"]["feature_id"],
        "modifier.bevel.high_valence_vertex_mesh"
    );

    for (index, blender_positions) in prefixes.iter().enumerate() {
        let mut prefix_document = document.clone();
        let gear = prefix_document
            .nodes
            .get_mut(&gear_id)
            .ok_or("gear node is missing")?;
        for (modifier_index, modifier) in gear.modifiers.iter_mut().enumerate() {
            modifier.enabled = modifier_index < index;
        }
        let snapshot = match Snapshot::evaluate(
            &prefix_document,
            &potter::eval::EvaluationContext::default(),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                assert_eq!(index, 1, "the source prefix must remain evaluable");
                assert_eq!(
                    error.code,
                    potter::error::ErrorCode::UnsupportedFeature,
                    "the first modifier prefix must use a typed feature gate"
                );
                assert_eq!(
                    error.details["feature_id"],
                    serde_json::json!("modifier.bevel.high_valence_vertex_mesh")
                );
                assert!(error.details["vertex_id"].is_number());
                assert_eq!(error.details["node_id"], serde_json::json!(gear_id));
                return Ok(());
            }
        };
        let evaluated = snapshot
            .meshes
            .get(&gear_id)
            .ok_or("Potter did not evaluate the gear mesh")?;
        let potter_positions: Vec<_> = evaluated.vertices.iter().map(|vertex| vertex.co).collect();
        let blender_positions = points_from_json(blender_positions)?;
        let (error, potter_index, blender_index, potter_point, blender_point) =
            symmetric_nearest_pair(&potter_positions, &blender_positions);
        assert!(
            error <= TOLERANCE,
            "gear stack prefix {} ({}): Potter has {} vertices, Blender has {}, maximum symmetric nearest-vertex error {error:.9e} exceeds {TOLERANCE:.1e}; nearest pair: Potter vertex {potter_index} {potter_point:?}, Blender vertex {blender_index} {blender_point:?}",
            index,
            ["source", "Bevel", "Array", "Subdivision", "Weighted Normal",][index],
            potter_positions.len(),
            blender_positions.len(),
        );
    }
    Ok(())
}
