use std::{collections::HashMap, error::Error, fs, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, SceneDoc},
};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const BOOLEAN_STACK_FIXTURE: &str = r#"
import bpy
import json
import math
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
def sample(owner):
    bpy.context.view_layer.update()
    evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return {
            "counts": [len(mesh.vertices), len(mesh.edges), len(mesh.polygons)],
            "positions": [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
                          for vertex in mesh.vertices],
            "triangles": [list(triangle.vertices) for triangle in mesh.loop_triangles],
        }
    finally:
        evaluated.to_mesh_clear()


prefixes = {}
for solver, host_name in (("EXACT", "HousingExact"), ("FLOAT", "HousingFloat")):
    bpy.ops.mesh.primitive_cube_add(size=1.8, location=(3.25, 0.0, 1.0))
    host = bpy.context.object
    host.name = host_name
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=40,
        radius=0.39,
        depth=2.8,
        location=(3.25, 0.0, 1.0),
        rotation=(math.radians(16), math.radians(8), math.radians(17)),
    )
    cutter = bpy.context.object
    cutter.name = host_name + "Cutter"
    boolean = host.modifiers.new("Exact topology probe", "BOOLEAN")
    boolean.operation = "DIFFERENCE"
    boolean.solver = solver
    boolean.object = cutter
    decimate = host.modifiers.new("Collapse topology probe", "DECIMATE")
    decimate.decimate_type = "COLLAPSE"
    decimate.ratio = 0.6
    solidify = host.modifiers.new("Solidify topology probe", "SOLIDIFY")
    solidify.thickness = 0.055
    results = []
    for prefix in range(1, len(host.modifiers) + 1):
        for index, modifier in enumerate(host.modifiers):
            modifier.show_viewport = index < prefix
        results.append(sample(host))
    for modifier in host.modifiers:
        modifier.show_viewport = True
    prefixes[host_name] = results




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

fn points_from_json(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("Blender vertex positions are missing")?
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

fn blender_triangles(geometry: &Value) -> TestResult<Vec<[DVec3; 3]>> {
    let positions = points_from_json(&geometry["positions"])?;
    geometry["triangles"]
        .as_array()
        .ok_or("Blender triangle indices are missing")?
        .iter()
        .map(|triangle| {
            let indices = triangle
                .as_array()
                .filter(|indices| indices.len() == 3)
                .ok_or("Blender triangle must have three indices")?;
            let index = |position: usize| {
                indices[position]
                    .as_u64()
                    .map(|index| index as usize)
                    .ok_or("Blender triangle index is invalid")
            };
            Ok([
                *positions
                    .get(index(0)?)
                    .ok_or("Blender triangle vertex is missing")?,
                *positions
                    .get(index(1)?)
                    .ok_or("Blender triangle vertex is missing")?,
                *positions
                    .get(index(2)?)
                    .ok_or("Blender triangle vertex is missing")?,
            ])
        })
        .collect()
}

fn potter_triangles(mesh: &Mesh) -> TestResult<Vec<[DVec3; 3]>> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    mesh.triangulate()?
        .iter()
        .map(|triangle| {
            Ok([
                *positions
                    .get(&triangle[0])
                    .ok_or("Potter triangle vertex is missing")?,
                *positions
                    .get(&triangle[1])
                    .ok_or("Potter triangle vertex is missing")?,
                *positions
                    .get(&triangle[2])
                    .ok_or("Potter triangle vertex is missing")?,
            ])
        })
        .collect()
}

fn point_triangle_distance_squared(point: DVec3, triangle: &[DVec3; 3]) -> f64 {
    fn segment_distance_squared(point: DVec3, first: DVec3, second: DVec3) -> f64 {
        let edge = second - first;
        let length_squared = edge.length_squared();
        let parameter = if length_squared > f64::MIN_POSITIVE {
            ((point - first).dot(edge) / length_squared).clamp(0.0, 1.0)
        } else {
            0.0
        };
        point.distance_squared(first + edge * parameter)
    }

    let [a, b, c] = *triangle;
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);
    let normal_squared = normal.length_squared();
    if normal_squared > f64::MIN_POSITIVE {
        let projection = point - normal * (normal.dot(point - a) / normal_squared);
        let projected = projection - a;
        let d00 = ab.dot(ab);
        let d01 = ab.dot(ac);
        let d11 = ac.dot(ac);
        let d20 = projected.dot(ab);
        let d21 = projected.dot(ac);
        let denominator = d00 * d11 - d01 * d01;
        if denominator > f64::MIN_POSITIVE {
            let u = (d11 * d20 - d01 * d21) / denominator;
            let v = (d00 * d21 - d01 * d20) / denominator;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                return point.distance_squared(projection);
            }
        }
    }
    segment_distance_squared(point, a, b)
        .min(segment_distance_squared(point, b, c))
        .min(segment_distance_squared(point, c, a))
}

fn directed_surface_distance(source: &[[DVec3; 3]], target: &[[DVec3; 3]]) -> f64 {
    const DIVISIONS: usize = 16;
    source
        .iter()
        .flat_map(|triangle| {
            (0..=DIVISIONS).flat_map(move |first_index| {
                (0..=DIVISIONS - first_index).map(move |second_index| {
                    let first_weight = first_index as f64 / DIVISIONS as f64;
                    let second_weight = second_index as f64 / DIVISIONS as f64;
                    triangle[0] * (1.0 - first_weight - second_weight)
                        + triangle[1] * first_weight
                        + triangle[2] * second_weight
                })
            })
        })
        .map(|point| {
            target
                .iter()
                .map(|triangle| point_triangle_distance_squared(point, triangle))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .fold(0.0_f64, f64::max)
}

fn symmetric_surface_distance(first: &[[DVec3; 3]], second: &[[DVec3; 3]]) -> f64 {
    directed_surface_distance(first, second).max(directed_surface_distance(second, first))
}

fn find_node(document: &SceneDoc, name: &str) -> TestResult<Id> {
    document
        .nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("imported node `{name}` is missing").into())
}
fn disable_other_modifier_stacks(document: &mut SceneDoc, enabled_node: &Id) {
    for (node_id, node) in &mut document.nodes {
        if node_id != enabled_node {
            for modifier in &mut node.modifiers {
                modifier.enabled = false;
            }
        }
    }
}

fn assert_catalog_gate() -> TestResult<()> {
    let catalog = potter_core::catalog::feature_catalog();
    let row = catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "modifier.boolean.downstream_topology")
        })
        .ok_or("Boolean topology limitation is missing from the feature catalog")?;
    assert_eq!(row["status"], "not_supported");
    assert_eq!(row["capabilities"]["evaluate"], false);
    Ok(())
}

#[test]
fn boolean_decimate_solidify_topology_divergence_is_gated_for_exact_and_float() -> TestResult<()> {
    const TOLERANCE: f64 = 1.0e-5;
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Boolean stack Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("boolean_stack.py");
    fs::write(&script, BOOLEAN_STACK_FIXTURE)?;
    let canonical_root = fs::canonicalize(root)?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(&canonical_root)
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender Boolean fixture failed: stdout={} stderr={}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );
    let blender_prefixes: Value =
        serde_json::from_slice(&fs::read(root.join("blender_prefixes.json"))?)?;
    let source_path = root
        .join("source.blend")
        .to_str()
        .ok_or("Blend path is not UTF-8")?
        .to_owned();
    let project = root.join("project");
    let project_path = project.to_str().ok_or("project path is not UTF-8")?;
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
    assert_catalog_gate()?;

    for (solver, name) in [("EXACT", "HousingExact"), ("FLOAT", "HousingFloat")] {
        let node_id = find_node(&document, name)?;
        let expected_prefixes = blender_prefixes[name]
            .as_array()
            .filter(|prefixes| prefixes.len() == 3)
            .ok_or_else(|| format!("Blender {solver} modifier prefixes are missing"))?;
        if solver == "EXACT" {
            assert_eq!(
                expected_prefixes[0]["counts"],
                serde_json::json!([88, 136, 48])
            );
        }

        let mut boolean_only = document.clone();
        disable_other_modifier_stacks(&mut boolean_only, &node_id);
        let node = boolean_only
            .nodes
            .get_mut(&node_id)
            .ok_or("Boolean host node is missing")?;
        assert_eq!(node.modifiers.len(), 3);
        for (index, modifier) in node.modifiers.iter_mut().enumerate() {
            modifier.enabled = index == 0;
        }
        let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())?;
        let potter_mesh = snapshot
            .meshes
            .get(&node_id)
            .ok_or("Potter did not evaluate the Boolean-only mesh")?;
        let potter_triangles = potter_triangles(potter_mesh)?;
        let blender_triangles = blender_triangles(&expected_prefixes[0])?;
        let surface_error = symmetric_surface_distance(&potter_triangles, &blender_triangles);
        assert!(
            surface_error <= TOLERANCE,
            "{solver} Boolean surface differs from Blender by {surface_error:.9e}, over {TOLERANCE:.1e}"
        );

        let mut full_stack = document.clone();
        disable_other_modifier_stacks(&mut full_stack, &node_id);
        let full_stack_error = Snapshot::evaluate(&full_stack, &EvaluationContext::default())
            .err()
            .ok_or("Boolean followed by Decimate and Solidify must be gated")?;
        assert_eq!(full_stack_error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            full_stack_error.details["feature_id"],
            "modifier.boolean.downstream_topology"
        );
        assert_eq!(full_stack_error.details["solver"], solver);
        assert_eq!(
            full_stack_error.details["downstream_modifier_type"],
            "decimate"
        );

        let mut solidify_only = document.clone();
        disable_other_modifier_stacks(&mut solidify_only, &node_id);
        let node = solidify_only
            .nodes
            .get_mut(&node_id)
            .ok_or("Boolean host node is missing")?;
        for modifier in &mut node.modifiers {
            modifier.enabled =
                modifier.modifier_type == "boolean" || modifier.modifier_type == "solidify";
        }
        let solidify_error = Snapshot::evaluate(&solidify_only, &EvaluationContext::default())
            .err()
            .ok_or("Boolean followed by Solidify must be gated")?;
        assert_eq!(solidify_error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            solidify_error.details["feature_id"],
            "modifier.boolean.downstream_topology"
        );
        assert_eq!(
            solidify_error.details["downstream_modifier_type"],
            "solidify"
        );
    }
    let exact_node_id = find_node(&document, "HousingExact")?;
    let exact_boolean_id = &document.nodes[&exact_node_id].modifiers[0].id;
    let apply_batch = root.join("apply_boolean.json");
    fs::write(
        &apply_batch,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "base_revision": 1,
            "operations": [{
                "op": "modifier.apply",
                "target": {"id": exact_node_id},
                "modifier_id": exact_boolean_id
            }]
        }))?,
    )?;
    let apply_output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(project_path)
        .arg("--file")
        .arg(&apply_batch)
        .arg("--json")
        .output()?;
    assert!(
        !apply_output.status.success(),
        "applying a Boolean before topology-dependent modifiers must be rejected"
    );
    let apply_error: Value = serde_json::from_slice(&apply_output.stdout)?;
    assert_eq!(apply_error["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        apply_error["error"]["details"]["feature_id"],
        "modifier.boolean.downstream_topology"
    );
    Ok(())
}
