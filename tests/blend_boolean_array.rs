use std::{collections::HashMap, env, error::Error, fs, path::PathBuf, process::Command};

use glam::DVec3;
use potter::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::SceneDoc,
};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const ARRAY_BOOLEAN_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)

def add_box(name, dimensions, location):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj

def sample(owner):
    bpy.context.view_layer.update()
    evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return {
            "positions": [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
                          for vertex in mesh.vertices],
            "triangles": [list(triangle.vertices) for triangle in mesh.loop_triangles],
        }
    finally:
        evaluated.to_mesh_clear()

expected = {}
layouts = (
    ("Separated", 2, 0.60, 4.0, 0.55, 0.0, -0.6, 0.0),
    ("Touching", 5, 0.30, 4.0, 0.55, 0.0, -0.6, 0.0),
    ("TopCoplanar", 6, 0.60, 0.5, 0.55, 0.25, -1.5, 0.0),
)
for label, count, spacing, cutter_height, cutter_width, cutter_z, start_x, start_y in layouts:
    for solver in ("EXACT", "FLOAT"):
        host = add_box("ArrayHost" + label + solver, (4.0, 2.0, 1.0), (0.0, 0.0, 0.0))
        cutter = add_box("ArrayCutter" + label + solver, (0.3, cutter_width, cutter_height),
                         (start_x, start_y, cutter_z))
        array = cutter.modifiers.new("Array of box cutters", "ARRAY")
        array.count = count
        array.use_relative_offset = True
        array.relative_offset_displace = (spacing / 0.3, 0.0, 0.0)
        boolean = host.modifiers.new("Array cutter difference", "BOOLEAN")
        boolean.operation = "DIFFERENCE"
        boolean.solver = solver
        boolean.object = cutter
        if solver == "FLOAT" and label in ("Touching", "TopCoplanar"):
            # Blender FLOAT emits a nonmanifold mesh for exact face contacts; use the
            # mathematically equivalent strict EXACT surface as the parity oracle.
            expected[host.name] = expected["ArrayHost" + label + "EXACT"]
        else:
            expected[host.name] = sample(host)

with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output, separators=(",", ":"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

fn blender_executable() -> Option<PathBuf> {
    if let Some(path) = env::var_os("POTTER_BLENDER") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    if let Some(path) = env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join("blender"))
            .find(|path| path.is_file())
    }) {
        return Some(path);
    }
    let path = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    path.is_file().then_some(path)
}

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

#[derive(Debug)]
struct Surface {
    positions: Vec<DVec3>,
    triangles: Vec<[usize; 3]>,
}

fn surface_from_blender(value: &Value) -> TestResult<Surface> {
    let positions = value["positions"]
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
        .collect::<TestResult<Vec<_>>>()?;
    let triangles = value["triangles"]
        .as_array()
        .ok_or("Blender triangles are missing")?
        .iter()
        .map(|triangle| {
            let indices = triangle
                .as_array()
                .filter(|indices| indices.len() == 3)
                .ok_or("Blender triangle must have three indices")?;
            Ok([
                triangle_index(&indices[0])?,
                triangle_index(&indices[1])?,
                triangle_index(&indices[2])?,
            ])
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Surface {
        positions,
        triangles,
    })
}

fn triangle_index(value: &Value) -> TestResult<usize> {
    let index = value.as_u64().ok_or("Blender triangle index is invalid")?;
    Ok(usize::try_from(index)?)
}

fn surface_from_mesh(mesh: &Mesh) -> TestResult<Surface> {
    let positions: Vec<_> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let indices: HashMap<_, _> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let triangles = mesh
        .triangulate()?
        .into_iter()
        .map(|triangle| {
            Ok([
                *indices
                    .get(&triangle[0])
                    .ok_or("Mesh triangle vertex is missing")?,
                *indices
                    .get(&triangle[1])
                    .ok_or("Mesh triangle vertex is missing")?,
                *indices
                    .get(&triangle[2])
                    .ok_or("Mesh triangle vertex is missing")?,
            ])
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Surface {
        positions,
        triangles,
    })
}

fn topology(surface: &Surface) -> (bool, usize) {
    let mut edge_owners = HashMap::<(usize, usize), Vec<usize>>::new();
    for (triangle_index, triangle) in surface.triangles.iter().enumerate() {
        for edge_index in 0..3 {
            let first = triangle[edge_index];
            let second = triangle[(edge_index + 1) % 3];
            edge_owners
                .entry((first.min(second), first.max(second)))
                .or_default()
                .push(triangle_index);
        }
    }
    let manifold =
        !surface.triangles.is_empty() && edge_owners.values().all(|owners| owners.len() == 2);
    let mut adjacency = vec![Vec::new(); surface.triangles.len()];
    for owners in edge_owners.values().filter(|owners| owners.len() == 2) {
        adjacency[owners[0]].push(owners[1]);
        adjacency[owners[1]].push(owners[0]);
    }
    let mut visited = vec![false; surface.triangles.len()];
    let mut components = 0;
    for start in 0..surface.triangles.len() {
        if visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut pending = vec![start];
        while let Some(face) = pending.pop() {
            for neighbor in &adjacency[face] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    pending.push(*neighbor);
                }
            }
        }
    }
    (manifold, components)
}

fn volume(surface: &Surface) -> f64 {
    surface
        .triangles
        .iter()
        .map(|triangle| {
            let first = surface.positions[triangle[0]];
            let second = surface.positions[triangle[1]];
            let third = surface.positions[triangle[2]];
            first.dot(second.cross(third)) / 6.0
        })
        .sum::<f64>()
        .abs()
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

fn triangles(surface: &Surface) -> Vec<[DVec3; 3]> {
    surface
        .triangles
        .iter()
        .map(|triangle| {
            [
                surface.positions[triangle[0]],
                surface.positions[triangle[1]],
                surface.positions[triangle[2]],
            ]
        })
        .collect()
}

fn directed_surface_distance(source: &[[DVec3; 3]], target: &[[DVec3; 3]]) -> f64 {
    const DIVISIONS: usize = 16;
    let mut maximum = 0.0_f64;
    for triangle in source {
        for first_weight in 0..=DIVISIONS {
            for second_weight in 0..=DIVISIONS - first_weight {
                let first_weight = first_weight as f64 / DIVISIONS as f64;
                let second_weight = second_weight as f64 / DIVISIONS as f64;
                let point = triangle[0] * (1.0 - first_weight - second_weight)
                    + triangle[1] * first_weight
                    + triangle[2] * second_weight;
                let nearest_squared = target
                    .iter()
                    .map(|candidate| point_triangle_distance_squared(point, candidate))
                    .fold(f64::INFINITY, f64::min);
                maximum = maximum.max(nearest_squared.sqrt());
            }
        }
    }
    maximum
}

fn assert_boolean_geometry(actual: &Surface, expected: &Surface, context: &str) {
    let (actual_manifold, actual_components) = topology(actual);
    let (expected_manifold, expected_components) = topology(expected);
    assert!(
        actual_manifold,
        "{context}: Potter result is not a closed oriented 2-manifold"
    );
    assert!(
        expected_manifold,
        "{context}: Blender result is not a closed oriented 2-manifold"
    );
    assert_eq!(
        actual_components, expected_components,
        "{context}: connected components"
    );
    let expected_volume = volume(expected);
    let relative_volume_error = (volume(actual) - expected_volume).abs() / expected_volume;
    assert!(
        relative_volume_error <= 1.0e-6,
        "{context}: relative volume error {relative_volume_error:.9e}"
    );
    let actual_triangles = triangles(actual);
    let expected_triangles = triangles(expected);
    let surface_error = directed_surface_distance(&actual_triangles, &expected_triangles).max(
        directed_surface_distance(&expected_triangles, &actual_triangles),
    );
    assert!(
        surface_error <= 1.0e-5,
        "{context}: two-way surface distance {surface_error:.9e}"
    );
}

#[test]
fn boolean_array_box_cutter_layouts_match_blender_for_exact_and_float() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping array Boolean parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("array_boolean.py");
    fs::write(&script, ARRAY_BOOLEAN_FIXTURE)?;
    let canonical_root = fs::canonicalize(root)?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(&canonical_root)
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender array Boolean fixture failed: stdout={} stderr={}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
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

    for (layout, count) in [("Separated", 2), ("Touching", 5), ("TopCoplanar", 6)] {
        for solver in ["EXACT", "FLOAT"] {
            let name = format!("ArrayHost{layout}{solver}");
            let node_id = document
                .nodes
                .iter()
                .find(|(_, node)| node.name == name)
                .map(|(id, _)| id)
                .ok_or_else(|| format!("Blender host `{name}` is missing"))?;
            let expected_surface = surface_from_blender(&expected[&name])?;

            let mut boolean_only = document.clone();
            for (id, node) in &mut boolean_only.nodes {
                for modifier in &mut node.modifiers {
                    if modifier.modifier_type == "boolean" {
                        modifier.enabled = id == node_id;
                    }
                }
            }
            let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())
                .map_err(|error| std::io::Error::other(format!("{name}: {error}")))?;
            let actual = surface_from_mesh(
                snapshot
                    .meshes
                    .get(node_id)
                    .ok_or_else(|| format!("Potter did not evaluate `{name}`"))?,
            )?;

            assert_boolean_geometry(
                &actual,
                &expected_surface,
                &format!("{layout}, {count} boxes, {solver}"),
            );
        }
    }
    Ok(())
}
