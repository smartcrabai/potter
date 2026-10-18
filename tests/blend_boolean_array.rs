use std::{collections::HashMap, error::Error, fs, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::SceneDoc,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const BLENDER_REQUIRED: &str = "Blender is required for the array Boolean parity test: install Blender 5.2, or set POTTER_BLENDER to its executable";

const ARRAY_BOOLEAN_FIXTURE: &str = r#"
import bpy
import json
import os
import sys
from mathutils import Vector

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
        matrix = evaluated.matrix_world
        positions = [[float(value) for value in matrix @ vertex.co] for vertex in mesh.vertices]
        edges = [[positions[index] for index in edge.vertices] for edge in mesh.edges]
        faces = [[positions[index] for index in polygon.vertices] for polygon in mesh.polygons]
        volume = 0.0
        for face in faces:
            first = Vector(face[0])
            for index in range(1, len(face) - 1):
                volume += first.dot(Vector(face[index]).cross(Vector(face[index + 1]))) / 6.0
        return {
            "vertices": positions,
            "edges": edges,
            "faces": faces,
            "signed_volume": volume,
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
        expected[host.name] = sample(host)

with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output, separators=(",", ":"))

for obj in bpy.data.objects:
    for modifier in obj.modifiers:
        if modifier.type == "BOOLEAN":
            enabled = obj.name == "ArrayHostSeparatedFLOAT"
            modifier.show_viewport = enabled
            modifier.show_render = enabled

# The CLI acceptance flow renders this imported scene after applying one FLOAT Boolean.
bpy.ops.object.camera_add(location=(8.0, -10.0, 8.0))
camera = bpy.context.object
camera.name = "BooleanArrayCamera"
camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = camera
bpy.ops.object.light_add(type="AREA", location=(4.0, -4.0, 7.0))
bpy.context.object.data.energy = 1200.0
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

#[derive(Debug, Clone)]
struct Surface {
    vertices: Vec<DVec3>,
    edges: Vec<[DVec3; 2]>,
    faces: Vec<Vec<DVec3>>,
    signed_volume: f64,
}

fn point_from_json(value: &Value) -> TestResult<DVec3> {
    let coordinates = value
        .as_array()
        .filter(|coordinates| coordinates.len() == 3)
        .ok_or("Blender vertex must have three coordinates")?;
    let coordinate = |index: usize| {
        coordinates[index]
            .as_f64()
            .ok_or("Blender vertex coordinate is invalid")
    };
    Ok(DVec3::new(coordinate(0)?, coordinate(1)?, coordinate(2)?))
}

fn surface_from_blender(value: &Value) -> TestResult<Surface> {
    let vertices = value["vertices"]
        .as_array()
        .ok_or("Blender vertices are missing")?
        .iter()
        .map(point_from_json)
        .collect::<TestResult<Vec<_>>>()?;
    let edges = value["edges"]
        .as_array()
        .ok_or("Blender edges are missing")?
        .iter()
        .map(|edge| {
            let points = edge
                .as_array()
                .filter(|points| points.len() == 2)
                .ok_or("Blender edge must have two endpoints")?;
            Ok([point_from_json(&points[0])?, point_from_json(&points[1])?])
        })
        .collect::<TestResult<Vec<_>>>()?;
    let faces = value["faces"]
        .as_array()
        .ok_or("Blender faces are missing")?
        .iter()
        .map(|face| {
            face.as_array()
                .ok_or("Blender face ring is missing")?
                .iter()
                .map(point_from_json)
                .collect::<TestResult<Vec<_>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Surface {
        vertices,
        edges,
        faces,
        signed_volume: value["signed_volume"]
            .as_f64()
            .ok_or("Blender signed volume is missing")?,
    })
}

fn signed_volume(faces: &[Vec<DVec3>]) -> f64 {
    faces
        .iter()
        .map(|face| {
            let Some(first) = face.first() else {
                return 0.0;
            };
            (1..face.len() - 1)
                .map(|index| first.dot(face[index].cross(face[index + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum()
}

fn surface_from_mesh(mesh: &Mesh, world: DMat4) -> TestResult<Surface> {
    let vertices: Vec<_> = mesh
        .vertices
        .iter()
        .map(|vertex| world.transform_point3(vertex.co))
        .collect();
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .zip(&vertices)
        .map(|(vertex, position)| (vertex.id, *position))
        .collect();
    let edges = mesh
        .edges
        .iter()
        .map(|edge| {
            Ok([
                positions
                    .get(&edge.vertices[0])
                    .copied()
                    .ok_or("Mesh edge vertex is missing")?,
                positions
                    .get(&edge.vertices[1])
                    .copied()
                    .ok_or("Mesh edge vertex is missing")?,
            ])
        })
        .collect::<TestResult<Vec<_>>>()?;
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    positions
                        .get(id)
                        .copied()
                        .ok_or_else(|| "Mesh face vertex is missing".into())
                })
                .collect::<TestResult<Vec<_>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Surface {
        vertices,
        edges,
        signed_volume: signed_volume(&faces),
        faces,
    })
}

fn same_point(first: DVec3, second: DVec3) -> bool {
    first.distance(second) <= 1.0e-5
}

fn assert_vertices(actual: &[DVec3], expected: &[DVec3], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: vertex count");
    let mut unmatched = expected.to_vec();
    for vertex in actual {
        let Some(index) = unmatched
            .iter()
            .position(|candidate| same_point(*vertex, *candidate))
        else {
            panic!("{context}: no Blender vertex matches {vertex:?}");
        };
        unmatched.swap_remove(index);
    }
    assert!(
        unmatched.is_empty(),
        "{context}: Blender has unmatched vertices"
    );
}

fn same_edge(actual: [DVec3; 2], expected: [DVec3; 2]) -> bool {
    (same_point(actual[0], expected[0]) && same_point(actual[1], expected[1]))
        || (same_point(actual[0], expected[1]) && same_point(actual[1], expected[0]))
}

fn assert_edges(actual: &[[DVec3; 2]], expected: &[[DVec3; 2]], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: edge count");
    let mut unmatched = expected.to_vec();
    for edge in actual {
        let Some(index) = unmatched
            .iter()
            .position(|candidate| same_edge(*edge, *candidate))
        else {
            panic!("{context}: no Blender edge matches {edge:?}");
        };
        unmatched.swap_remove(index);
    }
    assert!(
        unmatched.is_empty(),
        "{context}: Blender has unmatched edges"
    );
}

fn same_oriented_ring(actual: &[DVec3], expected: &[DVec3]) -> bool {
    actual.len() == expected.len()
        && (0..actual.len()).any(|start| {
            (0..actual.len())
                .all(|index| same_point(actual[index], expected[(start + index) % expected.len()]))
        })
}

fn assert_boolean_geometry(actual: &Surface, expected: &Surface, context: &str) {
    assert_vertices(&actual.vertices, &expected.vertices, context);
    assert_edges(&actual.edges, &expected.edges, context);
    assert_eq!(
        actual.faces.len(),
        expected.faces.len(),
        "{context}: face count"
    );

    let mut unmatched = expected.faces.clone();
    for face in &actual.faces {
        let Some(index) = unmatched
            .iter()
            .position(|candidate| same_oriented_ring(face, candidate))
        else {
            panic!("{context}: no Blender face matches oriented ring {face:?}");
        };
        unmatched.swap_remove(index);
    }
    assert!(
        unmatched.is_empty(),
        "{context}: Blender has unmatched faces"
    );

    let volume_error = (actual.signed_volume - expected.signed_volume).abs();
    let volume_tolerance = if expected.signed_volume.abs() <= 1.0e-9 {
        1.0e-9
    } else {
        expected.signed_volume.abs() * 1.0e-6
    };
    assert!(
        volume_error <= volume_tolerance,
        "{context}: signed volume {} differs from Blender {}",
        actual.signed_volume,
        expected.signed_volume
    );
}

#[test]
fn boolean_array_box_cutter_layouts_match_blender_for_exact_and_float() -> TestResult<()> {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
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
    assert_eq!(imported["result"]["losses"], json!([]));
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
                        // The fixture clears `show_viewport`/`show_render` on the EXACT hosts after
                        // sampling Blender's expectation, so the target is restored to the state it was sampled in.
                        modifier.enabled = id == node_id;
                        if id == node_id {
                            modifier.show_viewport = true;
                            modifier.show_render = true;
                        }
                    }
                }
            }
            let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())
                .map_err(|error| std::io::Error::other(format!("{name}: {error}")))?;
            let world = DMat4::from_cols_array(
                &snapshot
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| format!("Potter has no evaluated node `{name}`"))?
                    .world_matrix,
            );
            let actual = surface_from_mesh(
                snapshot
                    .meshes
                    .get(node_id)
                    .ok_or_else(|| format!("Potter did not evaluate `{name}`"))?,
                world,
            )?;
            assert_boolean_geometry(
                &actual,
                &expected_surface,
                &format!("{layout}, {count} boxes, {solver}"),
            );
        }
    }

    let float_name = "ArrayHostSeparatedFLOAT";
    let float_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == float_name)
        .map(|(id, _)| id)
        .ok_or("Blender separated FLOAT host is missing")?;
    let boolean_id = document.nodes[float_id]
        .modifiers
        .iter()
        .find(|modifier| modifier.modifier_type == "boolean")
        .map(|modifier| modifier.id.as_str().to_owned())
        .ok_or("Blender separated FLOAT Boolean modifier is missing")?;
    let batch = root.join("apply_float_boolean.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document.revision,
            "operations": [{
                "op": "modifier.apply",
                "target": {"id": float_id.as_str()},
                "modifier_id": boolean_id,
            }]
        }))?,
    )?;
    let batch_path = batch.to_str().ok_or("Apply batch path is not UTF-8")?;
    run_pot(&["apply", project_path, "--file", batch_path])?;

    let expected_surface = surface_from_blender(&expected[float_name])?;
    let applied: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert!(
        applied.nodes[float_id]
            .modifiers
            .iter()
            .all(|modifier| modifier.modifier_type != "boolean"),
        "applying the FLOAT Boolean must remove its modifier"
    );
    let applied_snapshot = Snapshot::evaluate(&applied, &EvaluationContext::default())?;
    let applied_world = DMat4::from_cols_array(
        &applied_snapshot
            .nodes
            .get(float_id)
            .ok_or("applied FLOAT host is missing from the evaluated scene")?
            .world_matrix,
    );
    let applied_surface = surface_from_mesh(
        applied_snapshot
            .meshes
            .get(float_id)
            .ok_or("applied FLOAT host has no evaluated mesh")?,
        applied_world,
    )?;
    assert_boolean_geometry(&applied_surface, &expected_surface, "applied FLOAT Boolean");

    let preview = root.join("preview");
    let preview_path = preview.to_str().ok_or("Preview path is not UTF-8")?;
    run_pot(&[
        "preview",
        project_path,
        "--views",
        "top",
        "--size",
        "64",
        "--out",
        preview_path,
    ])?;
    assert!(
        preview.join("top.png").is_file(),
        "preview image was not written"
    );

    let render_dir = root.join("render");
    let render_path = render_dir.to_str().ok_or("Render path is not UTF-8")?;
    let rendered = run_pot(&[
        "render",
        project_path,
        "--engine",
        "path",
        "--format",
        "png",
        "--out",
        render_path,
    ])?;
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .ok_or("rendered frame path is missing")?;
    assert!(std::path::Path::new(frame_path).is_file());

    let exported_blend = root.join("applied_float.blend");
    let exported_blend_path = exported_blend.to_str().ok_or("Export path is not UTF-8")?;
    run_pot(&[
        "export",
        project_path,
        "--format",
        "blend",
        "--out",
        exported_blend_path,
        "--blender",
        &blender_path,
    ])?;
    let reopened_project = root.join("reopened");
    let reopened_path = reopened_project
        .to_str()
        .ok_or("Reopened project path is not UTF-8")?;
    run_pot(&["init", reopened_path])?;
    let reopened = run_pot(&[
        "import",
        reopened_path,
        "--file",
        exported_blend_path,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        &blender_path,
    ])?;
    assert_eq!(reopened["result"]["losses"], json!([]));
    let reopened_doc: SceneDoc =
        serde_json::from_slice(&fs::read(reopened_project.join("scene.json"))?)?;
    let reopened_id = reopened_doc
        .nodes
        .iter()
        .find(|(_, node)| node.name == float_name)
        .map(|(id, _)| id)
        .ok_or("reopened FLOAT host is missing")?;
    let reopened_snapshot = Snapshot::evaluate(&reopened_doc, &EvaluationContext::default())?;
    let reopened_world = DMat4::from_cols_array(
        &reopened_snapshot
            .nodes
            .get(reopened_id)
            .ok_or("reopened FLOAT host is missing from the evaluated scene")?
            .world_matrix,
    );
    let reopened_surface = surface_from_mesh(
        reopened_snapshot
            .meshes
            .get(reopened_id)
            .ok_or("reopened FLOAT host has no evaluated mesh")?,
        reopened_world,
    )?;
    assert_boolean_geometry(
        &reopened_surface,
        &expected_surface,
        "reopened FLOAT export",
    );
    Ok(())
}
