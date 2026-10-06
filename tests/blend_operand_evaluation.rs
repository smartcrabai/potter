use std::{collections::HashMap, error::Error, fs, path::Path};

use glam::DVec3;
use potter::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, Modifier, SceneDoc},
};
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/pot_json.rs"]
mod pot_json_helper;

use blender_file::blender_executable;
use pot_json_helper::pot_json;

type TestResult<T> = Result<T, Box<dyn Error>>;

const FRAME: f64 = 5.0;
const TOLERANCE: f64 = 1.0e-4;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
if bpy.app.version != (5, 2, 2):
    raise RuntimeError("operand regression fixture requires Blender 5.2.2")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_set(5)


def make_mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def grid(name, axis_values, height):
    vertices = [(float(x), float(y), float(height(x, y)))
                for y in axis_values for x in axis_values]
    faces = []
    width = len(axis_values)
    for y in range(width - 1):
        for x in range(width - 1):
            index = y * width + x
            faces.append((index, index + 1, index + width + 1, index + width))
    return make_mesh_object(name, vertices, faces)


def cube_vertices(size):
    x, y, z = (component * 0.5 for component in size)
    return [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
            (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]

cube_faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

# Boolean: three arrayed cutters carry the second material slot into the result.
host = make_mesh_object("BooleanHost", cube_vertices((4.8, 2.0, 2.0)), cube_faces)
base_material = bpy.data.materials.new("BooleanBase")
cut_material = bpy.data.materials.new("BooleanCut")
for material in (base_material, cut_material):
    host.data.materials.append(material)
for polygon in host.data.polygons:
    polygon.material_index = 0
cutter = make_mesh_object(
    "BooleanCutter", cube_vertices((0.5, 3.0, 0.65)), cube_faces)
cutter.location.x = -1.25
for material in (base_material, cut_material):
    cutter.data.materials.append(material)
for polygon in cutter.data.polygons:
    polygon.material_index = 1
array = cutter.modifiers.new("Three cutter instances", "ARRAY")
array.count = 3
array.use_relative_offset = True
array.relative_offset_displace = (2.5, 0.0, 0.0)
boolean = host.modifiers.new("Boolean with arrayed operand", "BOOLEAN")
boolean.operation = "DIFFERENCE"
boolean.solver = "EXACT"
boolean.material_mode = "TRANSFER"
boolean.object = cutter

# Shrinkwrap must see the evaluated Catmull-Clark surface, not its coarse control grid.
shrink_target = grid(
    "ShrinkTarget", [-2, -1, 0, 1, 2],
    lambda x, y: 0.16 * x * x + 0.13 * x * y - 0.09 * y * y)
subdivision = shrink_target.modifiers.new("Subdivided shrink target", "SUBSURF")
subdivision.subdivision_type = "CATMULL_CLARK"
subdivision.levels = 2
subdivision.render_levels = 2
shrink_owner = make_mesh_object(
    "ShrinkOwner",
    [(-1.35, -1.1, 2.5), (0.35, -1.2, 2.5),
     (1.25, 0.75, 2.5), (-0.25, 1.35, 2.5)],
    [(0, 1, 2, 3)])
shrink = shrink_owner.modifiers.new("Shrink onto subdivided target", "SHRINKWRAP")
shrink.target = shrink_target
shrink.wrap_method = "NEAREST_SURFACEPOINT"
shrink.wrap_mode = "ON_SURFACE"

# Surface Deform is bound against the target's evaluated subdivision stack.
surface_target = grid(
    "SurfaceTarget", [-2, -1, 0, 1, 2],
    lambda x, y: 0.11 * x * x + 0.12 * x * y + 0.07 * y * y)
surface_target.location = (0.35, -0.2, 0.15)
surface_subdivision = surface_target.modifiers.new("Subdivided deform target", "SUBSURF")
surface_subdivision.subdivision_type = "CATMULL_CLARK"
surface_subdivision.levels = 2
surface_subdivision.render_levels = 2
surface_owner = make_mesh_object(
    "SurfaceOwner",
    [(-0.65, -0.65, 0.5), (0.65, -0.65, 0.5),
     (0.65, 0.65, 0.5), (-0.65, 0.65, 0.5)],
    [(0, 1, 2, 3)])
surface_owner.location = (-0.1, 0.2, 0.0)
surface_deform = surface_owner.modifiers.new("Bound to subdivided target", "SURFACE_DEFORM")
surface_deform.target = surface_target
bpy.ops.object.select_all(action="DESELECT")
surface_owner.select_set(True)
bpy.context.view_layer.objects.active = surface_owner
bpy.ops.object.surfacedeform_bind(modifier=surface_deform.name)
if not surface_deform.is_bound:
    raise RuntimeError("Surface Deform did not bind to the target")

# Surface Deform binds to the ordered evaluated mesh emitted by an Array modifier.
array_surface_target = grid(
    "ArraySurfaceTarget", [-2, -1, 0, 1, 2],
    lambda x, y: 0.08 * x * x + 0.06 * x * y - 0.03 * y * y)
array_surface_target.location = (0.35, -0.2, 0.15)
array_modifier = array_surface_target.modifiers.new("Array deform target", "ARRAY")
array_modifier.count = 2
array_modifier.use_relative_offset = True
array_modifier.relative_offset_displace = (1.5, 0.0, 0.0)
array_surface_owner = make_mesh_object(
    "ArraySurfaceOwner",
    [(-0.65, -0.65, 0.5), (0.65, -0.65, 0.5),
     (0.65, 0.65, 0.5), (-0.65, 0.65, 0.5)],
    [(0, 1, 2, 3)])
array_surface_owner.location = (5.85, 0.15, 0.0)
array_surface_deform = array_surface_owner.modifiers.new(
    "Bound to array target", "SURFACE_DEFORM")
array_surface_deform.target = array_surface_target
bpy.ops.object.select_all(action="DESELECT")
array_surface_owner.select_set(True)
bpy.context.view_layer.objects.active = array_surface_owner
bpy.ops.object.surfacedeform_bind(modifier=array_surface_deform.name)
if not array_surface_deform.is_bound:
    raise RuntimeError("Surface Deform did not bind to the array target")

# Data Transfer samples the second, UV-offset copy from the source Array stack.
transfer_source = make_mesh_object(
    "TransferSource", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    [(0, 1, 2, 3)])
uv_layer = transfer_source.data.uv_layers.new(name="UVMap")
for loop in transfer_source.data.loops:
    vertex = transfer_source.data.vertices[loop.vertex_index].co
    uv_layer.data[loop.index].uv = (float(vertex.x), float(vertex.y))
source_array = transfer_source.modifiers.new("UV-offset source copy", "ARRAY")
source_array.count = 2
source_array.use_relative_offset = True
source_array.relative_offset_displace = (3.0, 0.0, 0.0)
source_array.offset_u = 0.45
source_array.offset_v = -0.2
transfer_owner = make_mesh_object(
    "TransferOwner", [(3.1, 0.2, 0), (3.9, 0.2, 0),
                       (3.9, 0.8, 0), (3.1, 0.8, 0)],
    [(0, 1, 2, 3)])
transfer_owner.data.uv_layers.new(name="UVMap")
data_transfer = transfer_owner.modifiers.new("Transfer evaluated source UVs", "DATA_TRANSFER")
data_transfer.object = transfer_source
data_transfer.use_object_transform = False
data_transfer.use_vert_data = False
data_transfer.use_loop_data = True
data_transfer.data_types_loops = {"UV"}
data_transfer.loop_mapping = "POLYINTERP_NEAREST"
data_transfer.mix_mode = "REPLACE"
data_transfer.mix_factor = 1.0
data_transfer.layers_uv_select_src = "ALL"
data_transfer.layers_uv_select_dst = "NAME"


def sample(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        positions = [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
                     for vertex in mesh.vertices]
        triangles = []
        for triangle in mesh.loop_triangles:
            points = [positions[index] for index in triangle.vertices]
            triangles.append({
                "material": int(mesh.polygons[triangle.polygon_index].material_index),
                "points": points,
            })
        uv_records = []
        layer = mesh.uv_layers.get("UVMap")
        if layer is not None:
            for loop in mesh.loops:
                position = positions[loop.vertex_index]
                uv = layer.data[loop.index].uv
                uv_records.append(position + [float(uv.x), float(uv.y)])
        return {"positions": positions, "triangles": triangles,
                "material_indices": [int(poly.material_index) for poly in mesh.polygons],
                "uv_records": uv_records}
    finally:
        evaluated.to_mesh_clear()


scene.frame_set(5)
bpy.context.view_layer.update()
names = ["BooleanHost", "ShrinkOwner", "SurfaceOwner", "ArraySurfaceOwner", "TransferOwner"]
with open(os.path.join(root, "blender_evaluated.json"), "w", encoding="utf-8") as output:
    json.dump({name: sample(bpy.data.objects[name]) for name in names}, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "operand_source.blend"))
"#;

fn run_blender(blender: &Path, root: &Path) -> TestResult<()> {
    blender_script::run_blender_script(
        blender,
        "make_operand_scene.py",
        BLENDER_FIXTURE,
        root,
        "Blender fixture failed",
    )
}

fn blender_position(value: &Value) -> TestResult<DVec3> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 3)
        .ok_or("Blender vertex must have three coordinates")?;
    Ok(DVec3::new(
        components[0]
            .as_f64()
            .ok_or("invalid Blender x coordinate")?,
        components[1]
            .as_f64()
            .ok_or("invalid Blender y coordinate")?,
        components[2]
            .as_f64()
            .ok_or("invalid Blender z coordinate")?,
    ))
}

fn blender_positions(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("Blender vertex positions are missing")?
        .iter()
        .map(blender_position)
        .collect()
}

fn assert_vertex_cloud(actual: &[DVec3], expected: &[DVec3], label: &str) {
    assert!(
        !actual.is_empty() && !expected.is_empty(),
        "{label}: empty geometry"
    );
    let max_gap = actual
        .iter()
        .map(|point| {
            expected
                .iter()
                .map(|other| point.distance(*other))
                .fold(f64::INFINITY, f64::min)
        })
        .chain(expected.iter().map(|point| {
            actual
                .iter()
                .map(|other| point.distance(*other))
                .fold(f64::INFINITY, f64::min)
        }))
        .fold(0.0, f64::max);
    assert!(
        max_gap <= TOLERANCE,
        "{label}: symmetric vertex-cloud gap {max_gap} exceeds {TOLERANCE}"
    );
}

fn blender_triangles(geometry: &Value, material: Option<u32>) -> TestResult<Vec<[DVec3; 3]>> {
    geometry["triangles"]
        .as_array()
        .ok_or("Blender triangles are missing")?
        .iter()
        .filter(|triangle| {
            material.is_none_or(|index| triangle["material"].as_u64() == Some(u64::from(index)))
        })
        .map(|triangle| {
            let points = triangle["points"]
                .as_array()
                .filter(|points| points.len() == 3)
                .ok_or("Blender triangle must have three points")?;
            let mut parsed = [DVec3::ZERO; 3];
            for (index, point) in points.iter().enumerate() {
                parsed[index] = blender_position(point)?;
            }
            Ok(parsed)
        })
        .collect()
}

fn mesh_triangles(mesh: &Mesh, material: Option<u32>) -> TestResult<Vec<[DVec3; 3]>> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut triangles = Vec::new();
    for face in mesh
        .faces
        .iter()
        .filter(|face| material.is_none_or(|slot| face.material_index == slot))
    {
        let mut one_face = mesh.clone();
        one_face.faces = vec![face.clone()];
        for triangle in one_face.triangulate()? {
            triangles.push([
                *positions
                    .get(&triangle[0])
                    .ok_or("triangle vertex is missing")?,
                *positions
                    .get(&triangle[1])
                    .ok_or("triangle vertex is missing")?,
                *positions
                    .get(&triangle[2])
                    .ok_or("triangle vertex is missing")?,
            ]);
        }
    }
    Ok(triangles)
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
    const DIVISIONS: usize = 8;
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

fn assert_surface_parity(actual: &[[DVec3; 3]], expected: &[[DVec3; 3]], label: &str) {
    assert!(
        !actual.is_empty() && !expected.is_empty(),
        "{label}: empty surface"
    );
    let gap = directed_surface_distance(actual, expected)
        .max(directed_surface_distance(expected, actual));
    assert!(
        gap <= TOLERANCE,
        "{label}: symmetric surface gap {gap} exceeds {TOLERANCE}"
    );
}

fn object_mesh<'a>(
    snapshot: &'a Snapshot,
    document: &SceneDoc,
    name: &str,
) -> TestResult<&'a Mesh> {
    let id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .map(|(id, _)| id)
        .ok_or_else(|| format!("imported object {name} is missing"))?;
    snapshot
        .meshes
        .get(id)
        .ok_or_else(|| format!("evaluated mesh for {name} is missing").into())
}

fn mesh_positions(mesh: &Mesh) -> Vec<DVec3> {
    mesh.vertices.iter().map(|vertex| vertex.co).collect()
}

fn uv_records(mesh: &Mesh) -> TestResult<Vec<[f64; 5]>> {
    let entries = mesh.attributes["uv_map"]
        .as_array()
        .ok_or("evaluated Data Transfer output has no UV map")?;
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut records = Vec::new();
    for face in &mesh.faces {
        let uv = entries
            .iter()
            .find(|entry| {
                entry["face_id"].as_u64() == Some(u64::from(face.id))
                    && entry["layer"].as_str().unwrap_or("UVMap") == "UVMap"
            })
            .and_then(|entry| entry["uv"].as_array())
            .ok_or("Data Transfer output face has no UV values")?;
        if uv.len() != face.vertices.len() {
            return Err("Data Transfer output UV corner count differs from its face".into());
        }
        for (vertex_id, uv) in face.vertices.iter().zip(uv) {
            let position = positions
                .get(vertex_id)
                .ok_or("UV corner vertex is missing")?;
            let uv = uv
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or("invalid UV coordinate")?;
            records.push([
                position.x,
                position.y,
                position.z,
                uv[0].as_f64().ok_or("invalid UV u coordinate")?,
                uv[1].as_f64().ok_or("invalid UV v coordinate")?,
            ]);
        }
    }
    records.sort_by(|left, right| {
        left.iter()
            .zip(right)
            .find_map(|(a, b)| {
                let order = a.total_cmp(b);
                (order != std::cmp::Ordering::Equal).then_some(order)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(records)
}

fn expected_uv_records(value: &Value) -> TestResult<Vec<[f64; 5]>> {
    let mut records: Vec<[f64; 5]> = value["uv_records"]
        .as_array()
        .ok_or("Blender Data Transfer UV records are missing")?
        .iter()
        .map(|record| {
            let record = record
                .as_array()
                .filter(|record| record.len() == 5)
                .ok_or("invalid Blender UV record")?;
            let mut parsed = [0.0; 5];
            for (index, value) in record.iter().enumerate() {
                parsed[index] = value.as_f64().ok_or("invalid Blender UV value")?;
            }
            Ok(parsed)
        })
        .collect::<TestResult<_>>()?;
    records.sort_by(|left, right| {
        left.iter()
            .zip(right)
            .find_map(|(a, b)| {
                let order = a.total_cmp(b);
                (order != std::cmp::Ordering::Equal).then_some(order)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(records)
}

fn assert_uv_parity(actual: &[[f64; 5]], expected: &[[f64; 5]]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Data Transfer UV corner count"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error <= TOLERANCE,
            "Data Transfer UV record {index} differs by {error}: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn blender_object_mesh_operands_use_same_frame_evaluated_geometry() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping object-mesh operand regressions: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender(&blender, root)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_evaluated.json"))?)?;

    let project = root.join("operand_project");
    pot_json(&[
        "init",
        project.to_str().ok_or("project path was not UTF-8")?,
    ])?;
    let imported = pot_json(&[
        "import",
        project.to_str().ok_or("project path was not UTF-8")?,
        "--file",
        root.join("operand_source.blend")
            .to_str()
            .ok_or("blend path was not UTF-8")?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().ok_or("Blender path was not UTF-8")?,
    ])?;
    assert!(
        imported["result"]["losses"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "operand fixture import reported losses: {}",
        imported["result"]["losses"]
    );
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate(
        &document,
        &EvaluationContext {
            frame: Some(FRAME),
            ..EvaluationContext::default()
        },
    )?;

    let boolean_host = object_mesh(&snapshot, &document, "BooleanHost")?;
    assert_surface_parity(
        &mesh_triangles(boolean_host, None)?,
        &blender_triangles(&expected["BooleanHost"], None)?,
        "Boolean result with arrayed cutter",
    );
    let host_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "BooleanHost")
        .map(|(id, _)| id)
        .ok_or("Boolean host object is missing")?;
    assert_eq!(
        document.nodes[host_id].materials.len(),
        2,
        "Boolean host material slots"
    );
    let blender_cut_surface = blender_triangles(&expected["BooleanHost"], Some(1))?;
    let actual_cut_surface = mesh_triangles(boolean_host, Some(1))?;
    assert_surface_parity(
        &actual_cut_surface,
        &blender_cut_surface,
        "Boolean transferred second material slot on cut faces",
    );

    let cutter_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "BooleanCutter")
        .map(|(id, _)| id.clone())
        .ok_or("Boolean cutter object is missing")?;
    let mut cyclic_document = document.clone();
    let cycle_operand = cyclic_document
        .nodes
        .get_mut(&cutter_id)
        .ok_or("Boolean cutter node is missing")?;
    cycle_operand.modifiers.push(Modifier {
        id: Id::new("cycle_back")?,
        modifier_type: "boolean".to_owned(),
        name: "Cycle back to host".to_owned(),
        enabled: true,
        params: serde_json::json!({"object":host_id.as_str()})
            .as_object()
            .cloned()
            .ok_or("Boolean cycle operand parameters are invalid")?,
        binding_data: None,
        runtime: potter::model::ModifierRuntime::default(),
    });
    let cycle_error = Snapshot::evaluate(
        &cyclic_document,
        &EvaluationContext {
            frame: Some(FRAME),
            ..EvaluationContext::default()
        },
    )
    .err()
    .ok_or("two-object Boolean dependency cycle was not rejected")?;
    assert_eq!(cycle_error.code, ErrorCode::EvaluationFailed);
    let cycle = cycle_error.details["cycle"]
        .as_array()
        .ok_or("evaluated-mesh cycle witness is missing")?;
    assert_eq!(cycle.first(), cycle.last(), "cycle witness must close");
    assert!(
        cycle.iter().any(|id| id.as_str() == Some(host_id.as_str()))
            && cycle
                .iter()
                .any(|id| id.as_str() == Some(cutter_id.as_str())),
        "cycle witness must include both operands: {cycle:?}"
    );

    for (owner, label) in [
        ("ShrinkOwner", "subdivided-target Shrinkwrap"),
        ("SurfaceOwner", "Surface Deform bound to subdivided target"),
        (
            "ArraySurfaceOwner",
            "Surface Deform bound to arrayed target",
        ),
        ("TransferOwner", "Data Transfer from arrayed UV source"),
    ] {
        let actual = object_mesh(&snapshot, &document, owner)?;
        let expected_positions = blender_positions(&expected[owner]["positions"])?;
        assert_eq!(
            actual.vertices.len(),
            expected_positions.len(),
            "{label}: vertex count"
        );
        assert_vertex_cloud(&mesh_positions(actual), &expected_positions, label);
    }
    let mut changed_target_document = document.clone();
    let surface_target_id = changed_target_document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "SurfaceTarget")
        .map(|(id, _)| id.clone())
        .ok_or("subdivision target object is missing")?;
    let surface_target_data_id = changed_target_document
        .nodes
        .get(&surface_target_id)
        .and_then(|node| node.data.as_ref())
        .cloned()
        .ok_or("subdivision target mesh data is missing")?;
    changed_target_document
        .data_blocks
        .get_mut(&surface_target_data_id)
        .and_then(|data| data.mesh.as_mut())
        .and_then(|mesh| mesh.faces.pop())
        .ok_or("subdivision target polygon is missing")?;
    let changed_target_snapshot = Snapshot::evaluate(
        &changed_target_document,
        &EvaluationContext {
            frame: Some(FRAME),
            ..EvaluationContext::default()
        },
    )?;
    let topology_warning = changed_target_snapshot
        .warnings
        .iter()
        .find(|warning| {
            warning.details["feature_id"]
                .as_str()
                .is_some_and(|feature| feature == "modifier.surface_deform.target_topology_changed")
        })
        .ok_or("changed native Surface Deform target did not report a topology warning")?;
    assert_eq!(topology_warning.code, ErrorCode::EvaluationFailed);
    let changed_surface_owner = object_mesh(
        &changed_target_snapshot,
        &changed_target_document,
        "SurfaceOwner",
    )?;
    let surface_owner_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == "SurfaceOwner")
        .map(|(id, _)| id)
        .ok_or("subdivision Surface Deform owner is missing")?;
    let surface_owner_data_id = document.nodes[surface_owner_id]
        .data
        .as_ref()
        .ok_or("subdivision Surface Deform owner data is missing")?;
    let surface_owner_base = document.data_blocks[surface_owner_data_id]
        .mesh
        .as_ref()
        .ok_or("subdivision Surface Deform owner base mesh is missing")?;
    assert_eq!(
        mesh_positions(changed_surface_owner),
        mesh_positions(surface_owner_base),
        "a native Surface Deform with changed target topology must leave its source mesh undeformed"
    );

    let transfer = object_mesh(&snapshot, &document, "TransferOwner")?;
    assert_uv_parity(
        &uv_records(transfer)?,
        &expected_uv_records(&expected["TransferOwner"])?,
    );
    Ok(())
}
