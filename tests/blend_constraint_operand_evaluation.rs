use std::{error::Error, fs, path::Path};

use glam::DVec3;
use potter::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, SceneDoc},
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

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const FRAME: f64 = 5.0;
const TOLERANCE: f64 = 1.0e-5;
const ATTRIBUTE_NAME: &str = "position";
const ATTRIBUTE_SAMPLE_INDEX: usize = 0;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
if bpy.app.version != (5, 2, 2):
    raise RuntimeError("constraint operand fixture requires Blender 5.2.2")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_set(5)


def mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj




def mesh_positions(mesh):
    return [[float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
            for vertex in mesh.vertices]


def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


# The evaluated quad gains a center point, which is nearest to the owner; none
# of the four control vertices is. This makes modifier evaluation observable.
shrink_target = mesh_object(
    "ShrinkTarget",
    [(-2.0, -2.0, 0.0), (2.0, -2.0, 0.0),
     (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0)],
    [(0, 1, 2, 3)])
shrink_subdivision = shrink_target.modifiers.new("Evaluated target surface", "SUBSURF")
shrink_subdivision.subdivision_type = "CATMULL_CLARK"
shrink_subdivision.levels = 1
shrink_subdivision.render_levels = 1
shrink_owner = bpy.data.objects.new("ShrinkOwner", None)
scene.collection.objects.link(shrink_owner)
shrink_owner.location = (0.0, 0.0, 3.0)
shrink_constraint = shrink_owner.constraints.new("SHRINKWRAP")
shrink_constraint.target = shrink_target
shrink_constraint.shrinkwrap_type = "NEAREST_VERTEX"
shrink_constraint.wrap_mode = "ON_SURFACE"
shrink_constraint.distance = 0.0

# A same-topology modifier changes the built-in POINT position attribute
# before the Geometry Attribute constraint samples point zero.
attribute_target = mesh_object(
    "AttributeTarget",
    [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)],
    [(0, 1, 2, 3)])
attribute_target.location = (7.0, -4.0, 3.0)
attribute_displace = attribute_target.modifiers.new(
    "Displace evaluated target", "DISPLACE")
attribute_displace.direction = "Z"
attribute_displace.mid_level = 0.0
attribute_displace.strength = 1.0
attribute_owner = bpy.data.objects.new("AttributeOwner", None)
scene.collection.objects.link(attribute_owner)
attribute_owner.location = (0.0, 0.0, 0.0)
attribute_constraint = attribute_owner.constraints.new("GEOMETRY_ATTRIBUTE")
attribute_constraint.target = attribute_target
attribute_constraint.attribute_name = "position"
attribute_constraint.data_type = "VECTOR"
attribute_constraint.domain = "POINT"
attribute_constraint.sample_index = 0
attribute_constraint.mix_mode = "REPLACE"
attribute_constraint.apply_target_transform = False
attribute_constraint.mix_loc = True
attribute_constraint.mix_rot = False
attribute_constraint.mix_scl = False

scene.frame_set(5)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()

def evaluated_mesh(obj):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return mesh_positions(mesh)
    finally:
        evaluated.to_mesh_clear()


evaluated_attribute_target = attribute_target.evaluated_get(depsgraph)
evaluated_attribute_mesh = evaluated_attribute_target.to_mesh()
try:
    evaluated_attribute = evaluated_attribute_mesh.attributes.get("position")
    if evaluated_attribute is None:
        raise RuntimeError("evaluated position attribute is missing")
    attribute_values = [
        [float(component) for component in item.vector]
        for item in evaluated_attribute.data
    ]
finally:
    evaluated_attribute_target.to_mesh_clear()

result = {
    "frame": int(scene.frame_current),
    "ShrinkOwner": matrix_values(shrink_owner.evaluated_get(depsgraph).matrix_world),
    "ShrinkTarget": evaluated_mesh(shrink_target),
    "AttributeOwner": matrix_values(attribute_owner.evaluated_get(depsgraph).matrix_world),
    "AttributeTarget": evaluated_mesh(attribute_target),
    "attribute_sample_index": int(attribute_constraint.sample_index),
    "attribute_evaluated_values": attribute_values,
}
with open(os.path.join(root, "blender_evaluated.json"), "w", encoding="utf-8") as output:
    json.dump(result, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "constraint_operands.blend"))
"#;

fn run_blender(blender: &Path, root: &Path) -> TestResult {
    blender_script::run_blender_script(
        blender,
        "make_constraint_operands.py",
        BLENDER_FIXTURE,
        root,
        "Blender constraint operand fixture failed",
    )
}

fn node_named<'a>(doc: &'a SceneDoc, name: &str) -> TestResult<(&'a Id, &'a potter::model::Node)> {
    doc.nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .ok_or_else(|| format!("imported object {name} is missing").into())
}

fn blender_positions(value: &Value, label: &str) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} Blender evaluated positions are missing"))?
        .iter()
        .map(|position| {
            let components = position
                .as_array()
                .filter(|components| components.len() == 3)
                .ok_or_else(|| format!("{label} Blender position must have three components"))?;
            Ok(DVec3::new(
                components[0]
                    .as_f64()
                    .ok_or_else(|| format!("{label} Blender x position is invalid"))?,
                components[1]
                    .as_f64()
                    .ok_or_else(|| format!("{label} Blender y position is invalid"))?,
                components[2]
                    .as_f64()
                    .ok_or_else(|| format!("{label} Blender z position is invalid"))?,
            ))
        })
        .collect()
}

fn assert_vertex_cloud(actual: &[DVec3], expected: &[DVec3], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: evaluated vertex count differs"
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
        "{label}: symmetric evaluated-geometry gap {max_gap:e} exceeds {TOLERANCE:e}"
    );
}

fn matrix_values(value: &Value, label: &str) -> TestResult<Vec<f64>> {
    value
        .as_array()
        .filter(|values| values.len() == 16)
        .ok_or_else(|| format!("{label} Blender world matrix is missing or malformed"))?
        .iter()
        .map(|component| {
            component
                .as_f64()
                .ok_or_else(|| format!("{label} Blender matrix component is invalid").into())
        })
        .collect()
}

fn assert_matrix_parity(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), 16, "{label}: Potter matrix size");
    assert_eq!(expected.len(), 16, "{label}: Blender matrix size");
    let (index, max_error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    assert!(
        max_error <= TOLERANCE,
        "{label}: matrix component {index} differs by {max_error:e}; Potter={actual:?}, Blender={expected:?}"
    );
}

fn mesh_positions(mesh: &Mesh) -> Vec<DVec3> {
    mesh.vertices.iter().map(|vertex| vertex.co).collect()
}

fn attribute_at_vertex_index(mesh: &Mesh, name: &str, index: usize) -> TestResult<DVec3> {
    let vertex = mesh
        .vertices
        .get(index)
        .ok_or_else(|| format!("evaluated target has no point at index {index}"))?;
    if name == "position" {
        return Ok(vertex.co);
    }
    let attributes = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .ok_or("evaluated target has no Blender mesh attributes")?;
    let attribute = attributes
        .iter()
        .find(|attribute| attribute["name"].as_str() == Some(name))
        .ok_or_else(|| format!("evaluated target has no {name} attribute"))?;
    if attribute["domain"] != "POINT" || attribute["data_type"] != "FLOAT_VECTOR" {
        return Err(
            format!("evaluated {name} attribute has unexpected metadata: {attribute}").into(),
        );
    }
    let values = attribute["values"]
        .as_array()
        .ok_or_else(|| format!("evaluated {name} POINT attribute values are missing"))?;
    let value = values
        .get(index)
        .and_then(Value::as_array)
        .filter(|components| components.len() == 3)
        .ok_or_else(|| {
            format!(
                "evaluated {name} POINT attribute has {} values for {} evaluated vertices; index {index} is missing",
                values.len(),
                mesh.vertices.len()
            )
        })?;
    Ok(DVec3::new(
        value[0]
            .as_f64()
            .ok_or_else(|| format!("evaluated {name} x value is invalid"))?,
        value[1]
            .as_f64()
            .ok_or_else(|| format!("evaluated {name} y value is invalid"))?,
        value[2]
            .as_f64()
            .ok_or_else(|| format!("evaluated {name} z value is invalid"))?,
    ))
}

fn blender_vector(value: &Value, label: &str) -> TestResult<DVec3> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 3)
        .ok_or_else(|| format!("{label} Blender vector is missing or malformed"))?;
    Ok(DVec3::new(
        components[0]
            .as_f64()
            .ok_or_else(|| format!("{label} Blender x value is invalid"))?,
        components[1]
            .as_f64()
            .ok_or_else(|| format!("{label} Blender y value is invalid"))?,
        components[2]
            .as_f64()
            .ok_or_else(|| format!("{label} Blender z value is invalid"))?,
    ))
}

#[test]
fn blender_constraint_operands_use_same_frame_evaluated_meshes() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping constraint operand regressions: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender(&blender, root)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_evaluated.json"))?)?;
    assert_eq!(expected["frame"].as_f64(), Some(FRAME));

    let project = root.join("constraint_operand_project");
    pot_json(&["init", project.to_str().ok_or("project path is not UTF-8")?])?;
    let imported = pot_json(&[
        "import",
        project.to_str().ok_or("project path is not UTF-8")?,
        "--file",
        root.join("constraint_operands.blend")
            .to_str()
            .ok_or("Blend path is not UTF-8")?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender
            .to_str()
            .ok_or("Blender executable path is not UTF-8")?,
    ])?;
    assert_eq!(
        imported["result"]["losses"],
        serde_json::json!([]),
        "constraint operand fixture import losses"
    );
    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (shrink_owner_id, shrink_owner) = node_named(&doc, "ShrinkOwner")?;
    let (shrink_target_id, _) = node_named(&doc, "ShrinkTarget")?;
    let (attribute_owner_id, attribute_owner) = node_named(&doc, "AttributeOwner")?;
    let (attribute_target_id, _) = node_named(&doc, "AttributeTarget")?;
    assert_eq!(
        shrink_owner.constraints.len(),
        1,
        "Shrinkwrap constraint count"
    );
    assert_eq!(
        shrink_owner.constraints[0].target.as_ref(),
        Some(shrink_target_id),
        "Shrinkwrap target reference"
    );
    assert_eq!(
        attribute_owner.constraints.len(),
        1,
        "Geometry Attribute constraint count"
    );
    assert_eq!(
        attribute_owner.constraints[0].target.as_ref(),
        Some(attribute_target_id),
        "Geometry Attribute target reference"
    );
    assert_eq!(
        attribute_owner.constraints[0].params["sample_index"],
        serde_json::json!(ATTRIBUTE_SAMPLE_INDEX),
        "Geometry Attribute evaluated point sample index"
    );

    let context = EvaluationContext {
        frame: Some(FRAME),
        ..EvaluationContext::default()
    };
    let mut target_probe_doc = doc.clone();
    target_probe_doc
        .nodes
        .get_mut(attribute_owner_id)
        .ok_or("AttributeOwner disappeared from the imported scene")?
        .constraints
        .clear();
    let target_probe = Snapshot::evaluate(&target_probe_doc, &context)?;
    let target_probe_mesh = target_probe
        .meshes
        .get(attribute_target_id)
        .ok_or("Potter evaluated AttributeTarget mesh is missing")?;
    let blender_attribute_values = expected["attribute_evaluated_values"]
        .as_array()
        .ok_or("Blender evaluated point attribute values are missing")?;
    let blender_sample = blender_vector(
        blender_attribute_values
            .get(ATTRIBUTE_SAMPLE_INDEX)
            .ok_or("Blender evaluated point domain does not contain the sample index")?,
        "Geometry Attribute evaluated point sample",
    )?;
    let target_probe_point_count = target_probe_mesh.vertices.len();
    let blender_point_count = blender_attribute_values.len();

    assert!(
        (target_probe.frame - FRAME).abs() <= f64::EPSILON,
        "Potter evaluated frame"
    );

    let shrink_mesh = target_probe
        .meshes
        .get(shrink_target_id)
        .ok_or("Potter evaluated ShrinkTarget mesh is missing")?;
    let expected_shrink_positions = blender_positions(&expected["ShrinkTarget"], "ShrinkTarget")?;
    assert_vertex_cloud(
        &mesh_positions(shrink_mesh),
        &expected_shrink_positions,
        "subdivided Shrinkwrap target",
    );
    let shrink_actual = target_probe
        .nodes
        .get(shrink_owner_id)
        .ok_or("Potter evaluated ShrinkOwner transform is missing")?
        .world_matrix;
    let shrink_expected = matrix_values(&expected["ShrinkOwner"], "ShrinkOwner")?;
    assert_matrix_parity(
        &shrink_actual,
        &shrink_expected,
        "subdivided-target Shrinkwrap",
    );
    let shrink_location = DVec3::new(
        shrink_expected[12],
        shrink_expected[13],
        shrink_expected[14],
    );
    assert!(
        [
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(-2.0, 2.0, 0.0),
        ]
        .iter()
        .all(|point| shrink_location.distance(*point) > 1.0),
        "Blender Shrinkwrap did not select a generated subdivision vertex: {shrink_location:?}"
    );

    let attribute_mesh = target_probe_mesh;
    let expected_attribute_positions =
        blender_positions(&expected["AttributeTarget"], "AttributeTarget")?;
    assert_vertex_cloud(
        &mesh_positions(attribute_mesh),
        &expected_attribute_positions,
        "displaced Geometry Attribute target",
    );
    assert_eq!(
        blender_attribute_values.len(),
        expected_attribute_positions.len(),
        "Blender evaluated POINT attribute domain count"
    );
    let sample_index = expected["attribute_sample_index"]
        .as_u64()
        .ok_or("Blender Geometry Attribute sample index is missing")?
        as usize;
    assert_eq!(sample_index, ATTRIBUTE_SAMPLE_INDEX);
    assert!(
        blender_sample.distance(DVec3::new(-2.0, -2.0, 0.0)) > TOLERANCE,
        "Blender's sampled position was not changed by the target Displace modifier"
    );
    let potter_sample = attribute_at_vertex_index(attribute_mesh, ATTRIBUTE_NAME, sample_index)
        .map_err(|error| {
            std::io::Error::other(format!(
                "Geometry Attribute evaluated POINT sample mismatch: Blender has {blender_point_count} evaluated points and index {sample_index}={blender_sample:?}; Potter has {} evaluated vertices: {error}",
                attribute_mesh.vertices.len()
            ))
        })?;
    assert!(
        potter_sample.distance(blender_sample) <= TOLERANCE,
        "evaluated point attribute index {sample_index} differs: Potter={potter_sample:?}, Blender={blender_sample:?}"
    );
    let snapshot = Snapshot::evaluate(&doc, &context).map_err(|error| {
        std::io::Error::other(format!(
            "Potter Geometry Attribute evaluation did not match Blender's evaluated POINT sample: Blender has {blender_point_count} evaluated points and samples index {ATTRIBUTE_SAMPLE_INDEX} as {blender_sample:?}; Potter's same-frame target mesh has {target_probe_point_count} evaluated points with the Geometry Attribute owner constraint removed; evaluation error={error:?}"
        ))
    })?;
    assert!(
        (snapshot.frame - FRAME).abs() <= f64::EPSILON,
        "Potter evaluated frame"
    );
    let attribute_actual = snapshot
        .nodes
        .get(attribute_owner_id)
        .ok_or("Potter evaluated AttributeOwner transform is missing")?
        .world_matrix;
    let attribute_expected = matrix_values(&expected["AttributeOwner"], "AttributeOwner")?;
    assert_matrix_parity(
        &attribute_actual,
        &attribute_expected,
        "modified-target Geometry Attribute constraint",
    );
    assert!(
        DVec3::new(
            attribute_expected[12],
            attribute_expected[13],
            attribute_expected[14]
        )
        .abs_diff_eq(blender_sample, TOLERANCE),
        "Blender Geometry Attribute owner translation does not match evaluated point {sample_index}: owner={:?}, sampled={blender_sample:?}",
        [
            attribute_expected[12],
            attribute_expected[13],
            attribute_expected[14]
        ]
    );

    eprintln!(
        "Blender constraint operand parity at frame {FRAME}: Shrinkwrap matrix and target geometry, Geometry Attribute matrix and evaluated point {sample_index}; tolerance={TOLERANCE:e}"
    );
    Ok(())
}
