use std::{error::Error, fs, path::Path};

use glam::DVec3;
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

const FRAME: f64 = 5.0;
const TOLERANCE: f64 = 1.0e-5;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_set(1)

points = [(float(x), float(y), 0.0) for y in range(-2, 3) for x in range(-2, 3)]
faces = []
for y in range(4):
    for x in range(4):
        index = y * 5 + x
        faces.append((index, index + 1, index + 6, index + 5))
target_mesh = bpy.data.meshes.new("SurfaceTargetMesh")
target_mesh.from_pydata(points, [], faces)
target_mesh.update()
target = bpy.data.objects.new("SurfaceDeform_Target", target_mesh)
scene.collection.objects.link(target)
basis = target.shape_key_add(name="Basis", from_mix=False)
lift = target.shape_key_add(name="Lift", from_mix=False)
for point in lift.data:
    point.co.z += 0.3
lift.slider_min = -0.25
lift.slider_max = 1.25

muted = target.shape_key_add(name="Muted", from_mix=False)
for point in muted.data:
    point.co.z += 0.1
muted.value = 1.0
muted.mute = True
muted_curve_key = target.shape_key_add(name="MutedCurve", from_mix=False)
for point in muted_curve_key.data:
    point.co.z += 0.4
muted_curve_key.value = 0.5


key_data = target.data.shape_keys
action = bpy.data.actions.new("SurfaceTargetShapeAnimation")
slot = action.slots.new("KEY", key_data.name)
key_data.animation_data_create()
key_data.animation_data.action = action
key_data.animation_data.action_slot = slot
layer = action.layers.new("Shape Key Values")
strip = layer.strips.new(type="KEYFRAME")
channelbag = strip.channelbag(slot, ensure=True)
curve = channelbag.fcurves.new(data_path='key_blocks["Lift"].value', index=0)
curve.keyframe_points.insert(1.0, 0.0).interpolation = "LINEAR"
curve.keyframe_points.insert(9.0, 1.0).interpolation = "LINEAR"
curve.update()
muted_curve = channelbag.fcurves.new(data_path='key_blocks["MutedCurve"].value', index=0)
muted_curve.keyframe_points.insert(1.0, 0.0).interpolation = "LINEAR"
muted_curve.keyframe_points.insert(5.0, 1.0).interpolation = "LINEAR"
muted_curve.keyframe_points.insert(9.0, 0.0).interpolation = "LINEAR"
muted_curve.mute = True
muted_curve.update()

bound_mesh = bpy.data.meshes.new("BoundPlaneMesh")
bound_mesh.from_pydata(
    [(-0.7, -0.7, 0.1), (0.7, -0.7, 0.1), (0.7, 0.7, 0.1), (-0.7, 0.7, 0.1)],
    [], [(0, 1, 2, 3)])
bound_mesh.update()
bound = bpy.data.objects.new("Plane_SurfaceDeform_Bound", bound_mesh)
scene.collection.objects.link(bound)
modifier = bound.modifiers.new("BoundToAnimatedShape", "SURFACE_DEFORM")
modifier.target = target
bpy.ops.object.select_all(action="DESELECT")
bound.select_set(True)
bpy.context.view_layer.objects.active = bound
bpy.ops.object.surfacedeform_bind(modifier=modifier.name)
if not modifier.is_bound:
    raise RuntimeError("Surface Deform fixture did not bind")

def positions(obj, frame):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    result = [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return result

scene.frame_set(5)
bpy.context.view_layer.update()
shape_value = float(target.data.shape_keys.key_blocks["Lift"].value)
scene.frame_set(9)
bpy.context.view_layer.update()
shape_value_at_nine = float(target.data.shape_keys.key_blocks["Lift"].value)
with open(os.path.join(root, "blender_evaluated.json"), "w", encoding="utf-8") as handle:
    json.dump({
        "target": positions(target, 5),
        "bound": positions(bound, 5),
        "target_at_one": positions(target, 1),
        "bound_at_one": positions(bound, 1),
        "target_at_nine": positions(target, 9),
        "bound_at_nine": positions(bound, 9),
        "shape_value_at_nine": shape_value_at_nine,
        "shape_value": shape_value,
    }, handle)
scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "shape_key_source.blend"))
"#;

fn run_blender(blender: &Path, root: &Path) -> Result<(), Box<dyn Error>> {
    blender_script::run_blender_script(
        blender,
        "make_shape_key_scene.py",
        BLENDER_FIXTURE,
        root,
        "Blender fixture failed",
    )
}

fn expected_positions(value: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or("Blender positions were not an array")?
        .iter()
        .map(|point| {
            let components = point.as_array().ok_or("Blender vertex was not an array")?;
            Ok([
                components[0].as_f64().ok_or("vertex x was not numeric")?,
                components[1].as_f64().ok_or("vertex y was not numeric")?,
                components[2].as_f64().ok_or("vertex z was not numeric")?,
            ])
        })
        .collect()
}

fn assert_nearest_parity(actual: &[[f64; 3]], expected: &[[f64; 3]], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: vertex count");
    let actual_points: Vec<_> = actual.iter().copied().map(DVec3::from_array).collect();
    let expected_points: Vec<_> = expected.iter().copied().map(DVec3::from_array).collect();
    let max_error = actual_points
        .iter()
        .map(|point| {
            expected_points
                .iter()
                .map(|expected| point.distance(*expected))
                .fold(f64::INFINITY, f64::min)
        })
        .chain(expected_points.iter().map(|point| {
            actual_points
                .iter()
                .map(|actual| point.distance(*actual))
                .fold(f64::INFINITY, f64::min)
        }))
        .fold(0.0, f64::max);
    assert!(
        max_error < TOLERANCE,
        "{label}: symmetric nearest-vertex gap {max_error} exceeds {TOLERANCE}; actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn blender_key_action_deforms_surface_target_and_bound_object() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping shape-key parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender(&blender, root)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_evaluated.json"))?)?;
    assert!(
        (expected["shape_value"]
            .as_f64()
            .ok_or("frame-five shape value missing")?
            - 0.5)
            .abs()
            < 1.0e-6
    );
    let nine_value = expected["shape_value_at_nine"]
        .as_f64()
        .ok_or("frame-nine shape value missing")?;
    assert!((nine_value - 1.0).abs() < 1.0e-6);

    let project = root.join("shape_key_project");
    pot_json(&[
        "init",
        project.to_str().ok_or("project path was not UTF-8")?,
    ])?;
    let import = pot_json(&[
        "import",
        project.to_str().ok_or("project path was not UTF-8")?,
        "--file",
        root.join("shape_key_source.blend")
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
        import["result"]["losses"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "shape-key import reported losses: {}",
        import["result"]["losses"]
    );
    let mappings = import["result"]["id_mappings"]
        .as_object()
        .ok_or("import result omitted ID mappings")?;
    let target_id = mappings["Object:SurfaceDeform_Target"]
        .as_str()
        .ok_or("target object mapping was missing")?;
    let bound_id = mappings["Object:Plane_SurfaceDeform_Bound"]
        .as_str()
        .ok_or("bound object mapping was missing")?;
    let scene: potter::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let bound_node = &scene.nodes[&potter::model::Id::new(bound_id.to_owned())?];
    let modifier = bound_node
        .modifiers
        .first()
        .ok_or("Surface Deform modifier was not imported")?;
    assert_eq!(modifier.modifier_type, "surface_deform");
    assert!(
        modifier.enabled,
        "Surface Deform modifier was imported disabled"
    );
    assert_eq!(modifier.params["target"].as_str(), Some(target_id));
    let modifier_state = serde_json::to_value(modifier)?;
    assert!(
        modifier_state.get("binding_data").is_some(),
        "Surface Deform binding was not imported"
    );
    let target_node = &scene.nodes[&potter::model::Id::new(target_id.to_owned())?];
    let shape_keys = scene.data_blocks[target_node.data.as_ref().ok_or("target data missing")?]
        .shape_keys
        .as_ref()
        .ok_or("imported shape keys missing")?;
    assert!(
        shape_keys.action.is_some(),
        "Key datablock action was not imported"
    );
    assert!(
        shape_keys.action_slot.is_some(),
        "Key Action slot was not imported"
    );
    let lift = shape_keys
        .keys
        .values()
        .find(|key| key.name == "Lift")
        .ok_or("Lift shape key was not imported")?;
    assert_eq!(lift.slider_min, -0.25);
    assert!((lift.slider_max - 1.25).abs() < 1.0e-6);
    assert!(
        shape_keys
            .keys
            .values()
            .any(|key| key.name == "Muted" && key.mute),
        "muted shape-key state was not imported"
    );
    assert!(
        shape_keys
            .muted_action_curves
            .contains("key_blocks[\"MutedCurve\"].value"),
        "muted shape-key animation curve was not imported"
    );
    let snapshot = potter::eval::Snapshot::evaluate(
        &scene,
        &potter::eval::EvaluationContext {
            frame: Some(FRAME),
            ..potter::eval::EvaluationContext::default()
        },
    )?;
    let target_mesh = &snapshot.meshes[&potter::model::Id::new(target_id.to_owned())?];
    let bound_mesh = &snapshot.meshes[&potter::model::Id::new(bound_id.to_owned())?];
    let target_actual: Vec<_> = target_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    let bound_actual: Vec<_> = bound_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    assert_nearest_parity(
        &target_actual,
        &expected_positions(&expected["target"])?,
        "animated target",
    );
    assert_nearest_parity(
        &bound_actual,
        &expected_positions(&expected["bound"])?,
        "Surface Deform result",
    );
    let snapshot_nine = potter::eval::Snapshot::evaluate(
        &scene,
        &potter::eval::EvaluationContext {
            frame: Some(9.0),
            ..potter::eval::EvaluationContext::default()
        },
    )?;
    let target_nine: Vec<_> = snapshot_nine.meshes[&potter::model::Id::new(target_id.to_owned())?]
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    let bound_nine: Vec<_> = snapshot_nine.meshes[&potter::model::Id::new(bound_id.to_owned())?]
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    assert_nearest_parity(
        &target_nine,
        &expected_positions(&expected["target_at_nine"])?,
        "animated target at frame nine",
    );
    assert_nearest_parity(
        &bound_nine,
        &expected_positions(&expected["bound_at_nine"])?,
        "Surface Deform at the final action key",
    );

    let target_before = expected_positions(&expected["target_at_one"])?;
    let target_after = expected_positions(&expected["target"])?;
    let bound_before = expected_positions(&expected["bound_at_one"])?;
    let bound_after = expected_positions(&expected["bound"])?;
    assert!(
        target_after[0][2] - target_before[0][2] > 0.14,
        "animated shape key did not stimulate target geometry"
    );
    assert!(
        bound_after[0][2] - bound_before[0][2] > 0.14,
        "Surface Deform did not follow the evaluated target"
    );
    Ok(())
}
