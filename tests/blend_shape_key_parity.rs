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

const SHARED_OBJECT_KEY_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
mesh = bpy.data.meshes.new("SharedObjectKeyMesh")
mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [])
mesh.update()
target = bpy.data.objects.new("SharedObjectKeyTarget", mesh)
scene.collection.objects.link(target)
target.shape_key_add(name="Basis", from_mix=False)
smile = target.shape_key_add(name="Smile", from_mix=False)
smile.data[0].co.z += 1.0
key_data = mesh.shape_keys

frames = {
    1: ((0.0, 0.0, 0.0), 0.0),
    5: ((2.0, 1.0, -1.0), 0.5),
    9: ((4.0, 3.0, 2.0), 1.0),
}
for frame, (location, value) in frames.items():
    scene.frame_set(frame)
    target.location = location
    target.keyframe_insert(data_path="location", frame=frame)
    smile.value = value
    smile.keyframe_insert(data_path="value", frame=frame)
if (target.animation_data is None or key_data.animation_data is None
        or target.animation_data.action is None
        or target.animation_data.action != key_data.animation_data.action):
    raise RuntimeError("default keying did not share an Action between the Object and Key")
action = target.animation_data.action

slot_types = sorted(slot.target_id_type for slot in action.slots)
if slot_types != ["KEY", "OBJECT"]:
    raise RuntimeError("default keying did not create Object and Key Action slots: %r" % slot_types)

samples = {}
for frame in (1, 3, 5, 7, 9):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    samples[str(frame)] = {
        "location": [float(value) for value in target.location],
        "shape_value": float(key_data.key_blocks["Smile"].value),
        "shape_vertex_z": float(evaluated_mesh.vertices[0].co.z),
    }
    evaluated.to_mesh_clear()

with open(os.path.join(root, "shared_action_expected.json"), "w", encoding="utf-8") as handle:
    json.dump({"slot_types": slot_types, "frames": samples}, handle)
scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "shared_action_source.blend"))
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

fn assert_shared_action_parity(
    scene: &potter_core::model::SceneDoc,
    target_id: &potter_core::model::Id,
    expected: &Value,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    for frame in [1_u32, 3, 5, 7, 9] {
        let snapshot = potter_core::eval::Snapshot::evaluate(
            scene,
            &potter_core::eval::EvaluationContext {
                frame: Some(f64::from(frame)),
                ..potter_core::eval::EvaluationContext::default()
            },
        )?;
        let matrix = &snapshot
            .nodes
            .get(target_id)
            .ok_or("shared-action target was not evaluated")?
            .world_matrix;
        let sample = &expected["frames"][frame.to_string()];
        let location = sample["location"]
            .as_array()
            .ok_or("Blender location was not an array")?;
        for (index, component) in [matrix[12], matrix[13], matrix[14]].iter().enumerate() {
            let expected = location
                .get(index)
                .and_then(Value::as_f64)
                .ok_or("Blender location component was not numeric")?;
            assert!(
                (*component - expected).abs() < TOLERANCE,
                "{label}: Object location at frame {frame}, component {index}: Potter={component}, Blender={expected}"
            );
        }
        let shape_value = snapshot
            .meshes
            .get(target_id)
            .and_then(|mesh| mesh.vertices.first())
            .map(|vertex| vertex.co.z)
            .ok_or("evaluated shape-key vertex was missing")?;
        let expected_value = sample["shape_value"]
            .as_f64()
            .ok_or("Blender shape-key value was not numeric")?;
        let expected_vertex_z = sample["shape_vertex_z"]
            .as_f64()
            .ok_or("Blender shape-key vertex was not numeric")?;
        assert!((expected_value - expected_vertex_z).abs() < TOLERANCE);
        assert!(
            (shape_value - expected_value).abs() < TOLERANCE,
            "{label}: shape-key value at frame {frame}: Potter={shape_value}, Blender={expected_value}"
        );
    }
    Ok(())
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
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let bound_node = &scene.nodes[&potter_core::model::Id::new(bound_id.to_owned())?];
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
    let target_node = &scene.nodes[&potter_core::model::Id::new(target_id.to_owned())?];
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
    let snapshot = potter_core::eval::Snapshot::evaluate(
        &scene,
        &potter_core::eval::EvaluationContext {
            frame: Some(FRAME),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let target_mesh = &snapshot.meshes[&potter_core::model::Id::new(target_id.to_owned())?];
    let bound_mesh = &snapshot.meshes[&potter_core::model::Id::new(bound_id.to_owned())?];
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
    let snapshot_nine = potter_core::eval::Snapshot::evaluate(
        &scene,
        &potter_core::eval::EvaluationContext {
            frame: Some(9.0),
            ..potter_core::eval::EvaluationContext::default()
        },
    )?;
    let target_nine: Vec<_> = snapshot_nine.meshes
        [&potter_core::model::Id::new(target_id.to_owned())?]
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    let bound_nine: Vec<_> = snapshot_nine.meshes
        [&potter_core::model::Id::new(bound_id.to_owned())?]
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

#[test]
fn blender_default_keying_shared_object_key_action_round_trips_both_channels()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping shared Object+Key Action parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "make_shared_object_key_action.py",
        SHARED_OBJECT_KEY_FIXTURE,
        root,
        "Shared Object+Key Action fixture failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("shared_action_expected.json"))?)?;
    assert_eq!(expected["slot_types"], serde_json::json!(["KEY", "OBJECT"]));

    let project = root.join("shared_action_project");
    pot_json(&[
        "init",
        project.to_str().ok_or("project path was not UTF-8")?,
    ])?;
    let source = root.join("shared_action_source.blend");
    let import = pot_json(&[
        "import",
        project.to_str().ok_or("project path was not UTF-8")?,
        "--file",
        source.to_str().ok_or("blend path was not UTF-8")?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().ok_or("Blender path was not UTF-8")?,
    ])?;
    assert_eq!(import["result"]["losses"], serde_json::json!([]));
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let target_id = potter_core::model::Id::new(
        import["result"]["id_mappings"]["Object:SharedObjectKeyTarget"]
            .as_str()
            .ok_or("shared-action object mapping was missing")?
            .to_owned(),
    )?;
    let target = &scene.nodes[&target_id];
    let data_id = target
        .data
        .as_ref()
        .ok_or("shared-action mesh data was missing")?;
    let shape_keys = scene.data_blocks[data_id]
        .shape_keys
        .as_ref()
        .ok_or("shared-action shape keys were missing")?;
    let action_id = target
        .action
        .as_ref()
        .ok_or("Object Action was not imported")?;
    assert_eq!(shape_keys.action.as_ref(), Some(action_id));
    let action = &scene.actions[action_id];
    assert_eq!(action.slots.len(), 2);
    assert!(
        action
            .fcurves
            .iter()
            .any(|curve| curve.path == "transform.translation"),
        "Object transform curves were dropped from the shared Action"
    );
    assert!(
        action
            .fcurves
            .iter()
            .any(|curve| curve.path == "key_blocks[\"Smile\"].value"),
        "shape-key value curves were dropped from the shared Action"
    );
    assert_shared_action_parity(&scene, &target_id, &expected, "strict import")?;

    let exported = root.join("shared_action_roundtrip.blend");
    let export = pot_json(&[
        "export",
        project.to_str().ok_or("project path was not UTF-8")?,
        "--format",
        "blend",
        "--out",
        exported.to_str().ok_or("export path was not UTF-8")?,
        "--blender",
        blender.to_str().ok_or("Blender path was not UTF-8")?,
    ])?;
    assert_eq!(export["result"]["losses"], serde_json::json!([]));

    let roundtrip_project = root.join("shared_action_roundtrip_project");
    pot_json(&[
        "init",
        roundtrip_project
            .to_str()
            .ok_or("round-trip project path was not UTF-8")?,
    ])?;
    let roundtrip_import = pot_json(&[
        "import",
        roundtrip_project
            .to_str()
            .ok_or("round-trip project path was not UTF-8")?,
        "--file",
        exported
            .to_str()
            .ok_or("exported blend path was not UTF-8")?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().ok_or("Blender path was not UTF-8")?,
    ])?;
    assert_eq!(roundtrip_import["result"]["losses"], serde_json::json!([]));
    let roundtrip_scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(roundtrip_project.join("scene.json"))?)?;
    let roundtrip_target_id = potter_core::model::Id::new(
        roundtrip_import["result"]["id_mappings"]["Object:SharedObjectKeyTarget"]
            .as_str()
            .ok_or("round-trip object mapping was missing")?
            .to_owned(),
    )?;
    assert_shared_action_parity(
        &roundtrip_scene,
        &roundtrip_target_id,
        &expected,
        "round-trip import",
    )?;
    Ok(())
}
