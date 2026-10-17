#![expect(
    clippy::unwrap_used,
    reason = "Blender bridge fixtures use fixed names, paths, and control-point values"
)]

use std::{error::Error, fs, process::Command};

use potter_core::model::{Modifier, SceneDoc};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
#[path = "common/pot_json_guarded.rs"]
mod pot_json_guarded;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script_guarded::run_blender_script;
use pot_json_guarded::pot_json;
use process::run_guarded;

const SUBJECT_NAME: &str = "CurveMappingSubject";
const HOOK_NAME: &str = "HookCurveMapping";
const WARP_NAME: &str = "WarpCurveMapping";
const RAW_HOOK_NAME: &str = "RawHookCurveMapping";
const RAW_VWE_NAME: &str = "RawVertexWeightEditCurveMapping";
const HOOK_POINTS: [[f64; 2]; 4] = [[0.0, 0.125], [0.25, 0.875], [0.625, 0.375], [1.0, 0.75]];
const WARP_POINTS: [[f64; 2]; 4] = [[0.0, 0.25], [0.375, 0.875], [0.75, 0.125], [1.0, 0.5]];
const RAW_CURVE_POINTS: [[f64; 2]; 4] = [[0.0, 0.125], [1.5, 0.75], [-0.25, 0.875], [0.5, -0.25]];

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

arguments = sys.argv[sys.argv.index("--") + 1:]
root = os.path.realpath(arguments[0])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

mesh = bpy.data.meshes.new("CurveMappingMesh")
mesh.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
mesh.update()
subject = bpy.data.objects.new("CurveMappingSubject", mesh)
scene.collection.objects.link(subject)

hook_target = bpy.data.objects.new("HookTarget", None)
scene.collection.objects.link(hook_target)
warp_from = bpy.data.objects.new("WarpFrom", None)
scene.collection.objects.link(warp_from)
warp_to = bpy.data.objects.new("WarpTo", None)
scene.collection.objects.link(warp_to)


def set_curve(modifier, points):
    modifier.falloff_type = "CURVE"
    curve = modifier.falloff_curve.curves[0]
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    curve.points[0].location = tuple(points[0])
    curve.points[1].location = tuple(points[-1])
    for x, y in points[1:-1]:
        curve.points.new(float(x), float(y))
    modifier.falloff_curve.update()


def set_raw_curve(mapping, points):
    curve = mapping.curves[0]
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    # CurveMapPoints.new inserts by x, so use increasing placeholders before restoring raw order.
    for index in range(len(points) - 2):
        curve.points.new(float(index + 2), 0.5)
    # Do not update the mapping: that would sort and clip these intentionally raw controls.
    for point, coordinates in zip(curve.points, points):
        point.location = tuple(coordinates)


raw_points = [(0.0, 0.125), (1.5, 0.75), (-0.25, 0.875), (0.5, -0.25)]


hook = subject.modifiers.new("HookCurveMapping", "HOOK")
hook.object = hook_target
hook.vertex_indices_set([0, 1, 2, 3])
set_curve(hook, [(0.0, 0.125), (0.25, 0.875), (0.625, 0.375), (1.0, 0.75)])

warp = subject.modifiers.new("WarpCurveMapping", "WARP")
warp.object_from = warp_from
warp.object_to = warp_to
set_curve(warp, [(0.0, 0.25), (0.375, 0.875), (0.75, 0.125), (1.0, 0.5)])

raw_hook = subject.modifiers.new("RawHookCurveMapping", "HOOK")
raw_hook.object = hook_target
raw_hook.vertex_indices_set([0, 1, 2, 3])
raw_hook.falloff_type = "CURVE"
set_raw_curve(raw_hook.falloff_curve, raw_points)

raw_weight_group = subject.vertex_groups.new(name="RawCurveWeights")
raw_weight_group.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
raw_vwe = subject.modifiers.new("RawVertexWeightEditCurveMapping", "VERTEX_WEIGHT_EDIT")
raw_vwe.vertex_group = raw_weight_group.name
raw_vwe.falloff_type = "CURVE"
set_raw_curve(raw_vwe.map_curve, raw_points)


def modifier_record(modifier, curve_property):
    curve = getattr(modifier, curve_property).curves[0]
    return {
        "type": modifier.type,
        "falloff_type": modifier.falloff_type,
        curve_property: [
            [float(point.location[0]), float(point.location[1])]
            for point in curve.points
        ],
    }


bpy.ops.wm.save_as_mainfile(
    filepath=os.path.join(root, "curve_mapping_source.blend"),
    check_existing=False,
)
with open(os.path.join(root, "source_controls.json"), "w", encoding="utf-8") as handle:
    json.dump(
        {
            "HookCurveMapping": modifier_record(hook, "falloff_curve"),
            "WarpCurveMapping": modifier_record(warp, "falloff_curve"),
            "RawHookCurveMapping": modifier_record(raw_hook, "falloff_curve"),
            "RawVertexWeightEditCurveMapping": modifier_record(raw_vwe, "map_curve"),
        },
        handle,
    )
"#;

const BLENDER_REOPEN: &str = r#"
import bpy
import json
import os
import sys

arguments = sys.argv[sys.argv.index("--") + 1:]
output_path = os.path.realpath(arguments[0])
subject = bpy.data.objects["CurveMappingSubject"]


def modifier_record(name, curve_property):
    modifier = subject.modifiers[name]
    curve = getattr(modifier, curve_property).curves[0]
    return {
        "type": modifier.type,
        "falloff_type": modifier.falloff_type,
        curve_property: [
            [float(point.location[0]), float(point.location[1])]
            for point in curve.points
        ],
    }


with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "HookCurveMapping": modifier_record("HookCurveMapping", "falloff_curve"),
            "WarpCurveMapping": modifier_record("WarpCurveMapping", "falloff_curve"),
            "RawHookCurveMapping": modifier_record("RawHookCurveMapping", "falloff_curve"),
            "RawVertexWeightEditCurveMapping": modifier_record("RawVertexWeightEditCurveMapping", "map_curve"),
        },
        handle,
    )
"#;

fn assert_blender_curve(
    snapshot: &Value,
    name: &str,
    blender_type: &str,
    points_key: &str,
    expected: &[[f64; 2]],
) {
    let modifier = &snapshot[name];
    assert_eq!(modifier["type"], blender_type, "{name}: Blender modifier type");
    assert_eq!(modifier["falloff_type"], "CURVE", "{name}: falloff type");
    let points = &modifier[points_key];
    assert!(
        points.is_array(),
        "{name}: expected control-point array, got {points}"
    );
    assert_eq!(points, &json!(expected), "{name}: raw Blender point order/values");
}

fn find_modifier<'a>(scene: &'a SceneDoc, name: &str) -> &'a Modifier {
    scene
        .nodes
        .values()
        .find(|node| node.name == SUBJECT_NAME)
        .unwrap_or_else(|| panic!("missing Blender subject {SUBJECT_NAME}"))
        .modifiers
        .iter()
        .find(|modifier| modifier.name == name)
        .unwrap_or_else(|| panic!("missing modifier {name} on {SUBJECT_NAME}"))
}

fn assert_scene_curve(
    scene: &SceneDoc,
    name: &str,
    modifier_type: &str,
    points_key: &str,
    expected: &[[f64; 2]],
) {
    let modifier = find_modifier(scene, name);
    assert_eq!(modifier.modifier_type, modifier_type, "{name}: model type");
    assert_eq!(modifier.params["falloff_type"], "CURVE", "{name}: model falloff type");
    let points = &modifier.params[points_key];
    assert!(
        points.is_array(),
        "{name}: expected serialized control-point array, got {points}"
    );
    assert_eq!(points, &json!(expected), "{name}: imported control-point order/values");
}

#[test]
fn blender_curve_mapping_controls_round_trip_exactly() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Hook/Warp/VWE CurveMapping round-trip: Blender is unavailable");
        return Ok(());
    };

    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_curve_mapping_fixture.py",
        BLENDER_FIXTURE,
        root,
        &[],
        "Blender CurveMapping fixture failed",
        run_guarded,
    )?;

    let source: Value = serde_json::from_slice(&fs::read(root.join("source_controls.json"))?)?;
    assert_blender_curve(&source, HOOK_NAME, "HOOK", "falloff_curve", &HOOK_POINTS);
    assert_blender_curve(&source, WARP_NAME, "WARP", "falloff_curve", &WARP_POINTS);
    assert_blender_curve(
        &source,
        RAW_HOOK_NAME,
        "HOOK",
        "falloff_curve",
        &RAW_CURVE_POINTS,
    );
    assert_blender_curve(
        &source,
        RAW_VWE_NAME,
        "VERTEX_WEIGHT_EDIT",
        "map_curve",
        &RAW_CURVE_POINTS,
    );

    let project = root.join("project");
    let project_path = project.to_str().expect("temporary project path is UTF-8");
    let blender_path = blender.to_str().expect("Blender executable path is UTF-8");
    pot_json(&["init", project_path])?;
    let imported = pot_json(&[
        "import",
        project_path,
        "--file",
        root.join("curve_mapping_source.blend")
            .to_str()
            .expect("temporary source path is UTF-8"),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_path,
    ])?;
    assert_eq!(imported["result"]["losses"], json!([]), "{imported}");

    let scene: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_scene_curve(&scene, HOOK_NAME, "hook", "falloff_curve", &HOOK_POINTS);
    assert_scene_curve(&scene, WARP_NAME, "warp", "falloff_curve", &WARP_POINTS);
    assert_scene_curve(
        &scene,
        RAW_HOOK_NAME,
        "hook",
        "falloff_curve",
        &RAW_CURVE_POINTS,
    );
    assert_scene_curve(
        &scene,
        RAW_VWE_NAME,
        "vertex_weight_edit",
        "map_curve",
        &RAW_CURVE_POINTS,
    );

    let exported = root.join("curve_mapping_roundtrip.blend");
    pot_json(&[
        "export",
        project_path,
        "--format",
        "blend",
        "--out",
        exported.to_str().expect("temporary export path is UTF-8"),
        "--blender",
        blender_path,
    ])?;

    let reopen_script = root.join("reopen_curve_mapping.py");
    fs::write(&reopen_script, BLENDER_REOPEN)?;
    let reopened_path = root.join("reopened_controls.json");
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_script)
        .arg("--")
        .arg(&reopened_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender CurveMapping reopen failed: stdout={stdout} stderr={stderr}"
    );
    let reopened: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_blender_curve(&reopened, HOOK_NAME, "HOOK", "falloff_curve", &HOOK_POINTS);
    assert_blender_curve(&reopened, WARP_NAME, "WARP", "falloff_curve", &WARP_POINTS);
    assert_blender_curve(
        &reopened,
        RAW_HOOK_NAME,
        "HOOK",
        "falloff_curve",
        &RAW_CURVE_POINTS,
    );
    assert_blender_curve(
        &reopened,
        RAW_VWE_NAME,
        "VERTEX_WEIGHT_EDIT",
        "map_curve",
        &RAW_CURVE_POINTS,
    );

    Ok(())
}
