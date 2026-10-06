use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use potter::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10


def set_clamped_bezier(owner):
    action = owner.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                channelbag = strip.channelbag(slot)
                if channelbag is None:
                    continue
                for curve in channelbag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "BEZIER"
                        point.handle_left_type = "AUTO_CLAMPED"
                        point.handle_right_type = "AUTO_CLAMPED"
                    curve.update()


animated = bpy.data.objects.new("IK_Target_Animated", None)
scene.collection.objects.link(animated)
for frame, location in (
    (1, (-4.0, 4.0, 2.45)),
    (10, (-3.55, 4.10, 2.20)),
):
    animated.location = location
    animated.keyframe_insert(data_path="location", frame=frame)
set_clamped_bezier(animated)

path_data = bpy.data.curves.new("AnimatedPathData", type="CURVE")
path_data.dimensions = "3D"
path_data.use_path = True
path_data.path_duration = 10
path_data.eval_time = 0.0
spline = path_data.splines.new("POLY")
spline.points.add(1)
spline.points[0].co = (-1.0, 0.0, 0.0, 1.0)
spline.points[1].co = (9.0, 0.0, 0.0, 1.0)
path = bpy.data.objects.new("AnimatedPath", path_data)
scene.collection.objects.link(path)
for frame, value in ((1, 0.0), (10, 10.0)):
    path_data.eval_time = value
    path_data.keyframe_insert(data_path="eval_time", frame=frame)
set_clamped_bezier(path_data)

follower = bpy.data.objects.new("Path_Follower_Empty", None)
scene.collection.objects.link(follower)
follow = follower.constraints.new("FOLLOW_PATH")
follow.target = path
follow.use_fixed_location = False
follow.use_curve_follow = True

marker = bpy.data.objects.new("Path_Follower_CopyRotation_Marker", None)
scene.collection.objects.link(marker)
marker.parent = follower
copy_rotation = marker.constraints.new("COPY_ROTATION")
copy_rotation.target = follower
copy_rotation.influence = 0.5
copy_rotation.mix_mode = "REPLACE"


def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


expected = {}
for frame in (1, 5, 10):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    expected[str(frame)] = {
        name: matrix_values(bpy.data.objects[name].evaluated_get(depsgraph).matrix_world)
        for name in (
            "IK_Target_Animated",
            "Path_Follower_Empty",
            "Path_Follower_CopyRotation_Marker",
        )
    }
with open(os.path.join(root, "blender_expected.json"), "w", encoding="utf-8") as output:
    json.dump({"frames": expected}, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

fn blender_executable() -> Option<PathBuf> {
    fn usable(path: PathBuf) -> Option<PathBuf> {
        path.is_file().then_some(path)
    }

    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(path.into());
    }
    if let Some(path) = env::var_os("PATH")
        && let Some(path) = env::split_paths(&path)
            .map(|directory| directory.join("blender"))
            .find_map(usable)
    {
        return Some(path);
    }
    usable("/Applications/Blender.app/Contents/MacOS/Blender".into())
}

fn checked_output(mut command: Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn generate_blender_fixture(blender: &Path, root: &Path) -> Result<(), Box<dyn Error>> {
    let script_path = root.join("make_fixture.py");
    fs::write(&script_path, BLENDER_FIXTURE)?;
    let root = fs::canonicalize(root)?;
    let root = root
        .to_str()
        .ok_or_else(|| io::Error::other("temporary Blender fixture path is not UTF-8"))?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .args(["--", root]);
    checked_output(command, "Blender animation fixture")?;
    Ok(())
}

fn node_id(doc: &SceneDoc, name: &str) -> Result<Id, Box<dyn Error>> {
    doc.nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .map(|(id, _)| id.clone())
        .ok_or_else(|| io::Error::other(format!("imported object `{name}` is missing")).into())
}

fn assert_matrix_matches_blender(
    actual: &[f64; 16],
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let expected = expected
        .as_array()
        .filter(|matrix| matrix.len() == 16)
        .ok_or_else(|| io::Error::other(format!("{context}: Blender matrix is malformed")))?;
    let mut max_error = 0.0_f64;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected
            .as_f64()
            .ok_or_else(|| io::Error::other(format!("{context}: matrix element is not numeric")))?;
        max_error = max_error.max((actual - expected).abs());
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{context}: matrix element {index} differs: Potter={actual}, Blender={expected}; max error={max_error}"
        );
    }
    Ok(())
}

#[test]
fn blender_bezier_handles_drive_object_and_animated_path_evaluation() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender animation parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    generate_blender_fixture(&blender, root)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;

    let project = root.join("project");
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.arg("init").arg(&project).arg("--json");
    checked_output(init, "pot init")?;

    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .arg("import")
        .arg(&project)
        .args(["--file"])
        .arg(root.join("source.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let imported: Value =
        serde_json::from_slice(&checked_output(import, "pot blend import")?.stdout)?;
    assert_eq!(imported["result"]["losses"], json!([]));

    let doc: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let objects = [
        "IK_Target_Animated",
        "Path_Follower_Empty",
        "Path_Follower_CopyRotation_Marker",
    ];
    let ids = objects
        .map(|name| node_id(&doc, name))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    for frame in [1.0, 5.0, 10.0] {
        let snapshot = Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                frame: Some(frame),
                ..EvaluationContext::default()
            },
        )?;
        for (name, id) in objects.iter().zip(&ids) {
            let actual = &snapshot
                .nodes
                .get(id)
                .ok_or_else(|| io::Error::other(format!("evaluated object `{name}` is missing")))?
                .world_matrix;
            assert_matrix_matches_blender(
                actual,
                &expected["frames"][frame.to_string()][*name],
                &format!("{name} at frame {frame}"),
            )?;
        }
    }
    Ok(())
}
