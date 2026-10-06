use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> Result<std::process::Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    run_ok(pot().args(["init"]).arg(scene).arg("--json"))?;
    Ok(())
}

fn add_box(scene: &Path) -> Result<(), Box<dyn Error>> {
    let operations = scene.with_extension("operations.json");
    fs::write(&operations, include_str!("fixtures/first.json"))?;
    run_ok(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(&operations)
            .arg("--json"),
    )?;
    let path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&path)?)?;
    document["scenes"]["scene_main"]["frame_end"] = json!(2);
    document["nodes"]["body"]["action"] = json!("box_motion");
    document["actions"]["box_motion"] = json!({
        "name": "Box Motion",
        "fcurves": [
            {"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":0.0},{"frame":2.0,"value":2.0}]}
        ]
    });
    fs::write(path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn vertex_count(scene: &Path) -> Result<usize, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    Ok(document["data_blocks"]
        .as_object()
        .ok_or("missing data_blocks")?
        .values()
        .filter_map(|block| block["mesh"]["vertices"].as_array())
        .map(Vec::len)
        .sum())
}

fn export_import_round_trip(
    source: &Path,
    destination: &Path,
    format: &str,
    filename: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let output = source.join(filename);
    run_ok(
        pot()
            .args(["export"])
            .arg(source)
            .args(["--format", format, "--allow-lossy", "--out"])
            .arg(&output)
            .args(["--overwrite", "--json"]),
    )?;
    init(destination)?;
    run_ok(
        pot()
            .args(["import"])
            .arg(destination)
            .args(["--file"])
            .arg(&output)
            .args([
                "--format",
                format,
                "--allow-lossy",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ]),
    )?;
    assert_eq!(vertex_count(destination)?, vertex_count(source)?);
    let action_value = action_curve_value(destination, "body", "transform.translation", 0, 2.0)?;
    assert!(
        (action_value - 2.0).abs() < 1.0e-6,
        "{format} round-trip returned animation value {action_value}"
    );
    Ok(output)
}

fn blender_executable() -> Option<PathBuf> {
    fn usable(path: PathBuf) -> Option<PathBuf> {
        path.is_file().then_some(path)
    }
    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join("blender"))
            .find_map(usable)
    }) {
        return Some(path);
    }
    usable(PathBuf::from(
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ))
}

fn assert_blender_import(
    path: &Path,
    format: &str,
    expected_vertices: usize,
) -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender {format} import check: Blender is unavailable");
        return Ok(());
    };
    let script = r"
import bpy, sys
args = sys.argv[sys.argv.index('--') + 1:]
format_name, path = args
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
if format_name in ('usdc', 'usd'):
    bpy.ops.wm.usd_import(filepath=path)
elif format_name == 'alembic':
    bpy.ops.wm.alembic_import(filepath=path, as_background_job=False)
elif format_name == 'fbx-binary':
    bpy.ops.import_scene.fbx(filepath=path)
elif format_name == 'bvh':
    bpy.ops.import_anim.bvh(filepath=path)
count = sum(len(obj.data.vertices) for obj in bpy.context.scene.objects if obj.type == 'MESH')
print('POTTER_VERTEX_COUNT=' + str(count))
";
    let output = Command::new(&blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python-expr",
            script,
            "--",
            format,
        ])
        .arg(path)
        .output()?;
    assert!(
        output.status.success(),
        "Blender import of {} failed:\n{}\n{}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let output_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let count = output_text
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_VERTEX_COUNT="))
        .ok_or("Blender import did not report vertex count")?
        .parse::<usize>()?;
    assert_eq!(
        count,
        expected_vertices,
        "Blender vertex count after importing {}",
        path.display()
    );
    Ok(())
}

#[test]
fn usdc_alembic_and_binary_fbx_round_trip() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    init(&source)?;
    add_box(&source)?;
    let expected_vertices = vertex_count(&source)?;
    for (format, extension) in [
        ("usdc", "scene.usdc"),
        ("usd", "scene.usd"),
        ("alembic", "scene.abc"),
        ("fbx-binary", "scene.fbx"),
    ] {
        let destination = temporary.path().join(format!("destination-{format}.pot"));
        let output = export_import_round_trip(&source, &destination, format, extension)?;
        let bytes = fs::read(&output)?;
        match format {
            "usdc" | "usd" => assert!(bytes.starts_with(b"PXR-USDC"), "not a USD crate file"),
            "alembic" => assert!(bytes.starts_with(b"Ogawa"), "not an Alembic Ogawa file"),
            "fbx-binary" => assert!(
                bytes.starts_with(b"Kaydara FBX Binary  \0\x1a\0"),
                "not binary FBX"
            ),
            _ => unreachable!(),
        }
        assert_blender_import(&output, format, expected_vertices)?;
    }
    Ok(())
}

fn add_armature(scene: &Path) -> Result<(), Box<dyn Error>> {
    let path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&path)?)?;
    document["data_blocks"]["rig_data"] = json!({
        "type": "armature",
        "armature": {
            "bones": {
                "root": {"name":"Root", "parent":null, "head":[0.0,0.0,0.0], "tail":[0.0,0.0,1.0]},
                "child": {"name":"Child", "parent":"root", "head":[0.0,0.0,1.0], "tail":[0.0,0.0,2.0]}
            }
        }
    });
    document["nodes"]["rig"] =
        json!({"name":"Rig", "kind":"armature", "data":"rig_data", "action":"walk"});
    document["actions"]["walk"] = json!({
        "name": "Walk",
        "fcurves": [
            {"path":"pose.bones[\"root\"].location","index":0,"keyframes":[{"frame":1.0,"value":0.0},{"frame":250.0,"value":1.0}]},
            {"path":"pose.bones[\"child\"].rotation","index":2,"keyframes":[{"frame":1.0,"value":0.0},{"frame":250.0,"value":std::f64::consts::FRAC_1_SQRT_2}]},
            {"path":"pose.bones[\"child\"].rotation","index":3,"keyframes":[{"frame":1.0,"value":1.0},{"frame":250.0,"value":std::f64::consts::FRAC_1_SQRT_2}]}
        ]
    });
    document["collections"]["collection_root"]["objects"]
        .as_array_mut()
        .ok_or("root collection has no object list")?
        .push(json!("rig"));
    fs::write(path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn armature_bone_count(scene: &Path) -> Result<usize, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    Ok(document["data_blocks"]
        .as_object()
        .ok_or("missing data_blocks")?
        .values()
        .find_map(|block| block["armature"]["bones"].as_object())
        .map_or(0, serde_json::Map::len))
}
fn action_curve_value(
    scene: &Path,
    node_id: &str,
    path: &str,
    index: u32,
    frame: f64,
) -> Result<f64, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let action_id = document["nodes"][node_id]["action"]
        .as_str()
        .ok_or("node has no action")?;
    let action = document["actions"]
        .get(action_id)
        .ok_or("node action is missing")?;
    let curves = action["fcurves"]
        .as_array()
        .ok_or("missing action curves")?;
    for curve in curves {
        if curve["path"].as_str() != Some(path) || curve["index"].as_u64() != Some(u64::from(index))
        {
            continue;
        }
        let keyframes = curve["keyframes"].as_array().ok_or("missing keyframes")?;
        let keyframe = keyframes
            .iter()
            .find(|keyframe| keyframe["frame"].as_f64() == Some(frame))
            .ok_or("missing requested keyframe")?;
        return keyframe["value"]
            .as_f64()
            .ok_or_else(|| "keyframe value is not numeric".into());
    }
    Err("missing action curve".into())
}

#[test]
fn bvh_rig_round_trip_and_blender_import() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("rig.pot");
    init(&source)?;
    add_armature(&source)?;
    let output = source.join("rig.bvh");
    run_ok(
        pot()
            .args(["export"])
            .arg(&source)
            .args(["--format", "bvh", "--allow-lossy", "--out"])
            .arg(&output)
            .args(["--overwrite", "--json"]),
    )?;
    assert!(fs::read_to_string(&output)?.starts_with("HIERARCHY\n"));
    let destination = temporary.path().join("roundtrip.pot");
    init(&destination)?;
    run_ok(
        pot()
            .args(["import"])
            .arg(&destination)
            .args(["--file"])
            .arg(&output)
            .args([
                "--format",
                "bvh",
                "--allow-lossy",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ]),
    )?;
    assert_eq!(
        armature_bone_count(&destination)?,
        armature_bone_count(&source)?
    );
    let imported: Value = serde_json::from_slice(&fs::read(destination.join("scene.json"))?)?;
    assert_eq!(imported["scenes"]["scene_main"]["frame_end"], 250);
    assert_eq!(imported["scenes"]["scene_main"]["fps"], 24);
    assert!(
        (action_curve_value(
            &destination,
            "armature",
            "pose.bones[\"root\"].location",
            0,
            250.0
        )? - 1.0)
            .abs()
            < 1.0e-6
    );
    assert!(
        (action_curve_value(
            &destination,
            "armature",
            "pose.bones[\"child\"].rotation",
            2,
            250.0
        )? - std::f64::consts::FRAC_1_SQRT_2)
            .abs()
            < 1.0e-6
    );
    assert_blender_import(&output, "bvh", 0)?;
    Ok(())
}
