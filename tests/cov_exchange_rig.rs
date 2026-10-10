use std::{error::Error, fs, path::Path, process::Command};

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const FORMATS: [(&str, &str); 4] = [
    ("usda", "rig.usda"),
    ("gltf", "rig.gltf"),
    ("fbx", "rig.fbx"),
    ("alembic", "rig.abc"),
];

const BLENDER_IMPORT_SCRIPT: &str = r#"
import bpy
import json
import sys
from mathutils import Vector
from pxr import Gf, Usd, UsdGeom, UsdSkel

if bpy.app.version != (5, 2, 2):
    print("POTTER_BLENDER_VERSION_SKIP=" + str(bpy.app.version))
    sys.exit(0)
def usd_probe(path):
    stage = Usd.Stage.Open(path)
    mesh_prim = stage.GetPrimAtPath("/Scene/N_mesh/Geometry")
    binding = UsdSkel.BindingAPI(mesh_prim)
    skeleton_path = binding.GetSkeletonRel().GetTargets()[0]
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath(str(skeleton_path)))
    joints = list(skeleton.GetJointsAttr().Get())
    indices_primvar = binding.GetJointIndicesPrimvar()
    weights_primvar = binding.GetJointWeightsPrimvar()
    element_size = indices_primvar.GetElementSize()
    indices = list(indices_primvar.Get())
    weights = list(weights_primvar.Get())
    row = 1 * element_size
    root_weight = child_weight = None
    for joint_index, weight in zip(indices[row:row + element_size],
                                   weights[row:row + element_size]):
        if joints[joint_index] == "J_root":
            root_weight = float(weight)
        elif joints[joint_index] == "J_root/J_child":
            child_weight = float(weight)

    target_path = binding.GetBlendShapeTargetsRel().GetTargets()[0]
    blend_shape = UsdSkel.BlendShape(stage.GetPrimAtPath(str(target_path)))
    point_indices = list(blend_shape.GetPointIndicesAttr().Get())
    offsets = list(blend_shape.GetOffsetsAttr().Get())
    shape_weights = list(mesh_prim.GetAttribute("skel:blendShapeWeights").Get())
    points = list(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get())
    for index, offset in zip(point_indices, offsets):
        points[index] = points[index] + offset * shape_weights[0]
    center = Gf.Vec3d(
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )
    xform = stage.GetPrimAtPath("/Scene/N_mesh")
    def world_center(frame):
        matrix = UsdGeom.XformCache(Usd.TimeCode(frame)).GetLocalToWorldTransform(xform)
        result = matrix.Transform(center)
        return [float(result[index]) for index in range(3)]

    shape_delta = max(offset.GetLength() for offset in offsets)
    return {
        "bone_count": len(joints),
        "root_weight": root_weight,
        "child_weight": child_weight,
        "shape_delta": float(shape_delta),
        "start_position": world_center(1.0),
        "finish_position": world_center(3.0),
        "animated_delta": [
            right - left for left, right in zip(world_center(1.0), world_center(3.0))
        ],
    }


args = sys.argv[sys.argv.index("--") + 1:]
results = {}
for format_name, path in zip(args[0::2], args[1::2]):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if format_name in ("usd", "usda"):
        bpy.ops.wm.usd_import(filepath=path)
    elif format_name == "gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif format_name == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif format_name == "alembic":
        bpy.ops.wm.alembic_import(filepath=path, as_background_job=False)
    else:
        raise RuntimeError("unexpected format " + format_name)
    if format_name == "usda":
        results[format_name] = usd_probe(path)
        continue

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(format_name + " import contained no mesh")
    mesh = next((obj for obj in meshes if obj.data.shape_keys), meshes[0])

    def center(obj):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh.vertices:
                raise RuntimeError(obj.name + " has no vertices")
            return sum((evaluated.matrix_world @ vertex.co for vertex in mesh.vertices), Vector()) / len(mesh.vertices)
        finally:
            evaluated.to_mesh_clear()

    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    start = center(mesh)
    bpy.context.scene.frame_set(3)
    bpy.context.view_layer.update()
    finish = center(mesh)

    bones = sum(len(obj.data.bones) for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    group_weights = {}
    if len(mesh.data.vertices) > 1:
        groups = {group.index: group.name for group in mesh.vertex_groups}
        for assignment in mesh.data.vertices[1].groups:
            group_weights[groups[assignment.group]] = assignment.weight
    shape_delta = None
    if mesh.data.shape_keys:
        basis = mesh.data.shape_keys.key_blocks.get("Basis")
        target = mesh.data.shape_keys.key_blocks.get("Lift")
        if basis and target:
            shape_delta = (target.data[0].co - basis.data[0].co).length
    results[format_name] = {
        "bone_count": bones,
        "root_weight": group_weights.get("Root"),
        "child_weight": group_weights.get("Child"),
        "shape_delta": shape_delta,
        "start_position": [float(start[i]) for i in range(3)],
        "finish_position": [float(finish[i]) for i in range(3)],
        "animated_delta": [float(finish[i] - start[i]) for i in range(3)],
    }
print("POTTER_EXCHANGE_RIG=" + json.dumps(results, sort_keys=True))
"#;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> TestResult<std::process::Output> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "command failed ({}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(output)
}
fn assert_vec_close(actual: DVec3, expected: DVec3, format: &str, label: &str) {
    assert!(
        (actual - expected).length() < 1.0e-4,
        "{format}: {label} position {actual:?} differs from Potter evaluation {expected:?}"
    );
}

fn init(scene: &Path) -> TestResult {
    run_ok(pot().arg("init").arg(scene).arg("--json"))?;
    Ok(())
}

fn create_rigged_scene(scene: &Path) -> TestResult {
    let operations = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op":"node.create","id":"mesh","name":"RiggedPlane","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"arm","name":"Rig","kind":"armature"},
            {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]},
            {"op":"bone.create","target":{"id":"arm"},"id":"child","name":"Child","head":[0.0,0.0,1.0],"tail":[0.0,0.0,2.0],"parent":"root"},
            {"op":"vertex_group.create","target":{"id":"mesh"},"id":"root_group","name":"Root"},
            {"op":"vertex_group.create","target":{"id":"mesh"},"id":"child_group","name":"Child"},
            {"op":"vertex_group.assign","target":{"id":"mesh"},"group_id":"root_group","weights":[
                {"vertex_id":0,"weight":1.0},{"vertex_id":1,"weight":0.75},
                {"vertex_id":2,"weight":0.25},{"vertex_id":3,"weight":0.0}
            ]},
            {"op":"vertex_group.assign","target":{"id":"mesh"},"group_id":"child_group","weights":[
                {"vertex_id":0,"weight":0.0},{"vertex_id":1,"weight":0.25},
                {"vertex_id":2,"weight":0.75},{"vertex_id":3,"weight":1.0}
            ]},
            {"op":"modifier.create","target":{"id":"mesh"},"id":"skin","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
            {"op":"shape_key.create","target":{"id":"mesh"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,0.5]}},
            {"op":"shape_key.update","target":{"id":"mesh"},"id":"lift","set":{"value":1.0}},
            {"op":"action.create","id":"travel","name":"Travel"},
            {"op":"node.update","target":{"id":"mesh"},"set":{"action":"travel"}},
            {"op":"keyframe.insert","target":{"id":"mesh"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0,"interpolation":"linear"},
            {"op":"keyframe.insert","target":{"id":"mesh"},"path":"transform.translation","index":0,"frame":3.0,"value":2.0,"interpolation":"linear"}
        ]
    });
    let batch = scene.join("rig-operations.json");
    fs::write(&batch, serde_json::to_vec(&operations)?)?;
    run_ok(
        pot()
            .arg("apply")
            .arg(scene)
            .arg("--file")
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}

fn load_doc(scene: &Path) -> TestResult<SceneDoc> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

fn evaluate(scene: &Path, frame: f64) -> TestResult<Snapshot> {
    let document = load_doc(scene)?;
    Ok(Snapshot::evaluate(
        &document,
        &EvaluationContext {
            frame: Some(frame),
            ..EvaluationContext::default()
        },
    )?)
}

fn mesh_center(snapshot: &Snapshot, node_id: &str) -> TestResult<DVec3> {
    let id = Id::new(node_id.to_owned())?;
    let mesh = snapshot.meshes.get(&id).ok_or("evaluated mesh missing")?;
    let node = snapshot.nodes.get(&id).ok_or("evaluated node missing")?;
    let world = DMat4::from_cols_array(&node.world_matrix);
    let total = mesh.vertices.iter().fold(DVec3::ZERO, |sum, vertex| {
        sum + world.transform_point3(vertex.co)
    });
    Ok(total / mesh.vertices.len() as f64)
}

fn mesh_data<'a>(
    doc: &'a SceneDoc,
    node_id: &str,
) -> TestResult<&'a potter_core::model::DataBlock> {
    let id = Id::new(node_id.to_owned())?;
    let node = doc.nodes.get(&id).ok_or("mesh node missing")?;
    let data_id = node.data.as_ref().ok_or("mesh data reference missing")?;
    doc.data_blocks
        .get(data_id)
        .ok_or_else(|| "mesh data block missing".into())
}

fn assert_imported_rig(doc: &SceneDoc, format: &str) -> TestResult {
    let armature = doc.nodes.values().find(|node| node.kind == "armature");
    let armature = armature.ok_or_else(|| format!("{format} import lost the armature"))?;
    let armature_data_id = armature
        .data
        .as_ref()
        .ok_or("armature data reference missing")?;
    let bones = doc.data_blocks[armature_data_id]
        .armature
        .as_ref()
        .ok_or("armature data missing")?;
    assert_eq!(bones.bones.len(), 2, "{format}: bone count");

    let data = mesh_data(doc, "mesh")?;
    let lift = data
        .shape_keys
        .as_ref()
        .and_then(|keys| keys.keys.values().find(|key| key.name == "Lift"));
    let lift = lift.ok_or_else(|| format!("{format} import lost the Lift morph target"))?;
    assert!(
        lift.positions.iter().any(|(vertex_id, position)| {
            data.shape_keys
                .as_ref()
                .and_then(|keys| keys.basis.get(vertex_id))
                .is_some_and(|basis| {
                    (DVec3::from_array(*position) - DVec3::from_array(*basis)).length() > 0.49
                })
        }),
        "{format}: shape-key vertex delta was not retained"
    );
    for (group_name, expected) in [("Root", 0.75), ("Child", 0.25)] {
        let group = data
            .vertex_groups
            .iter()
            .find(|group| group.name == group_name)
            .ok_or_else(|| format!("{format} import lost vertex group {group_name}"))?;
        assert!(
            data.vertex_weights.values().any(|weights| weights
                .get(&group.id)
                .is_some_and(|weight| (*weight - expected).abs() < 1.0e-5)),
            "{format}: {group_name} skin weights did not preserve the authored blend"
        );
    }
    Ok(())
}

fn export(scene: &Path, format: &str, file: &Path) -> TestResult {
    run_ok(
        pot()
            .arg("export")
            .arg(scene)
            .args(["--format", format, "--allow-lossy", "--out"])
            .arg(file)
            .args(["--overwrite", "--json"]),
    )?;
    Ok(())
}

fn import(scene: &Path, file: &Path, format: &str) -> TestResult {
    run_ok(
        pot()
            .arg("import")
            .arg(scene)
            .arg("--file")
            .arg(file)
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
    Ok(())
}

fn blender_import(paths: &[(&str, std::path::PathBuf)]) -> TestResult<Option<Value>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping exchange rig Blender parity: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
        "--python-expr",
        BLENDER_IMPORT_SCRIPT,
        "--",
    ]);
    for (format, path) in paths {
        command.arg(*format).arg(path);
    }
    let output = command.output()?;
    assert!(
        output.status.success(),
        "Blender import failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let output_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if output_text.contains("POTTER_BLENDER_VERSION_SKIP=") {
        eprintln!("skipping exchange rig Blender parity: this fixture requires Blender 5.2.2");
        return Ok(None);
    }
    let payload = output_text
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_EXCHANGE_RIG="))
        .ok_or("Blender did not report exchange rig results")?;
    Ok(Some(serde_json::from_str(payload)?))
}

#[test]
fn rigged_exchange_formats_preserve_skin_morphs_and_animation() -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("source.pot");
    init(&source)?;
    create_rigged_scene(&source)?;

    let at_start = evaluate(&source, 1.0)?;
    let at_end = evaluate(&source, 3.0)?;
    let expected_start = mesh_center(&at_start, "mesh")?;
    let expected_end = mesh_center(&at_end, "mesh")?;
    let expected_motion = expected_end - expected_start;
    assert!((expected_motion - DVec3::new(2.0, 0.0, 0.0)).length() < 1.0e-9);
    let blender_available = blender_file::blender_executable().is_some();

    let mut blender_paths = Vec::<(&str, std::path::PathBuf)>::new();
    for (format, filename) in FORMATS {
        let output = directory.path().join(filename);
        export(&source, format, &output)?;
        let round_trip = directory.path().join(format!("round-trip-{format}"));
        init(&round_trip)?;
        import(&round_trip, &output, format)?;

        let imported = load_doc(&round_trip)?;
        let imported_start = evaluate(&round_trip, 1.0)?;
        let imported_end = evaluate(&round_trip, 3.0)?;
        let imported_start_position = mesh_center(&imported_start, "mesh")?;
        let imported_end_position = mesh_center(&imported_end, "mesh")?;
        assert_vec_close(
            imported_start_position,
            expected_start,
            format,
            "round-trip frame 1",
        );
        assert_vec_close(
            imported_end_position,
            expected_end,
            format,
            "round-trip frame 3",
        );
        if format == "alembic" {
            let cache_data = mesh_data(&imported, "mesh")?;
            let samples = &cache_data
                .mesh
                .as_ref()
                .ok_or("Alembic mesh missing")?
                .attributes["alembic_time_samples"];
            assert_eq!(
                samples["frames"].as_array().map(Vec::len),
                Some(2),
                "Alembic frame samples"
            );
            assert!(
                samples["positions"].is_array(),
                "Alembic point samples were not retained"
            );
        } else {
            assert_imported_rig(&imported, format)?;
        }
        if format == "fbx" {
            if blender_available {
                let binary_fbx = directory.path().join("rig-binary.fbx");
                export(&source, "fbx-binary", &binary_fbx)?;
                blender_paths.push(("fbx", binary_fbx));
            } else {
                eprintln!("skipping Blender FBX import parity: Blender is unavailable");
            }
        } else {
            blender_paths.push((format, output));
        }
    }

    if let Some(results) = blender_import(&blender_paths)? {
        for (format, _) in &blender_paths {
            let format = *format;
            let actual = &results[format];
            let actual_motion = DVec3::from_array([
                actual["animated_delta"][0]
                    .as_f64()
                    .ok_or("Blender animation delta missing")?,
                actual["animated_delta"][1]
                    .as_f64()
                    .ok_or("Blender animation delta missing")?,
                actual["animated_delta"][2]
                    .as_f64()
                    .ok_or("Blender animation delta missing")?,
            ]);
            let actual_start = DVec3::from_array([
                actual["start_position"][0]
                    .as_f64()
                    .ok_or("Blender start position missing")?,
                actual["start_position"][1]
                    .as_f64()
                    .ok_or("Blender start position missing")?,
                actual["start_position"][2]
                    .as_f64()
                    .ok_or("Blender start position missing")?,
            ]);
            let actual_end = DVec3::from_array([
                actual["finish_position"][0]
                    .as_f64()
                    .ok_or("Blender final position missing")?,
                actual["finish_position"][1]
                    .as_f64()
                    .ok_or("Blender final position missing")?,
                actual["finish_position"][2]
                    .as_f64()
                    .ok_or("Blender final position missing")?,
            ]);
            let blender_expected_start = if format == "alembic" {
                DVec3::new(expected_start.x, -expected_start.z, expected_start.y)
            } else {
                expected_start
            };
            let blender_expected_end = if format == "alembic" {
                DVec3::new(expected_end.x, -expected_end.z, expected_end.y)
            } else {
                expected_end
            };
            let blender_expected_motion = blender_expected_end - blender_expected_start;
            if matches!(format, "usda" | "alembic") {
                assert_vec_close(
                    actual_start,
                    blender_expected_start,
                    format,
                    "Blender frame 1",
                );
                assert_vec_close(actual_end, blender_expected_end, format, "Blender frame 3");
                assert_vec_close(
                    actual_motion,
                    blender_expected_motion,
                    format,
                    "Blender motion",
                );
            }
            if format != "alembic" {
                assert_eq!(
                    actual["bone_count"].as_u64(),
                    Some(2),
                    "{format}: Blender bone count"
                );
                assert!(
                    (actual["root_weight"]
                        .as_f64()
                        .ok_or_else(|| format!("{format}: Root weight missing; probe={actual}"))?
                        - 0.75)
                        .abs()
                        < 1.0e-5,
                    "{format}: Blender Root weight"
                );
                assert!(
                    (actual["child_weight"].as_f64().ok_or_else(|| format!(
                        "{format}: Child weight missing; probe={actual}"
                    ))? - 0.25)
                        .abs()
                        < 1.0e-5,
                    "{format}: Blender Child weight"
                );
                assert!(
                    (actual["shape_delta"].as_f64().ok_or_else(|| format!(
                        "{format}: Lift shape key missing; probe={actual}"
                    ))? - 0.5)
                        .abs()
                        < 1.0e-4,
                    "{format}: Blender Lift delta"
                );
            }
        }
    }
    Ok(())
}
