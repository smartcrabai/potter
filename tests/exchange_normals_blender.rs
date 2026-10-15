//! Blender 5.2.2 checks for exchange shading normals.
//!
//! Each test exports the shared fixture scene to one format, imports the file with Blender's own
//! importer, and compares Blender's world-space corner normals with the reference computed by
//! Blender for the same scene. PLY stores one normal per vertex, so its expectation is derived from
//! the reference: each corner takes the normalized sum of the reference corner normals at its
//! position.
//!
//! Alembic uses Blender's Y-up basis and reversed polygon winding, and Blender's importer undoes
//! both, so its import is checked against the same reference as the other formats.
//! FBX uses the binary variant because Blender 5.2.2 cannot import ASCII FBX.

use std::{collections::BTreeMap, env, fs, path::Path, process::Command};

use glam::DVec3;
use serde::Deserialize;
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "observed_corners and export_strict are not used by this test binary"
)]
mod exchange_normals;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use exchange_normals::{
    BLENDER_NORMAL_TOLERANCE, ObservedCorner, TestResult, export, near, pot, reference_corners,
    usd_world_corners, write_fixture_scene,
};

/// Imports the file given after `--` with the importer named by the first argument, then prints
/// every mesh object's world-space corner normals. Normals follow `mesh.corner_normals` mapped by
/// the inverse transpose of `matrix_world`, as in the reference.
const BLENDER_NORMALS_SCRIPT: &str = r#"
import bpy
import json
import sys
from mathutils import Vector

if bpy.app.version != (5, 2, 2):
    print("POTTER_BLENDER_VERSION_SKIP=" + str(bpy.app.version))
    sys.exit(0)

format_name, path = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
if format_name == "obj":
    bpy.ops.wm.obj_import(filepath=path)
elif format_name == "usda":
    bpy.ops.wm.usd_import(filepath=path)
elif format_name == "fbx":
    bpy.ops.import_scene.fbx(filepath=path)
elif format_name == "ply":
    bpy.ops.wm.ply_import(filepath=path)
elif format_name == "abc":
    bpy.ops.wm.alembic_import(filepath=path)
else:
    raise RuntimeError("unexpected format " + format_name)

def world_corners(obj):
    mesh = obj.data
    matrix = obj.matrix_world
    normal_matrix = matrix.to_3x3().inverted().transposed()
    corners = []
    for polygon in mesh.polygons:
        positions = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
        center = sum(positions, Vector()) / len(positions)
        for loop_index in polygon.loop_indices:
            position = matrix @ mesh.vertices[mesh.loops[loop_index].vertex_index].co
            normal = (normal_matrix @ Vector(mesh.corner_normals[loop_index].vector)).normalized()
            corners.append({
                "face_center": [float(center[axis]) for axis in range(3)],
                "corner": [float(position[axis]) for axis in range(3)],
                "normal": [float(normal[axis]) for axis in range(3)],
            })
    return corners

meshes = []
for obj in sorted((obj for obj in bpy.data.objects if obj.type == "MESH"), key=lambda obj: obj.name):
    meshes.append({
        "name": obj.name,
        "has_custom_normals": bool(obj.data.has_custom_normals),
        "corners": world_corners(obj),
    })
if not meshes:
    raise RuntimeError(format_name + " import contained no mesh")
print("POTTER_EXCHANGE_NORMALS=" + json.dumps({"meshes": meshes}, sort_keys=True))
"#;

#[derive(Deserialize)]
struct BlenderDump {
    meshes: Vec<BlenderMesh>,
}

#[derive(Deserialize)]
struct BlenderMesh {
    name: String,
    has_custom_normals: bool,
    corners: Vec<BlenderCorner>,
}

#[derive(Deserialize)]
struct BlenderCorner {
    face_center: [f64; 3],
    corner: [f64; 3],
    normal: [f64; 3],
}

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

/// Exports the fixture as `pot_format`, imports the file with Blender's `blender_format` importer,
/// and returns Blender's per-mesh corner normals. Returns `None` when the test should skip.
fn import_fixture(
    pot_format: &str,
    file_name: &str,
    blender_format: &str,
) -> TestResult<Option<Vec<BlenderMesh>>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping {pot_format} Blender normals: Blender is unavailable");
        return Ok(None);
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene.pot");
    write_fixture_scene(&scene)?;
    let file = export(&scene, pot_format, &directory.path().join(file_name))?;

    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python-expr",
            BLENDER_NORMALS_SCRIPT,
            "--",
        ])
        .arg(blender_format)
        .arg(&file);
    let output = process::run_guarded(command)?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "Blender {blender_format} import failed:\n{text}"
    );
    if text.contains("POTTER_BLENDER_VERSION_SKIP=") {
        assert!(
            !require_blender(),
            "POTTER_REQUIRE_BLENDER is set but Blender is not 5.2.2:\n{text}"
        );
        eprintln!("skipping {pot_format} Blender normals: this fixture requires Blender 5.2.2");
        return Ok(None);
    }
    let payload = text
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_EXCHANGE_NORMALS="))
        .ok_or("Blender did not report exchange normals")?;
    let dump: BlenderDump = serde_json::from_str(payload)?;
    Ok(Some(dump.meshes))
}

fn observed_corners_of(meshes: &[BlenderMesh]) -> Vec<ObservedCorner> {
    meshes
        .iter()
        .flat_map(|mesh| {
            mesh.corners.iter().map(|corner| ObservedCorner {
                face_center: corner.face_center,
                corner: corner.corner,
                normal: corner.normal,
            })
        })
        .collect()
}

fn assert_custom_normals(format: &str, meshes: &[BlenderMesh]) {
    for mesh in meshes {
        assert!(
            mesh.has_custom_normals,
            "{format}: Blender imported {} without custom normals",
            mesh.name
        );
    }
}

/// Expected corners for PLY: Blender's PLY import gives every corner the normal of its vertex.
/// Blender writes a vertex normal as normalize(N * sum of the object-local corner normals at that
/// vertex), with N the inverse transpose of the object matrix. Summing world corner normals instead
/// matches only orthogonal transforms, so the reference recovers local normals first.
fn ply_vertex_normal_reference() -> TestResult<Vec<ObservedCorner>> {
    let reference = reference_corners()?;
    // Fixture transforms (diagonal scale, no rotation): Mixed at the origin, Mirror at x = 5 with
    // scale (-1, 1, 1), and NonUni at y = 5 with scale (2, 1, 0.5).
    let scale_at = |corner: [f64; 3]| {
        if corner[1] >= 3.0 {
            DVec3::new(2.0, 1.0, 0.5)
        } else if corner[0] >= 3.0 {
            DVec3::new(-1.0, 1.0, 1.0)
        } else {
            DVec3::ONE
        }
    };
    Ok(reference
        .iter()
        .map(|corner| {
            let scale = scale_at(corner.corner);
            // World normal = normalize(M^-T n), so the local normal is proportional to M^T world.
            let local_sum = reference
                .iter()
                .filter(|other| near(other.corner, corner.corner))
                .fold(DVec3::ZERO, |sum, other| {
                    sum + (DVec3::from_array(other.normal) * scale).normalize()
                });
            ObservedCorner {
                face_center: corner.face_center,
                corner: corner.corner,
                normal: (local_sum / scale).normalize().to_array(),
            }
        })
        .collect())
}

/// Asserts each expected corner has exactly one observed corner at the same position and face
/// centre, with a normal within `tolerance`.
fn assert_matches_expected(
    format: &str,
    observed: &[ObservedCorner],
    expected: &[ObservedCorner],
    tolerance: f64,
) {
    assert_eq!(
        observed.len(),
        expected.len(),
        "{format}: corner count differs from the expected corners"
    );
    for want in expected {
        let found = observed
            .iter()
            .filter(|corner| {
                near(corner.corner, want.corner) && near(corner.face_center, want.face_center)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            found.len(),
            1,
            "{format}: expected one corner at {:?} in face centred {:?}",
            want.corner,
            want.face_center
        );
        let error = (0..3)
            .map(|axis| (found[0].normal[axis] - want.normal[axis]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error <= tolerance,
            "{format}: normal {:?} differs from expected {:?} at corner {:?} (error {error})",
            found[0].normal,
            want.normal,
            want.corner
        );
    }
}

fn assert_reference_normals(pot_format: &str, file_name: &str, blender_format: &str) -> TestResult {
    let Some(meshes) = import_fixture(pot_format, file_name, blender_format)? else {
        return Ok(());
    };
    assert_custom_normals(pot_format, &meshes);
    assert_matches_expected(
        pot_format,
        &observed_corners_of(&meshes),
        &reference_corners()?,
        BLENDER_NORMAL_TOLERANCE,
    );
    Ok(())
}

#[test]
fn obj_export_corner_normals_match_blender_import() -> TestResult {
    assert_reference_normals("obj", "fixture.obj", "obj")
}

#[test]
fn usda_export_corner_normals_match_blender_import() -> TestResult {
    assert_reference_normals("usda", "fixture.usda", "usda")
}

#[test]
fn fbx_export_corner_normals_match_blender_import() -> TestResult {
    assert_reference_normals("fbx-binary", "fixture.fbx", "fbx")
}

#[test]
fn ply_export_vertex_normals_match_blender_import() -> TestResult {
    let Some(meshes) = import_fixture("ply", "fixture.ply", "ply")? else {
        return Ok(());
    };
    assert_matches_expected(
        "ply",
        &observed_corners_of(&meshes),
        &ply_vertex_normal_reference()?,
        BLENDER_NORMAL_TOLERANCE,
    );
    Ok(())
}

#[test]
fn alembic_export_corner_normals_match_blender_import() -> TestResult {
    assert_reference_normals("alembic", "fixture.abc", "abc")
}

/// Imports an Alembic file with Blender's importer and prints every object's parent, scale, and
/// world translation, so placement can be compared with the scene.
const BLENDER_PLACEMENT_SCRIPT: &str = r#"
import bpy
import json
import sys

if bpy.app.version != (5, 2, 2):
    print("POTTER_BLENDER_VERSION_SKIP=" + str(bpy.app.version))
    sys.exit(0)

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.alembic_import(filepath=path)
objects = {}
for obj in bpy.data.objects:
    objects[obj.name] = {
        "parent": obj.parent.name if obj.parent else None,
        "scale": [float(value) for value in obj.scale],
        "world_translation": [float(value) for value in obj.matrix_world.translation],
    }
print("POTTER_BLENDER_PLACEMENT=" + json.dumps(objects, sort_keys=True))
"#;

/// Imports an Alembic file with Blender's importer, steps through the frames listed after `--`, and
/// prints each mesh's world-space corner normals at every frame.
const BLENDER_FRAMES_SCRIPT: &str = r#"
import bpy
import json
import sys
from mathutils import Vector

if bpy.app.version != (5, 2, 2):
    print("POTTER_BLENDER_VERSION_SKIP=" + str(bpy.app.version))
    sys.exit(0)

path, frame_list = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.alembic_import(filepath=path)
scene = bpy.context.scene
frames = {}
for frame in [int(value) for value in frame_list.split(",")]:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = []
    for obj in sorted((obj for obj in bpy.data.objects if obj.type == "MESH"), key=lambda obj: obj.name):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()
        corners = []
        for polygon in mesh.polygons:
            positions = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
            center = sum(positions, Vector()) / len(positions)
            for loop_index in polygon.loop_indices:
                position = matrix @ mesh.vertices[mesh.loops[loop_index].vertex_index].co
                normal = (normal_matrix @ Vector(mesh.corner_normals[loop_index].vector)).normalized()
                corners.append({
                    "face_center": [float(center[axis]) for axis in range(3)],
                    "corner": [float(position[axis]) for axis in range(3)],
                    "normal": [float(normal[axis]) for axis in range(3)],
                })
        meshes.append({"name": obj.name, "corners": corners})
        evaluated.to_mesh_clear()
    frames[str(frame)] = meshes
print("POTTER_BLENDER_FRAMES=" + json.dumps(frames, sort_keys=True))
"#;

#[derive(Deserialize)]
struct BlenderFrameMesh {
    corners: Vec<BlenderCorner>,
}

fn run_pot(command: &mut Command) -> TestResult {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "pot command failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

/// Creates a project at `scene` containing `operations`.
fn build_scene(scene: &Path, operations: &Value) -> TestResult {
    run_pot(pot().arg("init").arg(scene).arg("--json"))?;
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    run_pot(pot().arg("apply").arg(scene).arg("--file").arg(&batch_file))
}

/// A grid with a wave modifier and translation keys at frames 1 and 3 holding the transform at
/// identity, so the geometry changes per frame while the object stays at the origin.
fn wave_operations() -> Value {
    json!([
        {"op":"node.create","id":"wave","kind":"grid","params":{"size":2.0,"x_subdivisions":4,"y_subdivisions":4}},
        {"op":"modifier.create","target":{"id":"wave"},"id":"ripple","type":"wave","params":{"speed":0.25,"height":0.5,"width":1.5,"narrowness":1.5}},
        {"op":"action.create","id":"samples","name":"Samples"},
        {"op":"node.update","target":{"id":"wave"},"set":{"action":"samples"}},
        {"op":"keyframe.insert","target":{"id":"wave"},"path":"transform.translation","index":0,"frame":1.0,"value":0.0},
        {"op":"keyframe.insert","target":{"id":"wave"},"path":"transform.translation","index":0,"frame":3.0,"value":0.0}
    ])
}

/// Exports the scene's usda at `frame`, whose corner normals are Potter's for that frame.
fn export_usda_at_frame(scene: &Path, frame: u32, out: &Path) -> TestResult {
    run_pot(
        pot()
            .arg("export")
            .arg(scene)
            .args(["--format", "usda", "--frame"])
            .arg(frame.to_string())
            .arg("--out")
            .arg(out)
            .args(["--overwrite", "--allow-lossy", "--json"]),
    )
}

/// Runs Blender with `script` and `args` after `--`, and returns the payload after `marker`.
/// Returns `None` when Blender is not 5.2.2 and the test should skip.
fn run_blender_script(
    blender: &Path,
    script: &str,
    args: &[String],
    marker: &str,
) -> TestResult<Option<String>> {
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python-expr",
            script,
            "--",
        ])
        .args(args);
    let output = process::run_guarded(command)?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "Blender import failed:\n{text}");
    if text.contains("POTTER_BLENDER_VERSION_SKIP=") {
        assert!(
            !require_blender(),
            "POTTER_REQUIRE_BLENDER is set but Blender is not 5.2.2:\n{text}"
        );
        eprintln!("skipping Alembic Blender check: this check requires Blender 5.2.2");
        return Ok(None);
    }
    let payload = text
        .lines()
        .find_map(|line| line.strip_prefix(marker))
        .ok_or("Blender did not report the expected payload")?;
    Ok(Some(payload.to_owned()))
}

/// A zero-scale parent with a child, and a zero-scale root, must place in Blender with the same
/// scale and world position.
#[test]
fn alembic_zero_scale_nodes_place_in_blender() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Alembic zero-scale placement: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene.pot");
    build_scene(
        &scene,
        &json!([
            {"op":"node.create","id":"flat","kind":"box","params":{"size":2.0},"transform":{"translation":[0.0,0.0,0.0],"scale":[0.0,1.0,1.0]}},
            {"op":"node.create","id":"flat_child","kind":"box","params":{"size":2.0},"transform":{"translation":[0.0,3.0,0.0],"scale":[1.0,1.0,1.0]}},
            {"op":"node.parent","target":{"id":"flat_child"},"parent":"flat","keep_world":false},
            {"op":"node.create","id":"flat_root","kind":"box","params":{"size":2.0},"transform":{"translation":[5.0,0.0,0.0],"scale":[0.0,0.0,0.0]}}
        ]),
    )?;
    let file = export(&scene, "alembic", &directory.path().join("zero.abc"))?;
    let Some(payload) = run_blender_script(
        &blender,
        BLENDER_PLACEMENT_SCRIPT,
        &[file.display().to_string()],
        "POTTER_BLENDER_PLACEMENT=",
    )?
    else {
        return Ok(());
    };
    let objects: Value = serde_json::from_str(&payload)?;
    assert_eq!(objects["flat"]["scale"], json!([0.0, 1.0, 1.0]));
    assert_eq!(objects["flat"]["parent"], Value::Null);
    assert_eq!(objects["flat_child"]["parent"], json!("flat"));
    assert_eq!(objects["flat_root"]["scale"], json!([0.0, 0.0, 0.0]));
    let child: [f64; 3] =
        serde_json::from_value(objects["flat_child"]["world_translation"].clone())?;
    for (axis, expected) in [0.0, 3.0, 0.0].into_iter().enumerate() {
        assert!(
            (child[axis] - expected).abs() <= 1.0e-4,
            "flat_child world translation {child:?} differs from [0, 3, 0]"
        );
    }
    Ok(())
}

/// Blender's mesh-sequence cache at frames 1 and 3 must give the normals of Potter's per-frame
/// meshes, which change between the frames.
#[test]
fn alembic_animated_frames_match_blender_import() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Alembic animated normals: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene.pot");
    build_scene(&scene, &wave_operations())?;
    let file = export(&scene, "alembic", &directory.path().join("animated.abc"))?;
    let Some(payload) = run_blender_script(
        &blender,
        BLENDER_FRAMES_SCRIPT,
        &[file.display().to_string(), "1,3".to_owned()],
        "POTTER_BLENDER_FRAMES=",
    )?
    else {
        return Ok(());
    };
    let frames: BTreeMap<String, Vec<BlenderFrameMesh>> = serde_json::from_str(&payload)?;
    for frame in [1_u32, 3] {
        let usda = directory.path().join(format!("frame-{frame}.usda"));
        export_usda_at_frame(&scene, frame, &usda)?;
        let expected = usd_world_corners(&fs::read_to_string(&usda)?)?;
        let observed = frames
            .get(&frame.to_string())
            .ok_or("Blender reported no frame")?
            .iter()
            .flat_map(|mesh| {
                mesh.corners.iter().map(|corner| ObservedCorner {
                    face_center: corner.face_center,
                    corner: corner.corner,
                    normal: corner.normal,
                })
            })
            .collect::<Vec<_>>();
        assert_matches_expected(
            &format!("alembic frame {frame}"),
            &observed,
            &expected,
            BLENDER_NORMAL_TOLERANCE,
        );
    }
    Ok(())
}
