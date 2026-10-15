//! Blender 5.2.2 checks for exchange shading normals.
//!
//! Each test exports the shared fixture scene to one format, imports the file with Blender's own
//! importer, and compares Blender's world-space corner normals with the reference computed by
//! Blender for the same scene. PLY stores one normal per vertex, so its expectation is derived from
//! the reference: each corner takes the normalized sum of the reference corner normals at its
//! position.
//!
//! Alembic is excluded: Blender 5.2.2 imports Potter's Alembic with reversed polygon winding and a
//! Y-up basis on top-level Xforms. That is a pre-existing convention, not a normals issue.
//! FBX uses the binary variant because Blender 5.2.2 cannot import ASCII FBX.

use std::{env, process::Command};

use glam::DVec3;
use serde::Deserialize;
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
use exchange_normals::{ObservedCorner, export, reference_corners, write_fixture_scene};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Tolerance for corner normals after a Blender import. Blender stores custom normals as int16,
/// quantized to about 3e-5 per component (observed up to 9e-5), so this is looser than the
/// `NORMAL_TOLERANCE` used for Potter-side decoding.
const BLENDER_NORMAL_TOLERANCE: f64 = 2.0e-4;

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

fn same_point(left: [f64; 3], right: [f64; 3]) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= 1.0e-4)
}

/// Expected corners for PLY: Blender's PLY import gives every corner the normal of its vertex, and
/// a vertex normal is the normalized sum of the reference corner normals at that position.
fn ply_vertex_normal_reference() -> TestResult<Vec<ObservedCorner>> {
    let reference = reference_corners()?;
    Ok(reference
        .iter()
        .map(|corner| {
            let sum = reference
                .iter()
                .filter(|other| same_point(other.corner, corner.corner))
                .fold(DVec3::ZERO, |sum, other| {
                    sum + DVec3::from_array(other.normal)
                });
            ObservedCorner {
                face_center: corner.face_center,
                corner: corner.corner,
                normal: sum.normalize().to_array(),
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
                same_point(corner.corner, want.corner)
                    && same_point(corner.face_center, want.face_center)
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
