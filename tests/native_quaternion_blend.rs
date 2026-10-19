//! A generic `quaternion` attribute keeps its rotations through the `.blend` bridge.
//!
//! Potter stores quaternions as `[x, y, z, w]` (`docs/spec.md`), while Blender stores `w, x, y, z`.
//! The test sets two non-identity rotations in Potter, checks that Blender's dump of the exported
//! `.blend` holds them as `w, x, y, z`, then reimports that file and checks that Potter's native
//! store holds the original `[x, y, z, w]` values again.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use native_oracle::{
    TestResult, apply_operations, attribute, dump_blend, dumped_object, export, floats,
    import_blend, mesh_of_node, pot_json, read_scene,
};
use serde_json::json;
use tempfile::tempdir;

/// Potter's native `[x, y, z, w]` for a rotation of `angle` radians about `axis`.
fn native_rotation(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let length = axis
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let half = angle / 2.0;
    let scale = half.sin() / length;
    [
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        half.cos(),
    ]
}

fn same_quaternion(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-6)
}

#[test]
fn quaternion_attribute_round_trips_through_blend_in_potter_order() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let project = root.path().join("project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    // The second rotation's angle exceeds pi, so its w is negative: the sign must survive too.
    let first = native_rotation([1.0, 2.0, 3.0], 1.2);
    let second = native_rotation([0.0, 0.0, 1.0], 4.0);
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "node.create", "id": "box", "kind": "box", "params": {}},
            {"op": "mesh.attribute_create", "target": {"id": "box"}, "name": "Rotation", "domain": "point", "type": "quaternion"},
            {"op": "mesh.attribute_update", "target": {"id": "box"}, "elements": {"domain": "vertex", "ids": ["v0"]}, "name": "Rotation", "value": first},
            {"op": "mesh.attribute_update", "target": {"id": "box"}, "elements": {"domain": "vertex", "ids": ["v1"]}, "name": "Rotation", "value": second}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    // Blender's dump of the exported file: the same rotations, in Blender's w, x, y, z order.
    let scene = read_scene(&project)?;
    let mesh = mesh_of_node(&scene, "box")?;
    let storage_index = |id: u64| -> TestResult<usize> {
        mesh["vertices"]
            .as_array()
            .ok_or("mesh vertices are missing")?
            .iter()
            .position(|vertex| vertex["id"].as_u64() == Some(id))
            .ok_or_else(|| format!("vertex {id} is missing").into())
    };
    let exported = root.path().join("rotation.blend");
    export(&blender, &project, "blend", &exported)?;
    let dump = dump_blend(&blender, root.path(), &exported, "exported")?;
    let object = dumped_object(&dump, "box")?;
    let blender_attribute = object["attributes"]
        .as_array()
        .ok_or("Blender attributes are missing")?
        .iter()
        .find(|attribute| attribute["name"].as_str() == Some("Rotation"))
        .ok_or("Blender has no Rotation attribute")?;
    assert_eq!(blender_attribute["data_type"], "QUATERNION");
    assert_eq!(blender_attribute["domain"], "POINT");
    let blender_values = blender_attribute["values"]
        .as_array()
        .ok_or("Blender Rotation values are missing")?;
    let expected_blender = |native: [f64; 4]| [native[3], native[0], native[1], native[2]];
    let first_index = storage_index(0)?;
    let second_index = storage_index(1)?;
    assert!(
        same_quaternion(
            &floats(&blender_values[first_index])?,
            &expected_blender(first)
        ),
        "Blender holds {:?} for v0, expected w, x, y, z {:?}",
        blender_values[first_index],
        expected_blender(first)
    );
    assert!(
        same_quaternion(
            &floats(&blender_values[second_index])?,
            &expected_blender(second)
        ),
        "Blender holds {:?} for v1, expected w, x, y, z {:?}",
        blender_values[second_index],
        expected_blender(second)
    );
    // Vertex 2 has no stored rotation, so the bridge writes Potter's identity default, which Blender reads as
    // (1, 0, 0, 0) in w, x, y, z order.
    let unassigned_index = storage_index(2)?;
    assert!(
        same_quaternion(
            &floats(&blender_values[unassigned_index])?,
            &[1.0, 0.0, 0.0, 0.0]
        ),
        "Blender holds {:?} for the unassigned v2, expected the identity (1, 0, 0, 0)",
        blender_values[unassigned_index]
    );

    // Reimport the saved file: Potter's native store must hold the original [x, y, z, w] values.
    let (reimported, _) = import_blend(&blender, root.path(), "reimported", &exported)?;
    let reimported_scene = read_scene(&reimported)?;
    let reimported_mesh = mesh_of_node(&reimported_scene, "box")?;
    let rotation = attribute(reimported_mesh, "Rotation")?;
    assert_eq!(rotation["type"], "quaternion", "{rotation}");
    let stored: Vec<Vec<f64>> = rotation["values"]
        .as_object()
        .ok_or("reimported Rotation values are missing")?
        .values()
        .map(floats)
        .collect::<TestResult<_>>()?;
    assert!(
        stored.iter().any(|value| same_quaternion(value, &first)),
        "reimported Potter store lacks the first rotation {first:?}: {stored:?}"
    );
    assert!(
        stored.iter().any(|value| same_quaternion(value, &second)),
        "reimported Potter store lacks the second rotation {second:?}: {stored:?}"
    );
    Ok(())
}
