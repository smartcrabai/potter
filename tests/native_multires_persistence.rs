//! Blender 5.2.2 multires persistence through Potter's native MDisps bridge.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared Blender oracle exposes helpers used by the normal persistence binary"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::fs;

use native_oracle::{
    TestResult, blender, export, import_blend, read_scene, run_blender_arguments, write_script,
};
use serde_json::Value;
use tempfile::tempdir;

const MULTIRES_MASK_SOURCE: &str = r#"
import bpy
import sys

output = sys.argv[sys.argv.index("--") + 1]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
mesh = obj.data
modifier = obj.modifiers.new("Multires", "MULTIRES")
for _ in range(2):
    bpy.ops.object.multires_subdivide(modifier=modifier.name, mode="CATMULL_CLARK")
modifier.levels = modifier.total_levels
modifier.sculpt_levels = modifier.total_levels
modifier.render_levels = modifier.total_levels
mask = mesh.attributes.get(".sculpt_mask")
if mask is None:
    mask = mesh.attributes.new(name=".sculpt_mask", type="FLOAT", domain="POINT")
mask.data[0].value = 0.75
hidden = mesh.attributes.get(".hide_poly")
if hidden is None:
    hidden = mesh.attributes.new(name=".hide_poly", type="BOOLEAN", domain="FACE")
hidden.data[0].value = True
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
# Headless Blender aborts in the sculpt undo push unless an undo step exists first.
bpy.ops.ed.undo_push(message="multires mask fixture")
bpy.ops.object.mode_set(mode="SCULPT")
bpy.ops.paint.mask_flood_fill(mode="VALUE", value=0.25)
bpy.ops.object.mode_set(mode="OBJECT")
mask = mesh.attributes[".sculpt_mask"]
mask.data[0].value = 0.75
bpy.ops.wm.save_as_mainfile(filepath=output)
"#;

fn multires_record(scene: &Value) -> TestResult<&Value> {
    scene["data_blocks"]
        .as_object()
        .ok_or("scene data_blocks are invalid")?
        .values()
        .find_map(|data| {
            data.get("mesh")?
                .get("attributes")?
                .get("multires_displacement")
                .filter(|record| record.is_object())
        })
        .ok_or_else(|| "scene has no multires_displacement record".into())
}

fn hidden_face_attribute(scene: &Value) -> TestResult<&Value> {
    let mesh = scene["data_blocks"]
        .as_object()
        .ok_or("scene data_blocks are invalid")?
        .values()
        .find_map(|data| data["mesh"].as_object())
        .ok_or("scene has no mesh")?;
    mesh["attributes"]["blender_attributes"]
        .as_array()
        .ok_or("mesh has no retained Blender attributes")?
        .iter()
        .find(|attribute| attribute["name"].as_str() == Some(".hide_poly"))
        .ok_or_else(|| "mesh has no retained face-hide attribute".into())
}

fn check_mask_and_face_hide(scene: &Value, expected_mask: &Value) -> TestResult {
    let record = multires_record(scene)?;
    assert_eq!(&record["paint_mask"], expected_mask);
    assert!(
        record["paint_mask"]
            .as_array()
            .ok_or("paint_mask is not a per-loop array")?
            .iter()
            .any(|grid| grid["values"][4].as_f64() == Some(0.25)),
        "the interior GridPaintMask sample was not retained"
    );
    let mesh = scene["data_blocks"]
        .as_object()
        .ok_or("scene data_blocks are invalid")?
        .values()
        .find_map(|data| data["mesh"].as_object())
        .ok_or("scene has no mesh")?;
    assert!(
        mesh["attributes"]["blender_attributes"]
            .as_array()
            .ok_or("mesh has no retained Blender attributes")?
            .iter()
            .find(|attribute| attribute["name"].as_str() == Some(".sculpt_mask"))
            .and_then(|attribute| attribute["values"].as_array())
            .is_some_and(|values| values.iter().any(|value| value.as_f64() == Some(0.75))),
        "the asymmetric base control mask was not retained"
    );

    assert!(
        hidden_face_attribute(scene)?["values"]
            .as_array()
            .ok_or("face-hide values are not an array")?
            .iter()
            .any(|value| value.as_bool() == Some(true)),
        "the hidden face state was not retained"
    );
    Ok(())
}

#[test]
fn multires_grid_mask_and_face_hide_roundtrip_through_blend() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let source = directory.path().join("multires-mask.blend");
    let script = write_script(
        directory.path(),
        "make_multires_mask.py",
        MULTIRES_MASK_SOURCE,
    )?;
    run_blender_arguments(&blender, &script, None, &[source.as_path()])?;

    let (project, _) = import_blend(&blender, directory.path(), "first-import", &source)?;
    let first = read_scene(&project)?;
    let expected_mask = multires_record(&first)?["paint_mask"].clone();
    check_mask_and_face_hide(&first, &expected_mask)?;

    let exported = directory.path().join("multires-mask-roundtrip.blend");
    export(&blender, &project, "blend", &exported)?;
    let (roundtrip_project, _) =
        import_blend(&blender, directory.path(), "second-import", &exported)?;
    let roundtrip = read_scene(&roundtrip_project)?;
    check_mask_and_face_hide(&roundtrip, &expected_mask)?;
    assert_eq!(
        multires_record(&roundtrip)?["hidden"],
        multires_record(&first)?["hidden"],
        "the per-grid hidden bitmap changed during the round trip",
    );
    assert!(exported.is_file());
    fs::remove_file(exported)?;
    Ok(())
}
