use std::{fs, path::Path};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, export_raw, import_blend_allow_lossy, node_id, pot, pot_json, read_scene,
    run_blender_arguments, write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const BLENDER_FIXTURE: &str = r#"
import bpy
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"Cycles settings fixture requires Blender 5.2.2, found {bpy.app.version_string}")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"

world = bpy.data.worlds.new("Cycles Settings World")
world.color = (0.2, 0.3, 0.4)
scene.world = world
world.cycles.sampling_method = "MANUAL"
world.cycles.sample_map_resolution = 4096
world.cycles.max_bounces = 37
world.cycles.use_shadows = False
world.cycles.is_caustics_light = True
world.cycles_visibility.camera = False
world.cycles_visibility.diffuse = True
world.cycles_visibility.glossy = False
world.cycles_visibility.transmission = True
world.cycles_visibility.scatter = False
world.cycles.volume_sampling = "DISTANCE"
world.cycles.volume_interpolation = "CUBIC"
world.cycles.volume_step_size = 0.125

material = bpy.data.materials.new("Cycles Settings Material")
material.cycles.volume_sampling = "DISTANCE"
material.cycles.volume_interpolation = "CUBIC"
material.cycles.emission_sampling = "FRONT_BACK"
material.cycles.volume_step_rate = 2.5
bpy.ops.mesh.primitive_cube_add()
bpy.context.object.name = "Cycles Settings Material Holder"
bpy.context.object.data.materials.append(material)

light = bpy.data.lights.new("Cycles Settings Area", type="AREA")
light.cycles.is_portal = True
light.cycles.max_bounces = 31
light.cycles.is_caustics_light = True
light_object = bpy.data.objects.new("Cycles Settings Area", light)
scene.collection.objects.link(light_object)

bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
"#;

const BLENDER_READBACK: &str = r#"
import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"Cycles settings readback requires Blender 5.2.2, found {bpy.app.version_string}")
world = bpy.data.worlds.get("Cycles Settings World")
assert world is not None
assert world.cycles.sampling_method == "MANUAL"
assert world.cycles.sample_map_resolution == 4096
assert world.cycles.max_bounces == 37
assert world.cycles.use_shadows is False
assert world.cycles.is_caustics_light is True
assert {
    "camera": world.cycles_visibility.camera,
    "diffuse": world.cycles_visibility.diffuse,
    "glossy": world.cycles_visibility.glossy,
    "transmission": world.cycles_visibility.transmission,
    "scatter": world.cycles_visibility.scatter,
} == {
    "camera": False,
    "diffuse": True,
    "glossy": False,
    "transmission": True,
    "scatter": False,
}
assert world.cycles.volume_sampling == "DISTANCE"
assert world.cycles.volume_interpolation == "CUBIC"
assert world.cycles.volume_step_size == 0.125

material = bpy.data.materials.get("Cycles Settings Material")
assert material is not None
assert material.cycles.volume_sampling == "DISTANCE"
assert material.cycles.volume_interpolation == "CUBIC"
assert material.cycles.emission_sampling == "FRONT_BACK"
assert material.cycles.volume_step_rate == 2.5

light = bpy.data.lights.get("Cycles Settings Area")
assert light is not None and light.type == "AREA"
assert light.cycles.is_portal is True
assert light.cycles.max_bounces == 31
assert light.cycles.is_caustics_light is True
"#;

fn entity_id<'a>(scene: &'a Value, registry: &str, name: &str) -> TestResult<&'a str> {
    let entities = scene[registry]
        .as_object()
        .ok_or_else(|| format!("scene {registry} registry is missing"))?;
    if let Some((id, _)) = entities
        .iter()
        .find(|(_, entity)| entity["name"].as_str() == Some(name))
    {
        return Ok(id);
    }
    let available: Vec<_> = entities
        .values()
        .filter_map(|entity| entity["name"].as_str())
        .collect();
    Err(format!("{registry} entity {name:?} is missing; available: {available:?}").into())
}

fn inspect_id(project: &Path, id: &str) -> TestResult<Value> {
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(project).arg("--id").arg(id);
    })?;
    Ok(inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.first())
        .ok_or("inspect returned no selected item")?
        .clone())
}

fn assert_update_rejected(
    project: &Path,
    root: &Path,
    index: usize,
    operation: &Value,
    field: &str,
) -> TestResult {
    let revision = read_scene(project)?["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    let batch = root.join(format!("invalid-cycles-update-{index}.json"));
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": [operation],
        }))?,
    )?;
    let output = pot()
        .args(["apply"])
        .arg(project)
        .args(["--file"])
        .arg(batch)
        .arg("--json")
        .output()?;
    assert!(
        !output.status.success(),
        "invalid update for {field} unexpectedly succeeded"
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["ok"], false, "{field}: {report}");
    assert_eq!(
        report["error"]["code"], "INVALID_OPERATION",
        "{field}: {report}"
    );
    assert!(
        report.to_string().contains(field),
        "rejection did not identify {field}: {report}"
    );
    Ok(())
}

#[test]
fn cycles_world_material_and_light_settings_round_trip_through_blend() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Cycles settings Blender roundtrip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let fixture = write_script(root, "make_cycles_settings.py", BLENDER_FIXTURE)?;
    let source = root.join("cycles-settings-source.blend");
    run_blender_arguments(&blender, &fixture, None, &[source.as_path()])?;

    let (project, imported) = import_blend_allow_lossy(&blender, root, "project", &source)?;
    let import_losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import losses are missing")?;
    assert_eq!(
        import_losses.len(),
        2,
        "unexpected import losses: {imported}"
    );
    for feature in [
        "blender.world.cycles.is_caustics_light",
        "blender.light.cycles.is_caustics_light",
    ] {
        assert!(
            import_losses
                .iter()
                .any(|loss| loss["feature_id"] == feature),
            "missing import loss {feature}: {imported}"
        );
    }
    for feature in [
        "blender.world.cycles.max_bounces",
        "blender.world.cycles.use_shadows",
        "blender.world.cycles_visibility.camera",
        "blender.world.cycles_visibility.glossy",
        "blender.world.cycles_visibility.scatter",
        "blender.light.cycles.max_bounces",
        "blender.light.cycles.is_portal",
        "blender.world.cycles.sampling_method",
        "blender.world.cycles.sample_map_resolution",
        "blender.world.cycles.volume_sampling",
        "blender.world.cycles.volume_interpolation",
        "blender.world.cycles.volume_step_size",
        "blender.material.cycles.emission_sampling",
        "blender.material.cycles.volume_sampling",
        "blender.material.cycles.volume_interpolation",
        "blender.material.cycles.volume_step_rate",
        "blender.cycles.world.settings",
        "blender.cycles.material.settings",
        "blender.cycles.light.settings",
    ] {
        assert!(
            !import_losses
                .iter()
                .any(|loss| loss["feature_id"] == feature),
            "unexpected import loss {feature}: {imported}"
        );
    }
    let scene = read_scene(&project)?;
    let active_scene_id = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let world_id = scene["scenes"][active_scene_id]["world"]
        .as_str()
        .ok_or("active scene has no world")?;
    let material_id = entity_id(&scene, "materials", "Cycles Settings Material")?;
    let world = &scene["worlds"][world_id];
    assert_eq!(world["sampling_method"], "manual");
    assert_eq!(world["sample_map_resolution"], 4096);
    assert_eq!(world["max_bounces"], 37);
    assert_eq!(world["use_shadows"], false);
    assert_eq!(world["is_caustics_light"], true);
    assert_eq!(
        world["visibility"],
        json!({"camera": false, "glossy": false, "scatter": false})
    );
    assert!(world["visibility"]["diffuse"].as_bool().unwrap_or(true));
    assert!(
        world["visibility"]["transmission"]
            .as_bool()
            .unwrap_or(true)
    );
    assert_eq!(world["volume_sampling"], "distance");
    assert_eq!(world["volume_interpolation"], "cubic");
    assert_eq!(world["volume_step_size"], 0.125);

    let material = &scene["materials"][material_id];
    assert_eq!(material["volume_sampling"], "distance");
    assert_eq!(material["volume_interpolation"], "cubic");
    assert_eq!(material["emission_sampling"], "front_back");
    assert_eq!(material["volume_step_rate"], 2.5);

    let light_node_id = node_id(&scene, "Cycles Settings Area")?;
    let light_data_id = scene["nodes"][&light_node_id]["data"]
        .as_str()
        .ok_or("Cycles area light has no data block")?;
    let light = &scene["data_blocks"][light_data_id]["light"];
    assert_eq!(light["is_portal"], true);
    assert_eq!(light["max_bounces"], 31);
    assert_eq!(light["is_caustics_light"], true);

    let inspected_world = inspect_id(&project, world_id)?;
    assert_eq!(inspected_world["type"], "world");
    assert_eq!(inspected_world["id"], world_id);
    assert_eq!(inspected_world["color"], world["color"]);
    assert_eq!(inspected_world["strength"], world["strength"]);
    assert_eq!(
        inspected_world["visibility"],
        json!({
            "camera": false,
            "diffuse": true,
            "glossy": false,
            "transmission": true,
            "scatter": false
        })
    );
    for field in [
        "sampling_method",
        "sample_map_resolution",
        "max_bounces",
        "use_shadows",
        "is_caustics_light",
        "volume_sampling",
        "volume_interpolation",
        "volume_step_size",
    ] {
        assert_eq!(inspected_world[field], world[field], "World.{field}");
    }
    let inspected_material = inspect_id(&project, material_id)?;
    assert_eq!(inspected_material["type"], "material");
    assert_eq!(inspected_material["id"], material_id);
    assert_eq!(inspected_material["base_color"], material["base_color"]);
    assert_eq!(inspected_material["roughness"], material["roughness"]);
    for field in [
        "volume_sampling",
        "volume_interpolation",
        "emission_sampling",
        "volume_step_rate",
    ] {
        assert_eq!(
            inspected_material[field], material[field],
            "Material.{field}"
        );
    }
    let inspected_light = inspect_id(&project, &light_node_id)?;
    assert_eq!(inspected_light["light"]["is_portal"], true);
    assert_eq!(inspected_light["light"]["max_bounces"], 31);
    assert_eq!(inspected_light["light"]["is_caustics_light"], true);

    let invalid_updates = [
        (
            "sampling_method",
            json!({"op":"world.update","target":{"id":world_id},"set":{"sampling_method":"unknown"}}),
        ),
        (
            "volume_sampling",
            json!({"op":"world.update","target":{"id":world_id},"set":{"volume_sampling":"unknown"}}),
        ),
        (
            "volume_interpolation",
            json!({"op":"world.update","target":{"id":world_id},"set":{"volume_interpolation":"unknown"}}),
        ),
        (
            "emission_sampling",
            json!({"op":"material.update","target":{"id":material_id},"set":{"emission_sampling":"unknown"}}),
        ),
        (
            "volume_sampling",
            json!({"op":"material.update","target":{"id":material_id},"set":{"volume_sampling":"unknown"}}),
        ),
        (
            "volume_interpolation",
            json!({"op":"material.update","target":{"id":material_id},"set":{"volume_interpolation":"unknown"}}),
        ),
        (
            "sample_map_resolution",
            json!({"op":"world.update","target":{"id":world_id},"set":{"sample_map_resolution":3}}),
        ),
        (
            "volume_step_size",
            json!({"op":"world.update","target":{"id":world_id},"set":{"volume_step_size":0.0}}),
        ),
        (
            "volume_step_rate",
            json!({"op":"material.update","target":{"id":material_id},"set":{"volume_step_rate":0.0001}}),
        ),
        (
            "max_bounces",
            json!({"op":"world.update","target":{"id":world_id},"set":{"max_bounces":1025}}),
        ),
        (
            "max_bounces",
            json!({"op":"light.update","target":{"id":light_node_id},"set":{"max_bounces":1025}}),
        ),
    ];
    for (index, (field, operation)) in invalid_updates.into_iter().enumerate() {
        assert_update_rejected(&project, root, index, &operation, field)?;
    }
    assert_eq!(read_scene(&project)?["revision"], scene["revision"]);

    let exported = root.join("cycles-settings-roundtrip.blend");
    let export_output = export_raw(&blender, &project, "blend", &exported, true)?;
    assert!(
        export_output.status.success(),
        "lossy export failed: {}",
        String::from_utf8_lossy(&export_output.stdout)
    );
    let report: Value = serde_json::from_slice(&export_output.stdout)?;
    let export_losses = report["result"]["losses"]
        .as_array()
        .ok_or("export losses are missing")?;
    assert_eq!(export_losses.len(), 2, "unexpected export losses: {report}");
    for feature in [
        "blender.world.cycles.is_caustics_light",
        "blender.light.cycles.is_caustics_light",
    ] {
        assert!(
            export_losses
                .iter()
                .any(|loss| loss["feature_id"] == feature),
            "missing export loss {feature}: {report}"
        );
    }
    let readback = write_script(root, "read_cycles_settings.py", BLENDER_READBACK)?;
    run_blender_arguments(&blender, &readback, Some(&exported), &[])?;
    Ok(())
}
