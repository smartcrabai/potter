use std::{fs, path::Path};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, export_raw, import_blend, import_blend_allow_lossy, node_id, pot, pot_json,
    read_scene, run_blender_arguments, write_script,
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
scene.cycles.use_adaptive_sampling = False
scene.cycles.adaptive_threshold = 0.23
scene.cycles.adaptive_min_samples = 19
scene.cycles.tile_size = 3072
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.cycles.film_exposure = 1.75
scene.cycles.film_transparent_glass = True
scene.cycles.film_transparent_roughness = 0.37
scene.cycles.use_fast_gi = True
view_layer = bpy.context.view_layer
view_layer.use_solid = False
view_layer.use_strand = False
view_layer.use_volumes = False
view_layer.use_motion_blur = False
view_layer.cycles.use_denoising = False

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
view_layer.material_override = material
view_layer.world_override = world
bpy.ops.mesh.primitive_cube_add()
holder = bpy.context.object
holder.name = "Cycles Settings Material Holder"
holder.data.materials.append(material)
holder.is_holdout = True
holder.shadow_terminator_normal_offset = 0.42
holder.shadow_terminator_geometry_offset = 0.71
holder.shadow_terminator_shading_offset = 0.53
holder["_potter.internal"] = True
holder["visible_prop"] = "kept"
eevee_scene = bpy.data.scenes.new("EEVEE Terminator Scene")
eevee_scene.render.engine = "BLENDER_EEVEE"
terminator_mesh = bpy.data.meshes.new("EEVEE Terminator Mesh")
terminator_mesh.from_pydata(
    [(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
terminator_object = bpy.data.objects.new("EEVEE Terminator", terminator_mesh)
eevee_scene.collection.objects.link(terminator_object)
terminator_object.shadow_terminator_normal_offset = 0.25
camera_data = bpy.data.cameras.new("EEVEE Terminator Camera")
camera = bpy.data.objects.new("EEVEE Terminator Camera", camera_data)
eevee_scene.collection.objects.link(camera)
eevee_scene.camera = camera
light_data = bpy.data.lights.new("EEVEE Terminator Light", type="POINT")
light = bpy.data.objects.new("EEVEE Terminator Light", light_data)
eevee_scene.collection.objects.link(light)

light = bpy.data.lights.new("Cycles Settings Area", type="AREA")
light.cycles.is_portal = True
light.cycles.max_bounces = 31
light.cycles.is_caustics_light = True
light["_potter.light_tree_strength_scale"] = [7.0, 8.0, 9.0]
light["_potter.light_tree_strength_supported"] = False
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
scene = bpy.data.scenes.get("Scene")
assert scene is not None and scene.render.engine == "CYCLES"
assert scene.cycles.use_adaptive_sampling is False
assert scene.cycles.adaptive_threshold == 1.0
assert scene.cycles.adaptive_min_samples == 4096
assert scene.cycles.tile_size == 8192
assert scene.cycles.caustics_reflective is False
assert scene.cycles.caustics_refractive is False
assert scene.cycles.film_exposure == 1.75
assert scene.cycles.film_transparent_glass is True
assert abs(scene.cycles.film_transparent_roughness - 0.37) < 1e-6
view_layer = scene.view_layers[0]
assert view_layer.use_solid is False
assert view_layer.use_strand is False
assert view_layer.use_volumes is False
assert view_layer.use_motion_blur is False
assert view_layer.material_override.name == "Cycles Settings Material"
assert view_layer.world_override.name == "Cycles Settings World"
holder = bpy.data.objects.get("Cycles Settings Material Holder")
assert holder is not None and holder.is_holdout is True
assert abs(holder.shadow_terminator_normal_offset - 0.42) < 1e-6
assert abs(holder.shadow_terminator_geometry_offset - 0.71) < 1e-6
assert abs(holder.shadow_terminator_shading_offset - 0.53) < 1e-6
assert holder["visible_prop"] == "kept"
assert not any(key.startswith("_potter.") for key in holder.keys())
eevee_object = bpy.data.objects.get("EEVEE Terminator")
assert eevee_object is not None
assert abs(eevee_object.shadow_terminator_normal_offset - 0.25) < 1e-6

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
assert not any(key.startswith("_potter.") for key in light.keys())
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
        3,
        "unexpected import losses: {imported}"
    );
    for feature in [
        "blender.world.cycles.is_caustics_light",
        "blender.light.cycles.is_caustics_light",
        "blender.scene.cycles.use_fast_gi",
    ] {
        assert!(
            import_losses
                .iter()
                .any(|loss| loss["feature_id"] == feature),
            "missing import loss {feature}: {imported}"
        );
    }
    for feature in [
        "blender.scene.cycles.use_denoising",
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
        "blender.scene.cycles.caustics_reflective",
        "blender.scene.cycles.caustics_refractive",
        "blender.object.shadow_terminator_shading_offset",
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
    let scene_id = entity_id(&scene, "scenes", "Scene")?;
    let main_scene = &scene["scenes"][scene_id];
    let render = &main_scene["render"];
    assert_eq!(render["caustics_reflective"], false);
    assert_eq!(render["caustics_refractive"], false);
    assert_eq!(render["film_exposure"], 1.75);
    assert_eq!(render["film_transparent_glass"], true);
    assert_eq!(render["use_adaptive_sampling"], false);
    assert!(
        (render["adaptive_threshold"]
            .as_f64()
            .ok_or("adaptive threshold missing")?
            - 0.23)
            .abs()
            < 1e-6
    );
    assert_eq!(render["adaptive_min_samples"], 19);
    assert_eq!(render["tile_size"], 3072);
    assert!(
        (render["film_transparent_roughness"]
            .as_f64()
            .ok_or("film roughness missing")?
            - 0.37)
            .abs()
            < 1e-6
    );
    let view_layers = main_scene["view_layers"]
        .as_object()
        .ok_or("ViewLayer registry is missing")?;
    let (view_layer_id, view_layer) = view_layers
        .iter()
        .find(|(_, layer)| layer["name"] == "ViewLayer")
        .ok_or("default ViewLayer is missing")?;
    for field in ["use_solid", "use_strand", "use_volumes", "use_motion_blur"] {
        assert_eq!(view_layer[field], false, "ViewLayer.{field}");
    }
    assert_eq!(view_layer["material_override"], material_id);
    assert_eq!(view_layer["world_override"], world_id);
    let holder_id = node_id(&scene, "Cycles Settings Material Holder")?;
    let holder = &scene["nodes"][&holder_id];
    assert_eq!(holder["is_holdout"], true);
    for (field, expected) in [
        ("shadow_terminator_normal_offset", 0.42),
        ("shadow_terminator_geometry_offset", 0.71),
        ("shadow_terminator_shading_offset", 0.53),
    ] {
        assert!(
            (holder[field]
                .as_f64()
                .ok_or("terminator offset is missing")?
                - expected)
                .abs()
                < 1e-6,
            "Object.{field}"
        );
    }
    let holder_properties = holder["properties"]
        .as_object()
        .ok_or("Object properties are missing")?;
    assert_eq!(holder_properties["visible_prop"], json!("kept"));
    assert!(
        holder_properties
            .keys()
            .all(|key| !key.starts_with("_potter."))
    );
    let eevee_scene_id = entity_id(&scene, "scenes", "EEVEE Terminator Scene")?;
    let eevee_node_id = node_id(&scene, "EEVEE Terminator")?;
    assert_eq!(
        scene["scenes"][eevee_scene_id]["render"]["engine"],
        "realtime"
    );
    assert_eq!(
        scene["nodes"][&eevee_node_id]["shadow_terminator_normal_offset"],
        0.25
    );
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
    assert!(
        light["tree_strength_scale"].is_null()
            || light["tree_strength_scale"] == json!([1.0, 1.0, 1.0])
    );
    assert!(
        light["tree_strength_supported"]
            .as_bool()
            .is_none_or(|supported| supported)
    );

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
    let inspected_scene = inspect_id(&project, scene_id)?;
    for field in [
        "caustics_reflective",
        "caustics_refractive",
        "film_exposure",
        "film_transparent_glass",
        "film_transparent_roughness",
        "use_adaptive_sampling",
        "adaptive_threshold",
        "adaptive_min_samples",
        "tile_size",
    ] {
        assert_eq!(
            inspected_scene["render"][field], render[field],
            "render.{field}"
        );
    }
    assert_eq!(
        inspected_scene["view_layer_settings"][view_layer_id.as_str()],
        *view_layer
    );
    let inspected_holder = inspect_id(&project, &holder_id)?;
    for field in [
        "is_holdout",
        "shadow_terminator_normal_offset",
        "shadow_terminator_geometry_offset",
        "shadow_terminator_shading_offset",
    ] {
        assert_eq!(inspected_holder[field], holder[field], "Object.{field}");
    }

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
        (
            "adaptive_threshold",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"adaptive_threshold":-0.001}}),
        ),
        (
            "adaptive_threshold",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"adaptive_threshold":1.001}}),
        ),
        (
            "adaptive_min_samples",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"adaptive_min_samples":4097}}),
        ),
        (
            "tile_size",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"tile_size":7}}),
        ),
        (
            "tile_size",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"tile_size":8193}}),
        ),
        (
            "film_exposure",
            json!({"op":"render.update","target":{"id":scene_id},"set":{"film_exposure":4_294_967_297.0}}),
        ),
        (
            "shadow_terminator_shading_offset",
            json!({"op":"node.update","target":{"id":holder_id},"set":{"shadow_terminator_shading_offset":-0.01}}),
        ),
        (
            "use_solid",
            json!({"op":"view_layer.update","target":{"scene_id":scene_id,"id":view_layer_id},"set":{"use_solid":1}}),
        ),
    ];
    for (index, (field, operation)) in invalid_updates.into_iter().enumerate() {
        assert_update_rejected(&project, root, index, &operation, field)?;
    }
    assert_eq!(read_scene(&project)?["revision"], scene["revision"]);
    let update_revision = read_scene(&project)?["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    let valid_update = root.join("update-cycles-render-settings.json");
    fs::write(
        &valid_update,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": update_revision,
            "operations": [{
                "op": "render.update",
                "target": {"id": scene_id},
                "set": {
                    "use_adaptive_sampling": false,
                    "adaptive_threshold": 1.0,
                    "adaptive_min_samples": 4096,
                    "tile_size": 8192
                }
            }]
        }))?,
    )?;
    let update_output = pot()
        .args(["apply"])
        .arg(&project)
        .args(["--file"])
        .arg(&valid_update)
        .arg("--json")
        .output()?;
    assert!(
        update_output.status.success(),
        "render.update failed: {}",
        String::from_utf8_lossy(&update_output.stdout)
    );
    let update_report: Value = serde_json::from_slice(&update_output.stdout)?;
    assert_eq!(update_report["ok"], true, "{update_report}");
    let updated_scene = read_scene(&project)?;
    assert_ne!(updated_scene["revision"], scene["revision"]);
    let updated_render = &updated_scene["scenes"][scene_id]["render"];
    assert_eq!(updated_render["use_adaptive_sampling"], false);
    assert_eq!(updated_render["adaptive_threshold"], 1.0);
    assert_eq!(updated_render["adaptive_min_samples"], 4096);
    assert_eq!(updated_render["tile_size"], 8192);
    let updated_inspection = inspect_id(&project, scene_id)?;
    for field in [
        "use_adaptive_sampling",
        "adaptive_threshold",
        "adaptive_min_samples",
        "tile_size",
    ] {
        assert_eq!(
            updated_inspection["render"][field], updated_render[field],
            "render.update/read {field}"
        );
    }

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
    assert_eq!(export_losses.len(), 3, "unexpected export losses: {report}");
    for feature in [
        "blender.world.cycles.is_caustics_light",
        "blender.light.cycles.is_caustics_light",
        "blender.scene.cycles.use_fast_gi",
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
    let default_blend = root.join("default-startup.blend");
    let default_script = write_script(
        root,
        "save_default_startup.py",
        "import bpy, sys\nbpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index('--') + 1])\n",
    )?;
    run_blender_arguments(&blender, &default_script, None, &[default_blend.as_path()])?;
    let (_, default_import) = import_blend(&blender, root, "default-project", &default_blend)?;
    assert_eq!(
        default_import["result"]["losses"],
        json!([]),
        "default startup scene reported losses: {default_import}"
    );
    Ok(())
}
