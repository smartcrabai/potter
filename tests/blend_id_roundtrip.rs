#![expect(clippy::expect_used, reason = "integration test fixtures")]

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;

use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command, operation: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{operation} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn id_sets(document: &Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut result = BTreeMap::new();
    for registry in [
        "scenes",
        "collections",
        "nodes",
        "data_blocks",
        "materials",
        "images",
        "worlds",
        "node_groups",
        "actions",
    ] {
        let ids = document[registry]
            .as_object()
            .expect("scene registry is an object")
            .keys()
            .cloned()
            .collect();
        result.insert(registry.to_owned(), ids);
    }

    let mut view_layer_ids = BTreeSet::new();
    for scene in document["scenes"]
        .as_object()
        .expect("scenes are an object")
        .values()
    {
        view_layer_ids.extend(
            scene["view_layers"]
                .as_object()
                .expect("view layers are an object")
                .keys()
                .cloned(),
        );
    }
    result.insert("view_layers".to_owned(), view_layer_ids);

    let mut action_slot_ids = BTreeSet::new();
    for action in document["actions"]
        .as_object()
        .expect("actions are an object")
        .values()
    {
        action_slot_ids.extend(
            action["slots"]
                .as_array()
                .expect("action slots are an array")
                .iter()
                .map(|slot| slot["id"].as_str().expect("slot ID is a string").to_owned()),
        );
    }
    result.insert("action_slots".to_owned(), action_slot_ids);

    let mut graph_node_ids = BTreeSet::new();
    for group in document["node_groups"]
        .as_object()
        .expect("node groups are an object")
        .values()
    {
        graph_node_ids.extend(
            group["nodes"]
                .as_object()
                .expect("graph nodes are an object")
                .keys()
                .cloned(),
        );
    }
    result.insert("graph_nodes".to_owned(), graph_node_ids);
    result
}

fn assert_nullable_material_slots(doc: &SceneDoc) -> Result<(), Box<dyn Error>> {
    let body_id = Id::new("body".to_owned())?;
    let clay_id = Id::new("clay".to_owned())?;
    let ceramic_id = Id::new("ceramic".to_owned())?;
    let body = doc.nodes.get(&body_id).ok_or("body node is missing")?;
    assert_eq!(
        body.materials.as_slice(),
        &[Some(clay_id), None, Some(ceramic_id.clone())]
    );
    let data_id = body.data.as_ref().ok_or("body data ID is missing")?;
    let mesh = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.mesh.as_ref())
        .ok_or("body mesh is missing")?;
    assert!(mesh.faces.iter().any(|face| face.material_index == 1));
    assert!(mesh.faces.iter().any(|face| face.material_index == 2));

    let mut evaluated_doc = doc.clone();
    evaluated_doc
        .nodes
        .get_mut(&body_id)
        .ok_or("body node is missing")?
        .modifiers
        .clear();
    let snapshot = Snapshot::evaluate(&evaluated_doc, &EvaluationContext::default())?;
    let materials = &snapshot.nodes[&body_id].materials;
    assert_eq!(materials.len(), 3);
    assert_eq!(materials[1], None);
    assert_eq!(materials[2].as_ref(), Some(&ceramic_id));
    assert_eq!(materials.last().and_then(Option::as_ref), Some(&ceramic_id));
    Ok(())
}

fn add_representative_ids(scene: &Path) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    document["scenes"]["scene_main"]["camera"] = json!("camera_node");
    document["scenes"]["scene_main"]["world"] = json!("world_main");
    document["scenes"]["scene_main"]["view_layers"] = json!({
        "view_main": {"name": "View Layer", "excluded_collections": []},
        "view_review": {"name": "Review Layer", "excluded_collections": []}
    });
    document["actions"]["action_main"] = json!({
        "name": "Persistent Action",
        "fcurves": [],
        "slots": [{"id": "action_slot_main", "node": "body"}]
    });
    document["nodes"]["body"]["action"] = json!("action_main");
    document["node_groups"]["graph_main"] = json!({
        "name": "Persistent Geometry",
        "kind": "geometry",
        "interface": {
            "inputs": [],
            "outputs": [{
                "id": "geometry_output",
                "name": "Geometry",
                "socket_type": "geometry",
                "default": null
            }]
        },
        "nodes": {
            "graph_node_main": {
                "type": "NodeGroupOutput",
                "name": "Group Output",
                "location": [200.0, 0.0],
                "properties": {},
                "inputs": {}
            }
        },
        "links": []
    });
    document["worlds"]["world_main"]["node_tree"] = json!("graph_world");
    document["node_groups"]["graph_world"] = json!({
        "name": "Persistent World Nodes",
        "kind": "shader",
        "interface": {"inputs": [], "outputs": []},
        "nodes": {
            "world_background": {
                "type": "ShaderNodeBackground",
                "name": "Background",
                "location": [0.0, 0.0],
                "properties": {},
                "inputs": {"Color": [0.1, 0.2, 0.3, 1.0], "Strength": 0.5}
            },
            "world_output": {
                "type": "ShaderNodeOutputWorld",
                "name": "World Output",
                "location": [300.0, 0.0],
                "properties": {},
                "inputs": {}
            }
        },
        "links": [{
            "from_node": "world_background",
            "from_socket": "Background",
            "to_node": "world_output",
            "to_socket": "Surface"
        }]
    });
    document["nodes"]["body"]["modifiers"] = json!([{
        "id": "modifier_graph",
        "type": "nodes",
        "name": "Persistent Graph Modifier",
        "enabled": true,
        "params": {"node_group": "graph_main"}
    }]);
    fs::write(document_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

#[test]
fn blender_open_save_replace_preserves_all_represented_ids() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender ID round-trip; no executable Blender was found");
        return Ok(());
    };

    let directory = tempdir()?;
    let scene = directory.path().join("source");
    let mut init = pot();
    init.args(["init"]).arg(&scene).arg("--json");
    run_ok(&mut init, "initializing the scene")?;

    let first_operations = scene.with_extension("first.json");
    fs::write(&first_operations, include_str!("fixtures/first.json"))?;
    let mut apply_first = pot();
    apply_first
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&first_operations)
        .arg("--json");
    run_ok(&mut apply_first, "creating the mesh and material")?;

    let second_operations = scene.with_extension("second.json");
    fs::write(
        &second_operations,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "base_revision": 1,
            "operations": [
                {"op": "collection.create", "id": "collection_secondary", "name": "Secondary"},
                {"op": "camera.create", "id": "camera_node", "name": "Camera"},
                {"op": "light.create", "id": "light_node", "name": "Spot Light", "light_type": "spot", "spot_size": 0.75, "spot_blend": 0.2},
                {"op": "world.create", "id": "world_main", "color": [0.1, 0.2, 0.3], "strength": 0.5},
                {"op": "image.create", "id": "image_main", "name": "Round Trip Image", "width": 2, "height": 2},
                {"op": "material.create", "id": "ceramic", "name": "Ceramic"},
                {"op": "node.update", "target": {"id": "body"}, "set": {"materials": ["clay", null, "ceramic"]}},
                {"op": "mesh.set_attribute", "target": {"id": "body"}, "elements": {"domain": "face", "ids": ["f1"]}, "material_index": 1},
                {"op": "mesh.set_attribute", "target": {"id": "body"}, "elements": {"domain": "face", "ids": ["f2"]}, "material_index": 2}
            ]
        }))?,
    )?;
    let mut apply_second = pot();
    apply_second
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&second_operations)
        .arg("--json");
    run_ok(&mut apply_second, "creating the remaining represented IDs")?;
    add_representative_ids(&scene)?;
    let source_scene: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_nullable_material_slots(&source_scene)?;

    let source_document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let expected_ids = id_sets(&source_document);
    assert!(!expected_ids["data_blocks"].is_empty());
    for (registry, id) in [
        ("scenes", "scene_main"),
        ("collections", "collection_secondary"),
        ("nodes", "camera_node"),
        ("nodes", "light_node"),
        ("materials", "clay"),
        ("images", "image_main"),
        ("worlds", "world_main"),
        ("node_groups", "graph_main"),
        ("node_groups", "graph_world"),
        ("actions", "action_main"),
        ("view_layers", "view_review"),
        ("action_slots", "action_slot_main"),
        ("materials", "ceramic"),
        ("graph_nodes", "graph_node_main"),
        ("graph_nodes", "world_background"),
        ("graph_nodes", "world_output"),
    ] {
        assert!(
            expected_ids[registry].contains(id),
            "missing {registry} ID {id}"
        );
    }

    let blend_file = directory.path().join("round-trip.blend");
    let mut export = pot();
    export
        .args(["export"])
        .arg(&scene)
        .args(["--format", "blend", "--out"])
        .arg(&blend_file)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    run_ok(&mut export, "exporting the Blender file")?;

    let mut open_and_save = Command::new(&blender);
    open_and_save
        .args(["--background", "--disable-autoexec"])
        .arg(&blend_file)
        .args(["--python-exit-code", "1", "--python-expr"])
        .arg("import bpy; bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)");
    run_ok(&mut open_and_save, "opening and saving the Blender file")?;

    let mut import = pot();
    import
        .args(["import"])
        .arg(&scene)
        .args(["--file"])
        .arg(&blend_file)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "2",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    run_ok(&mut import, "replacing the scene from Blender")?;

    let imported_document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(id_sets(&imported_document), expected_ids);
    let imported_scene: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_nullable_material_slots(&imported_scene)?;
    let apply_operations = scene.with_extension("apply-slot-modifier.json");
    fs::write(
        &apply_operations,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "base_revision": imported_scene.revision,
            "operations": [
                {"op": "modifier.create", "target": {"id": "body"}, "id": "slot_probe", "type": "triangulate"},
                {"op": "modifier.reorder", "target": {"id": "body"}, "id": "slot_probe", "to_index": 0},
                {"op": "modifier.apply", "target": {"id": "body"}, "modifier_id": "slot_probe"}
            ]
        }))?,
    )?;
    let mut apply_modifier = pot();
    apply_modifier
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&apply_operations)
        .arg("--json");
    run_ok(&mut apply_modifier, "applying a modifier with nullable slots")?;
    let applied_scene: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_nullable_material_slots(&applied_scene)?;
    let source_light_id = source_document["nodes"]["light_node"]["data"]
        .as_str()
        .ok_or("source light data ID is missing")?;
    let imported_light_id = imported_document["nodes"]["light_node"]["data"]
        .as_str()
        .ok_or("imported light data ID is missing")?;
    let source_light = &source_document["data_blocks"][source_light_id]["light"];
    let imported_light = &imported_document["data_blocks"][imported_light_id]["light"];
    assert_eq!(source_light["light_type"], "spot");
    assert_eq!(source_light["spot_size"], json!(0.75));
    assert_eq!(source_light["spot_blend"], json!(0.2));
    for field in ["spot_size", "spot_blend"] {
        let expected = source_light[field]
            .as_f64()
            .ok_or("source spot-light setting is missing")?;
        let actual = imported_light[field]
            .as_f64()
            .ok_or("imported spot-light setting is missing")?;
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "spot light {field} changed from {expected} to {actual}"
        );
    }
    Ok(())
}

fn glb_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.get(0..4) != Some(&b"glTF"[..]) || bytes.get(16..20) != Some(&b"JSON"[..]) {
        return Err("GLB header or JSON chunk is invalid".into());
    }
    let chunk_length = usize::try_from(u32::from_le_bytes(
        bytes
            .get(12..16)
            .ok_or("GLB header is truncated")?
            .try_into()?,
    ))?;
    let chunk_end = 20_usize
        .checked_add(chunk_length)
        .ok_or("GLB JSON chunk length overflows")?;
    Ok(serde_json::from_slice(
        bytes
            .get(20..chunk_end)
            .ok_or("GLB JSON chunk is truncated")?,
    )?)
}

#[test]
fn blender_source_empty_slot_survives_import_apply_render_and_export() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender source material-slot regression; no executable Blender was found");
        return Ok(());
    };

    let directory = tempdir()?;
    let root = directory.path();
    let source = root.join("material-slots.blend");
    let source_path = serde_json::to_string(
        source
            .to_str()
            .ok_or("source Blender path is not UTF-8")?,
    )?;
    let script = r#"
import bpy
from mathutils import Vector

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def material(name, color):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    value.diffuse_color = color
    value["potter.id"] = name.lower()
    shader = value.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
    return value

material_zero = material("Mat0", (0.9, 0.05, 0.02, 1.0))
material_two = material("Mat2", (0.02, 0.1, 0.9, 1.0))
bpy.ops.mesh.primitive_cube_add(size=2.0)
owner = bpy.context.object
owner.name = "SlotOwner"
owner["potter.id"] = "slot_owner"
owner.data.name = "SlotMesh"
owner.data.materials.append(material_zero)
owner.data.materials.append(None)
owner.data.materials.append(material_two)
owner.data.update()
for polygon in owner.data.polygons:
    normal = polygon.normal
    if normal.z > 0.9:
        polygon.material_index = 1
    elif normal.x > 0.9:
        polygon.material_index = 0
    elif normal.y < -0.9:
        polygon.material_index = 2
    else:
        polygon.material_index = 0
owner.modifiers.new("Slot triangulation", "TRIANGULATE")

scene = bpy.context.scene
camera_data = bpy.data.cameras.new("SlotCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("SlotCamera", camera_data)
camera["potter.id"] = "slot_camera"
scene.collection.objects.link(camera)
camera.location = (4.0, -6.0, 4.0)
camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera
light_data = bpy.data.lights.new("SlotLight", "AREA")
light_data.energy = 800.0
light_data.size = 5.0
light = bpy.data.objects.new("SlotLight", light_data)
scene.collection.objects.link(light)
light.location = (4.0, -6.0, 6.0)
light.rotation_euler = (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat("-Z", "Y").to_euler()
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
bpy.ops.wm.save_as_mainfile(filepath=__OUTPUT__)
"#
    .replace("__OUTPUT__", &source_path);
    let mut create_source = Command::new(&blender);
    create_source
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
        ])
        .arg(script);
    run_ok(&mut create_source, "creating a Blender source scene with an empty slot")?;

    let project = root.join("project");
    let mut init = pot();
    init.args(["init"]).arg(&project).arg("--json");
    run_ok(&mut init, "initializing the source-slot project")?;
    let mut import = pot();
    import
        .args(["import"])
        .arg(&project)
        .args([
            "--file",
            source.to_str().ok_or("source path is not UTF-8")?,
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
    run_ok(&mut import, "importing the Blender source-slot scene")?;

    let scene_path = project.join("scene.json");
    let mut scene: SceneDoc = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let node_id = scene
        .nodes
        .iter()
        .find(|(_, node)| node.name == "SlotOwner")
        .map(|(id, _)| id.clone())
        .ok_or("imported material-slot object is missing")?;
    let material_slots = &scene.nodes[&node_id].materials;
    assert_eq!(material_slots.len(), 3);
    assert!(material_slots[1].is_none(), "middle Blender slot stays empty");
    let material_zero = material_slots[0]
        .clone()
        .ok_or("first Blender material slot is empty")?;
    let material_two = material_slots[2]
        .clone()
        .ok_or("last Blender material slot is empty")?;
    assert_eq!(scene.materials[&material_zero].name, "Mat0");
    assert_eq!(scene.materials[&material_two].name, "Mat2");
    let snapshot = Snapshot::evaluate(&scene, &EvaluationContext::default())?;
    let owner = &scene.nodes[&node_id];
    let materials = &snapshot.nodes[&node_id].materials;
    assert_eq!(materials.len(), 3);
    assert_eq!(materials[0].as_ref(), Some(&material_zero));
    assert_eq!(materials[1], None);
    assert_eq!(materials[2].as_ref(), Some(&material_two));
    assert_eq!(materials.last().and_then(Option::as_ref), Some(&material_two));
    let modifier_id = owner
        .modifiers
        .iter()
        .find(|modifier| modifier.modifier_type == "triangulate")
        .map(|modifier| modifier.id.as_str().to_owned())
        .ok_or("source Triangulate modifier is missing")?;
    let apply_operations = root.join("apply.json");
    fs::write(
        &apply_operations,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "base_revision": scene.revision,
            "operations": [{
                "op": "modifier.apply",
                "target": {"id": node_id.as_str()},
                "modifier_id": modifier_id
            }]
        }))?,
    )?;
    let mut apply_modifier = pot();
    apply_modifier
        .args(["apply"])
        .arg(&project)
        .args(["--file"])
        .arg(&apply_operations)
        .arg("--json");
    run_ok(&mut apply_modifier, "applying the imported Triangulate modifier")?;
    scene = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let owner = &scene.nodes[&node_id];
    assert_eq!(owner.materials.len(), 3);
    assert!(owner.materials[1].is_none());
    let snapshot = Snapshot::evaluate(&scene, &EvaluationContext::default())?;
    let materials = &snapshot.nodes[&node_id].materials;
    assert_eq!(materials.len(), 3);
    assert_eq!(materials[1], None);
    assert_eq!(materials[2].as_ref(), Some(&material_two));

    let preview = root.join("preview");
    let mut make_preview = pot();
    make_preview
        .args(["preview"])
        .arg(&project)
        .args(["--views", "top", "--size", "64", "--out"])
        .arg(&preview)
        .arg("--json");
    run_ok(&mut make_preview, "previewing the imported empty-slot mesh")?;
    assert!(preview.join("top.png").is_file());

    let render = root.join("render");
    let mut make_render = pot();
    make_render
        .args(["render"])
        .arg(&project)
        .args(["--engine", "path", "--format", "png"])
        .args(["--out"])
        .arg(&render)
        .arg("--json");
    let rendered = run_ok(&mut make_render, "rendering the imported empty-slot mesh")?;
    let rendered: Value = serde_json::from_slice(&rendered.stdout)?;
    let rendered_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .ok_or("rendered frame path is missing")?;
    assert!(Path::new(rendered_path).is_file());

    let glb = root.join("material-slots.glb");
    let mut export = pot();
    export
        .args(["export"])
        .arg(&project)
        .args(["--format", "glb", "--out"])
        .arg(&glb)
        .arg("--json");
    run_ok(&mut export, "exporting the imported empty-slot mesh")?;
    let exported = glb_json(&glb)?;
    let primitives = exported["meshes"][0]["primitives"]
        .as_array()
        .ok_or("GLB primitives are missing")?;
    assert_eq!(primitives.len(), 3);
    assert!(
        primitives[1].get("material").is_none(),
        "GLB empty-slot primitive must have no material"
    );
    let primitive_materials = primitives
        .iter()
        .map(|primitive| -> Result<Option<String>, Box<dyn Error>> {
            let Some(index) = primitive.get("material").and_then(Value::as_u64) else {
                return Ok(None);
            };
            let index = usize::try_from(index)?;
            Ok(Some(
                exported["materials"][index]["extras"]["potter"]["id"]
                    .as_str()
                    .ok_or("GLB material ID is missing")?
                    .to_owned(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        primitive_materials,
        vec![
            Some(material_zero.as_str().to_owned()),
            None,
            Some(material_two.as_str().to_owned())
        ]
    );
    Ok(())
}
