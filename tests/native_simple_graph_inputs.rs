//! `.blend` import must not overwrite an authored shader graph's Principled inputs with the Material's values.
//!
//! The exporter marks only the Principled node Potter generated (`potter_simple_material`). An authored graph has no
//! marker, so its inputs are its own. The test builds an authored shader graph whose Principled Roughness and Base
//! Colour differ from the Material's values, exports the `.blend`, reopens it in Blender 5.2.2, and reads the node
//! back. A second material with a Potter-generated graph still takes its Material values, so the marker path is
//! covered too.

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

use std::{fs, path::Path};

use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, build_oracle, export,
    import_blend, losses, node_id, number, read_scene, run_blender, write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

/// The authored graph's own Principled values.
const GRAPH_ROUGHNESS: f64 = 0.15;
const GRAPH_BASE: [f64; 4] = [0.2, 0.9, 0.2, 1.0];
/// The Material's values, which a generated material takes on import.
const MATERIAL_ROUGHNESS: f64 = 0.6;
const MATERIAL_BASE: [f64; 4] = [0.8, 0.1, 0.1, 1.0];

const PRINCIPLED_SCRIPT: &str = r#"
import bpy
import json
import sys

out_path = sys.argv[sys.argv.index("--") + 1]
result = {}
for material in bpy.data.materials:
    if material.node_tree is None:
        continue
    node = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if node is None:
        continue
    result[material.name] = {
        "roughness": float(node.inputs["Roughness"].default_value),
        "base_color": [float(c) for c in node.inputs["Base Color"].default_value],
    }
with open(out_path, "w") as handle:
    json.dump(result, handle)
"#;

fn blender_principled(blender: &Path, root: &Path, blend: &Path) -> TestResult<Value> {
    let script = write_script(root, "read_principled.py", PRINCIPLED_SCRIPT)?;
    let out = root.join("principled.json");
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

#[test]
fn authored_graph_inputs_survive_blend_round_trip() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (oracle, _) = build_oracle(&blender, root.path())?;
    let (project, report) = import_blend(&blender, root.path(), "project", &oracle)?;
    assert!(
        losses(&report).is_empty(),
        "oracle import lost data: {report}"
    );
    let scene = read_scene(&project)?;
    let plane = node_id(&scene, "TexturePlane")?;

    // An authored graph: a Principled with its own Roughness and Base Colour, linked to a Material Output. Its
    // material's Material values differ from the graph's, so the graph's values are the ones to keep.
    let authored = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "graph.create", "id": "authored_graph", "kind": "shader"},
            {"op": "graph.node_add", "graph": "authored_graph", "id": "output",
             "type": "ShaderNodeOutputMaterial"},
            {"op": "graph.node_add", "graph": "authored_graph", "id": "principled",
             "type": "ShaderNodeBsdfPrincipled",
             "inputs": {"Roughness": GRAPH_ROUGHNESS, "Base Color": GRAPH_BASE}},
            {"op": "graph.link", "graph": "authored_graph", "from_node": "principled",
             "from_socket": "BSDF", "to_node": "output", "to_socket": "Surface"},
            {"op": "material.create", "id": "authored_material", "name": "Authored Graph",
             "graph": "authored_graph", "base_color": MATERIAL_BASE, "roughness": MATERIAL_ROUGHNESS,
             "alpha_mode": "opaque"},
            {"op": "material.create", "id": "generated_material", "name": "Generated Material",
             "base_color": MATERIAL_BASE, "roughness": MATERIAL_ROUGHNESS, "alpha_mode": "opaque"},
            {"op": "node.create", "id": "generated_body", "kind": "box",
             "params": {"size": 0.5}, "material": "generated_material"},
            {"op": "node.update", "target": {"id": plane},
             "set": {"materials": ["authored_material"]}}
        ]),
    )?;
    assert_eq!(authored["ok"], true, "material setup failed: {authored}");

    let exported = root.path().join("simple-graph.blend");
    let export_report = export(&blender, &project, "blend", &exported)?;
    assert!(
        losses(&export_report).is_empty(),
        "export lost data: {export_report}"
    );

    let read = blender_principled(&blender, root.path(), &exported)?;
    let generated_node = &read["Generated Material"];
    assert_close(
        number(&generated_node["roughness"])?,
        MATERIAL_ROUGHNESS,
        FLOAT_TOLERANCE,
        "generated material roughness after import",
    );
    let authored_node = &read["Authored Graph"];
    assert_close(
        number(&authored_node["roughness"])?,
        GRAPH_ROUGHNESS,
        FLOAT_TOLERANCE,
        "authored graph roughness after import",
    );
    for (channel, expected) in GRAPH_BASE.iter().take(3).enumerate() {
        assert_close(
            number(&authored_node["base_color"][channel])?,
            *expected,
            FLOAT_TOLERANCE,
            "authored graph base colour after import",
        );
    }
    Ok(())
}

/// A Blender-authored material: a Principled graph with its own Base Colour and Roughness, which the import records as
/// the material's baseline.
const IMPORTED_FIXTURE: &str = r#"
import bpy
import sys

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
material = bpy.data.materials.new("Authored")
material.use_nodes = True
principled = material.node_tree.nodes["Principled BSDF"]
principled.inputs["Base Color"].default_value = (0.2, 0.9, 0.2, 1.0)
principled.inputs["Roughness"].default_value = 0.15
bpy.ops.mesh.primitive_plane_add(size=2.0)
bpy.context.object.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#;

/// A graph edit on an imported authored graph survives export when its Material is untouched, and the untouched fields
/// stay as the author made them: the import's baseline shows no Material field changed, so export writes nothing.
#[test]
fn graph_edit_survives_export_with_untouched_material() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = root.path().join("authored.blend");
    run_blender(
        &blender,
        &write_script(root.path(), "authored_fixture.py", IMPORTED_FIXTURE)?,
        None,
        &fixture,
    )?;
    let (project, report) = import_blend(&blender, root.path(), "project", &fixture)?;
    assert!(losses(&report).is_empty(), "import lost data: {report}");
    let scene = read_scene(&project)?;
    let graph_id = scene["materials"]
        .as_object()
        .and_then(|materials| {
            materials
                .values()
                .find(|material| material["name"] == "Authored")
        })
        .and_then(|material| material["node_tree"].as_str().map(str::to_owned))
        .ok_or("the imported Authored material has no shader graph")?;
    let principled_id = scene["node_groups"][&graph_id]["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["type"] == "ShaderNodeBsdfPrincipled")
        })
        .map(|(id, _)| id.clone())
        .ok_or("the Authored graph has no Principled node")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "graph.node_update", "graph": &graph_id, "node_id": &principled_id,
             "set": {"inputs": {"Roughness": 0.4}}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "graph edit failed: {applied}");
    let exported = root.path().join("graph-edit.blend");
    let export_report = export(&blender, &project, "blend", &exported)?;
    assert!(
        losses(&export_report).is_empty(),
        "export lost data: {export_report}"
    );
    let read = blender_principled(&blender, root.path(), &exported)?;
    let node = &read["Authored"];
    assert_close(
        number(&node["roughness"])?,
        0.4,
        FLOAT_TOLERANCE,
        "graph edit roughness after export",
    );
    for (channel, expected) in [0.2, 0.9, 0.2].iter().enumerate() {
        assert_close(
            number(&node["base_color"][channel])?,
            *expected,
            FLOAT_TOLERANCE,
            "untouched base colour after export",
        );
    }
    Ok(())
}

/// A Potter-authored graph is stamped with its material's baseline when the material is created, so a `material.update`
/// after that reaches the graph: the edited Base Colour replaces the graph's own. The untouched Roughness keeps the
/// graph's value, because the Material's Roughness still equals its baseline.
#[test]
fn material_edit_reaches_a_potter_authored_graph() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (oracle, _) = build_oracle(&blender, root.path())?;
    let (project, report) = import_blend(&blender, root.path(), "project", &oracle)?;
    assert!(
        losses(&report).is_empty(),
        "oracle import lost data: {report}"
    );

    let edited_base = [0.5, 0.5, 0.5, 1.0];
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "graph.create", "id": "authored_graph", "kind": "shader"},
            {"op": "graph.node_add", "graph": "authored_graph", "id": "output",
             "type": "ShaderNodeOutputMaterial"},
            {"op": "graph.node_add", "graph": "authored_graph", "id": "principled",
             "type": "ShaderNodeBsdfPrincipled",
             "inputs": {"Roughness": GRAPH_ROUGHNESS, "Base Color": GRAPH_BASE}},
            {"op": "graph.link", "graph": "authored_graph", "from_node": "principled",
             "from_socket": "BSDF", "to_node": "output", "to_socket": "Surface"},
            {"op": "material.create", "id": "authored_material", "name": "Authored Graph",
             "graph": "authored_graph", "base_color": MATERIAL_BASE,
             "roughness": MATERIAL_ROUGHNESS, "alpha_mode": "opaque"},
            {"op": "node.create", "id": "authored_body", "kind": "box",
             "params": {"size": 0.5}, "material": "authored_material"},
            {"op": "material.update", "target": {"id": "authored_material"},
             "set": {"base_color": edited_base}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "authoring failed: {applied}");

    let exported = root.path().join("authored-edit.blend");
    let export_report = export(&blender, &project, "blend", &exported)?;
    assert!(
        losses(&export_report).is_empty(),
        "export lost data: {export_report}"
    );
    let read = blender_principled(&blender, root.path(), &exported)?;
    let node = &read["Authored Graph"];
    assert_close(
        number(&node["roughness"])?,
        GRAPH_ROUGHNESS,
        FLOAT_TOLERANCE,
        "untouched roughness keeps the graph's value",
    );
    for (channel, expected) in edited_base.iter().take(3).enumerate() {
        assert_close(
            number(&node["base_color"][channel])?,
            *expected,
            FLOAT_TOLERANCE,
            "material base colour edit reaches the authored graph",
        );
    }
    Ok(())
}
