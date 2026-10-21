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

blend_path, out_path = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.open_mainfile(filepath=blend_path)
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
