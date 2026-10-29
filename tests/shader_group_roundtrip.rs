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

use std::fs;

use native_oracle::{
    TestResult, assert_close, export, import_blend, losses, read_scene, run_blender, write_script,
};
use potter_core::{
    model::SceneDoc,
    shader::{HitContext, evaluate_surface},
};
use serde_json::{Value, json};
use tempfile::tempdir;

const BUILD_FIXTURE: &str = r#"
import bpy
import sys

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)

inner = bpy.data.node_groups.new("Inner Shader", "ShaderNodeTree")
weight = inner.interface.new_socket(name="Weight", in_out="INPUT", socket_type="NodeSocketFloat")
weight.default_value = 0.375
tint = inner.interface.new_socket(name="Tint", in_out="INPUT", socket_type="NodeSocketColor")
tint.default_value = (0.15, 0.3, 0.45, 1.0)
factor = inner.interface.new_socket(name="Factor", in_out="OUTPUT", socket_type="NodeSocketFloat")
factor.default_value = 0.875
inner.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
group_input = inner.nodes.new("NodeGroupInput")
emission = inner.nodes.new("ShaderNodeEmission")
group_output = inner.nodes.new("NodeGroupOutput")
group_output.is_active_output = True
# The output interface default and the unlinked Group Output socket default are independent.
group_output.inputs["Factor"].default_value = 0.25
inner.links.new(group_input.outputs["Weight"], emission.inputs["Strength"])
inner.links.new(group_input.outputs["Tint"], emission.inputs["Color"])
inner.links.new(emission.outputs["Emission"], group_output.inputs["Shader"])

outer = bpy.data.node_groups.new("Outer Shader", "ShaderNodeTree")
outer_weight = outer.interface.new_socket(name="Weight", in_out="INPUT", socket_type="NodeSocketFloat")
outer_weight.default_value = 0.625
outer_tint = outer.interface.new_socket(name="Tint", in_out="INPUT", socket_type="NodeSocketColor")
outer_tint.default_value = (0.7, 0.2, 0.1, 1.0)
outer.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
outer_input = outer.nodes.new("NodeGroupInput")
nested = outer.nodes.new("ShaderNodeGroup")
nested.node_tree = inner
outer_output = outer.nodes.new("NodeGroupOutput")
outer_output.is_active_output = True
outer.links.new(outer_input.outputs["Weight"], nested.inputs["Weight"])
outer.links.new(outer_input.outputs["Tint"], nested.inputs["Tint"])
outer.links.new(nested.outputs["Shader"], outer_output.inputs["Shader"])

material = bpy.data.materials.new("Nested Group Material")
material.use_nodes = True
material.node_tree.nodes.clear()
root_group = material.node_tree.nodes.new("ShaderNodeGroup")
root_group.node_tree = outer
material_output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
material.node_tree.links.new(root_group.outputs["Shader"], material_output.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=2.0)
bpy.context.object.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#;

const READ_GROUPS: &str = r#"
import bpy
import json
import sys

path = sys.argv[sys.argv.index("--") + 1]
material = bpy.data.materials["Nested Group Material"]
root_group = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeGroup")
outer = root_group.node_tree
nested = next(node for node in outer.nodes if node.bl_idname == "ShaderNodeGroup")
inner = nested.node_tree

def interface_socket(tree, name, direction):
    return next(socket for socket in tree.interface.items_tree
                if socket.item_type == "SOCKET" and socket.name == name and socket.in_out == direction)

def links(tree):
    return sorted([[link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name]
                   for link in tree.links])

result = {
    "root_group": root_group.node_tree.name,
    "nested_group": nested.node_tree.name,
    "inner_weight_default": interface_socket(inner, "Weight", "INPUT").default_value,
    "inner_tint_default": list(interface_socket(inner, "Tint", "INPUT").default_value),
    "inner_factor_interface_default": interface_socket(inner, "Factor", "OUTPUT").default_value,
    "inner_factor_node_default": inner.nodes["Group Output"].inputs["Factor"].default_value,
    "outer_weight_default": interface_socket(outer, "Weight", "INPUT").default_value,
    "outer_tint_default": list(interface_socket(outer, "Tint", "INPUT").default_value),
    "inner_links": links(inner),
    "outer_links": links(outer),
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(result, output)
"#;

fn graph_named<'a>(scene: &'a Value, name: &str) -> TestResult<&'a Value> {
    scene["node_groups"]
        .as_object()
        .and_then(|groups| groups.values().find(|group| group["name"] == name))
        .ok_or_else(|| format!("imported graph {name} is missing").into())
}

fn assert_channels(actual: &Value, expected: [f64; 4], label: &str) -> TestResult {
    let values = actual
        .as_array()
        .ok_or_else(|| format!("{label} is not an array"))?;
    assert_eq!(values.len(), expected.len(), "{label} channel count");
    for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
        assert_close(
            actual
                .as_f64()
                .ok_or_else(|| format!("{label}[{index}] is not numeric"))?,
            expected,
            1.0e-6,
            &format!("{label}[{index}]"),
        );
    }
    Ok(())
}

fn has_link(
    links: &Value,
    from_node: &str,
    from_socket: &str,
    to_node: &str,
    to_socket: &str,
) -> bool {
    links.as_array().is_some_and(|items| {
        items.iter().any(|link| {
            link[0].as_str() == Some(from_node)
                && link[1].as_str() == Some(from_socket)
                && link[2].as_str() == Some(to_node)
                && link[3].as_str() == Some(to_socket)
        })
    })
}

#[test]
fn shader_group_interface_defaults_and_nested_links_survive_blend_round_trip() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = root.path().join("nested-shader-groups.blend");
    run_blender(
        &blender,
        &write_script(root.path(), "build_shader_groups.py", BUILD_FIXTURE)?,
        None,
        &fixture,
    )?;

    let (project, import_report) = import_blend(&blender, root.path(), "project", &fixture)?;
    assert!(
        losses(&import_report).is_empty(),
        "fixture import lost data: {import_report}"
    );
    let scene = read_scene(&project)?;
    let inner = graph_named(&scene, "Inner Shader")?;
    let inner_factor = inner["interface"]["outputs"]
        .as_array()
        .and_then(|sockets| sockets.iter().find(|socket| socket["name"] == "Factor"))
        .ok_or("the imported inner graph has no Factor output")?;
    assert_eq!(inner_factor["default"], json!(0.875));
    let outer = graph_named(&scene, "Outer Shader")?;
    let nested = outer["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .values()
                .find(|node| node["type"] == "ShaderNodeGroup")
        })
        .ok_or("the imported outer graph has no nested group node")?;
    let inner_group_id = scene["node_groups"]
        .as_object()
        .and_then(|groups| {
            groups
                .iter()
                .find_map(|(id, group)| (group["name"] == "Inner Shader").then_some(id.as_str()))
        })
        .ok_or("the imported inner group has no persistent ID")?;
    assert_eq!(
        nested["properties"]["node_tree"].as_str(),
        Some(inner_group_id)
    );
    let document: SceneDoc = serde_json::from_value(scene.clone())?;
    let material = document
        .materials
        .values()
        .find(|material| material.name == "Nested Group Material")
        .ok_or("the imported fixture has no nested-group material")?;
    let shading = evaluate_surface(material, &HitContext::new(&document.node_groups))?;
    let params = &shading
        .closures
        .first()
        .ok_or("the nested group produced no shader closure")?
        .params;
    for (actual, expected) in params.emission_color.iter().zip([0.7, 0.2, 0.1]) {
        assert!(
            (*actual - expected).abs() < 1.0e-6,
            "nested-group emission color was {actual}, expected {expected}"
        );
    }
    assert!(
        (params.emission_strength - 0.625).abs() < 1.0e-6,
        "nested-group emission strength was {}",
        params.emission_strength
    );

    let exported_blend = root.path().join("shader-groups-round-trip.blend");
    let export_report = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&export_report).is_empty(),
        "group export lost data: {export_report}"
    );
    let output_json = root.path().join("reopened-groups.json");
    run_blender(
        &blender,
        &write_script(root.path(), "read_shader_groups.py", READ_GROUPS)?,
        Some(&exported_blend),
        &output_json,
    )?;
    let actual: Value = serde_json::from_slice(&fs::read(output_json)?)?;
    assert_eq!(actual["root_group"], "Outer Shader");
    assert_eq!(actual["nested_group"], "Inner Shader");
    assert_eq!(actual["outer_weight_default"], json!(0.625));
    assert_channels(
        &actual["inner_tint_default"],
        [0.15, 0.3, 0.45, 1.0],
        "inner Tint",
    )?;
    assert_channels(
        &actual["outer_tint_default"],
        [0.7, 0.2, 0.1, 1.0],
        "outer Tint",
    )?;
    assert_eq!(actual["inner_links"].as_array().map(Vec::len), Some(3));
    assert_eq!(actual["outer_links"].as_array().map(Vec::len), Some(3));
    assert!(has_link(
        &actual["inner_links"],
        "Group Input",
        "Weight",
        "Emission",
        "Strength"
    ));
    assert!(has_link(
        &actual["inner_links"],
        "Group Input",
        "Tint",
        "Emission",
        "Color"
    ));
    assert!(has_link(
        &actual["outer_links"],
        "Group Input",
        "Weight",
        "Group",
        "Weight"
    ));
    assert!(has_link(
        &actual["outer_links"],
        "Group Input",
        "Tint",
        "Group",
        "Tint"
    ));
    assert!(has_link(
        &actual["outer_links"],
        "Group",
        "Shader",
        "Group Output",
        "Shader"
    ));
    Ok(())
}
