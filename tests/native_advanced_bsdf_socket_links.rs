use std::error::Error;

use potter_core::{
    error::{ErrorCode, Result},
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

const GRAPH: &str = "advanced_bsdfs";
const SCENE_ID: &str = "c8a8b9d4-2cbf-4bc4-9b17-6297b72e7d22";

fn apply(operations: &[Value]) -> Result<SceneDoc> {
    let doc = SceneDoc::new(SCENE_ID.to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    Ok(ops::apply_batch(&doc, &batch)?.doc)
}

#[test]
fn standalone_bsdf_nodes_accept_their_shader_and_value_sockets() -> Result<(), Box<dyn Error>> {
    let mut operations = vec![
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"color","type":"ShaderNodeRGB"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"roughness","type":"ShaderNodeValue"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"direction","type":"ShaderNodeCombineXYZ"}),
    ];
    let bsdfs = [
        ("sheen", "ShaderNodeBsdfSheen", "BSDF", Some("Color")),
        (
            "subsurface",
            "ShaderNodeSubsurfaceScattering",
            "BSSRDF",
            Some("Color"),
        ),
        (
            "translucent",
            "ShaderNodeBsdfTranslucent",
            "BSDF",
            Some("Color"),
        ),
        (
            "refraction",
            "ShaderNodeBsdfRefraction",
            "BSDF",
            Some("Color"),
        ),
        ("toon", "ShaderNodeBsdfToon", "BSDF", Some("Color")),
        (
            "metallic",
            "ShaderNodeBsdfMetallic",
            "BSDF",
            Some("Base Color"),
        ),
        ("hair", "ShaderNodeBsdfHair", "BSDF", Some("Color")),
        (
            "hair_principled",
            "ShaderNodeBsdfHairPrincipled",
            "BSDF",
            Some("Color"),
        ),
        (
            "ray_portal",
            "ShaderNodeBsdfRayPortal",
            "BSDF",
            Some("Color"),
        ),
        ("holdout", "ShaderNodeHoldout", "Holdout", None),
        (
            "transparent",
            "ShaderNodeBsdfTransparent",
            "BSDF",
            Some("Color"),
        ),
        ("glass", "ShaderNodeBsdfGlass", "BSDF", Some("Color")),
        ("glossy", "ShaderNodeBsdfGlossy", "BSDF", Some("Color")),
        (
            "anisotropic",
            "ShaderNodeBsdfAnisotropic",
            "BSDF",
            Some("Color"),
        ),
    ];
    for (id, node_type, output_socket, color_socket) in bsdfs {
        let output_id = format!("{id}_output");
        operations.push(json!({"op":"graph.node_add","graph":GRAPH,"id":id,"type":node_type}));
        operations.push(json!({"op":"graph.node_add","graph":GRAPH,"id":output_id,"type":"ShaderNodeOutputMaterial"}));
        if let Some(color_socket) = color_socket {
            operations.push(json!({
                "op":"graph.link","graph":GRAPH,"from_node":"color","from_socket":"Color",
                "to_node":id,"to_socket":color_socket
            }));
        }
        operations.push(json!({
            "op":"graph.link","graph":GRAPH,"from_node":id,"from_socket":output_socket,
            "to_node":output_id,"to_socket":"Surface"
        }));
    }
    operations.extend([
        json!({"op":"graph.link","graph":GRAPH,"from_node":"roughness","from_socket":"Value","to_node":"sheen","to_socket":"Roughness"}),
        json!({"op":"graph.link","graph":GRAPH,"from_node":"direction","from_socket":"Vector","to_node":"ray_portal","to_socket":"Direction"}),
    ]);

    let doc = apply(&operations)?;
    let graph_id = Id::new(GRAPH)?;
    let graph = doc
        .node_groups
        .get(&graph_id)
        .ok_or("advanced BSDF graph was not created")?;
    assert_eq!(graph.links.len(), 29);
    Ok(())
}

#[test]
fn advanced_bsdf_socket_types_and_dynamic_sockets_are_validated() -> Result<(), Box<dyn Error>> {
    let Err(type_mismatch) = apply(&[
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"color","type":"ShaderNodeRGB"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"sheen","type":"ShaderNodeBsdfSheen"}),
        json!({"op":"graph.link","graph":GRAPH,"from_node":"color","from_socket":"Color","to_node":"sheen","to_socket":"Roughness"}),
    ]) else {
        return Err("expected a color-to-roughness link type error".into());
    };
    assert_eq!(type_mismatch.code, ErrorCode::InvalidOperation);

    let dynamic_input = apply(&[
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"ior","type":"ShaderNodeCombineXYZ"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"metallic","type":"ShaderNodeBsdfMetallic","properties":{"fresnel_type":"PHYSICAL_CONDUCTOR"}}),
        json!({"op":"graph.link","graph":GRAPH,"from_node":"ior","from_socket":"Vector","to_node":"metallic","to_socket":"IOR"}),
    ])?;
    let graph_id = Id::new(GRAPH)?;
    let graph = dynamic_input
        .node_groups
        .get(&graph_id)
        .ok_or("dynamic Metallic graph was not created")?;
    assert_eq!(graph.links.len(), 1);

    let Err(unavailable_input) = apply(&[
        json!({"op":"graph.create","id":GRAPH,"kind":"shader"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"color","type":"ShaderNodeRGB"}),
        json!({"op":"graph.node_add","graph":GRAPH,"id":"metallic","type":"ShaderNodeBsdfMetallic","properties":{"fresnel_type":"PHYSICAL_CONDUCTOR"}}),
        json!({"op":"graph.link","graph":GRAPH,"from_node":"color","from_socket":"Color","to_node":"metallic","to_socket":"Base Color"}),
    ]) else {
        return Err("expected the unavailable F82 Base Color socket to reject links".into());
    };
    assert_eq!(unavailable_input.code, ErrorCode::InvalidOperation);
    Ok(())
}
