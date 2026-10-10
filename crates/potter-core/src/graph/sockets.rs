use crate::{
    error::{ErrorCode, PotError, Result},
    model::Id,
};

use super::{GraphLink, NodeGroup};

type Socket = (&'static str, &'static str);

const GEOMETRY_IN: &[Socket] = &[("Geometry", "geometry")];
const GEOMETRY_OUT: &[Socket] = &[("Geometry", "geometry")];
const MESH_OUT: &[Socket] = &[("Mesh", "geometry")];
const POINTS_OUT: &[Socket] = &[("Points", "geometry")];

#[must_use]
pub fn is_known_node_type(node_type: &str) -> bool {
    matches!(
        node_type,
        "NodeGroupInput"
            | "NodeGroupOutput"
            | "GeometryNodeMeshCube"
            | "GeometryNodeMeshIcoSphere"
            | "GeometryNodeTransform"
            | "GeometryNodeSetPosition"
            | "GeometryNodeJoinGeometry"
            | "GeometryNodeInstanceOnPoints"
            | "GeometryNodeRealizeInstances"
            | "GeometryNodeDistributePointsOnFaces"
            | "GeometryNodeMeshToPoints"
            | "GeometryNodeSubdivisionSurface"
            | "GeometryNodeMeshGrid"
            | "GeometryNodeMeshUVSphere"
            | "GeometryNodeMeshCylinder"
            | "GeometryNodeMeshCone"
            | "GeometryNodeMeshLine"
            | "GeometryNodeMeshCircle"
            | "ShaderNodeMath"
            | "ShaderNodeBackground"
            | "ShaderNodeEmission"
            | "ShaderNodeOutputMaterial"
            | "ShaderNodeOutputWorld"
            | "ShaderNodeTexSky"
            | "ShaderNodeTexEnvironment"
            | "ShaderNodeTexCoord"
            | "ShaderNodeMixRGB"
            | "ShaderNodeMix"
            | "ShaderNodeMixShader"
            | "ShaderNodeAddShader"
            | "ShaderNodeBsdfPrincipled"
            | "ShaderNodeVolumePrincipled"
            | "ShaderNodeVolumeAbsorption"
            | "ShaderNodeVolumeScatter"
            | "ShaderNodeDisplacement"
            | "ShaderNodeVectorMath"
            | "GeometryNodeInputPosition"
            | "GeometryNodeInputNormal"
            | "GeometryNodeInputIndex"
            | "GeometryNodeStoreNamedAttribute"
            | "GeometryNodeInputNamedAttribute"
            | "GeometryNodeCaptureAttribute"
            | "ShaderNodeCombineXYZ"
            | "ShaderNodeSeparateXYZ"
            | "FunctionNodeRandomValue"
            | "GeometryNodeRepeatInput"
            | "GeometryNodeSimulationInput"
            | "GeometryNodeSimulationOutput"
            | "GeometryNodeRepeatOutput"
            | "CompositorNodeRLayers"
            | "CompositorNodeExposure"
            | "CompositorNodeComposite"
            | "CompositorNodeMask"
    )
}

type SocketList = Vec<(String, String)>;
type SocketPair = (SocketList, SocketList);

pub(super) fn node_sockets(group: &NodeGroup, node_id: &Id) -> Option<SocketPair> {
    let node = group.nodes.get(node_id)?;
    let sockets = match node.node_type.as_str() {
        "NodeGroupInput" => (
            Vec::new(),
            group
                .interface
                .inputs
                .iter()
                .map(|socket| (socket.name.clone(), socket.socket_type.clone()))
                .collect(),
        ),
        "NodeGroupOutput" => {
            let inputs: Vec<_> = group
                .interface
                .outputs
                .iter()
                .map(|socket| (socket.name.clone(), socket.socket_type.clone()))
                .collect();
            if inputs.is_empty() && group.kind == super::GraphKind::Geometry {
                (sockets(&[("Geometry", "geometry")]), Vec::new())
            } else {
                (inputs, Vec::new())
            }
        }
        "GeometryNodeMeshCube" => (
            sockets(&[
                ("Size", "vector"),
                ("Vertices X", "integer"),
                ("Vertices Y", "integer"),
                ("Vertices Z", "integer"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshGrid" => (
            sockets(&[
                ("Size X", "float"),
                ("Size Y", "float"),
                ("Vertices X", "integer"),
                ("Vertices Y", "integer"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshUVSphere" => (
            sockets(&[
                ("Segments", "integer"),
                ("Rings", "integer"),
                ("Radius", "float"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshIcoSphere" => (
            sockets(&[("Radius", "float"), ("Subdivisions", "integer")]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshCylinder" => (
            sockets(&[
                ("Vertices", "integer"),
                ("Side Segments", "integer"),
                ("Fill Segments", "integer"),
                ("Radius", "float"),
                ("Depth", "float"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshCone" => (
            sockets(&[
                ("Vertices", "integer"),
                ("Side Segments", "integer"),
                ("Fill Segments", "integer"),
                ("Radius Top", "float"),
                ("Radius Bottom", "float"),
                ("Depth", "float"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshLine" => (
            sockets(&[
                ("Count", "integer"),
                ("Start Location", "vector"),
                ("Offset", "vector"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshCircle" => (
            sockets(&[("Vertices", "integer"), ("Radius", "float")]),
            sockets(MESH_OUT),
        ),
        "CompositorNodeRLayers" => (Vec::new(), sockets(&[("Image", "image")])),
        "CompositorNodeExposure" => (
            sockets(&[("Image", "image"), ("Exposure", "float")]),
            sockets(&[("Image", "image")]),
        ),
        "CompositorNodeComposite" => (sockets(&[("Image", "image")]), Vec::new()),
        "CompositorNodeMask" => (Vec::new(), sockets(&[("Mask", "image")])),
        "GeometryNodeTransform" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Translation", "vector"),
                ("Rotation", "rotation"),
                ("Scale", "vector"),
            ]),
            sockets(GEOMETRY_OUT),
        ),
        "GeometryNodeSetPosition" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Selection", "boolean"),
                ("Position", "vector"),
                ("Offset", "vector"),
            ]),
            sockets(GEOMETRY_OUT),
        ),
        "GeometryNodeJoinGeometry"
        | "GeometryNodeRealizeInstances"
        | "GeometryNodeRepeatOutput" => (sockets(GEOMETRY_IN), sockets(GEOMETRY_OUT)),
        "GeometryNodeInstanceOnPoints" => (
            sockets(&[
                ("Points", "geometry"),
                ("Selection", "boolean"),
                ("Instance", "geometry"),
                ("Pick Instance", "boolean"),
                ("Instance Index", "integer"),
                ("Rotation", "rotation"),
                ("Scale", "vector"),
            ]),
            sockets(&[("Instances", "geometry")]),
        ),
        "GeometryNodeDistributePointsOnFaces" => (
            sockets(&[
                ("Mesh", "geometry"),
                ("Selection", "boolean"),
                ("Distance Min", "float"),
                ("Density", "float"),
                ("Seed", "integer"),
            ]),
            sockets(&[
                ("Points", "geometry"),
                ("Rotation", "rotation"),
                ("Normal", "vector"),
            ]),
        ),
        "GeometryNodeMeshToPoints" => (
            sockets(&[
                ("Mesh", "geometry"),
                ("Selection", "boolean"),
                ("Position", "vector"),
                ("Radius", "float"),
            ]),
            sockets(POINTS_OUT),
        ),
        "GeometryNodeSubdivisionSurface" => (
            sockets(&[
                ("Mesh", "geometry"),
                ("Level", "integer"),
                ("Edge Crease", "float"),
                ("Vertex Crease", "float"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeInputPosition" => (Vec::new(), sockets(&[("Position", "vector")])),
        "GeometryNodeInputNormal" => (Vec::new(), sockets(&[("Normal", "vector")])),
        "GeometryNodeInputIndex" => (Vec::new(), sockets(&[("Index", "integer")])),
        "GeometryNodeStoreNamedAttribute" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Selection", "boolean"),
                ("Name", "string"),
                ("Value", "value"),
            ]),
            sockets(GEOMETRY_OUT),
        ),
        "GeometryNodeInputNamedAttribute" => (
            sockets(&[("Name", "string")]),
            sockets(&[("Attribute", "value"), ("Exists", "boolean")]),
        ),
        "GeometryNodeCaptureAttribute" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Selection", "boolean"),
                ("Value", "value"),
            ]),
            sockets(&[("Geometry", "geometry"), ("Attribute", "value")]),
        ),
        "ShaderNodeBackground" => (
            sockets(&[("Color", "color"), ("Strength", "float")]),
            sockets(&[("Background", "shader")]),
        ),
        "ShaderNodeEmission" => (
            sockets(&[("Color", "color"), ("Strength", "float")]),
            sockets(&[("Emission", "shader")]),
        ),
        "ShaderNodeOutputWorld" | "OutputWorld" => (sockets(&[("Surface", "shader")]), Vec::new()),
        "ShaderNodeOutputMaterial" | "OutputMaterial" => (
            sockets(&[
                ("Surface", "shader"),
                ("Volume", "shader"),
                ("Displacement", "vector"),
            ]),
            Vec::new(),
        ),
        "ShaderNodeTexSky" => (
            sockets(&[("Vector", "vector"), ("Sun Direction", "vector")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeTexEnvironment" => (
            sockets(&[("Vector", "vector")]),
            sockets(&[("Color", "color"), ("Alpha", "float")]),
        ),
        "ShaderNodeTexCoord" => (
            Vec::new(),
            sockets(&[
                ("Generated", "vector"),
                ("Normal", "vector"),
                ("UV", "vector"),
                ("Object", "vector"),
                ("Camera", "vector"),
                ("Window", "vector"),
                ("Incoming", "vector"),
            ]),
        ),
        "ShaderNodeMixRGB" | "ShaderNodeMix" => (
            sockets(&[("Fac", "float"), ("Color1", "color"), ("Color2", "color")]),
            sockets(&[("Color", "color"), ("Alpha", "float")]),
        ),
        "ShaderNodeMixShader" => (
            sockets(&[
                ("Fac", "float"),
                ("Shader", "shader"),
                ("Shader_001", "shader"),
            ]),
            sockets(&[("Shader", "shader")]),
        ),
        "ShaderNodeAddShader" => (
            sockets(&[("Shader", "shader"), ("Shader_001", "shader")]),
            sockets(&[("Shader", "shader")]),
        ),
        "ShaderNodeBsdfPrincipled" => (
            sockets(&[
                ("Base Color", "color"),
                ("Metallic", "float"),
                ("Roughness", "float"),
                ("IOR", "float"),
                ("Alpha", "float"),
                ("Transmission Weight", "float"),
                ("Emission Color", "color"),
                ("Emission Strength", "float"),
                ("Normal", "vector"),
            ]),
            sockets(&[("BSDF", "shader")]),
        ),
        "ShaderNodeVolumePrincipled" => (
            sockets(&[
                ("Color", "color"),
                ("Density", "float"),
                ("Anisotropy", "float"),
                ("Emission Color", "color"),
                ("Emission Strength", "float"),
            ]),
            sockets(&[("Volume", "shader")]),
        ),
        "ShaderNodeVolumeAbsorption" => (
            sockets(&[("Color", "color"), ("Density", "float")]),
            sockets(&[("Volume", "shader")]),
        ),
        "ShaderNodeVolumeScatter" => (
            sockets(&[
                ("Color", "color"),
                ("Density", "float"),
                ("Anisotropy", "float"),
            ]),
            sockets(&[("Volume", "shader")]),
        ),
        "ShaderNodeDisplacement" => (
            sockets(&[
                ("Height", "float"),
                ("Midlevel", "float"),
                ("Scale", "float"),
                ("Normal", "vector"),
            ]),
            sockets(&[("Displacement", "vector")]),
        ),
        "ShaderNodeMath" => (
            sockets(&[
                ("Value", "float"),
                ("Value_001", "float"),
                ("Value_002", "float"),
            ]),
            sockets(&[("Value", "float")]),
        ),
        "ShaderNodeVectorMath" => (
            sockets(&[
                ("Vector", "vector"),
                ("Vector_001", "vector"),
                ("Vector_002", "vector"),
                ("Scale", "float"),
            ]),
            sockets(&[("Vector", "vector"), ("Value", "float")]),
        ),
        "ShaderNodeCombineXYZ" => (
            sockets(&[("X", "float"), ("Y", "float"), ("Z", "float")]),
            sockets(&[("Vector", "vector")]),
        ),
        "ShaderNodeSeparateXYZ" => (
            sockets(&[("Vector", "vector")]),
            sockets(&[("X", "float"), ("Y", "float"), ("Z", "float")]),
        ),
        "FunctionNodeRandomValue" => (
            sockets(&[
                ("Min", "value"),
                ("Max", "value"),
                ("Seed", "integer"),
                ("ID", "integer"),
            ]),
            sockets(&[("Value", "value")]),
        ),
        "GeometryNodeRepeatInput" => (
            sockets(&[("Iterations", "integer"), ("Geometry", "geometry")]),
            sockets(&[("Iteration", "integer"), ("Geometry", "geometry")]),
        ),
        "GeometryNodeSimulationInput" => (
            sockets(&[("Geometry", "geometry")]),
            sockets(&[("Geometry", "geometry"), ("Delta Time", "float")]),
        ),
        "GeometryNodeSimulationOutput" => (
            sockets(&[("Geometry", "geometry"), ("Skip", "boolean")]),
            sockets(GEOMETRY_OUT),
        ),
        _ => return None,
    };
    Some(sockets)
}

fn sockets(items: &[Socket]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(name, socket_type)| ((*name).to_owned(), (*socket_type).to_owned()))
        .collect()
}

fn socket_type(group: &NodeGroup, node_id: &Id, socket_name: &str, output: bool) -> Option<String> {
    let node = group.nodes.get(node_id)?;
    if node.node_type == "ShaderNodeVectorMath" && output {
        let operation = node
            .properties
            .get("operation")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("ADD");
        let scalar_output = matches!(operation, "DOT_PRODUCT" | "DISTANCE" | "LENGTH");
        return match (socket_name, scalar_output) {
            ("Value", true) => Some("float".to_owned()),
            ("Vector", false) => Some("vector".to_owned()),
            _ => None,
        };
    }
    if node.node_type == "NodeGroupInput" && output {
        return group
            .interface
            .inputs
            .iter()
            .find(|socket| socket.name == socket_name || socket.id == socket_name)
            .map(|socket| socket.socket_type.clone());
    }
    if node.node_type == "NodeGroupOutput" && !output {
        if group.interface.outputs.is_empty() && socket_name == "Geometry" {
            return Some("geometry".to_owned());
        }
        return group
            .interface
            .outputs
            .iter()
            .find(|socket| socket.name == socket_name || socket.id == socket_name)
            .map(|socket| socket.socket_type.clone());
    }
    if let Some((inputs, outputs)) = node_sockets(group, node_id) {
        let sockets = if output { outputs } else { inputs };
        if let Some((_, socket_type)) = sockets.iter().find(|(name, _)| name == socket_name) {
            if socket_type == "value" {
                return Some(dynamic_value_type(node));
            }
            return Some(socket_type.clone());
        }
    }
    None
}

fn dynamic_value_type(node: &super::GraphNode) -> String {
    let data_type = node
        .properties
        .get("data_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("FLOAT");
    match data_type {
        "INT" | "INTEGER" => "integer".to_owned(),
        "BOOLEAN" | "BOOL" => "boolean".to_owned(),
        "VECTOR" | "FLOAT_VECTOR" => "vector".to_owned(),
        "ROTATION" => "rotation".to_owned(),
        "STRING" => "string".to_owned(),
        "FLOAT" => "float".to_owned(),
        _ => "value".to_owned(),
    }
}

/// Check that a link joins two existing, compatible sockets.
///
/// # Errors
///
/// Returns `INVALID_OPERATION` for absent nodes/sockets, type mismatches, or duplicate links.
pub fn validate_link(group: &NodeGroup, link: &GraphLink) -> Result<()> {
    if group.links.contains(link) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "graph link already exists",
        ));
    }
    validate_link_structure(group, link)?;
    if has_single_link_input(group, link)
        && group.links.iter().any(|existing| {
            existing.to_node == link.to_node && existing.to_socket == link.to_socket
        })
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "input socket already has a link",
        ));
    }
    Ok(())
}

pub(super) fn validate_group_links(group: &NodeGroup) -> Result<()> {
    for (index, link) in group.links.iter().enumerate() {
        validate_link_structure(group, link)?;
        if group.links[..index].contains(link) {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "graph link already exists",
            ));
        }
        if has_single_link_input(group, link)
            && group.links[..index].iter().any(|existing| {
                existing.to_node == link.to_node && existing.to_socket == link.to_socket
            })
        {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "input socket already has a link",
            ));
        }
    }
    Ok(())
}

fn validate_link_structure(group: &NodeGroup, link: &GraphLink) -> Result<()> {
    let from_type =
        socket_type(group, &link.from_node, &link.from_socket, true).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!(
                    "unknown output socket `{}` on `{}`",
                    link.from_socket, link.from_node
                ),
            )
        })?;
    let to_type = socket_type(group, &link.to_node, &link.to_socket, false).ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            format!(
                "unknown input socket `{}` on `{}`",
                link.to_socket, link.to_node
            ),
        )
    })?;
    if from_type != to_type {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("socket type mismatch: `{from_type}` cannot connect to `{to_type}`"),
        ));
    }
    Ok(())
}

fn has_single_link_input(group: &NodeGroup, link: &GraphLink) -> bool {
    link.to_socket != "Geometry"
        || group
            .nodes
            .get(&link.to_node)
            .is_none_or(|node| node.node_type != "GeometryNodeJoinGeometry")
}
