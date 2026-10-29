use crate::{
    error::{ErrorCode, PotError, Result},
    model::{Id, Registry},
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
            | "GeometryNodeGroup"
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
            | "FunctionNodeAlignEulerToVector"
            | "FunctionNodeAlignRotationToVector"
            | "FunctionNodeAxesToRotation"
            | "FunctionNodeAxisAngleToRotation"
            | "FunctionNodeCombineMatrix"
            | "FunctionNodeCombineTransform"
            | "FunctionNodeEulerToRotation"
            | "FunctionNodeInputRotation"
            | "FunctionNodeInvertMatrix"
            | "FunctionNodeInvertRotation"
            | "FunctionNodeMatrixDeterminant"
            | "FunctionNodeMatrixMultiply"
            | "FunctionNodeMatrixSVD"
            | "FunctionNodeProjectPoint"
            | "FunctionNodeQuaternionToRotation"
            | "FunctionNodeRotateEuler"
            | "FunctionNodeRotateRotation"
            | "FunctionNodeRotateVector"
            | "FunctionNodeRotationToAxisAngle"
            | "FunctionNodeRotationToEuler"
            | "FunctionNodeRotationToQuaternion"
            | "FunctionNodeSeparateMatrix"
            | "FunctionNodeSeparateTransform"
            | "FunctionNodeTransformDirection"
            | "FunctionNodeTransformPoint"
            | "FunctionNodeTransposeMatrix"
            | "ShaderNodeMath"
            | "ShaderNodeMapRange"
            | "ShaderNodeClamp"
            | "FunctionNodeIntegerMath"
            | "FunctionNodeBooleanMath"
            | "FunctionNodeCompare"
            | "ShaderNodeValToRGB"
            | "ShaderNodeFloatCurve"
            | "ShaderNodeVectorCurve"
            | "ShaderNodeRGBCurve"
            | "ShaderNodeBackground"
            | "ShaderNodeEmission"
            | "ShaderNodeOutputMaterial"
            | "ShaderNodeOutputWorld"
            | "ShaderNodeTexSky"
            | "ShaderNodeTexEnvironment"
            | "ShaderNodeTexCoord"
            | "ShaderNodeGroup"
            | "ShaderNodeMixRGB"
            | "ShaderNodeMix"
            | "ShaderNodeVertexColor"
            | "ShaderNodeAttribute"
            | "ShaderNodeAddShader"
            | "ShaderNodeMixShader"
            | "ShaderNodeBsdfPrincipled"
            | "ShaderNodeBsdfDiffuse"
            | "ShaderNodeVolumePrincipled"
            | "ShaderNodeVolumeAbsorption"
            | "ShaderNodeVolumeScatter"
            | "ShaderNodeDisplacement"
            | "ShaderNodeVectorMath"
            | "ShaderNodeRGB"
            | "ShaderNodeValue"
            | "ShaderNodeCombineColor"
            | "ShaderNodeSeparateColor"
            | "ShaderNodeRGBToBW"
            | "ShaderNodeInvert"
            | "ShaderNodeGamma"
            | "ShaderNodeBrightContrast"
            | "ShaderNodeHueSaturation"
            | "ShaderNodeVectorRotate"
            | "ShaderNodeVectorTransform"
            | "ShaderNodeNormal"
            | "ShaderNodeWavelength"
            | "ShaderNodeBlackbody"
            | "ShaderNodeLightFalloff"
            | "ShaderNodeObjectInfo"
            | "ShaderNodeNewGeometry"
            | "ShaderNodeCameraData"
            | "ShaderNodeUVMap"
            | "ShaderNodeBump"
            | "ShaderNodeNormalMap"
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
            | "GeometryNodeBake"
            | "CompositorNodeRLayers"
            | "CompositorNodeAlphaOver"
            | "CompositorNodeBlur"
            | "CompositorNodeOutputFile"
            | "CompositorNodeExposure"
            | "CompositorNodeComposite"
            | "CompositorNodeMask"
    )
}

type SocketList = Vec<(String, String)>;
type SocketPair = (SocketList, SocketList);

pub(super) fn node_sockets(group: &NodeGroup, node_id: &Id) -> Option<SocketPair> {
    let node = group.nodes.get(node_id)?;
    if node.node_type == "GeometryNodeBake" {
        let items = node.properties.get("bake_items")?.as_array()?;
        let bake_sockets = items
            .iter()
            .map(|item| {
                Some((
                    bake_item_identifier(item)?,
                    bake_item_socket_type(item.get("socket_type")?.as_str()?)?.to_owned(),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        return Some((bake_sockets.clone(), bake_sockets));
    }
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
                ("Resolution", "float"),
                ("Start Location", "vector"),
                ("Offset", "vector"),
            ]),
            sockets(MESH_OUT),
        ),
        "GeometryNodeMeshCircle" => (
            sockets(&[("Vertices", "integer"), ("Radius", "float")]),
            sockets(MESH_OUT),
        ),
        "FunctionNodeAlignEulerToVector" => (
            sockets(&[
                ("Rotation", "vector"),
                ("Factor", "float"),
                ("Vector", "vector"),
            ]),
            sockets(&[("Rotation", "vector")]),
        ),
        "FunctionNodeAlignRotationToVector" => (
            sockets(&[
                ("Rotation", "rotation"),
                ("Factor", "float"),
                ("Vector", "vector"),
            ]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeAxesToRotation" => (
            sockets(&[("Primary Axis", "vector"), ("Secondary Axis", "vector")]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeAxisAngleToRotation" => (
            sockets(&[("Axis", "vector"), ("Angle", "float")]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeCombineMatrix" => (
            sockets(&[
                ("Column 1 Row 1", "float"),
                ("Column 1 Row 2", "float"),
                ("Column 1 Row 3", "float"),
                ("Column 1 Row 4", "float"),
                ("Column 2 Row 1", "float"),
                ("Column 2 Row 2", "float"),
                ("Column 2 Row 3", "float"),
                ("Column 2 Row 4", "float"),
                ("Column 3 Row 1", "float"),
                ("Column 3 Row 2", "float"),
                ("Column 3 Row 3", "float"),
                ("Column 3 Row 4", "float"),
                ("Column 4 Row 1", "float"),
                ("Column 4 Row 2", "float"),
                ("Column 4 Row 3", "float"),
                ("Column 4 Row 4", "float"),
            ]),
            sockets(&[("Matrix", "matrix")]),
        ),
        "FunctionNodeCombineTransform" => (
            sockets(&[
                ("Translation", "vector"),
                ("Rotation", "rotation"),
                ("Scale", "vector"),
            ]),
            sockets(&[("Transform", "matrix")]),
        ),
        "FunctionNodeEulerToRotation" => (
            sockets(&[("Euler", "vector")]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeInputRotation" => (Vec::new(), sockets(&[("Rotation", "rotation")])),
        "FunctionNodeInvertMatrix" => (
            sockets(&[("Matrix", "matrix")]),
            sockets(&[("Matrix", "matrix"), ("Invertible", "boolean")]),
        ),
        "FunctionNodeInvertRotation" => (
            sockets(&[("Rotation", "rotation")]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeMatrixDeterminant" => (
            sockets(&[("Matrix", "matrix")]),
            sockets(&[("Determinant", "float")]),
        ),
        "FunctionNodeMatrixMultiply" => (
            sockets(&[("Matrix", "matrix"), ("Matrix_001", "matrix")]),
            sockets(&[("Matrix", "matrix")]),
        ),
        "FunctionNodeMatrixSVD" => (
            sockets(&[("Matrix", "matrix")]),
            sockets(&[("U", "matrix"), ("S", "vector"), ("V", "matrix")]),
        ),
        "FunctionNodeProjectPoint" => (
            sockets(&[("Vector", "vector"), ("Transform", "matrix")]),
            sockets(&[("Vector", "vector")]),
        ),
        "FunctionNodeQuaternionToRotation" => (
            sockets(&[
                ("W", "float"),
                ("X", "float"),
                ("Y", "float"),
                ("Z", "float"),
            ]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeRotateEuler" => {
            let inputs = match node
                .properties
                .get("rotation_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("EULER")
            {
                "EULER" => sockets(&[("Rotation", "vector"), ("Rotate By", "vector")]),
                "AXIS_ANGLE" => sockets(&[
                    ("Rotation", "vector"),
                    ("Axis", "vector"),
                    ("Angle", "float"),
                ]),
                _ => return None,
            };
            (inputs, sockets(&[("Rotation", "vector")]))
        }
        "FunctionNodeRotateRotation" => (
            sockets(&[("Rotation", "rotation"), ("Rotate By", "rotation")]),
            sockets(&[("Rotation", "rotation")]),
        ),
        "FunctionNodeRotateVector" => (
            sockets(&[("Vector", "vector"), ("Rotation", "rotation")]),
            sockets(&[("Vector", "vector")]),
        ),
        "FunctionNodeRotationToAxisAngle" => (
            sockets(&[("Rotation", "rotation")]),
            sockets(&[("Axis", "vector"), ("Angle", "float")]),
        ),
        "FunctionNodeRotationToEuler" => (
            sockets(&[("Rotation", "rotation")]),
            sockets(&[("Euler", "vector")]),
        ),
        "FunctionNodeRotationToQuaternion" => (
            sockets(&[("Rotation", "rotation")]),
            sockets(&[
                ("W", "float"),
                ("X", "float"),
                ("Y", "float"),
                ("Z", "float"),
            ]),
        ),
        "FunctionNodeSeparateMatrix" => {
            let outputs: Vec<_> = (1..=4)
                .flat_map(|column| {
                    (1..=4)
                        .map(move |row| (format!("Column {column} Row {row}"), "float".to_owned()))
                })
                .collect();
            (sockets(&[("Matrix", "matrix")]), outputs)
        }
        "FunctionNodeSeparateTransform" => (
            sockets(&[("Transform", "matrix")]),
            sockets(&[
                ("Translation", "vector"),
                ("Rotation", "rotation"),
                ("Scale", "vector"),
            ]),
        ),
        "FunctionNodeTransformDirection" => (
            sockets(&[("Direction", "vector"), ("Transform", "matrix")]),
            sockets(&[("Direction", "vector")]),
        ),
        "FunctionNodeTransformPoint" => (
            sockets(&[("Vector", "vector"), ("Transform", "matrix")]),
            sockets(&[("Vector", "vector")]),
        ),
        "FunctionNodeTransposeMatrix" => (
            sockets(&[("Matrix", "matrix")]),
            sockets(&[("Matrix", "matrix")]),
        ),
        "CompositorNodeRLayers" => (Vec::new(), sockets(&[("Image", "image")])),
        "CompositorNodeAlphaOver" => (
            sockets(&[
                ("Factor", "float"),
                ("Background", "image"),
                ("Foreground", "image"),
            ]),
            sockets(&[("Image", "image")]),
        ),
        "CompositorNodeBlur" => (
            sockets(&[("Image", "image"), ("Size", "vector")]),
            sockets(&[("Image", "image")]),
        ),
        "CompositorNodeOutputFile" | "CompositorNodeComposite" => {
            (sockets(&[("Image", "image")]), Vec::new())
        }
        "CompositorNodeExposure" => (
            sockets(&[("Image", "image"), ("Exposure", "float")]),
            sockets(&[("Image", "image")]),
        ),
        "CompositorNodeMask" => (Vec::new(), sockets(&[("Mask", "image")])),
        "GeometryNodeTransform" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Mode", "string"),
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
        "GeometryNodeJoinGeometry" | "GeometryNodeRepeatOutput" => {
            (sockets(GEOMETRY_IN), sockets(GEOMETRY_OUT))
        }
        "GeometryNodeRealizeInstances" => (
            sockets(&[
                ("Geometry", "geometry"),
                ("Selection", "boolean"),
                ("Realize All", "boolean"),
                ("Depth", "integer"),
            ]),
            sockets(GEOMETRY_OUT),
        ),
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
                ("Density Factor", "float"),
                ("Density Max", "float"),
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
                ("Limit Surface", "boolean"),
                ("Quality", "integer"),
                ("UV Smooth", "menu"),
                ("Boundary Smooth", "menu"),
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
        "ShaderNodeBsdfDiffuse" => (
            sockets(&[
                ("Color", "color"),
                ("Roughness", "float"),
                ("Normal", "vector"),
            ]),
            sockets(&[("BSDF", "shader")]),
        ),
        "ShaderNodeOutputWorld" | "OutputWorld" => (sockets(&[("Surface", "shader")]), Vec::new()),
        "ShaderNodeOutputMaterial" | "OutputMaterial" => (
            sockets(&[
                ("Surface", "shader"),
                ("Volume", "shader"),
                ("Displacement", "vector"),
                ("Thickness", "float"),
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
                ("Reflection", "vector"),
            ]),
        ),
        "ShaderNodeUVMap" => (Vec::new(), sockets(&[("UV", "vector")])),
        "ShaderNodeRGB" => (Vec::new(), sockets(&[("Color", "color")])),
        "ShaderNodeValue" => (Vec::new(), sockets(&[("Value", "float")])),
        "ShaderNodeCombineColor" => (
            sockets(&[("Red", "float"), ("Green", "float"), ("Blue", "float")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeSeparateColor" => (
            sockets(&[("Color", "color")]),
            sockets(&[("Red", "float"), ("Green", "float"), ("Blue", "float")]),
        ),
        "ShaderNodeRGBToBW" => (sockets(&[("Color", "color")]), sockets(&[("Val", "float")])),
        "ShaderNodeInvert" | "ShaderNodeRGBCurve" => (
            sockets(&[("Fac", "float"), ("Color", "color")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeGamma" => (
            sockets(&[("Color", "color"), ("Gamma", "float")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeBrightContrast" => (
            sockets(&[
                ("Color", "color"),
                ("Bright", "float"),
                ("Contrast", "float"),
            ]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeHueSaturation" => (
            sockets(&[
                ("Hue", "float"),
                ("Saturation", "float"),
                ("Value", "float"),
                ("Fac", "float"),
                ("Color", "color"),
            ]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeVectorRotate" => (
            sockets(&[
                ("Vector", "vector"),
                ("Center", "vector"),
                ("Axis", "vector"),
                ("Angle", "float"),
                ("Rotation", "vector"),
            ]),
            sockets(&[("Vector", "vector")]),
        ),
        "ShaderNodeVectorTransform" => (
            sockets(&[("Vector", "vector")]),
            sockets(&[("Vector", "vector")]),
        ),
        "ShaderNodeNormal" => (
            sockets(&[("Normal", "vector")]),
            sockets(&[("Normal", "vector"), ("Dot", "float")]),
        ),
        "ShaderNodeWavelength" => (
            sockets(&[("Wavelength", "float")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeBlackbody" => (
            sockets(&[("Temperature", "float")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeLightFalloff" => (
            sockets(&[("Strength", "float"), ("Smooth", "float")]),
            sockets(&[
                ("Quadratic", "float"),
                ("Linear", "float"),
                ("Constant", "float"),
            ]),
        ),
        "ShaderNodeObjectInfo" => (
            Vec::new(),
            sockets(&[
                ("Location", "vector"),
                ("Color", "color"),
                ("Alpha", "float"),
                ("Object Index", "float"),
                ("Material Index", "float"),
                ("Random", "float"),
            ]),
        ),
        "ShaderNodeNewGeometry" => (
            Vec::new(),
            sockets(&[
                ("Position", "vector"),
                ("Normal", "vector"),
                ("Tangent", "vector"),
                ("True Normal", "vector"),
                ("Incoming", "vector"),
                ("Parametric", "vector"),
                ("Backfacing", "float"),
                ("Pointiness", "float"),
                ("Random Per Island", "float"),
            ]),
        ),
        "ShaderNodeCameraData" => (
            Vec::new(),
            sockets(&[
                ("View Vector", "vector"),
                ("View Z Depth", "float"),
                ("View Distance", "float"),
            ]),
        ),
        "ShaderNodeBump" => (
            sockets(&[
                ("Strength", "float"),
                ("Distance", "float"),
                ("Filter Width", "float"),
                ("Height", "float"),
                ("Normal", "vector"),
            ]),
            sockets(&[("Normal", "vector")]),
        ),
        "ShaderNodeNormalMap" => (
            sockets(&[("Strength", "float"), ("Color", "color")]),
            sockets(&[("Normal", "vector")]),
        ),
        "ShaderNodeMixRGB" => (
            sockets(&[("Fac", "float"), ("Color1", "color"), ("Color2", "color")]),
            sockets(&[("Color", "color")]),
        ),
        "ShaderNodeVertexColor" => (
            sockets(&[("Layer Name", "string")]),
            sockets(&[("Color", "color"), ("Alpha", "float")]),
        ),
        "ShaderNodeAttribute" => (
            sockets(&[("Name", "string")]),
            sockets(&[
                ("Color", "color"),
                ("Vector", "vector"),
                ("Fac", "float"),
                ("Alpha", "float"),
            ]),
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
                ("Absorption Color", "color"),
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
        "ShaderNodeMath" => math_sockets(node),
        "ShaderNodeVectorMath" => vector_math_sockets(node),
        "FunctionNodeIntegerMath" => integer_math_sockets(node),
        "FunctionNodeBooleanMath" => boolean_math_sockets(node),
        "FunctionNodeCompare" => compare_sockets(node),
        "ShaderNodeMapRange" => map_range_sockets(node),
        "ShaderNodeClamp" => (
            sockets(&[("Value", "float"), ("Min", "float"), ("Max", "float")]),
            sockets(&[("Result", "float")]),
        ),
        "ShaderNodeMix" => mix_sockets(node),
        "ShaderNodeValToRGB" => (
            sockets(&[("Fac", "float")]),
            sockets(&[("Color", "color"), ("Alpha", "float")]),
        ),
        "ShaderNodeFloatCurve" => (
            sockets(&[("Factor", "float"), ("Value", "float")]),
            sockets(&[("Value", "float")]),
        ),
        "ShaderNodeVectorCurve" => (
            sockets(&[("Fac", "float"), ("Vector", "vector")]),
            sockets(&[("Vector", "vector")]),
        ),
        "ShaderNodeCombineXYZ" => (
            sockets(&[("X", "float"), ("Y", "float"), ("Z", "float")]),
            sockets(&[("Vector", "vector")]),
        ),
        "ShaderNodeSeparateXYZ" => (
            sockets(&[("Vector", "vector")]),
            sockets(&[("X", "float"), ("Y", "float"), ("Z", "float")]),
        ),
        "FunctionNodeRandomValue" => random_value_sockets(node),
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

fn math_sockets(_node: &super::GraphNode) -> SocketPair {
    (
        sockets(&[
            ("Value", "float"),
            ("Value_001", "float"),
            ("Value_002", "float"),
        ]),
        sockets(&[("Value", "float")]),
    )
}

fn vector_math_sockets(_node: &super::GraphNode) -> SocketPair {
    (
        sockets(&[
            ("Vector", "vector"),
            ("Vector_001", "vector"),
            ("Vector_002", "vector"),
            ("Scale", "float"),
        ]),
        sockets(&[("Vector", "vector"), ("Value", "float")]),
    )
}

fn property_enum(node: &super::GraphNode, key: &str, default: &str) -> String {
    node.properties
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or(default)
        .to_ascii_uppercase()
}

fn integer_math_sockets(_node: &super::GraphNode) -> SocketPair {
    (
        sockets(&[
            ("Value", "integer"),
            ("Value_001", "integer"),
            ("Value_002", "integer"),
        ]),
        sockets(&[("Value", "integer")]),
    )
}

fn boolean_math_sockets(node: &super::GraphNode) -> SocketPair {
    let operation = property_enum(node, "operation", "AND");
    let mut inputs = sockets(&[("Boolean", "boolean")]);
    if operation != "NOT" {
        inputs.push(("Boolean_001".to_owned(), "boolean".to_owned()));
    }
    (inputs, sockets(&[("Boolean", "boolean")]))
}

fn compare_socket_type(data_type: &str) -> &'static str {
    match data_type {
        "FLOAT" => "float",
        "INT" | "INTEGER" => "integer",
        "VECTOR" | "FLOAT_VECTOR" => "vector",
        "RGBA" | "COLOR" => "color",
        "BOOLEAN" | "BOOL" => "boolean",
        "STRING" => "string",
        "OBJECT" => "object",
        "IMAGE" => "image",
        "COLLECTION" => "collection",
        "MATERIAL" => "material",
        "FONT" => "font",
        "SOUND" => "sound",
        _ => "value",
    }
}

fn compare_sockets(node: &super::GraphNode) -> SocketPair {
    let data_type = property_enum(node, "data_type", "FLOAT");
    let operation = property_enum(node, "operation", "GREATER_THAN");
    let mode = property_enum(node, "mode", "ELEMENT");
    let input_type = compare_socket_type(&data_type);
    let mut inputs = sockets(&[("A", "value"), ("B", "value")]);
    for input in &mut inputs {
        input_type.clone_into(&mut input.1);
    }
    if matches!(data_type.as_str(), "VECTOR" | "FLOAT_VECTOR") && mode == "DOT_PRODUCT" {
        inputs.push(("C".to_owned(), "float".to_owned()));
    }
    if matches!(data_type.as_str(), "VECTOR" | "FLOAT_VECTOR") && mode == "DIRECTION" {
        inputs.push(("Angle".to_owned(), "float".to_owned()));
    }
    if matches!(operation.as_str(), "EQUAL" | "NOT_EQUAL")
        && matches!(
            data_type.as_str(),
            "FLOAT" | "VECTOR" | "FLOAT_VECTOR" | "RGBA"
        )
    {
        inputs.push(("Epsilon".to_owned(), "float".to_owned()));
    }
    (inputs, sockets(&[("Result", "boolean")]))
}

fn map_range_sockets(_node: &super::GraphNode) -> SocketPair {
    (
        sockets(&[
            ("Value", "float"),
            ("From Min", "float"),
            ("From Max", "float"),
            ("To Min", "float"),
            ("To Max", "float"),
            ("Steps", "float"),
            ("Vector", "vector"),
            ("From_Min_FLOAT3", "vector"),
            ("From_Max_FLOAT3", "vector"),
            ("To_Min_FLOAT3", "vector"),
            ("To_Max_FLOAT3", "vector"),
            ("Steps_FLOAT3", "vector"),
        ]),
        sockets(&[("Result", "float"), ("Vector", "vector")]),
    )
}

fn mix_sockets(_node: &super::GraphNode) -> SocketPair {
    (
        sockets(&[
            ("Factor_Float", "float"),
            ("Factor_Vector", "vector"),
            ("A_Float", "float"),
            ("B_Float", "float"),
            ("A_Vector", "vector"),
            ("B_Vector", "vector"),
            ("A_Color", "color"),
            ("B_Color", "color"),
            ("A_Rotation", "rotation"),
            ("B_Rotation", "rotation"),
        ]),
        sockets(&[
            ("Result_Float", "float"),
            ("Result_Vector", "vector"),
            ("Result_Color", "color"),
            ("Result_Rotation", "rotation"),
        ]),
    )
}

fn random_value_sockets(node: &super::GraphNode) -> SocketPair {
    let data_type = property_enum(node, "data_type", "FLOAT");
    let inputs = if matches!(data_type.as_str(), "BOOLEAN" | "BOOL") {
        sockets(&[
            ("Probability", "float"),
            ("Seed", "integer"),
            ("ID", "integer"),
        ])
    } else {
        sockets(&[
            ("Min", "value"),
            ("Max", "value"),
            ("Seed", "integer"),
            ("ID", "integer"),
        ])
    };
    (inputs, sockets(&[("Value", "value")]))
}

/// The socket identifier of a Bake item: its `id`, falling back to `identifier` and then `name`.
pub(super) fn bake_item_identifier(item: &serde_json::Value) -> Option<String> {
    item.get("id")
        .or_else(|| item.get("identifier"))
        .or_else(|| item.get("name"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}
fn bake_item_socket_type(value: &str) -> Option<&'static str> {
    match value {
        "geometry" | "GEOMETRY" | "NodeSocketGeometry" => Some("geometry"),
        "float" | "FLOAT" | "NodeSocketFloat" => Some("float"),
        "integer" | "INT" | "INTEGER" | "NodeSocketInt" => Some("integer"),
        "boolean" | "BOOL" | "BOOLEAN" | "NodeSocketBool" => Some("boolean"),
        "vector" | "VECTOR" | "FLOAT_VECTOR" | "NodeSocketVector" => Some("vector"),
        "rotation" | "ROTATION" | "NodeSocketRotation" => Some("rotation"),
        "color" | "COLOR" | "RGBA" | "NodeSocketColor" => Some("color"),
        "string" | "STRING" | "NodeSocketString" => Some("string"),
        _ => None,
    }
}

fn sockets(items: &[Socket]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(name, socket_type)| ((*name).to_owned(), (*socket_type).to_owned()))
        .collect()
}

fn socket_type(
    group: &NodeGroup,
    node_groups: Option<&Registry<NodeGroup>>,
    node_id: &Id,
    socket_name: &str,
    output: bool,
) -> Option<String> {
    let node = group.nodes.get(node_id)?;
    if matches!(
        node.node_type.as_str(),
        "ShaderNodeGroup" | "GeometryNodeGroup"
    ) {
        let group_id = node.properties.get("node_tree")?.as_str()?;
        let child_group = node_groups?.get(group_id)?;
        if node.node_type == "GeometryNodeGroup"
            && output
            && child_group.kind == super::GraphKind::Geometry
            && child_group.interface.outputs.is_empty()
            && socket_name == "Geometry"
        {
            return Some("geometry".to_owned());
        }
        let sockets = if output {
            &child_group.interface.outputs
        } else {
            &child_group.interface.inputs
        };
        let socket = sockets
            .iter()
            .find(|socket| socket.id == socket_name || socket.name == socket_name)?;
        return graph_socket_type(&socket.socket_type).map(str::to_owned);
    }
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
            .and_then(|socket| graph_socket_type(&socket.socket_type).map(str::to_owned));
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
            .and_then(|socket| graph_socket_type(&socket.socket_type).map(str::to_owned));
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

fn graph_socket_type(socket_type: &str) -> Option<&'static str> {
    match socket_type {
        "float"
        | "NodeSocketFloat"
        | "NodeSocketFloatFactor"
        | "NodeSocketFloatAngle"
        | "NodeSocketFloatDistance"
        | "NodeSocketFloatPercentage"
        | "NodeSocketFloatWavelength"
        | "NodeSocketFloatColorTemperature" => Some("float"),
        "integer" | "NodeSocketInt" => Some("integer"),
        "boolean" | "NodeSocketBool" => Some("boolean"),
        "vector"
        | "NodeSocketVector"
        | "NodeSocketVectorDirection"
        | "NodeSocketVectorEuler"
        | "NodeSocketVectorTranslation"
        | "NodeSocketVectorXYZ" => Some("vector"),
        "color" | "NodeSocketColor" => Some("color"),
        "rotation" | "NodeSocketRotation" => Some("rotation"),
        "matrix" | "NodeSocketMatrix" => Some("matrix"),
        "menu" | "NodeSocketMenu" => Some("menu"),
        "object" | "NodeSocketObject" => Some("object"),
        "collection" | "NodeSocketCollection" => Some("collection"),
        "material" | "NodeSocketMaterial" => Some("material"),
        "image" | "NodeSocketImage" => Some("image"),
        _ => None,
    }
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
    validate_link_with_registry(group, None, link)
}

pub fn validate_link_with_groups(
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    link: &GraphLink,
) -> Result<()> {
    validate_link_with_registry(group, Some(node_groups), link)
}

fn validate_link_with_registry(
    group: &NodeGroup,
    node_groups: Option<&Registry<NodeGroup>>,
    link: &GraphLink,
) -> Result<()> {
    if group.links.contains(link) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "graph link already exists",
        ));
    }
    validate_link_structure(group, node_groups, link)?;
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
        validate_link_structure(group, None, link)?;
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
pub(super) fn node_sockets_with_groups(
    group: &NodeGroup,
    node_id: &Id,
    node_groups: &Registry<NodeGroup>,
) -> Option<SocketPair> {
    let node = group.nodes.get(node_id)?;
    if node.node_type != "GeometryNodeGroup" {
        return node_sockets(group, node_id);
    }
    let group_id = node.properties.get("node_tree")?.as_str()?;
    let child_group = node_groups.get(group_id)?;
    let input_sockets = child_group
        .interface
        .inputs
        .iter()
        .flat_map(|socket| {
            std::iter::once((socket.id.clone(), socket.socket_type.clone())).chain(
                (socket.name != socket.id)
                    .then(|| (socket.name.clone(), socket.socket_type.clone())),
            )
        })
        .collect();
    let output_sockets = if child_group.kind == super::GraphKind::Geometry
        && child_group.interface.outputs.is_empty()
    {
        sockets(GEOMETRY_OUT)
    } else {
        child_group
            .interface
            .outputs
            .iter()
            .flat_map(|socket| {
                std::iter::once((socket.id.clone(), socket.socket_type.clone())).chain(
                    (socket.name != socket.id)
                        .then(|| (socket.name.clone(), socket.socket_type.clone())),
                )
            })
            .collect()
    };
    Some((input_sockets, output_sockets))
}
pub(super) fn validate_group_links_with_groups(
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
) -> Result<()> {
    for (index, link) in group.links.iter().enumerate() {
        validate_link_structure(group, Some(node_groups), link)?;
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

fn validate_link_structure(
    group: &NodeGroup,
    node_groups: Option<&Registry<NodeGroup>>,
    link: &GraphLink,
) -> Result<()> {
    let from_type = socket_type(group, node_groups, &link.from_node, &link.from_socket, true)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!(
                    "unknown output socket `{}` on `{}`",
                    link.from_socket, link.from_node
                ),
            )
        })?;
    let to_type = socket_type(group, node_groups, &link.to_node, &link.to_socket, false)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!(
                    "unknown input socket `{}` on `{}`",
                    link.to_socket, link.to_node
                ),
            )
        })?;
    if from_type != to_type && !value_socket_compatible(&from_type, &to_type) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("socket type mismatch: `{from_type}` cannot connect to `{to_type}`"),
        ));
    }
    Ok(())
}
fn value_socket_compatible(from_type: &str, to_type: &str) -> bool {
    let is_value = |socket_type| {
        matches!(
            socket_type,
            "value"
                | "float"
                | "integer"
                | "boolean"
                | "vector"
                | "rotation"
                | "color"
                | "string"
                | "object"
                | "image"
                | "collection"
                | "material"
                | "font"
                | "sound"
        )
    };
    (from_type == "value" && is_value(to_type)) || (to_type == "value" && is_value(from_type))
}

fn has_single_link_input(group: &NodeGroup, link: &GraphLink) -> bool {
    link.to_socket != "Geometry"
        || group
            .nodes
            .get(&link.to_node)
            .is_none_or(|node| node.node_type != "GeometryNodeJoinGeometry")
}
