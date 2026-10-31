#![expect(clippy::unwrap_used, reason = "integration tests")]

use std::{error::Error, fs};

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

use blender_file::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    eval::EvaluationContext,
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    graph::{self, GraphSocket, NodeGroup},
    model::{Id, SceneDoc},
    ops,
    store::Project,
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import sys
from mathutils import Euler, Matrix, Quaternion
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"geometry node parity requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])

def geometry_group(name):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return tree

def connect(links, source, output, target, input_name):
    links.new(source.outputs[output], target.inputs[input_name])

def store(tree, geometry, source, output, name, data_type):
    node = tree.nodes.new("GeometryNodeStoreNamedAttribute")
    node.domain = "POINT"
    node.data_type = data_type
    node.inputs["Name"].default_value = name
    tree.links.new(geometry, node.inputs["Geometry"])
    connect(tree.links, source, output, node, "Value")
    return node.outputs["Geometry"]

def matrix_node(tree, name, rows):
    node = tree.nodes.new("FunctionNodeCombineMatrix")
    node.name = name
    for row in range(4):
        for column in range(4):
            node.inputs[f"Column {column + 1} Row {row + 1}"].default_value = rows[row][column]
    return node

def matrix_multiply(tree, name, first, second):
    node = tree.nodes.new("FunctionNodeMatrixMultiply")
    node.name = name
    tree.links.new(first.outputs["Matrix"], node.inputs[0])
    tree.links.new(second.outputs["Matrix"], node.inputs[1])
    return node

def rotate_vector(tree, geometry, rotation, vector, name):
    node = tree.nodes.new("FunctionNodeRotateVector")
    node.inputs["Vector"].default_value = vector
    tree.links.new(rotation.outputs["Rotation"], node.inputs["Rotation"])
    return store(tree, geometry, node, "Vector", name, "FLOAT_VECTOR")

# The inner group consumes its unconnected interface defaults.  The incoming mesh begins away
# from the origin, so the child group proves both nested I/O and the default-driven offset.
child = geometry_group("GN Offset Child")
child_offset = child.interface.new_socket(name="Offset", in_out="INPUT", socket_type="NodeSocketVector")
child_offset.default_value = (0.25, -0.75, 0.5)
child_factor = child.interface.new_socket(name="Factor", in_out="INPUT", socket_type="NodeSocketFloat")
child_factor.default_value = 2.0
child_input = child.nodes.new("NodeGroupInput")
child_output = child.nodes.new("NodeGroupOutput")
child_scale = child.nodes.new("ShaderNodeVectorMath")
child_scale.operation = "SCALE"
child.links.new(child_input.outputs["Offset"], child_scale.inputs["Vector"])
child.links.new(child_input.outputs["Factor"], child_scale.inputs["Scale"])
child_position = child.nodes.new("GeometryNodeSetPosition")
child.links.new(child_input.outputs["Geometry"], child_position.inputs["Geometry"])
child.links.new(child_scale.outputs["Vector"], child_position.inputs["Offset"])
child.links.new(child_position.outputs["Geometry"], child_output.inputs["Geometry"])

# Typed interface defaults exercise ID references and value-shaped socket defaults on a group
# that is actually nested in the geometry graph.
resource_group = geometry_group("GN Resource Defaults")
resource_group.use_fake_user = True
resource_specs = [
    ("Object Default", "NodeSocketObject"),
    ("Collection Default", "NodeSocketCollection"),
    ("Material Default", "NodeSocketMaterial"),
    ("Image Default", "NodeSocketImage"),
    ("Matrix Default", "NodeSocketMatrix"),
    ("Rotation Default", "NodeSocketRotation"),
    ("Menu Default", "NodeSocketMenu"),
    ("String Default", "NodeSocketString"),
]
resource_sockets = {
    name: resource_group.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
    for name, socket_type in resource_specs
}

resource_collection = bpy.data.collections.new("GN Default Collection")
bpy.context.scene.collection.children.link(resource_collection)
resource_mesh = bpy.data.meshes.new("GN Default Object Mesh")
resource_object = bpy.data.objects.new("GN Default Object", resource_mesh)
resource_collection.objects.link(resource_object)
resource_material = bpy.data.materials.new("GN Default Material")
resource_image = bpy.data.images.new("GN Default Image", width=1, height=1, alpha=True)
resource_image.pixels = (0.125, 0.25, 0.5, 1.0)
resource_image.pack()
resource_sockets["Object Default"].default_value = resource_object
resource_sockets["Collection Default"].default_value = resource_collection
resource_sockets["Material Default"].default_value = resource_material
resource_sockets["Image Default"].default_value = resource_image
rotation_default = Euler((0.2, -0.4, 0.7), "XYZ")
resource_sockets["Rotation Default"].default_value = rotation_default
resource_sockets["String Default"].default_value = "nested group default"
resource_input = resource_group.nodes.new("NodeGroupInput")
menu_switch = resource_group.nodes.new("GeometryNodeMenuSwitch")
menu_switch.enum_items.new("MODE_A")
menu_switch.enum_items.new("MODE_B")
resource_group.links.new(resource_input.outputs["Menu Default"], menu_switch.inputs["Menu"])
resource_sockets["Menu Default"].default_value = "MODE_B"
resource_output = resource_group.nodes.new("NodeGroupOutput")
resource_group.links.new(resource_input.outputs["Geometry"], resource_output.inputs["Geometry"])

root_group = geometry_group("GN Group Rotation Matrix Root")
root_output = root_group.nodes.new("NodeGroupOutput")
line = root_group.nodes.new("GeometryNodeMeshLine")
line.mode = "OFFSET"
line.inputs["Count"].default_value = 1
line.inputs["Start Location"].default_value = (1.25, -0.5, 0.75)
child_call = root_group.nodes.new("GeometryNodeGroup")
child_call.node_tree = child
root_group.links.new(line.outputs["Mesh"], child_call.inputs["Geometry"])
geometry = child_call.outputs["Geometry"]
position = root_group.nodes.new("GeometryNodeInputPosition")
geometry = store(root_group, geometry, position, "Position", "nested_position", "FLOAT_VECTOR")

# Rotation conversion and operations: identity and zero-axis boundaries, normalized quaternion
# conversion, local Euler/rotation composition, and both vector alignment node families.
identity = root_group.nodes.new("FunctionNodeInputRotation")
geometry = rotate_vector(root_group, geometry, identity, (1.25, -2.0, 0.5), "input_rotation_vector")
identity_axis_angle = root_group.nodes.new("FunctionNodeRotationToAxisAngle")
root_group.links.new(identity.outputs["Rotation"], identity_axis_angle.inputs["Rotation"])
geometry = store(root_group, geometry, identity_axis_angle, "Axis", "identity_axis", "FLOAT_VECTOR")
geometry = store(root_group, geometry, identity_axis_angle, "Angle", "identity_angle", "FLOAT")
identity_euler = root_group.nodes.new("FunctionNodeRotationToEuler")
root_group.links.new(identity.outputs["Rotation"], identity_euler.inputs["Rotation"])
geometry = store(root_group, geometry, identity_euler, "Euler", "identity_euler", "FLOAT_VECTOR")
identity_quaternion = root_group.nodes.new("FunctionNodeRotationToQuaternion")
root_group.links.new(identity.outputs["Rotation"], identity_quaternion.inputs["Rotation"])
for component in ("W", "X", "Y", "Z"):
    geometry = store(root_group, geometry, identity_quaternion, component,
                     "identity_quaternion_" + component.lower(), "FLOAT")

zero_axis = root_group.nodes.new("FunctionNodeAxisAngleToRotation")
zero_axis.inputs["Axis"].default_value = (0.0, 0.0, 0.0)
zero_axis.inputs["Angle"].default_value = 1.1
geometry = rotate_vector(root_group, geometry, zero_axis, (0.25, 1.5, -0.75), "zero_axis_rotation_vector")
axis_angle = root_group.nodes.new("FunctionNodeAxisAngleToRotation")
axis_angle.inputs["Axis"].default_value = (-1.0, 2.0, 3.0)
axis_angle.inputs["Angle"].default_value = 1.25
axis_angle_read = root_group.nodes.new("FunctionNodeRotationToAxisAngle")
root_group.links.new(axis_angle.outputs["Rotation"], axis_angle_read.inputs["Rotation"])
geometry = store(root_group, geometry, axis_angle_read, "Axis", "axis_angle_axis", "FLOAT_VECTOR")
geometry = store(root_group, geometry, axis_angle_read, "Angle", "axis_angle_angle", "FLOAT")

source_euler = root_group.nodes.new("FunctionNodeEulerToRotation")
source_euler.inputs["Euler"].default_value = (0.3, -0.4, 0.8)
euler_read = root_group.nodes.new("FunctionNodeRotationToEuler")
root_group.links.new(source_euler.outputs["Rotation"], euler_read.inputs["Rotation"])
geometry = store(root_group, geometry, euler_read, "Euler", "euler_rotation_roundtrip", "FLOAT_VECTOR")

quaternion = Quaternion((0.81, 0.2, -0.3, 0.45))
quaternion.normalize()
quaternion_rotation = root_group.nodes.new("FunctionNodeQuaternionToRotation")
for component, value in zip(("W", "X", "Y", "Z"), quaternion):
    quaternion_rotation.inputs[component].default_value = value
geometry = rotate_vector(root_group, geometry, quaternion_rotation, (1.0, -2.0, 0.75), "quaternion_rotation_vector")
quaternion_read = root_group.nodes.new("FunctionNodeRotationToQuaternion")
root_group.links.new(quaternion_rotation.outputs["Rotation"], quaternion_read.inputs["Rotation"])
for component in ("W", "X", "Y", "Z"):
    geometry = store(root_group, geometry, quaternion_read, component,
                     "quaternion_roundtrip_" + component.lower(), "FLOAT")
quaternion_euler = root_group.nodes.new("FunctionNodeRotationToEuler")
root_group.links.new(quaternion_rotation.outputs["Rotation"], quaternion_euler.inputs["Rotation"])
geometry = store(root_group, geometry, quaternion_euler, "Euler", "quaternion_rotation_euler", "FLOAT_VECTOR")
inverted_rotation = root_group.nodes.new("FunctionNodeInvertRotation")
root_group.links.new(quaternion_rotation.outputs["Rotation"], inverted_rotation.inputs["Rotation"])
geometry = rotate_vector(root_group, geometry, inverted_rotation, (1.0, -2.0, 0.75), "inverse_rotation_vector")

rotate_rotation = root_group.nodes.new("FunctionNodeRotateRotation")
rotate_rotation.rotation_space = "LOCAL"
root_group.links.new(quaternion_rotation.outputs["Rotation"], rotate_rotation.inputs["Rotation"])
root_group.links.new(axis_angle.outputs["Rotation"], rotate_rotation.inputs["Rotate By"])
geometry = rotate_vector(root_group, geometry, rotate_rotation, (0.5, 1.0, -1.5), "rotated_rotation_vector")
rotate_rotation_global = root_group.nodes.new("FunctionNodeRotateRotation")
root_group.links.new(quaternion_rotation.outputs["Rotation"], rotate_rotation_global.inputs["Rotation"])
root_group.links.new(axis_angle.outputs["Rotation"], rotate_rotation_global.inputs["Rotate By"])
geometry = rotate_vector(root_group, geometry, rotate_rotation_global,
                         (0.5, 1.0, -1.5), "rotated_rotation_global_vector")
rotate_euler = root_group.nodes.new("FunctionNodeRotateEuler")
rotate_euler.rotation_type = "EULER"
rotate_euler.space = "LOCAL"
rotate_euler.inputs["Rotation"].default_value = (0.2, -0.3, 0.4)
rotate_euler.inputs["Rotate By"].default_value = (-0.4, 0.1, 0.6)
geometry = store(root_group, geometry, rotate_euler, "Rotation", "rotated_euler", "FLOAT_VECTOR")
rotate_euler_axis_angle = root_group.nodes.new("FunctionNodeRotateEuler")
rotate_euler_axis_angle.rotation_type = "AXIS_ANGLE"
rotate_euler_axis_angle.space = "OBJECT"
rotate_euler_axis_angle.inputs["Rotation"].default_value = (0.2, -0.3, 0.4)
rotate_euler_axis_angle.inputs["Axis"].default_value = (0.0, 1.0, 0.0)
rotate_euler_axis_angle.inputs["Angle"].default_value = 0.35
geometry = store(root_group, geometry, rotate_euler_axis_angle, "Rotation", "rotated_euler_axis_angle", "FLOAT_VECTOR")

align_euler = root_group.nodes.new("FunctionNodeAlignEulerToVector")
align_euler.axis = "X"
align_euler.pivot_axis = "Z"
align_euler.inputs["Rotation"].default_value = (0.1, -0.2, 0.3)
align_euler.inputs["Factor"].default_value = 0.75
align_euler.inputs["Vector"].default_value = (0.25, 1.0, -0.5)
geometry = store(root_group, geometry, align_euler, "Rotation", "align_euler_to_vector", "FLOAT_VECTOR")
align_rotation = root_group.nodes.new("FunctionNodeAlignRotationToVector")
align_rotation.axis = "Y"
align_rotation.pivot_axis = "X"
align_rotation.inputs["Factor"].default_value = 0.6
align_rotation.inputs["Vector"].default_value = (1.0, 0.5, -0.25)
root_group.links.new(quaternion_rotation.outputs["Rotation"], align_rotation.inputs["Rotation"])
geometry = rotate_vector(root_group, geometry, align_rotation, (1.0, 2.0, 3.0), "align_rotation_to_vector")
axes_rotation = root_group.nodes.new("FunctionNodeAxesToRotation")
axes_rotation.inputs["Primary Axis"].default_value = (1.0, 0.0, 0.0)
axes_rotation.inputs["Secondary Axis"].default_value = (0.0, 1.0, 0.0)
geometry = rotate_vector(root_group, geometry, axes_rotation, (1.0, 2.0, 3.0), "axes_to_rotation_vector")

# Two translated, differently rotated matrices deliberately do not commute.
first_matrix = matrix_node(root_group, "First Noncommuting Transform", (
    (0.0, -1.0, 0.0, 2.0),
    (1.0, 0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0, 0.5),
    (0.0, 0.0, 0.0, 1.0),
))
second_matrix = matrix_node(root_group, "Second Noncommuting Transform", (
    (1.0, 0.0, 0.0, -0.5),
    (0.0, 0.0, -1.0, 1.0),
    (0.0, 1.0, 0.0, 2.0),
    (0.0, 0.0, 0.0, 1.0),
))
product = matrix_multiply(root_group, "First Times Second", first_matrix, second_matrix)
reverse_product = matrix_multiply(root_group, "Second Times First", second_matrix, first_matrix)
for label, node in (("matrix_product", product), ("matrix_reverse_product", reverse_product)):
    separated = root_group.nodes.new("FunctionNodeSeparateMatrix")
    root_group.links.new(node.outputs["Matrix"], separated.inputs["Matrix"])
    for row in range(1, 4):
        geometry = store(root_group, geometry, separated, f"Column 4 Row {row}",
                         f"{label}_translation_{row}", "FLOAT")

determinant = root_group.nodes.new("FunctionNodeMatrixDeterminant")
root_group.links.new(product.outputs["Matrix"], determinant.inputs["Matrix"])
geometry = store(root_group, geometry, determinant, "Determinant", "matrix_determinant", "FLOAT")
inverse = root_group.nodes.new("FunctionNodeInvertMatrix")
root_group.links.new(product.outputs["Matrix"], inverse.inputs["Matrix"])
geometry = store(root_group, geometry, inverse, "Invertible", "matrix_invertible", "BOOLEAN")
inverse_parts = root_group.nodes.new("FunctionNodeSeparateMatrix")
root_group.links.new(inverse.outputs["Matrix"], inverse_parts.inputs["Matrix"])
for row in range(1, 4):
    geometry = store(root_group, geometry, inverse_parts, f"Column 4 Row {row}",
                     f"inverse_translation_{row}", "FLOAT")
transpose = root_group.nodes.new("FunctionNodeTransposeMatrix")
root_group.links.new(product.outputs["Matrix"], transpose.inputs["Matrix"])
transpose_parts = root_group.nodes.new("FunctionNodeSeparateMatrix")
root_group.links.new(transpose.outputs["Matrix"], transpose_parts.inputs["Matrix"])
geometry = store(root_group, geometry, transpose_parts, "Column 1 Row 2", "transpose_off_diagonal", "FLOAT")
svd = root_group.nodes.new("FunctionNodeMatrixSVD")
root_group.links.new(product.outputs["Matrix"], svd.inputs["Matrix"])
geometry = store(root_group, geometry, svd, "S", "matrix_svd_values", "FLOAT_VECTOR")
for socket, name in (("U", "matrix_svd_u_determinant"), ("V", "matrix_svd_v_determinant")):
    svd_determinant = root_group.nodes.new("FunctionNodeMatrixDeterminant")
    root_group.links.new(svd.outputs[socket], svd_determinant.inputs["Matrix"])
    geometry = store(root_group, geometry, svd_determinant, "Determinant", name, "FLOAT")

perspective = matrix_node(root_group, "Perspective Projector", (
    (1.1, 0.2, 0.0, 2.0),
    (0.1, 0.9, 0.3, -1.0),
    (0.0, 0.0, 1.2, 0.5),
    (0.15, -0.05, 0.1, 1.0),
))
project = root_group.nodes.new("FunctionNodeProjectPoint")
root_group.links.new(position.outputs["Position"], project.inputs["Vector"])
root_group.links.new(perspective.outputs["Matrix"], project.inputs["Transform"])
geometry = store(root_group, geometry, project, "Vector", "projected_point", "FLOAT_VECTOR")
transform_direction = root_group.nodes.new("FunctionNodeTransformDirection")
transform_direction.inputs["Direction"].default_value = (0.5, -1.0, 2.0)
root_group.links.new(product.outputs["Matrix"], transform_direction.inputs["Transform"])
geometry = store(root_group, geometry, transform_direction, "Direction", "matrix_direction", "FLOAT_VECTOR")
transform_point = root_group.nodes.new("FunctionNodeTransformPoint")
root_group.links.new(position.outputs["Position"], transform_point.inputs["Vector"])
root_group.links.new(product.outputs["Matrix"], transform_point.inputs["Transform"])
geometry = store(root_group, geometry, transform_point, "Vector", "matrix_point", "FLOAT_VECTOR")

singular_matrix = matrix_node(root_group, "Singular Transform", (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))
singular_determinant = root_group.nodes.new("FunctionNodeMatrixDeterminant")
root_group.links.new(singular_matrix.outputs["Matrix"], singular_determinant.inputs["Matrix"])
geometry = store(root_group, geometry, singular_determinant, "Determinant", "singular_determinant", "FLOAT")
singular_inverse = root_group.nodes.new("FunctionNodeInvertMatrix")
root_group.links.new(singular_matrix.outputs["Matrix"], singular_inverse.inputs["Matrix"])
geometry = store(root_group, geometry, singular_inverse, "Invertible", "singular_invertible", "BOOLEAN")
singular_point = root_group.nodes.new("FunctionNodeTransformPoint")
singular_point.inputs["Vector"].default_value = (0.25, 0.5, -1.0)
root_group.links.new(singular_inverse.outputs["Matrix"], singular_point.inputs["Transform"])
geometry = store(root_group, geometry, singular_point, "Vector", "singular_inverse_point", "FLOAT_VECTOR")
singular_svd = root_group.nodes.new("FunctionNodeMatrixSVD")
root_group.links.new(singular_matrix.outputs["Matrix"], singular_svd.inputs["Matrix"])
geometry = store(root_group, geometry, singular_svd, "S", "singular_svd_values", "FLOAT_VECTOR")

# The transform path checks composed non-uniform scale/rotation/translation separately from
# the matrix path, including both point and direction semantics.
combined_transform = root_group.nodes.new("FunctionNodeCombineTransform")
combined_transform.inputs["Translation"].default_value = (1.1, -0.2, 0.6)
combined_transform.inputs["Scale"].default_value = (2.0, 0.75, 1.5)
root_group.links.new(axis_angle.outputs["Rotation"], combined_transform.inputs["Rotation"])
separate_transform = root_group.nodes.new("FunctionNodeSeparateTransform")
root_group.links.new(combined_transform.outputs["Transform"], separate_transform.inputs["Transform"])
geometry = store(root_group, geometry, separate_transform, "Translation", "separate_transform_translation", "FLOAT_VECTOR")
geometry = store(root_group, geometry, separate_transform, "Scale", "separate_transform_scale", "FLOAT_VECTOR")
separate_rotation_euler = root_group.nodes.new("FunctionNodeRotationToEuler")
root_group.links.new(separate_transform.outputs["Rotation"], separate_rotation_euler.inputs["Rotation"])
geometry = store(root_group, geometry, separate_rotation_euler, "Euler", "separate_transform_rotation", "FLOAT_VECTOR")
combined_direction = root_group.nodes.new("FunctionNodeTransformDirection")
combined_direction.inputs["Direction"].default_value = (0.5, -1.0, 2.0)
root_group.links.new(combined_transform.outputs["Transform"], combined_direction.inputs["Transform"])
geometry = store(root_group, geometry, combined_direction, "Direction", "combined_transform_direction", "FLOAT_VECTOR")
combined_point = root_group.nodes.new("FunctionNodeTransformPoint")
combined_point.inputs["Vector"].default_value = (0.5, -1.0, 2.0)
root_group.links.new(combined_transform.outputs["Transform"], combined_point.inputs["Transform"])
geometry = store(root_group, geometry, combined_point, "Vector", "combined_transform_point", "FLOAT_VECTOR")

root_group.links.new(geometry, root_output.inputs["Geometry"])
subject_mesh = bpy.data.meshes.new("GN Group Subject Mesh")
subject = bpy.data.objects.new("GN Group Subject", subject_mesh)
bpy.context.collection.objects.link(subject)
modifier = subject.modifiers.new("GN Group Parity", "NODES")
modifier.node_group = root_group

def plain(value):
    if isinstance(value, Matrix):
        return [[float(component) for component in row] for row in value]
    if isinstance(value, Euler):
        return [float(component) for component in value]
    if isinstance(value, Quaternion):
        return [float(component) for component in value]
    if isinstance(value, bpy.types.ID):
        return {"id_type": value.bl_rna.identifier, "name": value.name_full}
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [plain(component) for component in value]
    return value

depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = subject.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
attribute_types = {node.inputs["Name"].default_value: node.data_type
                   for node in root_group.nodes if node.bl_idname == "GeometryNodeStoreNamedAttribute"}
attributes = {}
for name, data_type in attribute_types.items():
    attribute = evaluated_mesh.attributes.get(name)
    if attribute is None or len(attribute.data) != 1:
        raise RuntimeError(f"missing single-point attribute {name}")
    value = attribute.data[0]
    if data_type == "FLOAT_VECTOR":
        attributes[name] = [float(component) for component in value.vector]
    elif data_type == "BOOLEAN":
        attributes[name] = bool(value.value)
    else:
        attributes[name] = float(value.value)
evaluated.to_mesh_clear()

interface_inputs = {item.name: item for item in resource_group.interface.items_tree
                    if item.item_type == "SOCKET" and item.in_out == "INPUT"}
resource_defaults = {}
for name in ("Object Default", "Collection Default", "Material Default", "Image Default"):
    value = interface_inputs[name].default_value
    resource_defaults[name] = {"socket_type": interface_inputs[name].socket_type,
                                "id_type": value.bl_rna.identifier,
                                "name": value.name_full}
    if name == "Image Default":
        resource_defaults[name]["pixels"] = [float(component) for component in value.pixels]
value_defaults = {}
for name in ("Matrix Default", "Rotation Default", "Menu Default", "String Default"):
    socket = interface_inputs[name]
    value_defaults[name] = {"socket_type": socket.socket_type,
                            "value": plain(socket.default_value)
                                     if hasattr(socket, "default_value") else None}
with (root / "blender_expected.json").open("w", encoding="utf-8") as stream:
    json.dump({"attributes": attributes, "resource_defaults": resource_defaults,
               "value_defaults": value_defaults}, stream, sort_keys=True)
bpy.ops.wm.save_as_mainfile(filepath=str(root / "geometry_node_groups.blend"))
"#;

const BLENDER_READBACK: &str = r#"
import bpy
import json
import sys
from mathutils import Euler, Matrix, Quaternion
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"geometry group roundtrip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "geometry_node_groups_roundtrip.blend"))
tree = bpy.data.node_groups["GN Resource Defaults"]
sockets = {item.name: item for item in tree.interface.items_tree
           if item.item_type == "SOCKET" and item.in_out == "INPUT"}
resources = [
    ("Object Default", "NodeSocketObject", bpy.data.objects, "GN Default Object", "Object"),
    ("Collection Default", "NodeSocketCollection", bpy.data.collections, "GN Default Collection", "Collection"),
    ("Material Default", "NodeSocketMaterial", bpy.data.materials, "GN Default Material", "Material"),
    ("Image Default", "NodeSocketImage", bpy.data.images, "GN Default Image", "Image"),
]
resource_defaults = {}
for name, socket_type, datablocks, resource_name, id_type in resources:
    socket = sockets[name]
    value = socket.default_value
    assert socket.socket_type == socket_type, (name, socket.socket_type)
    resource = datablocks[resource_name]
    assert value.as_pointer() == resource.as_pointer(), (name, value, resource)
    assert value.bl_rna.identifier == id_type, (name, value.bl_rna.identifier)
    resource_defaults[name] = {"socket_type": socket.socket_type,
                               "id_type": value.bl_rna.identifier,
                               "name": value.name_full}
    if name == "Image Default":
        resource_defaults[name]["pixels"] = [float(component) for component in value.pixels]

def plain(value):
    if isinstance(value, Matrix):
        return [[float(component) for component in row] for row in value]
    if isinstance(value, Euler):
        return [float(component) for component in value]
    if isinstance(value, Quaternion):
        return [float(component) for component in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [plain(component) for component in value]
    return value

value_defaults = {}
for name, socket_type in (("Matrix Default", "NodeSocketMatrix"),
                          ("Rotation Default", "NodeSocketRotation"),
                          ("Menu Default", "NodeSocketMenu"),
                          ("String Default", "NodeSocketString")):
    socket = sockets[name]
    assert socket.socket_type == socket_type, (name, socket.socket_type)
    value_defaults[name] = {"socket_type": socket.socket_type,
                            "value": plain(socket.default_value)
                                     if hasattr(socket, "default_value") else None}
duplicate_id = "gn_resource_defaults_duplicate"
duplicate = next(group for group in bpy.data.node_groups
                 if group.get("potter.id") == duplicate_id)
original = bpy.data.node_groups["GN Resource Defaults"]
assert original.get("potter.id") != duplicate_id
subject = bpy.data.objects["GN Group Subject"]
assert any(mod.type == "NODES" and mod.node_group == duplicate for mod in subject.modifiers)
with (root / "blender_roundtrip_defaults.json").open("w", encoding="utf-8") as stream:
    json.dump({"resource_defaults": resource_defaults,
               "value_defaults": value_defaults}, stream, sort_keys=True)
"#;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn assert_value_close(actual: &Value, expected: &Value, label: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}: Potter {actual} differs from Blender {expected} by more than {tolerance}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{label}: value shape");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_value_close(actual, expected, &format!("{label}[{index}]"));
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "{label}: object keys"
            );
            for (key, expected) in expected {
                assert_value_close(&actual[key], expected, &format!("{label}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{label}"),
    }
}

fn input_socket<'a>(group: &'a NodeGroup, name: &str) -> &'a GraphSocket {
    group
        .interface
        .inputs
        .iter()
        .find(|socket| socket.name == name)
        .unwrap()
}

fn assert_resource_default(
    doc: &SceneDoc,
    group: &NodeGroup,
    socket_name: &str,
    socket_type: &str,
    id_type: &str,
    resource_name: &str,
) {
    let socket = input_socket(group, socket_name);
    assert_eq!(socket.socket_type, socket_type, "{socket_name} socket type");
    let reference = &socket.default;
    assert_eq!(reference["id_type"], id_type, "{socket_name} resource type");
    let id = Id::new(reference["id"].as_str().unwrap()).unwrap();
    let name = match id_type {
        "Object" => doc.nodes.get(&id).map(|resource| resource.name.as_str()),
        "Collection" => doc
            .collections
            .get(&id)
            .map(|resource| resource.name.as_str()),
        "Material" => doc
            .materials
            .get(&id)
            .map(|resource| resource.name.as_str()),
        "Image" => doc.images.get(&id).map(|resource| resource.name.as_str()),
        _ => None,
    };
    assert_eq!(name, Some(resource_name), "{socket_name} ID reference");
}

#[test]
fn blender_geometry_group_rotation_matrix_parity_and_resource_defaults_roundtrip() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping geometry node group parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_geometry_node_groups.py",
        BLENDER_FIXTURE,
        root,
        "Blender geometry node group fixture failed",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;
    let imported = import_blend(
        &root.join("geometry_node_groups.blend"),
        Some(blender.as_path()),
    )?;
    let root_id = imported
        .doc
        .node_groups
        .iter()
        .find_map(|(id, group)| (group.name == "GN Group Rotation Matrix Root").then(|| id.clone()))
        .ok_or("imported root geometry node group is missing")?;
    let evaluation = graph::evaluate_with_node_groups(
        &root_id,
        &imported.doc.node_groups,
        None,
        &Map::new(),
        "Blender geometry node group parity",
    )?;
    let expected_attributes = expected["attributes"]
        .as_object()
        .ok_or("Blender did not emit geometry node attributes")?;
    assert!(
        expected_attributes.len() >= 40,
        "fixture must exercise every rotation/matrix node family"
    );
    for (name, expected) in expected_attributes {
        let attribute = evaluation.mesh.attributes.get(name).unwrap();
        assert_eq!(attribute["domain"], "POINT", "{name} domain");
        let values = attribute["data"].as_array().unwrap();
        assert_eq!(values.len(), 1, "{name} point count");
        assert_value_close(&values[0], expected, name);
    }
    assert_ne!(
        expected_attributes["matrix_product_translation_1"],
        expected_attributes["matrix_reverse_product_translation_1"],
        "fixture transforms must be noncommuting"
    );
    assert_value_close(
        &expected_attributes["identity_axis"],
        &json!([0.0, 1.0, 0.0]),
        "identity axis",
    );
    assert_value_close(
        &expected_attributes["identity_angle"],
        &json!(0.0),
        "identity angle",
    );
    assert_value_close(
        &expected_attributes["zero_axis_rotation_vector"],
        &json!([0.25, 1.5, -0.75]),
        "zero-axis rotation",
    );
    assert_eq!(expected_attributes["singular_invertible"], json!(false));
    assert_eq!(expected_attributes["matrix_invertible"], json!(true));

    let resource_group = imported
        .doc
        .node_groups
        .values()
        .find(|group| group.name == "GN Resource Defaults")
        .ok_or("imported typed-default group is missing")?;
    for (socket, socket_type, id_type, name) in [
        ("Object Default", "object", "Object", "GN Default Object"),
        (
            "Collection Default",
            "collection",
            "Collection",
            "GN Default Collection",
        ),
        (
            "Material Default",
            "material",
            "Material",
            "GN Default Material",
        ),
        ("Image Default", "image", "Image", "GN Default Image"),
    ] {
        assert_resource_default(
            &imported.doc,
            resource_group,
            socket,
            socket_type,
            id_type,
            name,
        );
    }
    for (socket_name, socket_type) in [
        ("Matrix Default", "matrix"),
        ("Rotation Default", "rotation"),
        ("Menu Default", "menu"),
        ("String Default", "string"),
    ] {
        assert_eq!(
            input_socket(resource_group, socket_name).socket_type,
            socket_type
        );
    }
    assert!(
        input_socket(resource_group, "Matrix Default")
            .default
            .is_null()
    );
    assert_value_close(
        &input_socket(resource_group, "Rotation Default").default,
        &json!([0.2, -0.4, 0.7]),
        "rotation interface default",
    );
    assert_eq!(
        input_socket(resource_group, "Menu Default").default,
        json!("MODE_B")
    );
    assert_eq!(
        input_socket(resource_group, "String Default").default,
        json!("nested group default")
    );

    let subject_id = imported
        .doc
        .nodes
        .iter()
        .find_map(|(id, node)| (node.name == "GN Group Subject").then(|| id.clone()))
        .ok_or("imported Geometry Nodes subject is missing")?;
    let duplicate_group_id = "gn_resource_defaults_duplicate";
    let duplicate_operations = json!([
        {
            "op":"graph.create",
            "id":duplicate_group_id,
            "name":"GN Resource Defaults",
            "kind":"geometry"
        },
        {
            "op":"graph.interface_update",
            "graph":duplicate_group_id,
            "set":{
                "inputs":[{"id":"geometry_in","name":"Geometry","socket_type":"geometry"}],
                "outputs":[{"id":"geometry_out","name":"Geometry","socket_type":"geometry"}]
            }
        },
        {"op":"graph.node_add","graph":duplicate_group_id,"id":"input","type":"NodeGroupInput"},
        {"op":"graph.node_add","graph":duplicate_group_id,"id":"output","type":"NodeGroupOutput"},
        {
            "op":"graph.link",
            "graph":duplicate_group_id,
            "from_node":"input",
            "from_socket":"geometry_in",
            "to_node":"output",
            "to_socket":"geometry_out"
        },
        {
            "op":"modifier.create",
            "target":{"id":subject_id.as_str()},
            "id":"duplicate_group_modifier",
            "type":"nodes",
            "params":{"node_group":duplicate_group_id}
        }
    ]);
    let duplicate_doc = ops::apply_batch(
        &imported.doc,
        &json!({
            "schema_version":1,
            "base_revision":imported.doc.revision,
            "operations":duplicate_operations
        }),
    )?
    .doc;
    let mut project = Project::init(root.join("project"))?;
    let asset_blobs = imported
        .assets
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    project.commit_import_with_assets(
        duplicate_doc.clone(),
        json!({"fixture":"geometry_node_groups"}),
        json!({}),
        &asset_blobs,
    )?;
    let roundtrip_blend = root.join("geometry_node_groups_roundtrip.blend");
    export_blend(
        &duplicate_doc,
        &project,
        &roundtrip_blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &EvaluationContext::default(),
        },
    )?;
    run_blender_script(
        &blender,
        "read_geometry_node_group_defaults.py",
        BLENDER_READBACK,
        root,
        "Blender geometry node group resource-default readback failed",
    )?;
    let actual_defaults: Value =
        serde_json::from_slice(&fs::read(root.join("blender_roundtrip_defaults.json"))?)?;
    assert_eq!(
        actual_defaults["resource_defaults"],
        expected["resource_defaults"]
    );
    assert_value_close(
        &actual_defaults["value_defaults"],
        &expected["value_defaults"],
        "round-tripped interface defaults",
    );
    Ok(())
}
