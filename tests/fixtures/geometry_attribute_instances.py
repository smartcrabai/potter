# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate Blender 5.2.2 Geometry Attribute INSTANCE-domain oracles."""
import json
import os
import sys

import bpy
from mathutils import Euler, Quaternion

if bpy.app.version[:3] != (5, 2, 2):
    raise RuntimeError("this oracle requires Blender 5.2.2")

root = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

mix_modes = ["REPLACE", "BEFORE_FULL", "BEFORE_SPLIT", "AFTER_FULL", "AFTER_SPLIT"]
sample_indices = [0, 2, 999]
vector_name = "geometry_attribute_instance_vector"
float_name = "geometry_attribute_instance_float"
quaternion_name = "geometry_attribute_instance_quaternion"
int_name = "geometry_attribute_instance_int"
boolean_name = "geometry_attribute_instance_boolean"
float2_name = "geometry_attribute_instance_float2"
float_color_name = "geometry_attribute_instance_float_color"
byte_color_name = "geometry_attribute_instance_byte_color"
vertex_group_name = "geometry_attribute_instance_weight"
cases = []
guards = []


def matrix_columns(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def evaluated_mesh_counts(obj):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return {
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
        }
    finally:
        evaluated.to_mesh_clear()

def evaluated_mesh_positions(obj):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return sorted(
            [[float(component) for component in vertex.co] for vertex in mesh.vertices],
            key=lambda position: tuple(position),
        )
    finally:
        evaluated.to_mesh_clear()


def evaluated_attribute_values(obj, name):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.attributes.get(name)
        if attribute is None:
            return None
        if attribute.data_type == "FLOAT_VECTOR":
            return [
                [float(component) for component in item.vector]
                for item in attribute.data
            ]
        raise RuntimeError("unsupported oracle attribute type %s" % attribute.data_type)
    finally:
        evaluated.to_mesh_clear()





def make_mesh(name="GeometryAttributeInstancesMesh", with_face=False, with_string=False,
              with_face_attribute=True):
    mesh = bpy.data.meshes.new(name)
    vertices = [(1.0, 2.0, 3.0), (-2.0, 1.0, 0.5), (0.25, 4.0, -1.0)]
    faces = [(0, 1, 2)] if with_face else []
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    vector = mesh.attributes.new(vector_name, "FLOAT_VECTOR", "POINT")
    scalar = mesh.attributes.new(float_name, "FLOAT", "POINT")
    quaternion = mesh.attributes.new(quaternion_name, "QUATERNION", "POINT")
    integer = mesh.attributes.new(int_name, "INT", "POINT")
    boolean = mesh.attributes.new(boolean_name, "BOOLEAN", "POINT")
    float2 = mesh.attributes.new(float2_name, "FLOAT2", "POINT")
    float_color = mesh.color_attributes.new(float_color_name, "FLOAT_COLOR", "POINT")
    byte_color = mesh.color_attributes.new(byte_color_name, "BYTE_COLOR", "POINT")
    vector_values = [(0.7, -1.2, 2.3), (-2.4, 0.6, 1.1), (3.2, 1.7, -0.8)]
    scalar_values = [0.35, -0.8, 1.45]
    quaternion_values = [
        Quaternion(Euler((0.31, -0.22, 0.47), "XYZ")),
        Quaternion(Euler((-0.18, 0.63, 0.29), "XYZ")),
        Quaternion(Euler((0.52, 0.14, -0.36), "XYZ")),
    ]
    integer_values = [1, 3, 5]
    boolean_values = [True, False, True]
    float2_values = [(0.42, -0.37), (-0.25, 0.65), (1.1, -0.9)]
    float_color_values = [(0.25, 0.5, 0.75, 1.0), (0.4, 0.2, 0.8, 1.0), (0.7, 0.3, 0.2, 1.0)]
    byte_color_values = [(0.3, 0.6, 0.1, 1.0), (0.8, 0.25, 0.5, 1.0), (0.15, 0.45, 0.9, 1.0)]
    for index in range(len(vertices)):
        vector.data[index].vector = vector_values[index]
        scalar.data[index].value = scalar_values[index]
        quaternion.data[index].value = quaternion_values[index]
        integer.data[index].value = integer_values[index]
        boolean.data[index].value = boolean_values[index]
        float2.data[index].vector = float2_values[index]
        float_color.data[index].color = float_color_values[index]
        byte_color.data[index].color = byte_color_values[index]
    if with_face and with_face_attribute:
        face_scalar = mesh.attributes.new("geometry_attribute_instance_face", "FLOAT", "FACE")
        face_scalar.data[0].value = 0.625
    if with_string:
        string_attribute = mesh.attributes.new("geometry_attribute_instance_string", "STRING", "POINT")
        for index, value in enumerate((b"sample-zero", b"sample-one", b"sample-two")):
            string_attribute.data[index].value = value
    return mesh

def clean_mesh(name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(1.0, 2.0, 3.0), (-2.0, 1.0, 0.5), (0.25, 4.0, -1.0)], [], [])
    mesh.update()
    return mesh


def geometry_group(name, extra_geometry_input=False):
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    if extra_geometry_input:
        group.interface.new_socket(name="Extra Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return group, group.nodes.new("NodeGroupInput"), group.nodes.new("NodeGroupOutput")


def instance_on_points(group, points, instance, rotation, scale, name, pick=False):
    node = group.nodes.new("GeometryNodeInstanceOnPoints")
    node.label = name
    group.links.new(points, node.inputs["Points"])
    group.links.new(instance, node.inputs["Instance"])
    node.inputs["Rotation"].default_value = rotation
    node.inputs["Scale"].default_value = scale
    node.inputs["Pick Instance"].default_value = pick
    return node


def add_transform(group, source, translation, rotation, scale):
    node = group.nodes.new("GeometryNodeTransform")
    group.links.new(source, node.inputs["Geometry"])
    node.inputs["Translation"].default_value = translation
    node.inputs["Rotation"].default_value = rotation
    node.inputs["Scale"].default_value = scale
    return node


def new_target(name, mesh=None, vertex_group=False):
    target = bpy.data.objects.new(name, mesh if mesh is not None else make_mesh())
    scene.collection.objects.link(target)
    target.location = (0.7, -1.1, 2.3)
    target.rotation_mode = "QUATERNION"
    target.rotation_quaternion = Quaternion(Euler((0.28, -0.19, 0.42), "XYZ"))
    target.scale = (1.3, 0.75, 1.6)
    if vertex_group:
        group = target.vertex_groups.new(name=vertex_group_name)
        for index, weight in enumerate((0.2, 0.55, 0.9)):
            group.add([index], weight, "REPLACE")
    return target


def attach_group(target, group, name="InstanceGeometry"):
    modifier = target.modifiers.new(name, "NODES")
    modifier.node_group = group
    return modifier


def basic_group(name, rotation=(0.36, -0.27, 0.51), scale=(1.25, 0.7, 1.45)):
    group, input_node, output_node = geometry_group(name)
    instances = instance_on_points(
        group,
        input_node.outputs["Geometry"],
        input_node.outputs["Geometry"],
        rotation,
        scale,
        "Nonidentity Instance on Points",
    )
    transformed = add_transform(
        group,
        instances.outputs["Instances"],
        (0.25, -0.4, 0.75),
        (-0.23, 0.41, 0.32),
        (0.85, 1.2, 0.65),
    )
    group.links.new(transformed.outputs["Geometry"], output_node.inputs["Geometry"])
    return group


def make_owner_case(target, attribute, data_type, index, mix_mode, name, flags=None, finding=None):
    flags = flags or {"mix_loc": True, "mix_rot": True, "mix_scl": True}
    owner = bpy.data.objects.new(name, None)
    scene.collection.objects.link(owner)
    owner.location = (-0.8, 1.2, 2.5)
    owner.rotation_mode = "QUATERNION"
    owner.rotation_quaternion = Quaternion(Euler((0.16, 0.37, -0.25), "XYZ"))
    owner.scale = (0.8, 1.2, 1.1)
    baseline = matrix_columns(owner.matrix_basis)
    constraint = owner.constraints.new("GEOMETRY_ATTRIBUTE")
    constraint.target = target
    constraint.attribute_name = attribute
    constraint.data_type = data_type
    constraint.domain = "INSTANCE"
    constraint.sample_index = index
    constraint.mix_mode = mix_mode
    constraint.apply_target_transform = True
    constraint.mix_loc = flags["mix_loc"]
    constraint.mix_rot = flags["mix_rot"]
    constraint.mix_scl = flags["mix_scl"]
    case = {
        "id": name,
        "target": target.name,
        "attribute": attribute,
        "data_type": data_type,
        "domain": "INSTANCE",
        "sample_index": index,
        "mix_mode": mix_mode,
        "mix_loc": flags["mix_loc"],
        "mix_rot": flags["mix_rot"],
        "mix_scl": flags["mix_scl"],
        "baseline": baseline,
    }
    if finding:
        case["review_finding"] = finding
    cases.append(case)
    return owner


target = new_target("GeometryAttributeInstancesTarget")
attach_group(target, basic_group("GeometryAttributeInstances"))

# Each supported native/sample conversion covers every mix mode, two real indices, and clamping.
for data_type, attribute in [
    ("FLOAT4X4", "instance_transform"),
    ("QUATERNION", "instance_transform"),
    ("QUATERNION", quaternion_name),
    ("QUATERNION", vector_name),
    ("QUATERNION", float_name),
    ("QUATERNION", float2_name),
    ("VECTOR", vector_name),
    ("VECTOR", float_name),
    ("VECTOR", quaternion_name),
    ("VECTOR", int_name),
    ("VECTOR", boolean_name),
    ("VECTOR", float2_name),
    ("VECTOR", float_color_name),
    ("VECTOR", byte_color_name),
    ("FLOAT4X4", quaternion_name),
]:
    for index in sample_indices:
        for mix_mode in mix_modes:
            make_owner_case(
                target,
                attribute,
                data_type,
                index,
                mix_mode,
                "owner_%s_%s_%s_%s" % (
                    data_type.lower(),
                    attribute.rsplit("geometry_attribute_instance_", 1)[-1],
                    index,
                    mix_mode.lower(),
                ),
                finding="F4" if attribute == float_name else None,
            )

# Exercise each independent transform-component switch with a non-default mix mode.
for name, mix_mode, flags in [
    ("owner_partial_mix_loc", "REPLACE", {"mix_loc": False, "mix_rot": True, "mix_scl": True}),
    ("owner_partial_mix_rot", "BEFORE_SPLIT", {"mix_loc": True, "mix_rot": False, "mix_scl": True}),
    ("owner_partial_mix_scl", "AFTER_SPLIT", {"mix_loc": True, "mix_rot": True, "mix_scl": False}),
]:
    make_owner_case(target, "instance_transform", "FLOAT4X4", 1, mix_mode, name, flags)

# A's references are nested when IoP B uses A as its Instance input; only B has top-level transforms.
nested_group, nested_input, nested_output = geometry_group("NestedSourceInstanceGeometry")
source_instances = instance_on_points(
    nested_group, nested_input.outputs["Geometry"], nested_input.outputs["Geometry"],
    (-0.51, 0.18, 0.37), (1.4, 0.65, 1.1), "Nested source IoP A",
)
nested_output_node = instance_on_points(
    nested_group,
    nested_input.outputs["Geometry"],
    source_instances.outputs["Instances"],
    (0.27, -0.33, 0.44),
    (1.15, 0.9, 1.35),
    "Nested Instance on Points B",
)
nested_group.links.new(nested_output_node.outputs["Instances"], nested_output.inputs["Geometry"])
nested_target = new_target("NestedInstancesTarget")
attach_group(nested_target, nested_group)
for index in (0, 1, 2, 999):
    for mix_mode in mix_modes:
        make_owner_case(
            nested_target, "instance_transform", "FLOAT4X4", index, mix_mode,
            "owner_nested_source_%s_%s" % (index, mix_mode.lower()), finding="F1",
        )


# Join preserves both input instance arrays in order.
join_group, join_input, join_output = geometry_group("JoinedInstanceGeometry")
join_left = instance_on_points(
    join_group, join_input.outputs["Geometry"], join_input.outputs["Geometry"],
    (0.18, -0.36, 0.22), (0.9, 1.15, 0.7), "Join left",
)
join_right = instance_on_points(
    join_group, join_input.outputs["Geometry"], join_input.outputs["Geometry"],
    (-0.42, 0.31, 0.57), (1.35, 0.8, 1.2), "Join right",
)
join_node = join_group.nodes.new("GeometryNodeJoinGeometry")
interleaved_a = join_group.nodes.new("GeometryNodeTransform")
interleaved_a.inputs["Translation"].default_value = (0.1, 0.0, 0.0)
interleaved_b = join_group.nodes.new("GeometryNodeTransform")
interleaved_b.inputs["Translation"].default_value = (0.0, 0.2, 0.0)
join_left_link = join_group.links.new(join_left.outputs["Instances"], join_node.inputs["Geometry"])
join_group.links.new(join_input.outputs["Geometry"], interleaved_a.inputs["Geometry"])
join_right_link = join_group.links.new(join_right.outputs["Instances"], join_node.inputs["Geometry"])
join_group.links.new(join_input.outputs["Geometry"], interleaved_b.inputs["Geometry"])
# Change the sort IDs without changing tree.links order, as Blender's multi-input UI does.
join_left_link.swap_multi_input_sort_id(join_right_link)
join_group.links.new(join_node.outputs["Geometry"], join_output.inputs["Geometry"])
join_target = new_target("JoinedInstancesTarget")
attach_group(join_target, join_group)
for index, mix_mode in zip((0, 3, 999), ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(join_target, "instance_transform", "FLOAT4X4", index, mix_mode, "owner_join_%s" % index)

# Realize removes the instance domain; the consumer must still evaluate without a refusal.
realize_group, realize_input, realize_output = geometry_group("RealizedInstanceGeometry")
realize_instances = instance_on_points(
    realize_group, realize_input.outputs["Geometry"], realize_input.outputs["Geometry"],
    (0.36, -0.24, 0.19), (1.2, 0.85, 1.4), "Before realization",
)
realize_node = realize_group.nodes.new("GeometryNodeRealizeInstances")
realize_group.links.new(realize_instances.outputs["Instances"], realize_node.inputs["Geometry"])
realize_group.links.new(realize_node.outputs["Geometry"], realize_output.inputs["Geometry"])
realize_target = new_target("RealizedInstancesTarget", clean_mesh("RealizedInstancesMesh"))
attach_group(realize_target, realize_group)
make_owner_case(
    realize_target, "instance_transform", "FLOAT4X4", 0, "REPLACE", "owner_realized",
)

# Realize's output is a mesh and remains valid as Points for a later Instance on Points.
realized_points_group, realized_points_input, realized_points_output = geometry_group(
    "RealizedPointsThenInstanceGeometry",
)
realized_points_first = instance_on_points(
    realized_points_group,
    realized_points_input.outputs["Geometry"],
    realized_points_input.outputs["Geometry"],
    (0.2, -0.3, 0.4),
    (1.1, 0.8, 1.3),
    "Before realize",
)
realized_points_node = realized_points_group.nodes.new("GeometryNodeRealizeInstances")
realized_points_group.links.new(
    realized_points_first.outputs["Instances"],
    realized_points_node.inputs["Geometry"],
)
realized_points_second = instance_on_points(
    realized_points_group,
    realized_points_node.outputs["Geometry"],
    realized_points_input.outputs["Geometry"],
    (-0.4, 0.25, 0.3),
    (0.9, 1.2, 0.7),
    "After realize",
)
realized_points_group.links.new(
    realized_points_second.outputs["Instances"],
    realized_points_output.inputs["Geometry"],
)
realized_points_target = new_target(
    "RealizedPointsThenInstanceTarget",
    clean_mesh("RealizedPointsThenInstanceMesh"),
)
attach_group(realized_points_target, realized_points_group)
make_owner_case(
    realized_points_target,
    "instance_transform",
    "FLOAT4X4",
    1,
    "REPLACE",
    "owner_realized_then_instance",
)

# The modifier stack passes the first modifier's instances to the second Geometry input.
stack_target = new_target("StackedInstancesTarget")
first_group, first_input, first_output = geometry_group("StackedInstancesFirst")
first_iop = instance_on_points(
    first_group, first_input.outputs["Geometry"], first_input.outputs["Geometry"],
    (0.25, 0.17, -0.39), (0.9, 1.25, 0.8), "Stack source",
)
first_group.links.new(first_iop.outputs["Instances"], first_output.inputs["Geometry"])
attach_group(stack_target, first_group, "CreateInstances")
second_group, second_input, second_output = geometry_group("StackedInstancesSecond")
second_transform = add_transform(
    second_group, second_input.outputs["Geometry"], (0.15, -0.2, 0.65),
    (0.31, 0.22, -0.47), (1.1, 0.7, 1.3),
)
second_group.links.new(second_transform.outputs["Geometry"], second_output.inputs["Geometry"])
attach_group(stack_target, second_group, "TransformExistingInstances")
for index, mix_mode in zip(sample_indices, ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(
        stack_target, "instance_transform", "FLOAT4X4", index, mix_mode,
        "owner_stacked_%s" % index, finding="F2",
    )

# Set Position modifies each top-level instance translation.
set_position_group, position_input, position_output = geometry_group("SetPositionOnInstances")
position_instances = instance_on_points(
    set_position_group, position_input.outputs["Geometry"], position_input.outputs["Geometry"],
    (0.21, -0.44, 0.35), (1.3, 0.75, 1.1), "Before Set Position",
)
position_node = set_position_group.nodes.new("GeometryNodeSetPosition")
set_position_group.links.new(position_instances.outputs["Instances"], position_node.inputs["Geometry"])
position_node.inputs["Offset"].default_value = (0.35, -0.6, 0.9)
set_position_group.links.new(position_node.outputs["Geometry"], position_output.inputs["Geometry"])
position_target = new_target("SetPositionInstancesTarget")
attach_group(position_target, set_position_group)
for index, mix_mode in zip(sample_indices, ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(
        position_target, "instance_transform", "FLOAT4X4", index, mix_mode,
        "owner_set_position_%s" % index, finding="F3",
    )

# Distribute Points on Faces forwards the original top-level Instances component.
distribute_group, distribute_input, distribute_output = geometry_group("DistributePointsForwardsInstances")
distribute_instances = instance_on_points(
    distribute_group,
    distribute_input.outputs["Geometry"],
    distribute_input.outputs["Geometry"],
    (0.28, -0.17, 0.39),
    (1.2, 0.85, 1.35),
    "Instances before Distribute Points on Faces",
)
distribute_node = distribute_group.nodes.new("GeometryNodeDistributePointsOnFaces")
distribute_node.inputs["Density"].default_value = 1.0
distribute_group.links.new(distribute_instances.outputs["Instances"], distribute_node.inputs["Mesh"])
distribute_group.links.new(distribute_node.outputs["Points"], distribute_output.inputs["Geometry"])
distribute_target = new_target(
    "DistributePointsInstancesTarget",
    make_mesh("DistributePointsInstancesMesh", with_face=True, with_face_attribute=False),
)
attach_group(distribute_target, distribute_group)
for index, mix_mode in zip(sample_indices, ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(
        distribute_target, "instance_transform", "FLOAT4X4", index, mix_mode,
        "owner_distribute_points_%s" % index, finding="F3",
    )
distribute_control_group, distribute_control_input, distribute_control_output = geometry_group(
    "DistributePointsInstanceControl",
)
distribute_control_iop = instance_on_points(
    distribute_control_group,
    distribute_control_input.outputs["Geometry"],
    distribute_control_input.outputs["Geometry"],
    (0.28, -0.17, 0.39),
    (1.2, 0.85, 1.35),
    "Control IoP without Distribute Points on Faces",
)
distribute_control_group.links.new(
    distribute_control_iop.outputs["Instances"],
    distribute_control_output.inputs["Geometry"],
)
distribute_control_target = new_target(
    "DistributePointsControlTarget",
    make_mesh("DistributePointsControlMesh", with_face=True, with_face_attribute=False),
)
attach_group(distribute_control_target, distribute_control_group)
for index, mix_mode in zip(sample_indices, ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(
        distribute_control_target, "instance_transform", "FLOAT4X4", index, mix_mode,
        "owner_distribute_control_%s" % index,
    )

# Vertex groups are adapted from mesh points and propagated to the resulting instances.
vertex_target = new_target("VertexGroupInstancesTarget", vertex_group=True)
attach_group(vertex_target, basic_group("VertexGroupInstanceGeometry"))
for index, mix_mode in zip(sample_indices, ("REPLACE", "BEFORE_SPLIT", "AFTER_FULL")):
    make_owner_case(
        vertex_target, vertex_group_name, "VECTOR", index, mix_mode,
        "owner_vertex_group_%s" % index, finding="F5",
    )

# Only interface geometry input 0 receives the object's geometry. The second is empty.
second_input_group, second_input_node, second_output = geometry_group(
    "SecondGeometryInputIsEmpty", extra_geometry_input=True,
)
second_input_iop = instance_on_points(
    second_input_group,
    second_input_node.outputs["Geometry"],
    second_input_node.outputs["Extra Geometry"],
    (0.19, -0.23, 0.4),
    (1.2, 0.8, 1.1),
    "Second geometry socket as Instance",
)
second_realize = second_input_group.nodes.new("GeometryNodeRealizeInstances")
second_input_group.links.new(second_input_iop.outputs["Instances"], second_realize.inputs["Geometry"])
second_join = second_input_group.nodes.new("GeometryNodeJoinGeometry")
second_input_group.links.new(second_input_node.outputs["Geometry"], second_join.inputs["Geometry"])
second_input_group.links.new(second_input_iop.outputs["Instances"], second_join.inputs["Geometry"])
second_input_group.links.new(second_realize.outputs["Geometry"], second_join.inputs["Geometry"])
second_input_group.links.new(second_join.outputs["Geometry"], second_output.inputs["Geometry"])
second_input_target = new_target("SecondGeometryInputTarget", clean_mesh("SecondGeometryInputMesh"))
attach_group(second_input_target, second_input_group)
make_owner_case(
    second_input_target, "instance_transform", "FLOAT4X4", 0, "REPLACE",
    "owner_second_geometry_input", finding="F6",
)

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
targets = [
    target,
    nested_target,
    join_target,
    realize_target,
    realized_points_target,
    stack_target,
    position_target,
    distribute_target,
    distribute_control_target,
    vertex_target,
    second_input_target,
]
expected = {
    "cases": cases,
    "target_geometry": {
        item.name: evaluated_mesh_counts(item)
        for item in targets
    },
    "matrices": {
        case["id"]: matrix_columns(bpy.data.objects[case["id"]].evaluated_get(depsgraph).matrix_world)
        for case in cases
    },
    "join_link_oracle": {
        "sorted_multi_input": sorted(
            [
                {
                    "source_scale": [
                        float(component)
                        for component in link.from_node.inputs["Scale"].default_value
                    ],
                    "sort_id": int(link.multi_input_sort_id),
                }
                for link in join_group.links
                if link.to_node == join_node
            ],
            key=lambda item: item["sort_id"],
            reverse=True,
        ),
        "ordinary_transform_order": [
            [
                float(component)
                for component in link.to_node.inputs["Translation"].default_value
            ]
            for link in join_group.links
            if link.to_node.bl_idname == "GeometryNodeTransform"
        ],
    },
}
os.makedirs(root, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "geometry_attribute_instances.blend"))
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, sort_keys=True)


def clear_generated_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.node_groups, bpy.data.meshes):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def save_guard(
    name,
    feature_id,
    build,
    constraint_domain="INSTANCE",
    constraint_data_type=None,
    oracle_attribute=None,
):
    clear_generated_scene()
    target, attribute, data_type, sample_index = build()
    owner = bpy.data.objects.new("guard_owner", None)
    scene.collection.objects.link(owner)
    constraint = owner.constraints.new("GEOMETRY_ATTRIBUTE")
    constraint.target = target
    constraint.attribute_name = attribute
    constraint.data_type = constraint_data_type or data_type
    constraint.domain = constraint_domain
    constraint.sample_index = sample_index
    constraint.mix_mode = "REPLACE"
    constraint.apply_target_transform = True
    constraint.mix_loc = True
    constraint.mix_rot = True
    constraint.mix_scl = True
    scene.frame_set(1)
    bpy.context.view_layer.update()
    global depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "guard_%s.blend" % name))
    guard = {
        "id": name,
        "feature_id": feature_id,
        "inspect_id": owner.name,
    }
    if oracle_attribute is not None:
        values = evaluated_attribute_values(target, oracle_attribute)
        if not values:
            raise RuntimeError("Blender oracle omitted propagated attribute %s" % oracle_attribute)
        guard["oracle_attribute_values"] = values
    guards.append(guard)


def store_then_realize_attribute(target, name):
    group, input_node, output_node = geometry_group(name)
    instances = instance_on_points(
        group,
        input_node.outputs["Geometry"],
        input_node.outputs["Geometry"],
        (0.2, -0.3, 0.4),
        (1.1, 0.8, 1.3),
        "Store INSTANCE attribute source",
    )
    store = group.nodes.new("GeometryNodeStoreNamedAttribute")
    store.domain = "INSTANCE"
    store.data_type = "FLOAT_VECTOR"
    store.inputs["Name"].default_value = "realized_instance_vector"
    store.inputs["Value"].default_value = (0.5, -1.25, 2.0)
    group.links.new(instances.outputs["Instances"], store.inputs["Geometry"])
    realize = group.nodes.new("GeometryNodeRealizeInstances")
    group.links.new(store.outputs["Geometry"], realize.inputs["Geometry"])
    group.links.new(realize.outputs["Geometry"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target


def guard_realize_attribute_columns():
    target = new_target("GuardRealizeAttributeColumnsTarget")
    store_then_realize_attribute(target, "GuardRealizeAttributeColumns")
    return target, "realized_instance_vector", "VECTOR", 0


def save_plain_case(name, feature_id, build, oracle_attribute=None):
    clear_generated_scene()
    target = build()
    scene.frame_set(1)
    bpy.context.view_layer.update()
    global depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "guard_%s.blend" % name))
    guard = {
        "id": name,
        "feature_id": feature_id,
        "inspect_id": target.name,
        "oracle_positions": evaluated_mesh_positions(target),
    }
    if oracle_attribute is not None:
        values = evaluated_attribute_values(target, oracle_attribute)
        if not values:
            raise RuntimeError("Blender oracle omitted propagated attribute %s" % oracle_attribute)
        guard["oracle_attribute_values"] = values
    guards.append(guard)


def guard_store_named_attribute_uncaptured():
    target = new_target("GuardStoreNamedAttributeUncapturedTarget")
    store_then_realize_attribute(target, "GuardStoreNamedAttributeUncaptured")
    return target


def guard_transform_matrix_plain_mesh():
    target = new_target("GuardTransformMatrixPlainMeshTarget")
    group, input_node, output_node = geometry_group("GuardTransformMatrixPlainMesh")
    transform = group.nodes.new("GeometryNodeTransform")
    transform.inputs["Translation"].default_value = (1.0, 0.0, 0.0)
    transform.inputs["Mode"].default_value = "Matrix"
    group.links.new(input_node.outputs["Geometry"], transform.inputs["Geometry"])
    group.links.new(transform.outputs["Geometry"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target


def guard_picked_instance():
    target = new_target("GuardPickedInstanceTarget")
    group, input_node, output_node = geometry_group("GuardPickedInstance")
    instances = instance_on_points(
        group, input_node.outputs["Geometry"], input_node.outputs["Geometry"],
        (0.2, 0.3, -0.4), (1.2, 0.8, 1.1), "Picked Instance", pick=True,
    )
    group.links.new(instances.outputs["Instances"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target, "instance_transform", "FLOAT4X4", 0


def guard_realized_points():
    target = new_target("GuardRealizedPointsTarget")
    group, input_node, output_node = geometry_group("GuardRealizedPoints")
    first = instance_on_points(
        group, input_node.outputs["Geometry"], input_node.outputs["Geometry"],
        (0.2, 0.3, -0.4), (1.2, 0.8, 1.1), "Points-side IoP A",
    )
    points = group.nodes.new("GeometryNodeJoinGeometry")
    group.links.new(input_node.outputs["Geometry"], points.inputs["Geometry"])
    group.links.new(first.outputs["Instances"], points.inputs["Geometry"])
    second = instance_on_points(
        group, points.outputs["Geometry"], input_node.outputs["Geometry"],
        (-0.3, 0.1, 0.5), (0.9, 1.3, 0.7), "Points-side IoP B",
    )
    group.links.new(second.outputs["Instances"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target, "instance_transform", "FLOAT4X4", 0

def guard_non_point_attribute():
    target = new_target("GuardNonPointAttributeTarget", make_mesh("GuardNonPointAttributeMesh", True))
    group, input_node, output_node = geometry_group("GuardNonPointAttribute")
    instances = instance_on_points(
        group, input_node.outputs["Geometry"], input_node.outputs["Geometry"],
        (0.2, 0.3, -0.4), (1.2, 0.8, 1.1), "Non-point attribute source",
    )
    group.links.new(instances.outputs["Instances"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target, "geometry_attribute_instance_face", "VECTOR", 0


def guard_attribute_conversion():
    target = new_target(
        "GuardAttributeConversionTarget",
        make_mesh("GuardAttributeConversionMesh", with_string=True),
    )
    group, input_node, output_node = geometry_group("GuardAttributeConversion")
    instances = instance_on_points(
        group, input_node.outputs["Geometry"], input_node.outputs["Geometry"],
        (0.2, 0.3, -0.4), (1.2, 0.8, 1.1), "Unsupported STRING conversion",
    )
    group.links.new(instances.outputs["Instances"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target, "geometry_attribute_instance_string", "VECTOR", 0


def guard_transform_matrix_mode():
    target = new_target("GuardTransformMatrixModeTarget")
    group, input_node, output_node = geometry_group("GuardTransformMatrixMode")
    instances = instance_on_points(
        group, input_node.outputs["Geometry"], input_node.outputs["Geometry"],
        (0.2, 0.3, -0.4), (1.2, 0.8, 1.1), "Before matrix-mode Transform",
    )
    transform = group.nodes.new("GeometryNodeTransform")
    transform.inputs["Mode"].default_value = "Matrix"
    group.links.new(instances.outputs["Instances"], transform.inputs["Geometry"])
    group.links.new(transform.outputs["Geometry"], output_node.inputs["Geometry"])
    attach_group(target, group)
    return target, "instance_transform", "FLOAT4X4", 0


def save_fresh_session_case():
    clear_generated_scene()
    target = new_target("FreshSessionInstanceTarget")
    group, input_node, output_node = geometry_group("FreshSessionInstanceGeometry")
    instances = instance_on_points(
        group,
        input_node.outputs["Geometry"],
        input_node.outputs["Geometry"],
        (0.4, -0.2, 0.3),
        (1.2, 0.75, 1.1),
        "Fresh session instance source",
    )
    group.links.new(instances.outputs["Instances"], output_node.inputs["Geometry"])
    attach_group(target, group)
    owner = new_target("FreshSessionConstraintOwner", make_mesh("FreshSessionOwnerMesh", True))
    constraint = owner.constraints.new("GEOMETRY_ATTRIBUTE")
    constraint.target = target
    constraint.attribute_name = "instance_transform"
    constraint.data_type = "FLOAT4X4"
    constraint.domain = "INSTANCE"
    constraint.sample_index = 1
    constraint.mix_mode = "REPLACE"
    constraint.apply_target_transform = True
    constraint.mix_loc = True
    constraint.mix_rot = True
    constraint.mix_scl = True
    new_target("FreshSessionJoinDestination", make_mesh("FreshSessionDestinationMesh", True))
    scene.frame_set(1)
    bpy.context.view_layer.update()
    global depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {
        "owner_baseline": matrix_columns(owner.matrix_basis),
        "owner_matrix": matrix_columns(owner.evaluated_get(depsgraph).matrix_world),
    }
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "geometry_attribute_instances_fresh_session.blend"))
    return result


save_guard(
    "attribute_conversion",
    "constraint.geometry_attribute.instances.attribute_conversion",
    guard_attribute_conversion,
)
save_guard(
    "picked_instance",
    "constraint.geometry_attribute.instances.picked_instance",
    guard_picked_instance,
)
save_guard(
    "realized_points",
    "constraint.geometry_attribute.instances.realized_points",
    guard_realized_points,
)
save_guard(
    "non_point_attribute",
    "constraint.geometry_attribute.instances.non_point_attribute",
    guard_non_point_attribute,
)
save_guard(
    "transform_matrix_mode",
    "constraint.geometry_attribute.instances.transform_matrix_mode",
    guard_transform_matrix_mode,
)
save_plain_case(
    "store_named_attribute_uncaptured",
    "graph.instance_component.store_named_attribute_uncaptured",
    guard_store_named_attribute_uncaptured,
    oracle_attribute="realized_instance_vector",
)
save_guard(
    "realize_attribute_columns",
    "graph.instance_component.realize_attribute_columns",
    guard_realize_attribute_columns,
    constraint_domain="INSTANCE",
    constraint_data_type="VECTOR",
    oracle_attribute="realized_instance_vector",
)
save_plain_case(
    "transform_matrix_plain_mesh",
    "constraint.geometry_attribute.instances.transform_matrix_mode",
    guard_transform_matrix_plain_mesh,
)
fresh_session_oracle = save_fresh_session_case()

with open(os.path.join(root, "expected.json"), "r", encoding="utf-8") as handle:
    expected = json.load(handle)
expected["guards"] = guards
expected["fresh_session"] = fresh_session_oracle

# HALO particles are render-only; GN instance-domain sampling remains supported.
clear_generated_scene()
particle_target = new_target("ParticleHaloInstanceTarget")
bpy.ops.object.select_all(action="DESELECT")
particle_target.select_set(True)
bpy.context.view_layer.objects.active = particle_target
bpy.ops.object.particle_system_add()
particle_settings = particle_target.particle_systems[-1].settings
particle_settings.render_type = "HALO"
particle_settings.count = 1
attach_group(particle_target, basic_group("ParticleHaloInstanceGeometry"))
particle_owner = make_owner_case(
    particle_target, "instance_transform", "FLOAT4X4", 1, "REPLACE",
    "owner_particle_halo",
)
particle_case = cases.pop()
scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
expected["particle_halo"] = {
    "id": particle_owner.name,
    "target": particle_target.name,
    "render_type": particle_settings.render_type,
    "baseline": particle_case["baseline"],
    "matrix": matrix_columns(particle_owner.evaluated_get(depsgraph).matrix_world),
}
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "geometry_attribute_instances_particle_halo.blend"))
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, sort_keys=True)
