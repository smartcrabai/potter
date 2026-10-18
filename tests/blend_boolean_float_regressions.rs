//! Blender-derived FAST Boolean boundary parity, including cases absent from the fixed corpus.

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

#[path = "fixtures/boolean_parity/comparator.rs"]
mod comparator;
use comparator::{
    TestResult, evaluate_boolean, expected_canonical, find_node, import_blend_scene, mismatches,
    run_generator,
};

use std::{fs, path::Path};

use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
};
use serde_json::Value;
use tempfile::tempdir;

const BLENDER_REQUIRED: &str = "Blender is required for Boolean FLOAT boundary parity: install Blender 5.2, or set POTTER_BLENDER to its executable";

const FLOAT_CASES: [&str; 33] = [
    "self_difference_false",
    "self_difference_true",
    "self_union_false",
    "self_union_true",
    "self_intersect_false",
    "self_intersect_true",
    "open_difference",
    "open_union",
    "open_intersect",
    "open_difference_hole_tolerant_true",
    "collection_difference",
    "collection_union",
    "collection_empty_mesh_difference",
    "collection_empty_mesh_union",
    "threshold_default",
    "threshold_high",
    "tiny_normal_cutoff_difference",
    "collection_wire_member_union",
    "transformed_operand_difference",
    "singular_transform_flip_union",
    "object_null_over_data_union",
    "loose_geometry_difference",
    "ngon_difference",
    "missing_layers_union",
    "multi_named_uv_union",
    "float_attributes_union_transfer_mode",
    "float_attributes_union",
    "unset_object_difference",
    "one_sided_vertex_edge_boundary",
    "object_faceless_operand_union",
    "object_faceless_operand_intersect",
    "object_faceless_subject_union",
    "float_operand_material_index_oob",
];

const FLOAT_ERROR_CASES: [&str; 2] = ["collection_intersect_error", "empty_collection_error"];

const FLOAT_FIXTURE: &str = r#"
import bpy
import bmesh
import math
import json
import os
import runpy
import sys
from mathutils import Matrix, Quaternion, Vector

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
helpers = runpy.run_path(r"__GENERATOR_PATH__")
scene = bpy.context.scene.collection
box = helpers["box"]
join = helpers["join"]
part = helpers["part"]
remove_face = helpers["remove_face"]
make_object = helpers["make_object"]
build_case = helpers["build_case"]
describe = helpers["describe"]
make_case = helpers["case"]


def add_input_attributes(obj, source):
    mesh = obj.data
    point = mesh.attributes.new("point_signal", "FLOAT", "POINT")
    for index, item in enumerate(point.data):
        item.value = 0.125 + source * 10.0 + index * 0.375
    edge_rank = mesh.attributes.new("edge_rank", "INT", "EDGE")
    edge_flag = mesh.attributes.new("edge_flag", "BOOLEAN", "EDGE")
    for index, edge in enumerate(mesh.edges):
        edge_rank.data[index].value = 100 * source + index + 1
        edge_flag.data[index].value = (index + source) % 2 == 0
        edge.use_seam = (index + source) % 3 == 0
        edge.use_edge_sharp = (index + source) % 2 == 1
    crease = mesh.attributes.get("crease_edge")
    if crease is None:
        crease = mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
    for index, item in enumerate(crease.data):
        item.value = 0.125 + 0.25 * source + index * 0.03125
    face_score = mesh.attributes.new("face_score", "FLOAT", "FACE")
    for index, item in enumerate(face_score.data):
        item.value = 2.5 + 10.0 * source + index * 0.75
    corner_rank = mesh.attributes.new("corner_rank", "FLOAT", "CORNER")
    for index, item in enumerate(corner_rank.data):
        item.value = 0.25 + 20.0 * source + index * 0.125
    corner_integer = mesh.attributes.new("corner_integer", "INT", "CORNER")
    corner_flag = mesh.attributes.new("corner_flag", "BOOLEAN", "CORNER")
    for index in range(len(mesh.loops)):
        corner_integer.data[index].value = 100 * source + index + 1
        corner_flag.data[index].value = (index + source) % 3 == 0
    corner_rotation = mesh.attributes.new("corner_rotation", "QUATERNION", "CORNER")
    for index, item in enumerate(corner_rotation.data):
        axis = Vector((0.3 + 0.01 * index, 0.4, 0.5 + 0.02 * source)).normalized()
        angle = 0.0001 if index == 0 else 0.4 * (index + 1 + source)
        item.value = Quaternion(axis, angle)
    for domain in ("POINT", "EDGE", "FACE", "CORNER"):
        add_bmesh_copy_type_attributes(mesh, domain, source)

    point_paint = mesh.color_attributes.new("PointPaint", "FLOAT_COLOR", "POINT")
    for index, item in enumerate(point_paint.data):
        item.color = (
            0.1 + 0.04 * index + 0.2 * source,
            0.2 + 0.01 * index,
            0.35 + 0.05 * source,
            0.8,
        )
    corner_tint = mesh.color_attributes.new("CornerTint", "BYTE_COLOR", "CORNER")
    for index, item in enumerate(corner_tint.data):
        item.color_srgb = [
            ((17 * index + 31 * source) % 256) / 255.0,
            ((29 * index + 73 + source) % 256) / 255.0,
            ((43 * index + 101 + 19 * source) % 256) / 255.0,
            1.0 if index % 2 == 0 else 0.5,
        ]
    mesh.color_attributes.active_color_index = 0
    mesh.color_attributes.render_color_index = 1

    normals = []
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            normal = (
                Vector(polygon.normal)
                + Vector((
                    0.11 + 0.07 * source,
                    0.03 * (polygon.index + 1),
                    0.02 * (loop_index + 1),
                ))
            ).normalized()
            normals.append(normal)
    mesh.normals_split_custom_set(normals)
    mesh.update()
    for name in (
        "point_signal", "edge_rank", "edge_flag", "crease_edge", "face_score",
        "corner_rank", "corner_integer", "corner_flag", "corner_rotation",
    ):
        if mesh.attributes.get(name) is None:
            raise RuntimeError(f"missing authored mesh attribute {name}")
    for domain in ("point", "edge", "face", "corner"):
        for suffix in ("int8", "int16_2d", "int32_2d", "float4x4", "string"):
            if mesh.attributes.get(f"{domain}_{suffix}") is None:
                raise RuntimeError(f"missing authored {domain} copy/default attribute {suffix}")
    if not any(edge.use_seam for edge in mesh.edges):
        raise RuntimeError("the seam fixture has no marked edges")
    if not any(edge.use_edge_sharp for edge in mesh.edges):
        raise RuntimeError("the sharp-edge fixture has no marked edges")
    if mesh.color_attributes.get("PointPaint") is None or mesh.color_attributes.get("CornerTint") is None:
        raise RuntimeError("the point/corner color fixtures are missing")
    codes = mesh.attributes.get("custom_normal")
    if codes is None or not any(tuple(item.value) != (0, 0) for item in codes.data):
        raise RuntimeError("the custom-normal fixture has no authored code pairs")

def add_bmesh_copy_type_attributes(mesh, domain, source):
    records = (
        ("int8", "INT8"),
        ("int16_2d", "INT16_2D"),
        ("int32_2d", "INT32_2D"),
        ("float4x4", "FLOAT4X4"),
        ("string", "STRING"),
    )
    for suffix, data_type in records:
        attribute = mesh.attributes.new(f"{domain.lower()}_{suffix}", data_type, domain)
        for index, item in enumerate(attribute.data):
            if data_type == "INT8":
                item.value = 10 * source + index + 1
            elif data_type == "INT16_2D":
                item.value = (10 * source + index + 1, -index - source)
            elif data_type == "INT32_2D":
                item.value = (100 * source + index + 1, -index - source)
            elif data_type == "FLOAT4X4":
                value = Matrix.Identity(4)
                value[0][3] = 10.0 * source + index + 0.25
                value[1][3] = -float(index + source)
                item.value = value
            else:
                item.value = f"{domain.lower()}-{source}-{index}".encode("utf-8")

def validate_rich_record(record, generated_defaults=True):
    point_names = {name for row in record.get("point_attributes", []) for name in row["values"]}
    edge_names = {name for row in record.get("edge_attributes", []) for name in row["values"]}
    color_names = {layer["name"] for layer in record.get("color_layers", {}).get("layers", [])}
    if not {
        "point_signal", "PointPaint", "point_int8", "point_int16_2d",
        "point_int32_2d", "point_float4x4", "point_string",
    } <= point_names:
        raise RuntimeError("Blender FLOAT output omitted a POINT attribute")
    if not {
        "edge_rank", "edge_flag", "seam", "sharp", "crease_edge", "edge_int8",
        "edge_int16_2d", "edge_int32_2d", "edge_float4x4", "edge_string",
    } <= edge_names:
        raise RuntimeError("Blender FLOAT output omitted an EDGE attribute")
    if not {"PointPaint", "CornerTint"} <= color_names:
        raise RuntimeError("Blender FLOAT output omitted one of the color layers")
    if generated_defaults:
        group_rows = record["vertex_groups"].get("Host", [])
        if not group_rows or not any(row["weight"] is None for row in group_rows):
            raise RuntimeError("the Boolean cut has no generated vertex-group default")
        if not any(row["values"]["point_signal"] == 0.0 for row in record["point_attributes"]):
            raise RuntimeError("the Boolean cut has no generated POINT attribute default")
        if not any(row["values"]["edge_rank"] == 0 for row in record["edge_attributes"]):
            raise RuntimeError("the Boolean cut has no generated EDGE attribute default")
    if not record["faces"]:
        raise RuntimeError("Blender FLOAT attributes case unexpectedly has no faces")
    for face in record["faces"]:
        face_attribute_names = set(face.get("face_attributes", {}))
        required_face_names = {
            "face_score", "face_int8", "face_int16_2d", "face_int32_2d",
            "face_float4x4", "face_string",
        }
        if not required_face_names <= face_attribute_names or "fid" not in face:
            raise RuntimeError("Blender FLOAT output omitted a FACE attribute")
        corner_names = set(face.get("corner_attributes", {}))
        required_corner_names = {
            "corner_rank",
            "corner_integer",
            "corner_flag",
            "corner_int8",
            "corner_int16_2d",
            "corner_int32_2d",
            "corner_float4x4",
            "corner_string",
            "CornerTint",
            "corner_rotation",
        }
        if not required_corner_names <= corner_names:
            raise RuntimeError("Blender FLOAT output omitted CORNER attributes")
        if "uv" not in face or "custom_normal_codes" not in face or "custom_normals" not in face:
            raise RuntimeError("Blender FLOAT output omitted UV or custom normals")
    return record

def add_named_uv_layers(obj, names, source, active_index, render_index):
    mesh = obj.data
    if len(mesh.uv_layers) != 0:
        raise RuntimeError(f"{obj.name} unexpectedly already has UV layers")
    for layer_index, name in enumerate(names):
        layer = mesh.uv_layers.new(name=name)
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                item = layer.data[loop_index]
                item.uv = (
                    0.375 + source * 5.0 + layer_index * 1.25
                    + polygon.index * 0.0625 + loop_index * 0.015625,
                    -0.625 - source * 3.0 + layer_index * 0.875
                    - polygon.index * 0.03125 + loop_index * 0.0078125,
                )
                item.pin_uv = (loop_index + layer_index + source) % 2 == 0
    mesh.uv_layers.active_index = active_index
    mesh.uv_layers.active_render_index = render_index
    if (tuple(layer.name for layer in mesh.uv_layers) != tuple(names)
            or mesh.uv_layers.active_index != active_index
            or mesh.uv_layers.active_render_index != render_index):
        raise RuntimeError(f"{obj.name} has unexpected UV layer selections")


def add_case(name, operation, host, operands=(), *, collection=False, use_self=False,
             hole_tolerant=None, material_mode="INDEX", threshold=None, rich_attributes=False,
             generated_defaults=True):
    spec = make_case(
        "float_boundary", "PR17b", "FLOAT", operation, host, operands,
        collection=collection, use_self=use_self, hole_tolerant=hole_tolerant,
        material_mode=material_mode,
    )
    spec["id"] = name
    owner = build_case(spec)
    if rich_attributes:
        owner.vertex_groups[0].name = "Host"
        add_input_attributes(owner, 0)
        operand = bpy.data.objects[name + ".operand"]
        operand.vertex_groups[0].name = "Cutter"
        add_input_attributes(operand, 1)
    if threshold is not None:
        owner.modifiers[-1].double_threshold = threshold
    record = describe(owner, spec)
    expected[name] = validate_rich_record(record, generated_defaults) if rich_attributes else record
expected = {}
host = part(box(-1, 1, -1, 1, -1, 1))
offset = part(box(0.5, 2.5, -0.5, 1.5, 0, 2))
self_operand = part(join(box(0, 2, 0, 2, 0, 2), box(1, 3, 1, 3, 1, 3)))
big_host = part(box(-2, 2, -2, 2, -2, 2))
for operation in ("DIFFERENCE", "UNION", "INTERSECT"):
    for use_self in (False, True):
        suffix = "true" if use_self else "false"
        add_case(f"self_{operation.lower()}_{suffix}", operation, big_host,
                 [self_operand], use_self=use_self)

# Both face normals fall below Blender's 1e-35 squared-length cutoff, so no interior cut is built.
tiny_scale = 2.0 ** -30
tiny_subject = part((
    [(0.0, 0.0, 0.0), (tiny_scale, 0.0, 0.0), (0.0, tiny_scale, 0.0)],
    [(0, 1, 2)],
))
tiny_operand = part((
    [
        (tiny_scale / 4, tiny_scale / 4, -tiny_scale),
        (tiny_scale / 4, tiny_scale / 4, tiny_scale),
        (tiny_scale / 2, tiny_scale / 4, tiny_scale),
    ],
    [(0, 1, 2)],
))
add_case(
    "tiny_normal_cutoff_difference",
    "DIFFERENCE",
    tiny_subject,
    [tiny_operand],
    threshold=2.0 ** -40,
)

open_offset = part(remove_face(box(0.5, 2.5, -0.5, 1.5, 0, 2), 1))
for operation in ("DIFFERENCE", "UNION", "INTERSECT"):
    add_case(f"open_{operation.lower()}", operation, host, [open_offset])
add_case(
    "open_difference_hole_tolerant_true",
    "DIFFERENCE",
    host,
    [open_offset],
    hole_tolerant=True,
)

members = [
    part(box(0.5, 2.5, -0.5, 1.5, 0, 2)),
    part(box(1, 3, -0.5, 1.5, -0.5, 1)),
]
for operation in ("DIFFERENCE", "UNION"):
    add_case(f"collection_{operation.lower()}", operation, host,
             members, collection=True)
for operation in ("DIFFERENCE", "UNION"):
    empty_name = f"collection_empty_mesh_{operation.lower()}"
    empty_spec = make_case(
        "float_collection_empty_member",
        "PR17b",
        "FLOAT",
        operation,
        part(box(-1, 1, -1, 1, -1, 1), materials=helpers["split_by_x"]),
        [part(([], []))],
        collection=True,
    )
    empty_spec["id"] = empty_name
    empty_owner = build_case(empty_spec)
    empty_member = bpy.data.objects[f"{empty_name}.member0"]
    empty_member.location = (10.0, 0.0, 0.0)
    expected[empty_name] = describe(empty_owner, empty_spec)
wire_name = "collection_wire_member_union"
wire_spec = make_case(
    "float_collection_wire_member",
    "PR17b",
    "FLOAT",
    "UNION",
    host,
    [part(([(8.0, 4.0, 3.0), (9.0, 4.0, 3.0)], []))],
    collection=True,
)
wire_spec["id"] = wire_name
wire_owner = build_case(wire_spec)
wire_member = bpy.data.objects[wire_name + ".member0"]
wire_bm = bmesh.new()
wire_bm.from_mesh(wire_member.data)
wire_bm.verts.ensure_lookup_table()
wire_bm.edges.new((wire_bm.verts[0], wire_bm.verts[1]))
wire_bm.to_mesh(wire_member.data)
wire_bm.free()
wire_member.data.update()
wire_member.location = (10.0, 0.0, 0.0)
expected[wire_name] = describe(wire_owner, wire_spec)
add_case("collection_intersect_error", "INTERSECT", host,
         members, collection=True)
add_case("empty_collection_error", "DIFFERENCE", host, collection=True)

near_gap = part(box(-1, 1, -1, 1, 1.0001, 3.0001))
add_case("threshold_default", "UNION", host, [near_gap], threshold=0.000001)
add_case("threshold_high", "UNION", host, [near_gap], threshold=0.5)
transform_name = "transformed_operand_difference"
transform_spec = make_case("float_transform", "PR17b", "FLOAT", "DIFFERENCE", host, [offset])
transform_spec["id"] = transform_name
transform_owner = build_case(transform_spec)
transform_operand = bpy.data.objects[transform_name + ".operand"]
transform_owner.location = (0.5, -0.25, 0.25)
transform_owner.rotation_euler = (0.13, -0.27, 0.19)
transform_owner.scale = (1.2, 0.7, 1.1)
transform_operand.location = (0.9, 0.1, 0.0)
transform_operand.rotation_euler = (0.21, 0.1, -0.15)
transform_operand.scale = (-0.8, 1.3, 0.6)
expected[transform_name] = describe(transform_owner, transform_spec)
singular_name = "singular_transform_flip_union"
singular_plane = part((
    [(1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0)],
    [(0, 1, 2, 3)],
))
singular_spec = make_case(
    "float_singular_transform_flip",
    "PR17b",
    "FLOAT",
    "UNION",
    part(box(-1, 1, 8, 12, -1, 1)),
    [singular_plane],
)
singular_spec["id"] = singular_name
singular_owner = build_case(singular_spec)
singular_owner.scale = (-1.0, 1.0, 1.0)
bpy.data.objects[singular_name + ".operand"].scale = (0.0, 1.0, 1.0)
expected[singular_name] = describe(singular_owner, singular_spec)
object_null_name = "object_null_over_data_union"
object_null_spec = make_case(
    "float_object_null_link_data",
    "PR17b",
    "FLOAT",
    "UNION",
    part(box(-1, 1, -1, 1, -1, 1), materials=helpers["split_by_x"]),
    [part(box(-1, 1, 8, 12, -1, 1), materials=helpers["split_by_x"])],
)
object_null_spec["id"] = object_null_name
object_null_owner = build_case(object_null_spec)
object_null_operand = bpy.data.objects[object_null_name + ".operand"]
object_null_owner.material_slots[0].link = "OBJECT"
object_null_owner.material_slots[0].material = None
object_null_operand.data.materials[0] = object_null_owner.data.materials[0]
object_null_operand.material_slots[0].link = "DATA"
expected[object_null_name] = describe(object_null_owner, object_null_spec)
loose_name = "loose_geometry_difference"
loose_spec = make_case(
    "float_loose_geometry", "PR17b", "FLOAT", "DIFFERENCE", host, [offset]
)
loose_spec["id"] = loose_name
loose_owner = build_case(loose_spec)
loose_bm = bmesh.new()
loose_bm.from_mesh(loose_owner.data)
loose_a = loose_bm.verts.new((5.0, 4.0, 3.0))
loose_b = loose_bm.verts.new((6.0, 4.0, 3.0))
loose_edge = loose_bm.edges.new((loose_a, loose_b))
loose_edge.seam = True
loose_edge.smooth = False
loose_bm.verts.new((7.0, 4.0, 3.0))
loose_bm.to_mesh(loose_owner.data)
loose_bm.free()
loose_owner.data.update()
loose_operand = bpy.data.objects[loose_name + ".operand"]
operand_bm = bmesh.new()
operand_bm.from_mesh(loose_operand.data)
operand_a = operand_bm.verts.new((8.0, 4.0, 3.0))
operand_b = operand_bm.verts.new((9.0, 4.0, 3.0))
operand_edge = operand_bm.edges.new((operand_a, operand_b))
operand_edge.seam = False
operand_edge.smooth = True
operand_bm.to_mesh(loose_operand.data)
operand_bm.free()
loose_operand.data.update()
for source, owner in enumerate((loose_owner, loose_operand)):
    edge_attribute = owner.data.attributes.new("loose_edge_rank", "INT", "EDGE")
    for index, item in enumerate(edge_attribute.data):
        item.value = 1000 * source + index + 1
    point_attribute = owner.data.attributes.new("loose_point_rank", "INT", "POINT")
    for index, item in enumerate(point_attribute.data):
        item.value = 100 * source + index + 1
expected[loose_name] = describe(loose_owner, loose_spec)
angles = [2.0 * math.pi * index / 5.0 for index in range(5)]
lower = [(math.cos(angle), math.sin(angle), -1.0) for angle in angles]
upper = [(x, y, 1.0) for x, y, _ in lower]
ngon_faces = [tuple(reversed(range(5))), tuple(range(5, 10))]
ngon_faces.extend(
    (index, (index + 1) % 5, (index + 1) % 5 + 5, index + 5) for index in range(5)
)
ngon_host = part((lower + upper, ngon_faces))
add_case("ngon_difference", "DIFFERENCE", ngon_host, [offset])
missing_name = "missing_layers_union"
missing_spec = make_case(
    "float_missing_layers", "PR17b", "FLOAT", "UNION",
    part(box(-1, 1, -1, 1, -1, 1), uv=0.0), [offset],
)
missing_spec["id"] = missing_name
missing_owner = build_case(missing_spec)
host_color = missing_owner.data.color_attributes.new("HostPaint", "FLOAT_COLOR", "CORNER")
for index, item in enumerate(host_color.data):
    item.color = (0.2 + index * 0.01, 0.4, 0.6, 0.75)
missing_owner.data.update()
expected[missing_name] = describe(missing_owner, missing_spec)
multi_uv_name = "multi_named_uv_union"
multi_uv_host_names = ("HostUV", "SharedUV", "HostRenderUV")
multi_uv_operand_names = ("OperandUV", "OperandRenderUV", "SharedUV")
multi_uv_spec = make_case(
    "float_multi_named_uv", "PR17b", "FLOAT", "UNION",
    part(box(-1, 1, -1, 1, -1, 1)),
    [part(box(0.5, 2.5, -0.5, 1.5, 0, 2))],
)
multi_uv_spec["id"] = multi_uv_name
multi_uv_owner = build_case(multi_uv_spec)
multi_uv_operand = bpy.data.objects[multi_uv_name + ".operand"]
add_named_uv_layers(multi_uv_owner, multi_uv_host_names, 0, 1, 2)
add_named_uv_layers(multi_uv_operand, multi_uv_operand_names, 1, 1, 2)
multi_uv_record = describe(multi_uv_owner, multi_uv_spec)
expected_multi_uv_names = set(multi_uv_host_names) | set(multi_uv_operand_names)
output_multi_uv_names = {
    name for face in multi_uv_record["faces"] for name in face.get("uv_layers", {})
}
if not expected_multi_uv_names <= output_multi_uv_names:
    raise RuntimeError("Blender FLOAT union omitted a source-only named UV layer")
if not any(
        len({tuple(value) for value in layer["uv"]}) > 1
        for face in multi_uv_record["faces"]
        for layer in face["uv_layers"].values()
):
    raise RuntimeError("Blender FLOAT union has no per-loop named UV values")
if not any(
        len(set(layer["pins"])) > 1
        for face in multi_uv_record["faces"]
        for layer in face["uv_layers"].values()
):
    raise RuntimeError("Blender FLOAT union has no per-loop named UV pin flags")
expected[multi_uv_name] = multi_uv_record

attribute_host = part(
    box(-1, 1, -1, 1, -1, 1), fid=100, uv=0.0,
    materials=helpers["split_by_x"],
    weight=lambda index: 0.1 * (index + 1),
)
attribute_operand = part(
    box(0.5, 2.5, -0.5, 1.5, 0, 2), fid=200, uv=5.0,
    materials=helpers["split_by_z"],
    weight=lambda index: 0.9,
)
add_case("float_attributes_union", "UNION", attribute_host, [attribute_operand],
         rich_attributes=True)
add_case(
    "float_attributes_union_transfer_mode",
    "UNION",
    attribute_host,
    [attribute_operand],
    rich_attributes=True,
    material_mode="TRANSFER",
)
# The subject's z=0 boundary edge lies inside the operand triangle; the operand face must split alone.
boundary_host = part(
    ([(0.5, 0, 0), (0, 0.5, 0), (0, 0, 1)], [(0, 1, 2)]),
    fid=300,
    uv=2.0,
    materials=helpers["split_by_x"],
    weight=lambda index: 0.1 * (index + 1),
)
boundary_operand = part(
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]),
    fid=400,
    uv=6.0,
    materials=helpers["split_by_z"],
    weight=lambda index: 0.9,
)
add_case(
    "one_sided_vertex_edge_boundary",
    "UNION",
    boundary_host,
    [boundary_operand],
    rich_attributes=True,
    # Blender 5.2.2 interpolates this cut's weights, POINT signal and EDGE rank and creates no
    # default rows (probe /tmp/boundary_probe/probe2.log: Host weights 0.3..0.9, POINT signal
    # 0.125..10.875, EDGE rank 1..103); the default checks encode an older Blender here.
    generated_defaults=False,
)
boundary_faces = [
    face for face in expected["one_sided_vertex_edge_boundary"]["faces"]
    if face["fid"] == 400
]
cut_points = {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0)}
if not any(
        cut_points <= {tuple(point) for point in face["ring"]}
        for face in boundary_faces
):
    raise RuntimeError("Blender FLOAT did not retain the one-sided B-face cut in final output")
for operation in ("UNION", "INTERSECT"):
    quick_name = f"object_faceless_operand_{operation.lower()}"
    quick_host = (
        part(box(-1, 1, -1, 1, -1, 1), materials=helpers["split_by_x"])
        if operation == "INTERSECT" else host
    )
    quick_spec = make_case(
        "float_quick_mesh",
        "PR17b",
        "FLOAT",
        operation,
        quick_host,
        [part(([(8.0, 4.0, 3.0), (9.0, 4.0, 3.0)], []))],
    )
    quick_spec["id"] = quick_name
    quick_owner = build_case(quick_spec)
    quick_operand = bpy.data.objects[quick_name + ".operand"]
    quick_bm = bmesh.new()
    quick_bm.from_mesh(quick_operand.data)
    quick_bm.verts.ensure_lookup_table()
    quick_bm.edges.new((quick_bm.verts[0], quick_bm.verts[1]))
    quick_bm.to_mesh(quick_operand.data)
    quick_bm.free()
    quick_operand.data.update()
    expected[quick_name] = describe(quick_owner, quick_spec)
faceless_subject_name = "object_faceless_subject_union"
faceless_subject_spec = make_case(
    "float_quick_mesh",
    "PR17b",
    "FLOAT",
    "UNION",
    part(([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], []), materials=helpers["split_by_x"]),
    [part(box(-1, 1, -1, 1, -1, 1), materials=helpers["split_by_z"])],
)
faceless_subject_spec["id"] = faceless_subject_name
faceless_subject_owner = build_case(faceless_subject_spec)
expected[faceless_subject_name] = describe(faceless_subject_owner, faceless_subject_spec)
far_operand = part(box(4, 5, 4, 5, 4, 5))
material_spec = make_case(
    "float_material_index",
    "PR17b",
    "FLOAT",
    "UNION",
    part(box(-1, 1, -1, 1, -1, 1), materials=helpers["split_by_x"]),
    [far_operand],
)
material_spec["id"] = "float_operand_material_index_oob"
material_owner = build_case(material_spec)
material_operand = bpy.data.objects[material_spec["id"] + ".operand"]
material_operand.data.materials.clear()
material_operand.data.materials.append(
    bpy.data.materials.new(material_spec["id"] + ".single_slot")
)
for polygon in material_operand.data.polygons:
    polygon.material_index = 1
expected[material_spec["id"]] = describe(material_owner, material_spec)

unset_spec = make_case("unset_object", "PR17b", "FLOAT", "DIFFERENCE", host, [])
unset_spec["id"] = "unset_object_difference"
unset_host = make_object(unset_spec["id"], scene, host)
unset_boolean = unset_host.modifiers.new("Boolean", "BOOLEAN")
unset_boolean.operation = "DIFFERENCE"
unset_boolean.solver = "FLOAT"
expected[unset_spec["id"]] = describe(unset_host, unset_spec)

with open(os.path.join(root, "expected_float_boundaries.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output, separators=(",", ":"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "scene.blend"))
"#;

#[test]
fn boolean_float_boundaries_match_blender() -> TestResult {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let script = root.join("generate_float_boolean.py");
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/boolean_parity/generate_blender_expectations.py");
    let fixture = FLOAT_FIXTURE.replace(
        "__GENERATOR_PATH__",
        generator
            .to_str()
            .ok_or("Boolean generator path is not UTF-8")?,
    );
    fs::write(&script, fixture)?;
    run_generator(&blender, &script, &root)?;
    let scene = import_blend_scene(&blender, &root.join("scene.blend"), &root.join("project"))?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("expected_float_boundaries.json"))?)?;

    let mut failures = Vec::new();
    for case in FLOAT_CASES {
        let expected_json = expected
            .get(case)
            .ok_or_else(|| format!("Blender FLOAT expectation `{case}` is missing"))?;
        let node_id = find_node(&scene, case)?;
        match evaluate_boolean(&scene, &node_id) {
            Ok(actual) => {
                let canonical = expected_canonical(expected_json)?;
                failures.extend(
                    mismatches(&actual, &canonical)
                        .into_iter()
                        .map(|difference| format!("{case}: {difference}")),
                );
            }
            Err(error) => failures.push(format!("{case}: {error}")),
        }
    }

    let flipped_operand_ring = [
        [0.0, -1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
    ];
    let has_flipped_operand_face = expected["singular_transform_flip_union"]["faces"]
        .as_array()
        .is_some_and(|faces| {
            faces.iter().any(|face| {
                let Some(ring) = face.get("ring").and_then(Value::as_array) else {
                    return false;
                };
                ring.len() == flipped_operand_ring.len()
                    && ring
                        .iter()
                        .zip(flipped_operand_ring)
                        .all(|(point, expected)| {
                            point.as_array().is_some_and(|coordinates| {
                                coordinates.len() == 3
                                    && coordinates.iter().zip(expected).all(|(actual, expected)| {
                                        actual.as_f64().is_some_and(|actual| {
                                            (actual - expected).abs() <= 1.0e-6
                                        })
                                    })
                            })
                        })
            })
        });
    if !has_flipped_operand_face {
        failures.push(
            "Blender FLOAT singular-transform probe did not retain the source-flipped operand face"
                .to_owned(),
        );
    }
    for case in FLOAT_ERROR_CASES {
        let expected_json = expected
            .get(case)
            .ok_or_else(|| format!("Blender FLOAT error expectation `{case}` is missing"))?;
        let node_id = find_node(&scene, case)?;
        let mut input_only = scene.clone();
        input_only
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| format!("Boolean host `{case}` is missing"))?
            .modifiers
            .retain(|modifier| modifier.modifier_type != "boolean");
        match evaluate_boolean(&input_only, &node_id) {
            Ok(actual) => {
                let expected = expected_canonical(expected_json)?;
                failures.extend(
                    mismatches(&actual, &expected)
                        .into_iter()
                        .map(|difference| format!("{case} Blender error must leave the subject unchanged: {difference}")),
                );
            }
            Err(error) => failures.push(format!(
                "{case}: could not evaluate unmodified subject: {error}"
            )),
        }
        let expected_error = if case == "collection_intersect_error" {
            "intersect only available using exact solver"
        } else {
            "empty collection"
        };
        let mut only_boolean = scene.clone();
        for (id, node) in &mut only_boolean.nodes {
            for modifier in &mut node.modifiers {
                modifier.enabled = id == &node_id && modifier.modifier_type == "boolean";
            }
        }
        match Snapshot::evaluate(&only_boolean, &EvaluationContext::default()) {
            Ok(_) => failures.push(format!("{case}: FLOAT collection error was not rejected")),
            Err(error)
                if error.code == ErrorCode::InvalidOperation
                    && error.message.to_ascii_lowercase().contains(expected_error) => {}
            Err(error) => failures.push(format!(
                "{case}: expected INVALID_OPERATION containing `{expected_error}`, got {:?}: {}",
                error.code, error.message,
            )),
        }
    }

    let mut low_threshold = expected["threshold_default"].clone();
    let mut high_threshold = expected["threshold_high"].clone();
    low_threshold
        .as_object_mut()
        .ok_or("threshold expectation is not an object")?
        .remove("case");
    high_threshold
        .as_object_mut()
        .ok_or("threshold expectation is not an object")?
        .remove("case");
    if low_threshold != high_threshold {
        failures.push(
            "Blender FLOAT threshold probe unexpectedly changed the near-gap union".to_owned(),
        );
    }
    for operation in ["difference", "union", "intersect"] {
        let mut without_self = expected[&format!("self_{operation}_false")].clone();
        let mut with_self = expected[&format!("self_{operation}_true")].clone();
        without_self
            .as_object_mut()
            .ok_or("self-overlap expectation is not an object")?
            .remove("case");
        without_self
            .as_object_mut()
            .ok_or("self-overlap expectation is not an object")?
            .remove("use_self");
        with_self
            .as_object_mut()
            .ok_or("self-overlap expectation is not an object")?
            .remove("case");
        with_self
            .as_object_mut()
            .ok_or("self-overlap expectation is not an object")?
            .remove("use_self");
        if without_self != with_self {
            failures.push(format!(
                "Blender FLOAT use_self changed the {operation} self-overlap result"
            ));
        }
    }
    let mut open_default = expected["open_difference"].clone();
    let mut open_hole_tolerant = expected["open_difference_hole_tolerant_true"].clone();
    for record in [&mut open_default, &mut open_hole_tolerant] {
        let object = record
            .as_object_mut()
            .ok_or("open FLOAT expectation is not an object")?;
        for field in ["case", "hole_tolerant", "input_geometry"] {
            object.remove(field);
        }
    }
    if open_default != open_hole_tolerant {
        failures.push("Blender FLOAT changed its open difference for use_hole_tolerant".to_owned());
    }

    let mut index_materials = expected["float_attributes_union"].clone();
    let mut transfer_materials = expected["float_attributes_union_transfer_mode"].clone();
    for record in [&mut index_materials, &mut transfer_materials] {
        let object = record
            .as_object_mut()
            .ok_or("FLOAT material expectation is not an object")?;
        for field in [
            "case",
            "material_mode",
            "object_materials",
            "effective_materials",
            "data_materials",
            "material_links",
        ] {
            object.remove(field);
        }
    }
    if index_materials != transfer_materials {
        failures.push("Blender FLOAT unexpectedly applied material_mode TRANSFER".to_owned());
    }
    assert!(
        failures.is_empty(),
        "{} Blender FLOAT boundary mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
