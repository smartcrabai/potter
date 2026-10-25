# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the Geometry Attribute hair-Curves constraint oracle with Blender 5.2.2."""
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

if bpy.app.version[:3] != (5, 2, 2):
    raise RuntimeError("this oracle requires Blender 5.2.2")

root = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

mix_modes = ["REPLACE", "BEFORE_FULL", "BEFORE_SPLIT", "AFTER_FULL", "AFTER_SPLIT"]
point_counts = [3, 4, 2]
surface_mesh = bpy.data.meshes.new("GeometryAttributeSurfaceMesh")
surface_mesh.from_pydata([(-2.0, -2.0, 0.0), (2.0, -2.0, 0.0),
                          (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0)], [], [(0, 1, 2, 3)])
surface_mesh.uv_layers.new(name="GeometryAttributeSurfaceUV")
surface_object = bpy.data.objects.new("GeometryAttributeSurface", surface_mesh)
scene.collection.objects.link(surface_object)

curve_data = bpy.data.hair_curves.new("GeometryAttributeCurves")
curve_data.add_curves(point_counts)
point_positions = [
    (-1.0, 0.0, 0.0), (-0.5, 0.2, 0.4), (0.0, 0.5, 0.8),
    (0.4, -0.3, 0.2), (0.8, 0.1, 0.6), (1.2, 0.4, 1.0), (1.6, 0.7, 1.3),
    (2.0, -0.2, 0.3), (2.4, 0.3, 0.9),
]
for point, position in zip(curve_data.points, point_positions):
    point.position = position
    point.radius = 0.03 + point.index * 0.005

curve_data.surface = surface_object
curve_data.surface_uv_map = "GeometryAttributeSurfaceUV"
curve_data.use_mirror_x = True
curve_data.use_mirror_y = False
curve_data.use_mirror_z = True
curve_data.selection_domain = "CURVE"
target = bpy.data.objects.new("GeometryAttributeTarget", curve_data)
scene.collection.objects.link(target)
target.location = (0.7, -1.1, 2.3)
target.rotation_mode = "QUATERNION"
target.rotation_quaternion = Quaternion(Euler((0.28, -0.19, 0.42), "XYZ"))
target.scale = (1.3, 0.75, 1.6)

attribute_names = {
    ("VECTOR", "POINT"): "position",
    ("VECTOR", "CURVE"): "position_curve",
    ("QUATERNION", "POINT"): "rotation_point",
    ("QUATERNION", "CURVE"): "rotation_curve",
    ("FLOAT4X4", "POINT"): "scale_point",
    ("FLOAT4X4", "CURVE"): "scale_curve",
}
attribute_types = {"VECTOR": "FLOAT_VECTOR", "QUATERNION": "QUATERNION", "FLOAT4X4": "FLOAT4X4"}
attribute_domains = {"POINT": "POINT", "CURVE": "CURVE"}

def value_for(data_type, index):
    if data_type == "VECTOR":
        return Vector((2.3 + index * 0.37, -1.7 + index * 0.23, 0.8 - index * 0.13))
    if data_type == "QUATERNION":
        return Quaternion(Euler((0.21 + index * 0.09, -0.31 + index * 0.07,
                                 0.17 - index * 0.05), "XYZ"))
    rotation = Quaternion(Euler((0.16 + index * 0.04, 0.37 - index * 0.03,
                                 -0.25 + index * 0.02), "XYZ"))
    translation = Vector((2.4 + index * 0.41, -1.8 + index * 0.19, 0.6 - index * 0.11))
    scale = Vector((1.35 + index * 0.07, 0.72 + index * 0.05, 1.18 + index * 0.03))
    # The last curve's point matrices force Blender's negative-determinant decomposition path.
    if index in (7, 8):
        rotation = Quaternion(Euler((0.0, 0.0, (index - 7) * math.pi / 2.0), "XYZ"))
        scale = Vector((-1.0, 1.0, 1.0))
    return Matrix.LocRotScale(translation, rotation, scale)

for data_type in attribute_types:
    for domain in attribute_domains:
        name = attribute_names[(data_type, domain)]
        if name == "position":
            continue
        attribute = curve_data.attributes.new(name=name, type=attribute_types[data_type],
                                              domain=attribute_domains[domain])
        for index, element in enumerate(attribute.data):
            value = value_for(data_type, index)
            if data_type == "VECTOR":
                element.vector = value
            elif data_type == "QUATERNION":
                element.value = value
            else:
                element.value = value

owner_rotation = Quaternion(Euler((0.16, 0.37, -0.25), "XYZ"))
cases = []
for data_type in attribute_types:
    for domain in attribute_domains:
        attribute_name = attribute_names[(data_type, domain)]
        indices = [0, 2, 4, 1, 999]
        for mode_index, mix_mode in enumerate(mix_modes):
            owner_id = "owner_%s_%s_%s" % (data_type.lower(), domain.lower(), mode_index)
            owner = bpy.data.objects.new(owner_id, None)
            scene.collection.objects.link(owner)
            owner.location = (-0.8, 1.2, 2.5)
            owner.rotation_mode = "QUATERNION"
            owner.rotation_quaternion = owner_rotation
            owner.scale = (0.8, 1.2, 1.1)
            constraint = owner.constraints.new("GEOMETRY_ATTRIBUTE")
            constraint.target = target
            constraint.attribute_name = attribute_name
            constraint.data_type = data_type
            constraint.domain = domain
            constraint.sample_index = indices[mode_index]
            constraint.mix_mode = mix_mode
            constraint.apply_target_transform = True
            constraint.mix_loc = True
            constraint.mix_rot = True
            constraint.mix_scl = True
            cases.append({"id": owner_id, "attribute": attribute_name, "data_type": data_type,
                          "domain": domain, "mix_mode": mix_mode,
                          "sample_index": indices[mode_index]})

cross_domain_cases = [
    ("VECTOR", "CURVE", "position", 1),
    ("QUATERNION", "POINT", "rotation_curve", 4),
    ("FLOAT4X4", "CURVE", "scale_point", 2),
]
for index, (data_type, domain, attribute_name, sample_index) in enumerate(cross_domain_cases):
    owner_id = "owner_cross_domain_%s" % index
    owner = bpy.data.objects.new(owner_id, None)
    scene.collection.objects.link(owner)
    owner.location = (-0.8, 1.2, 2.5)
    owner.rotation_mode = "QUATERNION"
    owner.rotation_quaternion = owner_rotation
    owner.scale = (0.8, 1.2, 1.1)
    constraint = owner.constraints.new("GEOMETRY_ATTRIBUTE")
    constraint.target = target
    constraint.attribute_name = attribute_name
    constraint.data_type = data_type
    constraint.domain = domain
    constraint.sample_index = sample_index
    constraint.mix_mode = "REPLACE"
    constraint.apply_target_transform = True
    constraint.mix_loc = True
    constraint.mix_rot = True
    constraint.mix_scl = True
    cases.append({"id": owner_id, "attribute": attribute_name, "data_type": data_type,
                  "domain": domain, "mix_mode": "REPLACE", "sample_index": sample_index})

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()

def matrix_columns(obj):
    matrix = obj.evaluated_get(depsgraph).matrix_world
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

expected_curve_state = {
    "point_radii": [float(point.radius) for point in curve_data.points],
    "surface_binding": {
        "object": surface_object.name_full,
        "surface_uv_map": curve_data.surface_uv_map,
        "use_mirror_x": bool(curve_data.use_mirror_x),
        "use_mirror_y": bool(curve_data.use_mirror_y),
        "use_mirror_z": bool(curve_data.use_mirror_z),
        "selection_domain": curve_data.selection_domain,
    },
}
expected_curve_state["target_matrix"] = matrix_columns(target)
expected = {case["id"]: matrix_columns(bpy.data.objects[case["id"]]) for case in cases}
os.makedirs(root, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "geometry_attribute_curves.blend"))
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump({"cases": cases, "matrices": expected, **expected_curve_state}, handle, sort_keys=True)
