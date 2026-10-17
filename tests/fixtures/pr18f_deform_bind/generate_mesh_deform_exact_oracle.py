"""Blender 5.2.2 oracle for the tests/pr18f_deform_bind.rs asymmetric Mesh Deform scene.

Rebuilds exactly the scene `asymmetric_mesh_operations` builds in Rust: a 4x4-subdivision grid
subject and a box cage with their transforms, the cage's eight vertices moved to the asymmetric
positions before bind, the `Influence` vertex group, a MESH_DEFORM modifier at the given precision,
bind, then the three post-bind cage vertex moves. Writes the evaluated subject positions.

Usage:
  Blender -b --factory-startup --python generate_mesh_deform_exact_oracle.py -- OUT.json
"""

import bpy
import json
import sys
from mathutils import Quaternion, Vector


def out_path():
    return sys.argv[sys.argv.index("--") + 1]


BASE = [
    (-2.0, -2.0, -2.0), (2.0, -2.0, -2.0), (2.0, 2.0, -2.0), (-2.0, 2.0, -2.0),
    (-2.0, -2.0, 2.0), (2.0, -2.0, 2.0), (2.0, 2.0, 2.0), (-2.0, 2.0, 2.0),
]
ASYMMETRIC = [
    (-2.2, -1.8, -2.0), (2.1, -1.7, -2.1), (2.1, 1.9, -2.2), (-2.0, 1.7, -1.8),
    (-1.8, -1.9, 2.0), (2.1, -1.7, 2.2), (2.2, 1.8, 2.3), (-2.1, 1.6, 1.7),
]
SUBJECT_TRANSLATION = (0.2, -0.1, 0.5)
SUBJECT_ROTATION = (0.0, 0.0, 0.14943813247359922, 0.9887710779360422)  # x, y, z, w
SUBJECT_SCALE = (0.9, 1.2, 0.75)
CAGE_TRANSLATION = (-0.4, 0.3, -0.2)
CAGE_ROTATION = (0.0, 0.0, -0.09983341664682815, 0.9950041652780258)  # x, y, z, w
CAGE_SCALE = (1.3, 0.8, 1.1)
POST_BIND_MOVES = {0: (0.0, 0.0, 0.12), 2: (0.1, -0.03, 0.0), 6: (-0.04, 0.08, -0.06)}


def set_transform(obj, translation, rotation_xyzw, scale):
    obj.location = Vector(translation)
    q = Quaternion((rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]))
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = q
    obj.scale = Vector(scale)


def subject_grid(name):
    # Rust `grid` size 2.0 with 4x4 subdivisions: a 5x5 vertex grid spanning [-1, 1] in x and y.
    n = 5
    coords, faces = [], []
    for y in range(n):
        for x in range(n):
            coords.append((-1.0 + 0.5 * x, -1.0 + 0.5 * y, 0.0))
    for y in range(n - 1):
        for x in range(n - 1):
            i = y * n + x
            faces.append((i, i + 1, i + n + 1, i + n))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(coords, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    set_transform(obj, SUBJECT_TRANSLATION, SUBJECT_ROTATION, SUBJECT_SCALE)
    return obj


def cage_box(name):
    # Rust `box` size 4.0: corners at +-2, then the eight vertices moved to ASYMMETRIC.
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(list(ASYMMETRIC), [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    set_transform(obj, CAGE_TRANSLATION, CAGE_ROTATION, CAGE_SCALE)
    return obj


def activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def evaluated_positions(obj):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    result = [[round(c, 9) for c in v.co] for v in mesh.vertices]
    evaluated.to_mesh_clear()
    return result


def run_case(label, precision, dynamic, invert):
    source = subject_grid("Subject_" + label)
    cage = cage_box("Cage_" + label)
    group = source.vertex_groups.new(name="Influence")
    for i in range(25):
        group.add([i], 0.15 + 0.2 * (i % 5), 'REPLACE')
    md = source.modifiers.new("MeshDeform", "MESH_DEFORM")
    md.object = cage
    md.precision = precision
    md.use_dynamic_bind = dynamic
    md.vertex_group = group.name
    md.invert_vertex_group = invert
    activate(source)
    bpy.ops.object.meshdeform_bind(modifier=md.name)
    bpy.context.view_layer.update()
    for index, delta in POST_BIND_MOVES.items():
        cage.data.vertices[index].co += Vector(delta)
    cage.data.update()
    return {"bound": bool(md.is_bound), "positions": evaluated_positions(source)}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


results = {}
cases = [
    ("mesh_asymmetric_static_p2", 2, False, False),
    ("mesh_asymmetric_static_p3", 3, False, False),
    ("mesh_asymmetric_dynamic_p3_invert", 3, True, True),
]
for name, precision, dynamic, invert in cases:
    reset()
    results[name] = run_case(name, precision, dynamic, invert)

with open(out_path(), "w") as handle:
    json.dump(results, handle, indent=1, sort_keys=True)
print("WROTE", out_path(), sorted(results))
