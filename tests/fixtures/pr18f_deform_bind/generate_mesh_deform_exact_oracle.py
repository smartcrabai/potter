"""Blender 5.2.2 oracle for the tests/pr18f_deform_bind.rs Mesh Deform scenes.

Rebuilds the exact Rust scenes:
- `asymmetric_mesh_operations`: 4x4-subdivision grid subject, box cage (size 4) with the eight cage
  vertices moved to asymmetric positions before bind, the `Influence` group, MESH_DEFORM at the
  given precision, bind, then three post-bind cage vertex moves (cases `mesh_asymmetric_*`).
- `mesh_deform_nonuniform_bind_transform_matches_blender_translation`: the same subject grid, a box
  cage of size 6 (cage transform set), bind at precision 3, then all eight cage vertices translated
  by CAGE_TRANSLATION after bind (case `mesh_nonuniform_translation`).

Writes the evaluated subject positions for every case to OUT.json.

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

# Rust `CAGE_TRANSLATION` used by the nonuniform translation test (cage-local, all eight vertices).
TRANSLATION_CAGE_MOVE = (0.3, -0.2, 0.5)


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


def cage_box(name, corners, transform=True):
    # Winding is reversed so the face normals point outward. Mesh Deform's inside test needs outward
    # normals: with the opposite winding, Blender classifies every subject vertex as outside and the
    # bound subject never moves. Vertex indices are unchanged, so the Rust cage moves still apply.
    faces = [(3, 2, 1, 0), (5, 6, 7, 4), (1, 5, 4, 0), (2, 6, 5, 1), (3, 7, 6, 2), (0, 4, 7, 3)]
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(list(corners), [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if transform:
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


def bind_and_read(label, source, cage, precision, dynamic, invert, moves, group_weights):
    md = source.modifiers.new("MeshDeform", "MESH_DEFORM")
    md.object = cage
    md.precision = precision
    md.use_dynamic_bind = dynamic
    md.invert_vertex_group = invert
    if group_weights is not None:
        group = source.vertex_groups.new(name="Influence")
        for index, weight in enumerate(group_weights):
            group.add([index], weight, 'REPLACE')
        md.vertex_group = group.name
    activate(source)
    bpy.ops.object.meshdeform_bind(modifier=md.name)
    bpy.context.view_layer.update()
    for index, delta in moves.items():
        cage.data.vertices[index].co += Vector(delta)
    cage.data.update()
    return {"bound": bool(md.is_bound), "positions": evaluated_positions(source)}


def run_asymmetric(label, precision, dynamic, invert):
    source = subject_grid("Subject_" + label)
    cage = cage_box("Cage_" + label, ASYMMETRIC)
    weights = [0.15 + 0.2 * (i % 5) for i in range(25)]
    return bind_and_read(label, source, cage, precision, dynamic, invert, POST_BIND_MOVES, weights)


def run_nonuniform_translation(label):
    source = subject_grid("Subject_" + label)
    # Rust: box size 6.0 with the cage transform, no vertex group (all vertices weighted), precision 3.
    cage = cage_box("Cage_" + label, [tuple(c * 1.5 for c in corner) for corner in BASE])
    moves = {index: TRANSLATION_CAGE_MOVE for index in range(8)}
    return bind_and_read(label, source, cage, 3, False, False, moves, None)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


results = {}
reset()
results["mesh_asymmetric_static_p2"] = run_asymmetric("mesh_asymmetric_static_p2", 2, False, False)
reset()
results["mesh_asymmetric_static_p3"] = run_asymmetric("mesh_asymmetric_static_p3", 3, False, False)
reset()
results["mesh_asymmetric_dynamic_p3_invert"] = run_asymmetric(
    "mesh_asymmetric_dynamic_p3_invert", 3, True, True)
reset()
results["mesh_nonuniform_translation"] = run_nonuniform_translation("mesh_nonuniform_translation")

with open(out_path(), "w") as handle:
    json.dump(results, handle, indent=1, sort_keys=True)
print("WROTE", out_path(), sorted(results))
