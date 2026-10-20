"""Blender 5.2.2 oracle for the tests/pr18f_deform_bind.rs Mesh Deform scenes.

Rebuilds the exact Rust scenes:
- `asymmetric_mesh_operations`: 4x4-subdivision grid subject, box cage (size 4) with the eight cage
  vertices moved to asymmetric positions before bind, the `Influence` group, MESH_DEFORM at the
  given precision, bind, then three post-bind cage vertex moves (cases `mesh_asymmetric_*`).
- `mesh_deform_nonuniform_bind_transform_matches_blender_translation`: the same subject grid, a box
  cage of size 6 (cage transform set), bind at precision 3, then all eight cage vertices translated
  by CAGE_TRANSLATION after bind (case `mesh_nonuniform_translation`).

Subdivision cases: `mesh_asymmetric_subdiv_p3` is the `mesh_asymmetric_*` scene with a Subdivision level-1 modifier
before the Mesh Deform bind. `cor_grid_bind` and `cor_grid_subdiv_bind` bind a Corrective Smooth modifier (factor 0.5,
5 iterations, scale 1, SIMPLE, rest source BIND) on the flat grid, or after the same Subdivision level-1 modifier,
and then move four base vertices (`CORRECTIVE_MOVES`).

Writes the evaluated subject positions for every case to OUT.json. The committed copy is
tests/fixtures/pr18f_deform_bind/mesh_deform_blender_oracle.json, read by tests/pr18f_deform_bind.rs.

Rows that are analytic, not Blender-measured, are not generated here. The tests compute them directly:
uniform cage translation, the inverted and dynamic group scalings, the missing vertex group, the ignored
cage and owner motion, and the nonuniform translation constant (its Blender value is the
`mesh_nonuniform_translation` case above).

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
    (-2.0, -2.0, -2.0), (-2.0, -2.0, 2.0), (-2.0, 2.0, -2.0), (-2.0, 2.0, 2.0),
    (2.0, -2.0, -2.0), (2.0, -2.0, 2.0), (2.0, 2.0, -2.0), (2.0, 2.0, 2.0),
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
# Base-vertex moves after the Corrective Smooth bind (`run_corrective`).
CORRECTIVE_MOVES = {0: (0.0, 0.0, 0.12), 6: (-0.04, 0.08, -0.06), 12: (0.0, 0.0, 0.25), 18: (0.1, -0.03, 0.0)}

# Rust `CAGE_TRANSLATION` used by the nonuniform translation test (cage-local, all eight vertices).
TRANSLATION_CAGE_MOVE = (0.3, -0.2, 0.5)


def set_transform(obj, translation, rotation_xyzw, scale):
    obj.location = Vector(translation)
    q = Quaternion((rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]))
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = q
    obj.scale = Vector(scale)


def subject_grid(name, transform=True):
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
    if transform:
        set_transform(obj, SUBJECT_TRANSLATION, SUBJECT_ROTATION, SUBJECT_SCALE)
    return obj


def cage_box(name, corners, transform=True):
    # Blender's own cube (`primitive_cube_add`) gives the vertex order and face list Potter's
    # `Mesh::box_mesh` follows. Mesh Deform's index-based cage moves and corner triangulation depend on
    # that order, so the corners are assigned by index onto Blender's primitive.
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "Mesh"
    for vertex, corner in zip(obj.data.vertices, corners):
        vertex.co = corner
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


def bind_and_read(label, source, cage, precision, dynamic, invert, moves, group_weights,
                  subdivide=False):
    if subdivide:
        # Blender binds on the mesh evaluated before the bound modifier, so this level-1 subdivision is part of
        # the bind input.
        subdivision = source.modifiers.new("Subdivision", "SUBSURF")
        subdivision.levels = 1
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


def run_asymmetric(label, precision, dynamic, invert, subdivide=False):
    source = subject_grid("Subject_" + label)
    cage = cage_box("Cage_" + label, ASYMMETRIC)
    weights = [0.15 + 0.2 * (i % 5) for i in range(25)]
    return bind_and_read(
        label, source, cage, precision, dynamic, invert, POST_BIND_MOVES, weights,
        subdivide=subdivide)


def run_corrective(label, subdivide):
    # Rust: grid size 2.0 with 4x4 subdivisions and no object transform. The modifier is bound on the flat
    # grid, or after a Subdivision level-1 modifier when `subdivide`; then four base vertices move.
    source = subject_grid(label, transform=False)
    if subdivide:
        subdivision = source.modifiers.new("Subdivision", "SUBSURF")
        subdivision.levels = 1
    smooth = source.modifiers.new("Smooth", "CORRECTIVE_SMOOTH")
    smooth.factor = 0.5
    smooth.iterations = 5
    smooth.scale = 1.0
    smooth.smooth_type = 'SIMPLE'
    smooth.rest_source = 'BIND'
    activate(source)
    bpy.ops.object.correctivesmooth_bind(modifier=smooth.name)
    bpy.context.view_layer.update()
    for index, delta in CORRECTIVE_MOVES.items():
        source.data.vertices[index].co += Vector(delta)
    source.data.update()
    return {"bound": bool(smooth.is_bind), "positions": evaluated_positions(source)}


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
reset()
results["cor_grid_bind"] = run_corrective("cor_grid_bind", False)
reset()
results["cor_grid_subdiv_bind"] = run_corrective("cor_grid_subdiv_bind", True)
reset()
results["mesh_asymmetric_subdiv_p3"] = run_asymmetric(
    "mesh_asymmetric_subdiv_p3", 3, False, False, subdivide=True)

with open(out_path(), "w") as handle:
    json.dump(results, handle, indent=1, sort_keys=True)
print("WROTE", out_path(), sorted(results))
