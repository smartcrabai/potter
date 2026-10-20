"""Blender 5.2 oracle for tests/modifier_purpose_eval.rs.

Run headless from the repository root (`--factory-startup` keeps the run reproducible):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/purpose_eval/generate_purpose_eval_expectations.py -- \
        tests/fixtures/purpose_eval/expected.json

The optional path after `--` is the output file; it defaults to `expected.json` beside this script.

The scene is the one the Rust test builds. `host` is a box of size 2 with a Boolean DIFFERENCE
(EXACT, object `cutter`, a box of size 1 at the origin) followed by a Decimate (COLLAPSE, ratio 0.6).
`follower` is a box of size 1 with a Boolean UNION (FLOAT, object `host`). Every object keeps the
identity transform, so local and world coordinates coincide.

Each object is measured under two purposes:

- `preview`: the viewport depsgraph, read through `obj.evaluated_get()` and `to_mesh()`.
- `render`: the render depsgraph, read through `wm.obj_export` with `export_eval_mode`
  `DAG_EVAL_RENDER`. Python exposes no render depsgraph: `Depsgraph.mode` is read-only, and the
  `render_pre` handler still sees the viewport depsgraph. The exporter is the only Python path that
  evaluates in render mode.

OBJ writes six decimals, so the export is scaled by 1000 and the parser divides it back. That keeps
1e-9 scene units of resolution. The viewport export's counts are checked against `to_mesh()`, so a
parser error stops the run instead of reaching the expectations.

Volume is the divergence-theorem sum over each polygon's fan from its first corner. That sum is the
same for every triangulation only when the polygon is planar, and OBJ carries no triangulation, so
the generator refuses non-planar or zero-area polygons rather than choosing a triangulation.

Each entry records `vertices`, `edges` (unique undirected edges of the polygons plus loose edges),
`faces`, and `volume` (positive for outward winding).
"""

import json
import math
import os
import sys
import tempfile

import bpy

VIEWPORT = "DAG_EVAL_VIEWPORT"
RENDER = "DAG_EVAL_RENDER"
# OBJ writes coordinates with six decimals; this factor keeps 1e-9 scene units of resolution.
OBJ_SCALE = 1000.0
# Scene coordinates are f32; this admits f32 and OBJ rounding and nothing more.
PLANARITY_TOLERANCE = 1.0e-6
PURPOSES = ("preview", "render")


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def assert_planar(points, polygon):
    # Newell's normal is robust for n-gons; a zero normal means the polygon has no area.
    normal = [0.0, 0.0, 0.0]
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal[0] += (point[1] - following[1]) * (point[2] + following[2])
        normal[1] += (point[2] - following[2]) * (point[0] + following[0])
        normal[2] += (point[0] - following[0]) * (point[1] + following[1])
    length = math.sqrt(dot(normal, normal))
    if length == 0.0:
        raise RuntimeError(f"polygon {polygon} has zero area; its volume is not defined here")
    unit = [component / length for component in normal]
    centroid = [math.fsum(point[axis] for point in points) / len(points) for axis in range(3)]
    deviation = max(abs(dot(sub(point, centroid), unit)) for point in points)
    if deviation > PLANARITY_TOLERANCE:
        raise RuntimeError(f"polygon {polygon} is not planar (deviation {deviation})")


def measure_polygons(vertices, polygons, edge_count):
    terms = []
    for polygon in polygons:
        points = [vertices[index] for index in polygon]
        assert_planar(points, polygon)
        for k in range(1, len(points) - 1):
            terms.append(dot(points[0], cross(points[k], points[k + 1])))
    return {
        "vertices": len(vertices),
        "edges": edge_count,
        "faces": len(polygons),
        "volume": math.fsum(terms) / 6.0,
    }


def measure_viewport(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        vertices = [tuple(float(c) for c in vertex.co) for vertex in mesh.vertices]
        polygons = [
            [mesh.loops[index].vertex_index for index in polygon.loop_indices]
            for polygon in mesh.polygons
        ]
        return measure_polygons(vertices, polygons, len(mesh.edges))
    finally:
        evaluated.to_mesh_clear()


def export_obj(obj, mode, directory):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    path = os.path.join(directory, f"{obj.name}.{mode}.obj")
    bpy.ops.wm.obj_export(
        filepath=path,
        check_existing=False,
        export_selected_objects=True,
        export_eval_mode=mode,
        apply_modifiers=True,
        apply_transform=False,
        global_scale=OBJ_SCALE,
        forward_axis="Y",
        up_axis="Z",
    )
    return path


def read_obj(path):
    vertices, polygons, edges = [], [], set()
    with open(path, encoding="utf-8") as text:
        for line in text:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append(tuple(float(c) / OBJ_SCALE for c in fields[1:4]))
            elif fields[0] == "f":
                indices = [int(field.split("/")[0]) - 1 for field in fields[1:]]
                polygons.append(indices)
                for a, b in zip(indices, indices[1:] + indices[:1]):
                    edges.add((min(a, b), max(a, b)))
            elif fields[0] == "l":
                indices = [int(field) - 1 for field in fields[1:]]
                for a, b in zip(indices, indices[1:]):
                    edges.add((min(a, b), max(a, b)))
    return measure_polygons(vertices, polygons, len(edges))


def check_export(obj, directory):
    # The OBJ parser must reproduce the depsgraph counts; a mismatch means the parser is wrong.
    expected = measure_viewport(obj)
    exported = read_obj(export_obj(obj, VIEWPORT, directory))
    for key in ("vertices", "edges", "faces"):
        if expected[key] != exported[key]:
            raise RuntimeError(
                f"{obj.name} {key}: depsgraph {expected[key]}, viewport export {exported[key]}"
            )


def measure(purpose, obj, directory):
    if purpose == "preview":
        return measure_viewport(obj)
    return read_obj(export_obj(obj, RENDER, directory))


def set_visibility(modifier, enabled):
    # Blender reads show_viewport for the viewport depsgraph and show_render for the render one.
    modifier.show_viewport = enabled
    modifier.show_render = enabled


def add_box(name, size):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cutter = add_box("cutter", 1.0)
    host = add_box("host", 2.0)
    follower = add_box("follower", 1.0)

    boolean = host.modifiers.new("Boolean", "BOOLEAN")
    boolean.operation = "DIFFERENCE"
    boolean.solver = "EXACT"
    boolean.operand_type = "OBJECT"
    boolean.object = cutter

    decimate = host.modifiers.new("Decimate", "DECIMATE")
    decimate.decimate_type = "COLLAPSE"
    decimate.ratio = 0.6

    follow = follower.modifiers.new("Boolean", "BOOLEAN")
    follow.operation = "UNION"
    follow.solver = "FLOAT"
    follow.operand_type = "OBJECT"
    follow.object = host

    return {
        "cutter": cutter,
        "host": host,
        "follower": follower,
        "boolean": boolean,
        "decimate": decimate,
        "follow": follow,
    }


def measure_purpose(purpose, scene, directory):
    set_visibility(scene["boolean"], True)
    set_visibility(scene["follow"], True)
    # The Boolean-only host is the host with its Decimate disabled for both purposes.
    set_visibility(scene["decimate"], False)
    boolean_host = measure(purpose, scene["host"], directory)
    set_visibility(scene["decimate"], True)
    return {
        "cutter": measure(purpose, scene["cutter"], directory),
        "boolean_host": boolean_host,
        "decimated_host": measure(purpose, scene["host"], directory),
        "follower": measure(purpose, scene["follower"], directory),
    }


def main():
    scene = build_scene()
    if "--" in sys.argv:
        output = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
    else:
        output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "expected.json")

    results = {}
    with tempfile.TemporaryDirectory() as directory:
        set_visibility(scene["boolean"], True)
        set_visibility(scene["decimate"], True)
        check_export(scene["host"], directory)
        for purpose in PURPOSES:
            results[purpose] = measure_purpose(purpose, scene, directory)
            for name, entry in results[purpose].items():
                print(
                    f"{purpose} {name}: V={entry['vertices']} E={entry['edges']} "
                    f"F={entry['faces']} volume={entry['volume']:.9f}",
                    flush=True,
                )

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print("WROTE", output, flush=True)


main()
