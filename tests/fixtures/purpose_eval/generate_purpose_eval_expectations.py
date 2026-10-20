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

Volume is the divergence-theorem sum over Blender's loop triangles (`Mesh.calc_loop_triangles`),
Blender's own triangulation. The OBJ export carries no triangulation, and a decimated polygon can be
non-planar, where a fan of its corners depends on the diagonal chosen. So the render pass takes its
volume from the viewport loop triangles, after checking that the render export reproduces the viewport
mesh (same vertices, polygons and edges, within 1e-6 scene units). If the render evaluation differs,
the generator stops rather than recording a volume for a different mesh.

OBJ writes six decimals, so the export is scaled by 1000 and the parser divides it back. The viewport
export's counts are checked against `to_mesh()`, so a parser error stops the run.

Each entry records `vertices`, `edges` (`Mesh.edges`), `faces` (polygons), and `volume` (positive for
outward winding).
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
# Vertex positions must agree within this many scene units between the render export and the
# viewport mesh. Scene coordinates are f32; the export round trip is well inside this bound.
GEOMETRY_TOLERANCE = 1.0e-6
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


def volume(vertices, triangles):
    terms = [dot(vertices[a], cross(vertices[b], vertices[c])) for a, b, c in triangles]
    return math.fsum(terms) / 6.0


def summarize(geometry):
    return {
        "vertices": len(geometry["vertices"]),
        "edges": geometry["edges"],
        "faces": len(geometry["polygons"]),
        "volume": volume(geometry["vertices"], geometry["triangles"]),
    }


def evaluated_geometry(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        return {
            "vertices": [tuple(float(c) for c in vertex.co) for vertex in mesh.vertices],
            "polygons": [
                [mesh.loops[index].vertex_index for index in polygon.loop_indices]
                for polygon in mesh.polygons
            ],
            "triangles": [tuple(triangle.vertices) for triangle in mesh.loop_triangles],
            "edges": len(mesh.edges),
        }
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
    return {"vertices": vertices, "polygons": polygons, "edges": len(edges)}


def ring_points(geometry, ring):
    return [geometry["vertices"][index] for index in ring]


def same_point(p, q):
    return max(abs(a - b) for a, b in zip(p, q)) <= GEOMETRY_TOLERANCE


def same_ring(p, q):
    # A ring matches another under a cyclic shift and reversal, with each vertex within tolerance.
    if len(p) != len(q):
        return False
    count = len(p)
    for candidate in (q, q[::-1]):
        for shift in range(count):
            if all(same_point(p[i], candidate[(i + shift) % count]) for i in range(count)):
                return True
    return False


def same_geometry(exported, viewport):
    if len(exported["vertices"]) != len(viewport["vertices"]):
        return False
    if len(exported["polygons"]) != len(viewport["polygons"]):
        return False
    if exported["edges"] != viewport["edges"]:
        return False
    unmatched = [ring_points(viewport, ring) for ring in viewport["polygons"]]
    for ring in exported["polygons"]:
        points = ring_points(exported, ring)
        for index, candidate in enumerate(unmatched):
            if same_ring(points, candidate):
                del unmatched[index]
                break
        else:
            return False
    return not unmatched


def check_export(obj, directory):
    # The OBJ parser must reproduce the depsgraph counts; a mismatch means the parser is wrong.
    viewport = evaluated_geometry(obj)
    exported = read_obj(export_obj(obj, VIEWPORT, directory))
    checks = (
        ("vertices", len(viewport["vertices"]), len(exported["vertices"])),
        ("faces", len(viewport["polygons"]), len(exported["polygons"])),
        ("edges", viewport["edges"], exported["edges"]),
    )
    for key, depsgraph, export in checks:
        if depsgraph != export:
            raise RuntimeError(f"{obj.name} {key}: depsgraph {depsgraph}, viewport export {export}")


def measure(purpose, obj, directory):
    viewport = evaluated_geometry(obj)
    if purpose == "render":
        exported = read_obj(export_obj(obj, RENDER, directory))
        if not same_geometry(exported, viewport):
            raise RuntimeError(
                f"{obj.name}: render evaluation differs from the viewport mesh; "
                "the render volume would describe a different mesh"
            )
    return summarize(viewport)


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
