"""Blender 5.2 FLOAT union oracle for the near-coplanar box sliver in tests/modifiers_more.rs.

The proptest `boolean_float_union_of_overlapping_boxes_is_watertight` shrank to one input: a unit-cube
subject `[-1, 1]^3` and an operand box of size `size` centred at `(x, y, z)`. Here the operand's -x face
lies 2.8e-6 outside the subject's -x face, so the union keeps a thin sliver of volume.

Run headless from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/float_sliver/generate_union_sliver.py -- \
        tests/fixtures/boolean_parity/float_sliver

Writes `float_sliver_union.json`: for each threshold, the evaluated mesh's volume, face count, edge
manifold counts, and the analytic union volume in double precision (the test's formula).
"""

import json
import os
import sys

import bmesh
import bpy

OUTPUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])

# The shrunk proptest input, and controls with the same overlap pattern but no sub-threshold gap: the
# control's operand -x face is inside the subject (x - size/2 > -1). Blender stores coordinates as f32.
CASES = [
    {"name": "failing_sliver", "x": -0.2875819305642165, "y": 0.0, "z": 0.0, "size": 1.424841830952486},
    {"name": "control_no_gap", "x": -0.2875819305642165, "y": 0.0, "z": 0.0, "size": 1.3},
    {"name": "control_offset", "x": -0.25, "y": 0.0, "z": 0.0, "size": 1.424841830952486},
]
# Gap sweep: the operand's -x face sits `gap` outside the subject's -x face. The failing input has gap
# 2.846e-6 (x - size/2 + 1 = -gap); the sweep brackets the default double_threshold 1e-6.
SWEEP_SIZE = 1.424841830952486
SWEEP_GAPS = [1.0e-7, 5.0e-7, 1.0e-6, 2.0e-6, 2.846e-6, 5.0e-6, 2.0e-5, 1.0e-4, -1.0e-7, -1.0e-6, -2.846e-6, -2.0e-5, -1.0e-4]
CASES += [
    {"name": f"gap_{gap:g}", "x": -1.0 - gap + SWEEP_SIZE * 0.5, "y": 0.0, "z": 0.0, "size": SWEEP_SIZE}
    for gap in SWEEP_GAPS
]
# 1e-6 is the FLOAT default (`param_table.json`); 0.0 exercises the tree-epsilon clamp.
THRESHOLDS = [1.0e-6, 0.0]


def analytic_union_volume(size, x, y, z):
    """The test's `8 + size^3 - intersection_volume(size, x, y, z)` for a subject of size 2."""
    overlap = 1.0
    for offset in (x, y, z):
        lower = max(offset - size * 0.5, -1.0)
        upper = min(offset + size * 0.5, 1.0)
        overlap *= max(upper - lower, 0.0)
    return 8.0 + size**3 - overlap


def cube(name, size, location):
    mesh = bpy.data.meshes.new(name + ".mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.translate(bm, vec=location, verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def signed_volume_and_topology(mesh):
    mesh.calc_loop_triangles()
    volume = 0.0
    for tri in mesh.loop_triangles:
        a, b, c = (mesh.vertices[i].co for i in tri.vertices)
        volume += (
            a.x * (b.y * c.z - b.z * c.y)
            - a.y * (b.x * c.z - b.z * c.x)
            + a.z * (b.x * c.y - b.y * c.x)
        ) / 6.0
    edge_faces = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            edge_faces[key] = edge_faces.get(key, 0) + 1
    counts = list(edge_faces.values())
    return {
        "volume": volume,
        "faces": len(mesh.polygons),
        "vertices": len(mesh.vertices),
        "edges_not_two_faces": sum(1 for count in counts if count != 2),
    }


def run_case(case, results):
    x, y, z, size = case["x"], case["y"], case["z"], case["size"]
    for threshold in THRESHOLDS:
        subject = cube("subject", 2.0, (0.0, 0.0, 0.0))
        operand = cube("operand", size, (x, y, z))
        modifier = subject.modifiers.new("Boolean", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.solver = "FLOAT"
        modifier.operand_type = "OBJECT"
        modifier.object = operand
        modifier.double_threshold = threshold
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = subject.evaluated_get(depsgraph).to_mesh()
        record = signed_volume_and_topology(evaluated)
        record.update({"case": case["name"], "threshold": threshold, "analytic_volume": analytic_union_volume(size, x, y, z)})
        record["error"] = record["volume"] - record["analytic_volume"]
        results.append(record)
        print(json.dumps(record), flush=True)
        subject.evaluated_get(depsgraph).to_mesh_clear()
        bpy.data.objects.remove(subject, do_unlink=True)
        bpy.data.objects.remove(operand, do_unlink=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    results = []
    for case in CASES:
        run_case(case, results)
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, "float_sliver_union.json"), "w", encoding="utf-8") as handle:
        json.dump({"cases": CASES, "results": results}, handle, indent=2)
        handle.write("\n")


main()
