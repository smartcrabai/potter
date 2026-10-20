"""Blender 5.2.2 torus oracle for meshes of 1000 faces or more, built with one edge hash map.

Run: Blender -b --factory-startup --threads 1 --python generate_torus_single_map_oracle.py -- <out.json>

`primitive_torus_add` builds its mesh from Python vertices and faces, and Blender then runs
`mesh_calc_edges` (`blenkernel/intern/mesh_calc_edges.cc`). From 1000 faces, that function splits
the edges into `get_parallel_maps_count` hash maps: min(8, BLI_system_thread_count()), rounded down
to a power of two. The edge order therefore depends on the thread count of the host, while the
edge set does not. `--threads 1` gives one map, the order the Potter torus builder reproduces. The
case records that thread count, so the oracle is checked only against a single-map run.
"""
import bpy
import json
import sys

OUT = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_torus_add(
    major_segments=48, minor_segments=24, major_radius=1.0, minor_radius=0.25
)
mesh = bpy.context.object.data
case = {
    "vertices": [list(v.co) for v in mesh.vertices],
    "edges": [list(e.vertices) for e in mesh.edges],
    "faces": [list(p.vertices) for p in mesh.polygons],
    "loops": [loop.vertex_index for loop in mesh.loops],
    "uv_layers": [layer.name for layer in mesh.uv_layers],
    "uv": [list(d.uv) for d in mesh.uv_layers[0].data] if mesh.uv_layers else [],
    "kind": "torus",
    "operator": "mesh.primitive_torus_add",
    "kwargs": {"major_segments": 48, "minor_segments": 24, "major_radius": 1.0, "minor_radius": 0.25},
}
with open(OUT, "w", encoding="utf-8") as destination:
    json.dump({"blender_version": "5.2.2", "cases": {"torus_48x24_single_map": case}}, destination, indent=1)
    destination.write("\n")
print("CASES 1", len(case["faces"]), "faces")
