"""Dump one Blender 5.2.2 UV sphere exactly as the primitive emits it, as JSON.

Blender's UV sphere primitive keeps the same vertex positions in every process but emits its
vertices, edges, and faces in an order that changes from run to run. Each dump is one fixed Boolean
input (vertex, edge, and face order) and is committed under `inputs/` so the regression reproduces
that order.

Run once per dump from the repository root (each run is a fresh Blender process):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/regressions/dump_sphere_input.py -- <output.json>
"""

import json
import sys

import bpy

OUTPUT = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
mesh = bpy.context.object.data
data = {
    "verts": [list(vertex.co) for vertex in mesh.vertices],
    "edges": [list(edge.vertices) for edge in mesh.edges],
    "faces": [list(polygon.vertices) for polygon in mesh.polygons],
}
with open(OUTPUT, "w") as handle:
    json.dump(data, handle)
print("DUMPED", OUTPUT, len(data["verts"]), len(data["edges"]), len(data["faces"]), flush=True)
