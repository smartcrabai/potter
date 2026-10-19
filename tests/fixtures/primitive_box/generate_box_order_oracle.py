"""Blender 5.2.2 box (cube) topology oracle: element order of `primitive_cube_add`.

Run: Blender -b --factory-startup --python generate_box_order_oracle.py -- <out.json>

Potter's `box` primitive (`Mesh::box_mesh`) is Blender's `primitive_cube_add`, which runs the BMesh
`create_cube` operator (`bmesh/operators/bmo_primitive.cc`). Each case records the vertex coordinates,
edge endpoint pairs, face vertex lists and flat loop vertex indices in storage order, exactly as
Blender stores them. The `bmesh_round_trip` entry is the same cube after `BMesh.from_mesh` and
`BMesh.to_mesh`, the path `.blend` export takes for a descriptor primitive (`blend_bridge.py`); it must
match the plain entry.
"""
import bmesh
import bpy
import json
import sys

OUT = sys.argv[sys.argv.index("--") + 1]


def dump(mesh):
    return {
        "vertices": [list(vertex.co) for vertex in mesh.vertices],
        "edges": [list(edge.vertices) for edge in mesh.edges],
        "faces": [list(polygon.vertices) for polygon in mesh.polygons],
        "loops": [loop.vertex_index for loop in mesh.loops],
    }


def cube(size):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=size)
    mesh = bpy.context.object.data
    plain = dump(mesh)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    round_trip = bpy.data.meshes.new("box_round_trip")
    bm.to_mesh(round_trip)
    bm.free()
    assert dump(round_trip) == plain, "BMesh round trip changed the cube's storage order"
    return {"size": size, "plain": plain, "bmesh_round_trip": dump(round_trip)}


result = {
    "blender_version": "5.2.2",
    "source": "bpy.ops.mesh.primitive_cube_add -> BMesh create_cube",
    "cases": {"size_2": cube(2.0), "size_2_25": cube(2.25)},
}
with open(OUT, "w", encoding="utf-8") as destination:
    json.dump(result, destination, indent=1)
    destination.write("\n")
print("CASES", len(result["cases"]))
