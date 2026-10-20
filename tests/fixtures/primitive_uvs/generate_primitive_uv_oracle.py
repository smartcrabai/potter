"""Blender 5.2.2 primitive oracle: element order and calc_uvs UV layers of `primitive_*_add`.

Run: Blender -b --factory-startup --python generate_primitive_uv_oracle.py -- <out.json>

Each case calls one `bpy.ops.mesh.primitive_*_add` operator with its UV property left at the default (calc_uvs
True where the operator has it; torus has generate_uvs, default True) and records, in storage order: vertex
coordinates, edge endpoint pairs, face vertex lists, flat loop vertex indices, and the first UV map's values
per loop. The parameters are Potter's `node.create` names mapped to the operator's RNA names, so each case
states the same geometry on both sides.
"""
import bpy
import json
import sys

OUT = sys.argv[sys.argv.index("--") + 1]

# (case name, Potter kind, operator, operator kwargs)
CASES = [
    ("box_size_2", "box", bpy.ops.mesh.primitive_cube_add, {"size": 2.0}),
    ("plane_size_2", "plane", bpy.ops.mesh.primitive_plane_add, {"size": 2.0}),
    ("grid_2x2", "grid", bpy.ops.mesh.primitive_grid_add,
     {"x_subdivisions": 2, "y_subdivisions": 2, "size": 2.0}),
    ("uv_sphere_8x4", "uv_sphere", bpy.ops.mesh.primitive_uv_sphere_add,
     {"segments": 8, "ring_count": 4, "radius": 1.0}),
    ("uv_sphere_default", "uv_sphere", bpy.ops.mesh.primitive_uv_sphere_add, {}),
    ("ico_sphere_1", "icosphere", bpy.ops.mesh.primitive_ico_sphere_add,
     {"subdivisions": 1, "radius": 1.0}),
    ("ico_sphere_3", "icosphere", bpy.ops.mesh.primitive_ico_sphere_add,
     {"subdivisions": 3, "radius": 1.0}),
    ("ico_sphere_4", "icosphere", bpy.ops.mesh.primitive_ico_sphere_add,
     {"subdivisions": 4, "radius": 1.0}),
    ("ico_sphere_default", "icosphere", bpy.ops.mesh.primitive_ico_sphere_add, {}),
    ("cylinder_8", "cylinder", bpy.ops.mesh.primitive_cylinder_add,
     {"vertices": 8, "radius": 1.0, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cylinder_4_ngon", "cylinder", bpy.ops.mesh.primitive_cylinder_add,
     {"vertices": 4, "radius": 1.0, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cone_trunc_4_ngon", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 4, "radius1": 1.0, "radius2": 0.5, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cylinder_default", "cylinder", bpy.ops.mesh.primitive_cylinder_add, {}),
    ("cone_8", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 1.0, "radius2": 0.0, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cone_default", "cone", bpy.ops.mesh.primitive_cone_add, {}),
    ("circle_8", "circle", bpy.ops.mesh.primitive_circle_add,
     {"vertices": 8, "radius": 1.0, "fill_type": "NOTHING"}),
    ("circle_ngon_5", "circle", bpy.ops.mesh.primitive_circle_add,
     {"vertices": 5, "radius": 0.75, "fill_type": "NGON"}),
    ("torus_8x4", "torus", bpy.ops.mesh.primitive_torus_add,
     {"major_segments": 8, "minor_segments": 4, "major_radius": 1.0, "minor_radius": 0.25, "generate_uvs": True}),
    ("torus_default", "torus", bpy.ops.mesh.primitive_torus_add, {}),
    ("circle_trifan_5", "circle", bpy.ops.mesh.primitive_circle_add,
     {"vertices": 5, "radius": 0.75, "fill_type": "TRIFAN"}),
    ("cylinder_8_trifan", "cylinder", bpy.ops.mesh.primitive_cylinder_add,
     {"vertices": 8, "radius": 1.0, "depth": 2.0, "end_fill_type": "TRIFAN"}),
    ("cylinder_8_nothing", "cylinder", bpy.ops.mesh.primitive_cylinder_add,
     {"vertices": 8, "radius": 1.0, "depth": 2.0, "end_fill_type": "NOTHING"}),
    ("cone_trunc_8", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 1.0, "radius2": 0.5, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cone_8_trifan", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 1.0, "radius2": 0.0, "depth": 2.0, "end_fill_type": "TRIFAN"}),
    ("cone_8_nothing", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 1.0, "radius2": 0.0, "depth": 2.0, "end_fill_type": "NOTHING"}),
    ("cone_inverted_8_trifan", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 0.0, "radius2": 1.0, "depth": 2.0, "end_fill_type": "TRIFAN"}),
    ("cone_inverted_8_nothing", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 0.0, "radius2": 1.0, "depth": 2.0, "end_fill_type": "NOTHING"}),
    ("cone_inverted_32", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 32, "radius1": 0.0, "radius2": 1.0, "depth": 2.0, "end_fill_type": "NGON"}),
    ("cone_inverted_8", "cone", bpy.ops.mesh.primitive_cone_add,
     {"vertices": 8, "radius1": 0.0, "radius2": 1.0, "depth": 2.0, "end_fill_type": "NGON"}),
]


def dump(obj):
    mesh = obj.data
    layer = mesh.uv_layers[0] if len(mesh.uv_layers) else None
    return {
        "vertices": [list(v.co) for v in mesh.vertices],
        "edges": [list(e.vertices) for e in mesh.edges],
        "faces": [list(p.vertices) for p in mesh.polygons],
        "loops": [loop.vertex_index for loop in mesh.loops],
        "uv_layers": [l.name for l in mesh.uv_layers],
        "uv": [list(d.uv) for d in layer.data] if layer else [],
    }


# Blender's extrude_edge_only iterates boundary_map.out, a GHash keyed by BMEdge pointers
# (bmesh_operators.cc:100 creates the slot; bmo_extrude.cc bmo_extrude_edge_only_exec iterates it).
# Pointer-hash bucket order varies between processes, so UV sphere edge and face order does too.
RUN_DEPENDENT_SOURCE = "bmesh_operators.cc:100; bmo_extrude.cc bmo_extrude_edge_only_exec boundary_map.out"

cases = {}
for name, kind, op, kwargs in CASES:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    op(**kwargs)
    obj = bpy.context.object
    entry = dump(obj)
    entry["kind"] = kind
    entry["operator"] = op.idname()
    entry["kwargs"] = kwargs
    if kind == "uv_sphere":
        # Vertex order is reproducible across processes; edge and face order is not (see RUN_DEPENDENT_SOURCE).
        entry["edge_face_order"] = "run_dependent"
        entry["edge_face_order_source"] = RUN_DEPENDENT_SOURCE
    cases[name] = entry

with open(OUT, "w", encoding="utf-8") as destination:
    json.dump({"blender_version": "5.2.2", "cases": cases}, destination, indent=1)
    destination.write("\n")
print("CASES", len(cases))
