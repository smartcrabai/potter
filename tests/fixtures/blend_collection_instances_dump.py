"""Per-object color, vertex group, and seam dump for tests/blend_collection_instances.rs.

Run as: blender --background --factory-startup --python blend_collection_instances_dump.py -- <input.blend> <output.json>

Records are position-keyed so vertex and edge order may differ between writers: a color record is
[x, y, z, r, g, b, a] per corner, a vertex group record is [x, y, z, weight] per group, and a seam
record is the sorted pair of its end positions.
"""

import json
import sys

import bpy

ARGUMENTS = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=ARGUMENTS[0])


def rounded(values):
    return [round(float(value), 6) for value in values]


objects = {}
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    mesh = obj.data
    colors = {}
    for attribute in mesh.color_attributes:
        records = []
        if attribute.domain == "CORNER":
            for loop in mesh.loops:
                co = mesh.vertices[loop.vertex_index].co
                records.append(rounded([*co, *attribute.data[loop.index].color]))
        else:
            for vertex in mesh.vertices:
                records.append(rounded([*vertex.co, *attribute.data[vertex.index].color]))
        colors[attribute.name] = {"domain": attribute.domain, "records": records}
    groups = {}
    for vertex in mesh.vertices:
        for entry in vertex.groups:
            name = obj.vertex_groups[entry.group].name
            groups.setdefault(name, []).append(rounded([*vertex.co, entry.weight]))
    seams = []
    for edge in mesh.edges:
        if edge.use_seam:
            first = tuple(round(float(value), 6) for value in mesh.vertices[edge.vertices[0]].co)
            second = tuple(round(float(value), 6) for value in mesh.vertices[edge.vertices[1]].co)
            seams.append(list(min(first, second) + max(first, second)))
    objects[obj.name] = {"colors": colors, "vertex_groups": groups, "seams": seams}

with open(ARGUMENTS[1], "w", encoding="utf-8") as output:
    json.dump({"objects": objects}, output, sort_keys=True)
