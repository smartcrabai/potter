"""Dumps collection instance faces and per-object attributes for Blender oracle tests.

Run as: blender --background --factory-startup --python blend_collection_instances_dump.py -- <input.blend> <output.json>

Evaluated face records preserve material slot/name, UV layer names, loop count, and each corner's
world position, normal, and UV values. Object-level color, vertex-group, and seam records remain
position-keyed so storage-order changes do not affect the comparison.
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

def evaluated_instance_faces():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    faces = []
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != "MESH":
            continue
        mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        matrix = instance.matrix_world
        normal_matrix = matrix.to_3x3().inverted_safe().transposed()
        uv_layers = list(mesh.uv_layers)
        materials = list(mesh.materials)
        for polygon in mesh.polygons:
            slot = min(polygon.material_index, len(materials) - 1) if materials else None
            material = materials[slot] if slot is not None else None
            corners = []
            for loop_index in polygon.loop_indices:
                loop = mesh.loops[loop_index]
                position = matrix @ mesh.vertices[loop.vertex_index].co
                normal = normal_matrix @ mesh.corner_normals[loop_index].vector
                if normal.length:
                    normal.normalize()
                uvs = [rounded(layer.data[loop_index].uv) for layer in uv_layers]
                corners.append({
                    "position": rounded(position),
                    "normal": rounded(normal),
                    "uvs": uvs,
                })
            faces.append({
                "material": material.name if material is not None else None,
                "slot": slot,
                "uv_layers": [layer.name for layer in uv_layers],
                "corners": corners,
            })
        obj.to_mesh_clear()
    return faces


instance_faces = evaluated_instance_faces()


with open(ARGUMENTS[1], "w", encoding="utf-8") as output:
    json.dump({"objects": objects, "instance_faces": instance_faces}, output, sort_keys=True)
