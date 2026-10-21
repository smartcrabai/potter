"""PR9 oracle reader: dumps every mesh object of the open .blend as JSON (Blender 5.2.2).

Usage:
  Blender --factory-startup -b FILE.blend --python dump_blend.py -- OUT.json

Values are Blender's own: `corner_normals` is Blender's computed corner normal, `codes` is the raw
`custom_normal` INT16_2D pair per loop (None when the mesh has no custom normals), colors are
`Color Attribute.color` (linear), and material nodes and links are listed by node name.
"""

import json
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]


def plain(item):
    if hasattr(item, "vector"):
        return [float(c) for c in item.vector]
    if hasattr(item, "color"):
        return [float(c) for c in item.color]
    if hasattr(item, "value"):
        value = item.value
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return [float(c) if isinstance(c, float) else int(c) for c in value]
    return None


def attribute_dump(attribute):
    return {
        "name": attribute.name,
        "domain": attribute.domain,
        "data_type": attribute.data_type,
        "values": [plain(item) for item in attribute.data],
    }


def material_dump(material):
    if material is None:
        return None
    result = {"name": material.name, "nodes": [], "links": []}
    if not material.use_nodes or material.node_tree is None:
        return result
    tree = material.node_tree
    for node in tree.nodes:
        entry = {"name": node.name, "type": node.bl_idname}
        if node.bl_idname == "ShaderNodeTexImage":
            entry["image"] = node.image.name if node.image else None
            entry["interpolation"] = node.interpolation
            entry["colorspace"] = node.image.colorspace_settings.name if node.image else None
        if node.bl_idname == "ShaderNodeUVMap":
            entry["uv_map"] = node.uv_map
        if node.bl_idname == "ShaderNodeVertexColor":
            entry["layer_name"] = node.layer_name
        if node.bl_idname == "ShaderNodeNormalMap":
            entry["uv_map"] = node.uv_map
            entry["strength"] = float(node.inputs["Strength"].default_value)
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            entry["inputs"] = {}
            for socket in node.inputs:
                if socket.enabled and hasattr(socket, "default_value"):
                    value = socket.default_value
                    try:
                        entry["inputs"][socket.name] = [float(c) for c in value]
                    except TypeError:
                        entry["inputs"][socket.name] = float(value)
        result["nodes"].append(entry)
    for link in tree.links:
        result["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })
    return result


objects = []
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    mesh = obj.data
    loops = [
        {"vertex": loop.vertex_index, "edge": loop.edge_index}
        for loop in mesh.loops
    ]
    codes_attribute = mesh.attributes.get("custom_normal")
    codes = [list(item.value) for item in codes_attribute.data] if codes_attribute else None
    corner_normals = [[float(c) for c in n.vector] for n in mesh.corner_normals]
    uv_layers = []
    for layer in mesh.uv_layers:
        uv_layers.append({
            "name": layer.name,
            "values": [[float(c) for c in item.uv] for item in layer.data],
        })
    colors = []
    for color in mesh.color_attributes:
        colors.append({
            "name": color.name,
            "domain": color.domain,
            "data_type": color.data_type,
            "values": [[float(c) for c in item.color] for item in color.data],
        })
    generic = [
        attribute_dump(attribute)
        for attribute in mesh.attributes
        if not attribute.name.startswith(".")
        and attribute.name not in ("position", "sharp_face", "sharp_edge", "custom_normal", "UVMap")
        and attribute.name not in [c.name for c in mesh.color_attributes]
    ]
    groups = []
    for group in obj.vertex_groups:
        members = {}
        for vertex in mesh.vertices:
            for entry in vertex.groups:
                if entry.group == group.index:
                    members[str(vertex.index)] = float(entry.weight)
        groups.append({"name": group.name, "weights": members})
    objects.append({
        "object": obj.name,
        "mesh": mesh.name,
        "location": [float(c) for c in obj.location],
        "vertices": [[float(c) for c in v.co] for v in mesh.vertices],
        "edges": [
            {"v": list(edge.vertices), "sharp": bool(edge.use_edge_sharp), "seam": bool(edge.use_seam)}
            for edge in mesh.edges
        ],
        "faces": [
            {
                "v": list(poly.vertices),
                "smooth": bool(poly.use_smooth),
                "loop_start": poly.loop_start,
                "loop_total": poly.loop_total,
                "center": [float(c) for c in poly.center],
            }
            for poly in mesh.polygons
        ],
        "loops": loops,
        "codes": codes,
        "corner_normals": corner_normals,
        "uv_layers": uv_layers,
        "uv_active": mesh.uv_layers.active.name if mesh.uv_layers.active else None,
        "uv_render": next((layer.name for layer in mesh.uv_layers if layer.active_render), None),
        "color_layers": colors,
        "color_active": mesh.color_attributes.active_color_index,
        "color_render": mesh.color_attributes.render_color_index,
        "attributes": generic,
        "vertex_groups": groups,
        "materials": [material_dump(slot.material) for slot in obj.material_slots],
        "modifiers": [modifier.type for modifier in obj.modifiers],
        "texspace": {
            "auto": bool(mesh.use_auto_texspace),
            "location": [float(c) for c in mesh.texspace_location],
            "size": [float(c) for c in mesh.texspace_size],
        },
    })

result = {
    "objects": objects,
    "images": [
        {"name": image.name, "filepath": image.filepath, "packed": image.packed_file is not None,
         "colorspace": image.colorspace_settings.name}
        for image in bpy.data.images
    ],
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
