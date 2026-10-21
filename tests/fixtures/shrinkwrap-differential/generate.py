import json
import math
import sys
from pathlib import Path

import bpy

def set_transform(obj, raw):
    obj.location = raw["translation"]
    if "rotation_quaternion_xyzw" in raw:
        x, y, z, w = raw["rotation_quaternion_xyzw"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = [w, x, y, z]
    else:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = [math.radians(value) for value in raw["rotation_euler_deg"]]
    obj.scale = raw["scale"]


def make_mesh_object(name, raw, source=False):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(raw["vertices"], [], raw["faces"])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    set_transform(obj, raw["transform"])

    attributes = raw.get("attributes", {})
    for polygon, smooth in zip(mesh.polygons, attributes.get("shade_smooth", [])):
        polygon.use_smooth = smooth
    if attributes.get("shade_smooth_operator"):
        # Mesh.shade_smooth removes Blender's sharp_face attribute; per-polygon use_smooth keeps it.
        mesh.shade_smooth()
    for pair in raw.get("sharp_edges", []):
        key = tuple(sorted(pair))
        for edge in mesh.edges:
            if tuple(sorted(edge.vertices)) == key:
                edge.use_edge_sharp = True
    for entry in attributes.get("uv_map", []):
        layer_name = entry.get("layer", "UVMap")
        layer = mesh.uv_layers.get(layer_name) or mesh.uv_layers.new(name=layer_name)
        polygon = mesh.polygons[entry["face_id"]]
        for loop_index, uv in zip(polygon.loop_indices, entry["uv"]):
            layer.data[loop_index].uv = uv
    for attribute in attributes.get("blender_attributes", []):
        layer = mesh.color_attributes.new(
            name=attribute["name"], type=attribute["data_type"], domain=attribute["domain"]
        )
        for item, value in zip(layer.data, attribute["values"]):
            if attribute["data_type"] == "FLOAT_COLOR":
                item.color = value
            elif attribute["data_type"] == "BYTE_COLOR":
                item.color = value
            else:
                raise ValueError("unsupported recorded color attribute type: " + attribute["data_type"])

    if source and raw.get("vertex_group"):
        group_input = raw["vertex_group"]
        group = obj.vertex_groups.new(name=group_input["name"])
        for index, weight in enumerate(group_input["weights"]):
            if weight:
                group.add([index], weight, "REPLACE")
    return obj


def evaluated_result(owner):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_owner = owner.evaluated_get(depsgraph)
    mesh = evaluated_owner.to_mesh()
    try:
        faces = [list(polygon.vertices) for polygon in mesh.polygons]
        positions = [list(vertex.co) for vertex in mesh.vertices]
        edges = sorted([sorted(edge.vertices) for edge in mesh.edges])
        vertex_normals = [list(vertex.normal) for vertex in mesh.vertices]
        corner_normals = [list(corner.vector) for corner in mesh.corner_normals]
        uv_layers = [
            {"name": layer.name, "values": [list(item.uv) for item in layer.data]}
            for layer in mesh.uv_layers
        ]
        color_attributes = []
        for attribute in mesh.color_attributes:
            if attribute.data_type not in ("FLOAT_COLOR", "BYTE_COLOR"):
                continue
            color_attributes.append({
                "name": attribute.name,
                "domain": attribute.domain,
                "data_type": attribute.data_type,
                "values": [list(item.color) for item in attribute.data],
            })
        group_names = {group.index: group.name for group in evaluated_owner.vertex_groups}
        vertex_groups = []
        for group_index, group_name in sorted(group_names.items()):
            weights = [0.0] * len(mesh.vertices)
            for vertex in mesh.vertices:
                for assignment in vertex.groups:
                    if assignment.group == group_index:
                        weights[vertex.index] = assignment.weight
            vertex_groups.append({"name": group_name, "values": weights})
        return {
            "positions": positions,
            "edges": edges,
            "faces": faces,
            "vertex_normals": vertex_normals,
            "corner_normals": corner_normals,
            "uv_layers": uv_layers,
            "color_attributes": color_attributes,
            "vertex_groups": vertex_groups,
        }
    finally:
        evaluated_owner.to_mesh_clear()


def run_case(inputs, case, blend_path=None):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    owner = make_mesh_object("Source", inputs["source"], source=True)
    source_reference = evaluated_result(owner)
    objects = {
        name: make_mesh_object(name.title(), raw)
        for name, raw in inputs.items()
        if name != "source"
    }
    modifier = owner.modifiers.new(name="ShrinkwrapOracle", type="SHRINKWRAP")
    params = case["params"]
    for name, value in params.items():
        if name == "target":
            modifier.target = objects[value]
        elif name == "auxiliary_target":
            modifier.auxiliary_target = objects[value] if value is not None else None
        elif name == "vertex_group":
            modifier.vertex_group = value
        else:
            setattr(modifier, name, value)
    bpy.context.view_layer.update()
    if blend_path is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
    return source_reference, evaluated_result(owner)


def main():
    separator = sys.argv.index("--")
    arguments = sys.argv[separator + 1:]
    input_path, output_path = arguments[:2]
    blend_dir = Path(arguments[2]) if len(arguments) > 2 else None
    case_filter = arguments[3] if len(arguments) > 3 else None
    if blend_dir is not None:
        blend_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as stream:
        corpus = json.load(stream)
    defaults = corpus["defaults"]
    results = []
    source_reference = None
    for case in corpus["cases"]:
        params = dict(defaults)
        params.update(case["params"])
        case["params"] = params
        if case_filter is not None and case["id"] != case_filter:
            results.append({"id": case["id"], **case["blender"]})
            continue
        blend_path = None if blend_dir is None else blend_dir / f"{case['id']}.blend"
        source_reference, result = run_case(corpus["inputs"], case, blend_path)
        case["blender"] = result
        results.append({"id": case["id"], **result})
    if source_reference is None:
        raise ValueError("the selected Blender Shrinkwrap case does not exist")
    corpus["source_blender_input"] = source_reference
    corpus["blender_version"] = bpy.app.version_string
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(corpus, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("POTTER_SHRINKWRAP_ORACLE=" + json.dumps({
        "version": bpy.app.version_string,
        "cases": len(results),
        "ids": [case["id"] for case in corpus["cases"]],
    }, sort_keys=True))

main()


