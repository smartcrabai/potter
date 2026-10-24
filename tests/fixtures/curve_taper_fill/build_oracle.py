import bpy
import json
import os
import sys


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(
        f"curve taper/fill oracle requires Blender 5.2.2, found {bpy.app.version_string}"
    )

operations_path = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
output_root = os.path.realpath(sys.argv[sys.argv.index("--") + 2])
with open(operations_path, "r", encoding="utf-8") as stream:
    operations = json.load(stream)

bpy.ops.wm.read_factory_settings(use_empty=True)
nodes = {}
data_blocks = {}
curve_specs = {}
fill_modes = {"none": "NONE", "front": "FRONT", "back": "BACK", "both": "BOTH"}

for operation in operations:
    if operation["op"] != "curve.create":
        raise RuntimeError(f"unexpected fixture operation {operation['op']!r}")
    data_id = operation.get("data_id", operation["id"] + "_curve")
    data = bpy.data.curves.new(operation["name"] + " Data", "CURVE")
    data.dimensions = "2D" if operation.get("dimensions") == "two_d" else "3D"
    data.resolution_u = int(operation.get("resolution_u", 12))
    data.bevel_depth = float(operation.get("bevel_depth", 0.0))
    data.bevel_resolution = int(operation.get("bevel_resolution", 0))
    fill_mode = fill_modes[operation.get("fill_mode", "none")]
    if data.dimensions == "2D":
        data.fill_mode = {"NONE": "NONE", "BOTH": "BOTH"}.get(fill_mode, fill_mode)
    else:
        data.fill_mode = {"NONE": "FULL", "BOTH": "HALF"}.get(fill_mode, fill_mode)
    data.fill_solver = operation.get("fill_solver", "sweep_line").upper()
    data.fill_rule = operation.get("fill_rule", "even_odd").upper()
    obj = bpy.data.objects.new(operation["name"], data)
    bpy.context.scene.collection.objects.link(obj)
    nodes[operation["id"]] = obj
    data_blocks[data_id] = obj
    curve_specs[operation["id"]] = operation

    for spline_spec in operation["splines"]:
        spline_type = spline_spec["type"].upper()
        spline = data.splines.new(spline_type)
        points = spline_spec["points"]
        if spline_type == "BEZIER":
            spline.bezier_points.add(len(points) - 1)
            for bezier_point, point in zip(spline.bezier_points, points):
                bezier_point.co = point["co"]
                bezier_point.radius = float(point.get("radius", 1.0))
                bezier_point.tilt = float(point.get("tilt", 0.0))
                bezier_point.handle_left_type = point.get("handle_type", "AUTO").upper()
                bezier_point.handle_right_type = point.get("handle_type", "AUTO").upper()
        else:
            spline.points.add(len(points) - 1)
            for curve_point, point in zip(spline.points, points):
                curve_point.co = tuple(point["co"]) + (1.0,)
                curve_point.radius = float(point.get("radius", 1.0))
                curve_point.tilt = float(point.get("tilt", 0.0))
        spline.use_cyclic_u = bool(spline_spec.get("cyclic", False))
        spline.resolution_u = int(spline_spec.get("resolution", 12))
        if spline_type == "NURBS":
            spline.order_u = int(spline_spec.get("order", 3))
            spline.use_endpoint_u = bool(spline_spec.get("use_endpoint", False))

for node_id, operation in curve_specs.items():
    taper_id = operation.get("taper")
    if taper_id is not None:
        curve = nodes[node_id].data
        curve.taper_object = data_blocks[taper_id]
        taper_radius_mode = operation.get("taper_radius_mode", "override").upper()
        use_map_taper = bool(operation.get("use_map_taper", False))
        curve.taper_radius_mode = taper_radius_mode
        curve.use_map_taper = use_map_taper
        if curve.taper_radius_mode != taper_radius_mode or curve.use_map_taper != use_map_taper:
            raise RuntimeError(f"Blender did not preserve taper settings for {node_id}")


def sample(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        return {
            "positions": [
                [float(value) for value in matrix @ vertex.co]
                for vertex in mesh.vertices
            ],
            "faces": [
                [int(index) for index in polygon.vertices]
                for polygon in mesh.polygons
            ],
        }
    finally:
        evaluated.to_mesh_clear()


expected = {
    "blender_version": bpy.app.version_string,
    "objects": {
        node_id: sample(nodes[node_id])
        for node_id in (
            "taper_override",
            "taper_multiply",
            "taper_add",
            "taper_2d_contours",
            "bevelled_taper_host",
            "filled",
            "nurbs_endpoint_on",
            "nurbs_endpoint_off",
        )
}
}
os.makedirs(output_root, exist_ok=True)
with open(os.path.join(output_root, "expected.json"), "w", encoding="utf-8") as stream:
    json.dump(expected, stream, separators=(",", ":"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_root, "source.blend"))
