import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(
        f"Clamp To taper oracle requires Blender 5.2.2, found {bpy.app.version_string}"
    )

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

taper_data = bpy.data.curves.new("Clamp Taper Data", "CURVE")
taper_data.dimensions = "2D"
taper_spline = taper_data.splines.new("POLY")
taper_spline.points.add(2)
for point, coordinates in zip(
    taper_spline.points,
    ((0.0, 0.2, 0.0, 1.0), (0.5, 1.8, 0.0, 1.0), (1.0, 0.2, 0.0, 1.0)),
):
    point.co = coordinates
taper = bpy.data.objects.new("Clamp Taper", taper_data)
scene.collection.objects.link(taper)

path_data = bpy.data.curves.new("Clamp Route Data", "CURVE")
path_data.dimensions = "3D"
path_data.use_path = True
path_data.path_duration = 100
path_data.bevel_depth = 0.5
path_data.bevel_resolution = 2
path_data.taper_object = taper
path_data.taper_radius_mode = "OVERRIDE"
path_spline = path_data.splines.new("POLY")
path_spline.points.add(2)
for point, coordinates in zip(
    path_spline.points,
    ((0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 1.0), (4.0, 0.0, 0.0, 1.0)),
):
    point.co = coordinates
path = bpy.data.objects.new("Clamp Route", path_data)
scene.collection.objects.link(path)

target = bpy.data.objects.new("Clamp Owner", None)
scene.collection.objects.link(target)
target.location = (0.0, 0.4, 0.0)
clamp = target.constraints.new("CLAMP_TO")
clamp.target = path
clamp.main_axis = "CLAMPTO_Y"

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated_path = path.evaluated_get(depsgraph)
evaluated_mesh = evaluated_path.to_mesh()
try:
    positions = [evaluated_path.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices]
    bounds_min = [float(min(position[axis] for position in positions)) for axis in range(3)]
    bounds_max = [float(max(position[axis] for position in positions)) for axis in range(3)]
finally:
    evaluated_path.to_mesh_clear()
evaluated_target = target.evaluated_get(depsgraph)
world_matrix = [
    float(evaluated_target.matrix_world[row][column])
    for column in range(4)
    for row in range(4)
]
with open(os.path.join(root, "clamp_to_expected.json"), "w", encoding="utf-8") as output:
    json.dump(
        {
            "blender_version": bpy.app.version_string,
            "curve_bounds_min": bounds_min,
            "curve_bounds_max": bounds_max,
            "owner_world_matrix": world_matrix,
        },
        output,
        separators=(",", ":"),
    )
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "clamp_to_source.blend"))
