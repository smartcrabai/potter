import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"driver oracle requires Blender 5.2.2, found {bpy.app.version_string}")

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10


def add_euler_curves(obj):
    obj.rotation_mode = "XZY"
    for frame in (1, 10):
        obj.rotation_euler[0] = 0.5
        obj.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
        obj.rotation_euler[1] = 0.5
        obj.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)
        obj.rotation_euler[2] = 0.3
        obj.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)
    for layer in obj.animation_data.action.layers:
        for strip in layer.strips:
            for slot in obj.animation_data.action.slots:
                channelbag = strip.channelbag(slot)
                if channelbag is None:
                    continue
                for curve in channelbag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "CONSTANT"
                    curve.update()


source = bpy.data.objects.new("EulerDriverSource", None)
probe = bpy.data.objects.new("EulerDriverProbe", None)
scene.collection.objects.link(source)
scene.collection.objects.link(probe)
add_euler_curves(source)
add_euler_curves(probe)

curve = probe.driver_add("rotation_euler", 2)
driver = curve.driver
driver.type = "SCRIPTED"
driver.expression = "x"
variable = driver.variables.new()
variable.name = "x"
variable.type = "TRANSFORMS"
target = variable.targets[0]
target.id = source
target.transform_type = "ROT_X"
target.transform_space = "WORLD_SPACE"
target.rotation_mode = "XZY"

frame = 1
scene.frame_set(frame)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
evaluated = probe.evaluated_get(depsgraph)
world_matrix = [
    float(evaluated.matrix_world[row][column])
    for column in range(4)
    for row in range(4)
]
expected = {
    "blender_version": bpy.app.version_string,
    "frame": frame,
    "world_matrix": world_matrix,
    "driver_result": float(evaluated.rotation_euler[2]),
}
with open(os.path.join(root, "driver_expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output, indent=2, sort_keys=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "driver_source.blend"))
