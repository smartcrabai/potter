import json
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(
        f"F-curve channel fixture requires Blender 5.2.2, found {bpy.app.version_string}"
    )

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10
scene.render.image_settings.file_format = "PNG"
bpy.context.preferences.filepaths.save_version = 0



camera_data = bpy.data.cameras.new("Animated Camera Data")
camera = bpy.data.objects.new("F-Curve Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
for frame, location_x in ((1, 0.0), (10, 0.0)):
    camera.location.x = location_x
    camera.keyframe_insert(data_path="location", index=0, frame=frame)
object_action = camera.animation_data.action
for frame, lens in ((1, 24.0), (4, 84.0), (7, 42.0)):
    camera_data.lens = lens
    camera_data.keyframe_insert(data_path="lens", frame=frame)
if camera_data.animation_data.action != object_action:
    raise RuntimeError("Camera Data lens keying did not share the existing Object Action")
if {slot.target_id_type for slot in object_action.slots} != {"OBJECT", "CAMERA"}:
    raise RuntimeError("shared camera Action must retain Object and Camera slots")
camera_data.lens = 35.0

light_data = bpy.data.lights.new("Animated Light Data", type="POINT")
light = bpy.data.objects.new("F-Curve Light", light_data)
scene.collection.objects.link(light)
for frame, energy in ((2, 80.0), (6, 320.0)):
    light_data.energy = energy
    light_data.keyframe_insert(data_path="energy", frame=frame)
light_data.energy = 75.0

driver_source = bpy.data.objects.new("Lens Driver Source", None)
scene.collection.objects.link(driver_source)
for frame, location_x in ((2, 2.0), (6, 6.0)):
    driver_source.location.x = location_x
    driver_source.keyframe_insert(data_path="location", index=0, frame=frame)

driver_camera_data = bpy.data.cameras.new("Driven Camera Data")
driver_camera_data.lens = 55.0
driver_camera = bpy.data.objects.new("Driven F-Curve Camera", driver_camera_data)
scene.collection.objects.link(driver_camera)

scene_driver_light_data = bpy.data.lights.new("Scene Driver Light Data", type="POINT")
scene_driver_light_data.energy = 27.0
scene_driver_light = bpy.data.objects.new("Scene Driver Light", scene_driver_light_data)
scene.collection.objects.link(scene_driver_light)
scene_driver_curve = scene_driver_light_data.driver_add("energy")
scene_driver = scene_driver_curve.driver
scene_driver.type = "SCRIPTED"
scene_driver.expression = "frame * 2.0"
scene_variable = scene_driver.variables.new()
scene_variable.name = "frame"
scene_variable.type = "SINGLE_PROP"
scene_target = scene_variable.targets[0]
scene_target.id_type = "SCENE"
scene_target.id = scene
scene_target.data_path = "frame_current"

data_target_camera_data = bpy.data.cameras.new("Data Target Camera Data")
data_target_camera_data.lens = 75.0
data_target_camera = bpy.data.objects.new("Data Target Camera", data_target_camera_data)
scene.collection.objects.link(data_target_camera)
data_target_curve = data_target_camera_data.driver_add("lens")
data_target_driver = data_target_curve.driver
data_target_driver.type = "SCRIPTED"
data_target_driver.expression = "source_lens"
data_target_variable = data_target_driver.variables.new()
data_target_variable.name = "source_lens"
data_target_variable.type = "SINGLE_PROP"
data_target = data_target_variable.targets[0]
data_target.id_type = "CAMERA"
data_target.id = camera_data
data_target.data_path = "lens"


def channel_curve(owner, data_path):
    action = owner.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                channelbag = strip.channelbag(slot)
                if channelbag is None:
                    continue
                for curve in channelbag.fcurves:
                    if curve.data_path == data_path and curve.array_index == 0:
                        return curve
    raise RuntimeError(f"Missing F-Curve for {data_path}")


source_curve = channel_curve(driver_source, "location")
for point in source_curve.keyframe_points:
    point.interpolation = "LINEAR"
source_curve.update()

driver_curve = driver_camera_data.driver_add("lens")
driver = driver_curve.driver
driver.type = "SCRIPTED"
driver.expression = "drive * 5.0 + 1.0"
variable = driver.variables.new()
variable.name = "drive"
variable.type = "TRANSFORMS"
target = variable.targets[0]
target.id = driver_source
target.transform_type = "LOC_X"
target.transform_space = "LOCAL_SPACE"
# Keep the source path so Potter's imported LOC_X variable resolves to the animated translation.
target.data_path = "location"


camera_curve = channel_curve(camera_data, "lens")
for point in camera_curve.keyframe_points:
    point.interpolation = "BEZIER"
    point.handle_left_type = "AUTO_CLAMPED"
    point.handle_right_type = "AUTO_CLAMPED"
camera_curve.modifiers.new(type="CYCLES").mode_after = "REPEAT"
camera_curve.update()

light_curve = channel_curve(light_data, "energy")
for point in light_curve.keyframe_points:
    point.interpolation = "LINEAR"
light_cycle = light_curve.modifiers.new(type="CYCLES")
light_cycle.mode_after = "REPEAT_OFFSET"
light_curve.update()
scene.frame_set(1)
base_depsgraph = bpy.context.evaluated_depsgraph_get()
base_depsgraph.update()

frames = (2, 3, 4, 5, 6, 7, 8, 10)
expected = {
    "blender_version": bpy.app.version_string,
    "base": {
        "camera_lens_mm": 35.0,
        "light_energy": 75.0,
        "driven_camera_lens_mm": 55.0,
        "scene_driver_light_energy": float(scene_driver_light_data.energy),
        "data_target_camera_lens_mm": float(data_target_camera_data.lens),
    },
    "frames": {},
}
for frame in frames:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    expected["frames"][str(frame)] = {
        "camera_lens_mm": float(camera.evaluated_get(depsgraph).data.lens),
        "light_energy": float(light.evaluated_get(depsgraph).data.energy),
        "driven_camera_lens_mm": float(
            driver_camera.evaluated_get(depsgraph).data.lens
        ),
        "scene_driver_light_energy": float(
            scene_driver_light.evaluated_get(depsgraph).data.energy
        ),
        "data_target_camera_lens_mm": float(
            data_target_camera.evaluated_get(depsgraph).data.lens
        ),
    }

(ROOT / "expected.json").write_text(
    json.dumps(expected, indent=2) + "\n", encoding="utf-8"
)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "fcurve_non_transform_channels.blend"))
