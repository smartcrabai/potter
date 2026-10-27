"""Generate the Blender 5.2.2 F-Curve channel fixture and evaluated-state oracle.

Run with:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/fcurve_more_channels/generate.py
"""

import json
import math
import os
import tempfile
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view


ROOT = Path(__file__).resolve().parent
FRAMES = (1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10)
KEYS = (1, 4, 7)

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(
        f"F-Curve channel fixture requires Blender 5.2.2, found {bpy.app.version_string}"
    )

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("More Channels World")
world.color = (0.01, 0.01, 0.01)
scene.world = world
scene.frame_start = FRAMES[0]
scene.frame_end = FRAMES[-1]
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.dither_intensity = 0.0
bpy.context.preferences.filepaths.save_version = 0
scene.render.engine = "CYCLES"
scene.cycles.samples = 16
original_render_engine = scene.render.engine


def add_object(name, data=None):
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj

def set_frame(frame):
    integer_frame = math.floor(frame)
    scene.frame_set(integer_frame, subframe=frame - integer_frame)


def set_attr(owner, path, value):
    if path.startswith('["') and path.endswith('"]'):
        owner[path[2:-2]] = value
        return
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)


def animate(owner, data_path, values, *, index=None, interpolation="LINEAR"):
    for frame, value in zip(KEYS, values):
        if index is not None and data_path.startswith('["') and data_path.endswith('"]'):
            key = data_path[2:-2]
            array = list(owner[key])
            array[index] = value
            owner[key] = array
        elif data_path == "color" and index is not None:
            color = list(owner.color)
            color[index] = value
            owner.color = color
        else:
            set_attr(owner, data_path, value)
        if index is None:
            owner.keyframe_insert(data_path=data_path, frame=frame)
        else:
            owner.keyframe_insert(data_path=data_path, index=index, frame=frame)
    curve = channel_curve(owner, data_path, 0 if index is None else index)
    for point in curve.keyframe_points:
        point.interpolation = interpolation
        if interpolation == "BEZIER":
            point.handle_left_type = "AUTO_CLAMPED"
            point.handle_right_type = "AUTO_CLAMPED"
    curve.update()
    return curve


def channel_curve(owner, data_path, index=0):
    action = owner.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                channelbag = strip.channelbag(slot)
                if channelbag is None:
                    continue
                for curve in channelbag.fcurves:
                    if curve.data_path == data_path and curve.array_index == index:
                        return curve
    raise RuntimeError(f"Missing F-Curve for {data_path}[{index}]")

def set_raw_key_value(curve, frame, value):
    for point in curve.keyframe_points:
        if point.co.x == frame:
            point.co.y = value
            curve.update()
            return
    raise RuntimeError(f"Missing keyframe at {frame}")



# Camera data covers all newly exposed camera channels. The shifted X channel is
# driven from an animated Object transform, while the remaining channels are keyed.
camera_data = bpy.data.cameras.new("More Channels Camera Data")
camera_data.type = "ORTHO"
camera_data.lens = 50.0
camera_data.sensor_width = 36.0
camera_data.sensor_height = 24.0
camera_data.ortho_scale = 6.0
camera_data.clip_start = 0.1
camera_data.clip_end = 100.0
camera_data.shift_x = 0.0
camera_data.shift_y = 0.0
camera_data.dof.use_dof = True
camera_data.dof.focus_distance = 10.0
camera_data.dof.aperture_fstop = 2.8
camera = add_object("More Channels Camera", camera_data)
scene.camera = camera
camera.location = (0.0, 0.0, 5.0)

sensor_width_curve = animate(camera_data, "sensor_width", (30.0, 42.0, 34.0), interpolation="BEZIER")
set_raw_key_value(sensor_width_curve, 7, -1.0)
animate(camera_data, "sensor_height", (18.0, 26.0, 22.0), interpolation="CONSTANT")
ortho_scale_curve = animate(camera_data, "ortho_scale", (4.0, 9.0, 6.0))
set_raw_key_value(ortho_scale_curve, 7, -1.0)
clip_start_curve = animate(camera_data, "clip_start", (0.15, 0.45, 0.25))
set_raw_key_value(clip_start_curve, 7, -1.0)
animate(camera_data, "clip_end", (80.0, 140.0, 110.0))
animate(camera_data, "shift_y", (-0.2, 0.3, 0.1))
focus_distance_curve = animate(camera_data, "dof.focus_distance", (4.0, 18.0, 9.0))
set_raw_key_value(focus_distance_curve, 7, -1.0)
aperture_curve = animate(camera_data, "dof.aperture_fstop", (1.4, 5.6, 2.0))
set_raw_key_value(aperture_curve, 7, -1.0)

source = add_object("More Channels Driver Source")
for frame, location_x in ((1, 0.0), (7, 6.0)):
    source.location.x = location_x
    source.keyframe_insert(data_path="location", index=0, frame=frame)
source_curve = channel_curve(source, "location", 0)
for point in source_curve.keyframe_points:
    point.interpolation = "LINEAR"
source_curve.update()
driver_curve = camera_data.driver_add("shift_x")
driver = driver_curve.driver
driver.type = "SCRIPTED"
driver.expression = "drive * 0.02 + 0.05"
variable = driver.variables.new()
variable.name = "drive"
variable.type = "TRANSFORMS"
target = variable.targets[0]
target.id = source
target.transform_type = "LOC_X"
target.transform_space = "LOCAL_SPACE"
target.data_path = "location"

driver_spot_data = bpy.data.lights.new("More Channels Driven Spot Data", type="SPOT")
driver_spot_data.energy = 0.0
driver_spot_data.spot_size = 0.4
driver_spot = add_object("More Channels Driven Spot", driver_spot_data)
driver_spot.location = (10.0, 0.0, 4.0)
spot_driver_curve = driver_spot_data.driver_add("spot_size")
spot_driver = spot_driver_curve.driver
spot_driver.type = "SCRIPTED"
spot_driver.expression = "0.3 + drive * 0.02"
spot_variable = spot_driver.variables.new()
spot_variable.name = "drive"
spot_variable.type = "TRANSFORMS"
spot_target = spot_variable.targets[0]
spot_target.id = source
spot_target.transform_type = "LOC_X"
spot_target.transform_space = "LOCAL_SPACE"
spot_target.data_path = "location"

# Spot light: RGB F-Curves, a repeated modifier, and independent light scalars.
spot_data = bpy.data.lights.new("More Channels Spot Data", type="SPOT")
spot_data.energy = 80.0
spot_data.color = (0.8, 0.7, 0.6)
spot_data.spot_size = 0.75
spot_data.spot_blend = 0.2
spot_data.shadow_soft_size = 0.15
spot = add_object("More Channels Spot", spot_data)
spot.location = (0.0, 0.0, 4.0)
for component, values in enumerate(((0.1, 0.8, 0.3), (0.25, 0.65, 0.4), (0.7, 0.2, 0.9))):
    color_curve = animate(spot_data, "color", values, index=component)
    if component == 0:
        set_raw_key_value(color_curve, 7, -0.25)
spot_size_curve = animate(spot_data, "spot_size", (0.3, 1.2, 0.6))
set_raw_key_value(spot_size_curve, 7, -0.25)
spot_size_curve.modifiers.new(type="CYCLES").mode_after = "REPEAT"
spot_blend_curve = animate(spot_data, "spot_blend", (0.1, 0.7, 0.35), interpolation="CONSTANT")
set_raw_key_value(spot_blend_curve, 7, -0.25)
shadow_soft_size_curve = animate(spot_data, "shadow_soft_size", (0.05, 0.45, 0.2))
set_raw_key_value(shadow_soft_size_curve, 7, -0.25)

# Area and sun properties live on distinct Blender Light datablocks.
area_data = bpy.data.lights.new("More Channels Area Data", type="AREA")
area_data.energy = 0.0
area_data.shape = "RECTANGLE"
area_data.size = 0.5
area_data.size_y = 0.25
area = add_object("More Channels Area", area_data)
area_size_curve = animate(area_data, "size", (0.3, 1.1, 0.6))
set_raw_key_value(area_size_curve, 7, -0.25)
area_size_y_curve = animate(area_data, "size_y", (0.2, 0.8, 0.4))
set_raw_key_value(area_size_y_curve, 7, -0.25)

sun_data = bpy.data.lights.new("More Channels Sun Data", type="SUN")
sun_data.energy = 0.0
sun_data.angle = math.radians(0.526)
sun = add_object("More Channels Sun", sun_data)
sun_angle_curve = animate(sun_data, "angle", (0.01, 0.12, 0.04))
set_raw_key_value(sun_angle_curve, 7, -0.25)

# Direct F-Curve values preserve fractional Boolean samples at the four
# conversion boundaries; Blender's evaluated RNA values are captured below.
mesh = bpy.data.meshes.new("More Channels Render Plane Mesh")
mesh.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
mesh.update()
visibility = add_object("More Channels Visibility", mesh)
material = bpy.data.materials.new("More Channels Render Material")
material.diffuse_color = (0.8, 0.15, 0.05, 1.0)
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.8,
    0.15,
    0.05,
    1.0,
)
mesh.materials.append(material)
light_target_mesh = bpy.data.meshes.new("More Channels Light Target Mesh")
light_target_mesh.from_pydata(
    [(-0.4, -0.4, 0.0), (0.4, -0.4, 0.0), (0.4, 0.4, 0.0), (-0.4, 0.4, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
light_target_mesh.update()
light_target = add_object("More Channels Light Target", light_target_mesh)
light_target.location = (1.5, 0.0, 0.0)
light_target_mesh.materials.append(material)
mesh.uv_layers.new(name="UVMap")
uv_subject = add_object("Animated Camera UV Project Subject", mesh)
uv_subject.location = (10.0, 0.0, 0.0)
uv_project = uv_subject.modifiers.new("Animated Camera UV Projection", "UV_PROJECT")
uv_project.uv_layer = "UVMap"
uv_project.projector_count = 1
uv_project.projectors[0].object = camera
visibility["gain"] = 2.0
visibility.id_properties_ui("gain").update(min=0.0, max=4.0)
visibility["vector"] = [0.25, 0.5, 0.75]
visibility.id_properties_ui("vector").update(min=0.0, max=1.0)
visibility["steps"] = 0
visibility.id_properties_ui("steps").update(min=-2, max=2)
visibility["enabled"] = False
for data_path, keys in (
    ("hide_viewport", ((1, 0.0), (2, 0.0), (2.5, 0.5), (3, 0.9999999), (4, 1.0))),
    ("hide_render", ((1, 0.0), (2, 1.0), (3, 0.5), (4, 1.0))),
):
    for frame, _ in keys:
        visibility.keyframe_insert(data_path=data_path, frame=frame)
    curve = channel_curve(visibility, data_path)
    for point, (_, value) in zip(curve.keyframe_points, keys):
        point.co.y = value
        point.interpolation = "CONSTANT"
    curve.update()
gain_curve = animate(visibility, '["gain"]', (1.0, 5.0, 3.0), index=None)
set_raw_key_value(gain_curve, 7, -1.0)
vector_curve = animate(visibility, '["vector"]', (0.1, 0.9, 0.3), index=1)
set_raw_key_value(vector_curve, 7, -1.0)
steps_curve = animate(visibility, '["steps"]', (-1, 3, 0), index=None)
set_raw_key_value(steps_curve, 7, -3)
for frame, value in ((1, 0.0), (2.5, 0.5), (4, 1.0)):
    visibility.keyframe_insert(data_path='["enabled"]', frame=frame)
enabled_curve = channel_curve(visibility, '["enabled"]')
for point, (_, value) in zip(
    enabled_curve.keyframe_points, ((1, 0.0), (2.5, 0.5), (4, 1.0))
):
    point.co.y = value
    point.interpolation = "LINEAR"
enabled_curve.update()

# NLA Replace strips on Object custom properties at influence 0.6. Each Action drives one
# property, so the strips stay independent of the Visibility object's active Action.
nla_object = bpy.data.objects.new("More Channels NLA Object", None)
scene.collection.objects.link(nla_object)
nla_object.location = (0.0, 0.0, -20.0)
nla_object["nla_flag"] = False
nla_object["nla_steps"] = 0
nla_object.id_properties_ui("nla_steps").update(min=-2, max=2)
nla_object["nla_default_flag"] = False
nla_object.id_properties_ui("nla_default_flag").update(default=True)
nla_object["nla_default_steps"] = 0
nla_object.id_properties_ui("nla_default_steps").update(default=2)


def add_nla_replace_strip(owner, track_name, data_path, values):
    animation = owner.animation_data or owner.animation_data_create()
    animation.action = None
    animate(owner, data_path, values, interpolation="CONSTANT")
    action = animation.action
    animation.action = None
    track = animation.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(track_name, int(KEYS[0]), action)
    strip.blend_type = "REPLACE"
    # Blender applies a static influence only through an animated influence (`use_animated_influence`).
    strip.use_animated_influence = True
    animation.action = action
    strip.influence = 0.6
    for frame in KEYS:
        strip.keyframe_insert(data_path="influence", frame=frame)
    animation.action = None
    return strip


add_nla_replace_strip(nla_object, "More Channels NLA Flag", '["nla_flag"]', (True, False, True))
steps_strip = add_nla_replace_strip(
    nla_object, "More Channels NLA Steps", '["nla_steps"]', (3, 3, -3)
)
steps_strip.blend_in = 3.0
add_nla_replace_strip(nla_object, "More Channels NLA Default Flag", '["nla_default_flag"]', (True, True, True))
add_nla_replace_strip(
    nla_object, "More Channels NLA Default Steps", '["nla_default_steps"]', (3, 3, 3)
)

# Gain light: a spot pointing away from the frame, so Cycles light samples are unchanged. Its energy
# driver reads the Visibility object's animated, hard-clamped `gain` through a SINGLE_PROP variable.
gain_light_data = bpy.data.lights.new("More Channels Gain Light Data", type="SPOT")
gain_light_data.energy = 0.0
gain_light = add_object("More Channels Gain Light", gain_light_data)
gain_light.location = (0.0, 0.0, 100.0)
gain_light.rotation_euler = (math.pi, 0.0, 0.0)
gain_driver_curve = gain_light_data.driver_add("energy")
gain_driver = gain_driver_curve.driver
gain_driver.type = "SCRIPTED"
gain_driver.expression = "gain * 5.0"
gain_variable = gain_driver.variables.new()
gain_variable.name = "gain"
gain_variable.type = "SINGLE_PROP"
gain_target = gain_variable.targets[0]
gain_target.id_type = "OBJECT"
gain_target.id = visibility
gain_target.data_path = '["gain"]'


base = {
    "More Channels Camera": {
        "camera": {
            "sensor_width_mm": 36.0,
            "sensor_height_mm": 24.0,
            "ortho_scale": 6.0,
            "clip_start": 0.1,
            "clip_end": 100.0,
            "shift": [0.0, 0.0],
            "focus_distance": 10.0,
            "f_stop": 2.8,
        }
    },
    "More Channels Spot": {
        "light": {
            "color": [0.8, 0.7, 0.6],
            "spot_size": 0.75,
            "spot_blend": 0.2,
            "radius": 0.15,
        }
    },

    "More Channels Driven Spot": {"light": {"spot_size": 0.4}},
    "More Channels Area": {"light": {"area_size": 0.5, "area_size_y": 0.25}},
    "More Channels Sun": {"light": {"sun_angle": math.radians(0.526)}},
    "More Channels Visibility": {
        "visible": True,
        "render_visible": True,
        "properties": {
            "gain": 2.0,
            "vector": [0.25, 0.5, 0.75],
            "steps": 0,
            "enabled": False,
        },
    },
    "More Channels NLA Object": {
        "properties": {
            "nla_flag": False,
            "nla_steps": 0,
            "nla_default_flag": False,
            "nla_default_steps": 0,
        }
    },
    "More Channels Gain Light": {"light": {"energy": 0.0}},
}

frames = {}
for frame in FRAMES:
    set_frame(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    camera_eval = camera.evaluated_get(depsgraph).data
    spot_eval = spot.evaluated_get(depsgraph).data
    driven_spot_eval = driver_spot.evaluated_get(depsgraph).data
    area_eval = area.evaluated_get(depsgraph).data
    sun_eval = sun.evaluated_get(depsgraph).data
    visibility_eval = visibility.evaluated_get(depsgraph)
    nla_eval = nla_object.evaluated_get(depsgraph)
    gain_eval = gain_light.evaluated_get(depsgraph).data
    frames[str(frame)] = {
        "More Channels Camera": {
            "camera": {
                "sensor_width_mm": float(camera_eval.sensor_width),
                "sensor_height_mm": float(camera_eval.sensor_height),
                "ortho_scale": float(camera_eval.ortho_scale),
                "clip_start": float(camera_eval.clip_start),
                "clip_end": float(camera_eval.clip_end),
                "shift": [float(camera_eval.shift_x), float(camera_eval.shift_y)],
                "focus_distance": float(camera_eval.dof.focus_distance),
                "f_stop": float(camera_eval.dof.aperture_fstop),
            }
        },
        "More Channels Spot": {
            "light": {
                "color": [float(component) for component in spot_eval.color],
                "spot_size": float(spot_eval.spot_size),
                "spot_blend": float(spot_eval.spot_blend),
                "radius": float(spot_eval.shadow_soft_size),
            }
        },
        "More Channels Driven Spot": {
            "light": {"spot_size": float(driven_spot_eval.spot_size)}
        },
        "More Channels Area": {
            "light": {
                "area_size": float(area_eval.size),
                "area_size_y": float(area_eval.size_y),
            }
        },
        "More Channels Sun": {"light": {"sun_angle": float(sun_eval.angle)}},
        "More Channels Visibility": {
            "visible": not bool(visibility_eval.hide_viewport),
            "render_visible": not bool(visibility_eval.hide_render),
            "properties": {
                "gain": float(visibility_eval["gain"]),
                "vector": [float(component) for component in visibility_eval["vector"]],
                "steps": int(visibility_eval["steps"]),
                "enabled": bool(visibility_eval["enabled"]),
            },
        },
        "More Channels NLA Object": {
            "properties": {
                "nla_flag": bool(nla_eval["nla_flag"]),
                "nla_steps": int(nla_eval["nla_steps"]),
                "nla_default_flag": bool(nla_eval["nla_default_flag"]),
                "nla_default_steps": int(nla_eval["nla_default_steps"]),
            }
        },
        "More Channels Gain Light": {"light": {"energy": float(gain_eval.energy)}},
    }

original_filepath = scene.render.filepath
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.render_aa = "OFF"
scene.display.viewport_aa = "OFF"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.background_type = "WORLD"
bpy.context.view_layer.objects.active = None
bpy.ops.object.select_all(action="DESELECT")
render_coverage = {}
preview_coverage = {}
with tempfile.TemporaryDirectory(prefix="fcurve_more_channels_workbench_") as render_directory:
    def capture_workbench_coverage(frame, name):
        set_frame(frame)
        render_path = os.path.join(render_directory, name)
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(render_path, check_existing=False)
        pixels = image.pixels[:]
        background = pixels[:3]
        changed_pixels = sum(
            any(
                abs(pixels[offset + component] - background[component]) > 0.5 / 255.0
                for component in range(3)
            )
            for offset in range(0, len(pixels), 4)
        )
        coverage = changed_pixels / (image.size[0] * image.size[1])
        bpy.data.images.remove(image)
        return coverage

    for render_frame in (1, 2, 4):
        render_coverage[str(render_frame)] = capture_workbench_coverage(
            render_frame, f"render_{render_frame:04}.png"
        )
    hide_render_curve = channel_curve(visibility, "hide_render")
    hide_render_curve.mute = True
    set_frame(2)
    visibility.hide_render = False
    bpy.context.evaluated_depsgraph_get().update()
    preview_coverage["2"] = capture_workbench_coverage(2, "preview_0002.png")
    hide_render_curve.mute = False
    set_frame(1)
scene.render.engine = original_render_engine

light_samples = {}
with tempfile.TemporaryDirectory(prefix="fcurve_more_channels_cycles_") as render_directory:
    for render_frame in (1, 4):
        set_frame(render_frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        projected = world_to_camera_view(
            scene,
            camera.evaluated_get(depsgraph),
            light_target.matrix_world.translation,
        )
        pixel_x = int(projected.x * scene.render.resolution_x)
        pixel_y_bottom = int(projected.y * scene.render.resolution_y)
        pixel_y = scene.render.resolution_y - 1 - pixel_y_bottom
        if not (
            0 <= pixel_x < scene.render.resolution_x
            and 0 <= pixel_y_bottom < scene.render.resolution_y
        ):
            raise RuntimeError(f"Light sample is outside the camera frame at {render_frame}")
        render_path = os.path.join(render_directory, f"frame_{render_frame:04}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(render_path, check_existing=False)
        pixel_offset = (pixel_y_bottom * image.size[0] + pixel_x) * 4
        light_samples[str(render_frame)] = {
            "pixel": [pixel_x, pixel_y],
            "rgb": [float(value) for value in image.pixels[pixel_offset : pixel_offset + 3]],
        }
        bpy.data.images.remove(image)
scene.render.filepath = original_filepath
expected = {
    "blender_version": bpy.app.version_string,
    "tolerance": 1.0e-4,
    "frames": frames,
    "base": base,
    "render_coverage": render_coverage,
    "preview_coverage": preview_coverage,
    "light_samples": light_samples,
}
(ROOT / "expected.json").write_text(
    json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
scene.frame_set(FRAMES[0])
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "fcurve_more_channels.blend"))
print("wrote Blender 5.2.2 F-Curve channel fixture:", ROOT)
