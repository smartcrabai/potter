"""Blender 5.2.2 compositor Mask-node oracle for Bezier mask splines.

Run from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --disable-autoexec --threads 1 --python-exit-code 3 \
        --python tests/fixtures/native_oracle/mask_bezier_splines.py -- \
        tests/fixtures/native_oracle/mask_bezier

Each case in `mask_bezier_cases.json` becomes one `bpy.data.masks` datablock whose points are pixel
coordinates mapped into Blender mask space with `BKE_mask_coord_from_frame`'s aspect squeeze. A
fixed-size `CompositorNodeMask` feeds the group output, and the render writes a 32-bit RGBA EXR.
Feather widths use Blender's `1 / max(width, height)` mask-space pixel size.
"""

import json
import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

OUTPUT_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(FIXTURE_DIR, "mask_bezier_cases.json"), encoding="utf-8") as file:
    CASES = json.load(file)
DEFAULT_WIDTH = CASES["width"]
DEFAULT_HEIGHT = CASES["height"]
LAYER_NAME = "Oracle Layer"


def mask_space(pixel_co, width, height):
    x, y = pixel_co[0] / width, 1.0 - pixel_co[1] / height
    squeeze = min(width, height) / max(width, height)
    if width > height:
        y = (y - 0.5) * squeeze + 0.5
    elif height > width:
        x = (x - 0.5) * squeeze + 0.5
    return (x, y)


def make_mask(case, width, height):
    mask = bpy.data.masks.new(case["name"])
    layer = mask.layers.new(name=LAYER_NAME)
    spline = layer.splines.new()
    spline.points.add(len(case["points"]) - 1)
    spline.use_cyclic = case["cyclic"]
    spline.use_fill = True
    spline.use_self_intersection_check = False
    spline.offset_mode = "EVEN"
    spline.weight_interpolation = "EASE"

    for index, data in enumerate(case["points"]):
        point = spline.points[index]
        point.co = mask_space(data["position"], width, height)
        point.handle_left_type = "FREE"
        point.handle_right_type = "FREE"
        point.handle_left = mask_space(data.get("handle_left", data["position"]), width, height)
        point.handle_right = mask_space(data.get("handle_right", data["position"]), width, height)
        point.weight = data.get("weight", case["feather"]) / max(width, height)
        for key in data.get("keyframes", []):
            if "handle_right" in key:
                insert_handle_key(mask, index, key["frame"], key["handle_right"], width, height)
    return mask


def insert_handle_key(mask, point_index, frame, handle_pixel, width, height):
    """Keys a point's right handle. All keys of a mask share one action and one channelbag."""
    path = f'layers["{LAYER_NAME}"].splines[0].points[{point_index}].handle_right'
    if mask.animation_data is None:
        mask.animation_data_create()
    if mask.animation_data.action is None:
        action = bpy.data.actions.new(f"{mask.name} Handles")
        slot = action.slots.new(id_type="MASK", name=mask.name)
        action.layers.new("Handles").strips.new(type="KEYFRAME")
        mask.animation_data.action = action
        mask.animation_data.action_slot = slot
    slot = mask.animation_data.action_slot
    strip = mask.animation_data.action.layers[0].strips[0]
    channelbag = strip.channelbag(slot, ensure=True)
    value = mask_space(handle_pixel, width, height)
    for component in range(2):
        curve = channelbag.fcurves.find(path, index=component)
        if curve is None:
            curve = channelbag.fcurves.new(data_path=path, index=component)
        key = curve.keyframe_points.insert(frame, value[component], options={"FAST"})
        key.interpolation = "LINEAR"
        curve.update()


def make_scene(case, output_path):
    width = case.get("width", DEFAULT_WIDTH)
    height = case.get("height", DEFAULT_HEIGHT)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "NONE"
    scene.render.filepath = os.path.splitext(output_path)[0]
    scene.view_settings.view_transform = "Raw"

    camera_data = bpy.data.cameras.new("Oracle Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("Oracle Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    scene.camera = camera

    mask = make_mask(case, width, height)
    tree = bpy.data.node_groups.new("Mask Oracle", "CompositorNodeTree")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    scene.render.use_compositing = True

    mask_node = tree.nodes.new("CompositorNodeMask")
    mask_node.mask = mask
    mask_node.inputs["Size Source"].default_value = "Fixed"
    mask_node.inputs["Size X"].default_value = width
    mask_node.inputs["Size Y"].default_value = height
    mask_node.inputs["Feather"].default_value = case["use_feather"]
    mask_node.inputs["Motion Blur"].default_value = False
    group_output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(mask_node.outputs["Mask"], group_output.inputs["Image"])

    scene.frame_set(int(case["frame"]))
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    capture = scene.render.filepath + ".exr"
    if not os.path.isfile(capture):
        raise RuntimeError(f"Blender did not write {capture}")
    os.replace(capture, output_path)


os.makedirs(OUTPUT_DIR, exist_ok=True)
for case in CASES["cases"]:
    output_path = os.path.join(OUTPUT_DIR, case["name"] + ".exr")
    make_scene(case, output_path)
    print("CAPTURE", os.path.basename(output_path), os.path.getsize(output_path))
