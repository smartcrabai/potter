"""Blender 5.2.2 EEVEE/VSE oracle for an alpha-zero light-shape pixel through a Cross transition.

Regenerate the macOS consumer capture from the repository root with:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 \
        -P tests/fixtures/native_oracle/eevee_vse_light_shape_crossfade.py -- \
        tests/fixtures/native_oracle/eevee_macos/vse_light_shape_crossfade.blend \
        tests/fixtures/native_oracle/eevee_macos/vse_light_shape_crossfade.exr 1

The output scene's actual CROSS effect mixes an EEVEE Scene strip containing a camera-visible point
shape over an opaque Color strip. It saves and reopens the selected midpoint EXR consumer output.
"""

import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

blend_out, exr_out, render_flag = sys.argv[sys.argv.index("--") + 1 :][:3]
RENDER = render_flag != "0"
RESOLUTION = 64
CROSS_FRAME = 3


def configure_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "NONE"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def pixel_at(pixels, x, y):
    offset = (y * RESOLUTION + x) * 4
    return tuple(pixels[offset : offset + 4])


bpy.ops.wm.read_factory_settings(use_empty=True)
source = bpy.context.scene
source.name = "LightShapeCrossfadeSource"
configure_render(source)
source.eevee.taa_render_samples = 64
source.world = bpy.data.worlds.new("TransparentWorld")
source.world.use_nodes = True
source.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
source.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
camera_data = bpy.data.cameras.new("LightShapeCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 8.0
camera = bpy.data.objects.new("LightShapeCamera", camera_data)
source.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
source.camera = camera
light_data = bpy.data.lights.new("CameraVisiblePoint", "POINT")
light_data.energy = 5.0
light_data.shadow_soft_size = 1.0
light_data.use_soft_falloff = False
light = bpy.data.objects.new("CameraVisiblePoint", light_data)
source.collection.objects.link(light)
light.location = (0.0, 0.0, 0.0)
light.visible_camera = True
source.frame_start = 1
source.frame_end = 4

output = bpy.data.scenes.new("LightShapeVSECrossfade")
configure_render(output)
output.render.use_sequencer = True
output.frame_start = 1
output.frame_end = 4
editor = output.sequence_editor_create()
strips = editor.strips if hasattr(editor, "strips") else editor.sequences
background = strips.new_effect("OpaqueBackground", "COLOR", 1, 1, length=4)
background.color = (0.1, 0.2, 0.3)
background.blend_alpha = 1.0
scene_strip = strips.new_scene("CameraLightShapeScene", source, 2, 1)
transition = strips.new_effect(
    "CameraShapeCrossfade",
    "CROSS",
    3,
    1,
    length=4,
    input1=scene_strip,
    input2=background,
)
if bpy.context.window is not None:
    bpy.context.window.scene = output
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print("SAVED", blend_out)

if RENDER:
    output.frame_set(CROSS_FRAME)
    bpy.ops.render.render(write_still=False, scene=output.name)
    bpy.data.images["Render Result"].save_render(exr_out, scene=output)
    image = bpy.data.images.load(exr_out, check_existing=False)
    if tuple(image.size) != (RESOLUTION, RESOLUTION):
        raise RuntimeError(f"unexpected crossfade EXR size {image.size}")
    print("READBACK", os.path.basename(exr_out), image.alpha_mode, image.size[:], image.channels)
    for label, x, y in (("alpha0_shape", 32, 32), ("background", 4, 4)):
        print("PIXEL", label, x, y, tuple(round(value, 7) for value in pixel_at(image.pixels, x, y)))
    center = pixel_at(image.pixels, 32, 32)
    background = pixel_at(image.pixels, 4, 4)
    radiance_delta = max(center[channel] - background[channel] for channel in range(3))
    if (
        abs(center[3] - 0.5) > 1e-6
        or abs(background[3] - 0.5) > 1e-6
        or radiance_delta <= 0.01
    ):
        raise RuntimeError(
            f"crossfade midpoint lost alpha-zero radiance: center={center}, background={background}"
        )
    bpy.data.images.remove(image)
