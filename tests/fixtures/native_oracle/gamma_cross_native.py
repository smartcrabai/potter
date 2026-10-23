"""Blender 5.2.2 VSE GammaCross native consumer reference."""

import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

blend_out, exr_out = sys.argv[sys.argv.index("--") + 1 :][:2]
resolution = 64
frame = 3

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = "GammaCrossNativeFixture"
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.use_sequencer = True
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "NONE"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.frame_start = 1
scene.frame_end = 4

editor = scene.sequence_editor_create()
strips = editor.strips if hasattr(editor, "strips") else editor.sequences
black = strips.new_effect("GammaCrossBlack", "COLOR", 1, 1, length=4)
black.color = (0.0, 0.0, 0.0)
black.blend_alpha = 1.0
white = strips.new_effect("GammaCrossWhite", "COLOR", 2, 1, length=4)
white.color = (1.0, 1.0, 1.0)
white.blend_alpha = 1.0
strips.new_effect(
    "GammaCrossTransition",
    "GAMMA_CROSS",
    3,
    1,
    length=4,
    input1=black,
    input2=white,
)

# The previous capture was frame 1 because this producer never selected `frame`.
# Replacing it corrects producer provenance; it is not a numerical reference adjustment.
scene.frame_set(frame)
if scene.frame_current != frame:
    raise RuntimeError(f"expected frame {frame}, selected {scene.frame_current}")
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print("SAVED", blend_out)
bpy.ops.render.render(write_still=False, scene=scene.name)
image = bpy.data.images["Render Result"]
image.save_render(exr_out, scene=scene)
print("SAVED_EXR", exr_out)
