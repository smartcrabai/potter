"""Blender 5.2.2 EEVEE/VSE oracle for camera-light radiance in Scene strips.

Regenerate the macOS captures from the repository root with:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 \
        -P tests/fixtures/native_oracle/eevee_vse_light_shape_scene_strip.py -- \
        tests/fixtures/native_oracle/eevee_macos/vse_light_shape_scene_strip.blend \
        tests/fixtures/native_oracle/eevee_macos/vse_light_shape_scene_strip_replace.exr \
        tests/fixtures/native_oracle/eevee_macos/vse_light_shape_scene_strip_alpha_over.exr 1

The source scene has a camera-visible point-light shape on transparent film and a separate 50% emissive
base-only control plane. Two output scenes render its real Scene strip: Replace writes the source RGBA
without a background, while AlphaOver composites it over an opaque Color strip. Each capture is saved
as an EXR and read back from disk. Set the last argument to 0 to save only the .blend for Potter CLI
integration tests.
"""

import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

blend_out, replace_exr_out, over_exr_out, render_flag = sys.argv[sys.argv.index("--") + 1 :][:4]
RENDER = render_flag != "0"
RESOLUTION = 64


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


def capture(scene, exr_path, label):
    scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=False, scene=scene.name)
    # Save and reopen the actual VSE consumer output; Render Result alone is not the oracle.
    bpy.data.images["Render Result"].save_render(exr_path, scene=scene)
    image = bpy.data.images.load(exr_path, check_existing=False)
    if tuple(image.size) != (RESOLUTION, RESOLUTION):
        raise RuntimeError(f"unexpected {label} EXR size {image.size}")
    print("READBACK", os.path.basename(exr_path), image.alpha_mode, image.size[:], image.channels)
    probes = (
        ("alpha0_shape", 32, 32),
        ("fractional_base", 52, 51),
        ("background", 4, 4),
    )
    values = {}
    for probe, x, y in probes:
        value = pixel_at(image.pixels, x, y)
        values[probe] = value
        print("PIXEL", label, probe, x, y, tuple(round(channel, 7) for channel in value))
    if label == "replace":
        shape = values["alpha0_shape"]
        base = values["fractional_base"]
        if shape[3] != 0.0 or max(shape[:3]) <= 0.0:
            raise RuntimeError(f"VSE Replace must save nonzero emitter RGB at alpha zero: {shape}")
        if abs(base[3] - 0.5) > 0.01 or max(base[:3]) <= 0.0:
            raise RuntimeError(f"VSE Replace base-only control must have fractional alpha: {base}")
    else:
        for probe, value in values.items():
            if abs(value[3] - 1.0) > 1e-6:
                raise RuntimeError(f"opaque AlphaOver background lost alpha at {probe}: {value}")
    bpy.data.images.remove(image)


bpy.ops.wm.read_factory_settings(use_empty=True)
source = bpy.context.scene
source.name = "LightShapeSource"
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

# This isolated plane makes a 50% Combined pixel away from the visible point-light shape.
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(2.5, 2.5, 4.0))
control = bpy.context.object
control.name = "FractionalBaseControl"
material = bpy.data.materials.new("FractionalBaseControlMaterial")
material.use_nodes = True
material.surface_render_method = "BLENDED"
tree = material.node_tree
tree.nodes.clear()
output = tree.nodes.new("ShaderNodeOutputMaterial")
transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (0.25, 0.1, 0.05, 1.0)
emission.inputs["Strength"].default_value = 1.0
mix = tree.nodes.new("ShaderNodeMixShader")
mix.inputs["Fac"].default_value = 0.5
tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
tree.links.new(emission.outputs["Emission"], mix.inputs[2])
tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
control.data.materials.append(material)
source.frame_start = 1
source.frame_end = 1

replace_scene = bpy.data.scenes.new("LightShapeVSEReplace")
configure_render(replace_scene)
replace_scene.render.use_sequencer = True
replace_scene.frame_start = 1
replace_scene.frame_end = 1
replace_editor = replace_scene.sequence_editor_create()
replace_strips = replace_editor.strips if hasattr(replace_editor, "strips") else replace_editor.sequences
replace_strip = replace_strips.new_scene("CameraLightShapeReplace", source, 1, 1)
replace_strip.blend_type = "REPLACE"

over_scene = bpy.data.scenes.new("LightShapeVSEAlphaOver")
configure_render(over_scene)
over_scene.render.use_sequencer = True
over_scene.frame_start = 1
over_scene.frame_end = 1
over_editor = over_scene.sequence_editor_create()
over_strips = over_editor.strips if hasattr(over_editor, "strips") else over_editor.sequences
background = over_strips.new_effect("OpaqueBackground", "COLOR", 1, 1, length=1)
background.color = (0.1, 0.2, 0.3)
background.blend_alpha = 1.0
over_strip = over_strips.new_scene("CameraLightShapeAlphaOver", source, 2, 1)
over_strip.blend_type = "ALPHA_OVER"

if bpy.context.window is not None:
    bpy.context.window.scene = over_scene
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print("SAVED", blend_out)

if RENDER:
    capture(replace_scene, replace_exr_out, "replace")
    capture(over_scene, over_exr_out, "alpha_over")
