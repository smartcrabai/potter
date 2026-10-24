"""Blender 5.2.2 EEVEE raw EXR capture for a nonzero fractional-alpha base-only control.

Regenerate from this directory with Blender 5.2.2:

    Blender -b --factory-startup --threads 1 -P eevee_light_shape_fractional_base.py -- \
        pane.blend pane_with_base.blend fractional_base.exr 1

The generated scene adds one half-alpha emissive patch at (2.5, 2.5, 4), outside both the main pane and point-shape
footprint. Its actual EEVEE EXR readback pins ordinary Combined RGB at fractional alpha with no camera-shape radiance.
Set the final argument to 0 to save only the `.blend` for Potter integration tests.
"""

import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

pane_blend, output_blend, exr_out, render_flag = sys.argv[sys.argv.index("--") + 1 :][:4]
RENDER = render_flag != "0"
bpy.ops.wm.open_mainfile(filepath=pane_blend)
scene = bpy.context.scene

# The input scene generator's temporary controls live at negative x; keep this oracle scene to the one positive-x control.
for name in ("FractionalBaseSample", "FractionalBaseSampleBottom"):
    probe = bpy.data.objects.get(name)
    if probe is not None:
        bpy.data.objects.remove(probe, do_unlink=True)

bpy.ops.mesh.primitive_plane_add(size=1.0, location=(2.5, 2.5, 4.0))
plane = bpy.context.object
plane.name = "FractionalBaseSamplePositive"
material = bpy.data.materials.new("FractionalBaseSamplePositiveMaterial")
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
plane.data.materials.append(material)

bpy.ops.wm.save_as_mainfile(filepath=output_blend)
if RENDER:
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "NONE"
    scene.render.filepath = exr_out
    bpy.ops.render.render(write_still=False)
    bpy.data.images["Render Result"].save_render(exr_out, scene=scene)
    image = bpy.data.images.load(exr_out, check_existing=False)
    if tuple(image.size) != (64, 64):
        raise RuntimeError(f"unexpected fractional-base EXR size {image.size}")
    print("READBACK", os.path.basename(exr_out), image.alpha_mode, image.size[:], image.channels)
    bpy.data.images.remove(image)
else:
    print("SAVED", output_blend)
