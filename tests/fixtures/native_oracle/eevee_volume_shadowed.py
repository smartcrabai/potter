"""Render the canonical `volume_base` scene with EEVEE volumetric shadows enabled.

Run in Blender 5.2.2 after `eevee_volume_factor.py` saves its `volume_base` source blend; the script asserts the base inputs before toggling shadows:

    Blender -b --factory-startup -P eevee_volume_shadowed.py -- SOURCE.blend OUTPUT.blend RENDER.exr
"""

import sys

import bpy

source_blend, blend_out, exr_out = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.open_mainfile(filepath=source_blend)
scene = bpy.context.scene

def require_base_scene(condition, detail):
    if not condition:
        raise RuntimeError(f"shadow oracle requires the canonical volume_base scene: {detail}")


light_object = scene.objects.get("Light")
light = light_object.data if light_object is not None else None
material = bpy.data.materials.get("Scatter")
volume_node = (
    next((node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeVolumeScatter"), None)
    if material and material.node_tree
    else None
)
background = (
    scene.world.node_tree.nodes.get("Background")
    if scene.world and scene.world.use_nodes and scene.world.node_tree
    else None
)
camera = scene.camera
require_base_scene(light is not None and light.type == "POINT", "Light must be a point light")
require_base_scene(light.energy == 200.0 and light.volume_factor == 1.0, "Light power/factor must be 200/1")
require_base_scene(
    light.shadow_soft_size == 0.0 and light.use_soft_falloff,
    "the point-light radius/falloff differ from the canonical base",
)
require_base_scene(volume_node is not None, "the Scatter Volume node is missing")
require_base_scene(
    volume_node.inputs["Density"].default_value == 0.5
    and tuple(volume_node.inputs["Color"].default_value) == (1.0, 1.0, 1.0, 1.0)
    and volume_node.inputs["Anisotropy"].default_value == 0.0,
    "the Volume Scatter inputs must be density 0.5, white, and isotropic",
)
require_base_scene(
    scene.render.engine == "BLENDER_EEVEE"
    and scene.eevee.taa_render_samples == 64
    and scene.render.resolution_x == 96
    and scene.render.resolution_y == 96
    and scene.render.resolution_percentage == 100
    and scene.render.filter_size == 0.0,
    "the render engine/samples/frame/filter differ from the canonical base",
)
require_base_scene(
    scene.render.image_settings.file_format == "OPEN_EXR"
    and scene.render.image_settings.color_depth == "32"
    and scene.view_settings.view_transform == "Standard"
    and scene.view_settings.look == "None"
    and scene.view_settings.exposure == 0.0
    and scene.view_settings.gamma == 1.0
    and scene.display_settings.display_device == "sRGB",
    "the image and color-management settings differ from the canonical base",
)
require_base_scene(not scene.eevee.use_volumetric_shadows, "the source shadow flag must be false")
require_base_scene(
    background is not None
    and tuple(background.inputs["Color"].default_value) == (0.0, 0.0, 0.0, 1.0)
    and background.inputs["Strength"].default_value == 1.0,
    "the world must remain black at strength 1",
)
require_base_scene(
    camera is not None
    and camera.data.type == "ORTHO"
    and camera.data.ortho_scale == 3.0
    and tuple(camera.location) == (0.0, 0.0, 10.0),
    "the orthographic camera differs from the canonical base",
)
scene.eevee.use_volumetric_shadows = True
scene.render.filepath = exr_out
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
bpy.ops.wm.save_as_mainfile(filepath=blend_out)

if exr_out == "-":
    sys.exit(0)

bpy.ops.render.render(write_still=True)
