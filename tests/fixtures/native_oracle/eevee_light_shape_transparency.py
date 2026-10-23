"""Blender 5.2.2 EEVEE oracle for BLENDED-pane camera-shape visibility and transparent-film EXR output.

Regenerate from this directory with Blender 5.2.2:

    Blender -b --factory-startup --threads 1 -P eevee_light_shape_transparency.py -- \
        clear.blend clear.exr pane.blend pane.exr 1

The BLENDED pane is a 50% mix of a cyan-tinted Transparent BSDF and opaque black, covering the left half of the
orthographic view; its Combined alpha is 0.5 there. A small opaque backing plane behind the light tests additive
radiance at alpha 1, while an opaque emissive patch checks the base RGB at alpha 1. A separate half-alpha emissive patch,
away from the light shape and main pane, checks a nonzero fractional-alpha base-only sample. The pane render also has an
uncovered alpha-zero part of the emitter. The EXRs are Blender's actual transparent-film EEVEE write/readbacks; Potter's
integration test separately enables Combined and Z to exercise its multilayer EXR writer. Set the final argument to 0 to
save only the `.blend` scenes for Potter integration tests.
"""

import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

clear_blend, clear_exr, pane_blend, pane_exr, render_flag = sys.argv[sys.argv.index("--") + 1 :][:5]
RENDER = render_flag != "0"
RESOLUTION = 64
SAMPLES = 64

def add_emission_plane(name, location, size, color):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    plane = bpy.context.object
    plane.name = name
    material = bpy.data.materials.new(name + "Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(material)

def add_fractional_emission_plane(name, location, size, color):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    plane = bpy.context.object
    plane.name = name
    material = bpy.data.materials.new(name + "Material")
    material.use_nodes = True
    material.surface_render_method = "BLENDED"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.5
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    plane.data.materials.append(material)




def new_scene(with_pane):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "NONE"
    view = scene.view_settings
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0

    world = bpy.data.worlds.new("BlackWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

    camera_data = bpy.data.cameras.new("OracleCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 8.0
    camera = bpy.data.objects.new("OracleCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    scene.camera = camera

    point_data = bpy.data.lights.new("PointLight", "POINT")
    point_data.energy = 5.0
    point_data.shadow_soft_size = 1.0
    point_data.use_soft_falloff = False
    point = bpy.data.objects.new("PointLight", point_data)
    scene.collection.objects.link(point)
    point.location = (0.0, 0.0, 0.0)
    point.visible_camera = True

    if with_pane:
        # This 4 m square ends at x = 0 and covers the emitter's left half in the camera view.
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(-2.0, 0.0, 4.0))
        pane = bpy.context.object
        pane.name = "BlendPane"
        material = bpy.data.materials.new("TintedBlendPane")
        material.use_nodes = True
        material.surface_render_method = "BLENDED"
        tree = material.node_tree
        for node in list(tree.nodes):
            if node.type != "OUTPUT_MATERIAL":
                tree.nodes.remove(node)
        output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        transparent.inputs["Color"].default_value = (0.5, 1.0, 1.0, 1.0)
        opaque = tree.nodes.new("ShaderNodeEmission")
        opaque.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        opaque.inputs["Strength"].default_value = 0.0
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.4
        tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
        tree.links.new(opaque.outputs["Emission"], mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
        pane.data.materials.append(material)
        # This black opaque backing tests alpha 1 with visible light radiance.
        add_emission_plane("OpaqueBlackBacking", (-0.75, 0.0, -2.0), 0.5, (0.0, 0.0, 0.0))
        # An opaque sample away from the emitter verifies Combined's base RGB at alpha 1.
        add_emission_plane("OpaqueBaseSample", (2.5, 0.0, 4.0), 1.0, (0.25, 0.1, 0.05))
        # This base-only sample sits outside the main pane and emitter footprint.
        add_fractional_emission_plane("FractionalBaseSample", (-2.5, 2.5, 4.0), 1.0, (0.25, 0.1, 0.05))
        add_fractional_emission_plane("FractionalBaseSampleBottom", (-2.5, -2.5, 4.0), 1.0, (0.25, 0.1, 0.05))

    return scene


def render_capture(scene, exr_path):
    scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=False)
    # Read the final EXR back rather than capturing Render Result before file encoding.
    bpy.data.images["Render Result"].save_render(exr_path, scene=scene)
    image = bpy.data.images.load(exr_path, check_existing=False)
    if tuple(image.size) != (RESOLUTION, RESOLUTION):
        raise RuntimeError(f"unexpected Combined image size {image.size}")
    print("READBACK", os.path.basename(exr_path), image.alpha_mode, image.size[:], image.channels)
    bpy.data.images.remove(image)


for with_pane, blend_path, exr_path in (
    (False, clear_blend, clear_exr),
    (True, pane_blend, pane_exr),
):
    scene = new_scene(with_pane)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    if RENDER:
        render_capture(scene, exr_path)
    else:
        print("SAVED", blend_path)
