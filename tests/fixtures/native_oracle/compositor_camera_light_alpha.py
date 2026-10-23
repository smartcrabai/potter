"""Blender 5.2.2 EEVEE compositor-alpha oracle for camera-visible point-light radiance.

Capture batch A-compositor-alpha-oracle-20261023 (macOS arm64):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --disable-autoexec --threads 1 --python-exit-code 3 \
        --python tests/fixtures/native_oracle/compositor_camera_light_alpha.py -- \
        tests/fixtures/native_oracle/eevee_macos

The saved scene inputs omit compositor nodes so the consumer test applies the same graphs through
Potter's CLI. The AlphaOver capture composites the transparent EEVEE render over an opaque color;
the Blur capture is a compositor File Output node's saved float RGBA EXR, not Potter output.
"""

import glob
import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

OUTPUT_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
RESOLUTION = 64
SAMPLES = 64
LIGHT_COLOR = (1.0, 0.25, 0.05)
LIGHT_ENERGY = 1000.0
LIGHT_RADIUS = 1.5
OPAQUE_BACKGROUND = (0.1, 0.25, 0.5)


def new_scene(name, half_opaque_background=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = name
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
    scene.render.filepath = os.path.join(OUTPUT_DIR, "unused")
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

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

    point_data = bpy.data.lights.new("CameraPointLight", "POINT")
    point_data.energy = LIGHT_ENERGY
    point_data.color = LIGHT_COLOR
    point_data.shadow_soft_size = LIGHT_RADIUS
    point_data.use_soft_falloff = False
    point = bpy.data.objects.new("CameraPointLight", point_data)
    scene.collection.objects.link(point)
    point.location = (0.0, 0.0, 0.0)
    point.visible_camera = True

    if half_opaque_background:
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(-2.0, 0.0, -2.0))
        plane = bpy.context.object
        plane.name = "OpaqueHalfPlane"
        material = bpy.data.materials.new("OpaqueBlack")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        emission.inputs["Strength"].default_value = 0.0
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        plane.data.materials.append(material)
    return scene


def new_compositor_tree(scene, name):
    tree = bpy.data.node_groups.new(name, "CompositorNodeTree")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    return tree


def render_alpha_over(scene):
    tree = new_compositor_tree(scene, "Camera Light AlphaOver")
    layers = tree.nodes.new("CompositorNodeRLayers")
    alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
    alpha_over.inputs["Background"].default_value = (*OPAQUE_BACKGROUND, 1.0)
    group_output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], alpha_over.inputs["Foreground"])
    tree.links.new(alpha_over.outputs["Image"], group_output.inputs["Image"])
    capture = os.path.join(OUTPUT_DIR, "compositor_alpha_over_camera_light.exr")
    scene.render.filepath = os.path.splitext(capture)[0]
    bpy.ops.render.render(write_still=True)
    if not os.path.isfile(capture):
        raise RuntimeError(f"Blender did not write AlphaOver capture {capture}")


def render_blur_output(scene):
    tree = new_compositor_tree(scene, "Camera Light Blur")
    layers = tree.nodes.new("CompositorNodeRLayers")
    blur = tree.nodes.new("CompositorNodeBlur")
    blur.inputs["Size"].default_value = (2.0, 2.0)
    file_output = tree.nodes.new("CompositorNodeOutputFile")
    file_output.directory = OUTPUT_DIR
    file_output.file_name = "compositor_camera_light_blur_"
    file_output.file_output_items.new("RGBA", "Image")
    file_output.format.file_format = "OPEN_EXR_MULTILAYER"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "32"
    file_output.format.exr_codec = "NONE"
    group_output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], blur.inputs["Image"])
    tree.links.new(blur.outputs["Image"], file_output.inputs["Image"])
    tree.links.new(blur.outputs["Image"], group_output.inputs["Image"])
    scene.render.filepath = os.path.join(OUTPUT_DIR, "compositor_camera_light_blur_main")
    bpy.ops.render.render(write_still=False)
    matches = glob.glob(os.path.join(OUTPUT_DIR, "compositor_camera_light_blur_*.exr"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one saved Blur EXR, got {matches}")
    capture = os.path.join(OUTPUT_DIR, "compositor_blur_hdr_saved.exr")
    os.replace(matches[0], capture)


def save_scene(scene, name):
    path = os.path.join(OUTPUT_DIR, name)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Blender did not write scene input {path}")


os.makedirs(OUTPUT_DIR, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0
alpha_scene = new_scene("CameraLightAlphaOver")
alpha_scene_blend = os.path.join(OUTPUT_DIR, "compositor_alpha_over_scene.blend")
if not os.path.isfile(alpha_scene_blend):
    save_scene(alpha_scene, "compositor_alpha_over_scene.blend")
alpha_capture = os.path.join(OUTPUT_DIR, "compositor_alpha_over_camera_light.exr")
if not os.path.isfile(alpha_capture):
    render_alpha_over(alpha_scene)

blur_scene = new_scene("CameraLightBlurHDR", half_opaque_background=True)
blur_scene_blend = os.path.join(OUTPUT_DIR, "compositor_blur_hdr_scene.blend")
if not os.path.isfile(blur_scene_blend):
    save_scene(blur_scene, "compositor_blur_hdr_scene.blend")
blur_capture = os.path.join(OUTPUT_DIR, "compositor_blur_hdr_saved.exr")
if not os.path.isfile(blur_capture):
    render_blur_output(blur_scene)

for name in (
    "compositor_alpha_over_scene.blend",
    "compositor_alpha_over_camera_light.exr",
    "compositor_blur_hdr_scene.blend",
    "compositor_blur_hdr_saved.exr",
):
    print("CAPTURE", name, os.path.getsize(os.path.join(OUTPUT_DIR, name)))
