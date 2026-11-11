import json
import os
import sys

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
blend_path, oracle_path, case = argv

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"
scene.cycles.device = "CPU"
scene.cycles.samples = 512
scene.cycles.seed = 0
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.volume_bounces = 4
scene.cycles.volume_biased = False
scene.cycles.transparent_max_bounces = 8
scene.cycles.min_light_bounces = 0
scene.cycles.min_transparent_bounces = 0
scene.cycles.sample_clamp_direct = 0.0
scene.cycles.sample_clamp_indirect = 0.0
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_light_tree = False
scene.cycles.use_denoising = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.render.threads_mode = "FIXED"
scene.render.threads = 1
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

volume_world = bpy.data.worlds.new("Volume World")
scene.world = volume_world
volume_world.use_nodes = True
volume_tree = volume_world.node_tree
volume_tree.nodes.clear()
volume_output = volume_tree.nodes.new("ShaderNodeOutputWorld")
background = volume_tree.nodes.new("ShaderNodeBackground")
background.inputs["Color"].default_value = (0.15, 0.2, 0.3, 1.0)
background.inputs["Strength"].default_value = 0.3
fog = volume_tree.nodes.new("ShaderNodeVolumePrincipled")
fog.inputs["Color"].default_value = (
    (0.0, 0.0, 0.0, 1.0)
    if case == "scene_world_absorption_only"
    else (0.8, 0.7, 0.6, 1.0)
)
fog.inputs["Density"].default_value = 0.18
fog.inputs["Anisotropy"].default_value = 0.2
fog.inputs["Absorption Color"].default_value = (0.15, 0.25, 0.35, 1.0)
if case == "scene_world_scattering_only":
    fog.inputs["Absorption Color"].default_value = (1.0, 1.0, 1.0, 1.0)
volume_tree.links.new(background.outputs["Background"], volume_output.inputs["Surface"])
if case != "scene_world_no_volume":
    volume_tree.links.new(fog.outputs["Volume"], volume_output.inputs["Volume"])
volume_world.cycles.volume_sampling = (
    "DISTANCE" if case == "scene_world_distance" else "MULTIPLE_IMPORTANCE"
)
volume_world.cycles.volume_interpolation = "LINEAR"
volume_world.cycles.volume_step_size = 0.125
volume_world.cycles_visibility.scatter = True

if case == "view_layer_override":
    surface_world = bpy.data.worlds.new("Surface World")
    surface_world.color = (0.01, 0.01, 0.01)
    scene.world = surface_world
    layer = scene.view_layers[0]
    layer.world_override = volume_world
    layer.use_volumes = False
else:
    scene.world = volume_world

bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.0, 0.0, 0.0))
cube = bpy.context.object
cube.name = "Fog Receiver"
material = bpy.data.materials.new("Fog Receiver Material")
material.diffuse_color = (0.7, 0.3, 0.12, 1.0)
material.use_nodes = True
tree = material.node_tree
output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
for node in list(tree.nodes):
    if node.type == "BSDF_PRINCIPLED":
        tree.nodes.remove(node)
diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
diffuse.inputs["Color"].default_value = (0.7, 0.3, 0.12, 1.0)
tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
cube.data.materials.append(material)

light_data = bpy.data.lights.new("Fog Area", type="AREA")
light_data.energy = 70.0
light_data.shape = "DISK"
light_data.size = 1.5
light = bpy.data.objects.new("Fog Area", light_data)
scene.collection.objects.link(light)
light.visible_camera = False
light.visible_diffuse = True
light.visible_glossy = True
light.visible_transmission = True
light.visible_volume_scatter = True
light.visible_shadow = True
light.location = (1.5, -2.0, 2.0)
light.rotation_euler = (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat("-Z", "Y").to_euler()

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 1.5
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, -3.0, 1.0)
camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

scene.cycles.use_denoising = False
scene.view_layers[0].cycles.use_denoising = False
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

per_seed_files = []
for seed in range(64):
    scene.cycles.seed = seed
    output_path = os.path.join(os.path.dirname(oracle_path), f"{case}_cycles_{seed}")
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    per_seed_files.append(os.path.basename(output_path + ".exr"))

with open(oracle_path, "w", encoding="utf-8") as handle:
    json.dump(
        {"case": case, "samples": scene.cycles.samples, "per_seed_files": per_seed_files},
        handle,
        indent=2,
    )
