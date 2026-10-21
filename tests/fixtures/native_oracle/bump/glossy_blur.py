"""Filter Glossy's glossy blur, rendered by Blender 5.2.2 Cycles.

A top-down orthographic camera sees a white diffuse floor. The floor is lit only indirectly: a floor sample goes
up and hits a vertical roughness-0 metal wall, which reflects it onto an emitter standing across the scene. The
emitter's material turns light sampling off (`emission_sampling` NONE), so nothing lights the floor except a path
that bounces off the floor and then off the wall, which is what the blur acts on.

After the diffuse floor bounce the sampled pdf is cos/π, below one, so Cycles blurs the wall's specular closure
(`surface_shader_prepare_closures`): a sharp reflection becomes a rough one, which spreads the emitter's light over
the floor differently. With Filter Glossy 0 there is no blur and the wall stays a mirror. The frame mean therefore
depends on the blur.

Arguments after `--`: blend_out and json_out. The JSON has `seed_means` (Filter Glossy 1.0, the default) and
`seed_means_filter_glossy_0` (Filter Glossy 0.0).
"""

import json
import os
import sys
import tempfile

import bpy
from mathutils import Vector

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 2
scene.cycles.seed = 0
scene.cycles.blur_glossy = 1.0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"


def principled_material(name, base, metallic, roughness, specular_level):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = base
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Specular IOR Level"].default_value = specular_level
    return material


# The floor: a purely diffuse white, so its bounce is cosine-distributed.
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
floor = bpy.context.object
floor.data.materials.append(principled_material("Floor", (0.8, 0.8, 0.8, 1.0), 0.0, 0.5, 0.0))

# The wall: a vertical mirror at x = 4 facing the floor (normal -X), roughness 0 and metallic 1.
bpy.ops.mesh.primitive_plane_add(size=20.0, location=(4.0, 0.0, 5.0))
wall = bpy.context.object
wall.rotation_euler = (0.0, -1.5707963267948966, 0.0)
wall.data.materials.append(principled_material("Mirror", (1.0, 1.0, 1.0, 1.0), 1.0, 0.0, 0.5))

# The emitter: a vertical plane of 40 units at x = -4 facing the wall (normal +X). It is not light-sampled. Its size is
# measured: 20 units separates Filter Glossy 1 and 0 by 3.9 tolerances, 40 by 20.5, and smaller emitters less.
bpy.ops.mesh.primitive_plane_add(size=40.0, location=(-4.0, 0.0, 5.0))
emitter = bpy.context.object
emitter.rotation_euler = (0.0, 1.5707963267948966, 0.0)
emitter_material = bpy.data.materials.new("Emitter")
emitter_material.use_nodes = True
emitter_material.cycles.emission_sampling = "NONE"
tree = emitter_material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
emission.inputs["Strength"].default_value = 4.0
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
emitter.data.materials.append(emitter_material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 2.4
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = Vector((0.0, 0.0, 10.0))
scene.camera = camera

bpy.ops.wm.save_as_mainfile(filepath=blend_out)


def frame_mean(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    totals = [0.0, 0.0, 0.0]
    for index in range(width * height):
        for channel in range(3):
            totals[channel] += pixels[4 * index + channel]
    bpy.data.images.remove(image)
    return [total / (width * height) for total in totals]


work = tempfile.mkdtemp()
result = {"samples": 64}
for key, filter_glossy in (("seed_means", 1.0), ("seed_means_filter_glossy_0", 0.0)):
    scene.cycles.blur_glossy = filter_glossy
    seed_means = []
    for seed in range(8):
        scene.cycles.seed = seed
        scene.render.filepath = os.path.join(work, f"glossy_blur_{key}_{seed}.exr")
        bpy.ops.render.render(write_still=True)
        seed_means.append(frame_mean(scene.render.filepath))
    result[key] = seed_means

with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("DONE", flush=True)
