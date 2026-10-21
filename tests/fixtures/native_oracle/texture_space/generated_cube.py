"""Texture Coordinate Generated on a cube whose texture space is its automatic bounds, rendered by Blender 5.2.2 Cycles.

The cube's mesh is moved off the origin in its local space and the object is scaled to (2, 1, 3). The Generated
coordinate maps the local point through the object's automatic texture space, the bounds of its local mesh, so a
point at the local minimum corner is (0, 0, 0) and at the maximum corner (1, 1, 1), whatever the object's scale. A
mapping that ignored the texture-space location would leave the coordinate outside [0, 1]. The material emits the
Generated coordinate as a colour (Emission) and the world is black, so the frame mean is the mean of the coordinate
over the cube's visible pixels.

Arguments after `--`: blend_out and json_out.
"""

import json
import os
import sys
import tempfile

import bpy
from mathutils import Matrix, Vector

blend_out, json_out = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.cycles.seed = 0
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.object
cube.data.transform(Matrix.Translation((3.0, -2.0, 1.0)))
cube.scale = (2.0, 1.0, 3.0)

material = bpy.data.materials.new("Generated")
material.use_nodes = True
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
texcoord = tree.nodes.new("ShaderNodeTexCoord")
tree.links.new(texcoord.outputs["Generated"], emission.inputs["Color"])
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
cube.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 10.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = Vector((6.0, -10.0, 3.0))
camera.rotation_euler = (1.5707963267948966, 0.0, 0.0)
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
seed_means = []
for seed in range(8):
    scene.cycles.seed = seed
    scene.render.filepath = os.path.join(work, f"generated_seed_{seed}.exr")
    bpy.ops.render.render(write_still=True)
    seed_means.append(frame_mean(scene.render.filepath))

with open(json_out, "w") as handle:
    json.dump({"samples": 64, "seed_means": seed_means}, handle, indent=2)
print("DONE", flush=True)
