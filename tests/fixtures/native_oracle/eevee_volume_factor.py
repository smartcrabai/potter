"""EEVEE oracle for a light's volume factor, with EEVEE 5.2.2's default settings.

A 2 m cube with a pure scatterer (Volume Scatter, Color 1, Density 0.5, no absorption) hangs in a black world, so a
pixel's radiance is its volume in-scatter alone: the light's volume scattering. A point light of 200 W sits 3 m above the
cube, and an orthographic camera looks down from above. The argument names the case:
- `volume_base`: the light's Volume Factor at 1;
- `volume_zero`: 0, which switches the light's in-scatter off (`power[LIGHT_VOLUME] = volume_fac * point_power`);
- `volume_half`: 0.5.

EEVEE's in-scatter is linear in the factor, and the black world leaves no other term, so `volume_half` must be half of
`volume_base`, and `volume_zero` must be black. The JSON lists every pixel with its world position and RGB.

Arguments after `--`: case, blend_out, exr_out, json_out.
"""

import json
import sys

import bpy

case, blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:4]

SIZE = 96
ORTHO_SCALE = 3.0
FACTORS = {"volume_base": 1.0, "volume_zero": 0.0, "volume_half": 0.5}
factor = FACTORS[case]

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

world = bpy.data.worlds.new("Oracle")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world

view = scene.view_settings
view.view_transform = "Standard"
view.look = "None"
view.exposure = 0.0
view.gamma = 1.0
scene.display_settings.display_device = "sRGB"

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.object
cube.name = "Cube"
material = bpy.data.materials.new("Scatter")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
for node in list(nodes):
    if node.type != "OUTPUT_MATERIAL":
        nodes.remove(node)
scatter = nodes.new("ShaderNodeVolumeScatter")
scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
scatter.inputs["Density"].default_value = 0.5
links.new(scatter.outputs["Volume"], nodes["Material Output"].inputs["Volume"])
cube.data.materials.append(material)

light_data = bpy.data.lights.new("Light", "POINT")
light_data.energy = 200.0
light_data.shadow_soft_size = 0.0
light_data.use_soft_falloff = True
light_data.volume_factor = factor
light = bpy.data.objects.new("Light", light_data)
scene.collection.objects.link(light)
light.location = (0.0, 0.0, 3.0)

camera_data = bpy.data.cameras.new("OracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("OracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
scene.camera = camera

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 64
scene.render.resolution_x = SIZE
scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.filter_size = 0.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(exr_out, check_existing=False)
width, height = result.size
pixels = list(result.pixels)
pixel_size = ORTHO_SCALE / width
samples = []
for row in range(height):
    for column in range(width):
        start = (row * width + column) * 4
        world_x = -ORTHO_SCALE / 2.0 + (column + 0.5) * pixel_size
        world_y = -ORTHO_SCALE / 2.0 + (row + 0.5) * pixel_size
        samples.append({"world": [world_x, world_y], "rgb": pixels[start : start + 3]})

with open(json_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "case": case,
            "volume_factor": factor,
            "size": SIZE,
            "ortho_scale": ORTHO_SCALE,
            "light": {"location": [0.0, 0.0, 3.0], "energy": 200.0},
            "samples": samples,
        },
        handle,
    )
