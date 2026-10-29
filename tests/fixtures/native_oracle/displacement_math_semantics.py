"""Blender 5.2.2 consumer-visible oracle for shader displacement math semantics.

A UV map whose U coordinates are all outside [-1, 1] feeds ARCSINE, then a
Displacement node. EEVEE's shader math returns zero out of range; Cycles clamps
the input and displaces by pi/2. The render centroid measures the resulting
plane position, not an evaluator-only value.

Arguments after --: blend_out, expected_json.
"""

import json
import math
import sys

import bpy
from mathutils import Vector


def arguments():
    return sys.argv[sys.argv.index("--") + 1 :]


def centroid_bottom_up(image):
    width, height = image.size
    pixels = list(image.pixels)
    count = 0
    row_sum = 0.0
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 4
            if max(pixels[offset : offset + 3]) > 0.1:
                count += 1
                row_sum += y + 0.5
    if count == 0:
        raise RuntimeError(
            f"displacement oracle render contains no visible plane; max={max(pixels, default=0.0)}"
        )
    return row_sum / count / height


blend_out, expected_out = arguments()
scene = bpy.context.scene
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.display.shading.light = "STUDIO"
scene.world.color = (0.0, 0.0, 0.0)
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

mesh = bpy.data.meshes.new("DisplacementPlaneMesh")
mesh.from_pydata(
    [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
    [],
    [(0, 1, 2, 3)],
)
mesh.update()
uv_layer = mesh.uv_layers.new(name="UVMap")
for loop in mesh.loops:
    x = mesh.vertices[loop.vertex_index].co.x
    uv_layer.data[loop.index].uv = (2.0 + (x + 1.0) * 0.5, 0.5)
plane = bpy.data.objects.new("DisplacementPlane", mesh)
scene.collection.objects.link(plane)

material = bpy.data.materials.new("OutOfRangeArcsineDisplacement")
material.use_nodes = True
material.displacement_method = "DISPLACEMENT"
material.node_tree.nodes.clear()
uv = material.node_tree.nodes.new("ShaderNodeTexCoord")
math_node = material.node_tree.nodes.new("ShaderNodeMath")
math_node.operation = "ARCSINE"
displacement = material.node_tree.nodes.new("ShaderNodeDisplacement")
displacement.inputs["Midlevel"].default_value = 0.0
displacement.inputs["Scale"].default_value = 1.0
emission = material.node_tree.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
links = material.node_tree.links
links.new(uv.outputs["UV"], math_node.inputs[0])
links.new(math_node.outputs["Value"], displacement.inputs["Height"])
links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
links.new(emission.outputs["Emission"], output.inputs["Surface"])
plane.data.materials.append(material)

camera_data = bpy.data.cameras.new("DisplacementCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 6.0
camera = bpy.data.objects.new("DisplacementCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, -6.0, 4.0)
camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.render.filepath = "//displacement_oracle.png"
bpy.ops.wm.save_as_mainfile(filepath=blend_out)

results = {}
for name, engine in (("eevee", "BLENDER_EEVEE"), ("cycles", "CYCLES")):
    scene.render.engine = engine
    render_path = blend_out.removesuffix(".blend") + f"_{name}.png"
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    render_image = bpy.data.images.load(render_path, check_existing=False)
    results[name] = {"centroid_bottom_up": centroid_bottom_up(render_image)}
    bpy.data.images.remove(render_image)

with open(expected_out, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "blender_version": bpy.app.version_string,
            "out_of_range_input": 2.0,
            "eevee_height": 0.0,
            "cycles_height": math.pi / 2.0,
            "renders": results,
        },
        handle,
        indent=2,
    )
    handle.write("\n")
