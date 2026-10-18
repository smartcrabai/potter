"""Blender 5.2.2's Image Texture sampling of a multilayer EXR made from a real render, as an oracle for Potter's loader.

Regenerate from this directory with Blender 5.2.2, one thread, one run per case:

    Blender -b --factory-startup --threads 1 -P generate_multilayer_oracle.py -- \
        INTERPOLATION LINKED OUT.json MULTILAYER.exr

Stage 1 renders a 4x4 frame with film transparency: a plane covers the frame with a Mix Shader at 0.5 of an Emission
(0.8, 0.4, 0.2) and a Transparent BSDF, so every pixel has alpha 0.5. A Z pass is enabled, which makes Blender write the
Render Result as a multilayer EXR (`Combined` and `Depth` layers). The Render Result is saved to MULTILAYER.exr.

Stage 2 loads that file as an image, dumps its raw Combined pixels and its alpha mode, then samples it through an Image
Texture on a ramp plane over UVs -0.5..1.5, linked or unlinked (a linked Alpha output feeds a Mix Shader's Fac between two
Emission nodes that both read the Color). Each record is [u, v, r, g, b, sample_u, sample_v] as in generate_alpha_oracle.py.
The raw texels are written to the JSON, so the model can be checked against exactly what Blender loaded.
"""

import json
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
interpolation, linked_name, out_path, multilayer_path = argv[:4]
linked = linked_name == "linked"
out_path = os.path.abspath(out_path)
multilayer_path = os.path.abspath(multilayer_path)
SIZE = 4


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = SIZE
    render.resolution_y = SIZE
    render.resolution_percentage = 100
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 0.0001
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    camera_data = bpy.data.cameras.new("cam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("cam", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 2.0)
    scene.camera = camera
    return scene


def make_plane(name, size):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = name
    return plane


# Stage 1: the render that becomes the multilayer file.
scene = reset_scene()
scene.render.film_transparent = True
view_layer = scene.view_layers[0]
view_layer.use_pass_z = True
plane = make_plane("RenderPlane", 2.0)
mat = bpy.data.materials.new("render_alpha")
mat.use_nodes = True
nt = mat.node_tree
for node in list(nt.nodes):
    nt.nodes.remove(node)
emission = nt.nodes.new("ShaderNodeEmission")
emission.inputs["Color"].default_value = (0.8, 0.4, 0.2, 1.0)
emission.inputs["Strength"].default_value = 1.0
transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
mix = nt.nodes.new("ShaderNodeMixShader")
mix.inputs["Fac"].default_value = 0.5
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(emission.outputs["Emission"], mix.inputs[1])
nt.links.new(transparent.outputs["BSDF"], mix.inputs[2])
nt.links.new(mix.outputs["Shader"], output.inputs["Surface"])
plane.data.materials.append(mat)
image_settings = scene.render.image_settings
image_settings.file_format = "OPEN_EXR"
image_settings.color_depth = "32"
image_settings.color_mode = "RGBA"
image_settings.exr_codec = "NONE"
scene.render.filepath = os.path.join(os.path.dirname(multilayer_path), "render_stage_still.exr")
bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(multilayer_path)

# Stage 2: sample the multilayer file.
scene = reset_scene()
ml = bpy.data.images.load(multilayer_path, check_existing=False)
ml_alpha_mode = ml.alpha_mode
ml_width, ml_height = ml.size
ml_channels = ml.channels
texels = [list(ml.pixels[i : i + 4]) for i in range(0, len(ml.pixels), 4)]

sample_plane = make_plane("SamplePlane", 1.0)
mesh = sample_plane.data
layer = mesh.uv_layers.new(name="Lightmap")
mesh.uv_layers.active = layer
for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        layer.data[loop_index].uv = (2.0 * co.x + 0.5, 2.0 * co.y + 0.5)
scene.camera.data.ortho_scale = 1.0

sample_mat = bpy.data.materials.new("sample")
sample_mat.use_nodes = True
snt = sample_mat.node_tree
for node in list(snt.nodes):
    snt.nodes.remove(node)
tex = snt.nodes.new("ShaderNodeTexImage")
tex.image = ml
tex.interpolation = interpolation
tex.extension = "REPEAT"
uv = snt.nodes.new("ShaderNodeUVMap")
uv.uv_map = "Lightmap"
emission_a = snt.nodes.new("ShaderNodeEmission")
emission_a.inputs["Strength"].default_value = 1.0
emission_b = snt.nodes.new("ShaderNodeEmission")
emission_b.inputs["Strength"].default_value = 1.0
sample_output = snt.nodes.new("ShaderNodeOutputMaterial")
snt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
snt.links.new(tex.outputs["Color"], emission_a.inputs["Color"])
snt.links.new(tex.outputs["Color"], emission_b.inputs["Color"])
if linked:
    shader_mix = snt.nodes.new("ShaderNodeMixShader")
    snt.links.new(tex.outputs["Alpha"], shader_mix.inputs["Fac"])
    snt.links.new(emission_a.outputs["Emission"], shader_mix.inputs[1])
    snt.links.new(emission_b.outputs["Emission"], shader_mix.inputs[2])
    snt.links.new(shader_mix.outputs["Shader"], sample_output.inputs["Surface"])
else:
    snt.links.new(emission_a.outputs["Emission"], sample_output.inputs["Surface"])
sample_plane.data.materials.append(sample_mat)

render_settings = scene.render
render_settings.resolution_x = 32
render_settings.resolution_y = 32
scene.cycles.filter_width = 0.0001
render_settings.image_settings.file_format = "OPEN_EXR"
render_settings.image_settings.color_depth = "32"
render_settings.image_settings.exr_codec = "NONE"
work = os.path.dirname(multilayer_path)


def render_to(path):
    render_settings.filepath = path
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(path, check_existing=False)
    loaded.reload()
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    return pixels


color_pixels = render_to(os.path.join(work, "multilayer_colour.exr"))
for link in list(emission_a.inputs["Color"].links):
    snt.links.remove(link)
for link in list(emission_b.inputs["Color"].links):
    snt.links.remove(link)
snt.links.new(uv.outputs["UV"], emission_a.inputs["Color"])
snt.links.new(uv.outputs["UV"], emission_b.inputs["Color"])
uv_pixels = render_to(os.path.join(work, "multilayer_uv.exr"))

size = 32
records = []
for y in range(size):
    v = -0.5 + 2.0 * (y + 0.5) / size
    for x in range(size):
        u = -0.5 + 2.0 * (x + 0.5) / size
        offset = (y * size + x) * 4
        records.append(
            [round(u, 6), round(v, 6)]
            + [round(color_pixels[offset + c], 6) for c in range(3)]
            + [round(uv_pixels[offset], 7), round(uv_pixels[offset + 1], 7)]
        )
with open(out_path, "w") as handle:
    json.dump(
        {
            "container": "multilayer",
            "colorspace": "Linear",
            "interpolation": interpolation,
            "extension": "REPEAT",
            "alpha_output": linked_name,
            "alpha_mode": ml_alpha_mode,
            "width": ml_width,
            "height": ml_height,
            "channels": ml_channels,
            "texels": [[round(c, 9) for c in texel] for texel in texels],
            "size": size,
            "records": records,
        },
        handle,
    )
print("ORACLE_OK multilayer", interpolation, linked_name, ml_alpha_mode, ml_width, ml_height, ml_channels, len(records))
