# Blender 5.2.2 probe for single-channel (Z-like) texels read through a modifier image texture (BKE_texture_get_value,
# texture_image.cc: ibuf_get_color, imagewrap, BKE_texture_get_value_ex). The probe writes a one-channel 1x2 EXR (BW colour
# mode, values 0.25 bottom and 0.75 top), reloads it from disk, and samples it at LOCAL (-0.5, 0.5): the top texel.
# Vertex 2 carries weight 0.6 and SHARP falloff, so the mapped weight is w*w and the result inverts to the texture value:
#   result = w*w*f + w*(1-f), f = mask_constant * tex  =>  f = (result - w) / (w*w - w), tex = f / mask_constant.
# Usage: Blender --background --factory-startup --python depth_alpha_probe.py -- <output.exr> <output.json>
import bpy
import json
import os
import sys

arguments = sys.argv[sys.argv.index("--") + 1:]
exr_path = os.path.realpath(arguments[0])
output_path = os.path.realpath(arguments[1])

written = bpy.data.images.new("DepthProbeSource", width=1, height=2, alpha=False, float_buffer=True)
# Blender pixels are bottom-up: row 0 is the bottom texel (value 0.25), row 1 the top (value 0.75).
written.pixels.foreach_set([0.25, 0.25, 0.25, 1.0, 0.75, 0.75, 0.75, 1.0])
written.update()
scene = bpy.context.scene
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "BW"
scene.render.image_settings.color_depth = "32"
written.save_render(filepath=exr_path, scene=scene)
image = bpy.data.images.load(exr_path)

verts = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0)]
faces = [(0, 1, 3, 2)]
weights = [0.2, 0.4, 0.6, 0.8]
PROBED_VERTEX = 2
MASK_CONSTANT = 0.75


def probe(use_alpha, channel):
    texture = bpy.data.textures.new("DepthProbe", type="IMAGE")
    texture.image = image
    texture.use_interpolation = False
    texture.use_alpha = use_alpha
    mesh = bpy.data.meshes.new("ProbeMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Probe", mesh)
    bpy.context.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="A")
    for index, weight in enumerate(weights):
        group.add([index], weight, "REPLACE")
    modifier = obj.modifiers.new("WeightEdit", "VERTEX_WEIGHT_EDIT")
    modifier.vertex_group = "A"
    modifier.falloff_type = "SHARP"
    modifier.normalize = False
    modifier.mask_texture = texture
    modifier.mask_tex_use_channel = channel
    modifier.mask_tex_mapping = "LOCAL"
    modifier.mask_constant = MASK_CONSTANT
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    group_index = evaluated.vertex_groups["A"].index
    result = next(
        assignment.weight
        for assignment in evaluated_mesh.vertices[PROBED_VERTEX].groups
        if assignment.group == group_index
    )
    evaluated.to_mesh_clear()
    weight = weights[PROBED_VERTEX]
    fraction = (result - weight) / (weight * weight - weight)
    return fraction / MASK_CONSTANT


results = {}
for use_alpha in (True, False):
    key = "use_alpha_" + ("on" if use_alpha else "off")
    results[key] = {channel: probe(use_alpha, channel) for channel in ("RED", "INT", "ALPHA")}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, sort_keys=True)
print("DEPTH_ALPHA_PROBE_WRITTEN")
