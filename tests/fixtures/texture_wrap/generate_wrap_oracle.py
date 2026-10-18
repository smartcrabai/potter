"""Render Blender 5.2.2's sampling of a mixed-wrap glTF texture over a UV ramp, as the oracle for Potter's sampler.

Regenerate from this directory with Blender 5.2.2 (the three fixtures, sized 32):

    for pair in clamp_repeat mirror_clamp repeat_mirror; do
        Blender -b --factory-startup -P generate_wrap_oracle.py -- wrap_$pair.gltf blender_$pair.json 32
    done

The glTF is imported with Blender's glTF importer, which builds the per-axis Separate XYZ, Math and Combine XYZ UV
graph for a mixed wrap. The base colour Image Texture is set to Non-Color and wired to an Emission shader, so Blender
interpolates raw texel values exactly as Potter does. Potter's sRGB decode order is a separate question, so the image is
not decoded on either side. A quad spans Blender UVs -0.5..1.5, covering integer boundaries and half-texel edge bands.
The Raw view transform and a near-zero box filter give each pixel centre's exact value. Each record is
[u, v, r, g, b] with the Blender UV (V-up) at the pixel centre; values are rounded to six decimals.
"""

import json
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
gltf_path = os.path.abspath(argv[0])
out_path = os.path.abspath(argv[1])
size = int(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=gltf_path)

materials = [m for m in bpy.data.materials if m.use_nodes and m.node_tree]
assert materials, "the glTF imported no node material"
mat = materials[0]
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
image_node = bsdf.inputs["Base Color"].links[0].from_node
assert image_node.type == "TEX_IMAGE", image_node.type
image_node.image.colorspace_settings.name = "Non-Color"
output = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")

uv_name = "UVMap"
for node in nt.nodes:
    if node.type == "UVMAP" and node.uv_map:
        uv_name = node.uv_map
        break

emission = nt.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
nt.links.new(image_node.outputs["Color"], emission.inputs["Color"])
for link in list(output.inputs["Surface"].links):
    nt.links.remove(link)
nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])

# Replace every imported object with one plane that carries the imported material and the ramp UVs. The plane has its
# own default UV layer, so write into the layer the material reads (creating it only when absent).
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.active_object
mesh = plane.data
uv_layer = mesh.uv_layers.get(uv_name)
if uv_layer is None:
    uv_layer = mesh.uv_layers.new(name=uv_name)
mesh.uv_layers.active = uv_layer
for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        uv_layer.data[loop_index].uv = (2.0 * co.x + 0.5, 2.0 * co.y + 0.5)
mesh.materials.clear()
mesh.materials.append(mat)

camera_data = bpy.data.cameras.new("oracle_camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 1.0
camera = bpy.data.objects.new("oracle_camera", camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 2.0)
camera.rotation_euler = (0.0, 0.0, 0.0)

scene = bpy.context.scene
scene.camera = camera
render = scene.render
render.engine = "CYCLES"
render.resolution_x = size
render.resolution_y = size
render.resolution_percentage = 100
render.film_transparent = False
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

# Render to a 32-bit EXR in the temp directory and read it back for exact float values.
exr_path = os.path.join(tempfile.gettempdir(), os.path.basename(out_path) + ".exr")
render.image_settings.file_format = "OPEN_EXR"
render.image_settings.color_depth = "32"
render.image_settings.exr_codec = "NONE"
render.filepath = exr_path
bpy.ops.render.render(write_still=True)
loaded = bpy.data.images.load(exr_path, check_existing=False)
loaded.reload()
width, height = loaded.size
assert (width, height) == (size, size), (width, height)
pixels = list(loaded.pixels)

records = []
for y in range(height):
    v = -0.5 + 2.0 * (y + 0.5) / height
    for x in range(width):
        u = -0.5 + 2.0 * (x + 0.5) / width
        offset = (y * width + x) * 4
        records.append(
            [round(u, 6), round(v, 6)]
            + [round(pixels[offset + channel], 6) for channel in range(3)]
        )

with open(out_path, "w") as handle:
    json.dump(
        {"gltf": os.path.basename(gltf_path), "uv_layer": uv_name, "size": size, "records": records},
        handle,
    )
os.remove(exr_path)
print("ORACLE_OK", os.path.basename(gltf_path), width, height, uv_name, len(records))
