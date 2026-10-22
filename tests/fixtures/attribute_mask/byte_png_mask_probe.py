# Blender 5.2.2 oracle: a Vertex Weight Edit with an 8-bit sRGB image mask (straight alpha), UV mapping at texel
# centres, Closest sampling. Each channel selection (INT and RED) is written as the weight after the edit, from a
# base weight of 0.5. Run: blender --background --factory-startup --python byte_png_mask_probe.py -- <out-dir>
import bpy, json, os, sys
from mathutils import Vector
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
def make_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name + 'Mesh'); mesh.from_pydata(verts, [], faces); mesh.update()
    ob = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(ob); return ob
def read_weights(ob, group):
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = ev.to_mesh()
    vg = ev.vertex_groups.get(group); values = []
    for v in mesh.vertices:
        value = 0.0
        if vg:
            for a in v.groups:
                if a.group == vg.index: value = a.weight; break
        values.append(value)
    ev.to_mesh_clear(); return values
# Blender lists rows bottom-up: (x0,y0), (x1,y0), (x0,y1), (x1,y1); 8-bit sRGB texels, straight alpha.
texels = [(128, 64, 32, 128), (200, 200, 200, 255), (0, 0, 0, 0), (255, 128, 0, 64)]
image = bpy.data.images.new('MaskPng', width=2, height=2, alpha=True, float_buffer=False)
image.colorspace_settings.name = 'sRGB'
image.alpha_mode = 'STRAIGHT'
image.pixels = [c / 255.0 for texel in texels for c in texel]
image.update()
texture = bpy.data.textures.new('MaskPngTexture', type='IMAGE')
texture.image = image
texture.use_interpolation = False
quad = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
uvs = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
result = {}
for channel in ('INT', 'RED'):
    ob = make_object('Mask' + channel, quad, [(0, 1, 2, 3)])
    ob.data.uv_layers.new(name='UVMap')
    for loop in ob.data.loops:
        ob.data.uv_layers['UVMap'].data[loop.index].uv = uvs[loop.vertex_index]
    vg = ob.vertex_groups.new(name='A')
    vg.add([0, 1, 2, 3], 0.5, 'REPLACE')
    modifier = ob.modifiers.new('Edit', 'VERTEX_WEIGHT_EDIT')
    modifier.vertex_group = 'A'
    modifier.mask_texture = texture
    modifier.falloff_type = 'SHARP'
    modifier.mask_tex_use_channel = channel
    modifier.mask_tex_mapping = 'UV'
    modifier.mask_tex_uv_layer = 'UVMap'
    modifier.normalize = False
    bpy.context.view_layer.update()
    result[channel] = read_weights(ob, 'A')
with open(os.path.join(root, 'byte-png-mask.json'), 'w') as f: json.dump(result, f, indent=2)
