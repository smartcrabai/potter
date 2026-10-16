"""PR10 node-material probe: one UV sphere per material graph (P01-P45), Blender 5.2.2.

Run headless:
    Blender -b --factory-startup --python build_probe.py

Environment:
    POTTER_PROBE_OUT   directory for materials.blend and images/ (default: current directory)
    POTTER_PROBE_ONLY  optional comma-separated material names; builds only those probes
"""

import math
import os

import bpy

OUT = os.environ.get("POTTER_PROBE_OUT", os.getcwd())
IMG_DIR = os.path.join(OUT, "images")
os.makedirs(IMG_DIR, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)


def make_png(name, w, h, fn, colorspace="sRGB"):
    img = bpy.data.images.new(name, w, h, alpha=True)
    px = []
    for y in range(h):
        for x in range(w):
            px.extend(fn(x, y))
    img.pixels.foreach_set(px)
    path = os.path.join(IMG_DIR, name + ".png")
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    loaded = bpy.data.images.load(path, check_existing=False)
    loaded.name = name
    loaded.colorspace_settings.name = colorspace
    return loaded


img_base = make_png("probe_base", 4, 4, lambda x, y: ((1.0, 0.2, 0.2, 1.0) if (x + y) % 2 else (0.2, 0.8, 0.3, 1.0)))
img_rgba = make_png("probe_rgba", 4, 4, lambda x, y: (0.9, 0.9, 0.2, 0.9 if (x + y) % 2 else 0.2))
img_normal = make_png("probe_normal", 4, 4, lambda x, y: (0.6, 0.5, 0.9, 1.0), colorspace="Non-Color")
img_orm = make_png("probe_orm", 4, 4, lambda x, y: (0.8, 0.4, 0.9, 1.0), colorspace="Non-Color")
img_emit = make_png("probe_emit", 4, 4, lambda x, y: (1.0, 0.6, 0.1, 1.0))


def nn(nt, kind, loc=(0, 0)):
    n = nt.nodes.new(kind)
    n.location = loc
    return n


def link(nt, out_node, out_name, in_node, in_name):
    nt.links.new(out_node.outputs[out_name], in_node.inputs[in_name])


def sock_by_id(node, ident, is_input=True):
    socks = node.inputs if is_input else node.outputs
    for s in socks:
        if s.identifier == ident:
            return s
    raise KeyError(ident)


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL' and n.type != 'BSDF_PRINCIPLED':
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    return m, nt, out, bsdf


def image_node(nt, img, loc):
    t = nn(nt, 'ShaderNodeTexImage', loc)
    t.image = img
    return t


def uv_vector_for(nt, tex, uv_name=None, mapping=None):
    """Routes UVMap -> Mapping -> image Vector; either part is optional."""
    src = None
    if uv_name is not None:
        uvn = nn(nt, 'ShaderNodeUVMap', (-900, 0))
        uvn.uv_map = uv_name
        src = uvn
    if mapping is not None:
        mp = nn(nt, 'ShaderNodeMapping', (-600, 0))
        mp.inputs['Location'].default_value = mapping['loc']
        mp.inputs['Rotation'].default_value = mapping['rot']
        mp.inputs['Scale'].default_value = mapping['scale']
        if src is not None:
            nt.links.new(src.outputs['UV'], mp.inputs['Vector'])
        nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    elif src is not None:
        nt.links.new(src.outputs['UV'], tex.inputs['Vector'])


materials = []


def add(name, builder):
    materials.append((name, builder))


# P01 Blender default material (no edits)
def b_default(m, nt, out, bsdf):
    pass


add("P01_Default", b_default)


# P02 base color texture through UVMap -> Mapping (location, z-rotation, scale)
def b_basecolor_mapping(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-300, 200))
    uv_vector_for(nt, t, uv_name="UVMap", mapping={'loc': (0.1, 0.2, 0.0), 'rot': (0.0, 0.0, math.radians(30)), 'scale': (2.0, 3.0, 1.0)})
    link(nt, t, 'Color', bsdf, 'Base Color')
    bsdf.inputs['Metallic'].default_value = 0.25
    bsdf.inputs['Roughness'].default_value = 0.5


add("P02_BaseColorTexMapping", b_basecolor_mapping)


# P03 RGBA texture: color -> Base Color, texture alpha -> Alpha (direct, no math)
def b_alpha_direct(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-300, 200))
    link(nt, t, 'Color', bsdf, 'Base Color')
    link(nt, t, 'Alpha', bsdf, 'Alpha')


add("P03_AlphaTextureDirect", b_alpha_direct)


# P04 alpha clip: Alpha -> Math GREATER_THAN 0.5 -> Alpha
def b_alpha_clip_gt(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-300, 200))
    link(nt, t, 'Color', bsdf, 'Base Color')
    mth = nn(nt, 'ShaderNodeMath', (-100, -100))
    mth.operation = 'GREATER_THAN'
    mth.inputs[1].default_value = 0.5
    link(nt, t, 'Alpha', mth, 0)
    link(nt, mth, 'Value', bsdf, 'Alpha')


add("P04_AlphaClipGreaterThan", b_alpha_clip_gt)


# P05 alpha clip via Math ROUND
def b_alpha_round(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-300, 200))
    link(nt, t, 'Color', bsdf, 'Base Color')
    mth = nn(nt, 'ShaderNodeMath', (-100, -100))
    mth.operation = 'ROUND'
    link(nt, t, 'Alpha', mth, 0)
    link(nt, mth, 'Value', bsdf, 'Alpha')


add("P05_AlphaClipRound", b_alpha_round)


# P06 constant alpha 0.5
def b_alpha_const(m, nt, out, bsdf):
    bsdf.inputs['Base Color'].default_value = (0.2, 0.4, 0.8, 1.0)
    bsdf.inputs['Alpha'].default_value = 0.5


add("P06_AlphaConstant", b_alpha_const)


# P07 normal map (Strength 0.6) from Non-Color image
def b_normal(m, nt, out, bsdf):
    t = image_node(nt, img_normal, (-500, -200))
    nm = nn(nt, 'ShaderNodeNormalMap', (-200, -200))
    nm.space = 'TANGENT'
    nm.inputs['Strength'].default_value = 0.6
    link(nt, t, 'Color', nm, 'Color')
    link(nt, nm, 'Normal', bsdf, 'Normal')


add("P07_NormalMapStrength", b_normal)


# P08 Principled emission color constant with strength 5
def b_emit_principled(m, nt, out, bsdf):
    bsdf.inputs['Emission Color'].default_value = (1.0, 0.5, 0.2, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 5.0


add("P08_EmissionPrincipled", b_emit_principled)


# P09 emission texture into Principled Emission Color, strength 2
def b_emit_tex(m, nt, out, bsdf):
    t = image_node(nt, img_emit, (-300, 200))
    link(nt, t, 'Color', bsdf, 'Emission Color')
    bsdf.inputs['Emission Strength'].default_value = 2.0


add("P09_EmissionTexture", b_emit_tex)


# P10 Emission shader as Surface
def b_emission_shader(m, nt, out, bsdf):
    e = nn(nt, 'ShaderNodeEmission', (0, 0))
    e.inputs['Color'].default_value = (0.2, 0.6, 1.0, 1.0)
    e.inputs['Strength'].default_value = 3.0
    nt.links.new(e.outputs['Emission'], out.inputs['Surface'])


add("P10_EmissionShaderSurface", b_emission_shader)


# P11 Background as Surface (unlit)
def b_background(m, nt, out, bsdf):
    e = nn(nt, 'ShaderNodeBackground', (0, 0))
    e.inputs['Color'].default_value = (0.1, 0.2, 0.3, 1.0)
    e.inputs['Strength'].default_value = 1.0
    nt.links.new(e.outputs['Background'], out.inputs['Surface'])


add("P11_BackgroundUnlit", b_background)


# P12 Mix(Transparent, Principled) with Fac = texture alpha
def b_mix_transparent(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-500, 200))
    bsdf.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1.0)
    tr = nn(nt, 'ShaderNodeBsdfTransparent', (-100, -200))
    mx = nn(nt, 'ShaderNodeMixShader', (150, 0))
    nt.links.new(t.outputs['Alpha'], mx.inputs['Fac'])
    nt.links.new(tr.outputs['BSDF'], mx.inputs[1])
    nt.links.new(bsdf.outputs['BSDF'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])


add("P12_MixTransparentAlpha", b_mix_transparent)


# P13 ORM texture via Separate Color -> Roughness (Green) and Metallic (Blue)
def b_orm_sep(m, nt, out, bsdf):
    t = image_node(nt, img_orm, (-500, 0))
    sc = nn(nt, 'ShaderNodeSeparateColor', (-250, 0))
    link(nt, t, 'Color', sc, 'Color')
    nt.links.new(sc.outputs['Green'], bsdf.inputs['Roughness'])
    nt.links.new(sc.outputs['Blue'], bsdf.inputs['Metallic'])


add("P13_ORMSeparateColor", b_orm_sep)


# P14 ORM with occlusion in 'glTF Material Output' group (Red -> Occlusion input)
def b_orm_gltf_group(m, nt, out, bsdf):
    t = image_node(nt, img_orm, (-500, 0))
    sc = nn(nt, 'ShaderNodeSeparateColor', (-250, 0))
    link(nt, t, 'Color', sc, 'Color')
    nt.links.new(sc.outputs['Green'], bsdf.inputs['Roughness'])
    nt.links.new(sc.outputs['Blue'], bsdf.inputs['Metallic'])
    ng = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
    ng.interface.new_socket(name='Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    grp = nn(nt, 'ShaderNodeGroup', (-50, -250))
    grp.node_tree = ng
    nt.links.new(sc.outputs['Red'], grp.inputs['Occlusion'])


add("P14_ORMWithGltfOcclusionGroup", b_orm_gltf_group)


# P15 vertex color (layer 'Col') multiplied into base color texture
def b_vcolor_mul(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    vc = nn(nt, 'ShaderNodeVertexColor', (-500, -100))
    vc.layer_name = 'Col'
    mx = nn(nt, 'ShaderNodeMix', (-250, 100))
    mx.data_type = 'RGBA'
    mx.blend_type = 'MULTIPLY'
    mx.inputs['Factor'].default_value = 1.0
    nt.links.new(t.outputs['Color'], sock_by_id(mx, 'A_Color'))
    nt.links.new(vc.outputs['Color'], sock_by_id(mx, 'B_Color'))
    nt.links.new(mx.outputs[2] if len(mx.outputs) > 2 else mx.outputs[0], bsdf.inputs['Base Color'])


add("P15_VertexColorMultiply", b_vcolor_mul)


# P16 IOR + transmission + clearcoat + sheen + specular constants
def b_extensions(m, nt, out, bsdf):
    bsdf.inputs['IOR'].default_value = 1.33
    bsdf.inputs['Transmission Weight'].default_value = 0.5
    bsdf.inputs['Coat Weight'].default_value = 0.8
    bsdf.inputs['Coat Roughness'].default_value = 0.1
    bsdf.inputs['Sheen Weight'].default_value = 0.5
    bsdf.inputs['Sheen Tint'].default_value = (1.0, 0.0, 0.0, 1.0)
    bsdf.inputs['Sheen Roughness'].default_value = 0.4
    bsdf.inputs['Specular IOR Level'].default_value = 0.3
    bsdf.inputs['Specular Tint'].default_value = (0.5, 0.5, 1.0, 1.0)


add("P16_TransmissionCoatSheenSpecular", b_extensions)


# P17 Bump node (height from texture) -> Normal
def b_bump(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, -200))
    bp = nn(nt, 'ShaderNodeBump', (-200, -200))
    bp.inputs['Strength'].default_value = 0.5
    link(nt, t, 'Color', bp, 'Height')
    nt.links.new(bp.outputs['Normal'], bsdf.inputs['Normal'])


add("P17_BumpNode", b_bump)


# P18 Noise texture -> Base Color (procedural, no image)
def b_noise(m, nt, out, bsdf):
    nz = nn(nt, 'ShaderNodeTexNoise', (-300, 200))
    nt.links.new(nz.outputs['Color'], bsdf.inputs['Base Color'])


add("P18_NoiseBaseColor", b_noise)


# P19 Image -> Math MULTIPLY 0.5 -> Metallic
def b_metal_math(m, nt, out, bsdf):
    t = image_node(nt, img_orm, (-500, 0))
    mth = nn(nt, 'ShaderNodeMath', (-250, 0))
    mth.operation = 'MULTIPLY'
    mth.inputs[1].default_value = 0.5
    link(nt, t, 'Color', mth, 0)
    link(nt, mth, 'Value', bsdf, 'Metallic')


add("P19_MetallicMathFactor", b_metal_math)


# P20 UVMap named 'UVLight' (second UV layer) for base color
def b_uv_second(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-300, 200))
    uv_vector_for(nt, t, uv_name="UVLight")
    link(nt, t, 'Color', bsdf, 'Base Color')


add("P20_SecondUVLayer", b_uv_second)


# P21 image plugged straight into Roughness (no Separate Color)
def b_rough_direct(m, nt, out, bsdf):
    t = image_node(nt, img_orm, (-300, 0))
    link(nt, t, 'Color', bsdf, 'Roughness')


add("P21_RoughnessTextureDirect", b_rough_direct)


# P22 base color = Mix MULTIPLY(image, constant (1,0.5,0.5))
def b_const_mul(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    mx = nn(nt, 'ShaderNodeMix', (-250, 100))
    mx.data_type = 'RGBA'
    mx.blend_type = 'MULTIPLY'
    mx.inputs['Factor'].default_value = 1.0
    nt.links.new(t.outputs['Color'], sock_by_id(mx, 'A_Color'))
    sock_by_id(mx, 'B_Color').default_value = (1.0, 0.5, 0.5, 1.0)
    nt.links.new(mx.outputs[2] if len(mx.outputs) > 2 else mx.outputs[0], bsdf.inputs['Base Color'])


add("P22_BaseColorConstantMultiply", b_const_mul)


# P23 clearcoat weight from image (R channel)
def b_coat_tex(m, nt, out, bsdf):
    t = image_node(nt, img_orm, (-300, 0))
    link(nt, t, 'Color', bsdf, 'Coat Weight')
    bsdf.inputs['Coat Roughness'].default_value = 0.2


add("P23_ClearcoatTexture", b_coat_tex)


# P24 Emission shader mixed 50% with the Principled shader
def b_emit_mix(m, nt, out, bsdf):
    e = nn(nt, 'ShaderNodeEmission', (0, -200))
    e.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    e.inputs['Strength'].default_value = 1.0
    mx = nn(nt, 'ShaderNodeMixShader', (150, 0))
    mx.inputs['Fac'].default_value = 0.5
    nt.links.new(bsdf.outputs['BSDF'], mx.inputs[1])
    nt.links.new(e.outputs['Emission'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])


add("P24_MixEmissionShader", b_emit_mix)


# P25 backface culling off (double sided)
def b_double(m, nt, out, bsdf):
    m.use_backface_culling = False
    bsdf.inputs['Base Color'].default_value = (0.3, 0.6, 0.9, 1.0)


add("P25_DoubleSided", b_double)


# P26 Background with strength 2 and color 0.5
def b_bg_strength(m, nt, out, bsdf):
    e = nn(nt, 'ShaderNodeBackground', (0, 0))
    e.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    e.inputs['Strength'].default_value = 2.0
    nt.links.new(e.outputs['Background'], out.inputs['Surface'])


add("P26_BackgroundStrength2", b_bg_strength)


# P27 object-space normal map
def b_normal_object(m, nt, out, bsdf):
    t = image_node(nt, img_normal, (-500, -200))
    nm = nn(nt, 'ShaderNodeNormalMap', (-200, -200))
    nm.space = 'OBJECT'
    link(nt, t, 'Color', nm, 'Color')
    nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])


add("P27_NormalMapObjectSpace", b_normal_object)


# P28 alpha clip: Alpha -> Math LESS_THAN 0.5 -> SUBTRACT(1, x) -> Alpha
def b_alpha_inverse_clip(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-500, 200))
    lt = nn(nt, 'ShaderNodeMath', (-300, -100))
    lt.operation = 'LESS_THAN'
    lt.inputs[1].default_value = 0.5
    sb = nn(nt, 'ShaderNodeMath', (-100, -100))
    sb.operation = 'SUBTRACT'
    sb.inputs[0].default_value = 1.0
    link(nt, t, 'Alpha', lt, 0)
    link(nt, lt, 'Value', sb, 1)
    link(nt, sb, 'Value', bsdf, 'Alpha')


add("P28_AlphaClipOneMinusLessThan", b_alpha_inverse_clip)


# P29 Mapping with X/Y rotation (not representable in glTF)
def b_mapping_rotx(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    uv_vector_for(nt, t, uv_name="UVMap", mapping={'loc': (0.0, 0.0, 0.0), 'rot': (0.3, 0.0, 0.2), 'scale': (1.0, 1.0, 1.0)})
    link(nt, t, 'Color', bsdf, 'Base Color')


add("P29_MappingRotationXY", b_mapping_rotx)


# P30 Mapping in TEXTURE mode (inverted TRS)
def b_mapping_texture(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    mp = nn(nt, 'ShaderNodeMapping', (-300, 200))
    mp.vector_type = 'TEXTURE'
    mp.inputs['Location'].default_value = (0.25, 0.0, 0.0)
    mp.inputs['Scale'].default_value = (2.0, 2.0, 1.0)
    nt.links.new(mp.outputs['Vector'], t.inputs['Vector'])
    link(nt, t, 'Color', bsdf, 'Base Color')


add("P30_MappingTextureType", b_mapping_texture)


# P31 Closest interpolation with MIRROR extension
def b_interp_wrap(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    t.interpolation = 'Closest'
    t.extension = 'MIRROR'
    link(nt, t, 'Color', bsdf, 'Base Color')


add("P31_ClosestMirror", b_interp_wrap)


# P32 two images added (Mix ADD)
def b_two_images_add(m, nt, out, bsdf):
    a = image_node(nt, img_base, (-500, 200))
    b = image_node(nt, img_emit, (-500, -100))
    mx = nn(nt, 'ShaderNodeMix', (-250, 100))
    mx.data_type = 'RGBA'
    mx.blend_type = 'ADD'
    mx.inputs['Factor'].default_value = 0.5
    nt.links.new(a.outputs['Color'], sock_by_id(mx, 'A_Color'))
    nt.links.new(b.outputs['Color'], sock_by_id(mx, 'B_Color'))
    nt.links.new(mx.outputs[2] if len(mx.outputs) > 2 else mx.outputs[0], bsdf.inputs['Base Color'])


add("P32_TwoImagesMixAdd", b_two_images_add)


# P33 image added to a constant (Mix ADD)
def b_mix_add_const(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    mx = nn(nt, 'ShaderNodeMix', (-250, 100))
    mx.data_type = 'RGBA'
    mx.blend_type = 'ADD'
    mx.inputs['Factor'].default_value = 1.0
    nt.links.new(t.outputs['Color'], sock_by_id(mx, 'A_Color'))
    sock_by_id(mx, 'B_Color').default_value = (0.3, 0.3, 0.3, 1.0)
    nt.links.new(mx.outputs[2] if len(mx.outputs) > 2 else mx.outputs[0], bsdf.inputs['Base Color'])


add("P33_MixAddConstantBaseColor", b_mix_add_const)


# P34 Displacement output connected
def b_displacement(m, nt, out, bsdf):
    bsdf.inputs['Base Color'].default_value = (0.4, 0.8, 0.4, 1.0)
    t = image_node(nt, img_base, (-500, -200))
    dp = nn(nt, 'ShaderNodeDisplacement', (-200, -250))
    link(nt, t, 'Color', dp, 'Height')
    nt.links.new(dp.outputs['Displacement'], out.inputs['Displacement'])


add("P34_DisplacementOutput", b_displacement)


# P35 anisotropy and rotation, subsurface and thin film (no glTF representation for the last two)
def b_aniso_subsurface(m, nt, out, bsdf):
    bsdf.inputs['Anisotropic'].default_value = 0.5
    bsdf.inputs['Anisotropic Rotation'].default_value = 0.25
    bsdf.inputs['Subsurface Weight'].default_value = 0.5
    bsdf.inputs['Thin Film Thickness'].default_value = 300.0


add("P35_AnisotropySubsurfaceThinFilm", b_aniso_subsurface)


# P36 two Principled shaders mixed
def b_two_principled_mix(m, nt, out, bsdf):
    bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    b2 = nn(nt, 'ShaderNodeBsdfPrincipled', (0, -300))
    b2.inputs['Base Color'].default_value = (0.0, 0.0, 1.0, 1.0)
    mx = nn(nt, 'ShaderNodeMixShader', (150, 0))
    mx.inputs['Fac'].default_value = 0.5
    nt.links.new(bsdf.outputs['BSDF'], mx.inputs[1])
    nt.links.new(b2.outputs['BSDF'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])


add("P36_TwoPrincipledMix", b_two_principled_mix)


# P37 Glass shader as Surface
def b_glass(m, nt, out, bsdf):
    g = nn(nt, 'ShaderNodeBsdfGlass', (0, 0))
    g.inputs['Color'].default_value = (0.1, 0.9, 0.9, 1.0)
    g.inputs['IOR'].default_value = 1.45
    nt.links.new(g.outputs['BSDF'], out.inputs['Surface'])


add("P37_GlassOnly", b_glass)


# P38 Diffuse shader as Surface with an image color
def b_diffuse_only(m, nt, out, bsdf):
    t = image_node(nt, img_base, (-500, 200))
    d = nn(nt, 'ShaderNodeBsdfDiffuse', (0, 0))
    link(nt, t, 'Color', d, 'Color')
    nt.links.new(d.outputs['BSDF'], out.inputs['Surface'])


add("P38_DiffuseOnlyTexture", b_diffuse_only)


# P39 tangent normal map without strength change
def b_normal_tangent_only(m, nt, out, bsdf):
    t = image_node(nt, img_normal, (-500, -200))
    nm = nn(nt, 'ShaderNodeNormalMap', (-200, -200))
    nm.space = 'TANGENT'
    link(nt, t, 'Color', nm, 'Color')
    nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)


add("P39_NormalMapNoStrengthChange", b_normal_tangent_only)


# P40 vertex color directly into base color
def b_vertex_color_base_only(m, nt, out, bsdf):
    vc = nn(nt, 'ShaderNodeVertexColor', (-300, 0))
    vc.layer_name = 'Col'
    link(nt, vc, 'Color', bsdf, 'Base Color')


add("P40_VertexColorBaseColorOnly", b_vertex_color_base_only)


# P41 Principled added to an emission shader
def b_add_emission(m, nt, out, bsdf):
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    e = nn(nt, 'ShaderNodeEmission', (0, -200))
    e.inputs['Color'].default_value = (1.0, 0.8, 0.4, 1.0)
    e.inputs['Strength'].default_value = 4.0
    ad = nn(nt, 'ShaderNodeAddShader', (150, 0))
    nt.links.new(bsdf.outputs['BSDF'], ad.inputs[0])
    nt.links.new(e.outputs['Emission'], ad.inputs[1])
    nt.links.new(ad.outputs['Shader'], out.inputs['Surface'])


add("P41_AddShaderEmission", b_add_emission)


# P42 texture alpha multiplied by a constant
def b_alpha_factor_mul(m, nt, out, bsdf):
    t = image_node(nt, img_rgba, (-500, 200))
    link(nt, t, 'Color', bsdf, 'Base Color')
    mth = nn(nt, 'ShaderNodeMath', (-250, -100))
    mth.operation = 'MULTIPLY'
    mth.inputs[1].default_value = 0.5
    link(nt, t, 'Alpha', mth, 0)
    link(nt, mth, 'Value', bsdf, 'Alpha')


add("P42_AlphaTextureTimesConstant", b_alpha_factor_mul)


# P43 typical avatar material: base texture, alpha clip from a second image, tangent normal map
def b_avatar_typical(m, nt, out, bsdf):
    m.use_backface_culling = True
    t = image_node(nt, img_base, (-500, 200))
    uv_vector_for(nt, t, uv_name="UVMap")
    link(nt, t, 'Color', bsdf, 'Base Color')
    a = image_node(nt, img_rgba, (-500, -50))
    uv_vector_for(nt, a, uv_name="UVMap")
    mth = nn(nt, 'ShaderNodeMath', (-250, -50))
    mth.operation = 'GREATER_THAN'
    mth.inputs[1].default_value = 0.5
    link(nt, a, 'Alpha', mth, 0)
    link(nt, mth, 'Value', bsdf, 'Alpha')
    nm_img = image_node(nt, img_normal, (-500, -300))
    uv_vector_for(nt, nm_img, uv_name="UVMap")
    nm = nn(nt, 'ShaderNodeNormalMap', (-200, -300))
    nm.space = 'TANGENT'
    link(nt, nm_img, 'Color', nm, 'Color')
    nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = 0.6


add("P43_AvatarTypical", b_avatar_typical)


# P44 Principled mixed with Background (unlit in Blender)
def b_mix_bg_principled(m, nt, out, bsdf):
    bsdf.inputs['Base Color'].default_value = (0.9, 0.1, 0.1, 1.0)
    bg = nn(nt, 'ShaderNodeBackground', (0, -200))
    bg.inputs['Color'].default_value = (0.1, 0.1, 0.9, 1.0)
    mx = nn(nt, 'ShaderNodeMixShader', (150, 0))
    mx.inputs['Fac'].default_value = 0.5
    nt.links.new(bsdf.outputs['BSDF'], mx.inputs[1])
    nt.links.new(bg.outputs['Background'], mx.inputs[2])
    nt.links.new(mx.outputs['Shader'], out.inputs['Surface'])


add("P44_MixBackgroundPrincipled", b_mix_bg_principled)


# P45 P43 plus Principled emission (color (1, .5, .2) x strength 1.5: emissive strength extension)
def b_avatar_emission(m, nt, out, bsdf):
    b_avatar_typical(m, nt, out, bsdf)
    bsdf.inputs['Emission Color'].default_value = (1.0, 0.5, 0.2, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 1.5


add("P45_AvatarEmission", b_avatar_emission)


only = [name for name in os.environ.get("POTTER_PROBE_ONLY", "").split(",") if name]
known = {name for name, _ in materials}
unknown = [name for name in only if name not in known]
if unknown:
    raise RuntimeError("unknown probe names: " + ", ".join(unknown))
selected = [(name, builder) for name, builder in materials if not only or name in only]

# One UV sphere per probe, spaced along X.
for index, (name, builder) in enumerate(selected):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=(index * 2.5, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "_mesh"
    # Only P20 has a second UV layer; only P15 and P40 have a vertex color layer. Other meshes
    # keep Blender's default data, so the default sphere has no extra layers.
    if name == "P20_SecondUVLayer":
        uv2 = obj.data.uv_layers.new(name="UVLight")
        for loop in uv2.data:
            loop.uv = (loop.uv[1], loop.uv[0])
    if name in ("P15_VertexColorMultiply", "P40_VertexColorBaseColorOnly"):
        obj.data.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='CORNER')
    mat, nt, out, bsdf = new_mat(name)
    builder(mat, nt, out, bsdf)
    obj.data.materials.append(mat)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "materials.blend"))
