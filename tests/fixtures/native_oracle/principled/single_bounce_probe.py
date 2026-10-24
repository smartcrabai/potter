"""Single-direction probe: a plane (receiver) under one disk area light, seen from a fixed view.
With a black world, the plane's pixel is exactly the single-bounce integral of the material lobe.
Run: Blender -b --factory-startup --threads 1 --python probe.py -- OUT_DIR
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
os.makedirs(out, exist_ok=True)
THETA = math.radians(45.0)
DIST = 6.0
CAM = Vector((DIST * math.sin(THETA), 0.0, DIST * math.cos(THETA)))
LIGHT_DIST = 4.0
LIGHT = Vector((-LIGHT_DIST * math.sin(THETA), 0.0, LIGHT_DIST * math.cos(THETA)))
LIGHT_DIAMETER = float(os.environ.get("PROBE_DIAMETER", "1.2"))
POWER = 1000.0
BASE = (0.055, 0.15, 0.24, 1.0)
SAMPLES = 4096
VARIANTS = {
    "metal_r024": dict(metallic=1.0, roughness=0.24, base=BASE),
    "metal_r024_white": dict(metallic=1.0, roughness=0.24, base=(1.0, 1.0, 1.0, 1.0)),
    "diffuse_white": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.8, 0.8, 1.0), spec=0.0),
    "diffuse_roughness_r000": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.5, 0.3, 1.0), spec=0.0,
                                   diffuse_roughness=0.0),
    "diffuse_roughness_r050": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.5, 0.3, 1.0), spec=0.0,
                                   diffuse_roughness=0.5),
    "diffuse_roughness_r100": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.5, 0.3, 1.0), spec=0.0,
                                   diffuse_roughness=1.0),
    "diffuse_bsdf_r050": dict(node="diffuse", roughness=0.5, metallic=0.0,
                              base=(0.8, 0.5, 0.3, 1.0), diffuse_roughness=0.5),
    "spec_only_r024": dict(metallic=0.0, roughness=0.24, base=(0.0, 0.0, 0.0, 1.0)),
    "dielectric_white": dict(metallic=0.0, roughness=0.24, base=(0.8, 0.8, 0.8, 1.0)),
    # Coat alone: no base diffuse and no base dielectric layer (Specular IOR Level 0).
    "coat_r024": dict(metallic=0.0, roughness=0.24, base=(0.0, 0.0, 0.0, 1.0), spec=0.0,
                      coat=1.0, coat_roughness=0.24, coat_ior=1.5, coat_tint=(1.0, 1.0, 1.0)),
    # Coat over a white diffuse base with a tinted coat: exercises the layering and the tint.
    "coat_tint_diffuse": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.8, 0.8, 1.0),
                              coat=0.5, coat_roughness=0.1, coat_ior=1.5,
                              coat_tint=(1.0, 0.5, 0.25)),
    # Sheen alone: no base diffuse and no specular layer (Specular IOR Level 0 gives eta 1).
    "sheen_r050": dict(metallic=0.0, roughness=0.24, base=(0.0, 0.0, 0.0, 1.0), spec=0.0,
                       sheen=1.0, sheen_roughness=0.5, sheen_tint=(1.0, 1.0, 1.0)),
    "sheen_r030_tint": dict(metallic=0.0, roughness=0.24, base=(0.0, 0.0, 0.0, 1.0), spec=0.0,
                            sheen=0.6, sheen_roughness=0.3, sheen_tint=(1.0, 0.5, 0.25)),
    # Sheen over a white diffuse base with the default specular layer, so the sheen's layering of
    # the diffuse and the dielectric lobes is pinned.
    "sheen_r050_base08": dict(metallic=0.0, roughness=0.24, base=(0.8, 0.8, 0.8, 1.0),
                              sheen=1.0, sheen_roughness=0.5, sheen_tint=(1.0, 1.0, 1.0)),
}

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

def make(v, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = SAMPLES
    s.cycles.use_denoising = False
    s.cycles.seed = seed
    s.cycles.max_bounces = 4
    s.render.resolution_x = 16
    s.render.resolution_y = 16
    s.render.resolution_percentage = 100
    s.view_settings.view_transform = 'Standard'
    s.view_settings.look = 'None'
    s.view_settings.exposure = 0.0
    s.view_settings.gamma = 1.0
    s.render.image_settings.file_format = 'OPEN_EXR'
    s.render.image_settings.color_depth = '32'
    s.render.film_transparent = False
    w = bpy.data.worlds.new('Black'); w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    s.world = w
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new('Lobe'); mat.use_nodes = True
    nodes = mat.node_tree.nodes
    p = nodes['Principled BSDF']
    output = nodes['Material Output']
    if v.get('node') == 'diffuse':
        nodes.remove(p)
        b = nodes.new('ShaderNodeBsdfDiffuse')
        b.inputs['Color'].default_value = v['base']
        b.inputs['Roughness'].default_value = v['diffuse_roughness']
        mat.node_tree.links.new(b.outputs['BSDF'], output.inputs['Surface'])
    else:
        b = p
        b.inputs['Base Color'].default_value = v['base']
        b.inputs['Metallic'].default_value = v['metallic']
        b.inputs['Roughness'].default_value = v['roughness']
        b.inputs['IOR'].default_value = 1.5
        if 'spec' in v:
            b.inputs['Specular IOR Level'].default_value = v['spec']
        if 'diffuse_roughness' in v:
            b.inputs['Diffuse Roughness'].default_value = v['diffuse_roughness']
        if 'coat' in v:
            b.inputs['Coat Weight'].default_value = v['coat']
            b.inputs['Coat Roughness'].default_value = v['coat_roughness']
            b.inputs['Coat IOR'].default_value = v['coat_ior']
            b.inputs['Coat Tint'].default_value = (*v['coat_tint'], 1.0)
        if 'sheen' in v:
            b.inputs['Sheen Weight'].default_value = v['sheen']
            b.inputs['Sheen Roughness'].default_value = v['sheen_roughness']
            b.inputs['Sheen Tint'].default_value = (*v['sheen_tint'], 1.0)
    plane.data.materials.append(mat)
    ld = bpy.data.lights.new('Disk', 'AREA')
    ld.shape = 'DISK'
    ld.size = LIGHT_DIAMETER
    ld.energy = POWER
    light = bpy.data.objects.new('Disk', ld)
    s.collection.objects.link(light)
    light.location = LIGHT
    aim(light, (0, 0, 0))
    cd = bpy.data.cameras.new('Cam'); cd.type = 'ORTHO'; cd.ortho_scale = 0.2
    cam = bpy.data.objects.new('Cam', cd)
    s.collection.objects.link(cam)
    cam.location = CAM
    aim(cam, (0, 0, 0))
    s.camera = cam
    return s

def centre_mean(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size; ch = img.channels; px = list(img.pixels)
    acc = [0.0, 0.0, 0.0]; n = 0
    for y in range(h // 2 - 4, h // 2 + 4):
        for x in range(w // 2 - 4, w // 2 + 4):
            i = (y * w + x) * ch
            for c in range(3): acc[c] += px[i + c]
            n += 1
    bpy.data.images.remove(img)
    return [a / n for a in acc]

result = {}
# PROBE_VARIANTS=name,name... renders a subset; committed means are never regenerated in place.
args = sys.argv[sys.argv.index('--') + 1:]
out = args[0]
os.makedirs(out, exist_ok=True)
if len(args) > 1 and args[1] == '--save-only':
    name, diameter, blend_path = args[2], float(args[3]), args[4]
    LIGHT_DIAMETER = diameter
    make(VARIANTS[name], 0)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print('SAVED', name, blend_path, flush=True)
    sys.exit(0)
selected = os.environ.get('PROBE_VARIANTS')
for name, v in ((n, VARIANTS[n]) for n in (selected.split(',') if selected else VARIANTS)):
    vals = []
    for seed in range(16):
        s = make(v, seed)
        p = os.path.join(out, f'{name}_s{seed}.exr'); s.render.filepath = p
        bpy.ops.render.render(write_still=True)
        vals.append(centre_mean(p))
    m = [sum(x[c] for x in vals) / len(vals) for c in range(3)]
    stderr = [
        statistics.stdev(x[c] for x in vals) / math.sqrt(len(vals))
        for c in range(3)
    ]
    result[name] = {"mean": m, "stderr": stderr, "per_seed": vals}
    print('VARIANT', name, [round(x, 8) for x in m], flush=True)
json.dump({"view_theta_deg": 45.0, "dist": DIST, "light_dist": LIGHT_DIST,
           "light_diameter": LIGHT_DIAMETER, "power": POWER, "samples": SAMPLES,
           "results": result}, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
print('DONE', flush=True)
