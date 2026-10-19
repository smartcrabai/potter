"""Single-direction probe: a plane (receiver) under one disk area light, seen from a fixed view.
With a black world, the plane's pixel is exactly the single-bounce integral of the material lobe.
Run: Blender -b --factory-startup --threads 1 --python probe.py -- OUT_DIR
"""
import bpy, json, math, os, sys
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
os.makedirs(out, exist_ok=True)
THETA = math.radians(45.0)
DIST = 6.0
CAM = Vector((DIST * math.sin(THETA), 0.0, DIST * math.cos(THETA)))
LIGHT_DIST = 4.0
LIGHT = Vector((-LIGHT_DIST * math.sin(THETA), 0.0, LIGHT_DIST * math.cos(THETA)))
LIGHT_DIAMETER = 1.2
POWER = 1000.0
BASE = (0.055, 0.15, 0.24, 1.0)
SAMPLES = 4096
VARIANTS = {
    "metal_r024": dict(metallic=1.0, roughness=0.24, base=BASE),
    "metal_r024_white": dict(metallic=1.0, roughness=0.24, base=(1.0, 1.0, 1.0, 1.0)),
    "diffuse_white": dict(metallic=0.0, roughness=0.5, base=(0.8, 0.8, 0.8, 1.0), spec=0.0),
    "spec_only_r024": dict(metallic=0.0, roughness=0.24, base=(0.0, 0.0, 0.0, 1.0)),
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
    b = mat.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = v['base']
    b.inputs['Metallic'].default_value = v['metallic']
    b.inputs['Roughness'].default_value = v['roughness']
    b.inputs['IOR'].default_value = 1.5
    if 'spec' in v:
        b.inputs['Specular IOR Level'].default_value = v['spec']
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
for name, v in VARIANTS.items():
    vals = []
    for seed in range(16):
        s = make(v, seed)
        p = os.path.join(out, f'{name}_s{seed}.exr'); s.render.filepath = p
        bpy.ops.render.render(write_still=True)
        vals.append(centre_mean(p))
    m = [sum(x[c] for x in vals) / len(vals) for c in range(3)]
    result[name] = {"mean": m, "per_seed": vals}
    print('VARIANT', name, [round(x, 8) for x in m], flush=True)
json.dump({"view_theta_deg": 45.0, "dist": DIST, "light_dist": LIGHT_DIST,
           "light_diameter": LIGHT_DIAMETER, "power": POWER, "samples": SAMPLES,
           "results": result}, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
print('DONE', flush=True)
