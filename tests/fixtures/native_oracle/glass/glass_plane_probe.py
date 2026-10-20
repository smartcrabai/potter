"""Rough glass plane probe: a double-sided plane (receiver) with one glass lobe, lit by one disk area light.
"front" puts the light on the reflected side (above the plane); "back" puts it below the plane on the
refracted direction of the view ray, so the centre of the plane seen through the glass is the
single-bounce transmitted integral. Orthographic camera at 45 degrees, black world, central 8x8 patch of a
16x16 render. Cycles: CPU, 4096 samples, seeds 0..15, mean and stderr over seeds. EEVEE: one render per scene.
Run: Blender -b --factory-startup --threads 1 --python glass_plane_probe.py -- OUT_DIR
Optional: GLASS_VARIANTS=name,name... renders a subset (the committed JSON needs all variants).
Writes OUT_DIR/summary.json (committed schema) and OUT_DIR/per_seed.json (Cycles per-seed means, not committed).
"""
import bpy, json, math, os, sys
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
os.makedirs(out, exist_ok=True)

THETA = math.radians(45.0)
S, C = math.sin(THETA), math.cos(THETA)
DIST = 6.0
CAM = Vector((DIST * S, 0.0, DIST * C))
LIGHT_DIST = 4.0
# Blender's disk light `size` is the diameter; Cycles uses radius = size / 2 (intern/cycles/kernel/light/area.h).
LIGHT_SIZE = 2.4
POWER = 1000.0
SAMPLES = 4096
SEEDS = list(range(16))
RES = 16
PATCH = range(RES // 2 - 4, RES // 2 + 4)  # rows and columns 4..11: central 8x8
SIDES = ('front', 'back')

VARIANTS = {
    "P_r0_ior150": dict(node="principled", roughness=0.0, ior=1.5, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r0_ior145": dict(node="principled", roughness=0.0, ior=1.45, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r02_ior150": dict(node="principled", roughness=0.2, ior=1.5, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r02_ior145": dict(node="principled", roughness=0.2, ior=1.45, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r05_ior150": dict(node="principled", roughness=0.5, ior=1.5, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r05_ior145": dict(node="principled", roughness=0.5, ior=1.45, base=(1.0, 1.0, 1.0, 1.0)),
    "P_r02_ior150_tint": dict(node="principled", roughness=0.2, ior=1.5, base=(0.8, 0.5, 0.3, 1.0)),
    "G_r02_ior150_tint": dict(node="glass", roughness=0.2, ior=1.5, base=(0.8, 0.5, 0.3, 1.0)),
}


def front_centre():
    # Mirror direction of the view vector (S, 0, C) about the +Z normal.
    return Vector((-LIGHT_DIST * S, 0.0, LIGHT_DIST * C))


def back_centre(ior):
    # Snell from air into glass: sin(theta_t) = sin(45) / ior. The refracted continuation of the
    # incoming ray (-view) points down: t = (-sin_t, 0, -cos_t).
    sin_t = S / ior
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    t = Vector((-sin_t, 0.0, -cos_t))
    return t * LIGHT_DIST


def light_centre(v, side):
    return front_centre() if side == 'front' else back_centre(v['ior'])


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def make(name, v, side, seed, engine):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = engine
    if engine == 'CYCLES':
        s.cycles.device = 'CPU'
        s.cycles.samples = SAMPLES
        s.cycles.use_denoising = False
        s.cycles.seed = seed
        s.cycles.max_bounces = 4
    s.render.resolution_x = RES
    s.render.resolution_y = RES
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
    mat = bpy.data.materials.new('Glass_' + name); mat.use_nodes = True
    nt = mat.node_tree
    if v['node'] == 'principled':
        b = nt.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = v['base']
        b.inputs['Roughness'].default_value = v['roughness']
        b.inputs['IOR'].default_value = v['ior']
        b.inputs['Transmission Weight'].default_value = 1.0
    else:
        for n in list(nt.nodes):
            if n.type == 'BSDF_PRINCIPLED':
                nt.nodes.remove(n)
        g = nt.nodes.new('ShaderNodeBsdfGlass')
        g.inputs['Color'].default_value = v['base']
        g.inputs['Roughness'].default_value = v['roughness']
        g.inputs['IOR'].default_value = v['ior']
        nt.links.new(g.outputs['BSDF'], nt.nodes['Material Output'].inputs['Surface'])
    plane.data.materials.append(mat)

    ld = bpy.data.lights.new('Disk', 'AREA')
    ld.shape = 'DISK'
    ld.size = LIGHT_SIZE
    ld.energy = POWER
    light = bpy.data.objects.new('Disk', ld)
    s.collection.objects.link(light)
    light.location = light_centre(v, side)
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
            for c in range(3):
                acc[c] += px[i + c]
            n += 1
    bpy.data.images.remove(img)
    return [a / n for a in acc]


def render_to(s, path):
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


def stats(vals):
    n = len(vals)
    mean = [sum(x[c] for x in vals) / n for c in range(3)]
    sd = [math.sqrt(sum((x[c] - mean[c]) ** 2 for x in vals) / (n - 1)) for c in range(3)]
    return mean, [sd[c] / math.sqrt(n) for c in range(3)]


def r8(x):
    if isinstance(x, (list, tuple)):
        return [r8(e) for e in x]
    if isinstance(x, dict):
        return {k: r8(e) for k, e in x.items()}
    if isinstance(x, float):
        return float(f"{x:.8g}")
    return x


# Sanity: the centres must place the back light below the plane in the -x half-space, the front light above.
for ior in (1.5, 1.45):
    b = back_centre(ior)
    assert b.z < 0 and b.x < 0, b
    print('BACK_CENTRE ior', ior, [round(x, 8) for x in b], 'angle_from_-z_deg',
          round(math.degrees(math.acos(-b.z / b.length)), 6), flush=True)
f = front_centre()
assert f.z > 0 and f.x < 0, f
print('FRONT_CENTRE', [round(x, 8) for x in f], flush=True)

# EEVEE identifier in this Blender build.
eevee_items = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
EEVEE_ID = next(e for e in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE') if e in eevee_items)
print('ENGINES', eevee_items, 'EEVEE_ID', EEVEE_ID, flush=True)

selected = os.environ.get('GLASS_VARIANTS')
names = selected.split(',') if selected else list(VARIANTS)

cycles = {}
per_seed = {}
for name in names:
    v = VARIANTS[name]
    for side in SIDES:
        vals = []
        for seed in SEEDS:
            s = make(name, v, side, seed, 'CYCLES')
            assert s.render.engine == 'CYCLES'
            p = os.path.join(out, f'{name}_{side}_s{seed}.exr')
            render_to(s, p)
            vals.append(centre_mean(p))
        mean, stderr = stats(vals)
        cycles[(name, side)] = dict(mean=mean, stderr=stderr)
        per_seed[f'{name}/{side}'] = vals
        print('CYCLES', name, side, [round(x, 8) for x in mean], 'stderr', [round(x, 8) for x in stderr], flush=True)

json.dump(r8(per_seed), open(os.path.join(out, 'per_seed.json'), 'w'), indent=1)

eevee = {}
for name in names:
    v = VARIANTS[name]
    for side in SIDES:
        try:
            s = make(name, v, side, 0, EEVEE_ID)
            assert s.render.engine == EEVEE_ID
            p = os.path.join(out, f'eevee_{name}_{side}.exr')
            render_to(s, p)
            if not os.path.exists(p):
                raise RuntimeError('EEVEE produced no output file')
            eevee[(name, side)] = {"mean": centre_mean(p)}
        except Exception as e:
            eevee[(name, side)] = {"error": f"{type(e).__name__}: {e}"}
        print('EEVEE', name, side, eevee[(name, side)], flush=True)

results = {}
for name in names:
    v = VARIANTS[name]
    results[name] = {}
    for side in SIDES:
        c = cycles[(name, side)]
        results[name][side] = {
            "cycles": {"mean": c['mean'], "stderr": c['stderr']},
            "eevee": eevee[(name, side)],
            "ior": v['ior'],
            "roughness": v['roughness'],
            "base": list(v['base']),
            "node": 'principled' if v['node'] == 'principled' else 'glass',
        }

summary = {
    "provenance": {
        "blender": bpy.app.version_string,
        "engine": {"cycles": "CYCLES, device CPU, threads 1, max_bounces 4, denoising off",
                   "eevee": EEVEE_ID, "eevee_samples": "default"},
        "samples": SAMPLES,
        "seeds": SEEDS,
        "resolution": [RES, RES],
        "patch": "central 8x8 pixels (rows and columns 4..11 of 16x16), per-channel mean",
        "stderr": "sample std (ddof=1) of the 16 per-seed patch means / sqrt(16)",
        "view": {"transform": "Standard", "look": "None", "exposure": 0.0, "gamma": 1.0,
                 "film_transparent": False, "file": "OPEN_EXR 32-bit"},
        "geometry": {"receiver": "plane size 40 at z=0, normal +Z, double-sided (default)",
                     "camera": "ORTHO ortho_scale 0.2 at 6 units along (sin45, 0, cos45), aimed at origin"},
        "light": {"type": "AREA DISK", "size": LIGHT_SIZE,
                  "radius": LIGHT_SIZE / 2, "power_W": POWER, "aim": "-Z toward origin",
                  "front_centre": "4 * (-sin45, 0, cos45): mirror direction of the view",
                  "back_centre": "4 * t, t = (-sin(th_t), 0, -cos(th_t)), sin(th_t) = sin45 / ior "
                                 "(Snell from air into glass; t is the refracted continuation of -view)"},
        "variant_notes": {
            "principled": "Principled BSDF, Transmission Weight 1.0; only Base Color, Roughness, IOR set; "
                          "Cycles uses sqrt(base) as the transmission tint",
            "glass": "Glass BSDF node -> Material Output; distribution Multiscatter GGX (default); "
                     "transmission tinted by Color directly (no sqrt)",
        },
        "generator": "tests/fixtures/native_oracle/glass/glass_plane_probe.py",
        "run": "Blender -b --factory-startup --threads 1 --python glass_plane_probe.py -- OUT_DIR",
    },
    "light_centres": {
        f"ior_{ior}": {"front": list(front_centre()), "back": list(back_centre(ior))}
        for ior in (1.5, 1.45)
    },
    "results": results,
}
json.dump(r8(summary), open(os.path.join(out, 'summary.json'), 'w'), indent=2)
print('DONE', flush=True)
