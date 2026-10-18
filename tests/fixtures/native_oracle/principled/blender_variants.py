# Oracle generator for tests/native_principled_lobes.rs (Principled BSDF lobes vs Cycles).
# Run with Blender 5.2.2 headless, --threads 1 (LTS build d13f752e3b9c), 8 seeds (0..7), 512 spp,
# 128x128 frame, central 12x12 pixel patch of a radius-1 UV sphere under the area/world setup
# built in make_scene(). The per-variant output is distilled into blender_variant_means.json, the
# committed oracle. The generated EXRs are not committed.
#
"""Blender 5.2.2 Cycles Principled decomposition probe (W-bpy, --threads 1).

Sphere in the realistic scene's world/area setup, rendered for named material variants so that
the diffuse, dielectric-specular and metal contributions can be separated by Blender itself.
Patch: central 12x12 of a 128x128 linear OpenEXR. Eight seeds, 512 samples.
Run: Blender -b --factory-startup --threads 1 --python variants.py -- OUT_DIR
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

out = sys.argv[sys.argv.index('--') + 1]
os.makedirs(out, exist_ok=True)
W = H = 128
SAMPLES = 512
SEEDS = range(8)
PATCH = 6
WORLD = 0.05087608844041824
LIGHTS = [
    ("Key_Softbox", (0.0, -7.0, 13.0), 1800.0, 9.0, (0.80, 0.88, 1.0)),
    ("Warm_Rim", (7.0, 8.0, 10.0), 1300.0, 7.0, (1.0, 0.58, 0.31)),
    ("Cool_Fill", (-8.0, 1.0, 6.0), 900.0, 6.0, (0.34, 0.58, 1.0)),
]
BASE = (0.055, 0.15, 0.24, 1.0)
# name, lighting, base colour, metallic, specular IOR level
VARIANTS = [
    ("area_default", "area", BASE, 0.0, 0.5),
    ("area_spec0_pure_diffuse", "area", BASE, 0.0, 0.0),
    ("area_spec_only_black_base", "area", (0.0, 0.0, 0.0, 1.0), 0.0, 0.5),
    ("area_metal024", "area", BASE, 0.24, 0.5),
    ("area_metal1", "area", BASE, 1.0, 0.5),
    ("world_default", "world", BASE, 0.0, 0.5),
    ("world_spec0_pure_diffuse", "world", BASE, 0.0, 0.0),
    ("world_spec_only_black_base", "world", (0.0, 0.0, 0.0, 1.0), 0.0, 0.5),
    ("world_metal024", "world", BASE, 0.24, 0.5),
    ("world_metal1", "world", BASE, 1.0, 0.5),
]

def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()

def make_scene(lighting, base, metallic, spec_level, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.seed = seed
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 12
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = False

    world = bpy.data.worlds.new('ProbeWorld')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (WORLD, WORLD, WORLD, 1.0)
    background.inputs['Strength'].default_value = 1.0 if lighting == 'world' else 0.0
    scene.world = world

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0, location=(0, 0, 0))
    sphere = bpy.context.object
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new('Ceramic')
    material.use_nodes = True
    bsdf = material.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = base
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = 0.24
    bsdf.inputs['IOR'].default_value = 1.5
    bsdf.inputs['Specular IOR Level'].default_value = spec_level
    sphere.data.materials.append(material)

    if lighting == 'area':
        for name, location, energy, size, color in LIGHTS:
            data = bpy.data.lights.new(name, 'AREA')
            data.shape = 'DISK'
            data.size = size
            data.energy = energy
            data.color = color
            light = bpy.data.objects.new(name, data)
            scene.collection.objects.link(light)
            light.location = location
            aim(light, (0.0, 0.0, 0.0))

    camera_data = bpy.data.cameras.new('ProbeCamera')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 3.5
    camera = bpy.data.objects.new('ProbeCamera', camera_data)
    scene.collection.objects.link(camera)
    camera.location = (3.0, -5.0, 2.5)
    aim(camera, (0.0, 0.0, 0.0))
    scene.camera = camera
    return scene

def patch_mean(path):
    image = bpy.data.images.load(path, check_existing=False)
    w, h = image.size
    channels = image.channels
    pixels = list(image.pixels)
    sums = [0.0, 0.0, 0.0]
    count = 0
    for y in range(h // 2 - PATCH, h // 2 + PATCH):
        for x in range(w // 2 - PATCH, w // 2 + PATCH):
            i = (y * w + x) * channels
            for c in range(3):
                sums[c] += pixels[i + c]
            count += 1
    bpy.data.images.remove(image)
    return [s / count for s in sums]

summary = {}
for name, lighting, base, metallic, spec_level in VARIANTS:
    rows = []
    for seed in SEEDS:
        scene = make_scene(lighting, base, metallic, spec_level, seed)
        path = os.path.join(out, f'{name}_s{seed}.exr')
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        rows.append(patch_mean(path))
    mean = [statistics.mean(r[c] for r in rows) for c in range(3)]
    stderr = [statistics.stdev(r[c] for r in rows) / math.sqrt(len(rows)) for c in range(3)]
    summary[name] = {'mean': mean, 'stderr': stderr}
    print('VARIANT', name, 'mean', [round(x, 7) for x in mean], 'stderr', [round(x, 8) for x in stderr], flush=True)
with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print('DONE')
