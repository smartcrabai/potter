"""Single-bounce Principled thin-film probe: plane, disk area light, fixed view, black world.
Run: Blender -b --factory-startup --threads 1 --python single_bounce_thin_film_probe.py -- OUT_DIR
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

THETA = math.radians(45.0)
DIST = 6.0
CAM = Vector((DIST * math.sin(THETA), 0.0, DIST * math.cos(THETA)))
LIGHT_DIST = 4.0
LIGHT = Vector((-LIGHT_DIST * math.sin(THETA), 0.0, LIGHT_DIST * math.cos(THETA)))
LIGHT_DIAMETER = 2.4
POWER = 1000.0
SAMPLES = 4096
SEEDS = 16
BASE = (0.45, 0.62, 0.82, 1.0)
METAL_BASE = (0.18, 0.48, 0.76, 1.0)
VARIANTS = {
    "dielectric_no_film": dict(metallic=0.0, transmission=0.0, base=BASE,
                               thickness=0.0, film_ior=1.33),
    "dielectric_cutoff_0_1nm": dict(metallic=0.0, transmission=0.0, base=BASE,
                                    thickness=0.1, film_ior=1.33),
    "dielectric_120nm_ior133": dict(metallic=0.0, transmission=0.0, base=BASE,
                                    thickness=120.0, film_ior=1.33),
    "dielectric_450nm_ior170": dict(metallic=0.0, transmission=0.0, base=BASE,
                                    thickness=450.0, film_ior=1.7),
    "dielectric_900nm_ior133": dict(metallic=0.0, transmission=0.0, base=BASE,
                                    thickness=900.0, film_ior=1.33),
    "metallic_no_film": dict(metallic=1.0, transmission=0.0, base=METAL_BASE,
                             thickness=0.0, film_ior=1.33),
    "metallic_120nm_ior133": dict(metallic=1.0, transmission=0.0, base=METAL_BASE,
                                   thickness=120.0, film_ior=1.33),
    "metallic_450nm_ior170": dict(metallic=1.0, transmission=0.0, base=METAL_BASE,
                                   thickness=450.0, film_ior=1.7),
    "metallic_900nm_ior133": dict(metallic=1.0, transmission=0.0, base=METAL_BASE,
                                   thickness=900.0, film_ior=1.33),
    "transmission_1_thin_wall_off": dict(metallic=0.0, transmission=1.0, base=BASE,
                                         thickness=450.0, film_ior=1.7),
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
    world = bpy.data.worlds.new('Black')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    s.world = world

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new('ThinFilmProbe')
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = v['base']
    principled.inputs['Metallic'].default_value = v['metallic']
    principled.inputs['Roughness'].default_value = 0.24
    principled.inputs['IOR'].default_value = 1.5
    principled.inputs['Transmission Weight'].default_value = v['transmission']
    principled.inputs['Thin Film Thickness'].default_value = v['thickness']
    principled.inputs['Thin Film IOR'].default_value = v['film_ior']
    principled.inputs['Thin Wall'].default_value = False
    plane.data.materials.append(mat)

    light_data = bpy.data.lights.new('Disk', 'AREA')
    light_data.shape = 'DISK'
    light_data.size = LIGHT_DIAMETER
    light_data.energy = POWER
    light = bpy.data.objects.new('Disk', light_data)
    s.collection.objects.link(light)
    light.location = LIGHT
    aim(light, (0, 0, 0))
    camera_data = bpy.data.cameras.new('Cam')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 0.2
    camera = bpy.data.objects.new('Cam', camera_data)
    s.collection.objects.link(camera)
    camera.location = CAM
    aim(camera, (0, 0, 0))
    s.camera = camera
    return s

def centre_mean(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels)
    acc = [0.0, 0.0, 0.0]
    count = 0
    for y in range(height // 2 - 4, height // 2 + 4):
        for x in range(width // 2 - 4, width // 2 + 4):
            i = (y * width + x) * channels
            for c in range(3):
                acc[c] += pixels[i + c]
            count += 1
    bpy.data.images.remove(image)
    return [value / count for value in acc]

args = sys.argv[sys.argv.index('--') + 1:]
out = args[0]
os.makedirs(out, exist_ok=True)
if len(args) > 1 and args[1] == '--save-only':
    name, blend_path = args[2], args[3]
    make(VARIANTS[name], 0)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print('SAVED', name, blend_path, flush=True)
    sys.exit(0)

result = {}
for name, variant in VARIANTS.items():
    values = []
    for seed in range(SEEDS):
        scene = make(variant, seed)
        scene.render.filepath = os.path.join(out, f'{name}_s{seed}.exr')
        bpy.ops.render.render(write_still=True)
        values.append(centre_mean(scene.render.filepath))
    mean = [sum(sample[channel] for sample in values) / len(values) for channel in range(3)]
    stderr = [statistics.stdev(sample[channel] for sample in values) / math.sqrt(len(values))
              for channel in range(3)]
    result[name] = {"mean": mean, "stderr": stderr, "per_seed": values}
    print('VARIANT', name, [round(value, 8) for value in mean], flush=True)

assert result['dielectric_no_film']['per_seed'] == result['dielectric_cutoff_0_1nm']['per_seed']
capture_variants = {
    name: {key: item[key] for key in ("mean", "stderr")}
    for name, item in result.items()
}
summary = {
    "provenance": {
        "hash": "d13f752e3b9c4f8c261cda552b1021f8bcc0382c",
        "blender": "5.2.2",
        "engine": "CYCLES",
        "device": "CPU",
        "threads": 1,
        "samples_per_pixel": SAMPLES,
        "seeds": SEEDS,
        "resolution": [16, 16],
        "patch": "central 8x8",
        "generator": "tests/fixtures/native_oracle/principled/single_bounce_thin_film_probe.py",
        "geometry": {
            "receiver": "plane at origin, normal +Z",
            "view_theta_deg": 45.0,
            "view_distance": DIST,
            "light_distance": LIGHT_DIST,
            "light_diameter": LIGHT_DIAMETER,
            "power": POWER,
            "world": "black"
        },
        "material": "Principled BSDF; roughness 0.24, bulk IOR 1.5, thin wall off",
        "stderr": "standard error across 16 independent Blender seeds"
    },
    "variants": capture_variants
}
with open(os.path.join(out, 'single_bounce_thin_film_means.json'), 'w') as capture:
    json.dump(summary, capture, indent=2)
print('DONE', flush=True)
