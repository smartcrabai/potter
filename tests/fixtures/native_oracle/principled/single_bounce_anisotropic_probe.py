"""Single-bounce Principled anisotropy probe; run with Blender 5.2.2 in Cycles CPU mode.

`a09_r00_uv_sphere` adds a smooth UV sphere to expose cross-triangle Tangent UV_MAP differences.

Capture command:
  Blender --background --factory-startup --disable-autoexec --threads 1 --python-exit-code 3 \\
    --python single_bounce_anisotropic_probe.py -- OUT_DIR
Save one case for the Potter import/render test:
  Blender --background --factory-startup --disable-autoexec --python-exit-code 3 \\
    --python single_bounce_anisotropic_probe.py -- OUT_DIR --save-only CASE BLEND
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

OUT = sys.argv[sys.argv.index('--') + 1]
os.makedirs(OUT, exist_ok=True)
THETA = math.radians(45.0)
DIST = 6.0
CAM = Vector((DIST * math.sin(THETA), 0.0, DIST * math.cos(THETA)))
LIGHT_DIST = 4.0
LIGHT = Vector((-LIGHT_DIST * math.sin(THETA), 0.0, LIGHT_DIST * math.cos(THETA)))
LIGHT_DIAMETER = 2.4
POWER = 1000.0
BASE = (0.8, 0.5, 0.3, 1.0)
SAMPLES = 4096
SEEDS = range(16)
VARIANTS = {}
for amount in (0.5, 0.9):
    for rotation in (0.0, 0.25):
        for tangent in ('radial', 'uv_map'):
            name = f'a{int(amount * 10):02d}_r{int(rotation * 100):02d}_{tangent}'
            VARIANTS[name] = dict(anisotropy=amount, rotation=rotation, tangent=tangent)
VARIANTS['isotropic'] = dict(anisotropy=0.0, rotation=0.0, tangent=None)
VARIANTS['a09_r00_unlinked'] = dict(anisotropy=0.9, rotation=0.0, tangent=None)
VARIANTS['a09_r00_uv_sphere'] = dict(
    anisotropy=0.9, rotation=0.0, tangent='uv_map', geometry='uv_sphere')
VARIANTS['uv_sphere_tangent_emission'] = dict(
    geometry='uv_sphere', sphere_segments=8, sphere_rings=8, display_tangent=True)


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def make(v, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.seed = seed
    scene.cycles.max_bounces = 4
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = False

    world = bpy.data.worlds.new('Black')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    if v.get('geometry') == 'uv_sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=v.get('sphere_segments', 16), ring_count=v.get('sphere_rings', 8), radius=1.0)
        plane = bpy.context.object
        for polygon in plane.data.polygons:
            polygon.use_smooth = True
    else:
        bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.object
    material = bpy.data.materials.new('Principled anisotropy probe')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    if v.get('display_tangent'):
        output = nodes.get('Material Output')
        for node in list(nodes):
            if node != output:
                nodes.remove(node)
        tangent = nodes.new('ShaderNodeTangent')
        tangent.direction_type = 'UV_MAP'
        tangent.uv_map = 'UVMap'
        scale = nodes.new('ShaderNodeVectorMath')
        scale.operation = 'SCALE'
        scale.inputs['Scale'].default_value = 0.5
        offset = nodes.new('ShaderNodeVectorMath')
        offset.operation = 'ADD'
        offset.inputs[1].default_value = (0.5, 0.5, 0.5)
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 1.0
        links = material.node_tree.links
        links.new(tangent.outputs['Tangent'], scale.inputs['Vector'])
        links.new(scale.outputs['Vector'], offset.inputs[0])
        links.new(offset.outputs['Vector'], emission.inputs['Color'])
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
    else:
        principled = nodes['Principled BSDF']
        principled.inputs['Base Color'].default_value = BASE
        principled.inputs['Metallic'].default_value = 1.0
        principled.inputs['Roughness'].default_value = 0.35
        principled.inputs['Anisotropic'].default_value = v['anisotropy']
        principled.inputs['Anisotropic Rotation'].default_value = v['rotation']
        if v['tangent'] is not None:
            tangent = nodes.new('ShaderNodeTangent')
            tangent.direction_type = 'RADIAL' if v['tangent'] == 'radial' else 'UV_MAP'
            if v['tangent'] == 'radial':
                tangent.axis = 'Z'
            else:
                tangent.uv_map = 'UVMap'
            material.node_tree.links.new(tangent.outputs['Tangent'], principled.inputs['Tangent'])
    plane.data.materials.append(material)

    light_data = bpy.data.lights.new('Disk', 'AREA')
    light_data.shape = 'DISK'
    light_data.size = LIGHT_DIAMETER
    light_data.energy = POWER
    light = bpy.data.objects.new('Disk', light_data)
    scene.collection.objects.link(light)
    light.location = LIGHT
    aim(light, (0.0, 0.0, 0.0))

    camera_data = bpy.data.cameras.new('Cam')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 2.4 if v.get('geometry') == 'uv_sphere' else 0.2
    camera = bpy.data.objects.new('Cam', camera_data)
    scene.collection.objects.link(camera)
    camera.location = CAM
    aim(camera, (0.0, 0.0, 0.0))
    scene.camera = camera
    return scene


def centre_mean(path, name):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels)
    acc = [0.0, 0.0, 0.0]
    count = 0
    x_range = range(width // 2, width // 2 + 4) if name == 'uv_sphere_tangent_emission' else range(width // 2 - 4, width // 2 + 4)
    for y in range(height // 2 - 4, height // 2 + 4):
        for x in x_range:
            index = (y * width + x) * channels
            for channel in range(3):
                acc[channel] += pixels[index + channel]
            count += 1
    bpy.data.images.remove(image)
    return [value / count for value in acc]


args = sys.argv[sys.argv.index('--') + 1:]
if len(args) > 1 and args[1] == '--save-only':
    name, blend_path = args[2], args[3]
    make(VARIANTS[name], 0)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print('SAVED', name, blend_path, flush=True)
    sys.exit(0)

selected = os.environ.get('ANISOTROPY_VARIANTS')
names = selected.split(',') if selected else list(VARIANTS)
results = {}
for name in names:
    values = []
    for seed in SEEDS:
        scene = make(VARIANTS[name], seed)
        path = os.path.join(OUT, f'{name}_s{seed}.exr')
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        values.append(centre_mean(path, name))
        os.remove(path)
    mean = [sum(value[channel] for value in values) / len(values) for channel in range(3)]
    stderr = [statistics.stdev(value[channel] for value in values) / math.sqrt(len(values))
              for channel in range(3)]
    results[name] = {'mean': mean, 'stderr': stderr}
    print('VARIANT', name, 'mean', mean, 'stderr', stderr, flush=True)

summary = {
    'provenance': {
        'blender': bpy.app.version_string,
        'engine': 'CYCLES',
        'device': 'CPU',
        'threads': 1,
        'samples_per_pixel': SAMPLES,
        'seeds': list(SEEDS),
        'resolution': [16, 16],
        'patch': 'central 8x8; right central 4x8 for direct tangent emission',
        'generator': 'tests/fixtures/native_oracle/principled/single_bounce_anisotropic_probe.py',
        'geometry': {
            'receiver': 'plane at origin, normal +Z, size 40, built-in UVMap',
            'curved_receiver': 'smooth UV sphere, radius 1: 16×8 Principled, 8×8 direct tangent emission diagnostic',
            'view_theta_deg': 45.0,
            'view_distance': DIST,
            'light_distance': LIGHT_DIST,
            'light_diameter': LIGHT_DIAMETER,
            'power': POWER,
            'world': 'black',
        },
        'material': {
            'shader': 'Principled BSDF; direct Tangent UV_MAP diagnostic uses Emission',
            'base_color': BASE[:3],
            'metallic': 1.0,
            'roughness': 0.35,
            'tangent': 'Radial Tangent axis Z or UV_MAP tangent from UVMap',
            'variants': 'anisotropy 0.5 and 0.9; rotation 0 and 0.25; isotropic, unlinked-Tangent, curved Principled sphere, and direct Tangent UV_MAP emission diagnostic',
        },
        'stderr': 'standard error across 16 independent Cycles seeds',
    },
    'variants': results,
}
with open(os.path.join(OUT, 'single_bounce_anisotropic_means.json'), 'w') as capture:
    json.dump(summary, capture, indent=2)
print('DONE', flush=True)
