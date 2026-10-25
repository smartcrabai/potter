"""Single-bounce Principled Thin Wall probe for Blender 5.2.2 Cycles CPU.
Run: Blender -b --factory-startup --threads 1 --python single_bounce_thin_wall_probe.py -- OUT_DIR
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

SAMPLES = 2048
SEEDS = 16
RESOLUTION = 16
PATCH = 8
VARIANTS = {
    "transmission_pane_thin_wall": dict(kind="pane", thin_wall=True),
    "translucent_leaf_thin_wall": dict(kind="leaf", thin_wall=True),
    "transmission_pane_thin_wall_off": dict(kind="pane", thin_wall=False),
    "translucent_leaf_thin_wall_light_no_diffuse": dict(
        kind="leaf", thin_wall=True, visible_diffuse=False
    ),
    "translucent_leaf_thin_wall_light_no_transmission": dict(
        kind="leaf", thin_wall=True, visible_transmission=False
    ),
}


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def make(variant, seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.seed = seed
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("BlackWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world

    if variant["kind"] == "pane":
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
        surface = bpy.context.object
        surface.name = "ThinWallPane"
        base = (0.78, 0.9, 1.0, 1.0)
    else:
        count = 32
        vertices = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            vertices.append((1.05 * math.cos(angle), 0.48 * math.sin(angle), 0.0))
        mesh = bpy.data.meshes.new("ThinWallLeafMesh")
        mesh.from_pydata(vertices, [], [list(range(count))])
        mesh.update()
        surface = bpy.data.objects.new("ThinWallLeaf", mesh)
        scene.collection.objects.link(surface)
        base = (0.08, 0.42, 0.12, 1.0)

    material = bpy.data.materials.new("PrincipledThinWall")
    material.use_nodes = True
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = base
    principled.inputs["Roughness"].default_value = 0.18 if variant["kind"] == "pane" else 0.65
    principled.inputs["IOR"].default_value = 1.45
    principled.inputs["Thin Wall"].default_value = variant["thin_wall"]
    if variant["kind"] == "pane":
        principled.inputs["Transmission Weight"].default_value = 1.0
    else:
        principled.inputs["Subsurface Weight"].default_value = 0.85
        principled.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.2)
        principled.inputs["Subsurface Scale"].default_value = 0.16
    surface.data.materials.append(material)

    light_data = bpy.data.lights.new("Backlight", "AREA")
    light_data.shape = "DISK"
    light_data.size = 1.0
    light_data.energy = 100.0
    light = bpy.data.objects.new("Backlight", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, -1.5)
    aim(light, (0.0, 0.0, 0.0))
    if "visible_diffuse" in variant:
        light.visible_diffuse = variant["visible_diffuse"]
    if "visible_transmission" in variant:
        light.visible_transmission = variant["visible_transmission"]

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.4
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 5.0)
    aim(camera, (0.0, 0.0, 0.0))
    scene.camera = camera
    return scene


def centre_mean(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels)
    start = (RESOLUTION - PATCH) // 2
    sums = [0.0, 0.0, 0.0]
    for y in range(start, start + PATCH):
        for x in range(start, start + PATCH):
            offset = (y * width + x) * channels
            for channel in range(3):
                sums[channel] += pixels[offset + channel]
    bpy.data.images.remove(image)
    return [total / (PATCH * PATCH) for total in sums]


args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
os.makedirs(out, exist_ok=True)
if len(args) > 1 and args[1] == "--save-only":
    name, blend_path = args[2], args[3]
    os.makedirs(os.path.dirname(os.path.abspath(blend_path)), exist_ok=True)
    make(VARIANTS[name], 0)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print("SAVED", name, blend_path, flush=True)
    sys.exit(0)

results = {}
for name, variant in VARIANTS.items():
    values = []
    for seed in range(SEEDS):
        scene = make(variant, seed)
        scene.render.filepath = os.path.join(out, f"{name}_s{seed}")
        bpy.ops.render.render(write_still=True)
        values.append(centre_mean(scene.render.filepath + ".exr"))
    mean = [sum(sample[channel] for sample in values) / len(values) for channel in range(3)]
    stderr = [statistics.stdev(sample[channel] for sample in values) / math.sqrt(len(values))
              for channel in range(3)]
    results[name] = {"mean": mean, "stderr": stderr, "per_seed": values}
    print("VARIANT", name, "mean", mean, "stderr", stderr, flush=True)

capture = {
    "provenance": {
        "blender": "5.2.2 LTS",
        "engine": "CYCLES",
        "device": "CPU",
        "threads": 1,
        "samples_per_pixel": SAMPLES,
        "seeds": list(range(SEEDS)),
        "resolution": [RESOLUTION, RESOLUTION],
        "patch": "central 8x8 pixel mean",
        "generator": "tests/fixtures/native_oracle/principled/single_bounce_thin_wall_probe.py",
        "geometry": "plane or flat leaf; centered camera and area backlight",
        "stderr": "standard error across 16 independent Blender seeds",
    },
    "variants": results,
}
with open(os.path.join(out, "single_bounce_thin_wall_means.json"), "w") as output:
    json.dump(capture, output, indent=2)
print("DONE", flush=True)
