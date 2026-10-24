"""Cycles smooth Thin Wall indirect-light oracle for Blender 5.2.2 CPU.

Run: Blender -b --factory-startup --threads 1 --python smooth_thin_wall_indirect.py -- OUT_DIR
Save fixture: Blender -b --factory-startup --threads 1 --python smooth_thin_wall_indirect.py -- OUT_DIR --save-only BLEND_PATH
"""
import bpy, json, math, os, statistics, sys
from mathutils import Vector

SAMPLES = 4096
SEEDS = 16
RESOLUTION = 16
PATCH = 8


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def make(seed):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = seed
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("BlackWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "DiffuseFloor"
    floor_material = bpy.data.materials.new("DiffuseFloor")
    floor_material.use_nodes = True
    floor_bsdf = floor_material.node_tree.nodes["Principled BSDF"]
    floor_bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    floor_bsdf.inputs["Roughness"].default_value = 1.0
    floor.data.materials.append(floor_material)

    # The panel is behind the floor patch from the camera, but between that patch and the lamp.
    panel_mesh = bpy.data.meshes.new("SmoothThinWallPaneMesh")
    panel_mesh.from_pydata(
        [(-1.5, 1.5, 0.05), (1.5, 1.5, 0.05), (1.5, 1.5, 2.8), (-1.5, 1.5, 2.8)],
        [],
        [(0, 1, 2, 3)],
    )
    panel_mesh.update()
    panel = bpy.data.objects.new("SmoothThinWallPane", panel_mesh)
    scene.collection.objects.link(panel)
    pane_material = bpy.data.materials.new("SmoothThinWall")
    pane_material.use_nodes = True
    pane_bsdf = pane_material.node_tree.nodes["Principled BSDF"]
    pane_bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    pane_bsdf.inputs["Roughness"].default_value = 0.0
    pane_bsdf.inputs["IOR"].default_value = 1.45
    pane_bsdf.inputs["Thin Wall"].default_value = True
    pane_bsdf.inputs["Transmission Weight"].default_value = 1.0
    panel.data.materials.append(pane_material)

    light_data = bpy.data.lights.new("SmallBacklight", "AREA")
    light_data.shape = "DISK"
    light_data.size = 0.25
    light_data.energy = 1200.0
    light = bpy.data.objects.new("SmallBacklight", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 4.0, 2.8)
    aim(light, (0.0, 0.0, 0.0))

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.6
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, -6.0, 5.0)
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
    blend_path = args[2]
    os.makedirs(os.path.dirname(os.path.abspath(blend_path)), exist_ok=True)
    make(0)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print("SAVED", blend_path, flush=True)
    sys.exit(0)

values = []
for seed in range(SEEDS):
    scene = make(seed)
    scene.render.filepath = os.path.join(out, f"smooth_thin_wall_indirect_s{seed}")
    bpy.ops.render.render(write_still=True)
    values.append(centre_mean(scene.render.filepath + ".exr"))
mean = [sum(sample[channel] for sample in values) / len(values) for channel in range(3)]
stderr = [statistics.stdev(sample[channel] for sample in values) / math.sqrt(len(values))
          for channel in range(3)]
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
        "generator": "tests/fixtures/native_oracle/principled/smooth_thin_wall_indirect.py",
        "geometry": "diffuse floor, smooth transmissive Thin Wall panel between floor and small area light; camera sees only floor",
        "stderr": "standard error across 16 independent Blender seeds",
    },
    "mean": mean,
    "stderr": stderr,
}
with open(os.path.join(out, "smooth_thin_wall_indirect.json"), "w") as output:
    json.dump(capture, output, indent=2)
print("CAPTURE", mean, "stderr", stderr, flush=True)
