"""EEVEE oracle for the realtime light shapes beyond the area and point lights (Blender 5.2.2).

Two scenes, each on a black world, rendered by EEVEE and saved as `.blend` for Potter to import:

- Sun: a perspective camera at the origin looks up +Z, and a sun with `visible_camera` on shines down -Z, so its
  disc sits at the image centre. The disc has the sun's angular diameter (`SUN_ANGLE`), which EEVEE takes as
  `shape_radius = tan(angle / 2)` (`eevee_light.cc`). Its radiance is `energy · (1 + 1 / r²) / π`
  (`shape_display_light_radiance_get`).
- Spot and absorber: the orthographic top view of `eevee_light_shapes.py` with a spot light that faces the camera at
  (0, 2.5) m, and an absorbing slab, 2 m thick, over the area light. EEVEE multiplies each light shape by the
  transmittance in front of it (`eevee_light_shape_display.bsl.hh`, `vol.transmittance`), so the area light is
  dimmed by the slab. The spot's radiance on axis is the sphere-light formula `P / (4 π² r²)`.

Rows are reported top-down, as Potter's images are indexed.

Arguments after `--`: sun_blend, spot_blend, absorber_blend, json_out, sun_exr, spot_exr, absorber_exr.
"""

import json
import math
import sys

import bpy
import numpy as np

RESOLUTION = 64
WINDOW = 2
SAMPLES = 64

# Sun.
SUN_ANGLE = math.radians(8.0)
SUN_ENERGY = 0.01

# Spot and absorber.
ORTHO_SCALE = 8.0
AREA_SIZE = 2.0
AREA_CENTRE_X = -1.5
AREA_ENERGY = 10.0
POINT_CENTRE_X = 1.5
POINT_RADIUS = 1.0
POINT_ENERGY = 5.0
SPOT_CENTRE_Y = 2.5
SPOT_RADIUS = 0.5
SPOT_ENERGY = 5.0
SLAB_DENSITY = 0.5
SLAB_THICKNESS = 2.0

sun_blend, spot_blend, absorber_blend, json_out, sun_exr, spot_exr, absorber_exr = sys.argv[
    sys.argv.index("--") + 1 :
][:7]


def new_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Black")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    view = scene.view_settings
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    return scene


def add_camera(scene, camera_type, location, rotation):
    camera_data = bpy.data.cameras.new("OracleCamera")
    camera_data.type = camera_type
    if camera_type == "ORTHO":
        camera_data.ortho_scale = ORTHO_SCALE
    camera = bpy.data.objects.new("OracleCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = location
    camera.rotation_euler = rotation
    scene.camera = camera


def add_light(scene, name, light_type, location, rotation, visible=True):
    data = bpy.data.lights.new(name, light_type)
    data.color = (1.0, 1.0, 1.0)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    obj.visible_camera = visible
    return data


def render_and_read(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)[:, :, 0]
    bpy.data.images.remove(image)
    return pixels


def window_mean(pixels_bottom_up, column, row_top_down):
    row_bottom = RESOLUTION - 1 - row_top_down
    block = pixels_bottom_up[
        row_bottom - WINDOW:row_bottom + WINDOW + 1,
        column - WINDOW:column + WINDOW + 1,
    ]
    return float(block.mean())


result = {"resolution": RESOLUTION, "window": WINDOW}

# Sun scene.
scene = new_scene()
add_camera(scene, "PERSP", (0.0, 0.0, 0.0), (math.pi, 0.0, 0.0))
add_light(scene, "Sun", "SUN", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
sun = bpy.data.objects["Sun"].data
sun.angle = SUN_ANGLE
sun.energy = SUN_ENERGY
bpy.ops.wm.save_as_mainfile(filepath=sun_blend)
sun_pixels = render_and_read(scene, sun_exr)
radius = math.tan(SUN_ANGLE / 2.0)
result["sun_formula_radiance"] = SUN_ENERGY * (1.0 + 1.0 / radius**2) / math.pi
result["sun_centre_mean_red"] = window_mean(sun_pixels, RESOLUTION // 2, RESOLUTION // 2)
result["sun_corner_mean_red"] = window_mean(sun_pixels, RESOLUTION // 8, RESOLUTION // 8)

# Spot scene, without an absorber: EEVEE's volume grid is coarse and bleeds an absorber's edge onto nearby pixels, so
# the spot is measured where no volume is.
scene = new_scene()
add_camera(scene, "ORTHO", (0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
area = add_light(scene, "AreaLight", "AREA", (AREA_CENTRE_X, 0.0, 0.0), (math.pi, 0.0, 0.0))
area.shape = "SQUARE"
area.size = AREA_SIZE
area.energy = AREA_ENERGY
point = add_light(scene, "PointLight", "POINT", (POINT_CENTRE_X, 0.0, 0.0), (0.0, 0.0, 0.0))
point.shadow_soft_size = POINT_RADIUS
point.energy = POINT_ENERGY
spot = add_light(scene, "SpotLight", "SPOT", (0.0, SPOT_CENTRE_Y, 0.0), (math.pi, 0.0, 0.0))
spot.shadow_soft_size = SPOT_RADIUS
spot.energy = SPOT_ENERGY
spot.spot_size = math.radians(60.0)
spot.spot_blend = 0.15
bpy.ops.wm.save_as_mainfile(filepath=spot_blend)
spot_pixels = render_and_read(scene, spot_exr)
# Blender's image rows run bottom-up; the spot is at +2.5 m in Y, so its top-down row is (0.5 - y / size) · RES.
spot_row_top = int(math.floor((0.5 - SPOT_CENTRE_Y / ORTHO_SCALE) * RESOLUTION))
spot_column = RESOLUTION // 2
point_column = int(math.floor((POINT_CENTRE_X / ORTHO_SCALE + 0.5) * RESOLUTION))
area_column = int(math.floor((AREA_CENTRE_X / ORTHO_SCALE + 0.5) * RESOLUTION))
result["spot_row_top_down"] = spot_row_top
result["spot_column"] = spot_column
result["spot_mean_red"] = window_mean(spot_pixels, spot_column, spot_row_top)
result["spot_formula_radiance"] = SPOT_ENERGY / (4.0 * math.pi**2 * SPOT_RADIUS**2)
result["area_clear_mean_red"] = window_mean(spot_pixels, area_column, RESOLUTION // 2)
result["point_mean_red"] = window_mean(spot_pixels, point_column, RESOLUTION // 2)
result["area_column"] = area_column
result["point_column"] = point_column

# Absorber scene: the same area light, behind a slab of absorbing volume 2 m thick. EEVEE multiplies the light's shape by
# the transmittance in front of it (`eevee_light_shape_display.bsl.hh`, `vol.transmittance`).
scene = new_scene()
add_camera(scene, "ORTHO", (0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
area = add_light(scene, "AreaLight", "AREA", (AREA_CENTRE_X, 0.0, 0.0), (math.pi, 0.0, 0.0))
area.shape = "SQUARE"
area.size = AREA_SIZE
area.energy = AREA_ENERGY
# A 2.3 m wide slab, from x = -2.65 to -0.35 m, over the area light, spanning the frame in Y and 2 m thick in Z.
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(AREA_CENTRE_X, 0.0, 4.0))
slab = bpy.context.object
slab.name = "Absorber"
slab.scale = (2.3, ORTHO_SCALE, SLAB_THICKNESS)
material = bpy.data.materials.new("Absorber")
material.use_nodes = True
tree = material.node_tree
output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
for node in list(tree.nodes):
    if node.type == "BSDF_PRINCIPLED":
        tree.nodes.remove(node)
volume = tree.nodes.new("ShaderNodeVolumePrincipled")
volume.inputs["Density"].default_value = SLAB_DENSITY
volume.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
volume.inputs["Anisotropy"].default_value = 0.0
tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
slab.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=absorber_blend)
absorber_pixels = render_and_read(scene, absorber_exr)
result["area_through_absorber_mean_red"] = window_mean(absorber_pixels, area_column, RESOLUTION // 2)
result["absorber_transmittance"] = result["area_through_absorber_mean_red"] / result["area_clear_mean_red"]
result["absorber_density"] = SLAB_DENSITY
result["absorber_thickness"] = SLAB_THICKNESS
result["beer_lambert_reference"] = math.exp(-SLAB_DENSITY * SLAB_THICKNESS)
with open(json_out, "w") as handle:
    json.dump(result, handle, indent=2)
print("WROTE", json_out, json.dumps(result))

