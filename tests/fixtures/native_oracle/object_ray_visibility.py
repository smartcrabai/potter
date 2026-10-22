"""Object ray visibility, rendered by Blender's Cycles (v5.2.2) as the oracle for Potter.

A top-down orthographic view of a diffuse floor under one point light. Each object's `visible_*`
flags are set so that the flag changes the image (world coordinates in metres, +Y up in the image):

  HiddenCube       visible_camera=False. It casts a shadow on the floor. It is also
                   visible_glossy=False: with the default on, both mirrors reflect it into frame.
  Occluder         visible_shadow=False. The camera sees it; the floor under it stays lit.
  GlossyOffMirror  roughness-0 metal hovering over the floor, visible_glossy=False. The Reflector
                   cannot reflect it, so the Reflector shows the sky where it would be.
  Reflector        roughness-0 metal on the floor with default flags. Its disc shows the
                   GlossyOffMirror only while that object is visible_glossy.

The point light has no radius, so direct lighting is noise-free. Cycles and Potter both count
`max_bounces` as scatter events. Cycles uses a 1 px box filter, the same as Potter's jittered
pixel sampling. Adaptive sampling and clamping are off so the sample count is fixed.

Arguments after `--`: blend_out, exr_out. `tests/native_object_ray_visibility.rs` runs this
script headless into a temporary directory, imports the `.blend`, and compares Potter's render with
the EXR patch by patch. The patch centres in that test must match the world layout below.
"""

import sys

import bpy

WIDTH = 64
HEIGHT = 64
SAMPLES = 32
SEED = 0
MAX_BOUNCES = 2
# Orthographic frame: the view spans +-1.5 m on both axes.
ORTHO_SCALE = 3.0
LIGHT_ENERGY = 100.0
LIGHT_LOCATION = (-1.4, -0.9, 2.2)

# Ray visibility overrides per object name. Flags not listed keep Blender's default (True).
OBJECT_FLAGS = {
    "HiddenCube": {"visible_camera": False, "visible_glossy": False},
    "Occluder": {"visible_shadow": False},
    "GlossyOffMirror": {"visible_glossy": False},
}


def principled(name, base, metallic, roughness, specular):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    return material


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def smooth(obj):
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def named(obj, name, material, overrides):
    obj.name = name
    obj.data.materials.append(material)
    for flag, value in {**OBJECT_FLAGS.get(name, {}), **overrides.get(name, {})}.items():
        setattr(obj, flag, value)
    return obj


def build_scene(overrides=None):
    """Builds the scene in a fresh session. `overrides` maps an object name to extra ray visibility
    flags applied on top of `OBJECT_FLAGS`, so a sanity check can toggle one flag."""
    overrides = overrides or {}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = SEED
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = MAX_BOUNCES
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"

    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
    named(
        bpy.context.object,
        "Floor",
        principled("Floor", (0.8, 0.8, 0.8, 1.0), 0.0, 1.0, 0.0),
        overrides,
    )

    light_data = bpy.data.lights.new("Light", "POINT")
    light_data.energy = LIGHT_ENERGY
    light_data.shadow_soft_size = 0.0
    light = link(bpy.data.objects.new("Light", light_data))
    light.location = LIGHT_LOCATION

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera = link(bpy.data.objects.new("Camera", camera_data))
    camera.location = (0.0, 0.0, 10.0)
    scene.camera = camera

    matte = principled("Matte", (0.8, 0.8, 0.8, 1.0), 0.0, 1.0, 0.0)
    mirror = principled("Mirror", (1.0, 1.0, 1.0, 1.0), 1.0, 0.0, 0.5)

    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(-0.5, 0.45, 0.25))
    named(bpy.context.object, "HiddenCube", matte, overrides)

    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0.7, 0.6, 0.3))
    named(bpy.context.object, "Occluder", matte, overrides)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.4, segments=64, ring_count=32, location=(-0.2, -0.6, 0.95)
    )
    glossy_off = named(bpy.context.object, "GlossyOffMirror", mirror, overrides)
    smooth(glossy_off)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.3, segments=64, ring_count=32, location=(-0.95, -0.45, 0.3)
    )
    reflector = named(bpy.context.object, "Reflector", mirror, overrides)
    smooth(reflector)

    return scene


def render(exr_out):
    scene = bpy.context.scene
    scene.render.filepath = exr_out
    bpy.ops.render.render(write_still=True)


def main():
    blend_out, exr_out = sys.argv[sys.argv.index("--") + 1:][:2]
    build_scene()
    bpy.ops.wm.save_as_mainfile(filepath=blend_out)
    render(exr_out)


if __name__ == "__main__":
    main()
