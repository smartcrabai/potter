"""Light ray visibility, rendered by Blender's Cycles (v5.2.2) as the oracle for Potter.

A top-down orthographic view of a diffuse floor and a roughness-0 metal sphere, lit by two disk
area lights above them, with a third disk that the camera can see. The lights' `visible_*` flags change
what the camera sees:

  GlossyOff   visible_glossy=False. Its disk is absent from the sphere's mirror image, so the
              sphere's centre, where the mirror reflects it straight back up, is dark.
  DiffuseOff  visible_diffuse=False and visible_glossy=False. Cycles drops the diffuse closure from
              the disk's direct lighting as well as from diffuse bounce hits, so the floor under it
              receives no light from the disk. It is not reflected in the sphere.
  CameraLight visible_camera=True. Turned to face the camera, it is seen as a disk of radiance
              RADIANCE. The default flags hide it, so the camera sees the floor where it is.

Each light is also drawn with default flags into a second render. The test checks that the flags
change the reference, so a reference that lost the scene cannot pass by matching a broken render.
Blender ignores a light's `visible_shadow` (Cycles' `light_object_visibility_flags` has no shadow
bit), so the flagged scene sets none.

The disks emit radiance RADIANCE in W/(sr m²): power is radiance · π · area, with area π r².
The patch centres in `tests/native_light_ray_visibility.rs` must match the layout below.
The test's patch for CameraLight is centred at its world position, (-1, -0.9), as for the other lights.

Arguments after `--`: blend_out, flagged_exr_out, default_exr_out.
"""

import math
import sys

import bpy

WIDTH = 96
HEIGHT = 96
SAMPLES = 128
SEED = 0
MAX_BOUNCES = 1
# Orthographic frame: the view spans +-1.5 m on both axes.
ORTHO_SCALE = 3.0
LIGHT_HEIGHT = 2.0
DISK_SIZE = 1.0
RADIANCE = 4.0

# Light name, location, rotation, and the flags applied in the flagged scene only. The disks emit down
# (local -Z), except CameraLight, turned to emit up towards the camera.
LIGHTS = [
    ("GlossyOff", (0.0, 0.0, LIGHT_HEIGHT), (0.0, 0.0, 0.0), {"visible_glossy": False}),
    (
        "DiffuseOff",
        (1.0, -0.9, LIGHT_HEIGHT),
        (0.0, 0.0, 0.0),
        {"visible_diffuse": False, "visible_glossy": False},
    ),
    ("CameraLight", (-1.0, -0.9, LIGHT_HEIGHT), (math.pi, 0.0, 0.0), {"visible_camera": True}),
]


def principled(name, base, metallic, roughness, specular):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    return material


def smooth(obj):
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def build_scene(flagged):
    """Builds the scene in a fresh session. The default scene leaves every light flag at Blender's
    default, so the second render shows what the flags change."""
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

    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.name = "Floor"
    bpy.context.object.data.materials.append(
        principled("Floor", (0.8, 0.8, 0.8, 1.0), 0.0, 1.0, 0.0)
    )

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.3, segments=64, ring_count=32, location=(0.0, 0.0, 0.3)
    )
    sphere = bpy.context.object
    sphere.name = "Mirror"
    sphere.data.materials.append(principled("Mirror", (1.0, 1.0, 1.0, 1.0), 1.0, 0.0, 0.5))
    smooth(sphere)

    for name, location, rotation, flags in LIGHTS:
        data = bpy.data.lights.new(name, "AREA")
        data.shape = "DISK"
        data.size = DISK_SIZE
        # Cycles' disk radiance is P / (π · area), with area π r².
        data.energy = RADIANCE * math.pi * math.pi * (DISK_SIZE / 2.0) ** 2
        data.cycles.use_multiple_importance_sampling = True
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        # An area light emits along its local -Z, which is -Z in world space at the default rotation.
        light.location = location
        light.rotation_euler = rotation
        if flagged:
            for flag, value in flags.items():
                setattr(light, flag, value)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    # A camera with no rotation looks down -Z: the image's +X is world +X and its +Y is world +Y.
    camera.location = (0.0, 0.0, 10.0)
    scene.camera = camera
    return scene


def render(path):
    scene = bpy.context.scene
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    blend_out, flagged_exr, default_exr = sys.argv[sys.argv.index("--") + 1:][:3]
    build_scene(flagged=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_out)
    render(flagged_exr)
    build_scene(flagged=False)
    render(default_exr)


if __name__ == "__main__":
    main()
