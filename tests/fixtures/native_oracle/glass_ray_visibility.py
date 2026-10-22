"""Object transmission visibility, rendered by Blender's Cycles (v5.2.2) as the oracle for Potter.

A top-down orthographic view of a glass sphere in front of a white emitting backdrop. The sphere
is tinted red, so a path that refracts through it twice comes back red, and a path that cannot
refract comes back grey:

  Glass  transmission 1, roughness 0, base colour (1, 0.2, 0.2). Its `visible_transmission` is
         False in the flagged scene. Transmission rays then pass through it untinted, so the
         sphere's centre shows the backdrop grey. With the default flags the centre is red.

Each scene is rendered twice: flagged, and with every flag at Blender's default. The test checks
that the flag changes the reference, so a reference that lost the scene cannot pass by matching a
broken render. A patch on the backdrop, away from the sphere, is the control: it must not change.

The backdrop is a Principled surface with a black base and an emission of strength BACKDROP, so it
is seen directly with no light in the scene. The patch centres in
`tests/native_glass_ray_visibility.rs` must match the layout below.

Arguments after `--`: blend_out, flagged_exr_out, default_exr_out.
"""

import sys

import bpy

WIDTH = 48
HEIGHT = 48
SAMPLES = 96
SEED = 0
MAX_BOUNCES = 8
# Orthographic frame: the view spans +-1.5 m on both axes.
ORTHO_SCALE = 3.0
BACKDROP = 0.5
SPHERE_RADIUS = 0.8
SPHERE_LOCATION = (0.0, 0.0, 0.5)
BACKDROP_Z = -1.0


def configure(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = SEED
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = MAX_BOUNCES
    scene.cycles.transparent_max_bounces = MAX_BOUNCES
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


def principled(name, **inputs):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    for socket, value in inputs.items():
        bsdf.inputs[socket].default_value = value
    return material


def build_scene(flagged):
    """Builds the scene in a fresh session. The default scene leaves every flag at Blender's default,
    so the second render shows what the flag changes."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure(scene)

    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.0, 0.0, BACKDROP_Z))
    backdrop = bpy.context.object
    backdrop.name = "Backdrop"
    backdrop.data.materials.append(
        principled(
            "Backdrop",
            **{"Base Color": (0.0, 0.0, 0.0, 1.0), "Specular IOR Level": 0.0,
               "Emission Color": (1.0, 1.0, 1.0, 1.0), "Emission Strength": BACKDROP},
        )
    )

    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, segments=64, ring_count=32,
                                         location=SPHERE_LOCATION)
    glass = bpy.context.object
    glass.name = "Glass"
    glass.data.materials.append(
        principled(
            "Glass",
            **{"Base Color": (1.0, 0.2, 0.2, 1.0), "Metallic": 0.0, "Roughness": 0.0,
               "IOR": 1.45, "Transmission Weight": 1.0},
        )
    )
    for polygon in glass.data.polygons:
        polygon.use_smooth = True
    if flagged:
        glass.visible_transmission = False

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 5.0)
    scene.camera = camera
    return scene


def render(path):
    bpy.context.scene.render.filepath = path
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
