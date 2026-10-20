"""Light volume-scatter visibility, rendered by Blender's Cycles (v5.2.2) as the oracle for Potter.

A top-down orthographic view of a scattering fog cube under a point light, in front of an emitting
backdrop. The light is above the fog, so its scattered light reaches the camera only through the
fog's volume. The light's `visible_volume_scatter` is False in the flagged scene: then it does not
light the volume, and the fog shows only the backdrop it attenuates.

  Fog     a cube of Principled Volume, density DENSITY, white scatter, anisotropy 0. Its surface
          shader is removed, so it is a volume only.
  Lamp    point light, energy ENERGY, directly above the fog. visible_volume_scatter=False in the
          flagged scene.
  Backdrop a black surface with an emission of strength BACKDROP at z = BACKDROP_Z.

Each scene is rendered twice: flagged, and with every flag at Blender's default. The test checks
that the flag changes the fog patch, and that the backdrop control (outside the fog) does not. The
render uses one volume bounce and the same bounce budget as Potter's `max_bounces`, so both engines
count one scatter event.

Arguments after `--`: blend_out, flagged_exr_out, default_exr_out.
"""

import sys

import bpy

WIDTH = 96
HEIGHT = 96
SAMPLES = 256
SEED = 0
MAX_BOUNCES = 1
# Orthographic frame: the view spans +-1.5 m on both axes.
ORTHO_SCALE = 3.0
DENSITY = 0.5
FOG_SIZE = 2.0
ENERGY = 300.0
BACKDROP = 0.2
BACKDROP_Z = -3.0
LAMP_LOCATION = (0.0, 0.0, 3.0)


def configure(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = SEED
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = MAX_BOUNCES
    scene.cycles.volume_bounces = MAX_BOUNCES
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


def fog_material():
    material = bpy.data.materials.new("Fog")
    material.use_nodes = True
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    # The fog is a volume only: remove the default surface shader so the cube is not opaque.
    for node in list(tree.nodes):
        if node.type == "BSDF_PRINCIPLED":
            tree.nodes.remove(node)
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = DENSITY
    volume.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    volume.inputs["Anisotropy"].default_value = 0.0
    tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
    return material


def backdrop_material():
    material = bpy.data.materials.new("Backdrop")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = BACKDROP
    return material


def build_scene(flagged):
    """Builds the scene in a fresh session. The default scene leaves every flag at Blender's default,
    so the second render shows what the flag changes."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure(scene)

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, BACKDROP_Z))
    backdrop = bpy.context.object
    backdrop.name = "Backdrop"
    backdrop.data.materials.append(backdrop_material())

    bpy.ops.mesh.primitive_cube_add(size=FOG_SIZE, location=(0.0, 0.0, 0.0))
    fog = bpy.context.object
    fog.name = "Fog"
    fog.data.materials.append(fog_material())

    light_data = bpy.data.lights.new("Lamp", "POINT")
    light_data.energy = ENERGY
    light_data.shadow_soft_size = 0.0
    lamp = bpy.data.objects.new("Lamp", light_data)
    scene.collection.objects.link(lamp)
    lamp.location = LAMP_LOCATION
    if flagged:
        lamp.visible_volume_scatter = False

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 8.0)
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
