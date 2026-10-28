"""Blender 5.2.2 EEVEE oracle for realtime shader-converter behavior.

Builds a 96x48 row of nine emissive planes. Each center pixel probes an EEVEE-specific
converter/output (Bump, Normal Map, Blackbody, Wavelength, Geometry, HSV, or Camera Data).
The saved scene is imported and rendered by Potter's realtime consumer; the EXR and JSON here
are generated with Blender and the JSON stores the center pixels used as the oracle.

Arguments after `--`: blend_out, exr_out, json_out.
"""

import json
import sys

import bpy
if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

blend_out, exr_out, json_out = sys.argv[sys.argv.index("--") + 1 :][:3]

SIZE = (96, 48)
CAMERA_SCALE = 18.0
PLANE_SIZE = 1.5
CENTER_Y = 0.0
PROBES = [
    ("bump", "Bump"),
    ("normal_map", "Normal Map"),
    ("blackbody", "Blackbody"),
    ("wavelength", "Wavelength"),
    ("true_normal", "Geometry True Normal"),
    ("pointiness", "Geometry Pointiness"),
    ("random_per_island", "Geometry Random Per Island"),
    ("hsv", "HSV Clamp"),
    ("camera_data", "Camera Data"),
    ("camera_data_bump", "Camera Data Bump Height"),
]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("Black")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0


def material_for(probe):
    name = dict(PROBES)[probe]
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emission.outputs[0], output.inputs["Surface"])

    if probe == "bump":
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        separate_uv = tree.nodes.new("ShaderNodeSeparateXYZ")
        bump = tree.nodes.new("ShaderNodeBump")
        separate_normal = tree.nodes.new("ShaderNodeSeparateXYZ")
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = 1.0
        tree.links.new(coordinates.outputs["UV"], separate_uv.inputs[0])
        tree.links.new(separate_uv.outputs["X"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], separate_normal.inputs[0])
        tree.links.new(separate_normal.outputs["Z"], emission.inputs["Strength"])
    elif probe == "normal_map":
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Color"].default_value = (0.8, 0.5, 1.0, 1.0)
        tree.links.new(normal_map.outputs["Normal"], emission.inputs["Color"])
    elif probe == "blackbody":
        blackbody = tree.nodes.new("ShaderNodeBlackbody")
        blackbody.inputs["Temperature"].default_value = 3200.0
        tree.links.new(blackbody.outputs["Color"], emission.inputs["Color"])
    elif probe == "wavelength":
        wavelength = tree.nodes.new("ShaderNodeWavelength")
        wavelength.inputs["Wavelength"].default_value = 540.0
        tree.links.new(wavelength.outputs["Color"], emission.inputs["Color"])
    elif probe == "true_normal":
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        tree.links.new(geometry.outputs["True Normal"], emission.inputs["Color"])
    elif probe == "pointiness":
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        tree.links.new(geometry.outputs["Pointiness"], emission.inputs["Strength"])
    elif probe == "random_per_island":
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        tree.links.new(geometry.outputs["Random Per Island"], emission.inputs["Strength"])
    elif probe == "hsv":
        hue_saturation = tree.nodes.new("ShaderNodeHueSaturation")
        hue_saturation.inputs["Color"].default_value = (0.4, 0.2, 0.1, 1.0)
        hue_saturation.inputs["Value"].default_value = 3.0
        tree.links.new(hue_saturation.outputs["Color"], emission.inputs["Color"])
    elif probe == "camera_data":
        camera_data = tree.nodes.new("ShaderNodeCameraData")
        tree.links.new(camera_data.outputs["View Z Depth"], emission.inputs["Strength"])
    elif probe == "camera_data_bump":
        camera_data = tree.nodes.new("ShaderNodeCameraData")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs["Value_001"].default_value = 10.0
        bump = tree.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = 0.5
        separate_normal = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(camera_data.outputs["View Z Depth"], scale.inputs["Value"])
        tree.links.new(scale.outputs["Value"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], separate_normal.inputs["Vector"])
        tree.links.new(separate_normal.outputs["Z"], emission.inputs["Strength"])
    return material


def probe_center_x(index, probe):
    return 8.6 if probe == "camera_data_bump" else -7.40625 + 1.875 * index

for index, (probe, name) in enumerate(PROBES):
    center_x = probe_center_x(index, probe)
    bpy.ops.mesh.primitive_plane_add(
        size=PLANE_SIZE,
        location=(center_x, CENTER_Y, 0.0),
    )
    plane = bpy.context.object
    plane.name = name + " Plane"
    if probe == "camera_data_bump":
        plane.rotation_euler = (0.0, 0.5235987755982988, 0.0)
    plane.data.materials.append(material_for(probe))


camera_data = bpy.data.cameras.new("Converter Oracle Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = CAMERA_SCALE
camera = bpy.data.objects.new("Converter Oracle Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 16
scene.render.resolution_x, scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = exr_out

bpy.ops.wm.save_as_mainfile(filepath=blend_out, check_existing=False)
bpy.ops.render.render(write_still=True)
rendered = bpy.data.images.load(exr_out, check_existing=False)
pixels = list(rendered.pixels)
width, height = (int(value) for value in rendered.size)

probes = {}
for index, (probe, _) in enumerate(PROBES):
    x = round((probe_center_x(index, probe) + CAMERA_SCALE / 2.0) / CAMERA_SCALE * width - 0.5)
    y = height // 2
    offset = (y * width + x) * 4
    probes[probe] = {
        "uv": [(x + 0.5) / width, (y + 0.5) / height],
        "rgba": pixels[offset : offset + 4],
    }

with open(json_out, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "engine": "Blender 5.2.2 EEVEE",
            "resolution": [width, height],
            "probes": probes,
        },
        stream,
        indent=2,
    )
