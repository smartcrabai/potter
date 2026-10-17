"""Build the reserved-name color-layer oracle with Blender 5.2.2.

Usage: Blender --background --factory-startup --python build_reserved_color_oracle.py -- OUT_DIR

Writes reserved_color_oracle.blend and expected.json. The JSON is collected from Blender's own
mesh, group, material, and render APIs after authoring; it is never derived from Potter output.
"""

import json
import os
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)

BASE_FACTOR = (0.8, 0.6, 0.4, 1.0)
OBJECTS = [
    {
        "name": "ReservedLeft",
        "location_x": -0.95,
        "vertex_color": (0.25, 0.5, 0.75, 0.35),
        "corner_colors": [
            (0.94, 0.07, 0.21, 0.31),
            (0.81, 0.16, 0.43, 0.47),
            (0.72, 0.28, 0.12, 0.61),
            (0.63, 0.39, 0.24, 0.83),
        ],
        "hip_weights": (0.1, 0.35, 0.65, 0.9),
    },
    {
        "name": "ReservedRight",
        "location_x": 0.95,
        "vertex_color": (0.7, 0.2, 0.4, 0.8),
        "corner_colors": [
            (0.11, 0.82, 0.43, 0.93),
            (0.26, 0.73, 0.15, 0.76),
            (0.37, 0.64, 0.28, 0.58),
            (0.48, 0.55, 0.39, 0.42),
        ],
        "hip_weights": None,
    },
]


def read_color_layers(mesh):
    return [
        {
            "name": layer.name,
            "domain": layer.domain,
            "data_type": layer.data_type,
            "values": [[float(channel) for channel in item.color] for item in layer.data],
        }
        for layer in mesh.color_attributes
    ]


def read_groups(obj):
    result = {}
    for group in obj.vertex_groups:
        weights = {}
        for vertex in obj.data.vertices:
            for membership in vertex.groups:
                if membership.group == group.index:
                    weights[str(vertex.index)] = float(membership.weight)
        result[group.name] = weights
    return result


def make_material():
    material = bpy.data.materials.new("Reserved Shared Color")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.8
    vertex_color = tree.nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = "vertex_groups"
    multiply = tree.nodes.new("ShaderNodeMix")
    multiply.data_type = "RGBA"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1.0
    input_a = next(socket for socket in multiply.inputs if socket.name == "A" and socket.type == "RGBA")
    input_b = next(socket for socket in multiply.inputs if socket.name == "B" and socket.type == "RGBA")
    result = next(socket for socket in multiply.outputs if socket.name == "Result" and socket.type == "RGBA")
    input_a.default_value = BASE_FACTOR
    tree.links.new(vertex_color.outputs["Color"], input_b)
    tree.links.new(result, principled.inputs["Base Color"])
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, vertex_color, multiply, input_a


def make_mesh(spec, material):
    mesh = bpy.data.meshes.new(spec["name"] + "Mesh")
    mesh.from_pydata(
        [(-0.65, -0.65, 0.0), (0.65, -0.65, 0.0), (0.65, 0.65, 0.0), (-0.65, 0.65, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update(calc_edges=True)
    point = mesh.color_attributes.new(name="vertex_groups", type="FLOAT_COLOR", domain="POINT")
    for item in point.data:
        item.color = spec["vertex_color"]
    try:
        corner = mesh.color_attributes.new(name="color_layers", type="FLOAT_COLOR", domain="CORNER")
    except RuntimeError:
        corner = None
    if corner is not None and corner.name != "color_layers":
        mesh.color_attributes.remove(corner)
        corner = None
    if corner is not None:
        for item, color in zip(corner.data, spec["corner_colors"]):
            item.color = color
    mesh.color_attributes.active_color_index = 1 if corner is not None else 0
    mesh.color_attributes.render_color_index = 0

    obj = bpy.data.objects.new(spec["name"], mesh)
    obj.location.x = spec["location_x"]
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    if spec["hip_weights"] is not None:
        group = obj.vertex_groups.new(name="Hip")
        for index, weight in enumerate(spec["hip_weights"]):
            group.add([index], weight, "REPLACE")
    return obj


material, vertex_color_node, multiply_node, factor_socket = make_material()
objects = [make_mesh(spec, material) for spec in OBJECTS]

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.render.resolution_x = 128
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("Black World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

light_data = bpy.data.lights.new("Oracle Area", type="AREA")
light_data.energy = 1400.0
light_data.shape = "DISK"
light_data.size = 5.0
light = bpy.data.objects.new("Oracle Area", light_data)
light.location = (0.0, 0.0, 4.0)
scene.collection.objects.link(light)

camera_data = bpy.data.cameras.new("Oracle Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Oracle Camera", camera_data)
camera.location = (0.0, 0.0, 7.0)
scene.collection.objects.link(camera)
scene.camera = camera

blend_path = os.path.join(OUT, "reserved_color_oracle.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

# Render Blender's own numeric reference at the exact pixels the CLI test samples. In background
# mode `Render Result` keeps size (0, 0) even after a render, so the linear EXR is written and reloaded.
scene.render.filepath = os.path.join(OUT, "reserved_color_render.exr")
bpy.ops.render.render(write_still=True)
render_result = bpy.data.images.load(scene.render.filepath)
render_width, render_height = render_result.size
probe_pixels = {}
probe_uv = {}
for name, x in (("ReservedLeft", 32), ("ReservedRight", 96)):
    y = render_height // 2
    offset = (y * render_width + x) * 4
    probe_pixels[name] = [float(channel) for channel in render_result.pixels[offset:offset + 4]]
    probe_uv[name] = [(x + 0.5) / render_width, (y + 0.5) / render_height]

mesh_expectations = []
for obj in objects:
    group_weights = read_groups(obj)
    mesh_expectations.append(
        {
            "name": obj.name,
            "color_layers": read_color_layers(obj.data),
            "color_layer_order": [layer.name for layer in obj.data.color_attributes],
            "color_active_index": obj.data.color_attributes.active_color_index,
            "color_render_index": obj.data.color_attributes.render_color_index,
            "vertex_groups": group_weights,
        }
    )

expected = {
    "meshes": mesh_expectations,
    "material": {
        "name": material.name,
        "shared_by": [obj.name for obj in objects],
        "vertex_color_layer": vertex_color_node.layer_name,
        "vertex_color_blend": multiply_node.blend_type,
        "blend_data_type": multiply_node.data_type,
        "base_factor": [float(channel) for channel in factor_socket.default_value],
        "base_factor_socket": factor_socket.identifier,
    },
    "render_probe": {"pixels": probe_pixels, "uv": probe_uv, "size": [render_width, render_height]},
}
with open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, indent=2)
