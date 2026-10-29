"""Build and render linear shader-texture probes with Blender 5.2.2.

Run with Blender 5.2.2:
    Blender -b --factory-startup --threads 1 -P generate_oracle.py -- OUT.blend ORACLE.json MAPPED.blend WORLD.blend

Each tile is an emission-only consumer of one texture output. The report contains Blender's
Cycles and EEVEE pixel at each tile center, plus the socket defaults and RNA modes used to build
that tile. Potter's output is never used to construct an expectation.
"""
import json
import math
import os
import sys
import tempfile

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
BLEND_OUT = os.path.abspath(argv[0])
ORACLE_OUT = os.path.abspath(argv[1])
MAPPED_BLEND_OUT = os.path.abspath(argv[2])
WORLD_BLEND_OUT = os.path.abspath(argv[3])
PIXELS_PER_TILE = 4
COLUMNS = 16
CONSTANT_VECTOR = (0.17, 0.31, 0.53)
CONSTANT_W = 0.73
BRICK_ROW_VECTOR = (0.17, 0.62, 0.53)

NODE_TYPES = {
    "wave": "ShaderNodeTexWave",
    "magic": "ShaderNodeTexMagic",
    "brick": "ShaderNodeTexBrick",
    "white_noise": "ShaderNodeTexWhiteNoise",
    "gabor": "ShaderNodeTexGabor",
}
OUTPUTS = {
    "wave": ("Color", "Fac"),
    "magic": ("Color", "Fac"),
    "brick": ("Color", "Fac"),
    "white_noise": ("Color", "Value"),
    "gabor": ("Value", "Phase", "Intensity"),
}
INPUT_DEFAULTS = {
    "wave": {
        "Vector": CONSTANT_VECTOR,
        "Scale": 4.3,
        "Distortion": 2.1,
        "Detail": 3.0,
        "Detail Scale": 2.2,
        "Detail Roughness": 0.37,
        "Phase Offset": 0.29,
    },
    "magic": {
        "Vector": CONSTANT_VECTOR,
        "Scale": 5.1,
        "Distortion": 2.3,
    },
    "brick": {
        "Vector": CONSTANT_VECTOR,
        "Color1": (0.72, 0.19, 0.07, 1.0),
        "Color2": (0.21, 0.39, 0.68, 1.0),
        "Mortar": (0.08, 0.11, 0.16, 1.0),
        "Scale": 4.7,
        "Mortar Size": 0.12,
        "Mortar Smooth": 0.08,
        "Bias": 0.27,
        "Brick Width": 0.43,
        "Row Height": 0.31,
    },
    "white_noise": {"Vector": CONSTANT_VECTOR, "W": CONSTANT_W},
    "gabor": {
        "Vector": CONSTANT_VECTOR,
        "Scale": 3.7,
        "Frequency": 2.4,
        "Anisotropy": 0.63,
        "Orientation": 0.73,
    },
}


def plain(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    try:
        return [plain(component) for component in value]
    except TypeError:
        return str(value)


def enum_ids(node, name):
    prop = node.bl_rna.properties.get(name)
    if prop is None or prop.type != "ENUM":
        raise RuntimeError(f"{node.bl_idname} has no enum property {name}")
    return [item.identifier for item in prop.enum_items]


def prototype(bl_idname):
    material = bpy.data.materials.new("procedural_oracle_prototype")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    node = tree.nodes.new(bl_idname)
    return material, node


def make_modes():
    modes = {}
    material, node = prototype(NODE_TYPES["wave"])
    wave_types = enum_ids(node, "wave_type")
    profiles = enum_ids(node, "wave_profile")
    bands_directions = enum_ids(node, "bands_direction")
    rings_directions = enum_ids(node, "rings_direction")
    modes["wave"] = [
        {
            "wave_type": wave_type,
            "wave_profile": profile,
            "bands_direction": direction if wave_type == "BANDS" else bands_directions[-1],
            "rings_direction": direction if wave_type == "RINGS" else rings_directions[-1],
        }
        for wave_type in wave_types
        for profile in profiles
        for direction in (bands_directions if wave_type == "BANDS" else rings_directions)
    ]
    bpy.data.materials.remove(material)

    material, node = prototype(NODE_TYPES["white_noise"])
    modes["white_noise"] = [{"noise_dimensions": mode} for mode in enum_ids(node, "noise_dimensions")]
    bpy.data.materials.remove(material)

    material, node = prototype(NODE_TYPES["gabor"])
    modes["gabor"] = [{"gabor_type": mode} for mode in enum_ids(node, "gabor_type")]
    bpy.data.materials.remove(material)

    modes["magic"] = [{"turbulence_depth": 6}]
    modes["brick"] = [
        {
            "offset": 0.37,
            "offset_frequency": 3,
            "squash": 0.73,
            "squash_frequency": 4,
        }
    ]
    return modes


def set_input_defaults(node, node_key, properties):
    defaults = dict(INPUT_DEFAULTS[node_key])
    if node_key == "gabor":
        defaults["Orientation"] = (
            0.73 if properties["gabor_type"] == "2D" else (0.36, 0.48, 0.8)
        )
    for name, value in defaults.items():
        socket = node.inputs.get(name)
        if socket is not None and hasattr(socket, "default_value"):
            socket.default_value = value


def set_node_properties(node, properties):
    for name, value in properties.items():
        if not hasattr(node, name):
            raise RuntimeError(f"{node.bl_idname} has no property {name}")
        setattr(node, name, value)


def input_values(node):
    return [
        {
            "name": socket.name,
            "identifier": socket.identifier,
            "enabled": bool(socket.enabled),
            "default": plain(socket.default_value),
        }
        for socket in node.inputs
        if hasattr(socket, "default_value")
    ]


def property_values(node, requested):
    return {name: plain(getattr(node, name)) for name in requested}


def texture_mapping_values(mapping):
    return {
        name: plain(getattr(mapping, name))
        for name in (
            "vector_type", "translation", "rotation", "scale", "min", "max",
            "use_min", "use_max", "mapping_x", "mapping_y", "mapping_z", "mapping",
        )
    }


def color_mapping_values(mapping):
    ramp = mapping.color_ramp
    return {
        "use_color_ramp": bool(mapping.use_color_ramp),
        "brightness": float(mapping.brightness),
        "contrast": float(mapping.contrast),
        "saturation": float(mapping.saturation),
        "blend_type": str(mapping.blend_type),
        "blend_color": plain(mapping.blend_color),
        "blend_factor": float(mapping.blend_factor),
        "color_ramp": {
            "color_mode": str(ramp.color_mode),
            "hue_interpolation": str(ramp.hue_interpolation),
            "interpolation": str(ramp.interpolation),
            "elements": [
                {
                    "position": float(element.position),
                    "color": [float(channel) for channel in element.color],
                }
                for element in ramp.elements
            ],
        },
    }


def add_probe(scene, node_key, properties, coordinate_source, output_name, index):
    node_type = NODE_TYPES[node_key]
    probe_id = f"{node_key}_{index:03}_{coordinate_source}_{output_name.lower()}"
    material = bpy.data.materials.new(f"oracle_{probe_id}")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    texture = tree.nodes.new(node_type)
    texture.name = probe_id
    texture.label = probe_id
    set_node_properties(texture, properties)
    set_input_defaults(texture, node_key, properties)
    if node_key == "gabor" and coordinate_source == "constant_edge":
        texture.inputs["Anisotropy"].default_value = 0.5
        orientation = next(socket for socket in texture.inputs if socket.identifier == "Orientation 3D")
        orientation.default_value = (-1.0, 0.0, 0.0)
    if node_key == "brick" and coordinate_source == "constant_row9":
        texture.inputs["Vector"].default_value = BRICK_ROW_VECTOR
    if probe_id == "wave_000_constant_color":
        mapping = texture.color_mapping
        mapping.brightness = 1.25
        mapping.use_color_ramp = True
        mapping.color_ramp.interpolation = "CONSTANT"
        mapping.color_ramp.elements[1].position = 0.68
        mapping.color_ramp.elements[1].color = (0.22, 0.72, 0.31, 1.0)

    links = tree.links
    vector = texture.inputs.get("Vector")
    if coordinate_source == "generated":
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        coordinates.name = f"{probe_id}_coordinates"
        if vector is not None and vector.enabled:
            links.new(coordinates.outputs["Generated"], vector)
    elif vector is not None and vector.enabled:
        constant_vector = tree.nodes.new("ShaderNodeCombineXYZ")
        constant_vector.name = f"{probe_id}_constant_vector"
        for axis, component in zip(("X", "Y", "Z"), vector.default_value):
            constant_vector.inputs[axis].default_value = component
        links.new(constant_vector.outputs["Vector"], vector)
    w_input = texture.inputs.get("W")
    if w_input is not None and w_input.enabled and coordinate_source == "generated":
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        links.new(coordinates.outputs["Generated"], separate.inputs["Vector"])
        links.new(separate.outputs["X"], w_input)

    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    source = texture.outputs.get(output_name)
    if source is None:
        raise RuntimeError(f"{node_type} has no output socket {output_name}")
    if source.type == "VALUE":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(source, emission.inputs["Strength"])
    elif source.type in ("RGBA", "VECTOR"):
        links.new(source, emission.inputs["Color"])
    else:
        raise RuntimeError(f"cannot visualize {node_type}.{output_name} ({source.type})")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    row, column = divmod(index, COLUMNS)
    mesh = bpy.data.meshes.new(f"oracle_{probe_id}_mesh")
    mesh.from_pydata(
        [(-0.5, -0.5, -0.35), (0.5, -0.5, -0.35), (0.5, 0.5, 0.35), (-0.5, 0.5, 0.35)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.materials.append(material)
    obj = bpy.data.objects.new(f"oracle_{probe_id}", mesh)
    scene.collection.objects.link(obj)
    obj.location = (column + 0.5 - COLUMNS / 2, row + 0.5 - ROWS / 2, 0.0)
    return {
        "id": probe_id,
        "node_type": node_type,
        "node_name": probe_id,
        "output": output_name,
        "output_identifier": source.identifier,
        "output_type": source.type,
        "coordinate_source": coordinate_source,
        "mode": properties,
        "properties": property_values(texture, properties),
        "texture_mapping": (
            texture_mapping_values(texture.texture_mapping)
            if hasattr(texture, "texture_mapping")
            else None
        ),
        "color_mapping": (
            color_mapping_values(texture.color_mapping)
            if hasattr(texture, "color_mapping")
            else None
        ),
        "inputs": input_values(texture),
        "row": row,
        "column": column,
    }



def build_geometry_field(scene):
    group = bpy.data.node_groups.new("Procedural Texture Field", "GeometryNodeTree")
    group.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )
    nodes = group.nodes
    links = group.links
    line = nodes.new("GeometryNodeMeshLine")
    line.inputs["Count"].default_value = 3
    line.inputs["Start Location"].default_value = (0.17, 0.23, 0.41)
    line.inputs["Offset"].default_value = (0.4, 0.1, 0.2)
    position = nodes.new("GeometryNodeInputPosition")

    wave = nodes.new("ShaderNodeTexWave")
    wave.name = "GN Wave Field"
    wave.wave_type = "RINGS"
    wave.rings_direction = "SPHERICAL"
    wave.wave_profile = "TRI"
    for name, value in {
        "Scale": 3.2,
        "Distortion": 0.35,
        "Detail": 2.0,
        "Detail Scale": 1.7,
        "Detail Roughness": 0.42,
        "Phase Offset": 0.19,
    }.items():
        wave.inputs[name].default_value = value

    magic = nodes.new("ShaderNodeTexMagic")
    magic.name = "GN Magic Field"
    magic.turbulence_depth = 6
    magic.inputs["Scale"].default_value = 4.2
    magic.inputs["Distortion"].default_value = 0.7

    brick = nodes.new("ShaderNodeTexBrick")
    brick.name = "GN Brick Field"
    brick.offset = 0.37
    brick.offset_frequency = 3
    brick.squash = 0.73
    brick.squash_frequency = 4
    for name, value in {
        "Color1": (0.72, 0.19, 0.07, 1.0),
        "Color2": (0.21, 0.39, 0.68, 1.0),
        "Mortar": (0.08, 0.11, 0.16, 1.0),
        "Scale": 1.0,
        "Mortar Size": 0.0,
        "Mortar Smooth": 0.0,
        "Bias": 0.27,
        "Brick Width": 0.43,
        "Row Height": 0.31,
    }.items():
        brick.inputs[name].default_value = value

    white_noise = nodes.new("ShaderNodeTexWhiteNoise")
    white_noise.name = "GN White Noise Field"
    white_noise.noise_dimensions = "4D"
    white_noise.inputs["W"].default_value = 0.73

    gabor = nodes.new("ShaderNodeTexGabor")
    gabor.name = "GN Gabor Field"
    gabor.gabor_type = "2D"
    gabor.inputs["Scale"].default_value = 3.7
    gabor.inputs["Frequency"].default_value = 2.4
    gabor.inputs["Anisotropy"].default_value = 0.63
    gabor.inputs["Orientation"].default_value = 0.73

    fields = [
        ("procedural_wave_fac", "FLOAT", wave, "Fac"),
        ("procedural_magic_fac", "FLOAT", magic, "Fac"),
        ("procedural_brick_color", "FLOAT_COLOR", brick, "Color"),
        ("procedural_white_noise_color", "FLOAT_COLOR", white_noise, "Color"),
        ("procedural_gabor_phase", "FLOAT", gabor, "Phase"),
    ]
    geometry = line.outputs["Mesh"]
    for name, data_type, texture, socket in fields:
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.name = f"Store {name}"
        store.domain = "POINT"
        store.data_type = data_type
        store.inputs["Name"].default_value = name
        links.new(geometry, store.inputs["Geometry"])
        links.new(position.outputs["Position"], texture.inputs["Vector"])
        links.new(texture.outputs[socket], store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    output = nodes.new("NodeGroupOutput")
    output.is_active_output = True
    links.new(geometry, output.inputs["Geometry"])

    mesh = bpy.data.meshes.new("procedural_texture_field_host_mesh")
    host = bpy.data.objects.new("procedural_texture_field_consumer", mesh)
    scene.collection.objects.link(host)
    host.location = (1000.0, 0.0, 0.0)
    modifier = host.modifiers.new("Procedural Texture Field", "NODES")
    modifier.node_group = group
    bpy.context.view_layer.update()
    evaluated = host.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result = evaluated.to_mesh()
    try:
        attributes = []
        for name, data_type, texture, socket in fields:
            attribute = result.attributes.get(name)
            if attribute is None:
                raise RuntimeError(f"missing evaluated Geometry Nodes attribute {name}")
            values = [
                [float(component) for component in item.color]
                if data_type == "FLOAT_COLOR"
                else float(item.value)
                for item in attribute.data
            ]
            attributes.append({
                "name": name,
                "data_type": data_type,
                "node_type": texture.bl_idname,
                "output": socket,
                "values": values,
            })
    finally:
        evaluated.to_mesh_clear()
    return {
        "group_name": group.name,
        "positions": [
            [0.17, 0.23, 0.41],
            [0.57, 0.33, 0.61],
            [0.97, 0.43, 0.81],
        ],
        "attributes": attributes,
    }


def configure_render(scene):
    scene.render.resolution_x = COLUMNS * PIXELS_PER_TILE
    scene.render.resolution_y = ROWS * PIXELS_PER_TILE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1.0


def add_camera(scene):
    camera_data = bpy.data.cameras.new("oracle_camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(COLUMNS)
    camera = bpy.data.objects.new("oracle_camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera


def render_probe_pixels(scene, probes):
    scene.render.resolution_x = COLUMNS * PIXELS_PER_TILE
    scene.render.resolution_y = ROWS * PIXELS_PER_TILE
    bpy.ops.render.render(write_still=False)
    image = bpy.data.images.get("Render Result")
    width, height = image.size
    channels = image.channels
    pixels = list(image.pixels[:])
    result = []
    for probe in probes:
        x = probe["column"] * PIXELS_PER_TILE + PIXELS_PER_TILE // 2
        y = probe["row"] * PIXELS_PER_TILE + PIXELS_PER_TILE // 2
        offset = (y * width + x) * channels
        rgba = pixels[offset : offset + channels]
        if channels == 3:
            rgba.append(1.0)
        result.append([float(component) for component in rgba[:4]])
    return result



def build_mapping_fixture():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    material = bpy.data.materials.new("Mapped Wave Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.name = "Mapped Wave"
    wave.texture_mapping.scale = (2.0, 1.0, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(wave.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    bpy.ops.mesh.primitive_plane_add(size=2.0)
    bpy.context.object.name = "Mapped Wave Consumer"
    bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(0.0, 0.0, 5.0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 4.0
    scene.camera = camera
    os.makedirs(os.path.dirname(MAPPED_BLEND_OUT), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=MAPPED_BLEND_OUT, check_existing=False)



def build_world_fixture():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Procedural World Oracle")
    world.use_nodes = True
    scene.world = world
    tree = world.node_tree
    tree.nodes.clear()
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.wave_profile = "SIN"
    for name, value in {
        "Scale": 0.01,
        "Distortion": 0.0,
        "Detail": 0.0,
        "Phase Offset": 0.17,
    }.items():
        wave.inputs[name].default_value = value
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.is_active_output = True
    tree.links.new(coordinates.outputs["Generated"], wave.inputs["Vector"])
    tree.links.new(wave.outputs["Color"], background.inputs["Color"])
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    camera_data = bpy.data.cameras.new("procedural_world_camera")
    camera_data.type = "PERSP"
    camera_data.lens = 50.0
    camera = bpy.data.objects.new("procedural_world_camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = camera
    scene.render.resolution_x = 32
    scene.render.resolution_y = 24
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1.0

    os.makedirs(os.path.dirname(WORLD_BLEND_OUT), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=WORLD_BLEND_OUT, check_existing=False)
    coordinates_to_sample = [(2, 2), (8, 6), (16, 12), (24, 16), (29, 20)]
    renders = {}
    with tempfile.TemporaryDirectory(prefix="procedural_world_") as render_directory:
        for engine_key, blender_engine in (("cycles", "CYCLES"), ("eevee", "BLENDER_EEVEE")):
            scene.render.engine = blender_engine
            if engine_key == "cycles":
                scene.cycles.device = "CPU"
            elif hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = 16
            render_path = os.path.join(render_directory, f"{engine_key}.exr")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(render_path, check_existing=False)
            width, height = image.size
            channels = image.channels
            pixels = list(image.pixels[:])
            probes = []
            for x, y in coordinates_to_sample:
                offset = (y * width + x) * channels
                rgba = [float(value) for value in pixels[offset : offset + channels]]
                if channels == 3:
                    rgba.append(1.0)
                probes.append({"pixel": [x, y], "rgba": rgba[:4]})
            bpy.data.images.remove(image)
            renders[engine_key] = {
                "blender_engine": blender_engine,
                "resolution": [width, height],
                "probes": probes,
            }
    return {"renders": renders}
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
modes = make_modes()
probes = []
ROWS = math.ceil(
    (
        sum(
            len(node_modes) * (3 if node_key == "brick" else 2) * len(OUTPUTS[node_key])
            for node_key, node_modes in modes.items()
        )
        + 3
    )
    / COLUMNS
)
index = 0
for node_key, node_modes in modes.items():
    for mode in node_modes:
        for coordinate_source in ("constant", "generated"):
            for output_name in OUTPUTS[node_key]:
                probes.append(add_probe(scene, node_key, mode, coordinate_source, output_name, index))
                index += 1
        if node_key == "brick":
            for output_name in OUTPUTS[node_key]:
                probes.append(add_probe(scene, node_key, mode, "constant_row9", output_name, index))
                index += 1
for depth in (0, 10):
    probes.append(
        add_probe(scene, "magic", {"turbulence_depth": depth}, "constant", "Fac", index)
    )
    index += 1
probes.append(
    add_probe(
        scene,
        "gabor",
        {"gabor_type": "3D"},
        "constant_edge",
        "Value",
        index,
    )
)
gn_field = build_geometry_field(scene)
configure_render(scene)
add_camera(scene)

# The object meshes are one unit square and the orthographic camera spans the 16-column grid.
# Four pixels per tile keep every center probe clear of the neighboring material boundary.
blend_parent = os.path.dirname(BLEND_OUT)
os.makedirs(blend_parent, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT, check_existing=False)

renders = {}
for engine_key, blender_engine in (("cycles", "CYCLES"), ("eevee", "BLENDER_EEVEE")):
    scene.render.engine = blender_engine
    if engine_key == "cycles":
        scene.cycles.device = "CPU"
    renders[engine_key] = {
        "blender_engine": blender_engine,
        "resolution": [COLUMNS * PIXELS_PER_TILE, ROWS * PIXELS_PER_TILE],
        "pixels": render_probe_pixels(scene, probes),
    }
world_render = build_world_fixture()

with open(ORACLE_OUT, "w", encoding="utf-8") as output:
    json.dump(
        {
            "schema": "shader-procedural-textures-oracle/1",
            "blender_version": bpy.app.version_string,
            "color_settings": {
                "view_transform": "Raw",
                "look": "None",
                "exposure": 0.0,
                "gamma": 1.0,
            },
            "tile_pixels": PIXELS_PER_TILE,
            "columns": COLUMNS,
            "rows": ROWS,
            "probes": probes,
            "renders": renders,
            "gn_field": gn_field,
            "world_render": world_render,
        },
        output,
        indent=2,
    )
build_mapping_fixture()
print(f"PROBES {len(probes)}; RENDERS {len(renders)}")
