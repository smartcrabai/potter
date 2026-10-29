import json
import sys

import bpy
if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")

arguments = sys.argv[sys.argv.index("--") + 1 :]
blend_path, oracle_path, exr_path = arguments[:3]
probe_paths = arguments[3:]
if probe_paths and len(probe_paths) != 3:
    raise ValueError("expected optional Cycles probe paths: blend, oracle, EXR")
bpy.ops.wm.read_factory_settings(use_empty=True)


def material_with_nodes(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emission.outputs[0], output.inputs[0])
    return material, tree, emission

def generate_probe_oracle(probe_paths):
    probe_blend_path, probe_oracle_path, probe_exr_path = probe_paths
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.max_bounces = 3
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.render.resolution_x = 12
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    scene.world = None

    camera_data = bpy.data.cameras.new("Cycles Converter Probe Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 6.0
    camera = bpy.data.objects.new("Cycles Converter Probe Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 5.0)
    scene.camera = camera

    def encode_vector(tree, source, socket, emission):
        add = tree.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (1.0, 1.0, 1.0)
        scale = tree.nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs["Scale"].default_value = 0.5
        tree.links.new(source.outputs[socket], add.inputs[0])
        tree.links.new(add.outputs["Vector"], scale.inputs[0])
        tree.links.new(scale.outputs["Vector"], emission.inputs["Color"])

    incoming_material, incoming_tree, incoming_emission = material_with_nodes("Geometry Incoming")
    incoming_node = incoming_tree.nodes.new("ShaderNodeNewGeometry")
    encode_vector(incoming_tree, incoming_node, "Incoming", incoming_emission)

    reflection_material, reflection_tree, reflection_emission = material_with_nodes("TexCoord Reflection")
    texcoord = reflection_tree.nodes.new("ShaderNodeTexCoord")
    encode_vector(reflection_tree, texcoord, "Reflection", reflection_emission)
    hsv_material, hsv_tree, hsv_emission = material_with_nodes("Cycles HSV Overdrive")
    hue_saturation = hsv_tree.nodes.new("ShaderNodeHueSaturation")
    hue_saturation.inputs["Color"].default_value = (0.3, 0.1, 0.1, 1.0)
    hue_saturation.inputs["Saturation"].default_value = 1.0
    hue_saturation.inputs["Value"].default_value = -1.0
    hsv_tree.links.new(hue_saturation.outputs["Color"], hsv_emission.inputs["Color"])

    normal_material, normal_tree, normal_emission = material_with_nodes(
        "Object Normal Map Backface"
    )
    normal_map = normal_tree.nodes.new("ShaderNodeNormalMap")
    normal_map.space = "OBJECT"
    normal_map.inputs["Color"].default_value = (0.75, 0.25, 0.9, 1.0)
    add = normal_tree.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (1.0, 1.0, 1.0)
    scale = normal_tree.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 0.5
    normal_tree.links.new(normal_map.outputs["Normal"], add.inputs[0])
    normal_tree.links.new(add.outputs["Vector"], scale.inputs[0])
    normal_tree.links.new(scale.outputs["Vector"], normal_emission.inputs["Color"])

    depth_material, depth_tree, depth_emission = material_with_nodes("Camera Data Z Depth")
    camera_data_node = depth_tree.nodes.new("ShaderNodeCameraData")
    depth_offset = depth_tree.nodes.new("ShaderNodeMath")
    depth_offset.operation = "ADD"
    depth_offset.inputs["Value_001"].default_value = 6.0
    depth_tree.links.new(camera_data_node.outputs["View Z Depth"], depth_offset.inputs["Value"])
    depth_tree.links.new(depth_offset.outputs["Value"], depth_emission.inputs["Strength"])

    secondary_depth_material, secondary_depth_tree, secondary_depth_emission = material_with_nodes(
        "Camera Data Secondary Z Depth"
    )
    secondary_camera_data = secondary_depth_tree.nodes.new("ShaderNodeCameraData")
    secondary_depth_offset = secondary_depth_tree.nodes.new("ShaderNodeMath")
    secondary_depth_offset.operation = "ADD"
    secondary_depth_offset.inputs["Value_001"].default_value = 2.0
    secondary_depth_tree.links.new(
        secondary_camera_data.outputs["View Z Depth"],
        secondary_depth_offset.inputs["Value"],
    )
    secondary_depth_tree.links.new(
        secondary_depth_offset.outputs["Value"],
        secondary_depth_emission.inputs["Strength"],
    )
    camera_bump_material, camera_bump_tree, camera_bump_emission = material_with_nodes(
        "Camera Data Bump Height"
    )
    camera_bump_camera = camera_bump_tree.nodes.new("ShaderNodeCameraData")
    camera_bump_scale = camera_bump_tree.nodes.new("ShaderNodeMath")
    camera_bump_scale.operation = "MULTIPLY"
    camera_bump_scale.inputs["Value_001"].default_value = 10.0
    camera_bump = camera_bump_tree.nodes.new("ShaderNodeBump")
    camera_bump.inputs["Distance"].default_value = 0.5
    camera_bump_normal = camera_bump_tree.nodes.new("ShaderNodeSeparateXYZ")
    camera_bump_tree.links.new(
        camera_bump_camera.outputs["View Z Depth"], camera_bump_scale.inputs["Value"]
    )
    camera_bump_tree.links.new(
        camera_bump_scale.outputs["Value"], camera_bump.inputs["Height"]
    )
    camera_bump_tree.links.new(
        camera_bump.outputs["Normal"], camera_bump_normal.inputs["Vector"]
    )
    camera_bump_tree.links.new(
        camera_bump_normal.outputs["Z"], camera_bump_emission.inputs["Strength"]
    )
    geometry_bump_material, geometry_bump_tree, geometry_bump_emission = material_with_nodes(
        "Geometry Normal Bump Height"
    )
    geometry_bump_geometry = geometry_bump_tree.nodes.new("ShaderNodeNewGeometry")
    geometry_bump_z = geometry_bump_tree.nodes.new("ShaderNodeSeparateXYZ")
    geometry_bump = geometry_bump_tree.nodes.new("ShaderNodeBump")
    geometry_bump_normal = geometry_bump_tree.nodes.new("ShaderNodeSeparateXYZ")
    geometry_bump_tree.links.new(
        geometry_bump_geometry.outputs["Normal"], geometry_bump_z.inputs["Vector"]
    )
    geometry_bump_tree.links.new(
        geometry_bump_z.outputs["Z"], geometry_bump.inputs["Height"]
    )
    geometry_bump_tree.links.new(
        geometry_bump.outputs["Normal"], geometry_bump_normal.inputs["Vector"]
    )
    geometry_bump_tree.links.new(
        geometry_bump_normal.outputs["Z"], geometry_bump_emission.inputs["Strength"]
    )

    mirror_material = bpy.data.materials.new("Secondary Mirror")
    mirror_material.use_nodes = True
    mirror_tree = mirror_material.node_tree
    mirror_tree.nodes.clear()
    mirror_output = mirror_tree.nodes.new("ShaderNodeOutputMaterial")
    mirror_bsdf = mirror_tree.nodes.new("ShaderNodeBsdfGlossy")
    mirror_bsdf.inputs["Roughness"].default_value = 0.0
    mirror_tree.links.new(mirror_bsdf.outputs["BSDF"], mirror_output.inputs["Surface"])

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-1.25, 0.0, 0.0))
    mirror_plane = bpy.context.object
    mirror_plane.name = "Secondary Mirror"
    mirror_plane.scale = (0.4, 1.0, 1.0)
    mirror_plane.data.materials.append(mirror_material)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-1.25, 0.0, 6.0))
    secondary_plane = bpy.context.object
    secondary_plane.name = "Behind Camera Depth Target"
    secondary_plane.scale = (0.4, 1.0, 1.0)
    secondary_plane.data.materials.append(secondary_depth_material)
    probes = [
        ("incoming", incoming_material, -1.75, False),
        ("reflection", reflection_material, -0.25, False),
        ("hsv_overdrive", hsv_material, 0.25, False),
        ("camera_z_depth", depth_material, 0.75, False),
        ("normal_map_object_backface", normal_material, 1.75, True),
        ("camera_data_bump", camera_bump_material, 2.25, False),
        ("geometry_normal_bump", geometry_bump_material, -0.75, False),
    ]
    for name, material, center_x, backface in probes:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(center_x, 0.0, 0.0))
        plane = bpy.context.object
        plane.name = name
        plane.scale = (0.4, 1.0, 1.0)
        if name == "reflection":
            plane.rotation_euler = (0.0, 0.7853981633974483, 0.0)
        if name == "camera_data_bump":
            plane.rotation_euler = (0.0, 0.5235987755982988, 0.0)
        if backface:
            plane.rotation_euler = (3.141592653589793, 0.0, 0.0)
        plane.data.materials.append(material)

    bpy.ops.wm.save_as_mainfile(filepath=probe_blend_path, check_existing=False)
    scene.render.filepath = probe_exr_path
    bpy.ops.render.render(write_still=True)
    rendered = bpy.data.images.load(probe_exr_path, check_existing=False)
    probe_pixels = list(rendered.pixels)
    pixel_x = {
        "incoming": 2,
        "reflection": 5,
        "hsv_overdrive": 6,
        "camera_z_depth": 7,
        "normal_map_object_backface": 9,
        "geometry_normal_bump": 4,
        "camera_data_bump": 10,
    }
    probe_oracle = {
        "engine": "Blender 5.2.2 Cycles",
        "resolution": [12, 12],
        "probes": {
            name: {
                "uv": [(pixel_x[name] + 0.5) / 12.0, 6.5 / 12.0],
                "rgba": probe_pixels[(6 * 12 + pixel_x[name]) * 4 : (6 * 12 + pixel_x[name]) * 4 + 4],
            }
            for name, _, _, _ in probes
        },
    }
    front_depth = probe_oracle["probes"]["camera_z_depth"]["rgba"][0] - 6.0
    probe_oracle["probes"]["camera_z_depth"]["view_z_depth"] = front_depth
    probe_oracle["probes"]["camera_z_depth"]["camera_space_z"] = -camera.location.z
    secondary_offset = (6 * 12 + 3) * 4
    secondary_rgba = probe_pixels[secondary_offset : secondary_offset + 4]
    secondary_view_z_depth = secondary_rgba[0] - 2.0
    probe_oracle["probes"]["camera_z_depth_secondary"] = {
        "uv": [3.5 / 12.0, 6.5 / 12.0],
        "rgba": secondary_rgba,
        "view_z_depth": secondary_view_z_depth,
        "camera_space_z": -secondary_view_z_depth,
    }
    with open(probe_oracle_path, "w", encoding="utf-8") as stream:
        json.dump(probe_oracle, stream, indent=2)
    return
if probe_paths:
    generate_probe_oracle(probe_paths)
    sys.exit(0)


color_material, tree, emission = material_with_nodes("Color Converters")
rgb = tree.nodes.new("ShaderNodeRGB")
rgb.outputs[0].default_value = (0.2, 0.4, 0.6, 1.0)
invert = tree.nodes.new("ShaderNodeInvert")
gamma = tree.nodes.new("ShaderNodeGamma")
gamma.inputs["Gamma"].default_value = 2.0
bright_contrast = tree.nodes.new("ShaderNodeBrightContrast")
bright_contrast.inputs["Brightness"].default_value = 0.1
bright_contrast.inputs["Contrast"].default_value = 0.5
hue_saturation = tree.nodes.new("ShaderNodeHueSaturation")
value = tree.nodes.new("ShaderNodeValue")
value.outputs[0].default_value = 0.0
separate_hsv = tree.nodes.new("ShaderNodeSeparateColor")
separate_hsv.mode = "HSV"
combine_hsv = tree.nodes.new("ShaderNodeCombineColor")
combine_hsv.mode = "HSV"
separate_hsl = tree.nodes.new("ShaderNodeSeparateColor")
separate_hsl.mode = "HSL"
combine_hsl = tree.nodes.new("ShaderNodeCombineColor")
combine_hsl.mode = "HSL"
rgb_to_bw = tree.nodes.new("ShaderNodeRGBToBW")
links = tree.links
links.new(rgb.outputs[0], invert.inputs[1])
links.new(invert.outputs[0], gamma.inputs[0])
links.new(gamma.outputs[0], bright_contrast.inputs[0])
links.new(bright_contrast.outputs[0], hue_saturation.inputs[4])
links.new(value.outputs[0], hue_saturation.inputs[3])
links.new(hue_saturation.outputs[0], separate_hsv.inputs[0])
for index in range(3):
    links.new(separate_hsv.outputs[index], combine_hsv.inputs[index])
links.new(combine_hsv.outputs[0], separate_hsl.inputs[0])
for index in range(3):
    links.new(separate_hsl.outputs[index], combine_hsl.inputs[index])
links.new(combine_hsl.outputs[0], emission.inputs[0])
links.new(combine_hsl.outputs[0], rgb_to_bw.inputs[0])
links.new(rgb_to_bw.outputs[0], emission.inputs[1])

blackbody_material, blackbody_tree, blackbody_emission = material_with_nodes("Blackbody")
blackbody = blackbody_tree.nodes.new("ShaderNodeBlackbody")
blackbody.inputs[0].default_value = 3200.0
blackbody_tree.links.new(blackbody.outputs[0], blackbody_emission.inputs[0])

wavelength_material, wavelength_tree, wavelength_emission = material_with_nodes("Wavelength")
wavelength = wavelength_tree.nodes.new("ShaderNodeWavelength")
wavelength.inputs[0].default_value = 540.0
wavelength_tree.links.new(wavelength.outputs[0], wavelength_emission.inputs[0])

falloff_material, falloff_tree, falloff_emission = material_with_nodes("Light Falloff")
falloff = falloff_tree.nodes.new("ShaderNodeLightFalloff")
falloff.inputs["Strength"].default_value = 2.0
falloff.inputs["Smooth"].default_value = 0.5
falloff_tree.links.new(falloff.outputs["Constant"], falloff_emission.inputs["Strength"])

object_color_material, object_color_tree, object_color_emission = material_with_nodes(
    "Object Info Color"
)
object_color_info = object_color_tree.nodes.new("ShaderNodeObjectInfo")
object_color_tree.links.new(object_color_info.outputs["Color"], object_color_emission.inputs["Color"])
object_color_tree.links.new(object_color_info.outputs["Alpha"], object_color_emission.inputs["Strength"])

object_index_material, object_index_tree, object_index_emission = material_with_nodes(
    "Object Info Object Index"
)
object_index_info = object_index_tree.nodes.new("ShaderNodeObjectInfo")
object_index_tree.links.new(object_index_info.outputs["Object Index"], object_index_emission.inputs["Strength"])

material_index_material, material_index_tree, material_index_emission = material_with_nodes(
    "Object Info Material Index"
)
material_index_material.pass_index = 23
material_index_info = material_index_tree.nodes.new("ShaderNodeObjectInfo")
material_index_tree.links.new(
    material_index_info.outputs["Material Index"], material_index_emission.inputs["Strength"])

random_material, random_tree, random_emission = material_with_nodes("Object Info Random")
random_info = random_tree.nodes.new("ShaderNodeObjectInfo")
random_tree.links.new(random_info.outputs["Random"], random_emission.inputs["Strength"])

location_material, location_tree, location_emission = material_with_nodes("Object Info Location")
location_info = location_tree.nodes.new("ShaderNodeObjectInfo")
vector_length = location_tree.nodes.new("ShaderNodeVectorMath")
vector_length.operation = "LENGTH"
location_tree.links.new(location_info.outputs["Location"], vector_length.inputs[0])
location_tree.links.new(vector_length.outputs["Value"], location_emission.inputs["Strength"])

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.max_bounces = 1
scene.cycles.use_denoising = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01
scene.render.resolution_x = 9
scene.render.resolution_y = 4
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.view_settings.view_transform = "Standard"
scene.world = None
camera_data = bpy.data.cameras.new("Oracle Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 9.0
camera = bpy.data.objects.new("Oracle Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 5.0)
scene.camera = camera

materials = [
    color_material,
    blackbody_material,
    wavelength_material,
    falloff_material,
    object_color_material,
    object_index_material,
    material_index_material,
    random_material,
    location_material,
]
for x, material in zip(range(-4, 5), materials):
    bpy.ops.mesh.primitive_plane_add(size=1.2, location=(float(x), 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "Random Target" if material is random_material else material.name + " Plane"
    if material is object_color_material:
        plane.color = (0.2, 0.4, 0.6, 0.8)
    if material is object_index_material:
        plane.pass_index = 17
    if material is random_material:
        plane.pass_index = 29
    plane.data.materials.append(material)

bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
bpy.ops.render.render(write_still=False)
bpy.data.images["Render Result"].save_render(exr_path)
rendered = bpy.data.images.load(exr_path, check_existing=False)
pixels = list(rendered.pixels)
sample_indices = [(1 * 9 + x) * 4 for x in range(9)]
with open(oracle_path, "w", encoding="utf-8") as stream:
    json.dump(
        {"pixels": [pixels[index : index + 4] for index in sample_indices]},
        stream,
    )

