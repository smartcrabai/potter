"""Blender oracle scenes for tests/blend_collection_instances.rs.

Run as: blender --background --factory-startup --python blend_collection_instances.py -- <root> <case>

Writes the original quad scene as `<case>.blend`, plus `<case>-faces.blend` for Blender face-loop
checks. Only the latter drops the missing-UV material's image node, since Potter's blend bridge
cannot export that external image reference; its source GLB still covers the textured refusal. A
geometry-only copy strips colors, groups, and seams, triangulates a copy of each mesh, and feeds the
GLB triangle-corner multiset comparison.
"""

import math
import os
import sys

import bmesh
import bpy

ARGUMENTS = sys.argv[sys.argv.index("--") + 1:]
ROOT = os.path.realpath(ARGUMENTS[0])
CASE = ARGUMENTS[1]

CORNERS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]
FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
RED = (0.8, 0.1, 0.1, 1.0)
BLUE = (0.1, 0.2, 0.9, 1.0)


def unit(vector):
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector)


def varied_normal(index):
    return (0.2 + 0.1 * (index % 3), -0.3 + 0.2 * ((index // 3) % 2), 0.9)


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
    return mat


def checker_image(path):
    image = bpy.data.images.new("checker", width=2, height=2)
    image.pixels = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    return image


def textured_material(name, image):
    mat = material(name, (1.0, 1.0, 1.0, 1.0))
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    mat.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def cube(name, materials=(), location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
         scale=(1.0, 1.0, 1.0), size=1.0, uv_layers=("UVMap",)):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata([(x * size, y * size, z * size) for x, y, z in CORNERS], [], FACES)
    mesh.update()
    for mat in materials:
        mesh.materials.append(mat)
    for layer_index, layer_name in enumerate(uv_layers):
        layer = mesh.uv_layers.new(name=layer_name)
        for loop in mesh.loops:
            co = mesh.vertices[loop.vertex_index].co
            layer.data[loop.index].uv = (
                co.x * 0.5 + 0.5 + 0.25 * layer_index,
                co.z * 0.5 + 0.5 - 0.1 * layer_index,
            )
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    return obj


def smooth(obj):
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def custom_normals(obj, vector_for):
    obj.data.normals_split_custom_set(
        [unit(vector_for(index)) for index in range(len(obj.data.loops))]
    )


def place_instance(target, name, source, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
                   scale=(1.0, 1.0, 1.0), potter_id=None):
    empty = bpy.data.objects.new(name, None)
    empty.instance_type = "COLLECTION"
    empty.instance_collection = source
    empty.location = location
    empty.rotation_euler = rotation
    empty.scale = scale
    if potter_id is not None:
        empty["potter.id"] = potter_id
    target.objects.link(empty)
    return empty


def nested_instances(scene):
    jacket_mat = material("jacket", (0.8, 0.1, 0.1, 1.0))
    skin_mat = material("skin", (0.9, 0.7, 0.5, 1.0))
    members = bpy.data.collections.new("Members")
    outer = bpy.data.collections.new("Outer")
    scene.collection.children.link(outer)

    jacket = cube("Jacket", [jacket_mat, skin_mat])
    for polygon in jacket.data.polygons:
        polygon.material_index = 1 if polygon.index == 1 else 0
    members.objects.link(jacket)

    skin = cube("Skin", [skin_mat], location=(2.5, 0.0, 0.0))
    smooth(skin)
    skin.data.normals_split_custom_set([(0.0, 0.6, 0.8)] * len(skin.data.loops))
    for edge in skin.data.edges:
        edge.use_seam = edge.index < 4
    color = skin.data.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    for index, datum in enumerate(color.data):
        datum.color = (0.1 * index % 1.0, 0.5, 0.25, 1.0)
    group = skin.vertex_groups.new(name="Weight")
    group.add(list(range(8)), 0.5, "REPLACE")
    members.objects.link(skin)

    place_instance(outer, "InnerInstance", members)
    outer.objects.link(cube("Hat", [skin_mat], location=(0.0, 3.0, 0.0), size=0.5))
    place_instance(scene.collection, "Instance", outer, location=(0.0, 0.0, 5.0),
                   rotation=(0.0, 0.0, 0.5235987755982988), scale=(-1.0, 1.0, 1.0))


def rotated_member(scene):
    members = bpy.data.collections.new("Members")
    members.objects.link(cube("Turned", [material("red", RED)],
                              location=(1.2, 0.4, -0.3), rotation=(0.37, -0.81, 1.23)))
    members.objects.link(cube("Level", [material("blue", BLUE)], location=(-2.0, 0.0, 0.0)))
    place_instance(scene.collection, "Instance", members,
                   location=(0.5, 0.0, 0.0), rotation=(0.0, 0.0, 0.4))


def mirrored_member(scene):
    members = bpy.data.collections.new("Members")
    members.objects.link(cube("Mirror", [material("red", RED)], location=(2.0, 0.0, 0.0),
                              rotation=(0.0, 0.5, 0.0), scale=(-1.0, 1.0, 1.0)))
    members.objects.link(cube("Plain", [material("blue", BLUE)], location=(-1.5, 0.5, 0.0)))
    place_instance(scene.collection, "Instance", members)


def uv_layer_names(scene):
    members = bpy.data.collections.new("Members")
    red = material("red", RED)
    blue = material("blue", BLUE)
    members.objects.link(cube("Jacket", [red], uv_layers=("UVMap",)))
    members.objects.link(cube("Skin", [blue], location=(2.5, 0.0, 0.0),
                              uv_layers=("Body", "Atlas")))
    place_instance(scene.collection, "Instance", members)


def potter_unwrap_mixed(scene):
    members = bpy.data.collections.new("Members")
    members.objects.link(cube("Blend", [material("red", RED)], uv_layers=("UVMap",)))
    place_instance(scene.collection, "Instance", members)


def missing_uv_member(scene):
    members = bpy.data.collections.new("Members")
    members.objects.link(cube("Mapped", [material("red", RED)], uv_layers=("UVMap",)))
    members.objects.link(cube("Bare", [material("blue", BLUE)], location=(2.5, 0.0, 0.0),
                              uv_layers=()))
    place_instance(scene.collection, "Instance", members)


def missing_uv_textured(scene):
    members = bpy.data.collections.new("Members")
    image = checker_image(os.path.join(ROOT, "checker.png"))
    members.objects.link(cube("Mapped", [material("red", RED)], uv_layers=("UVMap",)))
    members.objects.link(cube("Textured", [textured_material("textured", image)],
                              location=(2.5, 0.0, 0.0), uv_layers=()))
    place_instance(scene.collection, "Instance", members)


def mixed_smooth(scene):
    members = bpy.data.collections.new("Members")
    custom = cube("Custom", [material("red", RED)])
    smooth(custom)
    custom_normals(custom, varied_normal)
    poly = cube("Poly", [material("blue", BLUE)], location=(2.5, 0.0, 0.0))
    smooth(poly)
    members.objects.link(custom)
    members.objects.link(poly)
    members.objects.link(cube("Flat", [material("red", RED)], location=(5.0, 0.0, 0.0)))
    place_instance(scene.collection, "Instance", members, location=(0.0, 0.0, -3.0))


def custom_normals_rotated(scene):
    members = bpy.data.collections.new("Members")
    turned = cube("Turned", [material("red", RED)], location=(1.0, 0.5, 0.0),
                  rotation=(0.2, -0.9, 1.1))
    smooth(turned)
    custom_normals(turned, varied_normal)
    level = cube("Level", [material("blue", BLUE)], location=(-2.0, 0.0, 0.0),
                 rotation=(0.0, 0.3, 0.0))
    smooth(level)
    custom_normals(level, lambda index: (0.9, 0.1 * (index % 4), 0.3 - 0.1 * (index % 2)))
    members.objects.link(turned)
    members.objects.link(level)
    place_instance(scene.collection, "Instance", members, rotation=(0.0, 0.0, 1.0))


def out_of_range_material(scene):
    # Empty (None) slot preservation is covered once Node.materials carries Option entries (PR17a).
    members = bpy.data.collections.new("Members")
    ranged = cube("Ranged", [material("red", RED), material("blue", BLUE)])
    for polygon in ranged.data.polygons:
        polygon.material_index = 0 if polygon.index < 3 else 5
    members.objects.link(ranged)
    members.objects.link(cube("Plain", [], location=(2.5, 0.0, 0.0)))
    place_instance(scene.collection, "Instance", members)


def color_type_conflict(scene):
    members = bpy.data.collections.new("Members")
    byte = cube("ByteColor", [material("red", RED)])
    byte_color = byte.data.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    for index, datum in enumerate(byte_color.data):
        datum.color = (0.2, 0.4 + 0.01 * index, 0.6, 1.0)
    flat = cube("FloatColor", [material("blue", BLUE)], location=(2.5, 0.0, 0.0))
    float_color = flat.data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    for index, datum in enumerate(float_color.data):
        datum.color = (0.9, 0.1 + 0.01 * index, 0.3, 1.0)
    members.objects.link(byte)
    members.objects.link(flat)
    place_instance(scene.collection, "Instance", members)


def potter_id_reimport(scene):
    members = bpy.data.collections.new("Members")
    alpha = cube("Alpha", [material("red", RED)])
    alpha["potter.id"] = "member_alpha"
    alpha.data["potter.id"] = "mesh_alpha"
    beta = cube("Beta", [material("blue", BLUE)], location=(2.5, 0.0, 0.0))
    beta["potter.id"] = "member_beta"
    beta.data["potter.id"] = "mesh_beta"
    members.objects.link(alpha)
    members.objects.link(beta)
    place_instance(scene.collection, "Instance", members, potter_id="instance_main")


def metadata_conflict(scene):
    clash_members = bpy.data.collections.new("ClashMembers")
    left = cube("Left", [material("red", RED)])
    left.data["note"] = "a"
    right = cube("Right", [material("red", RED)], location=(2.5, 0.0, 0.0))
    right.data["note"] = "b"
    clash_members.objects.link(left)
    clash_members.objects.link(right)
    clean_members = bpy.data.collections.new("CleanMembers")
    clean_members.objects.link(cube("Solo", [material("blue", BLUE)], location=(0.0, 0.0, 4.0)))
    place_instance(scene.collection, "ClashInstance", clash_members,
                   potter_id="clash_instance")
    place_instance(scene.collection, "CleanInstance", clean_members,
                   location=(0.0, 0.0, -4.0), potter_id="clean_instance")


CASES = {
    "nested_instances": nested_instances,
    "rotated_member": rotated_member,
    "mirrored_member": mirrored_member,
    "uv_layer_names": uv_layer_names,
    "potter_unwrap_mixed": potter_unwrap_mixed,
    "missing_uv_member": missing_uv_member,
    "missing_uv_textured": missing_uv_textured,
    "mixed_smooth": mixed_smooth,
    "custom_normals_rotated": custom_normals_rotated,
    "out_of_range_material": out_of_range_material,
    "color_type_conflict": color_type_conflict,
    "potter_id_reimport": potter_id_reimport,
    "metadata_conflict": metadata_conflict,
}

def strip_attributes():
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        while len(mesh.color_attributes):
            mesh.color_attributes.remove(mesh.color_attributes[0])
        for group in list(obj.vertex_groups):
            obj.vertex_groups.remove(group)
        for edge in mesh.edges:
            edge.use_seam = False


def triangulate_geometry():
    for mesh in bpy.data.meshes:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
CASES[CASE](scene)
full_scene = os.path.join(ROOT, CASE + ".blend")
bpy.ops.wm.save_as_mainfile(filepath=full_scene, check_existing=False)
if CASE == "missing_uv_textured":
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in list(material.node_tree.nodes):
                if node.type == "TEX_IMAGE":
                    material.node_tree.nodes.remove(node)
bpy.ops.wm.save_as_mainfile(
    filepath=os.path.join(ROOT, CASE + "-faces.blend"),
    check_existing=False,
)
bpy.ops.wm.open_mainfile(filepath=full_scene)
strip_attributes()
triangulate_geometry()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, CASE + "-geometry.blend"), check_existing=False)
bpy.ops.export_scene.gltf(
    filepath=os.path.join(ROOT, CASE + "-geometry.blender.glb"),
    export_format="GLB",
)
