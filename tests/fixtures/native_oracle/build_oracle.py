"""PR9 native-data oracle generator (Blender 5.2.2, background).

Usage:
  Blender --factory-startup -b --python build_oracle.py -- OUT_DIR

Writes into OUT_DIR:
  oracle.blend   the scene Potter imports
  expected.json  explicit numbers written by this script (never read back from Potter)

Scene:
  Oracle   a 2x2-per-face cube (26 vertices, 24 faces, 48 edges) with mixed flat/smooth faces,
           sharp and seam edges, and authored/automatic status per loop-normal fan (not per isolated
           corner). Faces can mix status where a sharp edge splits fans, 3 UV layers with active and
           render different, 4 color layers (POINT/CORNER x FLOAT_COLOR/BYTE_COLOR) with active and
           render different, user generic attributes on every domain, two vertex groups, and
           material "Surface" whose Principled base color is multiplied by the Vertex Color node
           reading the non-render layer "Paint".
  Regress  the same cube topology, all smooth, partial custom normals on faces 0..11, no
           material. `custom_normal` (Auto/authored status rows) and `custom_normal_codes` (raw Blender
           code pairs, (0, 0) is Auto) hold its per-corner state. `regress.reference_corners` in
           expected.json holds the corners Blender produces after a Subdivision (level 1, Use Custom
           Normals on) and a Displace (strength 0.1, mid level 0) modifier.
  Plain    one quad with material "Bare" (no color layers, no vertex color node).

  TexturePlane a separate square with two UV layers, a 4x4 asymmetric multirow image, and a
           Blender-authored reference material saved as `reference.blend` in the output directory.

BYTE_COLOR values are written as exact sRGB bytes converted to linear, so Blender's byte storage
round-trips them exactly. Alpha is linear byte storage (value = byte / 255).
"""

import json
import math
import os
import struct
import sys
import zlib

import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)

REGION_FACES = {"Oracle": 6, "Regress": 12}
UV_NAMES = ["Base", "Light", "Detail"]
UV_ACTIVE = 2
UV_RENDER = 1
COLOR_ACTIVE = 1
COLOR_RENDER = 3


def round4(value):
    return round(value, 4)


def srgb_to_linear(value):
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def byte_linear(byte):
    return srgb_to_linear(byte / 255.0)


def grid_cube(cuts):
    """Cube [-1, 1]^3 with `cuts` x `cuts` quads per face. Quads wind counter-clockwise outward."""
    verts = []
    index = {}
    faces = []

    def vertex(point):
        key = tuple(round(c, 9) for c in point)
        if key not in index:
            index[key] = len(verts)
            verts.append(tuple(float(c) for c in point))
        return index[key]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            a1, a2 = [k for k in range(3) if k != axis]
            normal = [0.0, 0.0, 0.0]
            normal[axis] = sign
            for i in range(cuts):
                for j in range(cuts):
                    u0 = -1.0 + 2.0 * i / cuts
                    u1 = -1.0 + 2.0 * (i + 1) / cuts
                    v0 = -1.0 + 2.0 * j / cuts
                    v1 = -1.0 + 2.0 * (j + 1) / cuts
                    quad = []
                    for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                        point = [0.0, 0.0, 0.0]
                        point[axis] = sign
                        point[a1] = u
                        point[a2] = v
                        quad.append(point)
                    c0, c1, c2 = (Vector(p) for p in quad[:3])
                    if (c1 - c0).cross(c2 - c0).dot(Vector(normal)) < 0.0:
                        quad.reverse()
                    faces.append([vertex(p) for p in quad])
    return verts, faces


def make_object(name, verts, faces, location=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def unit(vector):
    length = math.sqrt(sum(c * c for c in vector))
    return tuple(c / length for c in vector)


def set_shading(mesh, smooth_for_face):
    for poly in mesh.polygons:
        poly.use_smooth = smooth_for_face(poly.index)
    for edge in mesh.edges:
        a, b = sorted(edge.vertices)
        edge.use_edge_sharp = (a + b) % 5 == 0
        edge.use_seam = (a * 3 + b) % 7 == 3


def authored_loop(loop_index):
    return loop_index % 3 != 0


def authored_vector(loop_index):
    x = 0.2 + 0.07 * ((loop_index * 5) % 11)
    y = -0.3 + 0.05 * ((loop_index * 7) % 9)
    z = 0.9 - 0.03 * ((loop_index * 3) % 7)
    return unit((x, y, z))


def corner_fans(mesh):
    """Blender's loop-normal spaces. Corners around a vertex join one fan across every non-sharp
    edge whose two faces are both smooth; a flat face's corners stand alone. Returns the fan root
    of every loop and the face index of every loop."""
    count = len(mesh.loops)
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    loop_face = [0] * count
    groups = {}
    for poly in mesh.polygons:
        for k in range(poly.loop_total):
            li = poly.loop_start + k
            loop_face[li] = poly.index
            previous = poly.loop_start + (k - 1) % poly.loop_total
            loop = mesh.loops[li]
            groups.setdefault((loop.vertex_index, loop.edge_index), []).append(li)
            groups.setdefault((loop.vertex_index, mesh.loops[previous].edge_index), []).append(li)
    for (vertex, edge), members in groups.items():
        if mesh.edges[edge].use_edge_sharp:
            continue
        if not all(mesh.polygons[loop_face[li]].use_smooth for li in members):
            continue
        for other in members[1:]:
            root_a, root_b = find(members[0]), find(other)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    return [find(li) for li in range(count)], loop_face


def set_partial_custom_normals(mesh, region_faces):
    """Authored fans get `authored_vector`; every other fan is automatic. A fan is authored when
    its first loop lies in a region face and has index % 3 != 0. Blender stores custom normals per
    fan, so all loops of a fan share one status. Automatic fans get code pair (0, 0), written
    directly (Blender's encoder does not map an exact fan normal to (0, 0)), and must then follow
    the fan normal computed before any custom normal existed. Returns {loop index: vector}."""
    mesh.update()
    base = [tuple(n.vector) for n in mesh.corner_normals]
    roots, loop_face = corner_fans(mesh)
    members_by_root = {}
    for loop_index, root in enumerate(roots):
        members_by_root.setdefault(root, []).append(loop_index)
    vectors = list(base)
    authored = {}
    automatic = []
    for members in members_by_root.values():
        first = min(members)
        if loop_face[first] < region_faces and authored_loop(first):
            vector = authored_vector(first)
            for loop_index in members:
                authored[loop_index] = vector
                vectors[loop_index] = vector
        else:
            automatic.extend(members)
    mesh.normals_split_custom_set(vectors)
    codes_attribute = mesh.attributes["custom_normal"]
    for loop_index in automatic:
        codes_attribute.data[loop_index].value = (0, 0)
    mesh.update()
    codes = [tuple(item.value) for item in mesh.attributes["custom_normal"].data]
    after = [tuple(n.vector) for n in mesh.corner_normals]
    for loop_index, code in enumerate(codes):
        if loop_index in authored:
            if code == (0, 0):
                raise SystemExit("authored loop %d encoded as automatic" % loop_index)
            expected = authored[loop_index]
        else:
            if code != (0, 0):
                raise SystemExit("automatic loop %d kept code %r" % (loop_index, code))
            expected = base[loop_index]
        if max(abs(a - b) for a, b in zip(after[loop_index], expected)) > 1e-3:
            raise SystemExit("loop %d normal %r differs from %r" % (loop_index, after[loop_index], expected))
    return authored


def custom_normal_rows(mesh, authored):
    rows = []
    for poly in mesh.polygons:
        row = []
        for k in range(poly.loop_total):
            loop_index = poly.loop_start + k
            if loop_index in authored:
                row.append(list(authored[loop_index]))
            else:
                row.append(None)
        rows.append(row)
    return rows


def custom_normal_codes(mesh):
    """Raw `custom_normal` INT16_2D pairs per face corner, exactly as Blender stores them. (0, 0)
    is Auto; an authored fan holds its nonzero pair."""
    data = mesh.attributes["custom_normal"].data
    return [
        [[int(c) for c in data[poly.loop_start + k].value] for k in range(poly.loop_total)]
        for poly in mesh.polygons
    ]


def loop_count(mesh):
    return len(mesh.loops)


def color_point_value(index):
    return [
        round4(0.0625 + 0.0525 * ((index * 7) % 13)),
        round4(0.9 - 0.0375 * ((index * 5) % 11)),
        round4(0.2 + 0.0125 * ((index * 3) % 17)),
        round4(0.25 + 0.75 * ((index % 4) / 3.0)),
    ]


def color_corner_value(loop_index):
    r = round4(0.031 + 0.0457 * ((loop_index * 11) % 23))
    if loop_index == 5:
        r = 1.5  # HDR value: stored unclamped
    return [
        r,
        round4(0.5 + 0.4 * math.sin(loop_index)),
        round4(0.05 + 0.0231 * ((loop_index * 13) % 31)),
        round4(0.4 + 0.5 * ((loop_index % 5) / 4.0)),
    ]


def byte_point_value(index):
    rgb = [(index * 53 + 17) % 256, (index * 91 + 3) % 256, (index * 29 + 101) % 256]
    alpha = 255 if index % 3 == 0 else (index * 17 + 40) % 256
    return rgb, alpha


def byte_corner_value(loop_index):
    rgb = [(loop_index * 47 + 9) % 256, (loop_index * 23 + 77) % 256, (loop_index * 61 + 31) % 256]
    alpha = 255 if loop_index % 4 == 0 else (loop_index * 19 + 13) % 256
    return rgb, alpha


def add_color_layers(mesh):
    layers = {}

    def linear_rgba(rgb, alpha):
        return [byte_linear(rgb[0]), byte_linear(rgb[1]), byte_linear(rgb[2]), alpha / 255.0]

    paint = mesh.color_attributes.new("Paint", "FLOAT_COLOR", "POINT")
    values = []
    for index, item in enumerate(paint.data):
        value = color_point_value(index)
        item.color = value
        values.append(value)
    layers["Paint"] = {"domain": "point", "source": "FLOAT_COLOR", "values": values}

    glow = mesh.color_attributes.new("Glow", "FLOAT_COLOR", "CORNER")
    values = []
    for loop_index, item in enumerate(glow.data):
        value = color_corner_value(loop_index)
        item.color = value
        values.append(value)
    layers["Glow"] = {"domain": "corner", "source": "FLOAT_COLOR", "values": values}

    dirt = mesh.color_attributes.new("Dirt", "BYTE_COLOR", "POINT")
    values = []
    for index, item in enumerate(dirt.data):
        rgb, alpha = byte_point_value(index)
        value = linear_rgba(rgb, alpha)
        item.color = value
        values.append(value)
    layers["Dirt"] = {"domain": "point", "source": "BYTE_COLOR", "values": values}

    trim = mesh.color_attributes.new("Trim", "BYTE_COLOR", "CORNER")
    values = []
    for loop_index, item in enumerate(trim.data):
        rgb, alpha = byte_corner_value(loop_index)
        value = linear_rgba(rgb, alpha)
        item.color = value
        values.append(value)
    layers["Trim"] = {"domain": "corner", "source": "BYTE_COLOR", "values": values}

    mesh.color_attributes.active_color_index = COLOR_ACTIVE
    mesh.color_attributes.render_color_index = COLOR_RENDER
    for item in mesh.color_attributes:
        expected_values = layers[item.name]["values"]
        for element, expected in zip(item.data, expected_values):
            for a, b in zip(tuple(element.color), expected):
                if abs(a - b) > 1e-6:
                    raise SystemExit("color readback differs for %s: %r vs %r" % (item.name, tuple(element.color), expected))
    return layers


def add_uv_layers(mesh):
    layers = {}
    for layer_index, name in enumerate(UV_NAMES):
        layer = mesh.uv_layers.new(name=name)
        values = []
        for loop_index, item in enumerate(layer.data):
            uv = [round4(0.1 * (layer_index + 1) + 0.013 * loop_index), round4(0.9 - 0.007 * loop_index - 0.2 * layer_index)]
            item.uv = uv
            values.append(uv)
        layers[name] = values
    mesh.uv_layers.active = mesh.uv_layers[UV_NAMES[UV_ACTIVE]]
    mesh.uv_layers[UV_NAMES[UV_RENDER]].active_render = True
    return layers


POTTER_TYPE = {"FLOAT": "float", "INT": "int", "BOOLEAN": "bool", "FLOAT_VECTOR": "float3"}


def add_generic_attributes(mesh):
    specs = {}

    def create(name, data_type, domain, values, setter):
        attribute = mesh.attributes.new(name, data_type, domain)
        for index, item in enumerate(attribute.data):
            setter(item, values[index])
        specs[name] = {"domain": domain.lower(), "type": POTTER_TYPE[data_type], "values": values}

    def scalar(item, value):
        item.value = value

    def vector(item, value):
        item.vector = value

    point_wear = [round4(0.125 * (i % 9) - 0.25) for i in range(len(mesh.vertices))]
    create("wear", "FLOAT", "POINT", point_wear, scalar)
    point_cluster = [(i % 5) - 2 for i in range(len(mesh.vertices))]
    create("cluster", "INT", "POINT", point_cluster, scalar)
    edge_crease = [round4(0.5 + 0.0625 * (i % 8)) for i in range(len(mesh.edges))]
    create("crease_hint", "FLOAT", "EDGE", edge_crease, scalar)
    edge_keep = [i % 3 == 0 for i in range(len(mesh.edges))]
    create("keep", "BOOLEAN", "EDGE", edge_keep, scalar)
    face_island = [i % 4 for i in range(len(mesh.polygons))]
    create("island", "INT", "FACE", face_island, scalar)
    face_bake = [i % 2 == 1 for i in range(len(mesh.polygons))]
    create("bake_me", "BOOLEAN", "FACE", face_bake, scalar)
    face_tint = [(round4(0.1 * (i % 5)), 0.5, round4(0.2 + 0.05 * (i % 7))) for i in range(len(mesh.polygons))]
    create("face_tint", "FLOAT_VECTOR", "FACE", face_tint, vector)
    corner_rank = [round4(0.01 * i + 0.3) for i in range(loop_count(mesh))]
    create("corner_rank", "FLOAT", "CORNER", corner_rank, scalar)
    corner_dir = [(0.0, round4(-0.1 * (i % 4)), 1.0) for i in range(loop_count(mesh))]
    create("corner_dir", "FLOAT_VECTOR", "CORNER", corner_dir, vector)
    return specs


def add_vertex_groups(obj):
    top = obj.vertex_groups.new(name="Top")
    side = obj.vertex_groups.new(name="Side")
    top_members = {}
    side_members = {}
    for vertex in obj.data.vertices:
        if vertex.co.z > 0.0:
            top_members[vertex.index] = 1.0
        if vertex.co.x > 0.0:
            side_members[vertex.index] = 0.25
    for index, weight in top_members.items():
        top.add([index], weight, "REPLACE")
    for index, weight in side_members.items():
        side.add([index], weight, "REPLACE")
    return {"Top": {str(k): v for k, v in sorted(top_members.items())},
            "Side": {str(k): v for k, v in sorted(side_members.items())}}


def build_surface_material():
    material = bpy.data.materials.new("Surface")
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    base = (0.2, 0.4, 0.6, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.25
    bsdf.inputs["Roughness"].default_value = 0.35
    color_node = tree.nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = "Paint"
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    socket_a = [s for s in mix.inputs if s.name == "A" and s.type == "RGBA"][0]
    socket_b = [s for s in mix.inputs if s.name == "B" and s.type == "RGBA"][0]
    socket_a.default_value = base
    result = [s for s in mix.outputs if s.name == "Result" and s.type == "RGBA"][0]
    tree.links.new(color_node.outputs["Color"], socket_b)
    tree.links.new(result, bsdf.inputs["Base Color"])
    return material, {
        "name": "Surface",
        "base_color": list(base),
        "metallic": 0.25,
        "roughness": 0.35,
        "vertex_color_layer": "Paint",
        "vertex_color_blend": "MULTIPLY",
    }


def build_bare_material():
    material = bpy.data.materials.new("Bare")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    return material


# ---- Oracle -----------------------------------------------------------------------------------
verts, faces = grid_cube(2)
oracle = make_object("Oracle", verts, faces)
oracle_mesh = oracle.data
set_shading(oracle_mesh, lambda index: index % 4 != 0)
oracle_authored = set_partial_custom_normals(oracle_mesh, REGION_FACES["Oracle"])
oracle_uv = add_uv_layers(oracle_mesh)
oracle_colors = add_color_layers(oracle_mesh)
oracle_generic = add_generic_attributes(oracle_mesh)
oracle_groups = add_vertex_groups(oracle)
surface, surface_info = build_surface_material()
oracle_mesh.materials.append(surface)

# ---- Regress ----------------------------------------------------------------------------------
regress = make_object("Regress", verts, faces, location=(4.0, 0.0, 0.0))
regress_mesh = regress.data
set_shading(regress_mesh, lambda index: True)
for edge in regress_mesh.edges:
    edge.use_edge_sharp = False
    edge.use_seam = False
regress_authored = set_partial_custom_normals(regress_mesh, REGION_FACES["Regress"])
regress_rows = custom_normal_rows(regress_mesh, regress_authored)
regress_codes = custom_normal_codes(regress_mesh)

subdivision = regress.modifiers.new("Subdivision", "SUBSURF")
subdivision.levels = 1
subdivision.render_levels = 1
subdivision.use_custom_normals = True
displace = regress.modifiers.new("Displace", "DISPLACE")
displace.strength = 0.1
displace.mid_level = 0.0
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = regress.evaluated_get(depsgraph).to_mesh()
evaluated.update()
reference_normals = [tuple(n.vector) for n in evaluated.corner_normals]
reference_corners = []
for poly in evaluated.polygons:
    center = list(poly.center)
    for k in range(poly.loop_total):
        loop_index = poly.loop_start + k
        co = evaluated.vertices[evaluated.loops[loop_index].vertex_index].co
        reference_corners.append({
            "co": list(co),
            "center": center,
            "normal": list(reference_normals[loop_index]),
        })
regress.evaluated_get(depsgraph).to_mesh_clear()
regress.modifiers.clear()

# ---- Plain ------------------------------------------------------------------------------------
plain_verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
plain = make_object("Plain", plain_verts, [[0, 1, 2, 3]], location=(-4.0, 0.0, 0.0))
plain.data.materials.append(build_bare_material())
# ---- Emitter ----------------------------------------------------------------------------------
# A 2x2 quad at z = 3 facing the camera, above the cube. Its emission is the Vertex Color layer
# "Ink" (constant, linear) at strength 1, so a render pixel equals `INK` exactly.
INK = (0.0625, 0.25, 0.5, 1.0)
emitter = make_object("Emitter", [(-1.0, -1.0, 3.0), (1.0, -1.0, 3.0), (1.0, 1.0, 3.0), (-1.0, 1.0, 3.0)], [[0, 1, 2, 3]])
ink = emitter.data.color_attributes.new("Ink", "FLOAT_COLOR", "CORNER")
for item in ink.data:
    item.color = INK
emit_material = bpy.data.materials.new("Ink")
emit_material.use_nodes = True
emit_tree = emit_material.node_tree
emit_tree.nodes.remove(emit_tree.nodes["Principled BSDF"])
emit_color = emit_tree.nodes.new("ShaderNodeVertexColor")
emit_color.layer_name = "Ink"
emission = emit_tree.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 1.0
emit_tree.links.new(emit_color.outputs["Color"], emission.inputs["Color"])
emit_tree.links.new(emission.outputs["Emission"], emit_tree.nodes["Material Output"].inputs["Surface"])
emitter.data.materials.append(emit_material)

# ---- TexturePlane -----------------------------------------------------------------------------
# A 4 x 4 texel image, asymmetric in rows and columns, shown on a square plane at PLANE_LOCATION
# (away from the other objects). Square because Blender's minimum render height is 4 pixels: the
# render is exactly the texel grid. Its UV layers: "Base" is flipped vertically (v' = 1 - v) and
# "Light" is the identity. Base is the active layer and Light the render layer.
TEX_WIDTH = 4
TEX_HEIGHT = 4
# File rows, top row first. Every row and every column holds distinct colors; the last row is blue.
TEX_FILE_ROWS = [
    [(230, 20, 20), (230, 90, 20), (230, 160, 20), (230, 230, 20)],
    [(20, 230, 20), (90, 230, 20), (160, 230, 20), (230, 230, 230)],
    [(20, 20, 230), (20, 90, 230), (20, 160, 230), (20, 20, 120)],
    [(20, 20, 200), (20, 60, 230), (60, 20, 200), (20, 40, 120)],
]
PLANE_LOCATION = (8.0, 0.0, 0.0)
plane_verts = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]
texture_plane = make_object("TexturePlane", plane_verts, [[0, 1, 2, 3]], location=PLANE_LOCATION)
plane_mesh = texture_plane.data
IDENTITY_UV = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
FLIPPED_UV = [(u, 1.0 - v) for u, v in IDENTITY_UV]
for layer_name, corner_uvs in (("Base", FLIPPED_UV), ("Light", IDENTITY_UV)):
    layer = plane_mesh.uv_layers.new(name=layer_name)
    for corner, item in enumerate(layer.data):
        item.uv = corner_uvs[corner]
plane_mesh.uv_layers.active = plane_mesh.uv_layers["Base"]
plane_mesh.uv_layers["Light"].active_render = True


def write_png(path, width, height, rows):
    """Writes an 8-bit RGB PNG. `rows` lists the file rows, top row first."""
    raw = b"".join(
        b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in rows
    )

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    assert len(rows) == height and all(len(row) == width for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


texture_path = os.path.join(OUT, "texture.png")
write_png(texture_path, TEX_WIDTH, TEX_HEIGHT, TEX_FILE_ROWS)

# ---- Write ------------------------------------------------------------------------------------
oracle_faces = [
    {"v": list(poly.vertices), "smooth": bool(poly.use_smooth)} for poly in oracle_mesh.polygons
]
oracle_edges = [
    {"v": list(edge.vertices), "sharp": bool(edge.use_edge_sharp), "seam": bool(edge.use_seam)}
    for edge in oracle_mesh.edges
]
oracle_custom = custom_normal_rows(oracle_mesh, oracle_authored)
oracle_codes = custom_normal_codes(oracle_mesh)

expected = {
    "format": "pr9-native-oracle/1",
    "blender": bpy.app.version_string,
    "byte_tolerance": 1.0 / 255.0,
    "float_tolerance": 1.0e-6,
    "normal_tolerance": 2.0e-4,
    "oracle": {
        "object": "Oracle",
        "vertices": [list(v.co) for v in oracle_mesh.vertices],
        "edges": oracle_edges,
        "faces": oracle_faces,
        "custom_normal": oracle_custom,
        "custom_normal_codes": oracle_codes,
        "uv_layers": {
            "names": UV_NAMES,
            "active": UV_NAMES[UV_ACTIVE],
            "render": UV_NAMES[UV_RENDER],
            "values": oracle_uv,
        },
        "color_layers": {
            "order": ["Paint", "Glow", "Dirt", "Trim"],
            "active": "Glow",
            "render": "Trim",
            "layers": oracle_colors,
        },
        "generic": oracle_generic,
        "vertex_groups": oracle_groups,
        "material": surface_info,
    },
    "regress": {
        "object": "Regress",
        "vertices": [list(v.co) for v in regress_mesh.vertices],
        "faces": [{"v": list(p.vertices), "smooth": bool(p.use_smooth)} for p in regress_mesh.polygons],
        "custom_normal": regress_rows,
        "custom_normal_codes": regress_codes,
        "modifiers": [
            {"type": "subdivision", "levels": 1},
            {"type": "displace", "strength": 0.1, "mid_level": 0.0},
        ],
        "reference_corners": reference_corners,
    },
    "plain": {
        "object": "Plain",
        "material": "Bare",
        "color_layers": [],
    },
    "emitter": {
        "object": "Emitter",
        "layer": "Ink",
        "value": list(INK),
        "camera": {"location": [0.0, 0.0, 6.0], "projection": "orthographic", "ortho_scale": 4.0},
        "pixel": [0.5, 0.5],
    },
}

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "oracle.blend"))

# Blender's own reading of the PNG. `image.pixels` lists rows bottom row first.
texture_image = bpy.data.images.load(texture_path, check_existing=False)
if tuple(texture_image.size) != (TEX_WIDTH, TEX_HEIGHT):
    raise SystemExit("texture size is %r" % (tuple(texture_image.size),))
values = list(texture_image.pixels)
observed = [
    [[values[(y * TEX_WIDTH + x) * 4 + c] for c in range(3)] for x in range(TEX_WIDTH)]
    for y in range(TEX_HEIGHT)
]
file_rows_linear = [[[srgb_to_linear(c / 255.0) for c in pixel] for pixel in row] for row in TEX_FILE_ROWS]
file_rows_unit = [[[c / 255.0 for c in pixel] for pixel in row] for row in TEX_FILE_ROWS]
# Blender row y must hold file row (height - 1 - y).
bottom_up_linear = [file_rows_linear[TEX_HEIGHT - 1 - y] for y in range(TEX_HEIGHT)]
bottom_up_unit = [file_rows_unit[TEX_HEIGHT - 1 - y] for y in range(TEX_HEIGHT)]


def matches(actual, wanted):
    return all(
        abs(a - b) <= 1.0e-4
        for row_a, row_b in zip(actual, wanted)
        for pixel_a, pixel_b in zip(row_a, row_b)
        for a, b in zip(pixel_a, pixel_b)
    )


if matches(observed, bottom_up_linear):
    transfer = "linear"
elif matches(observed, bottom_up_unit):
    transfer = "srgb"
else:
    raise SystemExit("Blender pixels match neither the linear nor the sRGB file rows")
if matches(observed, file_rows_linear if transfer == "linear" else file_rows_unit):
    raise SystemExit("Blender pixels are in file order (top row first); expected bottom row first")

# The field's base color factor (Potter's default material base color, spec "The default material"), which
# multiplies the texture in both the Potter field and the Blender reference's MixRGB MULTIPLY.
TEXTURE_BASE_FACTOR = 0.6
# Expected render of the texture plane, bottom row first. The render camera sees the plane's
# texel grid one-to-one, so a render pixel shows the texel its UV falls in, times the factor. Base is
# flipped (v' = 1 - v), so its row py is file row py; Light is the identity, so its row py is file row
# (height - 1 - py). A texture that names the wrong layer, or flips twice, renders the other one.
expected["texture"] = {
    "file": "texture.png",
    "width": TEX_WIDTH,
    "height": TEX_HEIGHT,
    "file_rows_top_down": [[list(pixel) for pixel in row] for row in TEX_FILE_ROWS],
    "file_rows_top_down_linear": file_rows_linear,
    "blender_pixel_transfer": transfer,
    "blender_pixels_bottom_up": observed,
    "blender_row_y_holds_file_row": [TEX_HEIGHT - 1 - y for y in range(TEX_HEIGHT)],
    "render_bottom_up_base_layer": [[[TEXTURE_BASE_FACTOR * c for c in pixel] for pixel in file_rows_linear[py]] for py in range(TEX_HEIGHT)],
    "render_bottom_up_light_layer": [[[TEXTURE_BASE_FACTOR * c for c in pixel] for pixel in file_rows_linear[TEX_HEIGHT - 1 - py]] for py in range(TEX_HEIGHT)],
    "plane": {
        "object": "TexturePlane",
        "location": list(PLANE_LOCATION),
        "size": [2.0, 2.0],
        "camera_ortho_scale": 2.0,
    },
}

# The reference: Blender's own nodes. Image Texture (Closest) with its vector from a UV Map node
# naming "Base"; the texture is multiplied by the field's base color factor through a MixRGB MULTIPLY
# (Color2 is the texture), which feeds Base Color, as Potter's export writes it.
reference = bpy.data.materials.new("Reference")
reference.use_nodes = True
tree = reference.node_tree
tex = tree.nodes.new("ShaderNodeTexImage")
tex.image = texture_image
tex.interpolation = "Closest"
uv_node = tree.nodes.new("ShaderNodeUVMap")
uv_node.uv_map = "Base"
factor_mix = tree.nodes.new("ShaderNodeMixRGB")
factor_mix.blend_type = "MULTIPLY"
factor_mix.inputs["Fac"].default_value = 1.0
factor_mix.inputs["Color1"].default_value = (TEXTURE_BASE_FACTOR, TEXTURE_BASE_FACTOR, TEXTURE_BASE_FACTOR, 1.0)
tree.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
tree.links.new(tex.outputs["Color"], factor_mix.inputs["Color2"])
tree.links.new(factor_mix.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
texture_plane.data.materials.append(reference)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "reference.blend"))

with open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, indent=1)
print("wrote", OUT)
