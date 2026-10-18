# Usage: /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tests/fixtures/bevel_parity_oracle.py -- <scratch-dir>
"""Blender 5.2.2 bevel parity oracle: geometry, edge and face flags, UVs, face strength, corner normals.

Writes tests/fixtures/bevel_parity_oracle.json and a copy under <scratch-dir>. Every float is the
little-endian IEEE-754 binary32 bit pattern as lowercase hex, normalised so -0.0 is written as 0.0.
"""
import bpy
import ctypes
import json
import math
import os
import random
import struct
import sys
from mathutils import Vector


if bpy.app.version != (5, 2, 2):
    raise RuntimeError('need Blender 5.2.2, got ' + str(bpy.app.version))
if '--' not in sys.argv or not sys.argv[sys.argv.index('--') + 1:]:
    raise RuntimeError('pass a scratch directory after --')
OUT = sys.argv[sys.argv.index('--') + 1]
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bevel_parity_oracle.json')
SCHEMA_VERSION = 2
ACOS_TRIGO_THRESHOLD = math.acos(0.9999)  # Blender LNOR_SPACE_TRIGO_THRESHOLD in custom-normal codes
STEP_16_BIT = math.pi / 32767.0
os.makedirs(OUT, exist_ok=True)


def f32_round(value):
    """The binary32 value nearest to `value` as a Python float; -0.0 becomes 0.0."""
    return struct.unpack('<f', struct.pack('<f', float(value)))[0] + 0.0


def f32(value):
    return struct.pack('<f', f32_round(value)).hex()


def f32_value(text):
    return struct.unpack('<f', bytes.fromhex(text))[0] + 0.0


def edge_key(a, b):
    return (min(a, b), max(a, b))


def orient_closed_faces(vertices, faces):
    center = sum((Vector(vertex) for vertex in vertices), Vector()) / len(vertices)
    result = []
    for face in faces:
        points = [Vector(vertices[index]) for index in face]
        normal = (points[1] - points[0]).cross(points[2] - points[0])
        face_center = sum(points, Vector()) / len(points)
        result.append(tuple(reversed(face)) if normal.dot(face_center - center) < 0.0 else tuple(face))
    return result


def shape_data():
    cube_vertices = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    cube_faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    tetra_vertices = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
    tetra_faces = orient_closed_faces(tetra_vertices, [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)])
    base = [(0, 0, 0), (2, 0, 0), (1.7, 1.0, 0), (-0.2, 1.2, 0)]
    top = [(x + 0.45, y + 0.18, 1.4) for x, y, _ in base]
    prism_vertices = base + top
    prism_faces = orient_closed_faces(prism_vertices, [
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ])
    pyramid_vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0.2, 0.1, 1.8)]
    pyramid_faces = orient_closed_faces(pyramid_vertices, [
        (0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
    ])
    outline = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    side_count = len(outline)
    reflex_vertices = [(x, y, z) for z in (0, 1) for x, y in outline]
    reflex_faces = [tuple(reversed(range(side_count))), tuple(range(side_count, 2 * side_count))]
    reflex_faces.extend((index, (index + 1) % side_count,
                         (index + 1) % side_count + side_count, index + side_count)
                        for index in range(side_count))
    # Valence 5 apex with unequal base radii: every dihedral angle is non-right.
    pentagon_radius = (1.0, 0.8, 1.1, 0.9, 1.25)
    pentagon_base = [(r * math.cos(2 * math.pi * k / 5), r * math.sin(2 * math.pi * k / 5), 0.0)
                     for k, r in enumerate(pentagon_radius)]
    pentagon_vertices = pentagon_base + [(0.15, -0.1, 1.6)]
    pentagon_faces = orient_closed_faces(pentagon_vertices, [tuple(range(5))] + [
        (k, (k + 1) % 5, 5) for k in range(5)])
    # Valence 6 apex.
    hexagon_radius = (1.2, 1.0, 1.3, 0.9, 1.15, 1.05)
    hexagon_base = [(r * math.cos(2 * math.pi * k / 6 + 0.1), r * math.sin(2 * math.pi * k / 6 + 0.1), 0.0)
                    for k, r in enumerate(hexagon_radius)]
    hexagon_vertices = hexagon_base + [(0.05, 0.1, 1.5)]
    hexagon_faces = orient_closed_faces(hexagon_vertices, [tuple(range(6))] + [
        (k, (k + 1) % 6, 6) for k in range(6)])
    # Valence 4 everywhere, stretched so no dihedral angle is the regular 109.47 degrees.
    octahedron_vertices = [(1.2, 0, 0), (-1, 0.1, 0), (0, 1.1, 0), (0, -0.9, 0), (0, 0, 1.3), (0.1, 0, -0.8)]
    octahedron_faces = orient_closed_faces(octahedron_vertices, [
        (4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0),
        (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3),
    ])
    # Open surfaces omit one face, so their rim edges have a single face and are never EDGES-selected.
    # The fan is a valence-5 centre with a 240-degree open rim of triangles.
    open_fan_radius = (1.0, 0.8, 1.1, 0.9, 1.2)
    open_fan_vertices = [(0.0, 0.0, 0.5)] + [
        (r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3), 0.15 * k - 0.2)
        for k, r in enumerate(open_fan_radius)]
    open_fan_faces = [(0, k, k + 1) for k in range(1, 5)]
    return {
        'cube': (cube_vertices, cube_faces),
        'tetrahedron': (tetra_vertices, tetra_faces),
        'oblique_prism': (prism_vertices, prism_faces),
        'valence4_pyramid': (pyramid_vertices, pyramid_faces),
        'concave_reflex_prism': (reflex_vertices, reflex_faces),
        'boundary_quad': ([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [(0, 1, 2, 3)]),
        'nonmanifold_three_faces': (
            [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -0.8, 0), (0, 0, 1)],
            [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
        ),
        'pentagonal_pyramid': (pentagon_vertices, pentagon_faces),
        'hexagonal_pyramid': (hexagon_vertices, hexagon_faces),
        'irregular_octahedron': (octahedron_vertices, octahedron_faces),
        'open_fan': (open_fan_vertices, open_fan_faces),
        'open_pyramid': (pyramid_vertices, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]),
        'open_box': (cube_vertices, [face for face in cube_faces if face != (4, 5, 6, 7)]),
    }


def make_object(name, vertices, faces, custom_normals):
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    materials = []
    for material_name, color in (
        ('bevel_oracle_cool', (0.2, 0.4, 0.8, 1.0)),
        ('bevel_oracle_warm', (0.8, 0.25, 0.1, 1.0)),
    ):
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(material_name)
        material.diffuse_color = color
        mesh.materials.append(material)
        materials.append(material_name)
    for polygon in mesh.polygons:
        polygon.material_index = polygon.index % 2
        polygon.use_smooth = polygon.index % 2 == 0

    uv_layer = mesh.uv_layers.new(name='UVMap')
    uv_input = []
    uv_pins = []
    for polygon in mesh.polygons:
        count = len(polygon.loop_indices)
        for corner, loop_index in enumerate(polygon.loop_indices):
            angle = 2.0 * math.pi * corner / count
            offset = polygon.index * 0.375
            uv = (offset + 0.5 + 0.5 * math.cos(angle),
                  0.25 * polygon.index + 0.5 + 0.5 * math.sin(angle))
            uv_layer.data[loop_index].uv = uv
            uv_input.append([f32(component) for component in uv])
            uv_pins.append(bool(uv_layer.data[loop_index].pin_uv))

    group_names = ('weight_ramp', 'weight_inverse')
    groups = [obj.vertex_groups.new(name=group_name) for group_name in group_names]
    group_weights = []
    for vertex in mesh.vertices:
        ramp = 0.15 + (vertex.index % 5) * 0.16
        inverse = 0.95 - ramp
        groups[0].add([vertex.index], ramp, 'REPLACE')
        groups[1].add([vertex.index], inverse, 'REPLACE')
        group_weights.append([f32(ramp), f32(inverse)])

    # A deterministic one-third of the edges (in sorted endpoint order) carry bevel weight 1.
    selected_edges = set(sorted({edge_key(*edge.vertices) for edge in mesh.edges})[::3])
    weight_attr = mesh.attributes.new('bevel_weight_edge', 'FLOAT', 'EDGE')
    edge_input = []
    for edge in mesh.edges:
        weight = 1.0 if edge_key(*edge.vertices) in selected_edges else 0.0
        weight_attr.data[edge.index].value = weight
        edge.use_seam = edge.index % 3 == 0
        edge.use_edge_sharp = edge.index % 4 == 0
        edge_input.append({
            'vertices': list(edge.vertices),
            'seam': bool(edge.use_seam),
            'sharp': bool(edge.use_edge_sharp),
            'bevel_weight': f32(weight),
        })

    # Every other vertex carries bevel weight 1 for VERTICES-affect weight limits.
    vertex_weight_attr = mesh.attributes.new('bevel_weight_vert', 'FLOAT', 'POINT')
    vertex_weight_input = []
    for vertex in mesh.vertices:
        weight = 1.0 if vertex.index % 2 == 0 else 0.0
        vertex_weight_attr.data[vertex.index].value = weight
        vertex_weight_input.append(f32(weight))

    custom_normal_codes = None
    custom_normal_vectors = None
    if custom_normals:
        custom = []
        for polygon in mesh.polygons:
            face_normal = polygon.normal
            for loop_index in polygon.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                perturb = Vector((0.025 * (vertex_index % 3 - 1),
                                  0.02 * (polygon.index % 3 - 1),
                                  0.015 * ((vertex_index + polygon.index) % 3 - 1)))
                custom.append((face_normal + perturb).normalized())
        mesh.normals_split_custom_set(custom)
        mesh.update()
        custom_normal = mesh.attributes.get('custom_normal')
        if custom_normal is not None:
            custom_normal_codes = []
            custom_normal_vectors = []
            for loop_index, item in enumerate(custom_normal.data):
                code = [int(component) for component in item.value]
                custom_normal_codes.append(code)
                custom_normal_vectors.append(
                    None if code == [0, 0] else
                    [f32(component) for component in mesh.corner_normals[loop_index].vector])
    input_data = {
        'vertices': [[f32(component) for component in vertex.co] for vertex in mesh.vertices],
        'faces': [list(polygon.vertices) for polygon in mesh.polygons],
        'face_material_indices': [int(polygon.material_index) for polygon in mesh.polygons],
        'face_smooth': [bool(polygon.use_smooth) for polygon in mesh.polygons],
        'edges': edge_input,
        'vertex_bevel_weight': vertex_weight_input,
        'materials': materials,
        'vertex_groups': {'names': list(group_names), 'weights': group_weights},
        'uv_layers': {'UVMap': uv_input},
        'custom_normal_codes': custom_normal_codes,
        'custom_normal_vectors': custom_normal_vectors,
    }
    if any(uv_pins):
        input_data['uv_pins'] = uv_pins
    return obj, input_data


def profile_point_handles(point):
    """Read the stored CurveProfilePoint handles, which RNA does not expose."""
    h1x, h1y, h2x, h2y = struct.unpack('<ffff', ctypes.string_at(point.as_pointer() + 0xc, 16))
    return [f32_round(h1x), f32_round(h1y)], [f32_round(h2x), f32_round(h2y)]


# Blender's RNA identifier is ALIGN; the parameter contract spells it ALIGNED.
HANDLE_TO_CONTRACT = {'AUTO': 'AUTO', 'VECTOR': 'VECTOR', 'ALIGN': 'ALIGNED', 'FREE': 'FREE'}
HANDLE_CODES = {'FREE': 0, 'AUTO': 1, 'VECTOR': 2, 'ALIGNED': 3}


def custom_point_spec(spec):
    if len(spec) == 3:
        return spec[0], spec[1], spec[2], None
    if len(spec) == 5:
        return spec[0], spec[1], spec[2], (spec[3], spec[4])
    raise RuntimeError('custom profile point must have location and handle types, optionally handles')


def capture_profile(profile):
    points = []
    for point in profile.points:
        handle_1, handle_2 = profile_point_handles(point)
        points.append({
            'location': [f32_round(point.location[0]), f32_round(point.location[1])],
            'handle_1': handle_1,
            'handle_2': handle_2,
            'handle_type_1': HANDLE_TO_CONTRACT[point.handle_type_1],
            'handle_type_2': HANDLE_TO_CONTRACT[point.handle_type_2],
        })
    return {
        'points': points,
        'use_clip': bool(profile.use_clip),
        'use_sample_straight_edges': bool(profile.use_sample_straight_edges),
        'use_sample_even_lengths': bool(profile.use_sample_even_lengths),
    }


def apply_custom_profile(profile, spec):
    # Blender's LINE preset resets the flags and the endpoints; the spec is applied on top of it.
    profile.preset = 'LINE'
    profile.use_clip = spec.get('use_clip', profile.use_clip)
    profile.use_sample_straight_edges = spec.get('use_sample_straight_edges', False)
    profile.use_sample_even_lengths = spec.get('use_sample_even_lengths', False)
    # Point additions reallocate and re-order the collection; retain and shift stable indices.
    authored_points = []
    claimed = set()
    for point_spec in spec.get('points', []):
        location, handle_1, handle_2, handles = custom_point_spec(point_spec)
        wanted = (f32_round(location[0]), f32_round(location[1]))
        target_index = next((index for index, point in enumerate(profile.points)
                             if index not in claimed
                             and (f32_round(point.location[0]), f32_round(point.location[1])) == wanted),
                            None)
        if target_index is None:
            inserted = profile.points.add(location[0], location[1])
            pointer = inserted.as_pointer()
            target_index = next(index for index, point in enumerate(profile.points)
                                if point.as_pointer() == pointer)
            authored_points = [
                (index + (index >= target_index), first, second, authored_handles)
                for index, first, second, authored_handles in authored_points
            ]
            claimed = {index + (index >= target_index) for index in claimed}
        authored_points.append((target_index, handle_1, handle_2, handles))
        claimed.add(target_index)
    # Initialize AUTO handles at final point locations before authoring FREE/ALIGNED handles.
    profile.initialize(2)
    for index, handle_1, handle_2, _ in authored_points:
        ctypes.memmove(
            profile.points[index].as_pointer() + 0xa,
            bytes([HANDLE_CODES[handle_1], HANDLE_CODES[handle_2]]),
            2,
        )
    for index, _, _, handles in authored_points:
        if handles is not None:
            h1, h2 = handles
            ctypes.memmove(
                profile.points[index].as_pointer() + 0xc,
                struct.pack('<ffff', f32_round(h1[0]), f32_round(h1[1]),
                            f32_round(h2[0]), f32_round(h2[1])),
                16,
            )
    profile.initialize(2)


def capture_mesh(obj, mesh):
    group_names = [group.name for group in obj.vertex_groups]
    group_count = len(group_names)
    raw_vertices = []
    for vertex in mesh.vertices:
        if vertex.hide:
            raise RuntimeError('bevel output contains a hidden vertex')
        weights = [0.0] * group_count
        for assignment in vertex.groups:
            if assignment.group < group_count:
                weights[assignment.group] = f32_round(assignment.weight)
        raw_vertices.append((tuple(f32_round(component) for component in vertex.co), tuple(weights)))
    # Keep distinct Blender vertices even when position and weights coincide; topology identifies them.
    ordered_indices = sorted(range(len(raw_vertices)), key=lambda index: (raw_vertices[index], index))
    ordered_vertices = [raw_vertices[index] for index in ordered_indices]
    vertex_ids = [0] * len(raw_vertices)
    for label, index in enumerate(ordered_indices):
        vertex_ids[index] = label

    weight_attribute = mesh.attributes.get('bevel_weight_edge')
    edges = []
    for edge in mesh.edges:
        if edge.hide:
            raise RuntimeError('bevel output contains a hidden edge')
        first, second = sorted(vertex_ids[index] for index in edge.vertices)
        weight = f32_round(weight_attribute.data[edge.index].value) if weight_attribute is not None else None
        edges.append((first, second, bool(edge.use_seam), bool(edge.use_edge_sharp), weight))
    edges.sort(key=lambda row: (row[0], row[1], row[2], row[3],
                                float('-inf') if row[4] is None else row[4]))

    uv_layer = mesh.uv_layers.get('UVMap')
    if uv_layer is None:
        raise RuntimeError('evaluated bevel mesh lost UVMap')
    sharp_faces = mesh.attributes.get('sharp_face')
    face_strength = mesh.attributes.get('__mod_weightednormals_faceweight')
    faces = []
    for polygon in mesh.polygons:
        if polygon.hide:
            raise RuntimeError('bevel output contains a hidden face')
        smooth = bool(polygon.use_smooth)
        if sharp_faces is not None and bool(sharp_faces.data[polygon.index].value) == smooth:
            raise RuntimeError('sharp_face disagrees with use_smooth')
        ring = []
        for loop_index in polygon.loop_indices:
            loop = mesh.loops[loop_index]
            uv = uv_layer.data[loop_index].uv
            normal = mesh.corner_normals[loop_index].vector
            ring.append((vertex_ids[loop.vertex_index], f32_round(uv.x), f32_round(uv.y),
                         *(f32_round(component) for component in normal)))
        # Rings start at the lexicographic minimum rotation over (position, canonical vertex id).
        corner_key = [(ordered_vertices[corner[0]][0], corner[0]) for corner in ring]
        size = len(ring)
        start = min(range(size), key=lambda shift: [corner_key[(shift + step) % size]
                                                    for step in range(size)])
        ring = ring[start:] + ring[:start]
        strength = int(face_strength.data[polygon.index].value) if face_strength is not None else None
        faces.append((int(polygon.material_index), smooth, strength, ring))
    faces.sort(key=lambda row: ([corner[0] for corner in row[3]], row[0], row[1],
                                -2 ** 40 if row[2] is None else row[2]))
    return {
        'group_names': group_names,
        'vertices': [[[f32(component) for component in position], [f32(weight) for weight in weights]]
                     for position, weights in ordered_vertices],
        'edges': [[first, second, seam, sharp, None if weight is None else f32(weight)]
                  for first, second, seam, sharp, weight in edges],
        'faces': [[material, smooth, strength,
                   [[corner[0], f32(corner[1]), f32(corner[2]),
                     f32(corner[3]), f32(corner[4]), f32(corner[5])] for corner in ring]]
                  for material, smooth, strength, ring in faces],
    }


DEFAULTS = {
    'segments': 1, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
    'width': 0.17, 'width_pct': 17.0, 'limit_method': 'NONE', 'angle_limit': 0.5,
    'edge_weight': 'bevel_weight_edge', 'vertex_weight': 'bevel_weight_vert',
    'vertex_group': '', 'invert_vertex_group': False,
    'use_clamp_overlap': True, 'profile_type': 'SUPERELLIPSE', 'loop_slide': True,
    'miter_outer': 'MITER_SHARP', 'miter_inner': 'MITER_SHARP', 'spread': 0.1,
    'vmesh_method': 'ADJ', 'mark_seam': False, 'mark_sharp': False, 'harden_normals': False,
    'face_strength_mode': 'FSTR_NONE', 'material': -1,
}
FLOAT_PARAMETERS = ('profile', 'spread', 'angle_limit', 'width', 'width_pct')


def evaluate(obj, shape, name, config):
    parameters = dict(DEFAULTS)
    parameters.update(config)
    custom_spec = parameters.pop('custom_profile', None)
    modifier = obj.modifiers.new('bevel_oracle', 'BEVEL')
    modifier.segments = parameters['segments']
    modifier.profile = parameters['profile']
    modifier.affect = parameters['affect']
    # RNA `width` and `width_pct` share one storage field; write only the one the type reads.
    modifier.offset_type = parameters['offset_type']
    if parameters['offset_type'] == 'PERCENT':
        modifier.width_pct = parameters['width_pct']
        if abs(modifier.width_pct - parameters['width_pct']) > 1e-6:
            raise RuntimeError('width_pct did not round-trip for ' + name)
    else:
        modifier.width = parameters['width']
        if abs(modifier.width - parameters['width']) > 1e-6:
            raise RuntimeError('width did not round-trip for ' + name)
    modifier.limit_method = parameters['limit_method']
    modifier.angle_limit = parameters['angle_limit']
    modifier.edge_weight = parameters['edge_weight']
    modifier.vertex_weight = parameters['vertex_weight']
    modifier.vertex_group = parameters['vertex_group']
    modifier.invert_vertex_group = parameters['invert_vertex_group']
    modifier.use_clamp_overlap = parameters['use_clamp_overlap']
    modifier.profile_type = parameters['profile_type']
    modifier.spread = parameters['spread']
    modifier.loop_slide = parameters['loop_slide']
    modifier.miter_outer = parameters['miter_outer']
    modifier.miter_inner = parameters['miter_inner']
    modifier.vmesh_method = parameters['vmesh_method']
    modifier.mark_seam = parameters['mark_seam']
    modifier.mark_sharp = parameters['mark_sharp']
    modifier.harden_normals = parameters['harden_normals']
    modifier.face_strength_mode = parameters['face_strength_mode']
    modifier.material = parameters['material']
    if parameters['profile_type'] == 'CUSTOM':
        apply_custom_profile(modifier.custom_profile, custom_spec or {})
        parameters['custom_profile'] = capture_profile(modifier.custom_profile)
    # Record the values Blender actually stores (single-precision), so Potter receives the same inputs.
    for key in FLOAT_PARAMETERS:
        parameters[key] = f32_round(getattr(modifier, key))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if evaluated_mesh is None:
        raise RuntimeError('Blender returned no evaluated mesh for ' + name)
    try:
        output = capture_mesh(evaluated_obj, evaluated_mesh)
    finally:
        evaluated_obj.to_mesh_clear()
        obj.modifiers.remove(modifier)
    return {'name': name, 'shape': shape, 'parameters': parameters, 'output': output}


def normal_quantization_probe():
    """Round-trip error of Blender's 16-bit custom-normal codes over random directions.

    Each triangle is its own smooth fan, so its codes are relative to the face normal. The result
    bounds the difference between a custom normal and Blender's stored copy of it.
    """
    rng = random.Random(20260416)
    triangles = 65536
    verts, faces = [], []
    for _ in range(triangles):
        x, y, z = (rng.uniform(-5.0, 5.0) for _ in range(3))
        base = len(verts)
        verts.extend([(x, y, z), (x + 1.0, y + 0.1, z), (x + 0.2, y + 1.0, z + 0.3)])
        faces.append((base, base + 1, base + 2))
    mesh = bpy.data.meshes.new('bevel_oracle_normal_probe')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    wanted = [Vector((rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))).normalized()
              for _ in range(3 * triangles)]
    mesh.normals_split_custom_set(wanted)
    mesh.update()
    stored = [normal.vector.copy() for normal in mesh.corner_normals]
    # Potter's inputs are Blender's stored values, and its outputs re-encode them. Store the
    # stored values again and read them back: the codec must map a stored value to itself.
    mesh.normals_split_custom_set(stored)
    mesh.update()
    restored = [normal.vector.copy() for normal in mesh.corner_normals]
    bpy.data.meshes.remove(mesh)
    max_component = 0.0
    max_angle = 0.0
    for want, got in zip(wanted, stored):
        max_component = max(max_component, abs(want.x - got.x), abs(want.y - got.y), abs(want.z - got.z))
        max_angle = max(max_angle, math.acos(max(-1.0, min(1.0, want.dot(got.normalized())))))
    restore_component = 0.0
    restore_angle = 0.0
    for got, again in zip(stored, restored):
        restore_component = max(restore_component, abs(got.x - again.x), abs(got.y - again.y), abs(got.z - again.z))
        restore_angle = max(restore_angle, math.acos(max(-1.0, min(1.0, got.normalized().dot(again.normalized())))))
    # Measured: a stored value re-stored decodes up to ~0.0141 rad away (the trigonometric snap
    # threshold), so the bound is that threshold plus one 16-bit code step.
    if restore_angle > ACOS_TRIGO_THRESHOLD + STEP_16_BIT:
        raise RuntimeError('a stored custom normal does not restore within the codec bound')
    return {
        'samples': len(wanted),
        'wanted_to_stored_max_abs_component_error': max_component,
        'wanted_to_stored_max_angle_error_radians': max_angle,
        'stored_restore_max_abs_component_error': restore_component,
        'stored_restore_max_angle_error_radians': restore_angle,
        'trigo_threshold_angle_radians': ACOS_TRIGO_THRESHOLD,
        'code_step_radians': STEP_16_BIT,
    }


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


clear_scene()
shape_inputs = shape_data()
objects = {}
inputs = {}
for shape_name, (vertices, faces) in shape_inputs.items():
    objects[shape_name], inputs[shape_name] = make_object(shape_name, vertices, faces, True)
cube_vertices, cube_faces = shape_inputs['cube']


def max_valence(faces):
    neighbours = {}
    for face in faces:
        for first, second in zip(face, face[1:] + face[:1]):
            neighbours.setdefault(first, set()).add(second)
            neighbours.setdefault(second, set()).add(first)
    return max(len(values) for values in neighbours.values())


assert max_valence(shape_inputs['pentagonal_pyramid'][1]) == 5
assert max_valence(shape_inputs['hexagonal_pyramid'][1]) == 6
assert max_valence(shape_inputs['irregular_octahedron'][1]) == 4
objects['cube_plain'], inputs['cube_plain'] = make_object('cube_plain', cube_vertices, cube_faces, False)

cube_edge_weights = [edge['bevel_weight'] for edge in inputs['cube']['edges']]
assert 0 < sum(weight == f32(1.0) for weight in cube_edge_weights) < len(cube_edge_weights)
assert sum(1 for weight in inputs['cube']['vertex_bevel_weight'] if weight == f32(1.0)) == 4
assert inputs['cube_plain']['custom_normal_codes'] is None

# Pairwise matrix covers every segment/profile pairing and every pair of
# segments, profiles, affect modes, and width modes without the 300-case product.
width_types = ('OFFSET', 'WIDTH', 'DEPTH', 'PERCENT', 'ABSOLUTE')
configs = []
for segment in range(1, 7):
    for profile_index, profile in enumerate((0.0, 0.25, 0.5, 0.8, 1.0)):
        affect = ('EDGES', 'VERTICES')[(segment - 1 + profile_index) % 2]
        width_type = width_types[(segment - 1 + 2 * profile_index) % len(width_types)]
        configs.append({'segments': segment, 'profile': profile, 'affect': affect,
                        'offset_type': width_type})
assert len(configs) == 30
factors = (
    ('segments', tuple(range(1, 7))),
    ('profile', (0.0, 0.25, 0.5, 0.8, 1.0)),
    ('affect', ('EDGES', 'VERTICES')),
    ('offset_type', width_types),
)
for first_index, (first_name, first_values) in enumerate(factors):
    for second_name, second_values in factors[first_index + 1:]:
        observed = {(config[first_name], config[second_name]) for config in configs}
        assert set((first, second) for first in first_values for second in second_values) <= observed

cases = [
    evaluate(objects['cube'], 'cube',
             'cube_s%d_p%d_%s_%s' % (config['segments'], int(config['profile'] * 100),
                                    config['affect'].lower(), config['offset_type'].lower()),
             config)
    for config in configs
]
cases.extend([
    evaluate(objects['cube'], 'cube', 'cube_selected_edges_s3_p80', {
        'segments': 3, 'profile': 0.8, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'limit_method': 'WEIGHT', 'face_strength_mode': 'FSTR_ALL', 'material': 1,
        'use_clamp_overlap': False, 'loop_slide': False,
    }),
    evaluate(objects['cube'], 'cube', 'cube_hardened_custom_normals_s2_p50', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'harden_normals': True,
    }),
    evaluate(objects['oblique_prism'], 'oblique_prism', 'oblique_prism_angle_limit_patch_miter_strength_new', {
        'segments': 3, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'limit_method': 'ANGLE', 'angle_limit': 1.4, 'miter_outer': 'MITER_PATCH',
        'miter_inner': 'MITER_ARC', 'face_strength_mode': 'FSTR_NEW', 'material': 0,
        'mark_sharp': True,
    }),
    evaluate(objects['valence4_pyramid'], 'valence4_pyramid', 'pyramid_vgroup_arc_cutoff_strength_affected', {
        'segments': 3, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'WIDTH',
        'limit_method': 'VGROUP', 'vertex_group': 'weight_ramp', 'invert_vertex_group': True,
        'miter_outer': 'MITER_ARC', 'miter_inner': 'MITER_SHARP', 'vmesh_method': 'CUTOFF',
        'face_strength_mode': 'FSTR_AFFECTED', 'material': 1, 'mark_seam': True,
    }),
    evaluate(objects['tetrahedron'], 'tetrahedron', 'tetrahedron_s2_p50', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
    }),
    evaluate(objects['oblique_prism'], 'oblique_prism', 'oblique_prism_s4_p80', {
        'segments': 4, 'profile': 0.8, 'affect': 'VERTICES', 'offset_type': 'DEPTH',
    }),
    evaluate(objects['valence4_pyramid'], 'valence4_pyramid', 'valence4_pyramid_s3_p25', {
        'segments': 3, 'profile': 0.25, 'affect': 'EDGES', 'offset_type': 'WIDTH',
    }),
    evaluate(objects['boundary_quad'], 'boundary_quad', 'boundary_quad_s2_p100', {
        'segments': 2, 'profile': 1.0, 'affect': 'EDGES', 'offset_type': 'ABSOLUTE',
    }),
    evaluate(objects['nonmanifold_three_faces'], 'nonmanifold_three_faces', 'nonmanifold_three_faces_s5_p0', {
        'segments': 5, 'profile': 0.0, 'affect': 'VERTICES', 'offset_type': 'PERCENT',
    }),
])

cases.append(evaluate(objects['cube'], 'cube', 'cube_custom_profile_default_line', {
    'segments': 3, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
    'profile_type': 'CUSTOM', 'custom_profile': {},
}))

reflex_miter_cases = {}
reflex_base = {'segments': 3, 'profile': 0.5, 'affect': 'EDGES',
               'offset_type': 'OFFSET', 'width': 0.25}
for miter_outer in ('MITER_SHARP', 'MITER_PATCH', 'MITER_ARC'):
    case = evaluate(objects['concave_reflex_prism'], 'concave_reflex_prism',
                    'reflex_outer_%s_spread_010' % miter_outer.lower(), {
                        **reflex_base, 'miter_outer': miter_outer,
                        'miter_inner': 'MITER_SHARP', 'spread': 0.1,
                    })
    cases.append(case)
    reflex_miter_cases[(miter_outer, 'MITER_SHARP', 0.1)] = case
for spread in (0.05, 0.5, 1.0):
    for miter_inner in ('MITER_SHARP', 'MITER_ARC'):
        case = evaluate(objects['concave_reflex_prism'], 'concave_reflex_prism',
                        'reflex_inner_%s_spread_%03d' % (miter_inner.lower(), int(spread * 100)), {
                            **reflex_base, 'miter_outer': 'MITER_SHARP',
                            'miter_inner': miter_inner, 'spread': spread,
                        })
        cases.append(case)
        reflex_miter_cases[('MITER_SHARP', miter_inner, spread)] = case
clamp_pairs = {}
for width_type in ('OFFSET', 'WIDTH', 'PERCENT', 'ABSOLUTE'):
    for size_name, width, width_pct in (('small', 0.001, 0.1), ('large', 5.0, 500.0)):
        outputs = {}
        for clamp in (True, False):
            modifier_case = evaluate(objects['cube'], 'cube', 'clamp_%s_%s_%s' % (
                size_name, width_type.lower(), 'on' if clamp else 'off'), {
                    'segments': 2, 'profile': 0.5, 'affect': 'EDGES',
                    'offset_type': width_type, 'width': width, 'width_pct': width_pct,
                    'use_clamp_overlap': clamp,
                })
            cases.append(modifier_case)
            outputs[clamp] = modifier_case['output']
        clamp_pairs[(width_type, size_name)] = outputs

# Corpus extension. Each block covers one axis the matrix above does not: cutoff at every segment
# count, valence 5 and 6 apexes and a non-regular valence-4 octahedron, partial selections by
# edge weight, vertex weight and vertex group, angle limits, seams and sharp edges, harden normals
# with and without custom normals, face strength, miter combinations, and CUSTOM profiles.
for segment in range(1, 7):
    cases.append(evaluate(objects['cube'], 'cube', 'cube_cutoff_s%d_p50' % segment, {
        'segments': segment, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'vmesh_method': 'CUTOFF',
    }))
for shape in ('pentagonal_pyramid', 'hexagonal_pyramid', 'irregular_octahedron'):
    for segment in (1, 3):
        for vmesh in ('ADJ', 'CUTOFF'):
            for affect in ('EDGES', 'VERTICES'):
                cases.append(evaluate(objects[shape], shape, '%s_s%d_%s_%s' % (
                    shape, segment, vmesh.lower(), affect.lower()), {
                        'segments': segment, 'profile': 0.5, 'affect': affect,
                        'offset_type': 'OFFSET', 'vmesh_method': vmesh,
                    }))
cases.extend([
    evaluate(objects['pentagonal_pyramid'], 'pentagonal_pyramid', 'pentagonal_pyramid_selected_edges_s2', {
        'segments': 2, 'profile': 0.8, 'affect': 'EDGES', 'offset_type': 'WIDTH', 'width': 0.2,
        'limit_method': 'WEIGHT', 'face_strength_mode': 'FSTR_NEW',
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_selected_vertices_s2', {
                 'segments': 2, 'profile': 0.8, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
                 'limit_method': 'WEIGHT', 'mark_seam': True,
             }),
    evaluate(objects['cube'], 'cube', 'cube_selected_vertices_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
        'limit_method': 'WEIGHT',
    }),
    evaluate(objects['cube'], 'cube', 'cube_vgroup_edges_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'limit_method': 'VGROUP', 'vertex_group': 'weight_ramp',
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_vgroup_edges_inverted_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
                 'limit_method': 'VGROUP', 'vertex_group': 'weight_inverse',
                 'invert_vertex_group': True,
             }),
    evaluate(objects['hexagonal_pyramid'], 'hexagonal_pyramid',
             'hexagonal_pyramid_vgroup_vertices_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
                 'limit_method': 'VGROUP', 'vertex_group': 'weight_ramp',
                 'miter_outer': 'MITER_ARC', 'miter_inner': 'MITER_ARC',
             }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_angle_limit_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
                 'limit_method': 'ANGLE', 'angle_limit': 0.8,
             }),
    evaluate(objects['cube'], 'cube', 'cube_mark_seam_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET', 'mark_seam': True,
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_mark_seam_sharp_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
                 'mark_seam': True, 'mark_sharp': True,
             }),
    evaluate(objects['hexagonal_pyramid'], 'hexagonal_pyramid',
             'hexagonal_pyramid_mark_sharp_vertices_s3', {
                 'segments': 3, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
                 'mark_sharp': True,
             }),
    evaluate(objects['cube'], 'cube', 'cube_harden_s1_p50', {
        'segments': 1, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'harden_normals': True,
    }),
    evaluate(objects['cube'], 'cube', 'cube_harden_s3_p80', {
        'segments': 3, 'profile': 0.8, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'harden_normals': True,
    }),
    evaluate(objects['cube'], 'cube', 'cube_harden_s6_p50', {
        'segments': 6, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'harden_normals': True,
    }),
    evaluate(objects['cube_plain'], 'cube_plain', 'cube_plain_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
    }),
    evaluate(objects['cube_plain'], 'cube_plain', 'cube_plain_harden_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'harden_normals': True,
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_harden_vertices_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
                 'harden_normals': True,
             }),
    evaluate(objects['pentagonal_pyramid'], 'pentagonal_pyramid', 'pentagonal_pyramid_fstr_new_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'face_strength_mode': 'FSTR_NEW',
    }),
    evaluate(objects['hexagonal_pyramid'], 'hexagonal_pyramid', 'hexagonal_pyramid_fstr_affected_vertices_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
        'face_strength_mode': 'FSTR_AFFECTED', 'material': 0,
    }),
    evaluate(objects['cube'], 'cube', 'cube_fstr_all_s1_p100', {
        'segments': 1, 'profile': 1.0, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'face_strength_mode': 'FSTR_ALL',
    }),
    evaluate(objects['pentagonal_pyramid'], 'pentagonal_pyramid', 'pentagonal_pyramid_miter_patch_arc_s3', {
        'segments': 3, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'miter_outer': 'MITER_PATCH', 'miter_inner': 'MITER_ARC', 'spread': 0.3,
    }),
    evaluate(objects['hexagonal_pyramid'], 'hexagonal_pyramid', 'hexagonal_pyramid_miter_arc_sharp_vertices_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
        'miter_outer': 'MITER_ARC', 'miter_inner': 'MITER_SHARP', 'spread': 0.5,
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_material0_fstr_all_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
                 'material': 0, 'face_strength_mode': 'FSTR_ALL',
             }),
    evaluate(objects['cube'], 'cube', 'cube_custom_free_handles_s3', {
        'segments': 3, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'profile_type': 'CUSTOM',
        'custom_profile': {'points': [
            ((0.7, 0.15), 'FREE', 'FREE', (0.62, 0.08), (0.82, 0.12)),
            ((0.35, 0.6), 'ALIGNED', 'VECTOR', (0.24, 0.72), (0.4, 0.49)),
            ((1.0, 0.0), 'AUTO', 'AUTO'),
            ((0.0, 1.0), 'VECTOR', 'AUTO'),
        ]},
    }),
    evaluate(objects['irregular_octahedron'], 'irregular_octahedron',
             'irregular_octahedron_custom_straight_even_s4', {
                 'segments': 4, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
                 'profile_type': 'CUSTOM',
                 'custom_profile': {
                     'points': [((0.5, 0.5), 'VECTOR', 'VECTOR')],
                     'use_clip': False,
                     'use_sample_straight_edges': True,
                     'use_sample_even_lengths': True,
                 },
             }),
    evaluate(objects['pentagonal_pyramid'], 'pentagonal_pyramid',
             'pentagonal_pyramid_custom_clip_vertices_s2', {
                 'segments': 2, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
                 'profile_type': 'CUSTOM',
                 'custom_profile': {
                     'points': [((0.8, 0.3), 'FREE', 'ALIGNED', (0.7, 0.2), (0.9, 0.4)),
                                ((0.2, 0.9), 'AUTO', 'FREE', (0.12, 0.8), (0.3, 0.96))],
                     'use_clip': True,
                 },
             }),
    evaluate(objects['cube'], 'cube', 'cube_custom_even_s6', {
        'segments': 6, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'profile_type': 'CUSTOM',
        'custom_profile': {'points': [((0.25, 0.5), 'AUTO', 'AUTO')], 'use_sample_even_lengths': True},
    }),
])
# Open-boundary corpus: each surface omits a face, so its rim edges have one face. With EDGES affect
# Blender bevels only manifold edges; boundary edges still join the fan of their endpoint vertices.
def open_boundary_topology(name, faces):
    directed = set()
    for face in faces:
        for first, second in zip(face, face[1:] + face[:1]):
            assert (first, second) not in directed, name + ' has inconsistent face winding'
            directed.add((first, second))
    boundary = {edge_key(first, second) for first, second in directed if (second, first) not in directed}
    manifold = {edge_key(first, second) for first, second in directed if (second, first) in directed}
    assert boundary and manifold, name + ' needs both boundary and manifold edges'
    return boundary, manifold


def vertices_of(edges):
    return {vertex for edge in edges for vertex in edge}


open_topology = {name: open_boundary_topology(name, shape_inputs[name][1])
                 for name in ('open_fan', 'open_pyramid', 'open_box')}
open_boundary_cases = [
    evaluate(objects['open_fan'], 'open_fan', 'open_fan_edges_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
    }),
    evaluate(objects['open_fan'], 'open_fan', 'open_fan_vertices_cutoff_s3', {
        'segments': 3, 'profile': 0.5, 'affect': 'VERTICES', 'offset_type': 'OFFSET',
        'vmesh_method': 'CUTOFF',
    }),
    evaluate(objects['open_pyramid'], 'open_pyramid', 'open_pyramid_vertices_s2', {
        'segments': 2, 'profile': 0.8, 'affect': 'VERTICES', 'offset_type': 'WIDTH', 'width': 0.2,
    }),
    evaluate(objects['open_box'], 'open_box', 'open_box_selected_edges_s2', {
        'segments': 2, 'profile': 0.5, 'affect': 'EDGES', 'offset_type': 'OFFSET',
        'limit_method': 'WEIGHT',
    }),
    evaluate(objects['open_box'], 'open_box', 'open_box_selected_vertices_s3_p80', {
        'segments': 3, 'profile': 0.8, 'affect': 'VERTICES', 'offset_type': 'DEPTH',
        'limit_method': 'WEIGHT',
    }),
]
for case in open_boundary_cases:
    source = inputs[case['shape']]
    assert len(case['output']['vertices']) > len(source['vertices']), case['name']
    assert len(case['output']['faces']) > len(source['faces']), case['name']
# Partial selections beside the open rim of open_box (the rim is the vertex set of its boundary edges).
box_boundary, box_manifold = open_topology['open_box']
box_rim = vertices_of(box_boundary)
selected_box_edges = {edge_key(*edge['vertices']) for edge in inputs['open_box']['edges']
                      if edge['bevel_weight'] == f32(1.0)} & box_manifold
assert selected_box_edges and box_manifold - selected_box_edges
assert vertices_of(selected_box_edges) & box_rim and vertices_of(box_manifold - selected_box_edges) & box_rim
selected_box_vertices = {index for index, weight in enumerate(inputs['open_box']['vertex_bevel_weight'])
                         if weight == f32(1.0)}
assert selected_box_vertices & box_rim and box_rim - selected_box_vertices
cases.extend(open_boundary_cases)
assert len({case['name'] for case in cases}) == len(cases)



def geometry_signature(output):
    positions = [tuple(f32_value(component) for component in row[0]) for row in output['vertices']]
    rings = tuple(sorted(tuple(positions[corner[0]] for corner in face[3]) for face in output['faces']))
    return tuple(sorted(set(positions))), rings


outer_reference = geometry_signature(reflex_miter_cases[('MITER_SHARP', 'MITER_SHARP', 0.1)]['output'])
for miter_outer in ('MITER_PATCH', 'MITER_ARC'):
    assert geometry_signature(reflex_miter_cases[(miter_outer, 'MITER_SHARP', 0.1)]['output']) != outer_reference
inner_arc_signatures = []
for spread in (0.05, 0.5, 1.0):
    sharp_geometry = geometry_signature(
        reflex_miter_cases[('MITER_SHARP', 'MITER_SHARP', spread)]['output'])
    arc_geometry = geometry_signature(
        reflex_miter_cases[('MITER_SHARP', 'MITER_ARC', spread)]['output'])
    assert arc_geometry != sharp_geometry
    inner_arc_signatures.append(arc_geometry)
assert len(set(inner_arc_signatures)) == len(inner_arc_signatures)
for width_type in ('OFFSET', 'WIDTH', 'PERCENT', 'ABSOLUTE'):
    assert geometry_signature(clamp_pairs[(width_type, 'small')][True]) == geometry_signature(
        clamp_pairs[(width_type, 'small')][False])
    assert geometry_signature(clamp_pairs[(width_type, 'large')][True]) != geometry_signature(
        clamp_pairs[(width_type, 'large')][False])
custom_profile_case = next(case for case in cases if case['name'] == 'cube_custom_profile_default_line')
profile_points = custom_profile_case['parameters']['custom_profile']['points']
assert [
    {key: point[key] for key in ('location', 'handle_type_1', 'handle_type_2')}
    for point in profile_points
] == [
    {'location': [1.0, 0.0], 'handle_type_1': 'AUTO', 'handle_type_2': 'AUTO'},
    {'location': [0.0, 1.0], 'handle_type_1': 'AUTO', 'handle_type_2': 'AUTO'},
]
assert all(len(point['handle_1']) == 2 and len(point['handle_2']) == 2 for point in profile_points)
assert custom_profile_case['parameters']['custom_profile']['use_clip'] is True
assert custom_profile_case['parameters']['custom_profile']['use_sample_straight_edges'] is False
assert custom_profile_case['parameters']['custom_profile']['use_sample_even_lengths'] is False

assert {case['parameters']['miter_outer'] for case in cases} == {
    'MITER_SHARP', 'MITER_PATCH', 'MITER_ARC'}
assert {case['parameters']['miter_inner'] for case in cases} == {'MITER_SHARP', 'MITER_ARC'}
assert {case['parameters']['vmesh_method'] for case in cases} == {'ADJ', 'CUTOFF'}
assert {case['parameters']['limit_method'] for case in cases} == {
    'NONE', 'ANGLE', 'WEIGHT', 'VGROUP'}
assert {case['parameters']['face_strength_mode'] for case in cases} == {
    'FSTR_NONE', 'FSTR_NEW', 'FSTR_AFFECTED', 'FSTR_ALL'}
assert {case['parameters']['material'] for case in cases} == {-1, 0, 1}
assert any(case['parameters']['mark_sharp'] for case in cases)
assert any(case['parameters']['mark_seam'] for case in cases)
assert any(case['parameters']['harden_normals'] for case in cases)
for case in cases:
    if case['parameters']['face_strength_mode'] != 'FSTR_NONE':
        assert any(face[2] is not None for face in case['output']['faces'])
        assert all(face[2] in (None, 16384, -16384, 0) for face in case['output']['faces'])
# Blender's bevel_set_weighted_normal_face_strength: corner faces (triangles here) are WEAK,
# strip faces (quads) are MEDIUM, and the six source faces (quads) are STRONG.
fstr_faces = next(case for case in cases if case['name'] == 'cube_fstr_all_s1_p100')['output']['faces']
assert sorted({(len(face[3]), face[2]) for face in fstr_faces}) == [(3, -16384), (4, 0), (4, 16384)]
assert sum(1 for face in fstr_faces if face[2] == 16384) == 6
assert any(edge[2] for case in cases for edge in case['output']['edges'])
assert any(edge[3] for case in cases for edge in case['output']['edges'])
assert any(face[1] is False for case in cases for face in case['output']['faces'])

build_hash = bpy.app.build_hash
if isinstance(build_hash, bytes):
    build_hash = build_hash.decode('ascii')
result = {
    'schema_version': SCHEMA_VERSION,
    'blender_version': bpy.app.version_string,
    'blender_build_hash': str(build_hash),
    'float_encoding': 'little-endian IEEE-754 binary32 hex',
    'mesh_encoding': {
        'vertex': ['position_bits[3]', 'group_weight_bits[]'],
        'edge': ['canonical_vertex_id[2]', 'use_seam', 'use_edge_sharp', 'bevel_weight_edge_bits_or_null'],
        'face': ['material_index', 'use_smooth', '__mod_weightednormals_faceweight_or_null',
                 'corner_ring'],
        'corner': ['canonical_vertex_id', 'uv_x_bits', 'uv_y_bits',
                   'normal_x_bits', 'normal_y_bits', 'normal_z_bits'],
        'canonicalization': 'vertices sort numerically by (position, group weights, Blender vertex index), preserving distinct coincident rows; '
                            'edges sort by (ids, seam, sharp, weight); faces sort by (ring ids, material, '
                            'smooth, strength); each ring starts at the lexicographic minimum rotation over '
                            '(position, vertex id)',
        'hidden_elements': 'none; the generator rejects hidden output elements',
        'sharp_face': 'Blender sharp_face equals not use_smooth; the generator rejects disagreement',
    },
    'normal_quantization': normal_quantization_probe(),
    'inputs': inputs,
    'cases': cases,
}
text = json.dumps(result, separators=(',', ':')) + '\n'
with open(FIXTURE, 'w') as fixture_file:
    fixture_file.write(text)
with open(os.path.join(OUT, 'bevel_parity_oracle.json'), 'w') as output_file:
    output_file.write(text)
print('[ok] bevel_parity_oracle: %d cases, fixture=%s' % (len(cases), FIXTURE))
