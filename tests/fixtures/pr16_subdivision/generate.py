import bpy
import json
import math
import os
import sys


ROUND_DIGITS = 7
DEFAULTS = {"levels": 2, "quality": 2, "boundary_smooth": "ALL", "uv_smooth": "PRESERVE_BOUNDARIES",
            "use_creases": False, "use_custom_normals": False, "use_limit_surface": True,
            "subdivision_type": "CATMULL_CLARK"}
UV_MODES = ["NONE", "PRESERVE_CORNERS", "PRESERVE_CORNERS_AND_JUNCTIONS",
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE", "PRESERVE_BOUNDARIES", "SMOOTH_ALL"]
BOUNDARY_MODES = ["ALL", "PRESERVE_CORNERS"]


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, ROUND_DIGITS)
    if hasattr(value, "to_tuple"):
        return rounded(value.to_tuple())
    if hasattr(value, "__iter__"):
        return [rounded(item) for item in value]
    raise TypeError(type(value).__name__)


def edge_key(a, b):
    return tuple(sorted((int(a), int(b))))


def face_edge_counts(faces):
    counts = {}
    for face in faces:
        for index, vertex in enumerate(face):
            key = edge_key(vertex, face[(index + 1) % len(face)])
            counts[key] = counts.get(key, 0) + 1
    return counts


def all_edges(faces):
    return sorted(face_edge_counts(faces))


def border_edges(faces):
    return [edge for edge, count in sorted(face_edge_counts(faces).items()) if count == 1]


def make_mesh(vertices, faces, seams=(), creases=(), loose_edges=(), smooth=None, materials=None,
              sharp_edges=(), vertex_creases=(), rotated_storage=False, custom_normals=None):
    uv_layers = [{"name": "UVMap", "values": []}, {"name": "Secondary", "values": []}]
    generated_normals = []
    for face_index, face in enumerate(faces):
        shift = 0.37 if face_index % 2 else 0.0
        for corner, vertex_index in enumerate(face):
            x, y, z = vertices[vertex_index]
            uv_layers[0]["values"].append([0.5 + x * 0.31 + shift, 0.5 + y * 0.29 - shift * 0.4])
            uv_layers[1]["values"].append([0.2 - x * 0.17 + shift * 0.6, 0.8 + y * 0.23 + shift])
            vector = (0.13 + 0.025 * face_index + 0.02 * corner,
                      -0.11 + 0.015 * vertex_index,
                      1.0 + 0.03 * x - 0.02 * y + 0.01 * z)
            length = math.sqrt(sum(component * component for component in vector))
            generated_normals.append([component / length for component in vector])
    return {
        "vertices": vertices,
        "faces": faces,
        "loose_edges": [list(edge) for edge in loose_edges],
        "uv_layers": uv_layers,
        "smooth": [True for _ in faces] if smooth is None else list(smooth),
        "materials": [0 for _ in faces] if materials is None else list(materials),
        "vertex_groups": {},
        "seams": [list(edge) for edge in sorted({edge_key(*edge) for edge in seams})],
        "creases": [[int(a), int(b), float(value)] for a, b, value in creases],
        "sharp_edges": [list(edge) for edge in sorted({edge_key(*edge) for edge in sharp_edges})],
        "vertex_creases": [[int(vertex), float(value)] for vertex, value in vertex_creases],
        "rotated_storage": bool(rotated_storage),
        "order_attributes": [],
        "custom_normals": generated_normals if custom_normals is None else custom_normals,
    }


def grid_meshes():
    grid_vertices = []
    for y in range(3):
        for x in range(4):
            px, py = x - 1.5, y - 1.0
            grid_vertices.append([px, py, 0.12 * px * px + 0.08 * px * py - 0.06 * py * py + 0.03 * px])
    grid_faces = []
    for y in range(2):
        for x in range(3):
            i = y * 4 + x
            grid_faces.append([i, i + 1, i + 5, i + 4])
    seams = []
    for y in range(3):
        for x in range(1, 3):
            if (x + y) % 2 == 0:
                seams.append((y * 4 + x, y * 4 + x + 1))
    for x in range(4):
        if (x + 1) % 2 == 0:
            seams.append((4 + x, 8 + x))
    return grid_vertices, grid_faces, seams


def fixture_set():
    grid_vertices, grid_faces, grid_seams = grid_meshes()
    grid = make_mesh(grid_vertices, grid_faces, grid_seams)
    quad_vertices = [[-1.0, -0.8, 0.12], [1.1, -0.8, -0.07],
                     [1.0, 0.9, 0.21], [-0.9, 0.8, -0.16]]
    quad = make_mesh(quad_vertices, [[0, 1, 2, 3]], [(0, 1), (2, 3)])
    quad["uv_layers"][0]["pinned"] = [True, False, False, False]
    quad["vertex_groups"] = {"Ramp": [0.1, 0.35, 0.65, 0.9]}
    cube_vertices = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    cube_faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                  [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    cube_seams = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 3), (1, 2)]
    cube = make_mesh(cube_vertices, cube_faces, cube_seams, [(0, 1, 0.25), (2, 3, 0.75)])
    cube_flat = make_mesh(cube_vertices, cube_faces, cube_seams, [(0, 1, 0.25), (2, 3, 0.75)],
                          smooth=[False, True, False, True, True, False])
    ngon_vertices = []
    for i in range(7):
        angle = 2.0 * math.pi * i / 7.0
        radius = 1.0 + (0.12 if i % 2 else 0.0)
        ngon_vertices.append([radius * math.cos(angle), radius * math.sin(angle), 0.06 * (i % 3)])
    ngon = make_mesh(ngon_vertices, [list(range(7))], [(0, 1), (2, 3), (4, 5)])
    nonmanifold = make_mesh([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0],
                             [-1, 0, 0], [-1, -1, 0], [0, 1, 0], [0, 2, 1]],
                            [[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 6, 7]],
                            [(0, 1), (2, 3), (4, 5), (6, 7)])
    mixed = make_mesh([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, 0.2], [0.0, 1.0, 0.0],
                       [2.0, 0.5, 0.3], [-0.9, -0.6, 0.1], [-1.4, 0.5, 0.2], [-0.7, 1.5, 0.0]],
                      [[0, 1, 2, 3], [1, 4, 2], [0, 5, 6, 7, 3]],
                      [(0, 1), (3, 0), (6, 7)], [(1, 2, 0.5)])
    loose = make_mesh([[-1.0, -0.8, 0.12], [1.1, -0.8, -0.07], [1.0, 0.9, 0.21],
                       [-0.9, 0.8, -0.16], [2.0, 0.0, 0.3], [3.0, 1.0, 0.1], [-2.0, -2.0, 0.5]],
                      [[0, 1, 2, 3]], [(0, 1)], loose_edges=[(4, 5)])
    # The corpus "mixed" pentagon traverses edge 3-0 in the quad's direction (non-manifold winding);
    # "mixed_consistent" reverses it so every shared edge is traversed once each way.
    mixed_consistent = make_mesh(mixed["vertices"], [[0, 1, 2, 3], [1, 4, 2], [3, 7, 6, 5, 0]],
                                 [(0, 1), (3, 0), (6, 7)], [(1, 2, 0.5)], materials=[0, 1, 2])
    meshes = {"curved_grid": grid, "quad": quad, "cube": cube, "cube_flat": cube_flat,
              "ngon": ngon, "nonmanifold": nonmanifold, "mixed": mixed,
              "mixed_consistent": mixed_consistent, "loose": loose}
    for value in (0.1, 0.25, 0.5, 0.9, 1.0):
        name = f"grid_crease_{str(value).replace('.', '_')}"
        meshes[name] = make_mesh(grid_vertices, grid_faces, grid_seams,
                                 [(5, 6, value), (1, 2, value), (5, 9, value)])
    meshes["cube_crease_mix"] = make_mesh(cube_vertices, cube_faces, cube_seams,
                                          [(0, 1, 0.3), (2, 3, 1.0), (4, 5, 0.6), (1, 2, 0.5)])
    meshes["ngon_crease"] = make_mesh(ngon_vertices, [list(range(7))], [(0, 1), (2, 3), (4, 5)],
                                      [(0, 1, 0.5), (3, 4, 1.0)])
    meshes["nonmanifold_crease"] = make_mesh(meshes["nonmanifold"]["vertices"],
                                             meshes["nonmanifold"]["faces"],
                                             [(0, 1), (2, 3), (4, 5), (6, 7)],
                                             [(0, 1, 0.5), (2, 3, 1.0)])
    meshes["mixed_crease"] = make_mesh(mixed["vertices"], mixed["faces"],
                                       [(0, 1), (3, 0), (6, 7)], [(1, 2, 0.5), (0, 3, 0.8)])
    dependent_plus1 = make_mesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
         [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0]],
        [[0, 1, 2, 3], [0, 3, 4, 5], [0, 5, 6, 7], [0, 7, 8, 1]],
        creases=[(0, 3, 0.5)])
    dependent_plus1["uv_layers"] = [{
        "name": "UVMap",
        "values": [[0.2, 0.4], [0.37, 0.4], [0.37, 0.53], [0.2, 0.53],
                   [0.2, 0.4], [0.2, 0.53], [0.03, 0.53], [0.03, 0.4],
                   [2.2, 0.4], [2.03, 0.4], [2.03, 0.27], [2.2, 0.27],
                   [2.2, 0.4], [2.2, 0.27], [2.37, 0.27], [2.37, 0.4]],
    }]
    meshes["dependent_uv_plus1"] = dependent_plus1
    dependent_plus2 = make_mesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
         [-1, 0, 0], [0, -1, 0]],
        [[0, 1, 2, 3], [0, 3, 4, 5], [0, 5, 6, 1]],
        creases=[(0, 3, 0.5)])
    dependent_plus2["uv_layers"] = [{
        "name": "UVMap",
        "values": [[0.2, 0.4], [0.37, 0.4], [0.37, 0.53], [0.2, 0.53],
                   [0.2, 0.4], [0.2, 0.53], [0.03, 0.53], [0.03, 0.4],
                   [2.2, 0.4], [2.03, 0.4], [2.2, 0.27], [2.37, 0.4]],
    }]
    meshes["dependent_uv_plus2"] = dependent_plus2
    plane_vertices = [[x, y, 0.1 * x * x - 0.05 * y * y + 0.03 * x * y]
                      for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    # Vertex 4 is the interior vertex at the centre; faces are stored top row first.
    plane_faces = [[3, 4, 7, 6], [4, 5, 8, 7], [0, 1, 4, 3], [1, 2, 5, 4]]
    # Sharp edges 3-4 and 1-4 form an L at vertex 4, and both lie in face [0, 1, 4, 3].
    # Uniform custom normals: the setter splits fans only where custom normals disagree, so the
    # generator's varying normals would add sharp edges and erase the L.
    meshes["l_sharp_plane"] = make_mesh(plane_vertices, plane_faces, sharp_edges=[(3, 4), (1, 4)],
                                        custom_normals=[[0.0, 0.0, 1.0]] * 16)
    meshes["plane_crease"] = make_mesh(plane_vertices, plane_faces,
                                       creases=[(a, b, 0.2) for a, b in all_edges(plane_faces)])
    meshes["plane_border_crease"] = make_mesh(
        plane_vertices, plane_faces, creases=[(a, b, 1.0) for a, b in border_edges(plane_faces)])
    # Corner UVs at vertex 4 in face order are a, c, b, then one distinct value. Blender groups
    # a, b and c (a and b are 0.9e-4 apart, b and c 0.6e-4 apart), so the grouping order matters.
    meshes["uv_group"] = make_mesh(plane_vertices, plane_faces)
    uv_values = meshes["uv_group"]["uv_layers"][0]["values"]
    uv_values[1] = [0.0, 0.0]
    uv_values[4] = [1.5e-4, 0.0]
    uv_values[10] = [0.9e-4, 0.0]
    uv_values[15] = [0.5, 0.5]
    meshes["cube_crease_0_2"] = make_mesh(cube_vertices, cube_faces, cube_seams,
                                          [(a, b, 0.2) for a, b in all_edges(cube_faces)])
    meshes["quad_all_crease"] = make_mesh(quad_vertices, [[0, 1, 2, 3]], [(0, 1), (2, 3)],
                                          [(a, b, 1.0) for a, b in all_edges([[0, 1, 2, 3]])])
    # Vertex 0 has two creased edges (crease 0.2, sharpness 0.4 each) and its own vertex crease (0.2,
    # sharpness 0.4). Blender adds the smaller edge sharpness to the vertex's own sharpness, clamped
    # to 10 (topology_refiner_factory.cc 238-244): 0.4 + 0.4 = 0.8. Taking only the larger value gives
    # 0.4. The sharpnesses stay below 1, where the fractional weight, and so the limit, depends on
    # the value.
    meshes["quad_vertex_corner"] = make_mesh(quad_vertices, [[0, 1, 2, 3]], [],
                                             [(0, 1, 0.2), (0, 3, 0.2)],
                                             vertex_creases=[(0, 0.2)])
    meshes["loose_corner"] = make_mesh(quad_vertices + [[2.0, 0.2, 0.3]], [[0, 1, 2, 3]],
                                       [(0, 1), (2, 3)], loose_edges=[(2, 4)])
    # Loose chains that bend, so the B-spline differs from the midpoint: a free chain whose end points
    # have one edge (extrapolated neighbours), and a chain attached to face corner 2, whose interior
    # vertices have two edges each (real neighbours through the chain).
    meshes["loose_chain"] = make_mesh(quad_vertices + [[2.0, 0.2, 0.3], [2.6, 1.2, 0.5],
                                                       [1.8, 1.9, -0.4], [0.6, 2.4, 0.2]],
                                      [[0, 1, 2, 3]], [], loose_edges=[(4, 5), (5, 6), (6, 7)])
    meshes["loose_corner_chain"] = make_mesh(quad_vertices + [[2.0, 0.2, 0.3], [2.6, 1.2, 0.5],
                                                              [1.8, 1.9, -0.4]],
                                             [[0, 1, 2, 3]], [(0, 1), (2, 3)],
                                             loose_edges=[(2, 4), (4, 5), (5, 6)])
    meshes["cube_rotated_storage"] = make_mesh(cube_vertices, cube_faces, cube_seams,
                                               [(0, 1, 0.25), (2, 3, 0.75)], rotated_storage=True)
    meshes["ngon_rotated_storage"] = make_mesh(ngon_vertices, [list(range(7))],
                                               [(0, 1), (2, 3), (4, 5)], rotated_storage=True)
    meshes["mixed_rotated_storage"] = make_mesh(mixed["vertices"], mixed["faces"],
                                                [(0, 1), (3, 0), (6, 7)], [(1, 2, 0.5)],
                                                rotated_storage=True)
    # Each generic value identifies its element: POINT by vertex index, EDGE by edge index, FACE by face
    # index, and CORNER by Blender loop index, so a wrong slot or corner frame changes a value. The n-gon
    # also carries integer and quaternion data, which Blender mixes per ptex slot.
    float_orders = [
        {"name": "OrderPoint", "domain": "POINT", "type": "FLOAT"},
        {"name": "OrderEdge", "domain": "EDGE", "type": "FLOAT"},
        {"name": "OrderFace", "domain": "FACE", "type": "FLOAT"},
        {"name": "OrderCorner", "domain": "CORNER", "type": "FLOAT"},
    ]
    for name in ("quad", "curved_grid", "mixed_rotated_storage"):
        meshes[name]["order_attributes"] = float_orders
    meshes["ngon"]["order_attributes"] = float_orders + [
        {"name": "OrderInt", "domain": "POINT", "type": "INT"},
        {"name": "OrderIntCorner", "domain": "CORNER", "type": "INT"},
        {"name": "OrderQuaternion", "domain": "POINT", "type": "QUATERNION"},
    ]
    # BYTE_COLOR layers hold sRGB bytes that Blender interpolates per slot and truncates (encode_byte_colors).
    byte_colors = [
        {"name": "ByteColorPoint", "domain": "POINT"},
        {"name": "ByteColorCorner", "domain": "CORNER"},
    ]
    meshes["quad"]["byte_colors"] = byte_colors
    meshes["ngon"]["byte_colors"] = byte_colors
    meshes["vertex_crease_grid"] = make_mesh(grid_vertices, grid_faces, grid_seams,
                                             vertex_creases=[(5, 1.0), (6, 0.5), (1, 0.3)])
    meshes["vertex_crease_cube"] = make_mesh(cube_vertices, cube_faces, cube_seams,
                                             vertex_creases=[(0, 1.0), (6, 0.5)])
    return meshes


def build_cases():
    cases = []

    def add(name, mesh, **overrides):
        params = dict(DEFAULTS)
        params.update(overrides)
        cases.append({"name": name, "mesh": mesh, "params": params})

    # Quadrilateral limit surface: every level and quality pairing, plus L>Q and L<Q.
    for level in range(1, 6):
        add(f"level_{level}_quality_{level}", "quad", levels=level, quality=level,
            boundary_smooth="ALL", uv_smooth="SMOOTH_ALL", use_custom_normals=False)
    add("quality_6", "quad", levels=3, quality=6, boundary_smooth="PRESERVE_CORNERS",
        uv_smooth="SMOOTH_ALL")
    add("quad_l3_quality_1", "quad", levels=3, quality=1, uv_smooth="SMOOTH_ALL")
    add("quad_l1_quality_6", "quad", levels=1, quality=6, uv_smooth="SMOOTH_ALL")
    add("quad_l4_quality_2", "quad", levels=4, quality=2, uv_smooth="SMOOTH_ALL")
    add("quad_l2_quality_10", "quad", levels=2, quality=10, uv_smooth="SMOOTH_ALL")
    # Exercise every finite quality at every output level on the same curved quad.
    existing_quad_pairs = {
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (3, 6), (3, 1), (1, 6), (4, 2),
    }
    for level in range(1, 6):
        for quality in range(1, 7):
            if (level, quality) not in existing_quad_pairs:
                add(f"quad_l{level}_q{quality}", "quad", levels=level, quality=quality,
                    boundary_smooth="ALL", uv_smooth="SMOOTH_ALL", use_custom_normals=False)
    for level in (4, 5):
        add(f"quad_default_uv_l{level}", "quad", levels=level, quality=2,
            boundary_smooth="ALL", uv_smooth="PRESERVE_BOUNDARIES")

    # Levels zero is the identity and quality does not matter.
    add("levels_0_cube", "cube", levels=0, quality=3)
    add("levels_0_grid", "curved_grid", levels=0, quality=3, boundary_smooth="PRESERVE_CORNERS")

    # Every uv_smooth mode on the cube (interior extraordinary vertices) and on other topology.
    for quality, uv_mode in enumerate(UV_MODES, 1):
        add(f"uv_{uv_mode.lower()}", "cube", levels=2, quality=quality, uv_smooth=uv_mode)
    for mesh, quality in (("curved_grid", 2), ("ngon", 3), ("nonmanifold", 3), ("mixed", 2)):
        for uv_mode in UV_MODES:
            add(f"{mesh}_uv_{uv_mode.lower()}", mesh, levels=2, quality=quality, uv_smooth=uv_mode,
                boundary_smooth="PRESERVE_CORNERS")

    # Boundary rules across levels and qualities on boundary-rich topology.
    for mesh in ("curved_grid", "mixed", "nonmanifold"):
        for boundary in BOUNDARY_MODES:
            for level, quality in ((1, 1), (1, 6), (3, 1), (3, 3)):
                add(f"{mesh}_{boundary.lower()}_l{level}_q{quality}", mesh, levels=level,
                    quality=quality, boundary_smooth=boundary, uv_smooth="PRESERVE_BOUNDARIES")

    # Semi-sharp and infinite creases, crease rules across boundary modes.
    for value in (0.1, 0.25, 0.5, 0.9, 1.0):
        name = f"grid_crease_{str(value).replace('.', '_')}"
        for level, quality in ((2, 4), (3, 1)):
            add(f"{name}_l{level}_q{quality}", name, levels=level, quality=quality,
                use_creases=True, uv_smooth="PRESERVE_BOUNDARIES")
    add("grid_crease_preserve_corners", "grid_crease_0_5", levels=2, quality=4,
        use_creases=True, boundary_smooth="PRESERVE_CORNERS")
    add("cube_crease_mix_l1_q1", "cube_crease_mix", levels=1, quality=1, use_creases=True)
    add("cube_crease_mix_l3_q2", "cube_crease_mix", levels=3, quality=2, use_creases=True)
    add("cube_crease_mix_l2_q6", "cube_crease_mix", levels=2, quality=6, use_creases=True,
        boundary_smooth="PRESERVE_CORNERS", uv_smooth="PRESERVE_CORNERS")
    add("cube_crease_disabled", "cube_crease_mix", levels=2, quality=2, use_creases=False)

    # Quad n-gons and triangles at several levels.
    for level, quality in ((1, 1), (2, 3), (3, 6), (2, 1)):
        add(f"ngon_l{level}_q{quality}", "ngon", levels=level, quality=quality)
    add("ngon_crease_l2_q3", "ngon_crease", levels=2, quality=3, use_creases=True)
    add("mixed_crease_l2_q2", "mixed_crease", levels=2, quality=2, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("dependent_uv_plus1_crease", "dependent_uv_plus1", levels=3, quality=3,
        use_creases=True, uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("dependent_uv_plus2_crease", "dependent_uv_plus2", levels=3, quality=3,
        use_creases=True, uv_smooth="PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")
    for level, quality, boundary in ((1, 1, "ALL"), (3, 2, "PRESERVE_CORNERS"), (2, 6, "ALL")):
        add(f"mixed_consistent_l{level}_q{quality}_{boundary.lower()}", "mixed_consistent",
            levels=level, quality=quality, boundary_smooth=boundary, uv_smooth="SMOOTH_ALL",
            use_creases=True)
    add("custom_normals_mixed_consistent", "mixed_consistent", levels=2, quality=2,
        use_custom_normals=True)
    add("nolimit_mixed_consistent_l2", "mixed_consistent", levels=2, use_limit_surface=False)

    # Non-manifold fans, with and without creases.
    for level, quality in ((1, 2), (2, 6), (3, 1)):
        add(f"topology_nonmanifold_l{level}_q{quality}", "nonmanifold", levels=level,
            quality=quality, boundary_smooth="PRESERVE_CORNERS",
            uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS", use_creases=True)
    add("nonmanifold_crease_l2_q3", "nonmanifold_crease", levels=2, quality=3, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")

    # Limit surface disabled: uniform refinement, quality ignored.
    for mesh, level in (("quad", 1), ("quad", 3), ("cube", 1), ("cube", 3), ("curved_grid", 2),
                        ("ngon", 2), ("nonmanifold", 2), ("mixed", 2)):
        add(f"nolimit_{mesh}_l{level}", mesh, levels=level, quality=4, use_limit_surface=False,
            uv_smooth="SMOOTH_ALL", use_creases=True)

    # Custom normals with and without limit surface, on every topology class.
    for enabled in (False, True):
        add(f"custom_normals_{'enabled' if enabled else 'disabled'}", "curved_grid", levels=2,
            quality=4, use_custom_normals=enabled)
    add("custom_normals_cube_l2_q2", "cube", levels=2, quality=2, use_custom_normals=True)
    add("custom_normals_cube_nolimit", "cube", levels=2, quality=2, use_custom_normals=True,
        use_limit_surface=False)
    add("custom_normals_ngon", "ngon", levels=2, quality=3, use_custom_normals=True)
    add("custom_normals_nonmanifold", "nonmanifold", levels=2, quality=3, use_custom_normals=True,
        use_creases=True)
    add("custom_normals_mixed", "mixed", levels=2, quality=2, use_custom_normals=True)
    add("custom_normals_flat_faces", "cube_flat", levels=2, quality=2, use_custom_normals=True)

    # Bilinear subdivision keeps linear positions and UVs.
    for mesh, level in (("cube", 1), ("cube", 3), ("curved_grid", 2), ("ngon", 2),
                        ("nonmanifold", 2), ("mixed", 2)):
        add(f"simple_{mesh}_l{level}", mesh, levels=level, subdivision_type="SIMPLE",
            uv_smooth="PRESERVE_BOUNDARIES", use_creases=True)
    add("simple_cube_custom", "cube", levels=2, subdivision_type="SIMPLE",
        use_custom_normals=True)

    # Loose edges and vertices are carried through and split at their midpoints.
    add("loose_l2_q2", "loose", levels=2, quality=2, use_creases=True)
    add("loose_l1_q4", "loose", levels=1, quality=4, use_limit_surface=True)
    add("flat_cube_l2_q3", "cube_flat", levels=2, quality=3)
    # L-shaped infinite crease, with and without the limit surface, levels 1 and 2. Level 2 keeps the
    # two perpendicular sharp edges at interior vertex 4 sharing face [0, 1, 4, 3].
    add("l_sharp_plane_l1", "l_sharp_plane", levels=1, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("l_sharp_plane_l2", "l_sharp_plane", levels=2, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("l_sharp_plane_l2_concave", "l_sharp_plane", levels=2, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")
    add("l_sharp_plane_l1_nolimit", "l_sharp_plane", levels=1, use_creases=True,
        use_limit_surface=False, uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("l_sharp_plane_l2_nolimit", "l_sharp_plane", levels=2, use_creases=True,
        use_limit_surface=False, uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    # Limit surface off with every edge creased at 0.2: uniform positions use the crease rules.
    for level in (1, 2):
        add(f"nolimit_cube_crease_0_2_l{level}", "cube_crease_0_2", levels=level,
            use_creases=True, use_limit_surface=False)
        add(f"nolimit_plane_crease_0_2_l{level}", "plane_crease", levels=level,
            use_creases=True, use_limit_surface=False)
    # Rotated storage: each face's stored corners are rotated by (face + 1) mod n, with the Blender
    # loop order attached as corner_order. Blender's own result is the unrotated mesh's.
    add("rotated_storage_cube_l2", "cube_rotated_storage", levels=2, use_creases=True)
    add("rotated_storage_cube_custom_l2", "cube_rotated_storage", levels=2, use_custom_normals=True)
    add("rotated_storage_ngon_custom_l2", "ngon_rotated_storage", levels=2, quality=3,
        use_custom_normals=True)
    add("rotated_storage_mixed_custom_l2", "mixed_rotated_storage", levels=2,
        use_custom_normals=True, uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    # Creases of 1.0 on the border of a plane (corners have exactly two edges) and on every edge of
    # a single quad, with boundary_smooth ALL.
    for level in (1, 2):
        add(f"plane_border_crease_l{level}", "plane_border_crease", levels=level, use_creases=True)
        add(f"quad_all_crease_l{level}", "quad_all_crease", levels=level, use_creases=True)
    # A vertex crease combined with two creased edges at the same corner: the vertex takes its own
    # sharpness plus the smaller edge sharpness (Blender topology_refiner_factory.cc 238-244).
    for level, limit in ((1, True), (2, True), (1, False)):
        add(f"quad_vertex_corner_l{level}{'' if limit else '_nolimit'}", "quad_vertex_corner",
            levels=level, use_creases=True, use_limit_surface=limit)
    # A loose edge touching face corner 2 makes that corner infinitely sharp.
    add("loose_corner_l1_q2", "loose_corner", levels=1, quality=2, use_creases=True)
    add("loose_corner_l2_q4", "loose_corner", levels=2, quality=4, use_creases=True)
    # Bent loose chains: the B-spline through base neighbours, at limit on, limit off, and SIMPLE.
    add("loose_chain_l1", "loose_chain", levels=1)
    add("loose_chain_l2_q4", "loose_chain", levels=2, quality=4)
    add("loose_chain_l2_nolimit", "loose_chain", levels=2, use_limit_surface=False)
    add("loose_chain_simple_l2", "loose_chain", levels=2, subdivision_type="SIMPLE")
    add("loose_corner_chain_l1", "loose_corner_chain", levels=1)
    add("loose_corner_chain_l2_q4", "loose_corner_chain", levels=2, quality=4)
    add("loose_corner_chain_l2_nolimit", "loose_corner_chain", levels=2, use_limit_surface=False)
    add("loose_corner_chain_simple_l2", "loose_corner_chain", levels=2, subdivision_type="SIMPLE")
    # Depth-2 seams with creases: UV selection must follow the same topology as positions.
    add("seams_uv_depth2_junctions", "grid_crease_0_5", levels=2, quality=4, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_AND_JUNCTIONS")
    add("seams_uv_depth2_concave", "grid_crease_0_5", levels=2, quality=4, use_creases=True,
        uv_smooth="PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")
    # Corner UV grouping at vertex 4 (see fixture_set).
    add("uv_group_l1_smooth_all", "uv_group", levels=1, uv_smooth="SMOOTH_ALL")
    add("uv_group_l2_preserve_corners", "uv_group", levels=2, uv_smooth="PRESERVE_CORNERS")
    add("uv_group_l2_smooth_all", "uv_group", levels=2, uv_smooth="SMOOTH_ALL")
    # Vertex creases (crease_vert), used when Use Creases is on.
    add("vertex_crease_grid_l2", "vertex_crease_grid", levels=2, quality=4, use_creases=True)
    add("vertex_crease_grid_l1_q1", "vertex_crease_grid", levels=1, quality=1, use_creases=True)
    add("vertex_crease_grid_nolimit_l2", "vertex_crease_grid", levels=2, use_creases=True,
        use_limit_surface=False)
    add("vertex_crease_cube_l2", "vertex_crease_cube", levels=2, use_creases=True)
    return cases

def output_mesh(obj, params):
    coarse_edges = [list(map(int, edge.vertices)) for edge in obj.data.edges]
    modifier = obj.modifiers.new("OracleSubdivision", "SUBSURF")
    modifier.subdivision_type = params["subdivision_type"]
    modifier.levels = params["levels"]
    modifier.render_levels = params["levels"]
    modifier.quality = params["quality"]
    modifier.boundary_smooth = params["boundary_smooth"]
    modifier.uv_smooth = params["uv_smooth"]
    modifier.use_creases = params["use_creases"]
    modifier.use_custom_normals = params["use_custom_normals"]
    modifier.use_limit_surface = params["use_limit_surface"]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        uv = {layer.name: [list(item.uv) for item in layer.data] for layer in mesh.uv_layers}
        uv_pins = {layer.name: [bool(item.pin_uv) for item in layer.data] for layer in mesh.uv_layers}
        crease = mesh.attributes.get("crease_edge")
        sharp = mesh.attributes.get("sharp_edge")
        edges = [
            {
                "v": list(map(int, edge.vertices)),
                "seam": bool(edge.use_seam),
                "sharp": False if sharp is None else bool(sharp.data[edge.index].value),
                "crease": 0.0 if crease is None else float(crease.data[edge.index].value),
            }
            for edge in mesh.edges
        ]
        faces = []
        for polygon in mesh.polygons:
            loops = list(polygon.loop_indices)
            face = {
                "v": [int(mesh.loops[index].vertex_index) for index in loops],
                "uvs": {name: [uv[name][index] for index in loops] for name in uv},
                "uv_pins": {name: [uv_pins[name][index] for index in loops] for name in uv},
                "smooth": bool(polygon.use_smooth),
                "material": int(polygon.material_index),
            }
            normal_attribute = mesh.attributes.get("custom_normal")
            if normal_attribute is not None:
                face["normal_codes"] = [
                    None if tuple(normal_attribute.data[index].value) == (0, 0)
                    else list(map(int, normal_attribute.data[index].value))
                    for index in loops
                ]
            faces.append(face)
        group_indices = {group.index: group.name for group in obj.vertex_groups}
        group_weights = {
            group.name: [0.0] * len(mesh.vertices) for group in obj.vertex_groups
        }
        for vertex in mesh.vertices:
            for membership in vertex.groups:
                group_name = group_indices.get(membership.group)
                if group_name is not None:
                    group_weights[group_name][vertex.index] = float(membership.weight)
        ordered_attributes = []
        for attribute in mesh.attributes:
            if attribute.name.startswith("Order"):
                ordered_attributes.append({
                    "name": attribute.name, "domain": attribute.domain,
                    "data_type": attribute.data_type,
                    "values": [attribute_value(item.value, attribute.data_type) for item in attribute.data],
                })
        byte_colors = []
        for attribute in mesh.color_attributes:
            if attribute.name.startswith("ByteColor"):
                byte_colors.append({
                    "name": attribute.name, "domain": attribute.domain,
                    "data_type": attribute.data_type,
                    "values": [[int(round(channel * 255)) for channel in item.color_srgb] for item in attribute.data],
                })
        return {"input_edges": coarse_edges, "vertices": [list(vertex.co) for vertex in mesh.vertices], "edges": edges,
                "faces": faces, "vertex_groups": group_weights, "ordered_attributes": ordered_attributes,
                "byte_colors": byte_colors, "has_custom_normals": bool(mesh.has_custom_normals)}
    finally:
        evaluated.to_mesh_clear()


def quaternion_for(index):
    """A unit quaternion (w, x, y, z); for some vertices the angle exceeds pi, so w is negative."""
    angle = 0.3 + 0.37 * index
    axis = (1.0, 0.5 * (index % 3), 0.25 * (index % 5) + 0.1)
    length = math.sqrt(sum(component * component for component in axis))
    scale = math.sin(angle / 2) / length
    return (math.cos(angle / 2), axis[0] * scale, axis[1] * scale, axis[2] * scale)


def order_values(mesh, domain, data_type):
    """Blender values of an order test attribute. POINT, EDGE and FACE index their element; CORNER uses
    the Blender loop index, so every loop carries a value of its own."""
    if domain == "POINT" and data_type == "FLOAT":
        return [index + 0.25 for index in range(len(mesh.vertices))]
    if domain == "POINT" and data_type == "INT":
        return [(index * 3) % 7 for index in range(len(mesh.vertices))]
    if domain == "POINT" and data_type == "QUATERNION":
        return [quaternion_for(index) for index in range(len(mesh.vertices))]
    if domain == "EDGE" and data_type == "FLOAT":
        return [100.25 + index for index in range(len(mesh.edges))]
    if domain == "FACE" and data_type == "FLOAT":
        return [200.25 + index for index in range(len(mesh.polygons))]
    if domain == "CORNER" and data_type == "FLOAT":
        return [300.25 + loop.index for loop in mesh.loops]
    if domain == "CORNER" and data_type == "INT":
        return [(loop.index * 5) % 13 for loop in mesh.loops]
    raise RuntimeError(f"unsupported order test attribute {data_type} on {domain}")


def attribute_value(value, data_type):
    """A generic value in Potter's canonical order: a quaternion is Blender's (w, x, y, z) read as (x, y, z, w)."""
    if data_type == "QUATERNION":
        w, x, y, z = (float(component) for component in value)
        return [x, y, z, w]
    if hasattr(value, "__len__"):
        return [float(component) for component in value]
    return value if isinstance(value, int) else float(value)


def byte_color_values(domain, mesh):
    """Deterministic sRGB bytes (r, g, b, a) of each BYTE_COLOR element, by Blender loop or vertex index. The
    Rust fixture builds the same bytes from the same formula."""
    base = 17 if domain == "POINT" else 53
    count = len(mesh.vertices) if domain == "POINT" else len(mesh.loops)
    return [[(index * 37 + channel * 91 + base) % 256 for channel in range(3)]
            + [(index * 29 + (101 if domain == "POINT" else 7)) % 256] for index in range(count)]



def make_object(name, spec):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(spec["vertices"], [tuple(edge) for edge in spec["loose_edges"]], spec["faces"])
    mesh.update()
    material_indices = spec.get("materials", [0] * len(spec["faces"]))
    for material_index in range(max(material_indices, default=0) + 1):
        mesh.materials.append(bpy.data.materials.new(f"{name}Material{material_index}"))
    seam_set = {edge_key(*edge) for edge in spec["seams"]}
    sharp_set = {edge_key(*edge) for edge in spec["sharp_edges"]}
    crease_values = {edge_key(*edge[:2]): float(edge[2]) for edge in spec["creases"]}
    crease = mesh.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
    for edge in mesh.edges:
        key = edge_key(*edge.vertices)
        edge.use_seam = key in seam_set
        edge.use_edge_sharp = key in sharp_set
        crease.data[edge.index].value = crease_values.get(key, 0.0)
    vertex_crease_values = {int(vertex): float(value) for vertex, value in spec["vertex_creases"]}
    if vertex_crease_values:
        vertex_crease = mesh.attributes.new(name="crease_vert", type="FLOAT", domain="POINT")
        for vertex in mesh.vertices:
            vertex_crease.data[vertex.index].value = vertex_crease_values.get(vertex.index, 0.0)
    for definition in spec.get("order_attributes", []):
        domain = definition["domain"]
        data_type = definition["type"]
        attribute = mesh.attributes.new(name=definition["name"], type=data_type, domain=domain)
        for item, value in zip(attribute.data, order_values(mesh, domain, data_type)):
            item.value = value
    for definition in spec.get("byte_colors", []):
        attribute = mesh.color_attributes.new(name=definition["name"], type="BYTE_COLOR",
                                              domain=definition["domain"])
        for item, bytes_ in zip(attribute.data, byte_color_values(definition["domain"], mesh)):
            item.color_srgb = tuple(channel / 255.0 for channel in bytes_)
    for spec_layer in spec["uv_layers"]:
        layer = mesh.uv_layers.new(name=spec_layer["name"])
        for index, value in enumerate(spec_layer["values"]):
            layer.data[index].uv = value
            if "pinned" in spec_layer:
                layer.data[index].pin_uv = bool(spec_layer["pinned"][index])
    for index, polygon in enumerate(mesh.polygons):
        polygon.use_smooth = spec["smooth"][index]
        polygon.material_index = material_indices[index]
    mesh.normals_split_custom_set(spec["custom_normals"])
    normal_codes = mesh.attributes.get("custom_normal")
    spec["custom_normal_codes"] = [
        None if tuple(value.value) == (0, 0) else list(map(int, value.value))
        for value in normal_codes.data
    ]
    sharp_edge = mesh.attributes.get("sharp_edge")
    spec["sharp_edges"] = [
        list(map(int, edge.vertices))
        for edge in mesh.edges
        if sharp_edge is not None and bool(sharp_edge.data[edge.index].value)
    ]
    spec["seams"] = [
        list(map(int, edge.vertices)) for edge in mesh.edges if edge.use_seam
    ]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for group_name, weights in spec.get("vertex_groups", {}).items():
        if len(weights) != len(mesh.vertices):
            raise RuntimeError(f"{group_name} weights do not match the source vertices")
        group = obj.vertex_groups.new(name=group_name)
        for vertex_index, weight in enumerate(weights):
            if weight > 0.0:
                group.add([vertex_index], float(weight), "REPLACE")
    return obj


def main():
    if bpy.app.version != (5, 2, 2):
        raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 1:
        raise RuntimeError("usage: generate.py -- <oracle.json>")
    output_path = os.path.abspath(args[0])
    meshes = fixture_set()
    cases = build_cases()
    results = []
    for case in cases:
        obj = make_object(case["name"], meshes[case["mesh"]])
        result = dict(case)
        result["expected"] = output_mesh(obj, case["params"])
        results.append(result)
        bpy.data.objects.remove(obj, do_unlink=True)
    oracle = {"generator": "tests/fixtures/pr16_subdivision/generate.py",
              "blender_version": bpy.app.version_string, "round_digits": ROUND_DIGITS,
              "meshes": meshes, "cases": rounded(results)}
    with open(output_path, "w", encoding="utf-8", newline="\n") as output:
        json.dump(oracle, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(f"wrote {len(results)} subdivision cases to {output_path}")


main()
