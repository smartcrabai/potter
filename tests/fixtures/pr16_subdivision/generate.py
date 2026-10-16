import bpy
import json
import math
import os
import sys


ROUND_DIGITS = 7


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
def max_rounding_error(value):
    if isinstance(value, dict):
        return max((max_rounding_error(item) for item in value.values()), default=0.0)
    if isinstance(value, (list, tuple)):
        return max((max_rounding_error(item) for item in value), default=0.0)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return 0.0
    if isinstance(value, (int, float)):
        return abs(float(value) - round(float(value), ROUND_DIGITS))
    if hasattr(value, "to_tuple"):
        return max_rounding_error(value.to_tuple())
    raise TypeError(type(value).__name__)


def edge_key(a, b):
    return tuple(sorted((int(a), int(b))))


def make_mesh(vertices, faces, seams=(), creases=()):
    uv_layers = [{"name": "UVMap", "values": []}, {"name": "Secondary", "values": []}]
    custom_normals = []
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
            custom_normals.append([component / length for component in vector])
    return {
        "vertices": vertices,
        "faces": faces,
        "uv_layers": uv_layers,
        "smooth": [True for _ in faces],
        "seams": [list(edge) for edge in sorted({edge_key(*edge) for edge in seams})],
        "creases": [[int(a), int(b), float(value)] for a, b, value in creases],
        "custom_normals": custom_normals,
    }


def fixture_set():
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
    grid = make_mesh(grid_vertices, grid_faces, seams)

    quad = make_mesh([[-1.0, -0.8, 0.12], [1.1, -0.8, -0.07],
                      [1.0, 0.9, 0.21], [-0.9, 0.8, -0.16]],
                     [[0, 1, 2, 3]], [(0, 1), (2, 3)])
    cube = make_mesh([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                     [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
                     [(0, 1), (2, 3), (4, 5), (6, 7), (0, 3), (1, 2)],
                     [(0, 1, 0.25), (2, 3, 0.75)])
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
    meshes = {"curved_grid": grid, "quad": quad, "cube": cube,
              "ngon": ngon, "nonmanifold": nonmanifold}
    cases = []
    for level, quality in zip(range(1, 6), range(1, 6)):
        cases.append({"name": f"level_{level}_quality_{quality}", "mesh": "quad",
                      "params": {"levels": level, "quality": quality, "boundary_smooth": "ALL",
                                 "uv_smooth": "SMOOTH_ALL", "use_creases": False,
                                 "use_custom_normals": False, "use_limit_surface": True}})
    cases.append({"name": "quality_6", "mesh": "quad",
                  "params": {"levels": 3, "quality": 6, "boundary_smooth": "PRESERVE_CORNERS",
                             "uv_smooth": "SMOOTH_ALL", "use_creases": False,
                             "use_custom_normals": False, "use_limit_surface": True}})
    uv_modes = ["NONE", "PRESERVE_CORNERS", "PRESERVE_CORNERS_AND_JUNCTIONS",
                "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE", "PRESERVE_BOUNDARIES", "SMOOTH_ALL"]
    for quality, uv_mode in enumerate(uv_modes, 1):
        cases.append({"name": f"uv_{uv_mode.lower()}", "mesh": "cube",
                      "params": {"levels": 2, "quality": quality, "boundary_smooth": "ALL",
                                 "uv_smooth": uv_mode, "use_creases": False,
                                 "use_custom_normals": False, "use_limit_surface": True}})
    for value in (0.0, 0.25, 0.5, 1.0):
        name = f"crease_{str(value).replace('.', '_')}"
        meshes[name] = make_mesh(grid_vertices, grid_faces, seams,
                                 [(5, 6, value), (1, 2, value), (5, 9, value)])
        cases.append({"name": name, "mesh": name,
                      "params": {"levels": 2, "quality": 4, "boundary_smooth": "ALL",
                                 "uv_smooth": "PRESERVE_BOUNDARIES", "use_creases": True,
                                 "use_custom_normals": False, "use_limit_surface": True}})
    for boundary in ("ALL", "PRESERVE_CORNERS"):
        cases.append({"name": f"boundary_{boundary.lower()}", "mesh": "curved_grid",
                      "params": {"levels": 2, "quality": 5, "boundary_smooth": boundary,
                                 "uv_smooth": "PRESERVE_BOUNDARIES", "use_creases": False,
                                 "use_custom_normals": False, "use_limit_surface": True}})
    for name, quality in (("cube", 2), ("ngon", 3), ("nonmanifold", 6)):
        cases.append({"name": f"topology_{name}", "mesh": name,
                      "params": {"levels": 2, "quality": quality, "boundary_smooth": "PRESERVE_CORNERS",
                                 "uv_smooth": "PRESERVE_CORNERS_AND_JUNCTIONS", "use_creases": True,
                                 "use_custom_normals": False, "use_limit_surface": True}})
    for enabled in (False, True):
        cases.append({"name": f"custom_normals_{'enabled' if enabled else 'disabled'}",
                      "mesh": "curved_grid",
                      "params": {"levels": 2, "quality": 4, "boundary_smooth": "ALL",
                                 "uv_smooth": "PRESERVE_BOUNDARIES", "use_creases": False,
                                 "use_custom_normals": enabled, "use_limit_surface": True}})
    return meshes, cases


def output_mesh(obj, params):
    modifier = obj.modifiers.new("OracleSubdivision", "SUBSURF")
    modifier.subdivision_type = "CATMULL_CLARK"
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
        crease = mesh.attributes.get("crease_edge")
        edges = [{"v": list(map(int, edge.vertices)), "seam": bool(edge.use_seam),
                  "crease": 0.0 if crease is None else float(crease.data[edge.index].value)}
                 for edge in mesh.edges]
        faces = []
        for polygon in mesh.polygons:
            loops = list(polygon.loop_indices)
            face = {"v": [int(mesh.loops[index].vertex_index) for index in loops],
                    "uvs": {name: [uv[name][index] for index in loops] for name in uv},
                    "smooth": bool(polygon.use_smooth)}
            if params["use_custom_normals"]:
                face["normals"] = [list(mesh.corner_normals[index].vector) for index in loops]
            faces.append(face)
        return {"vertices": [list(vertex.co) for vertex in mesh.vertices], "edges": edges,
                "faces": faces, "has_custom_normals": bool(mesh.has_custom_normals),
                "normals_domain": str(mesh.normals_domain)}
    finally:
        evaluated.to_mesh_clear()


def make_object(name, spec):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(spec["vertices"], [], spec["faces"])
    mesh.update()
    seam_set = {edge_key(*edge) for edge in spec["seams"]}
    crease_values = {edge_key(*edge[:2]): float(edge[2]) for edge in spec["creases"]}
    crease = mesh.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
    for edge in mesh.edges:
        key = edge_key(*edge.vertices)
        edge.use_seam = key in seam_set
        crease.data[edge.index].value = crease_values.get(key, 0.0)
    for spec_layer in spec["uv_layers"]:
        layer = mesh.uv_layers.new(name=spec_layer["name"])
        for index, value in enumerate(spec_layer["values"]):
            layer.data[index].uv = value
    for index, polygon in enumerate(mesh.polygons):
        polygon.use_smooth = spec["smooth"][index]
    mesh.normals_split_custom_set(spec["custom_normals"])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def main():
    if bpy.app.version != (5, 2, 2):
        raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 1:
        raise RuntimeError("usage: generate.py -- <oracle.json>")
    output_path = os.path.abspath(args[0])
    meshes, cases = fixture_set()
    results = []
    for case in cases:
        obj = make_object(case["name"], meshes[case["mesh"]])
        result = dict(case)
        result["expected"] = output_mesh(obj, case["params"])
        results.append(result)
        bpy.data.objects.remove(obj, do_unlink=True)
    oracle = {"generator": "tests/fixtures/pr16_subdivision/generate.py",
              "blender_version": bpy.app.version_string, "round_digits": ROUND_DIGITS,
              "meshes": meshes, "cases": results}
    oracle["max_serialization_rounding_error"] = f"{max_rounding_error(oracle):.12g}"
    with open(output_path, "w", encoding="utf-8", newline="\n") as output:
        json.dump(rounded(oracle), output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(f"wrote {len(results)} subdivision cases to {output_path}")


main()
