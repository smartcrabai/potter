# Usage: /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/fixtures/solidify/generate.py
"""Blender 5.2.2 ground truth for the Solidify modifier (EXTRUDE and NON_MANIFOLD).

Writes two files per case:
  inputs/<case>.json    Potter Mesh JSON (serde shape) plus the complete non-UI Solidify parameter set,
                        with the material slot count (`totcol`). Potter reads this file unchanged.
  expected/<case>.json  The evaluated mesh (`depsgraph` `evaluated_get().to_mesh()`) of the same
                        geometry and parameters, recorded by Blender. Nothing here comes from Potter.

Every input is built here, so Blender and Potter receive the identical mesh. Blender supplies the
parameter defaults (read from a fresh modifier), so the corpus never hand-codes Blender's defaults.

Cases (purpose of each, also stored as `purpose` in its input file):
  grid_extrude_default      open grid, thickness only; use_pin_to_last/use_apply_on_spline set (Solidify ignores them)
  grid_offset_fraction      offset 0.37 (not 0 or +-1), use_rim off, sharp and smooth flags on the input
  grid_offset_outward_flip  offset +1, use_flip_normals, use_flat_faces, two UV layers
  grid_even_quality_inward  offset -0.6, use_even_offset, use_quality_normals
  grid_rim_only             use_rim_only with offset +0.5
  grid_rim_off_flat         use_rim off, use_flat_faces, smooth flags
  grid_negative_thickness   thickness -0.2 (Blender accepts negative thickness)
  grid_negative_thickness_even_out  thickness -0.25 with offset +1, use_even_offset, use_flip_normals
  grid_zero_thickness       thickness 0: Blender keeps coincident shells (duplicate positions), not identity
  grid_vg_invert            thickness vertex group "thick" inverted, thickness_vertex_group 0.6
  grid_vg_direct            thickness vertex group "thick" not inverted, thickness_vertex_group 0.4
  grid_shell_rim_groups     output shell/rim vertex-group weights
  grid_mat_slots1           totcol 1 disables offsets; original material indices are preserved
                            even if they exceed the only material slot
  grid_mat_slots3           totcol 3 with negative material_offset and rim offset 2
  grid_mat_slots4_wide_offset  totcol 4 with material offsets beyond the slot range
  grid_creases              input crease_edge values plus edge_crease_inner/outer/rim
  grid_custom_uv            custom normals, two UV layers, named corner color and sharp/seam flags
  grid_loose_edges          open grid with two loose edges (one a diagonal of a face) and a loose vertex pair
  cube_closed_extrude       closed asymmetric cube, inward-default offset, thickness 0.05
  cube_angle_clamp          closed cube, thickness_clamp 0.12 with use_thickness_angle_clamp
  cube_near_clamp           closed cube, thickness 0.5 with thickness_clamp 0.45 (near the edge-length limit)
  near_collinear_angle_clamp  nearly collinear neighbours and a sliver triangle, angle clamp on
  near_collinear_extrude    same mesh with negative thickness and use_quality_normals
  asym_fan_extrude          asymmetric open triangle fan, negative thickness, flip, even offset
  ngon_concave_extrude      concave L-shaped n-gon, flat faces, quality normals
  tee_extrude               three quads sharing one edge (T-edge), EXTRUDE mode
  book4_extrude             four quads sharing one edge (valence 4), EXTRUDE mode, rim only
  tee_nm_fixed_none         T-edge, NON_MANIFOLD, FIXED thickness, boundary NONE, merge threshold 0.001
  tee_nm_even_round         T-edge, NON_MANIFOLD, EVEN, ROUND, bevel_convex 0.5
  book4_nm_constraints_flat valence-4 edge, NON_MANIFOLD, CONSTRAINTS, FLAT, bevel_convex -0.4
  grid_nm_fixed_none        open grid in NON_MANIFOLD mode (boundary edges only)
  asym_fan_nm_round_even    NON_MANIFOLD output preserves named corner color and a seam mark
  grid_nm_empty_groups_flat  flat faces with named but unassigned shell/rim groups (no deform weights)
  grid_nm_flat_no_groups     control for `use_flat_faces` when the mesh has no deform data
  grid_nm_unrelated_vg_empty unrelated weights with default empty vertex_group
  grid_nm_unrelated_vg_missing unrelated weights with unresolved vertex_group name
  grid_nm_null_slot_group  null material slot plus unrelated thickness weights
  cube_nm_constraints_round closed cube in NON_MANIFOLD, CONSTRAINTS, ROUND, bevel_convex 0.25
  ngon_nm_flat_fixed        concave n-gon in NON_MANIFOLD, FLAT, FIXED
  grid_nm_groups_mats_creases  NON_MANIFOLD with inverted thickness group, two material slots, creases
  edges_only_extrude        only loose edges (no faces): Blender duplicates the edges, no faces
  grid_extrude_bevel_weights  EXTRUDE bevel_convex with nonzero input edge and vertex bevel weights
  grid_nm_bevel_weights      NON_MANIFOLD bevel_convex with nonzero input edge and vertex bevel weights
  collinear_triangle_nm_fixed_none  collinear triangle with FIXED thickness and NONE boundary
  quad_nm_sharp_seam_boundary_connector  two-group boundary connectors copy edge flags
  book4_nm_sharp_seam_boundary_closing  larger-group closing boundary edges copy edge flags
"""

import hashlib
import json
import math
import os
import traceback

import bpy

FIXTURES = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(FIXTURES, "inputs")
EXPECTED = os.path.join(FIXTURES, "expected")
assert bpy.app.version == (5, 2, 2), f"need Blender 5.2.2, got {bpy.app.version}"

T, F = True, False

# Non-UI RNA properties of the Solidify modifier. `show_*`, `is_active`, `name`, and the other
# UI or bookkeeping properties are excluded.
PARAM_NAMES = [
    "solidify_mode", "thickness", "thickness_clamp", "use_thickness_angle_clamp",
    "thickness_vertex_group", "offset", "edge_crease_inner", "edge_crease_outer", "edge_crease_rim",
    "material_offset", "material_offset_rim", "vertex_group", "shell_vertex_group",
    "rim_vertex_group", "use_rim", "use_even_offset", "use_quality_normals", "invert_vertex_group",
    "use_flat_faces", "use_flip_normals", "use_rim_only", "nonmanifold_thickness_mode",
    "nonmanifold_boundary_mode", "nonmanifold_merge_threshold", "bevel_convex",
    "use_pin_to_last", "use_apply_on_spline",
]


# ---------------------------------------------------------------- base geometry

def grid_base():
    verts = []
    for j in range(3):
        for i in range(4):
            verts.append((i + 0.12 * j, 0.9 * j + 0.07 * i * i, 0.25 * math.sin(1.3 * i + 0.7 * j)))
    faces = []
    for j in range(2):
        for i in range(3):
            a = j * 4 + i
            faces.append([a, a + 1, a + 5, a + 4])
    return dict(verts=verts, faces=faces, loose=[])


def grid_loose_base():
    base = grid_base()
    base["verts"] = base["verts"] + [(0.5, -0.6, 0.2), (1.5, -0.9, 0.1)]
    base["loose"] = [(12, 13), (0, 5)]
    return base


CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]


def cube_base():
    verts = [(x + 0.3, 0.6 * y - 0.2, 1.4 * z + 0.1) for x, y, z in CUBE_V]
    return dict(verts=verts, faces=[list(f) for f in CUBE_F], loose=[])


def ngon_base():
    verts = [(0, 0, 0), (2, 0, 0), (2, 1, 0.2), (1, 1, 0.1), (1, 2, 0.4), (0, 2, 0)]
    return dict(verts=verts, faces=[[0, 1, 2, 3, 4, 5]], loose=[])


def single_quad_base():
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    return dict(verts=verts, faces=[[0, 1, 2, 3]], loose=[])


def book_base(pages):
    """Quads that all share the edge (0,0,0)-(0,0,1); each page is (direction x, direction y, length)."""
    verts = [(0, 0, 0), (0, 0, 1)]
    faces = []
    for dx, dy, length in pages:
        a = len(verts)
        verts.append((dx * length, dy * length, 0.0))
        verts.append((dx * length, dy * length, 1.0))
        faces.append([0, a, a + 1, 1])
    return dict(verts=verts, faces=faces, loose=[])


def tee_base():
    return book_base([(1.0, 0.0, 1.0), (0.3, 1.0, 0.8), (-0.9, -0.4, 1.2)])


def book4_base():
    return book_base([(1.0, 0.0, 1.0), (0.2, 1.0, 0.9), (-1.0, 0.3, 1.2), (-0.4, -1.0, 0.7)])


def asym_fan_base():
    verts = [(0, 0, 0), (2.0, 0.2, 0.1), (2.6, 1.5, -0.1), (0.9, 2.1, 0.3), (-0.5, 1.2, 0.0), (1.1, 0.9, 0.5)]
    faces = [[0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]]
    return dict(verts=verts, faces=faces, loose=[])


def near_collinear_base():
    verts = [
        (0, 0, 0), (1, 0.0005, 0), (2, 0, 0), (3, 0.0003, 0),
        (0, 1, 0.1), (1, 1, 0.12), (2, 1.0004, 0.1), (3, 1, 0.15),
        (3.00005, 0.2, 0.15),
    ]
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 8, 7]]
    return dict(verts=verts, faces=faces, loose=[])


def collinear_triangle_base():
    return dict(verts=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], faces=[[0, 1, 2]], loose=[])


def edges_only_base():
    return dict(verts=[(0, 0, 0), (1, 0, 0), (1, 1, 0)], faces=[], loose=[(0, 1), (1, 2)])


BASES = {
    "grid": grid_base,
    "grid_loose": grid_loose_base,
    "cube": cube_base,
    "ngon": ngon_base,
    "single_quad": single_quad_base,
    "tee": tee_base,
    "book4": book4_base,
    "asym_fan": asym_fan_base,
    "near_collinear": near_collinear_base,
    "collinear_triangle": collinear_triangle_base,
    "edges_only": edges_only_base,
}


# ---------------------------------------------------------------- derived attributes

def edge_key(a, b):
    return (min(a, b), max(a, b))


def newell(verts, face):
    nx = ny = nz = 0.0
    for k in range(len(face)):
        p = verts[face[k]]
        q = verts[face[(k + 1) % len(face)]]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return normalize([nx, ny, nz])


def normalize(v):
    length = math.sqrt(sum(c * c for c in v))
    assert length > 0.0
    return [c / length for c in v]


def synthetic_normal(normal, f, c):
    """Unit custom normal tilted off the face normal; varies per face f and corner c."""
    t = [math.sin(1.7 * f + 0.9 * c), math.cos(1.3 * f + 0.6 * c), math.sin(0.7 * f + 1.1 * c)]
    d = sum(t[k] * normal[k] for k in range(3))
    tp = normalize([t[k] - d * normal[k] for k in range(3)])
    return normalize([normal[k] + 0.8 * tp[k] for k in range(3)])


def uv_layer(verts, faces, offsets, layer):
    values = []
    for f, face in enumerate(faces):
        for k, v in enumerate(face):
            x, y, z = verts[v]
            loop = offsets[f] + k
            if layer == 0:
                values.append([0.8 * x + 0.1 * y + 0.013 * loop, 0.5 * y - 0.2 * z + 0.007 * loop])
            else:
                values.append([0.3 * (loop % 5) + 0.05 * x, 0.9 - 0.11 * (loop % 7) + 0.02 * y])
    return values


def decorate(base, smooth=False, sharp=(), seam=(), creases=(), materials=None, uv_layers=0,
             custom=False, groups=(), colors=False, bevel_edges=None, bevel_vertices=None):
    """Input attributes for a base mesh. Every value is deterministic and stored in both files."""
    verts, faces, loose = base["verts"], base["faces"], base["loose"]
    edges = []
    seen = set()
    for face in faces:
        for k in range(len(face)):
            key = edge_key(face[k], face[(k + 1) % len(face)])
            if key not in seen:
                seen.add(key)
                edges.append(key)
    for a, b in loose:
        key = edge_key(a, b)
        assert key not in seen, f"loose edge {key} duplicates a face edge"
        seen.add(key)
        edges.append(key)
    edge_ids = {key: i for i, key in enumerate(edges)}
    offsets, total = [], 0
    for face in faces:
        offsets.append(total)
        total += len(face)
    face_smooth = [smooth] * len(faces) if isinstance(smooth, bool) else list(smooth)
    sharp_set = {edge_key(a, b) for a, b in sharp}
    seam_set = {edge_key(a, b) for a, b in seam}
    for key in sharp_set | seam_set:
        assert key in edge_ids, f"marked edge {key} is not an edge"
    crease_map = {}
    for a, b, value in creases:
        assert edge_key(a, b) in edge_ids, f"crease edge {(a, b)} is not an edge"
        crease_map[edge_key(a, b)] = value
    bevel_edge_values = None
    if bevel_edges is not None:
        bevel_edge_values = [0.0] * len(edges)
        for a, b, value in bevel_edges:
            key = edge_key(a, b)
            assert key in edge_ids, f"bevel-weight edge {(a, b)} is not an edge"
            bevel_edge_values[edge_ids[key]] = value
    bevel_vertex_values = None
    if bevel_vertices is not None:
        assert len(bevel_vertices) == len(verts)
        bevel_vertex_values = list(bevel_vertices)
    face_materials = list(materials) if materials is not None else [0] * len(faces)
    uvs = [uv_layer(verts, faces, offsets, layer) for layer in range(uv_layers)]
    normals = None
    if custom:
        normals = []
        for f, face in enumerate(faces):
            n = newell(verts, face)
            for k in range(len(face)):
                normals.append(synthetic_normal(n, f, k))
    color_values = None
    if colors:
        color_values = []
        for f, face in enumerate(faces):
            for k, vertex in enumerate(face):
                color_values.append([
                    0.11 + 0.03 * ((vertex + f) % 7),
                    0.17 + 0.04 * ((2 * vertex + k) % 9),
                    0.23 + 0.02 * ((vertex + 3 * k + f) % 11),
                    0.5 + 0.1 * ((f + k) % 5),
                ])
    group_values = {}
    for name in groups:
        assert name == "thick"
        group_values[name] = [((7 * i + 3) % 11) / 10.0 for i in range(len(verts))]
    return dict(verts=verts, faces=faces, edges=edges, edge_ids=edge_ids, offsets=offsets,
                smooth=face_smooth, sharp=sharp_set, seam=seam_set, creases=crease_map,
                bevel_edge_weights=bevel_edge_values, bevel_vertex_weights=bevel_vertex_values,
                materials=face_materials, uvs=uvs, normals=normals, colors=color_values,
                groups=group_values, loose=loose)


def potter_mesh(spec, output_groups=()):
    """Potter `Mesh` serde JSON: stable ids equal indices, every edge (face and loose) is listed."""
    attributes = {
        "shade_smooth": {"domain": "face", "type": "bool",
                         "values": {f"f{i}": v for i, v in enumerate(spec["smooth"])}},
        "sharp": {"domain": "edge", "type": "bool",
                  "values": {f"e{i}": edge in spec["sharp"] for i, edge in enumerate(spec["edges"])}},
        "blender_edge_flags": [
            {"sharp": edge in spec["sharp"], "seam": edge in spec["seam"]}
            for edge in spec["edges"]
        ],
    }
    if spec["creases"]:
        attributes["crease_edge"] = {"domain": "edge", "type": "float",
                                     "values": {f"e{i}": spec["creases"].get(edge, 0.0)
                                                for i, edge in enumerate(spec["edges"])}}
    if spec["bevel_edge_weights"] is not None:
        attributes["bevel_weight_edge"] = {
            "domain": "edge", "type": "float",
            "values": {f"e{i}": weight for i, weight in enumerate(spec["bevel_edge_weights"])},
        }
    if spec["bevel_vertex_weights"] is not None:
        attributes["bevel_weight_vert"] = {
            "domain": "point", "type": "float",
            "values": {f"v{i}": weight for i, weight in enumerate(spec["bevel_vertex_weights"])},
        }
    if spec["normals"] is not None:
        values, loop = {}, 0
        for f, face in enumerate(spec["faces"]):
            values[f"f{f}"] = [spec["normals"][loop + k] for k in range(len(face))]
            loop += len(face)
        attributes["custom_normal"] = {"domain": "corner", "type": "float3", "values": values}
    if spec["uvs"]:
        names = ["UVMap", "UV2"]
        attributes["blender_uv_layers"] = [{"name": names[i], "values": uvs}
                                           for i, uvs in enumerate(spec["uvs"])]
    if spec["colors"] is not None:
        attributes["blender_attributes"] = [{
            "name": "SolidifyColor", "domain": "CORNER", "data_type": "FLOAT_COLOR",
            "values": spec["colors"],
        }]
    group_names = set(spec["groups"]) | {name for name in output_groups if name}
    if group_names:
        attributes["vertex_groups"] = {
            name: {f"v{i}": w for i, w in enumerate(spec["groups"].get(name, []))}
            for name in group_names
        }
    return {
        "vertices": [{"id": i, "co": [float(c) for c in v]} for i, v in enumerate(spec["verts"])],
        "edges": [{"id": i, "v": list(edge)} for i, edge in enumerate(spec["edges"])],
        "faces": [{"id": i, "v": list(face), "material_index": spec["materials"][i]}
                  for i, face in enumerate(spec["faces"])],
        "attributes": attributes,
        "next_id": {"vertex": len(spec["verts"]), "edge": len(spec["edges"]), "face": len(spec["faces"])},
    }


# ---------------------------------------------------------------- Blender build and readback

def default_params():
    me = bpy.data.meshes.new("default_probe")
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    ob = bpy.data.objects.new("default_probe", me)
    bpy.context.scene.collection.objects.link(ob)
    modifier = ob.modifiers.new("probe", "SOLIDIFY")
    defaults = {name: plain(getattr(modifier, name)) for name in PARAM_NAMES}
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    return defaults


def plain(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"unexpected Blender property value {value!r}")


def build_object(name, spec, slots, params):
    me = bpy.data.meshes.new(name)
    me.from_pydata(spec["verts"], spec["loose"], spec["faces"])
    me.update()
    for i, poly in enumerate(me.polygons):
        poly.use_smooth = spec["smooth"][i]
        poly.material_index = spec["materials"][i]
    for edge in me.edges:
        key = edge_key(edge.vertices[0], edge.vertices[1])
        edge.use_edge_sharp = key in spec["sharp"]
        edge.use_seam = key in spec["seam"]
    if spec["creases"]:
        crease = me.attributes.new("crease_edge", "FLOAT", "EDGE")
        for edge in me.edges:
            crease.data[edge.index].value = spec["creases"].get(
                edge_key(edge.vertices[0], edge.vertices[1]), 0.0
            )
    if spec["bevel_edge_weights"] is not None:
        bevel = me.attributes.new("bevel_weight_edge", "FLOAT", "EDGE")
        for edge in me.edges:
            key = edge_key(edge.vertices[0], edge.vertices[1])
            bevel.data[edge.index].value = spec["bevel_edge_weights"][spec["edge_ids"][key]]
    if spec["bevel_vertex_weights"] is not None:
        bevel = me.attributes.new("bevel_weight_vert", "FLOAT", "POINT")
        for index, value in enumerate(spec["bevel_vertex_weights"]):
            bevel.data[index].value = value
    slot_table = [True] * slots if isinstance(slots, int) else slots
    for index, occupied in enumerate(slot_table):
        material = bpy.data.materials.new(f"{name}_slot{index}") if occupied else None
        me.materials.append(material)
    for layer, uvs in enumerate(spec["uvs"]):
        uv_map = me.uv_layers.new(name=["UVMap", "UV2"][layer])
        for loop, uv in enumerate(uvs):
            uv_map.data[loop].uv = uv
    if spec["colors"] is not None:
        color = me.color_attributes.new(name="SolidifyColor", type="FLOAT_COLOR", domain="CORNER")
        for loop, value in enumerate(spec["colors"]):
            color.data[loop].color = value
    if spec["normals"] is not None:
        me.normals_split_custom_set([tuple(n) for n in spec["normals"]])
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for group, weights in spec["groups"].items():
        vg = ob.vertex_groups.new(name=group)
        for i, w in enumerate(weights):
            vg.add([i], w, "REPLACE")
    for group in (params["shell_vertex_group"], params["rim_vertex_group"]):
        if group and ob.vertex_groups.get(group) is None:
            ob.vertex_groups.new(name=group)
    modifier = ob.modifiers.new("Solidify", "SOLIDIFY")
    for key in PARAM_NAMES:
        setattr(modifier, key, params[key])
    return ob, me


def codec_error(me, spec):
    """Largest per-component error of Blender's stored custom normals against the requested unit normals."""
    if spec["normals"] is None:
        return None
    worst = 0.0
    for loop, target in enumerate(spec["normals"]):
        got = me.corner_normals[loop].vector
        worst = max(worst, max(abs(got[k] - target[k]) for k in range(3)))
    return worst
def codec_roundtrip_error(mesh):
    """Maximum per-component output-normal codec error after writing and decoding Blender's vectors."""
    if not mesh.has_custom_normals:
        return None
    requested = [tuple(mesh.corner_normals[i].vector) for i in range(len(mesh.loops))]
    probe = mesh.copy()
    try:
        probe.normals_split_custom_set(requested)
        return max(
            abs(probe.corner_normals[i].vector[axis] - requested[i][axis])
            for i in range(len(requested))
            for axis in range(3)
        )
    finally:
        bpy.data.meshes.remove(probe)




def canonical_hash(verts, faces):
    positions = sorted(tuple(round(c, 5) + 0.0 for c in p) for p in verts)
    face_sets = sorted(sorted(tuple(round(c, 5) + 0.0 for c in verts[i]) for i in face) for face in faces)
    text = json.dumps({"vertices": [list(p) for p in positions],
                       "faces": [[list(p) for p in f] for f in face_sets]}, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def record_output(ob, me_e):
    verts = [[float(c) for c in v.co] for v in me_e.vertices]
    crease_attr = me_e.attributes.get("crease_edge")
    bevel_weight_edge = me_e.attributes.get("bevel_weight_edge")
    edges = []
    for edge in me_e.edges:
        crease = float(crease_attr.data[edge.index].value) if crease_attr is not None else 0.0
        bevel_weight = (
            float(bevel_weight_edge.data[edge.index].value)
            if bevel_weight_edge is not None else 0.0
        )
        edges.append({
            "v": [int(edge.vertices[0]), int(edge.vertices[1])],
            "sharp": bool(edge.use_edge_sharp),
            "seam": bool(edge.use_seam),
            "crease": crease,
            "bevel_weight": bevel_weight,
        })
    uv_names = [layer.name for layer in me_e.uv_layers]
    custom = bool(me_e.has_custom_normals)
    normals = None
    if custom:
        normals = [[float(c) for c in me_e.corner_normals[i].vector] for i in range(len(me_e.loops))]
    color = me_e.attributes.get("SolidifyColor")
    color_names = [
        attribute.name
        for attribute in me_e.color_attributes
        if attribute.domain == "CORNER" and attribute.data_type == "FLOAT_COLOR"
    ]
    faces = []
    loops_total = 0
    for poly in me_e.polygons:
        loops = range(poly.loop_start, poly.loop_start + poly.loop_total)
        loops_total += poly.loop_total
        face = {
            "v": [int(v) for v in poly.vertices],
            "material_index": int(poly.material_index),
            "smooth": bool(poly.use_smooth),
            "uv": {name: [[float(c) for c in me_e.uv_layers[name].data[i].uv] for i in loops]
                   for name in uv_names},
            "colors": {
                "SolidifyColor": [[float(c) for c in color.data[i].color] for i in loops]
            } if color is not None else {},
        }
        if normals is not None:
            face["normals"] = [normals[i] for i in loops]
        faces.append(face)
    group_names = [group.name for group in ob.vertex_groups]
    groups = {name: [0.0] * len(verts) for name in group_names}
    for i, vertex in enumerate(me_e.vertices):
        for element in vertex.groups:
            groups[group_names[element.group]][i] = float(element.weight)
    face_edges = set()
    for face in faces:
        for k in range(len(face["v"])):
            face_edges.add(edge_key(face["v"][k], face["v"][(k + 1) % len(face["v"])]))
    loose = sum(1 for e in edges if edge_key(e["v"][0], e["v"][1]) not in face_edges)
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "vertex_groups": groups,
        "uv_layer_names": uv_names,
        "color_attribute_names": color_names,
        "has_custom_normals": custom,
        "has_bevel_weight_edge": bevel_weight_edge is not None,
        "topology": {"vertices": len(verts), "edges": len(edges), "faces": len(faces),
                     "loops": loops_total, "loose_edges": loose},
        "canonical_hash": canonical_hash(verts, [f["v"] for f in faces]),
    }


# ---------------------------------------------------------------- cases

def pairs(*items):
    return [tuple(p) for p in items]


MATS_3 = [0, 1, 2, 0, 1, 2]
GRID_SMOOTH = [T, F, T, F, T, F]
GRID_SHARP = pairs((1, 5), (5, 6))
GRID_SEAM = pairs((0, 4), (1, 5))
GRID_CREASES = [(0, 1, 0.25), (4, 5, 0.5), (5, 6, 0.75)]

GRID_BEVEL_EDGES = [(0, 1, 0.25), (4, 5, 0.75), (5, 6, 0.5)]
GRID_BEVEL_VERTICES = [
    0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75,
]
QUAD_EDGES = pairs((0, 1), (1, 2), (2, 3), (3, 0))
BOOK4_EDGES = pairs((0, 1), (0, 2), (2, 3), (1, 3), (0, 4), (4, 5), (1, 5),
                    (0, 6), (6, 7), (1, 7), (0, 8), (8, 9), (1, 9))

# (case, base, decorate kwargs, material slot count, parameter overrides, purpose)
CASES = [
    ("grid_extrude_default", "grid", {}, 1,
     dict(thickness=0.1, use_pin_to_last=True, use_apply_on_spline=True),
     "open grid, thickness only; use_pin_to_last/use_apply_on_spline set (Solidify ignores them)"),
    ("grid_extrude_bevel_weights", "grid",
     dict(bevel_edges=GRID_BEVEL_EDGES, bevel_vertices=GRID_BEVEL_VERTICES), 1,
     dict(thickness=0.2, offset=0.0, bevel_convex=0.4),
     "EXTRUDE bevel_convex with nonzero input edge and vertex bevel weights"),
    ("grid_offset_fraction", "grid", dict(smooth=GRID_SMOOTH, sharp=GRID_SHARP), 1,
     dict(thickness=0.25, offset=0.37, use_rim=False),
     "offset 0.37 (not 0 or +-1), rim off, smooth and sharp flags on the input"),
    ("grid_offset_outward_flip", "grid", dict(uv_layers=2), 1,
     dict(thickness=0.3, offset=1.0, use_flip_normals=True, use_flat_faces=True),
     "offset +1 with flipped normals and flat faces, two UV layers"),
    ("grid_even_quality_inward", "grid", {}, 1,
     dict(thickness=0.3, offset=-0.6, use_even_offset=True, use_quality_normals=True),
     "offset -0.6 with even offset and quality normals"),
    ("grid_rim_only", "grid", {}, 1,
     dict(thickness=0.2, offset=0.5, use_rim_only=True),
     "rim-only output with offset +0.5"),
    ("grid_rim_off_flat", "grid", dict(smooth=GRID_SMOOTH), 1,
     dict(thickness=0.2, offset=0.8, use_rim=False, use_flat_faces=True),
     "rim off, flat faces, smooth flags on the input"),
    ("grid_negative_thickness", "grid", {}, 1,
     dict(thickness=-0.2, offset=-0.5),
     "negative thickness (Blender accepts it)"),
    ("grid_negative_thickness_even_out", "grid", {}, 1,
     dict(thickness=-0.25, offset=1.0, use_even_offset=True, use_flip_normals=True),
     "negative thickness with offset +1, even offset and flipped normals"),
    ("grid_zero_thickness", "grid", {}, 1,
     dict(thickness=0.0),
     "zero thickness: Blender keeps coincident shells (duplicate positions)"),
    ("grid_vg_invert", "grid", dict(groups=("thick",)), 1,
     dict(thickness=0.3, offset=-0.3, vertex_group="thick", invert_vertex_group=True,
          thickness_vertex_group=0.6),
     "thickness vertex group inverted, thickness factor 0.6"),
    ("grid_vg_direct", "grid", dict(groups=("thick",)), 1,
     dict(thickness=0.3, offset=0.0, vertex_group="thick", thickness_vertex_group=0.4),
     "thickness vertex group not inverted, thickness factor 0.4, offset 0"),
    ("grid_shell_rim_groups", "grid", {}, 1,
     dict(thickness=0.2, offset=-1.0, shell_vertex_group="shell", rim_vertex_group="rim"),
     "shell and rim output vertex-group weights"),
    ("grid_mat_slots1", "grid", dict(materials=MATS_3), 1,
     dict(thickness=0.2, offset=-0.4, material_offset=2, material_offset_rim=-1),
     "totcol 1 disables offsets; original face material indices remain untouched"),
    ("grid_mat_slots2", "grid", dict(materials=MATS_3), 2,
     dict(thickness=0.2, offset=0.4, material_offset=1, material_offset_rim=1),
     "totcol 2, positive material offsets"),
    ("grid_mat_slots3", "grid", dict(materials=MATS_3), 3,
     dict(thickness=0.2, offset=-0.4, material_offset=-1, material_offset_rim=2),
     "totcol 3, negative material offset and rim offset 2"),
    ("grid_mat_slots4_wide_offset", "grid", dict(materials=MATS_3), 4,
     dict(thickness=0.2, material_offset=5, material_offset_rim=-7, use_rim=True),
     "totcol 4, material offsets beyond the slot range"),
    ("grid_creases", "grid", dict(creases=GRID_CREASES), 1,
     dict(thickness=0.2, edge_crease_inner=0.6, edge_crease_outer=0.3, edge_crease_rim=0.8),
     "input crease_edge values plus edge_crease_inner/outer/rim"),
    ("grid_custom_uv", "grid", dict(smooth=GRID_SMOOTH, sharp=GRID_SHARP, seam=GRID_SEAM,
                                     uv_layers=2, custom=True, colors=True), 1,
     dict(thickness=0.2, offset=0.5, use_quality_normals=True),
     "custom normals, two UV layers, named corner color, and sharp/seam flags"),
    ("grid_loose_edges", "grid_loose", {}, 1,
     dict(thickness=0.15, offset=0.0),
     "open grid with loose edges (one a diagonal of a face) and a loose vertex pair"),
    ("cube_closed_extrude", "cube", {}, 1,
     dict(thickness=0.05, offset=-1.0),
     "closed asymmetric cube, inward offset"),
    ("cube_angle_clamp", "cube", dict(smooth=T), 1,
     dict(thickness=0.3, offset=0.0, thickness_clamp=0.12, use_thickness_angle_clamp=True),
     "closed cube, thickness_clamp 0.12 with the angle clamp"),
    ("cube_near_clamp", "cube", {}, 1,
     dict(thickness=0.5, offset=0.5, thickness_clamp=0.45),
     "closed cube, thickness 0.5 with thickness_clamp 0.45 (near the edge-length limit)"),
    ("near_collinear_angle_clamp", "near_collinear", {}, 1,
     dict(thickness=0.2, offset=0.6, use_thickness_angle_clamp=True, thickness_clamp=0.5),
     "nearly collinear neighbours and a sliver triangle, angle clamp on"),
    ("near_collinear_extrude", "near_collinear", dict(smooth=T), 1,
     dict(thickness=-0.15, offset=-0.9, use_quality_normals=True),
     "nearly collinear neighbours, negative thickness, quality normals"),
    ("asym_fan_extrude", "asym_fan", {}, 1,
     dict(thickness=-0.3, offset=0.8, use_flip_normals=True, use_even_offset=True),
     "asymmetric open triangle fan, negative thickness, flipped, even offset"),
    ("ngon_concave_extrude", "ngon", dict(smooth=T), 1,
     dict(thickness=0.3, offset=-0.5, use_flat_faces=True, use_quality_normals=True),
     "concave L-shaped n-gon, flat faces, quality normals"),
    ("tee_extrude", "tee", {}, 1,
     dict(thickness=0.2, offset=-0.4),
     "three quads sharing one edge (T-edge), EXTRUDE mode"),
    ("book4_extrude", "book4", {}, 1,
     dict(thickness=0.15, offset=0.5, use_rim_only=True),
     "four quads sharing one edge (valence 4), EXTRUDE, rim only"),
    ("tee_nm_fixed_none", "tee", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=-0.3,
          nonmanifold_merge_threshold=0.001),
     "T-edge, NON_MANIFOLD, FIXED thickness, boundary NONE"),
    ("grid_nm_bevel_weights", "grid",
     dict(bevel_edges=GRID_BEVEL_EDGES, bevel_vertices=GRID_BEVEL_VERTICES), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.25,
          bevel_convex=0.35, nonmanifold_merge_threshold=0.001),
     "NON_MANIFOLD input edge and vertex bevel weights with bevel_convex"),
    ("collinear_triangle_nm_fixed_none", "collinear_triangle", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=1.0, offset=0.0,
          nonmanifold_merge_threshold=0.001),
     "collinear triangle, FIXED thickness, NONE boundary"),
    ("quad_nm_sharp_seam_boundary_connector", "single_quad",
     dict(sharp=QUAD_EDGES, seam=QUAD_EDGES), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.0,
          use_rim=True, use_rim_only=False, nonmanifold_merge_threshold=0.001),
     "one-quad two-group boundary connectors copy sharp and seam from source edges"),
    ("book4_nm_sharp_seam_boundary_closing", "book4",
     dict(sharp=BOOK4_EDGES, seam=BOOK4_EDGES), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="CONSTRAINTS",
          nonmanifold_boundary_mode="FLAT", thickness=0.15, offset=0.0,
          use_rim=True, use_rim_only=False, nonmanifold_merge_threshold=0.05),
     "four page groups exercise closing boundary edges with sharp and seam provenance"),
    ("tee_nm_even_round", "tee", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="EVEN",
          nonmanifold_boundary_mode="ROUND", thickness=0.25, offset=0.4,
          bevel_convex=0.5, nonmanifold_merge_threshold=0.01),
     "T-edge, NON_MANIFOLD, EVEN, ROUND, bevel_convex 0.5"),
    ("book4_nm_constraints_flat", "book4", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="CONSTRAINTS",
          nonmanifold_boundary_mode="FLAT", thickness=0.15, offset=0.0,
          bevel_convex=-0.4, nonmanifold_merge_threshold=0.05),
     "valence-4 edge, NON_MANIFOLD, CONSTRAINTS, FLAT, bevel_convex -0.4"),
    ("grid_nm_fixed_none", "grid", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=-1.0),
     "open grid in NON_MANIFOLD mode (boundary edges only)"),
    ("grid_nm_empty_groups_flat", "grid", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.4,
          use_flat_faces=True, shell_vertex_group="shell", rim_vertex_group="rim"),
     "empty shell/rim groups do not allocate deform weights; flat_faces uses Blender's null dvert"),
    ("grid_nm_flat_no_groups", "grid", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.4,
          use_flat_faces=True),
     "flat_faces with no deform-vertex data is a Blender no-op"),
    ("grid_nm_unrelated_vg_empty", "grid", dict(groups=("thick",)), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.4,
          use_flat_faces=True),
     "unrelated deform weights with empty vertex_group do not suppress thickness"),
    ("grid_nm_unrelated_vg_missing", "grid", dict(groups=("thick",)), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="NONE", thickness=0.2, offset=0.4,
          use_flat_faces=True, vertex_group="missing"),
     "unrelated deform weights with missing vertex_group do not suppress thickness"),
    ("grid_nm_null_slot_group", "grid", dict(materials=MATS_3, groups=("thick",)),
     [True, None, True],
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="ROUND", thickness=0.2, offset=0.4,
          use_flat_faces=True, material_offset=1, material_offset_rim=1),
     "null slot is retained for material offsets while unrelated weights do not suppress thickness"),
    ("asym_fan_nm_round_even", "asym_fan",
     dict(smooth=T, seam=pairs((0, 5)), colors=True, custom=True, uv_layers=2), 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="EVEN",
          nonmanifold_boundary_mode="ROUND", thickness=0.3, offset=0.7, use_flip_normals=True),
     "asymmetric NON_MANIFOLD, EVEN, ROUND output preserves normal, UV, color and seam data"),
    ("cube_nm_constraints_round", "cube", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="CONSTRAINTS",
          nonmanifold_boundary_mode="ROUND", thickness=0.1, offset=-0.5, bevel_convex=0.25),
     "closed cube in NON_MANIFOLD, CONSTRAINTS, ROUND, bevel_convex 0.25"),
    ("ngon_nm_flat_fixed", "ngon", {}, 1,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="FIXED",
          nonmanifold_boundary_mode="FLAT", thickness=0.3, offset=-1.0),
     "concave n-gon in NON_MANIFOLD, FIXED, FLAT"),
    ("grid_nm_groups_mats_creases", "grid", dict(materials=MATS_3, groups=("thick",), creases=GRID_CREASES), 2,
     dict(solidify_mode="NON_MANIFOLD", nonmanifold_thickness_mode="CONSTRAINTS",
          nonmanifold_boundary_mode="ROUND", thickness=0.2, vertex_group="thick",
          invert_vertex_group=True, material_offset=1, material_offset_rim=-1,
          edge_crease_inner=0.4, edge_crease_outer=0.2, edge_crease_rim=0.9),
     "NON_MANIFOLD with the inverted thickness group, two material slots, creases"),
    ("edges_only_extrude", "edges_only", {}, 1,
     dict(thickness=0.2),
     "only loose edges (no faces): Blender duplicates the edges"),
]


def run_case(name, base_name, decor_kwargs, slots, overrides, purpose, defaults):
    base = BASES[base_name]()
    spec = decorate(base, **decor_kwargs)
    params = dict(defaults)
    params.update(overrides)
    slot_table = [True] * slots if isinstance(slots, int) else slots
    input_doc = {
        "case": name,
        "purpose": purpose,
        "material_slots": len(slot_table),
        "material_slot_layout": slot_table,
        "modifier": {"type": "solidify", "params": params},
        "mesh": potter_mesh(
            spec, (params["shell_vertex_group"], params["rim_vertex_group"])
        ),
    }
    expected = {"case": name, "blender_version": "5.2.2"}
    ob = me = None
    try:
        ob, me = build_object(name, spec, slot_table, params)
        expected["input_normal_codec_max_error"] = codec_error(me, spec)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = ob.evaluated_get(dg)
        me_e = ev.to_mesh()
        try:
            expected["outcome"] = "ok"
            expected["output"] = record_output(ob, me_e)
            expected["output_normal_codec_max_error"] = codec_roundtrip_error(me_e)
        finally:
            ev.to_mesh_clear()
    except Exception as error:  # Blender refusals are recorded, not hidden
        expected["outcome"] = "error"
        expected["error"] = f"{type(error).__name__}: {error}"
        expected["traceback"] = traceback.format_exc()
    finally:
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None:
            for material in list(me.materials):
                if material is not None and material.users <= 1:
                    bpy.data.materials.remove(material)
            bpy.data.meshes.remove(me)
    return input_doc, expected


def write_json(path, document):
    with open(path, "w") as handle:
        json.dump(document, handle, indent=1, allow_nan=False)
        handle.write("\n")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    os.makedirs(INPUTS, exist_ok=True)
    os.makedirs(EXPECTED, exist_ok=True)
    for directory in (INPUTS, EXPECTED):
        for stale in os.listdir(directory):
            if stale.endswith(".json"):
                os.remove(os.path.join(directory, stale))
    defaults = default_params()
    names = [case[0] for case in CASES]
    assert len(names) == len(set(names)), "duplicate case names"
    failures = []
    for name, base_name, decor_kwargs, slots, overrides, purpose in CASES:
        input_doc, expected = run_case(name, base_name, decor_kwargs, slots, overrides, purpose, defaults)
        write_json(os.path.join(INPUTS, f"{name}.json"), input_doc)
        write_json(os.path.join(EXPECTED, f"{name}.json"), expected)
        if expected["outcome"] == "ok":
            topo = expected["output"]["topology"]
            print(f"SOLIDIFY {name}: ok v={topo['vertices']} e={topo['edges']} f={topo['faces']} "
                  f"loose={topo['loose_edges']} custom={expected['output']['has_custom_normals']}")
        else:
            failures.append(name)
            print(f"SOLIDIFY {name}: ERROR {expected['error']}")
    print(f"SOLIDIFY wrote {len(CASES)} cases; blender errors: {failures or 'none'}")


main()
