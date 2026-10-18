"""PR9 persistence reference (Blender 5.2.2, background).

Usage:
  Blender --factory-startup -b oracle.blend --python authoring_reference.py -- OUT.json

Evaluates the oracle's `Regress` object through a Subdivision (levels 1, Use Custom Normals off),
so the result has no custom-normal layer. Then authors custom normals on that evaluated topology
with Blender's own setter, using the fan rules of build_oracle.py: a fan whose first loop index is
not a multiple of 3 is authored with authored_vector(first loop); every other fan is automatic
(code pair (0, 0)).

Writes one entry per loop of that result, in Blender's own loop order:
  co, center  the loop's vertex position and its face center, as Blender reports them
  vector      the authored input vector, or null for an automatic fan (what Potter must receive)
  normal      Blender's corner normal after authoring (the expected per-corner normal)
  code        Blender's raw custom_normal pair after authoring
"""

import json
import math
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]


def unit(vector):
    length = math.sqrt(sum(c * c for c in vector))
    return tuple(c / length for c in vector)


def authored_vector(loop_index):
    x = 0.2 + 0.07 * ((loop_index * 5) % 11)
    y = -0.3 + 0.05 * ((loop_index * 7) % 9)
    z = 0.9 - 0.03 * ((loop_index * 3) % 7)
    return unit((x, y, z))


def corner_fans(mesh):
    """Blender's loop-normal spaces, as in build_oracle.py. Returns the fan root of every loop."""
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
    return [find(li) for li in range(count)]


regress = bpy.data.objects["Regress"]
regress.modifiers.clear()
subdivision = regress.modifiers.new("Subdivision", "SUBSURF")
subdivision.levels = 1
subdivision.render_levels = 1
subdivision.use_custom_normals = False
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = regress.evaluated_get(depsgraph).to_mesh()
mesh = evaluated.copy()
regress.evaluated_get(depsgraph).to_mesh_clear()
regress.modifiers.clear()
mesh.update()
if mesh.attributes.get("custom_normal") is not None:
    raise SystemExit("Subdivision without use_custom_normals kept a custom_normal layer")

count = len(mesh.loops)
base = [tuple(n.vector) for n in mesh.corner_normals]
roots = corner_fans(mesh)
members_by_root = {}
for loop_index, root in enumerate(roots):
    members_by_root.setdefault(root, []).append(loop_index)
authored = {}
for members in members_by_root.values():
    first = min(members)
    if first % 3 != 0:
        vector = authored_vector(first)
        for loop_index in members:
            authored[loop_index] = vector
vectors = [authored.get(i, base[i]) for i in range(count)]
mesh.normals_split_custom_set(vectors)
codes_attribute = mesh.attributes["custom_normal"]
for loop_index in range(count):
    if loop_index not in authored:
        codes_attribute.data[loop_index].value = (0, 0)
mesh.update()
codes = [tuple(int(c) for c in item.value) for item in mesh.attributes["custom_normal"].data]
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

loops = []
for poly in mesh.polygons:
    center = [float(c) for c in poly.center]
    for k in range(poly.loop_total):
        loop_index = poly.loop_start + k
        co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        loops.append({
            "co": [float(c) for c in co],
            "center": center,
            "vector": [float(c) for c in authored[loop_index]] if loop_index in authored else None,
            "normal": [float(c) for c in after[loop_index]],
            "code": list(codes[loop_index]),
        })

result = {
    "blender": bpy.app.version_string,
    "authored": len(authored),
    "automatic": count - len(authored),
    "loops": loops,
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
