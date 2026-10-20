"""Blender 5.2.2 polygon-fill oracle for Potter's kd-tree ear clipper (`geom/triangulate/ear_clip.rs`).

Run headless from the repository root (`--factory-startup` keeps the run reproducible):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/polyfill/generate_polyfill_rings.py -- tests/fixtures/polyfill/degenerate_rings.json

Each ring is a clockwise n-gon (n >= 5) in the z = 0 plane, so Blender's face normal is -Z and the
projection is the identity. The rings are integer grids, zig-zag combs and integer stars, which
produce duplicate points, collinear corners and ties, the cases where the brute-force ear search
disagreed with Blender. For each ring the generator records:

- `loop_triangles`: `Mesh.calc_loop_triangles()` in Blender's order, as corner indices. Mesh
  tessellation runs `BLI_polyfill_calc_arena` (`blenkernel/intern/mesh_tessellate.cc:116`) and Boolean
  n-gon triangulation runs `BLI_polyfill_calc` (`blenlib/intern/mesh_intersect.cc:1953`), so this is
  the exact order of both consumers' polyfill.
- `beauty_triangles`: `bmesh.ops.triangulate` with `ngon_method='BEAUTY'` (`BLI_polyfill_beautify`
  after the polyfill), as sorted corner triples. BMesh's own triangle order is not part of the
  oracle, so only the set is recorded.

Quads take Blender's quad split instead of the polyfill, so they are not in this corpus.
"""

import bmesh
import bpy
import json
import os
import random
import sys

OUTPUT = sys.argv[sys.argv.index("--") + 1]
SEED = 20260424
RINGS_PER_KIND = 80


def signed_area2(ring):
    return sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )


def grid_ring(rng, n):
    return [[float(rng.randint(0, 6)), float(rng.randint(0, 6))] for _ in range(n)]


def comb_ring(rng, n):
    return [[float(i), 0.0 if i % 2 == 0 else float(1 + rng.randint(0, 3))] for i in range(n)]


def star_ring(rng, n):
    import math

    angles = sorted(rng.randint(0, 63) * 2.0 * math.pi / 64.0 for _ in range(n))
    return [
        [float(round(rng.randint(1, 3) * math.cos(t) * 4.0)),
         float(round(rng.randint(1, 3) * math.sin(t) * 4.0))]
        for t in angles
    ]


KINDS = {"grid": grid_ring, "comb": comb_ring, "star": star_ring}


def clockwise_rings():
    rng = random.Random(SEED)
    rings = []
    for kind, make in KINDS.items():
        made = 0
        while made < RINGS_PER_KIND:
            n = rng.randint(5, 14)
            ring = make(rng, n)
            area2 = signed_area2(ring)
            if area2 == 0.0:
                continue
            if area2 > 0.0:
                ring.reverse()
            rings.append({"kind": kind, "coords": ring})
            made += 1
    return rings


def blender_outputs(ring):
    coords = ring["coords"]
    mesh = bpy.data.meshes.new("polyfill_ring")
    mesh.from_pydata([(x, y, 0.0) for x, y in coords], [], [list(range(len(coords)))])
    mesh.update()
    mesh.calc_loop_triangles()
    normal = [round(c) for c in mesh.polygons[0].normal]
    loop_triangles = [list(tri.vertices) for tri in mesh.loop_triangles]
    bpy.data.meshes.remove(mesh)

    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0.0)) for x, y in coords]
    face = bm.faces.new(verts)
    bm.normal_update()
    result = bmesh.ops.triangulate(bm, faces=[face], quad_method="FIXED", ngon_method="BEAUTY")
    beauty = sorted(sorted(v.index for v in f.verts) for f in result["faces"])
    bm.free()
    return normal, loop_triangles, beauty


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = {
        "blender_version": bpy.app.version_string,
        "generator": "tests/fixtures/polyfill/generate_polyfill_rings.py",
        "seed": SEED,
        "rings": [],
    }
    for ring in clockwise_rings():
        normal, loop_triangles, beauty = blender_outputs(ring)
        if normal != [0, 0, -1]:
            raise RuntimeError(f"ring {ring['coords']} has normal {normal}, expected -Z")
        duplicates = len({tuple(p) for p in ring["coords"]}) != len(ring["coords"])
        out["rings"].append({
            "kind": ring["kind"],
            "duplicate_points": duplicates,
            "coords": ring["coords"],
            "loop_triangles": loop_triangles,
            "beauty_triangles": beauty,
        })
    with open(OUTPUT, "w") as handle:
        json.dump(out, handle, indent=1)
    print("WROTE", len(out["rings"]), "rings to", OUTPUT, flush=True)


main()
