"""Export the four UV-sphere Boolean hosts from a generated regression scene as pinned input data.

Blender's UV sphere primitive computes its vertex coordinates with the platform's float libm. The
macOS arm64 and Linux x86_64 builds differ by a few float32 ulps, and the canonical vertex order
depends on those bits, so generating the spheres per platform made the Boolean input platform
dependent. Each host is therefore exported once, with exact float32 values (Python's float repr
round-trips a double that holds a float32 exactly) and its edge, face and vertex order, to
`hosts/<case>.json`. The generator builds the four sphere hosts from these files on every platform.

Provenance: run once on macOS (Blender 5.2.2 arm64, the oracle platform) on the `scene.blend` written
by the generator. Those hosts are exactly the input the committed `expected/sphere_*` files were
recorded from; that equality is checked when the generator is run again.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/boolean_parity/regressions/dump_sphere_hosts.py -- \
        <scene.blend> tests/fixtures/boolean_parity/regressions/hosts
"""

import json
import os
import sys

import bpy

CASES = (
    "sphere_open64_hole_tolerant_on",
    "sphere_open64_hole_tolerant_off",
    "sphere_open128_hole_tolerant_on",
    "sphere_closed64_hole_tolerant_on",
)

SCENE, OUTPUT = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.open_mainfile(filepath=SCENE)
os.makedirs(OUTPUT, exist_ok=True)
for case in CASES:
    mesh = bpy.data.objects[case].data
    data = {
        "verts": [list(vertex.co) for vertex in mesh.vertices],
        "edges": [list(edge.vertices) for edge in mesh.edges],
        "faces": [list(polygon.vertices) for polygon in mesh.polygons],
    }
    with open(os.path.join(OUTPUT, case + ".json"), "w") as handle:
        json.dump(data, handle)
    print("HOST", case, len(data["verts"]), len(data["edges"]), len(data["faces"]), flush=True)
