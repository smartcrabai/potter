"""Blender 5.2.2 Wave modifier oracle with `use_normal` (MOD_wave.cc: `co += ... * vert_normals()`).

Run: Blender -b --factory-startup --threads 1 --python generate_wave_oracle.py -- <out.json>

The mesh is a jittered 4x4 quad grid: unequal faces and corner angles, so the angle-weighted vertex
normals differ from area-weighted or unweighted sums. Each case records the input positions, the
faces, the modifier parameters and Blender's evaluated positions at frame 20. The `use_normal` case
and the case without it share every other parameter, so the normal term is the only difference.
"""
import json
import random
import struct
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
rng = random.Random(20261020)
FRAME = 20

PARAMS = {
    "speed": 0.25,
    "height": 0.5,
    "width": 1.5,
    "narrowness": 1.5,
    "time_offset": 0.0,
    "lifetime": 0.0,
    "damping_time": 10.0,
    "falloff_radius": 0.0,
    "start_position_x": 0.2,
    "start_position_y": -0.1,
    "use_x": True,
    "use_y": True,
    "use_cyclic": True,
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def jittered_grid():
    points = []
    for j in range(5):
        for i in range(5):
            points.append((
                f32(i + rng.uniform(-0.3, 0.3)),
                f32(j + rng.uniform(-0.3, 0.3)),
                f32(rng.uniform(-0.3, 0.3)),
            ))
    faces = []
    for j in range(4):
        for i in range(4):
            a = j * 5 + i
            faces.append((a, a + 1, a + 6, a + 5))
    return points, faces


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_set(FRAME)
points, faces = jittered_grid()


def evaluate(kind, use_normal):
    me = bpy.data.meshes.new("wave_case")
    me.from_pydata(points, [], faces)
    me.update()
    obj = bpy.data.objects.new("wave_case", me)
    bpy.context.collection.objects.link(obj)
    mod = obj.modifiers.new("Wave", "WAVE")
    for key, value in PARAMS.items():
        setattr(mod, key, value)
    mod.use_normal = use_normal
    mod.use_normal_x = True
    mod.use_normal_y = True
    mod.use_normal_z = True
    scene.frame_set(FRAME)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    result = evaluated.to_mesh()
    output = [[float(component) for component in vertex.co] for vertex in result.vertices]
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(me)
    assert len(output) == len(points)
    return {
        "kind": kind,
        "frame": FRAME,
        "params": dict(PARAMS, use_normal=use_normal, use_normal_x=True, use_normal_y=True, use_normal_z=True),
        "positions": [list(point) for point in points],
        "faces": [list(face) for face in faces],
        "output": output,
    }


cases = [evaluate("wave_use_normal", True), evaluate("wave_no_normal", False)]
with open(OUT, "w") as handle:
    json.dump({"blender_version": "5.2.2", "cases": cases}, handle, indent=0)
print("CASES", len(cases))
