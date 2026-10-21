"""Generates the expected plane positions in procedural_oracle.json for tests/pr18f_displacement.rs.

Each case in the file is a plane (size 2, Blender primitive_plane_add) with one procedural texture
and one Displace modifier. Blender-backed cases evaluate the texture in Blender 5.2.2 headless and
record the evaluated vertex positions in Blender's storage order. Blender's storage order for this
primitive matches Potter's grid order (-1,-1), (1,-1), (-1,1), (1,1); the script asserts it.

`native_value_noise` has no Blender counterpart: NATIVE_VALUE_NOISE is Potter's former Volume
Displace value noise, kept as a native texture type. Its backend is a port of that former sampler
(`value_noise_color` and `hash_value_noise` in `crates/potter-core/src/geom/modifiers/simulation.rs`
before commit 3790f51c). It is evaluated here in Python, independently of the Rust sampler.

The spec fields are inputs and `expected` is the output. Running the script rewrites the file.

Usage:
  Blender -b --factory-startup --python generate_procedural_oracle.py -- procedural_oracle.json
"""

import bpy
import json
import math
import sys

MASK64 = (1 << 64) - 1
PLANE_ORDER = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]


def path_arg():
    return sys.argv[sys.argv.index("--") + 1]


def round_position(co):
    return [round(float(c), 9) for c in co]


def blender_expected(case, size):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=size)
    obj = bpy.context.active_object
    rest = [tuple(round(float(c), 9) for c in v.co) for v in obj.data.vertices]
    assert rest == PLANE_ORDER, "Blender plane order differs from Potter's: %r" % (rest,)

    tex = bpy.data.textures.new(case["id"], type=case["texture_type"])
    for key, value in case["settings"].items():
        assert hasattr(tex, key), "%s has no Texture setting %s" % (case["texture_type"], key)
        setattr(tex, key, value)

    mod = obj.modifiers.new("deform", "DISPLACE")
    mod.texture = tex
    mod.texture_coords = case["texture_coords"]
    mod.direction = case["direction"]
    mod.strength = case["strength"]
    mod.mid_level = case["mid_level"]

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    positions = [round_position(v.co) for v in mesh.vertices]
    evaluated.to_mesh_clear()
    assert len(positions) == 4, "expected four plane vertices, got %d" % len(positions)
    return positions


def hash_value_noise(point, channel):
    value = (
        ((point[0] & MASK64) * 0x9E3779B9)
        ^ ((point[1] & MASK64) * 0x85EBCA6B)
        ^ ((point[2] & MASK64) * 0xC2B2AE35)
        ^ ((channel & MASK64) * 0x27D4EB2D)
    ) & MASK64
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK64
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK64
    value ^= value >> 16
    return float(value) / float(MASK64)


def native_value_noise(point):
    if not all(math.isfinite(c) for c in point) or max(abs(c) for c in point) > 9.0e18:
        return [0.5, 0.5, 0.5]
    floor = [math.floor(c) for c in point]
    base = [int(f) for f in floor]
    fade = []
    for axis in range(3):
        d = point[axis] - floor[axis]
        fade.append(d * d * (3.0 - 2.0 * d))
    channels = []
    for channel in range(3):
        value = 0.0
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    cell = [base[0] + x, base[1] + y, base[2] + z]
                    wx = fade[0] if x else 1.0 - fade[0]
                    wy = fade[1] if y else 1.0 - fade[1]
                    wz = fade[2] if z else 1.0 - fade[2]
                    value += hash_value_noise(cell, channel) * (wx * wy * wz)
        channels.append(value)
    return channels


def native_expected(case, size):
    # Rest positions in Potter's storage order; the displacement is (rgb - mid) * strength per axis.
    out = []
    for rest in PLANE_ORDER:
        rgb = native_value_noise([rest[0] * (size / 2.0), rest[1] * (size / 2.0), rest[2]])
        offset = [(c - case["mid_level"]) * case["strength"] for c in rgb]
        out.append(round_position([rest[0] + offset[0], rest[1] + offset[1], rest[2] + offset[2]]))
    return out


def main():
    path = path_arg()
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    size = float(spec["plane"]["size"])
    for case in spec["cases"]:
        if case["backend"] == "blender":
            case["expected"] = blender_expected(case, size)
        elif case["backend"] == "potter_legacy_value_noise":
            assert case["texture_type"] == "NATIVE_VALUE_NOISE"
            case["expected"] = native_expected(case, size)
        else:
            raise ValueError("unknown backend %s" % case["backend"])
        print("GENERATED", case["id"], case["expected"])
    spec["generated_by"] = "tests/fixtures/pr18f_displacement/generate_procedural_oracle.py"
    spec["blender_version"] = bpy.app.version_string
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=1)
        handle.write("\n")


main()
