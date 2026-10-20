"""Derives the tolerance of `tests/realtime_transparent_shadow.rs` from measurements and geometry.

The oracle compares Potter's realtime shadowed-to-lit ratio with EEVEE's. Each side has an error against the exact
ratio, so the tolerance is the sum of three terms, per light and per occluder state:

1. EEVEE's noise: four times the sample standard deviation of the ratio over independent noise realisations. EEVEE has
   no seed, so each realisation is the generator at a different y shift of the whole scene (a sub-millimetre move that
   leaves the expectation unchanged and redraws its world-position hashed transparency noise).
2. EEVEE's bias: the absolute difference between EEVEE's mean ratio and the exact ratio, computed here from the
   occluder geometry on the 81 sampled pixel centres (a weighted average of the shadow factor, weighted by the lit
   radiance of each floor point for the point light).
3. Potter's edge placement: the largest change of the exact ratio when the occluder's shadow edges move by the
   bound on Potter's placement error. For a sun the PCF jitter is 0.003 rad at 1 unit from the floor. For a point light
   the depth map has 64 texels per 90-degree face, and nearest-texel lookup misplaces an edge by at most half a texel,
   taken at the occluder plane and scaled to the floor.

The shift sweep takes the worst case over every shift in the bound's square, not the nominal geometry alone.

Usage: python3 measure_realtime_transparent_shadow_spread.py BLENDER OUT_JSON
"""

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = os.path.join(HERE, "realtime_transparent_shadow.py")
SHIFTS = [0.0, 0.0003, 0.0006, 0.0009, 0.0012]
LIGHTS = ["SUN", "POINT"]
OCCLUDER_ALPHA = 0.5
# The occluder square at z = 1, centred at (-3, 0), 0.9 units wide: x0, x1, y0, y1.
OCCLUDER = (-3.45, -2.55, -0.45, 0.45)
# The 9 by 9 grid over the 64 by 64 frame, and the ortho camera at x = -1.5 with ortho scale 2.
PATCH = [0.25 + 0.5 * index / 8 for index in range(9)]
RESOLUTION = 64
POINT_LIGHT = (-4.0, 0.0, 2.5)
SUN_BOUND = 0.003
POINT_DEPTH_TO_OCCLUDER = POINT_LIGHT[2] - 1.0
POINT_DEPTH_TO_FLOOR = POINT_LIGHT[2]


def sample_points():
    points = []
    for v in PATCH:
        for u in PATCH:
            px = min(RESOLUTION - 1, int(u * RESOLUTION))
            py = min(RESOLUTION - 1, int(v * RESOLUTION))
            points.append((-2.5 + 2.0 * (px + 0.5) / RESOLUTION, 1.0 - 2.0 * (py + 0.5) / RESOLUTION))
    return points


def inside(square, x, y):
    return square[0] <= x <= square[1] and square[2] <= y <= square[3]


def sun_factors(square):
    # The sun travels along (+x, -z)/sqrt(2); a floor point (x, y) reaches the occluder plane at (x - 1, y).
    return [0.0 if inside(square, x - 1.0, y) else 1.0 for x, y in sample_points()], [1.0] * 81


def point_factors(square):
    factors, weights = [], []
    for x, y in sample_points():
        d = (POINT_LIGHT[0] - x, POINT_LIGHT[1] - y, POINT_LIGHT[2])
        distance_squared = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
        distance = math.sqrt(distance_squared)
        weights.append((d[2] / distance) / distance_squared)
        t = 1.0 / d[2]
        factors.append(0.0 if inside(square, x + t * d[0], y + t * d[1]) else 1.0)
    return factors, weights


def ratios(light, square):
    """Exact shadowed-to-lit ratios: weighted means of the shadow factor, for the alpha and the opaque occluder."""
    factors, weights = sun_factors(square) if light == "SUN" else point_factors(square)
    total = sum(weights)
    transparent = sum(w * (1.0 if f else OCCLUDER_ALPHA_PASS) for w, f in zip(weights, factors)) / total
    opaque = sum(w * (1.0 if f else 0.0) for w, f in zip(weights, factors)) / total
    return transparent, opaque


OCCLUDER_ALPHA_PASS = 1.0 - OCCLUDER_ALPHA


def potter_bound(light, delta, steps=41):
    """The largest change of the exact ratios when the occluder square moves by up to `delta` in each direction."""
    nominal_alpha, nominal_off = ratios(light, OCCLUDER)
    worst_alpha = worst_off = 0.0
    for i in range(steps):
        for j in range(steps):
            dx = -delta + 2.0 * delta * i / (steps - 1)
            dy = -delta + 2.0 * delta * j / (steps - 1)
            moved = (OCCLUDER[0] + dx, OCCLUDER[1] + dx, OCCLUDER[2] + dy, OCCLUDER[3] + dy)
            alpha, off = ratios(light, moved)
            worst_alpha = max(worst_alpha, abs(alpha - nominal_alpha))
            worst_off = max(worst_off, abs(off - nominal_off))
    return worst_alpha, worst_off


def potter_edge_bound(light):
    if light == "SUN":
        return potter_bound(light, SUN_BOUND)
    texel_at_occluder = 2.0 * POINT_DEPTH_TO_OCCLUDER / RESOLUTION
    half_texel_on_floor = 0.5 * texel_at_occluder * POINT_DEPTH_TO_FLOOR / POINT_DEPTH_TO_OCCLUDER
    return potter_bound(light, half_texel_on_floor)


def render(blender, directory, light, occluder, shift):
    name = f"{light}_{occluder}_{shift:.4f}"
    json_out = os.path.join(directory, name + ".json")
    subprocess.run(
        [
            blender, "-b", "--factory-startup", "--python", GENERATOR, "--",
            os.path.join(directory, name + ".blend"),
            os.path.join(directory, name + ".exr"),
            json_out, light, occluder, str(shift),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=900,
    )
    with open(json_out, encoding="utf-8") as handle:
        values = json.load(handle)["patch_mean"]
    return sum(values) / len(values)


def main():
    blender, out_json = sys.argv[1], sys.argv[2]
    report = {"shifts_y": SHIFTS, "lights": {}}
    with tempfile.TemporaryDirectory() as directory:
        for light in LIGHTS:
            runs = []
            for shift in SHIFTS:
                none = render(blender, directory, light, "NONE", shift)
                alpha = render(blender, directory, light, "ALPHA", shift)
                alpha_off = render(blender, directory, light, "ALPHA_OFF", shift)
                runs.append({
                    "none": none,
                    "alpha_ratio": alpha / none,
                    "alpha_off_ratio": alpha_off / none,
                })
            exact_alpha, exact_off = ratios(light, OCCLUDER)
            bound_alpha, bound_off = potter_edge_bound(light)
            entry = {"runs": runs, "exact": {"alpha_ratio": exact_alpha, "alpha_off_ratio": exact_off},
                     "potter_edge_bound": {"alpha_ratio": bound_alpha, "alpha_off_ratio": bound_off}}
            for key, exact, bound in (("alpha_ratio", exact_alpha, bound_alpha),
                                      ("alpha_off_ratio", exact_off, bound_off)):
                values = [run[key] for run in runs]
                stdev = statistics.stdev(values)
                mean = statistics.mean(values)
                bias = abs(mean - exact)
                entry[key] = {
                    "mean": mean,
                    "stdev": stdev,
                    "eevee_bias": bias,
                    "tolerance": 4.0 * stdev + bias + bound,
                }
            report["lights"][light] = entry
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
