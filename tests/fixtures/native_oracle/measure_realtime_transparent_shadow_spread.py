"""Derives the tolerance of `tests/realtime_transparent_shadow.rs` from measurements and geometry.

The oracle compares Potter's realtime shadowed-to-lit ratio with EEVEE's. Each side has an error against the exact
ratio, so the tolerance is the sum of three terms, per light and per occluder state:

1. EEVEE's noise: four times the sample standard deviation of the ratio over independent noise realisations. EEVEE has
   no seed, so each realisation is the generator at a different y shift of the whole scene (a sub-millimetre move that
   leaves the expectation unchanged and redraws its world-position hashed transparency noise).
2. EEVEE's bias: the absolute difference between EEVEE's mean ratio and the exact ratio. The exact ratio averages the
   shadow over each pixel's footprint (a 6 by 6 subsample per pixel), as both renderers antialias a pixel. It is
   weighted by the lit radiance of each floor point for the point light.
3. Potter's edge placement: the largest change of the exact ratio when the occluder's shadow edges move by the bound on
   Potter's placement error, measured in the occluder plane where the occluder is defined:
   - Sun: Potter's sun disc is 16 Vogel rays over `tan(angle / 2)`, the default angle's 0.00459 at the floor's distance
     from the occluder (1 unit), so the edge moves by that radius.
   - Point, transparent occluder: none. Potter traces that occluder with an exact ray to the light
     (`point_light_transparency`), and the depth map leaves it out, so no texel places its edge.
   - Point, opaque occluder: the depth map has 64 texels per 90-degree face at the occluder plane, and its 3 by 3 PCF
     spreads an edge over one texel either side, plus half a texel of rounding: 1.5 texels of the occluder plane.

The shift sweep takes the worst case over every shift in the bound's square, not the nominal geometry alone.

Two modes. The default renders the generator at every shift and then derives the tolerances. `--analyse-only` takes the
runs already in the output file and only re-derives the analysis, so the bounds can change without rendering again.

Usage: python3 measure_realtime_transparent_shadow_spread.py BLENDER OUT_JSON
       python3 measure_realtime_transparent_shadow_spread.py --analyse-only OUT_JSON
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
SUBSAMPLES = 6
POINT_LIGHT = (-4.0, 0.0, 2.5)
# `sun_disc_radius` of the default Blender sun angle, 0.526 degrees: tan(angle / 2) = 0.00459 (`light_model.rs`).
SUN_BOUND = math.tan(math.radians(0.526) / 2.0)
POINT_DEPTH_TO_OCCLUDER = POINT_LIGHT[2] - 1.0
# The 3 by 3 PCF's edge spread plus rounding, in depth-map texels.
POINT_EDGE_TEXELS = 1.5


def pixel_subsamples():
    """For each of the 81 patch pixels, a SUBSAMPLES by SUBSAMPLES grid of floor points across its footprint."""
    pixels = []
    for v in PATCH:
        for u in PATCH:
            px = min(RESOLUTION - 1, int(u * RESOLUTION))
            py = min(RESOLUTION - 1, int(v * RESOLUTION))
            points = []
            for i in range(SUBSAMPLES):
                for j in range(SUBSAMPLES):
                    fx = px + (i + 0.5) / SUBSAMPLES
                    fy = py + (j + 0.5) / SUBSAMPLES
                    points.append((-2.5 + 2.0 * fx / RESOLUTION, 1.0 - 2.0 * fy / RESOLUTION))
            pixels.append(points)
    return pixels


PIXELS = pixel_subsamples()


def inside(square, x, y):
    return square[0] <= x <= square[1] and square[2] <= y <= square[3]


def ratios(light, square):
    """Exact shadowed-to-lit ratios over the patch: the lit-weighted mean of the shadow pass, for the transparent
    occluder and for the opaque one. A floor point is shadowed by the occluder when its ray to the light crosses the
    occluder square on the occluder plane."""
    transparent_sum = opaque_sum = weight_sum = 0.0
    for points in PIXELS:
        for x, y in points:
            if light == "SUN":
                # The sun travels along (+x, -z)/sqrt(2); a floor point reaches the occluder plane at (x - 1, y).
                weight = 1.0
                hit = inside(square, x - 1.0, y)
            else:
                d = (POINT_LIGHT[0] - x, POINT_LIGHT[1] - y, POINT_LIGHT[2])
                distance_squared = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
                weight = (d[2] / math.sqrt(distance_squared)) / distance_squared
                t = 1.0 / d[2]
                hit = inside(square, x + t * d[0], y + t * d[1])
            weight_sum += weight
            # A crossed transparent occluder passes its `1 - alpha` of the light; an opaque one passes none.
            transparent_sum += weight * ((1.0 - OCCLUDER_ALPHA) if hit else 1.0)
            opaque_sum += weight * (0.0 if hit else 1.0)
    return transparent_sum / weight_sum, opaque_sum / weight_sum


def potter_bound(light, delta, steps=21):
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
    """Potter's edge-placement bound for the transparent and the opaque ratio of `light`."""
    if light == "SUN":
        return potter_bound(light, SUN_BOUND)
    texel_on_occluder = 2.0 * POINT_DEPTH_TO_OCCLUDER / RESOLUTION
    _, bound_off = potter_bound(light, POINT_EDGE_TEXELS * texel_on_occluder)
    # The transparent occluder is traced exactly, so its edge has no placement error.
    return 0.0, bound_off


def derive(light, runs):
    """The analysis of one light's runs: the exact ratios, EEVEE's mean and spread, its bias and the tolerances."""
    exact_alpha, exact_off = ratios(light, OCCLUDER)
    bound_alpha, bound_off = potter_edge_bound(light)
    entry = {
        "runs": runs,
        "exact": {"alpha_ratio": exact_alpha, "alpha_off_ratio": exact_off},
        "potter_edge_bound": {"alpha_ratio": bound_alpha, "alpha_off_ratio": bound_off},
    }
    for key, exact, bound in (
        ("alpha_ratio", exact_alpha, bound_alpha),
        ("alpha_off_ratio", exact_off, bound_off),
    ):
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
    if not entry["alpha_ratio"]["mean"] > entry["alpha_off_ratio"]["mean"]:
        raise SystemExit(f"{light}: EEVEE's transparent shadow must be lighter than its opaque one")
    return entry


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


def collect(blender):
    """Renders every light at every shift, and returns the runs per light."""
    runs_by_light = {}
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
            runs_by_light[light] = runs
    return runs_by_light


def main():
    if sys.argv[1] == "--analyse-only":
        out_json = sys.argv[2]
        with open(out_json, encoding="utf-8") as handle:
            previous = json.load(handle)
        runs_by_light = {light: previous["lights"][light]["runs"] for light in LIGHTS}
    else:
        blender, out_json = sys.argv[1], sys.argv[2]
        runs_by_light = collect(blender)
    report = {"shifts_y": SHIFTS, "lights": {}}
    for light in LIGHTS:
        report["lights"][light] = derive(light, runs_by_light[light])
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
