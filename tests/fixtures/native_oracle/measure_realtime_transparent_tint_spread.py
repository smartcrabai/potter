"""Measures EEVEE's patch-mean spread for the Transparent tint scene over sub-millimetre shifts.

Each shift moves the whole scene along y by `k * 0.0001` (k = 1..8), which gives an independent realisation of EEVEE's
dither noise (`realtime_transparent_tint.py` hashes world position). The statistics here are the source of the values in
`realtime_transparent_tint.json`: the mean, the sample standard deviation, and the standard error over the shifts.

Usage (system Python; Blender runs the generator for each shift):

    python3 measure_realtime_transparent_tint_spread.py BLENDER OUT_JSON

Writes OUT_JSON with, per method, the per-shift patch means and their statistics. Nothing here renders Potter.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = os.path.join(HERE, "realtime_transparent_tint.py")
METHODS = ["BLENDED", "DITHERED"]
SHIFTS = [k * 0.0001 for k in range(1, 9)]


def render(blender, method, shift, workdir):
    tag = "%s_%d" % (method, round(shift * 1e4))
    blend = os.path.join(workdir, "scene_%s.blend" % tag)
    image = os.path.join(workdir, "eevee_%s.exr" % tag)
    out = os.path.join(workdir, "eevee_%s.json" % tag)
    subprocess.run(
        [blender, "-b", "--factory-startup", "--python", GENERATOR, "--",
         blend, image, out, method, "1", repr(shift)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open(out, encoding="utf-8") as handle:
        return json.load(handle)["patch_mean"]


def statistics(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd, sd / math.sqrt(n)


def main():
    blender, out_path = sys.argv[1], sys.argv[2]
    result = {"shifts_y": SHIFTS, "methods": {}}
    with tempfile.TemporaryDirectory() as workdir:
        for method in METHODS:
            per_shift = [render(blender, method, shift, workdir) for shift in SHIFTS]
            channels = []
            for channel in range(3):
                mean, sd, se = statistics([patch[channel] for patch in per_shift])
                channels.append({"mean": mean, "sd": sd, "standard_error": se})
            result["methods"][method] = {"per_shift_patch_mean": per_shift, "channels": channels}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
