"""Deviation of EEVEE's glass pixels from the transmission model, from the `GLASS_PX` lines of a test log.

`tests/realtime_glass_transmission.rs` prints, for every checked pixel, the EEVEE value of the platform that rendered the
oracle and Potter's value (`GLASS_SPHERE name` opens a sphere's lines, then `GLASS_PX uv_u uv_v eevee_rgb potter_rgb max`).
This script predicts each pixel with the transmission model (`eevee_glass_direct_mirror.py`'s `glass_split`, with the
orthographic view EEVEE uses) and reports, per sphere and channel, the largest and mean of |EEVEE - model| over the same
interior pixels. Run it on the log of a CI run to measure the platform difference, and on the local Blender to measure the
model's own bound.

Usage: python3 eevee_glass_transmission_deviation.py <log> [--table <f32le>]
"""

import importlib.util
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("direct_mirror", HERE / "eevee_glass_direct_mirror.py")
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

IOR = 1.45
F0 = ((IOR - 1.0) / (IOR + 1.0)) ** 2
ORTHO_SCALE = 7.6
# The oracle's spheres: centre along X, roughness, and Base Color (the transmission tint is its square root).
SPHERES = {
    "clear": (-2.4, 0.0, (1.0, 1.0, 1.0)),
    "tinted": (0.0, 0.0, (0.9, 0.5, 0.2)),
    "rough": (2.4, 0.5, (1.0, 1.0, 1.0)),
}
LINE = re.compile(r"^GLASS_PX (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)$")
HEADER = re.compile(r"^GLASS_SPHERE (\w+)$")


def model(name, u, v):
    """The model's RGB at the uniform white world (radiance 1), with the thickness squared (two events)."""
    centre, roughness, base = SPHERES[name]
    world_x = (u - 0.5) * ORTHO_SCALE
    world_y = (v - 0.5) * ORTHO_SCALE
    x = world_x - centre
    r2 = x * x + world_y * world_y
    if r2 >= 1.0:
        return None
    nv = math.sqrt(1.0 - r2)
    values = []
    for channel in range(3):
        tint = math.sqrt(base[channel])
        reflectance, transmittance = mirror.glass_split(IOR, roughness, nv, F0, tint)
        values.append(reflectance + transmittance * transmittance)
    return values


def parse(lines):
    """`(sphere, uv, eevee rgb, potter rgb)` for every pixel line, in log order."""
    sphere = None
    rows = []
    for line in lines:
        header = HEADER.match(line.strip())
        if header:
            sphere = header.group(1)
            continue
        pixel = LINE.match(line.strip())
        if pixel and sphere is not None:
            values = [float(value) for value in pixel.groups()]
            rows.append((sphere, (values[0], values[1]), values[2:5], values[5:8]))
    return rows


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    if "--table" in argv:
        mirror.TABLES = mirror.Tables(Path(argv[argv.index("--table") + 1]))
    else:
        mirror.TABLES = mirror.Tables(mirror.DEFAULT_TABLE)
    rows = parse(Path(argv[1]).read_text(encoding="utf-8").splitlines())
    for name in SPHERES:
        worst = [0.0, 0.0, 0.0]
        total = [0.0, 0.0, 0.0]
        count = 0
        worst_at = None
        for sphere, (u, v), eevee, _ in rows:
            if sphere != name:
                continue
            predicted = model(name, u, v)
            if predicted is None:
                continue
            count += 1
            for channel in range(3):
                error = abs(eevee[channel] - predicted[channel])
                total[channel] += error
                if error > worst[channel]:
                    worst[channel] = error
                    worst_at = (u, v, channel)
        if count == 0:
            print(f"{name}: no pixels")
            continue
        print(
            f"{name}: pixels {count}  max |EEVEE - model| {[round(w, 5) for w in worst]}  "
            f"mean {[round(t / count, 6) for t in total]}  worst at uv {worst_at[:2]} channel {worst_at[2]}"
        )


if __name__ == "__main__":
    main(sys.argv)
