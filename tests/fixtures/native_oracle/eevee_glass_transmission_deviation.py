"""Model bound of the glass transmission test: the transmission model against the committed macOS arm64 EEVEE render
(`eevee_macos/glass_transmission.f32le`), over every interior pixel of each sphere.

The model is `eevee_glass_direct_mirror.py`'s `glass_split` at the uniform white world (radiance 1), with the thickness
squared (two events): `reflectance + transmittance²` per channel, the tint `sqrt(Base Color)`. The test's tolerance per
sphere is the largest deviation printed here, rounded up.

Usage: python3 eevee_glass_transmission_deviation.py [--table <f32le>]
"""

import importlib.util
import math
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("direct_mirror", HERE / "eevee_glass_direct_mirror.py")
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

IOR = 1.45
F0 = ((IOR - 1.0) / (IOR + 1.0)) ** 2
ORTHO_SCALE = 7.6
SIZE = 96
INTERIOR = 0.75
FIXTURE = HERE / "eevee_macos" / "glass_transmission.f32le"
# The oracle's spheres: centre along X, roughness, and Base Color (the transmission tint is its square root).
SPHERES = {
    "clear": (-2.4, 0.0, (1.0, 1.0, 1.0)),
    "tinted": (0.0, 0.0, (0.9, 0.5, 0.2)),
    "rough": (2.4, 0.5, (1.0, 1.0, 1.0)),
}


def model(name, u, v):
    """The model's RGB at the uniform white world, or `None` outside the sphere."""
    centre, roughness, base = SPHERES[name]
    x = (u - 0.5) * ORTHO_SCALE - centre
    y = (v - 0.5) * ORTHO_SCALE
    r2 = x * x + y * y
    if r2 >= 1.0:
        return None
    nv = math.sqrt(1.0 - r2)
    values = []
    for channel in range(3):
        tint = math.sqrt(base[channel])
        reflectance, transmittance = mirror.glass_split(IOR, roughness, nv, F0, tint)
        values.append(reflectance + transmittance * transmittance)
    return values


def main(argv):
    mirror.TABLES = mirror.Tables(Path(argv[argv.index("--table") + 1]) if "--table" in argv else mirror.DEFAULT_TABLE)
    raw = FIXTURE.read_bytes()
    if len(raw) != SIZE * SIZE * 12:
        raise SystemExit(f"{FIXTURE.name} holds {len(raw)} bytes, expected {SIZE * SIZE * 12}")
    eevee = array("f")
    eevee.frombytes(raw)
    scale = SIZE / ORTHO_SCALE
    for name, (centre_x, _, _) in SPHERES.items():
        centre_px = (centre_x + ORTHO_SCALE / 2.0) * scale
        worst = [0.0, 0.0, 0.0]
        total = [0.0, 0.0, 0.0]
        count = 0
        worst_at = None
        for row in range(SIZE):
            for column in range(SIZE):
                dx = (column + 0.5 - centre_px) / scale
                dy = (row + 0.5 - SIZE / 2.0) / scale
                if math.hypot(dx, dy) > INTERIOR:
                    continue
                u = (column + 0.5) / SIZE
                v = (row + 0.5) / SIZE
                predicted = model(name, u, v)
                start = (row * SIZE + column) * 3
                wanted = eevee[start : start + 3]
                count += 1
                for channel in range(3):
                    error = abs(wanted[channel] - predicted[channel])
                    total[channel] += error
                    if error > worst[channel]:
                        worst[channel] = error
                        worst_at = (u, v, channel)
        print(
            f"{name}: pixels {count}  max |EEVEE - model| {[round(w, 5) for w in worst]}  "
            f"mean {[round(t / count, 6) for t in total]}  worst at uv ({worst_at[0]:.6f}, {worst_at[1]:.6f}) "
            f"channel {worst_at[2]}"
        )


if __name__ == "__main__":
    main(sys.argv)
