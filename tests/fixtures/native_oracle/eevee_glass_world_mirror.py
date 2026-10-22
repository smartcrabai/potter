"""Model bound of the glass world test: the transmission model of `eevee_transmission.rs` over the world of
`eevee_glass_world.py`, against the committed macOS arm64 EEVEE render (`eevee_macos/glass_world.f32le`).

The port follows the Rust functions in order: `probe_radiance` (EEVEE's UFLOAT 11-11-10 capture, six mantissa bits for red
and green and five for blue, truncated), `prefiltered_with` (five taps, each captured, then averaged), `reflection_direction`,
`transmission_roughness`, `dominant_transmission`, `sphere_exit_normal` and `transmission_direction` (a sphere thickness,
so two refraction events), and `glass_closures`' weights (the reflectance, and the transmittance squared with a thickness).
`glass_split` is `eevee_glass_direct_mirror.py`'s port of `bsdf_lut`, called once per channel with the tint
`sqrt(Base Color)`, since its normalisation depends on the tint.

The world is `t = (z + 1) / 2`, colour `(0.25 + 0.5 t, 0.5, 0.75 - 0.5 t)` by the direction's elevation `z`.

The test's tolerance is the largest deviation printed here, rounded up. Run it with no arguments:

    python3 eevee_glass_world_mirror.py
"""

import importlib.util
import math
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
FIXTURE = HERE / "eevee_macos" / "glass_world.f32le"
# The oracle's spheres: name, centre X, Base Color, roughness.
SPHERES = [
    ("clear", -2.4, (1.0, 1.0, 1.0), 0.0),
    ("tinted", 0.0, (0.9, 0.5, 0.2), 0.0),
    ("rough", 2.4, (1.0, 1.0, 1.0), 0.5),
]


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, s):
    return tuple(x * s for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalize_or(v, fallback):
    length = math.sqrt(dot(v, v))
    return fallback if length == 0.0 else mul(v, 1.0 / length)


def lerp(a, b, t):
    return add(a, mul(sub(b, a), t))


def world(direction):
    t = (direction[2] + 1.0) / 2.0
    return (0.25 + 0.5 * t, 0.5, 0.75 - 0.5 * t)


def truncate_ufloat(value, mantissa_bits):
    """`truncate_ufloat` in `eevee_transmission.rs`: a 5-bit exponent (bias 15), truncated, no rounding."""
    if value <= 0.0:
        return 0.0
    largest = (2.0 - 2.0 ** -mantissa_bits) * 32768.0
    value = min(value, largest)
    exponent = min(max(math.floor(math.log2(value)), -14), 15)
    step = 2.0 ** (exponent - mantissa_bits)
    return math.floor(value / step) * step


def probe_radiance(radiance):
    return (
        truncate_ufloat(radiance[0], 6),
        truncate_ufloat(radiance[1], 6),
        truncate_ufloat(radiance[2], 5),
    )


def prefiltered(axis, roughness):
    """`prefiltered_with`: five taps over the roughness's spread, each captured, averaged."""
    axis = normalize_or(axis, (0.0, 0.0, 0.0))
    if axis == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 0.0)
    if abs(axis[2]) < 0.999:
        tangent = normalize_or(cross(axis, (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))
    else:
        tangent = normalize_or(cross(axis, (0.0, 1.0, 0.0)), (0.0, 0.0, 0.0))
    bitangent = normalize_or(cross(axis, tangent), (0.0, 0.0, 0.0))
    spread = min(max(roughness, 0.0), 1.0) ** 2 * 0.5
    total = (0.0, 0.0, 0.0)
    for offset_x, offset_y in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        direction = normalize_or(
            add(axis, add(mul(tangent, spread * offset_x), mul(bitangent, spread * offset_y))),
            (0.0, 0.0, 0.0),
        )
        total = add(total, probe_radiance(world(direction)))
    return mul(total, 1.0 / 5.0)


def dominant_mix(roughness):
    m = roughness * roughness
    smoothness = 1.0 - m
    return smoothness * (math.sqrt(smoothness) + m)


def reflect(incident, normal):
    return sub(incident, mul(normal, 2.0 * dot(normal, incident)))


def refract(incident, normal, eta):
    cos_incident = dot(normal, incident)
    k = 1.0 - eta * eta * (1.0 - cos_incident * cos_incident)
    if k < 0.0:
        return (0.0, 0.0, 0.0)
    return sub(mul(incident, eta), mul(normal, eta * cos_incident + math.sqrt(k)))


def reflection_direction(normal, view, roughness):
    mirror_direction = reflect(mul(view, -1.0), normal)
    return normalize_or(lerp(normal, mirror_direction, dominant_mix(roughness)), normal)


def dominant_transmission(normal, view, ior, roughness):
    refracted = refract(mul(view, -1.0), normal, 1.0 / ior)
    return normalize_or(lerp(mul(normal, -1.0), refracted, dominant_mix(roughness)), mul(normal, -1.0))


def transmission_roughness(roughness, ior):
    factor = 1.0 - 1.0 / ior if ior > 1.0 else 0.0
    return roughness * math.sqrt(max(factor, 0.0))


def transmission_direction(normal, view, ior, roughness):
    """A sphere thickness (`Thickness::Sphere`): the entry direction, amended by the sphere's exit normal."""
    perceived = transmission_roughness(roughness, ior)
    entry = dominant_transmission(normal, view, ior, perceived)
    cos_alpha = dot(entry, mul(normal, -1.0))
    exit_normal = normalize_or(add(normal, mul(entry, 2.0 * cos_alpha)), normal)
    return dominant_transmission(mul(exit_normal, -1.0), mul(entry, -1.0), 1.0 / ior, perceived)


def model(centre, base, roughness, u, v):
    """The model's RGB at one interior pixel, or `None` outside the sphere."""
    x = (u - 0.5) * ORTHO_SCALE - centre
    y = (v - 0.5) * ORTHO_SCALE
    r2 = x * x + y * y
    if r2 >= 1.0:
        return None
    normal = (x, y, math.sqrt(1.0 - r2))
    view = (0.0, 0.0, 1.0)
    normal_dot_view = normal[2]
    reflected = prefiltered(reflection_direction(normal, view, roughness), roughness)
    transmitted = prefiltered(
        transmission_direction(normal, view, IOR, roughness),
        transmission_roughness(roughness, IOR),
    )
    values = []
    for channel in range(3):
        tint = math.sqrt(base[channel])
        reflectance, transmittance = mirror.glass_split(IOR, roughness, normal_dot_view, F0, tint)
        values.append(reflectance * reflected[channel] + transmittance * transmittance * transmitted[channel])
    return values


def main():
    mirror.TABLES = mirror.Tables(mirror.DEFAULT_TABLE)
    raw = FIXTURE.read_bytes()
    if len(raw) != SIZE * SIZE * 12:
        raise SystemExit(f"{FIXTURE.name} holds {len(raw)} bytes, expected {SIZE * SIZE * 12}")
    eevee = array("f")
    eevee.frombytes(raw)
    scale = SIZE / ORTHO_SCALE
    overall = 0.0
    for name, centre_x, base, roughness in SPHERES:
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
                predicted = model(centre_x, base, roughness, u, v)
                wanted = eevee[(row * SIZE + column) * 3 : (row * SIZE + column) * 3 + 3]
                count += 1
                for channel in range(3):
                    error = abs(wanted[channel] - predicted[channel])
                    total[channel] += error
                    if error > worst[channel]:
                        worst[channel] = error
                        worst_at = (u, v, channel)
        if count < 100:
            raise SystemExit(f"{name}: only {count} interior pixels")
        overall = max(overall, max(worst))
        print(
            f"{name}: pixels {count}  max |EEVEE - model| {[round(w, 5) for w in worst]}  "
            f"mean {[round(t / count, 6) for t in total]}  worst at uv ({worst_at[0]:.6f}, {worst_at[1]:.6f}) "
            f"channel {worst_at[2]}"
        )
    print(f"largest deviation {overall:.5f}")


if __name__ == "__main__":
    main()
