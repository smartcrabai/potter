"""Python mirror of the light closure factors and the sun's disc, checked against `eevee_light_factors.py` (EEVEE 5.2.2).

The factors (EEVEE's `eevee_light.cc`: `power[LIGHT_DIFFUSE] = diff_fac`, `power[LIGHT_SPECULAR] = spec_fac`,
`power[LIGHT_TRANSMISSION] = transmission_fac`) scale a light's contribution per closure class, so each prediction is linear:
- diffuse: `diffuse_half = 0.5 · diffuse_base`;
- specular: `specular_half = 0.5 · specular_base`;
- glass: the reflection does not depend on the transmission factor, so `glass_half = zero + 0.5 · (base - zero)`, where `zero`
  is the reflection alone.

The sun's disc (`eevee_light.cc` `shape_radius`, the disc of `tan(angle / 2)`) integrates the specular lobe over the disc. Over a
uniform plane seen from above, the lobe's shape is the same at every pixel, so the disc's effect is one number: the ratio of the
lobe averaged over the disc to its value at the centre. `R = mean_i f(L_i) cos θ_i / (f(L_c) cos θ_c)`, with `f` the GGX
distribution, the height-correlated Smith G and a constant Fresnel (which varies by well under a per cent across the disc).
EEVEE's `sun_disc` pixel is then `sun_delta · R`, to the accuracy of EEVEE's LTC integral. Potter takes the disc's mean over its
16 Vogel-spiral samples (`SUN_DISC_SAMPLES`), which this mirror also evaluates at 16 and 256 to show the sampling is converged.

Usage: python3 eevee_light_factors_mirror.py <case dir with f_<case>.json>
"""

import json
import math
import sys
from pathlib import Path

SAMPLES = 16


def ggx_value_times_cosine(cos_light, cos_view, cos_half, alpha):
    """The GGX lobe times the light's cosine: D · G2 / (4 N·V) · N·L, with Smith's height-correlated G2."""
    a2 = alpha * alpha
    d = a2 / (math.pi * ((cos_half * cos_half) * (a2 - 1.0) + 1.0) ** 2)

    def lambda_of(c):
        return (-1.0 + math.sqrt(1.0 + a2 * (1.0 - c * c) / (c * c))) / 2.0

    g = 1.0 / (1.0 + lambda_of(cos_view) + lambda_of(cos_light))
    return d * g / (4.0 * cos_view * cos_light) * cos_light


def lobe_at(direction, roughness):
    """The lobe at the light direction `direction` (a unit vector) seen from +Z, with the half vector between them."""
    alpha = roughness * roughness
    x, y, z = direction
    hx, hy, hz = x, y, z + 1.0
    h = math.sqrt(hx * hx + hy * hy + hz * hz)
    return ggx_value_times_cosine(z, 1.0, hz / h, alpha)


def disc_directions(radius_tangent, count):
    """Vogel spiral directions over the disc of tangent radius `radius_tangent` about +Z, as `sun_disc_directions` gives them."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        r = math.sqrt((index + 0.5) / count) * radius_tangent
        angle = index * golden
        x, y, z = r * math.cos(angle), r * math.sin(angle), 1.0
        n = math.sqrt(x * x + y * y + z * z)
        yield (x / n, y / n, z / n)


def disc_ratio(angle_degrees, roughness, count):
    radius = math.tan(math.radians(angle_degrees) * 0.5)
    centre = lobe_at((0.0, 0.0, 1.0), roughness)
    total = sum(lobe_at(direction, roughness) for direction in disc_directions(radius, count))
    return total / count / centre


def load(directory, name):
    return json.loads((Path(directory) / f"f_{name}.json").read_text(encoding="utf-8"))


def rgb_of(case):
    return [sample["rgb"][0] for sample in case["samples"]]


def max_dev(a, b, floor=0.0):
    worst = 0.0
    for x, y in zip(a, b):
        if max(abs(x), abs(y)) > floor:
            worst = max(worst, abs(x - y))
    return worst


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    directory = argv[1]
    report = {}
    base, half = rgb_of(load(directory, "diffuse_base")), rgb_of(load(directory, "diffuse_half"))
    report["diffuse_half_vs_half_base"] = max_dev(half, [0.5 * b for b in base])

    base, half = rgb_of(load(directory, "specular_base")), rgb_of(load(directory, "specular_half"))
    report["specular_half_vs_half_base"] = max_dev(half, [0.5 * b for b in base])

    base = rgb_of(load(directory, "glass_base"))
    zero = rgb_of(load(directory, "glass_zero"))
    half = rgb_of(load(directory, "glass_half"))
    report["glass_half_vs_model"] = max_dev(half, [z + 0.5 * (b - z) for b, z in zip(base, zero)])
    report["glass_transmitted_peak"] = max(b - z for b, z in zip(base, zero))

    delta = rgb_of(load(directory, "sun_delta"))
    disc = rgb_of(load(directory, "sun_disc"))
    angle = load(directory, "sun_disc")["light"]["angle_degrees"]
    roughness = 0.3
    predicted_16 = [d * disc_ratio(angle, roughness, SAMPLES) for d in delta]
    predicted_256 = [d * disc_ratio(angle, roughness, 256) for d in delta]
    report["sun_disc_ratio_potter_16"] = disc_ratio(angle, roughness, SAMPLES)
    report["sun_disc_ratio_converged"] = disc_ratio(angle, roughness, 256)
    report["sun_disc_ratio_eevee"] = sum(disc) / sum(delta)
    report["sun_disc_vs_potter_16"] = max_dev(disc, predicted_16, floor=0.05)
    report["sun_disc_vs_converged"] = max_dev(disc, predicted_256, floor=0.05)
    report["sun_disc_vs_centre_only"] = max_dev(disc, delta, floor=0.05)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv)
