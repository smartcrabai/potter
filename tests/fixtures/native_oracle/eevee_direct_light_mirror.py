"""Python mirror of the realtime direct light's cut-offs, spot cone and sun disc, checked against
`eevee_direct_light_cases.py` (EEVEE 5.2.2).

The mirror is the specification of Potter's realtime direct light (`render/light_model.rs` `local_light_influence` and
`spot_cone`, `render/scene.rs` `influence_at`, `light_contribution` and `shadow_visibility`). It predicts each plane pixel of
the oracle's JSON and reports the per-channel deviation from EEVEE's pixel.

Sources (Blender v5.2.2):
- `draw/engines/eevee/shaders/eevee_light_lib.bsl.hh`: `light_attenuation_surface`, `light_influence_attenuation`,
  `light_attenuation_common`, `light_spot_attenuation`, `light_attenuation_facing`.
- `draw/engines/eevee/eevee_light.cc`: `attenuation_radius_get` (the influence radius `sqrt(power / threshold)`, or the custom
  distance), `shape_parameters_set` (the spot's `spot_size_inv`, `spot_mul` and `spot_bias`), `shape_radiance_get` and
  `point_radiance_get`, and the sun's `shape_radius` (`tan(angle / 2)`, clamped to [0.001, 20]).
- `draw/engines/eevee/shaders/eevee_shadow_tracing.bsl.hh`: `shadow_ray_generate_directional` (the sun's shadow cone).
- `blenkernel/intern/light.cc`: `BKE_light_power` (the energy times 2^exposure).

The mirror predicts Potter, not EEVEE's LTC: the point light is the delta of Potter's `punctual_radiance`, and the area
light is the exact rectangle irradiance of `area_irradiance_per_radiance`. The sun's shadow is the mean over the 16
Vogel-spiral rays of the disc (`sun_disc_visibility`), which is the default variant. `--exact` uses 64 rays, `--pcf9` is
the nine-tap filter at the new radius, and `--old` is Potter's nine-tap filter at the old fixed 0.003 rad.

Usage: python3 eevee_direct_light_mirror.py <case.json> [--dump <out.json>]
"""

from __future__ import annotations

import json
import math
import sys

LIGHT_THRESHOLD_DEFAULT = 0.01
# The sun's penumbra before the disc: Potter's nine-tap PCF at a fixed 0.003 rad.
SUN_ANGULAR_RADIUS_POTTER_OLD = 0.003
# Potter's sun disc (`SUN_DISC_SAMPLES` in `scene.rs`).
SUN_DISC_SAMPLES = 16


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    n = length(a)
    return (0.0, 0.0, 0.0) if n == 0.0 else mul(a, 1.0 / n)


def saturate(x):
    return min(max(x, 0.0), 1.0)


def smoothstep(x):
    t = saturate(x)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------------------------------------------------
# Influence (`light_influence_attenuation`, with `attenuation_radius_get`).


def influence(light, distance, threshold):
    """`light_influence_attenuation`: `(1 - (d² / R²)²)²`, zero beyond the radius and zero at zero radius (culled)."""
    if light["use_custom_distance"]:
        radius_squared = light["cutoff_distance"] ** 2
    else:
        max_power = max(light["color"]) * abs(light["energy"]) / 100.0
        radius_squared = max_power / threshold
    if radius_squared <= 0.0:
        return 0.0
    factor = saturate(1.0 - (distance * distance / radius_squared) ** 2)
    return factor * factor


# ---------------------------------------------------------------------------------------------------------------------
# Spot cone (`light_spot_attenuation`, EEVEE's form) and Potter's Cycles form for comparison.


def spot_cone_eevee(light_matrix, scale, to_light, spot_size, spot_blend):
    """The spot's cone with EEVEE's ellipse: the direction to the light in the light's frame, stretched by its scale."""
    local = (dot(light_matrix[0], to_light), dot(light_matrix[1], to_light), dot(light_matrix[2], to_light))
    if local[2] <= 0.0:
        return 0.0
    cos_half = math.cos(spot_size * 0.5)
    blend_width = (1.0 - cos_half) * spot_blend
    spot_mul = 1.0 / max(1e-8, blend_width)
    spot_bias = -cos_half * spot_mul
    size_inv = (scale[2] / max(scale[0], 1e-8), scale[2] / max(scale[1], 1e-8))
    stretched = ((local[0] * size_inv[0]) / local[2], (local[1] * size_inv[1]) / local[2])
    ellipse = 1.0 / math.sqrt(1.0 + stretched[0] ** 2 + stretched[1] ** 2)
    return smoothstep(ellipse * spot_mul + spot_bias)


def spot_cone_cycles(axis, to_light, spot_size, spot_blend):
    """Potter's Cycles form: `smoothstep((cos θ - cos(angle / 2)) / ((1 - cos(angle / 2)) · blend))`."""
    cos_half = math.cos(spot_size * 0.5)
    spread = (1.0 - cos_half) * spot_blend
    cosine = dot(axis, mul(to_light, -1.0))
    return smoothstep((cosine - cos_half) / spread) if spread > 0.0 else float(cosine > cos_half)


# ---------------------------------------------------------------------------------------------------------------------
# Irradiance per unit radiance of a rectangle, Lambert (`area_irradiance_per_radiance`: Arvo's polygon form).


def rectangle_irradiance(corners, point, normal):
    total = 0.0
    for index in range(4):
        a = sub(corners[index], point)
        b = sub(corners[(index + 1) % 4], point)
        c = cross(a, b)
        c_length = length(c)
        if c_length <= 1e-12:
            continue
        subtended = math.atan2(c_length, dot(a, b))
        total += subtended * dot(normal, c) / c_length
    return max(0.5 * total, 0.0)


# ---------------------------------------------------------------------------------------------------------------------
# Sun shadows: Potter's nine-tap PCF over the cone (`shadow_visibility`), and the exact disc.


def blocked(sphere_centre, sphere_radius, origin, direction, distance):
    oc = sub(origin, sphere_centre)
    b = dot(oc, direction)
    c = dot(oc, oc) - sphere_radius * sphere_radius
    disc = b * b - c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if 1e-5 < t < distance:
            return True
    return False


def tangent_frame(normal):
    """`shadow_visibility`'s tangent and bitangent of the receiver's normal."""
    if abs(normal[2]) < 0.999:
        tangent = normalize(cross(normal, (0.0, 0.0, 1.0)))
    else:
        tangent = normalize(cross(normal, (0.0, 1.0, 0.0)))
    return tangent, normalize(cross(normal, tangent))


def sun_visibility_potter(occluder, point, normal, direction, angular_radius):
    offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    tangent, bitangent = tangent_frame(normal)
    visible = 0.0
    origin = add(point, mul(normal, 1e-5))
    for ox, oy in offsets:
        sample = normalize(add(direction, add(mul(tangent, ox * angular_radius), mul(bitangent, oy * angular_radius))))
        visible += 0.0 if blocked(occluder[0], occluder[1], origin, sample, math.inf) else 1.0
    return visible / 9.0


def sun_visibility_exact(occluder, point, normal, direction, angular_radius, samples=64):
    """The fraction of the sun's disc that is unoccluded, from a Vogel spiral over the disc of `angular_radius`."""
    tangent, bitangent = tangent_frame(direction)
    origin = add(point, mul(normal, 1e-5))
    visible = 0.0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(samples):
        radius = math.sqrt((index + 0.5) / samples) * angular_radius
        angle = index * golden
        sample = normalize(add(direction, add(mul(tangent, radius * math.cos(angle)), mul(bitangent, radius * math.sin(angle)))))
        visible += 0.0 if blocked(occluder[0], occluder[1], origin, sample, math.inf) else 1.0
    return visible / samples


def sun_radius(angle_degrees):
    """`eevee_light.cc` `shape_radius`: `tan(angle / 2)`, clamped to [0.001, 20] (the same for Potter's disc)."""
    return min(max(math.tan(math.radians(angle_degrees) * 0.5), 0.001), 20.0)


# ---------------------------------------------------------------------------------------------------------------------
# The plane's pixel.


def predict(case, variant):
    """The predicted RGB of a plane pixel, for the case, with `variant` choosing the spot cone or the sun's shadow."""
    light = case["light"]
    albedo = case["albedo"]
    threshold = light["threshold"]
    normal = (0.0, 0.0, 1.0)
    location = tuple(light["location"])
    predictions = []
    light_dict = {
        "use_custom_distance": light["custom_distance"] is not None,
        "cutoff_distance": light["custom_distance"] or 40.0,
        "color": (1.0, 1.0, 1.0),
        "energy": light["energy"],
    }
    scale = tuple(light["scale"])
    occluder = None
    if case["occluder"] is not None:
        occluder = (tuple(case["occluder"]["centre"]), case["occluder"]["radius"])
    for sample in case["samples"]:
        point = (sample["world"][0], sample["world"][1], 0.0)
        to_light = sub(location, point)
        distance = length(to_light)
        direction_to_light = mul(to_light, 1.0 / distance)
        cos_surface = dot(normal, direction_to_light)
        if light["type"] == "POINT":
            intensity = light["energy"] / (4.0 * math.pi)
            radiance = albedo / math.pi * intensity / (distance * distance) * max(cos_surface, 0.0)
            value = radiance * influence(light_dict, distance, threshold)
        elif light["type"] == "SPOT":
            intensity = light["energy"] / (4.0 * math.pi)
            axis = normalize(mul((0.0, 0.0, 1.0), -1.0))
            light_matrix = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            spot_size = math.radians(light["spot_size_degrees"])
            if variant == "cycles":
                cone = spot_cone_cycles(axis, direction_to_light, spot_size, light["spot_blend"])
            else:
                cone = spot_cone_eevee(light_matrix, scale, direction_to_light, spot_size, light["spot_blend"])
            value = albedo / math.pi * intensity / (distance * distance) * max(cos_surface, 0.0) * cone
            value *= influence(light_dict, distance, threshold)
        elif light["type"] == "AREA":
            half = light["area_size"] / 2.0
            corners = [
                (location[0] - half, location[1] - half, location[2]),
                (location[0] + half, location[1] - half, location[2]),
                (location[0] + half, location[1] + half, location[2]),
                (location[0] - half, location[1] + half, location[2]),
            ]
            area = light["area_size"] ** 2
            radiance = light["energy"] / (math.pi * area)
            irradiance = rectangle_irradiance(corners, point, normal)
            value = albedo / math.pi * radiance * irradiance
            value *= influence(light_dict, distance, threshold)
        else:  # SUN
            strength = light["energy"]
            direction = (0.0, 0.0, 1.0)
            radius = sun_radius(light["angle_degrees"])
            if variant == "exact":
                visibility = sun_visibility_exact(occluder, point, normal, direction, radius, samples=64)
            elif variant == "old":
                visibility = sun_visibility_potter(occluder, point, normal, direction, SUN_ANGULAR_RADIUS_POTTER_OLD)
            elif variant == "pcf9":
                visibility = sun_visibility_potter(occluder, point, normal, direction, radius)
            else:
                visibility = sun_visibility_exact(occluder, point, normal, direction, radius, samples=SUN_DISC_SAMPLES)
            value = albedo / math.pi * strength * max(dot(normal, direction), 0.0) * visibility
        predictions.append([value, value, value])
    return predictions


def deviation(case, predictions):
    """The largest and mean per-channel deviation over the plane pixels, leaving out a sun case's occluder: the camera sees
    the sphere itself there, not the plane."""
    occluder = case["occluder"]
    worst = 0.0
    total = 0.0
    counted = 0
    worst_at = None
    for sample, predicted in zip(case["samples"], predictions):
        if occluder is not None and math.hypot(sample["world"][0] - occluder["centre"][0], sample["world"][1] - occluder["centre"][1]) < occluder["radius"]:
            continue
        counted += 1
        error = max(abs(p - w) for p, w in zip(predicted, sample["rgb"]))
        total += error
        if error > worst:
            worst = error
            worst_at = sample["world"]
    return {"max": worst, "mean": total / counted, "at": worst_at}


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    case = json.loads(open(argv[1], encoding="utf-8").read())
    variant = "cycles" if case["light"]["type"] == "SPOT" and "--cycles" in argv else "eevee"
    if case["light"]["type"] == "SUN":
        variant = next((name for name in ("exact", "old", "pcf9") if f"--{name}" in argv), "potter")
    predictions = predict(case, variant)
    report = {"case": case["case"], "variant": variant, "deviation": deviation(case, predictions)}
    print(json.dumps(report, indent=2))
    if "--dump" in argv:
        with open(argv[argv.index("--dump") + 1], "w", encoding="utf-8") as handle:
            json.dump(predictions, handle)


if __name__ == "__main__":
    main(sys.argv)
