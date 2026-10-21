"""Single-scatter reference for the volume NEE oracle, and its comparison with Blender's render.

For the camera ray through a pixel at (x0, y0), the single-scatter radiance (one scattering event, no
environment) is

    L = integral over the cube of  sigma_s(x) * p(cos) * E(x) * T_cam(x) dz,

with T_cam = exp(-integral of sigma_t from the camera to x), E(x) = I * T_light(x) / |P - x|^2 the point
light's irradiance (T_light is the attenuation from x to the light), and the Henyey-Greenstein phase

    p(cos) = (1 - g^2) / (4 pi (1 + g^2 - 2 g cos)^(3/2)),

with cos = dot(d, L) for the camera direction d = (0, 0, -1) and L the unit vector from x to the light.
That is the physical scattering angle (forward scattering for g > 0). I = 1 here; Blender's light unit is
calibrated from the diffuse case. The printed ratios are Blender / analytic, and the diffuse ratio is the
light-unit calibration that the volume ratios must match.

Run: python3 analytic_volume_nee.py BLENDER.json
"""

import json
import math
import sys

POSITION = None
GRID = 400
HALF = 0.5


def hg(cosine, g):
    return (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * cosine) ** 1.5)


def density_at(point, kind, density, gradient_scale):
    x, y, z = point
    if abs(x) > HALF or abs(y) > HALF or abs(z) > HALF:
        return 0.0
    if kind == "homogeneous":
        return density
    return gradient_scale * (x + 0.5)


def light_transmittance(point, light, kind, density, gradient_scale):
    dx = [light[i] - point[i] for i in range(3)]
    length = math.sqrt(sum(c * c for c in dx))
    steps = GRID
    ds = length / steps
    optical = 0.0
    for step in range(steps):
        t = (step + 0.5) * ds
        q = [point[i] + dx[i] / length * t for i in range(3)]
        optical += density_at(q, kind, density, gradient_scale) * ds
    return math.exp(-optical), length


def diffuse_radiance(x, y, light):
    # White Lambert plane z = 0, irradiance I cos / d^2, radiance = E / pi.
    point = (x, y, 0.0)
    dx = [light[i] - point[i] for i in range(3)]
    d2 = sum(c * c for c in dx)
    d = math.sqrt(d2)
    cos_surface = dx[2] / d
    return cos_surface / d2 / math.pi


def volume_radiance(x, y, light, kind, density, gradient_scale, g):
    dz = 1.0 / GRID
    # Camera attenuation accumulated from the cube's top face (z = +0.5) downward.
    optical_camera = 0.0
    total = 0.0
    for step in range(GRID):
        z = HALF - (step + 0.5) * dz
        point = (x, y, z)
        sigma = density_at(point, kind, density, gradient_scale)
        t_camera = math.exp(-optical_camera - 0.5 * sigma * dz)
        optical_camera += sigma * dz
        t_light, distance = light_transmittance(point, light, kind, density, gradient_scale)
        to_light = [(light[i] - point[i]) / distance for i in range(3)]
        cosine = (0.0 * to_light[0] + 0.0 * to_light[1] + -1.0 * to_light[2])
        phase = hg(cosine, g)
        irradiance = t_light / (distance * distance)
        total += sigma * phase * irradiance * t_camera * dz
    return total


def main():
    blender = json.load(open(sys.argv[1]))
    light = tuple(blender["position"])
    density = blender["density"]
    gradient_scale = blender["gradient_scale"]
    centres = blender["pixel_centres"]
    points = [(x, y) for x in centres for y in centres]
    cases = {
        "diffuse": lambda x, y: diffuse_radiance(x, y, light),
        "homogeneous_g0": lambda x, y: volume_radiance(x, y, light, "homogeneous", density, gradient_scale, 0.0),
        "homogeneous_g05": lambda x, y: volume_radiance(x, y, light, "homogeneous", density, gradient_scale, 0.5),
        "gradient_g0": lambda x, y: volume_radiance(x, y, light, "gradient", density, gradient_scale, 0.0),
        "gradient_g05": lambda x, y: volume_radiance(x, y, light, "gradient", density, gradient_scale, 0.5),
    }
    for name, function in cases.items():
        analytic = sum(function(x, y) for x, y in points) / len(points)
        measured = blender["cases"][name]["mean"]
        print(f"{name:16s} analytic(I=1) {analytic:.6f} blender {measured:.6f} ratio {measured / analytic:.4f}")


if __name__ == "__main__":
    main()
