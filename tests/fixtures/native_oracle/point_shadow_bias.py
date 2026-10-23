"""Shadow-map self-shadowing for a point light on a unit sphere, measured against exact rays.

Potter's point-light shadow map (`render/scene.rs`, `PointShadowMap`) is six 64 x 64 faces over the light, each texel holding the
nearest depth of the triangles it covers. A receiver is visible when its depth is within a bias of the stored depth at its
texel and the PCF taps around it. On a curved surface the neighbouring taps see the surface at a slightly different depth, so
a bias below a texel's worth of slope puts the lit side of the sphere in shadow. This script rasterizes the sphere the way
Potter does, then counts how many points that are lit (checked against exact ray casts) the map shadows, for a bias of
`0.001 · depth` (the old value) and for a bias of `k` texels (a texel is `2 · depth / 64` across at depth `depth`).

It also counts the wrong results on a plane under the sphere, where the true shadow is known, so a bias that is too large
(peter-panning) would show.

Usage: python3 point_shadow_bias.py
"""

import importlib.util
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("direct_mirror", HERE / "eevee_glass_direct_mirror.py")
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

SIZE = 64
LIGHT = (0.0, 0.0, 4.0)
CENTRE = (-2.4, 0.0, 0.0)
verts, tris = mirror.uv_sphere(CENTRE, 1.0)

BASES = [
    ((1, 0, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1)),
    ((0, -1, 0), (0, 0, 1)),
    ((0, 0, 1), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0)),
]


def rasterize_faces():
    """The six depth faces: `(forward, right, up, depth)` with the nearest depth per texel, as `rasterize_shadow_triangle`."""
    faces = []
    for forward, up_hint in BASES:
        forward = mirror.normalize(forward)
        right = mirror.normalize(mirror.cross(forward, up_hint))
        up = mirror.normalize(mirror.cross(right, forward))
        depth = [math.inf] * (SIZE * SIZE)
        for tri in tris:
            corners = [verts[k] for k in tri]
            relative = [mirror.sub(p, LIGHT) for p in corners]
            depths = [mirror.dot(r, forward) for r in relative]
            if min(depths) <= 1e-6:
                continue
            projected = [
                (
                    (mirror.dot(r, right) / d * 0.5 + 0.5) * SIZE,
                    (0.5 - mirror.dot(r, up) / d * 0.5) * SIZE,
                    d,
                )
                for r, d in zip(relative, depths)
            ]
            a, b, c = projected
            area = edge(a, b, c)
            if abs(area) < 1e-12:
                continue
            x0 = max(0, int(math.floor(min(p[0] for p in projected))))
            x1 = min(SIZE - 1, int(math.ceil(max(p[0] for p in projected))))
            y0 = max(0, int(math.floor(min(p[1] for p in projected))))
            y1 = min(SIZE - 1, int(math.ceil(max(p[1] for p in projected))))
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    sample = (x + 0.5, y + 0.5, 0.0)
                    w = [edge(b, c, sample) / area, edge(c, a, sample) / area, edge(a, b, sample) / area]
                    if min(w) < -1e-9:
                        continue
                    reciprocal = w[0] / a[2] + w[1] / b[2] + w[2] / c[2]
                    if reciprocal <= 1e-12:
                        continue
                    value = 1.0 / reciprocal
                    index = y * SIZE + x
                    if value < depth[index]:
                        depth[index] = value
        faces.append((forward, right, up, depth))
    return faces


def edge(first, second, point):
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])


FACES = rasterize_faces()


def map_visible(point, bias_of_depth):
    """Potter's PCF visibility of `point` from the map, with `bias_of_depth(depth)` as the bias."""
    relative = mirror.sub(point, LIGHT)
    absolute = [abs(c) for c in relative]
    if absolute[0] >= absolute[1] and absolute[0] >= absolute[2]:
        index = 0 if relative[0] >= 0 else 1
    elif absolute[1] >= absolute[2]:
        index = 2 if relative[1] >= 0 else 3
    else:
        index = 4 if relative[2] >= 0 else 5
    forward, right, up, depth_buffer = FACES[index]
    depth = mirror.dot(relative, forward)
    x = mirror.dot(relative, right) / depth
    y = mirror.dot(relative, up) / depth
    centre_x = round((x * 0.5 + 0.5) * SIZE - 0.5)
    centre_y = round((0.5 - y * 0.5) * SIZE - 0.5)
    bias = bias_of_depth(depth)
    taps = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    visible = 0
    for dx, dy in taps:
        sx, sy = centre_x + dx, centre_y + dy
        if sx < 0 or sy < 0 or sx >= SIZE or sy >= SIZE:
            visible += 1
            continue
        if depth <= depth_buffer[sy * SIZE + sx] + bias:
            visible += 1
    return visible / len(taps)


def exact_visible(point, normal):
    to_light = mirror.sub(LIGHT, point)
    distance = mirror.length(to_light)
    direction = mirror.normalize(to_light)
    origin = mirror.add(point, mirror.mul(normal, 1e-5))
    for tri in tris:
        hit = mirror.ray_triangle(origin, direction, *(verts[k] for k in tri))
        if hit and 1e-5 < hit[0] < distance:
            return 0.0
    return 1.0


def main():
    random.seed(1)
    lit = []
    while len(lit) < 250:
        x, y = random.uniform(-0.95, 0.95), random.uniform(-0.95, 0.95)
        if x * x + y * y >= 0.9:
            continue
        normal = (x, y, math.sqrt(1 - x * x - y * y))
        point = mirror.add(CENTRE, normal)
        to_light = mirror.normalize(mirror.sub(LIGHT, point))
        if mirror.dot(normal, to_light) < 0.15:
            continue
        lit.append((point, normal))
    truth = [exact_visible(p, n) for p, n in lit]
    print(f"lit points {len(lit)}, exactly visible {int(sum(truth))}")

    def self_shadowed(bias_of_depth):
        return sum(1 for (p, n), t in zip(lit, truth) if abs(map_visible(p, bias_of_depth) - t) > 0.5)

    print(f"bias 0.001 * depth (old):      {self_shadowed(lambda d: max(d * 0.001, 1e-4))} lit points shadowed")
    for k in (0.5, 1.0, 1.5):
        count = self_shadowed(lambda d, k=k: k * 2.0 * d / SIZE)
        print(f"bias {k} texel (2 depth / {SIZE}):  {count} lit points shadowed")

    plane = []
    for i in range(-12, 13):
        for j in range(-12, 13):
            plane.append((CENTRE[0] + i * 0.1, j * 0.1, -1.2))
    plane_truth = [exact_visible(p, (0.0, 0.0, 1.0)) for p in plane]
    print(f"plane points {len(plane)}, truly shadowed {int(len(plane) - sum(plane_truth))}")

    def wrong_on_plane(bias_of_depth):
        return sum(1 for p, t in zip(plane, plane_truth) if abs(map_visible(p, bias_of_depth) - t) > 0.5)

    print(f"bias 0.001 * depth: {wrong_on_plane(lambda d: max(d * 0.001, 1e-4))} wrong on the plane")
    for k in (0.5, 1.0, 1.5):
        print(f"bias {k} texel: {wrong_on_plane(lambda d, k=k: k * 2.0 * d / SIZE)} wrong on the plane")


if __name__ == "__main__":
    main()
