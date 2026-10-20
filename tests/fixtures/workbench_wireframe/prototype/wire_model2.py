"""Workbench wireframe model v2: line records per pixel, Blender's overlay AA (3x3 cross), and linear compositing.

Sources (Blender 5.2.2):
- overlay_wireframe_vert.glsl / _frag.glsl: colour per vertex (facing mix), line record per fragment
  (pack_line_data: perp direction, signed distance, depth), last-writer-wins per pixel.
- overlay_antialiasing.bsl.hh: centre coverage, neighbour cross (+x,-x,+y,-y), alpha over / under by depth,
  the dense-line fix, premultiplied output over the background.
"""
import json
import math
import sys

from pngread import read_png
from wire_model import (DISC_RADIUS, LINE_SMOOTH_END, LINE_SMOOTH_START, clamp01, display, dot3, mat_mul_vec,
                        mix, smoothstep, srgb_to_linear, wire_rgb_linear)


LINE_SPAN = 0.5


def cov(dist):
    # line_coverage(dist, kernel 0, smooth) = smoothstep(END, START, |dist|).
    return smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, abs(dist))


def project(vp, v):
    clip = mat_mul_vec(vp, v + [1.0])
    return (clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3])


def build_records(geometry, oracle, theme_wire, theme_back, facing_variant):
    width, height = oracle["region"]
    view = oracle["view_matrix"]
    proj = oracle["window_matrix"]
    vp = [[sum(proj[r][k] * view[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
    view_z = [view[2][0], view[2][1], view[2][2]]
    rim_lin = [srgb_to_linear(c) for c in theme_wire]
    back_lin = [srgb_to_linear(c) for c in theme_back]
    records = {}  # pixel -> dict(colour, dist, dir, depth)
    for name, ob in geometry.items():
        ndc = [project(vp, v) for v in ob["vertices"]]
        screen = [((x + 1) * 0.5 * width, (1 - (y + 1) * 0.5) * height, z) for x, y, z in ndc]
        face_of_edge = {}
        for fi, face in enumerate(ob["faces"]):
            start = ob["loop_start"][fi]
            for k in range(len(face)):
                a, b = face[k], face[(k + 1) % len(face)]
                face_of_edge.setdefault((min(a, b), max(a, b)), []).append((fi, start + k))
        for a, b in ob["edges"]:
            adjacent = face_of_edge.get((min(a, b), max(a, b)), [])
            if facing_variant == "both":
                uses = adjacent
            elif facing_variant == "last":
                uses = adjacent[-1:]
            else:
                uses = adjacent[:1]
            for fi, corner in uses:
                ca = ob["corner_normals"][corner_of(ob, corner, a)]
                cb = ob["corner_normals"][corner_of(ob, corner, b)]
                col_a = wire_rgb_linear(clamp01(abs(dot3(ca, view_z))), rim_lin, back_lin)
                col_b = wire_rgb_linear(clamp01(abs(dot3(cb, view_z))), rim_lin, back_lin)
                draw(records, width, height, screen[a], screen[b], col_a, col_b)
    return records, back_lin


def corner_of(ob, corner, vertex):
    if ob["loop_vertices"][corner] == vertex:
        return corner
    face = ob["loop_faces"][corner]
    for c in range(len(ob["loop_vertices"])):
        if ob["loop_vertices"][c] == vertex and ob["loop_faces"][c] == face:
            return c
    return corner


def draw(records, width, height, p0, p1, c0, c1):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return
    ux, uy = dx / length, dy / length
    perp = (-uy, ux)
    minx = max(0, int(math.floor(min(x0, x1) - 2)))
    maxx = min(width - 1, int(math.ceil(max(x0, x1) + 2)))
    miny = max(0, int(math.floor(min(y0, y1) - 2)))
    maxy = min(height - 1, int(math.ceil(max(y0, y1) + 2)))
    for py in range(miny, maxy + 1):
        for px in range(minx, maxx + 1):
            rx, ry = px + 0.5 - x0, py + 0.5 - y0
            t = rx * ux + ry * uy
            if t < -LINE_SPAN or t > length + LINE_SPAN:
                continue
            d = rx * perp[0] + ry * perp[1]  # signed distance to the infinite line
            if abs(d) > 0.5:
                continue
            f = min(max(t / length, 0.0), 1.0)
            colour = [mix(a, b, f) for a, b in zip(c0, c1)]
            depth = z0 + (z1 - z0) * f
            # The fragment's `line_output` (perp, dist) and colour; the last drawn fragment wins.
            # Depth test LESS_EQUAL against the cleared depth buffer (window depth, 1.0 when empty): the nearest
            # line wins a pixel, and an equal depth goes to the later fragment.
            window_depth = (depth + 1.0) * 0.5
            current = records.get(py * width + px)
            if current is not None and window_depth > current["depth"]:
                continue
            records[py * width + px] = {"colour": colour, "dist": d, "dir": perp, "depth": window_depth}


def antialias(records, width, height, bg):
    """overlay_antialiasing.bsl.hh frag_main for every pixel that has a line at or next to it."""
    out = {}
    cands = set()
    for i in records:
        y, x = divmod(i, width)
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                nx, ny = x + ddx, y + ddy
                if 0 <= nx < width and 0 <= ny < height:
                    cands.add(ny * width + nx)
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in cands:
        y, x = divmod(i, width)
        centre = records.get(i)
        c_color = [0.0, 0.0, 0.0]
        c_alpha = 0.0
        c_depth = 1.0
        if centre is not None:
            k = cov(centre["dist"])
            c_color = [c * k for c in centre["colour"]]
            c_alpha = k
            c_depth = centre["depth"]
        neigh = []
        for ox, oy in offsets:
            n = records.get((y + oy) * width + (x + ox))
            if n is None:
                neigh.append(None)
                continue
            is_dir_h = abs(n["dir"][0]) > abs(n["dir"][1])
            is_ofs_h = ox != 0
            if is_dir_h != is_ofs_h:
                nd = 1e10
            else:
                nd = n["dist"] + (ox * -n["dir"][0] + oy * -n["dir"][1])
            neigh.append((n, cov(nd) if nd < 1e9 else 0.0))
        for (n, k) in [q for q in neigh if q is not None]:
            if k <= 0.0:
                continue
            nc = [c * k for c in n["colour"]]
            na = k
            if n["depth"] < c_depth:
                # alpha over: target = neighbour + target * (1 - neighbour.a)
                c_color = [a + b * (1 - na) for a, b in zip(nc, c_color)]
                c_alpha = na + c_alpha * (1 - na)
                c_depth = n["depth"]
            else:
                c_color = [a + b * (1 - c_alpha) for a, b in zip(c_color, nc)]
                c_alpha = c_alpha + na * (1 - c_alpha)
        # Dense-line fix: `original_center_has_alpha` is false for every line pixel (alpha 1 before AA).
        if centre is not None and c_alpha > 0.0:
            count = sum(1 for q in neigh if q is not None and q[0] is not None)
            blend = clamp01((0.25 * count) * 2 - 1)
            un = [c / c_alpha for c in c_color]
            c_color = [mix(a, b, blend) for a, b in zip(c_color, un)]
        if c_alpha == 0.0 and all(v == 0 for v in c_color):
            continue
        out[i] = [c + bg_c * (1 - c_alpha) for c, bg_c in zip(c_color, bg)]
    return out


if __name__ == "__main__":
    geom = json.load(open(sys.argv[1]))
    oracle = json.load(open(sys.argv[2]))
    png = sys.argv[3]
    variant = sys.argv[4] if len(sys.argv) > 4 else "first"
    theme = oracle["theme_view_3d"]
    records, bg = build_records(geom, oracle, theme["wire"][:3], oracle["theme_background"]["high_gradient"][:3], variant)
    width, height, pixels = read_png(png)
    result = antialias(records, width, height, bg)
    errs_edge, errs_int, worst = [], [], []
    for y in range(150, 550):
        for x in range(150, 650):
            i = y * width + x
            model = tuple(display(c) for c in result.get(i, bg))
            d = max(abs(a - b) for a, b in zip(model, pixels[i]))
            (errs_edge if (i in records or i in result and i in result and any(
                (j in records) for j in [i-1,i+1,i-width,i+width])) else errs_int).append(d)
            if d > 6:
                worst.append((x, y, model, pixels[i]))
    def stats(e):
        return (len(e), round(sum(e) / max(len(e), 1), 3), max(e) if e else 0, sum(1 for v in e if v > 6))
    print("variant", variant, "interior (n, mean, max, >6):", stats(errs_int), "edge (n, mean, max, >6):", stats(errs_edge))
    print("worst", worst[:8])
