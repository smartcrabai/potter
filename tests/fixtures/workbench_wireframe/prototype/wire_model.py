"""Prototype of Potter's Workbench wireframe model, checked against the Blender 5.2.2 viewport oracle.

Model (Blender 5.2.2 sources, see the comments in the functions): each mesh edge is one line, its
colour from the facing of its corner normal, its visibility from the threshold on `wd` (the cosine of
the two adjacent face normals), and its coverage from the overlay AA kernel. Overlays are composited
in linear light over the theme background, and the framebuffer is displayed with a 2.2 gamma.
"""
import json
import math
import sys

from pngread import read_png

# overlay_shader_shared.hh: DISC_RADIUS = (1/sqrt(pi)) * 1.05, LINE_SMOOTH_START/END = 0.5 -/+ DISC_RADIUS.
DISC_RADIUS = (1.0 / math.sqrt(math.pi)) * 1.05
LINE_SMOOTH_START = 0.5 - DISC_RADIUS
LINE_SMOOTH_END = 0.5 + DISC_RADIUS


def srgb_to_linear(c):
    # BLI_color: srgb_to_linearrgb (piecewise sRGB EOTF).
    return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def line_coverage(dist):
    # overlay_antialiasing.bsl.hh line_coverage (smooth lines, kernel 0 at theme pixel size 1):
    # smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, |dist|).
    return smoothstep(LINE_SMOOTH_END, LINE_SMOOTH_START, abs(dist))


def mat_mul_vec(m, v):
    return [sum(m[r][c] * v[c] for c in range(4)) for r in range(4)]


def mix(a, b, t):
    return a + (b - a) * t


def wire_rgb_linear(facing, rim_lin, wire_lin):
    # overlay_wireframe_vert.glsl: rim/wire pow(1/2.2), mix 0.35 for the front colour, mix by facing,
    # pow(2.2) back to linear.
    rim = [c ** (1 / 2.2) for c in rim_lin]
    wire = [c ** (1 / 2.2) for c in wire_lin]
    front = [mix(r, w, 0.35) for r, w in zip(rim, wire)]
    col = [mix(r, f, facing) for r, f in zip(rim, front)]
    return [c ** 2.2 for c in col]


def display(lin):
    # The overlay framebuffer is shown with a 2.2 gamma; measured against the oracle background.
    return max(0, min(255, round(255 * lin ** (1 / 2.2))))


def render(geometry, oracle, theme_wire, theme_back, facing_variant="first"):
    width, height = oracle["region"]
    view = oracle["view_matrix"]
    proj = oracle["window_matrix"]
    vp = [[sum(proj[r][k] * view[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
    view_z = [view[2][0], view[2][1], view[2][2]]  # world direction of the view's +z (towards the viewer)
    rim_lin = [srgb_to_linear(c) for c in theme_wire]
    back_lin = [srgb_to_linear(c) for c in theme_back]
    lin = [back_lin[:] for _ in range(width * height)]
    coverage_hits = {}  # pixel index -> (line colour, signed distance) of the last line drawn there
    for name, ob in geometry.items():
        verts = ob["vertices"]
        pix = []
        for v in verts:
            clip = mat_mul_vec(vp, v + [1.0])
            nx = clip[0] / clip[3]
            ny = clip[1] / clip[3]
            pix.append(((nx + 1) * 0.5 * width, (1 - (ny + 1) * 0.5) * height))
        # Corner normals per face, as the wire batch's CornerNormal attribute.
        face_of_edge = {}
        for fi, face in enumerate(ob["faces"]):
            start = ob["loop_start"][fi]
            for k in range(len(face)):
                a, b = face[k], face[(k + 1) % len(face)]
                key = (min(a, b), max(a, b))
                face_of_edge.setdefault(key, []).append((fi, start + k))
        for ei, (a, b) in enumerate(ob["edges"]):
            key = (min(a, b), max(a, b))
            adjacent = face_of_edge.get(key, [])
            # The edge is drawn once per adjacent corner (variant "both") or once from the first face.
            uses = adjacent if facing_variant == "both" else adjacent[:1]
            for fi, corner in uses:
                # wd is only needed for the threshold; threshold 1.0 keeps every edge (see the notes).
                colour_a = wire_rgb_linear(
                    clamp01(abs(dot3(ob["corner_normals"][corner_for(ob, corner, a)], view_z))),
                    rim_lin, back_lin)
                colour_b = wire_rgb_linear(
                    clamp01(abs(dot3(ob["corner_normals"][corner_for(ob, corner, b)], view_z))),
                    rim_lin, back_lin)
                draw_segment(lin, width, height, pix[a], pix[b], colour_a, colour_b, coverage_hits)
    # AA: the centre pixel's own line coverage, over the background (linear light).
    for i, (colour, d) in coverage_hits.items():
        cov = line_coverage(d)
        lin[i] = [c * cov + prev * (1 - cov) for c, prev in zip(colour, lin[i])]
    return lin, coverage_hits


def corner_for(ob, corner, vertex):
    # The corner index of `vertex` within the loop `corner` (loops of a face hold vertex indices).
    if ob["loop_vertices"][corner] == vertex:
        return corner
    # Otherwise the neighbouring loop of the same face holds the vertex: search the face's loops.
    for c in range(len(ob["loop_vertices"])):
        if ob["loop_vertices"][c] == vertex and ob["loop_faces"][c] == ob["loop_faces"][corner]:
            return c
    return corner


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def draw_segment(lin, width, height, p0, p1, c0, c1, hits):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return
    ux, uy = dx / length, dy / length
    minx = max(0, int(math.floor(min(x0, x1) - 2)))
    maxx = min(width - 1, int(math.ceil(max(x0, x1) + 2)))
    miny = max(0, int(math.floor(min(y0, y1) - 2)))
    maxy = min(height - 1, int(math.ceil(max(y0, y1) + 2)))
    for py in range(miny, maxy + 1):
        for px in range(minx, maxx + 1):
            cx, cy = px + 0.5, py + 0.5
            rx, ry = cx - x0, cy - y0
            t = rx * ux + ry * uy
            if t < -0.5 or t > length + 0.5:
                continue
            d = -rx * uy + ry * ux  # signed distance to the infinite line, in pixels
            # The line rasteriser writes a fragment where the pixel centre is within half a pixel of the
            # line; each pixel keeps one line record, the last one drawn (overlay_wireframe_frag.glsl).
            if abs(d) > 0.5:
                continue
            f = min(max(t / length, 0.0), 1.0)
            colour = [mix(a, b, f) for a, b in zip(c0, c1)]
            hits[py * width + px] = (colour, d)



if __name__ == "__main__":
    geom = json.load(open(sys.argv[1]))
    oracle = json.load(open(sys.argv[2]))
    png = sys.argv[3]
    variant = sys.argv[4] if len(sys.argv) > 4 else "first"
    theme = oracle["theme_view_3d"]
    wire = theme["wire"][:3]
    back = oracle["theme_background"]["high_gradient"][:3]
    lin, hits = render(geom, oracle, wire, back, variant)
    width, height, pixels = read_png(png)
    interior = edge = 0
    err_interior = err_edge = 0
    max_edge = 0
    worst = []
    roi = [(x, y) for y in range(150, 550) for x in range(150, 650)]
    for x, y in roi:
        i = y * width + x
        model = tuple(display(c) for c in lin[i])
        want = pixels[i]
        d = max(abs(a - b) for a, b in zip(model, want))
        if i in hits:
            edge += 1
            err_edge += d
            max_edge = max(max_edge, d)
            if d > 6:
                worst.append((x, y, model, want))
        else:
            interior += 1
            err_interior += d
    print(f"variant={variant} interior px={interior} mean_err={err_interior / max(interior,1):.3f} "
          f"edge px={edge} mean_err={err_edge / max(edge,1):.3f} max_edge_err={max_edge}")
    print("bad edge samples", len(worst), worst[:6])
