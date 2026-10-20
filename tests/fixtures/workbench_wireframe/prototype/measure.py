import json, math, sys
sys.path.insert(0, '/tmp/pr-wire/oracle')
import wire_model2 as m
from pngread import read_png
from build_lines_spec import SEGMENTS  # world endpoints

cam = json.load(open('lines.json'))
W, H = cam['region']
vp = [[sum(cam['window_matrix'][r][k] * cam['view_matrix'][k][c] for k in range(4)) for c in range(4)] for r in range(4)]
bg = m.srgb_to_linear(61/255)
bgd = m.display(bg)
w, h, px = read_png('lines.png')

def to_screen(v):
    x, y, z = m.project(vp, v)
    return ((x + 1) * 0.5 * W, (1 - (y + 1) * 0.5) * H, (z + 1) * 0.5)

records = {}
for (a, b) in SEGMENTS:
    p0, p1 = to_screen(a), to_screen(b)
    m.draw(records, W, H, p0, p1, [0.0] * 3, [0.0] * 3)
res = m.antialias(records, W, H, [bg] * 3)

def ink_model(seg):
    p0, p1 = to_screen(seg[0]), to_screen(seg[1])
    return band_ink(lambda i: res.get(i, [bg]*3), p0, p1, lambda i: tuple(m.display(c) for c in res.get(i, [bg]*3)))

def band_ink(_, p0, p1, value_of):
    (x0, y0, _), (x1, y1, _) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    total = 0
    for py in range(max(0, int(min(y0, y1)) - 5), min(H, int(max(y0, y1)) + 6)):
        for pxx in range(max(0, int(min(x0, x1)) - 5), min(W, int(max(x0, x1)) + 6)):
            rx, ry = pxx + 0.5 - x0, py + 0.5 - y0
            t = rx * ux + ry * uy
            d = -rx * uy + ry * ux
            if -1 <= t <= L + 1 and abs(d) <= 2.5:
                v = value_of(py * W + pxx)
                total += sum(bgd - c for c in v) / 1.0 if isinstance(v, tuple) else 0
    return total / L

def ink_capture(seg):
    p0, p1 = to_screen(seg[0]), to_screen(seg[1])
    (x0, y0, _), (x1, y1, _) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    total = 0
    for py in range(max(0, int(min(y0, y1)) - 5), min(H, int(max(y0, y1)) + 6)):
        for pxx in range(max(0, int(min(x0, x1)) - 5), min(W, int(max(x0, x1)) + 6)):
            rx, ry = pxx + 0.5 - x0, py + 0.5 - y0
            t = rx * ux + ry * uy
            d = -rx * uy + ry * ux
            if -1 <= t <= L + 1 and abs(d) <= 2.5:
                total += sum(bgd - c for c in px[py * W + pxx])
    return total / L

for seg, angle in zip(SEGMENTS, [0,15,30,45,60,75,90]):
    km = ink_model(seg); kc = ink_capture(seg)
    print(f"angle {angle:3d}  model ink/len {km:8.2f}  capture ink/len {kc:8.2f}  ratio {km/kc:6.3f}")
