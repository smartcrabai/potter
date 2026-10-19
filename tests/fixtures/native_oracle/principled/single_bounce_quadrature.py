"""Quadrature of the Cycles v5.2.2 Principled lobes for the single-bounce probe.

Receiver at the origin, normal +Z, view 45 degrees off the normal, a disk emitter of radius R and
flux P (radiance P / (pi area)), the central 8x8 pixel patch of a 16 px orthographic camera of
width 0.2 units. The image's vertical axis maps to receiver x with a factor cos(45 deg), so one
pixel is pixel / cos in x and pixel in y. Energy compensation uses the Cycles GGX tables, read from
crates/potter-core/src/render/principled_tables_data.rs (generated from intern/cycles/scene/shader.tables).

Run: python3 single_bounce_quadrature.py DIAMETER GRID   (needs numpy; DIAMETER 1.2 is radius 0.6)
"""
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
DATA = ROOT / "crates/potter-core/src/render/principled_tables_data.rs"


def table(name):
    text = DATA.read_text()
    m = re.search(r"static " + name + r": \[f32; \d+\] = \[(.*?)\];", text, re.S)
    return np.array([float(t.replace("_", "")) for t in re.findall(r"[-0-9_.e]+", m.group(1))])


E2 = table("GGX_E").reshape(32, 32)   # E2[mu_index, rough_index]
EAVG = table("GGX_E_AVG")


def lut1(t, x):
    x = min(max(x, 0.0), 1.0) * (len(t) - 1)
    i = min(int(math.floor(x)), len(t) - 1)
    j = min(i + 1, len(t) - 1)
    f = x - i
    return t[i] if f == 0 else (1 - f) * t[i] + f * t[j]


def lut_e(rough, mu):
    mu = min(max(mu, 0.0), 1.0) * 31
    i = min(int(math.floor(mu)), 31)
    j = min(i + 1, 31)
    f = mu - i
    r0 = lut1(E2[i], rough)
    r1 = lut1(E2[j], rough)
    return r0 if f == 0 else (1 - f) * r0 + f * r1


def setup(radius, power=1000.0, dist=4.0, theta=math.radians(45.0)):
    cam = np.array([6.0 * math.sin(theta), 0.0, 6.0 * math.cos(theta)])
    centre = np.array([-dist * math.sin(theta), 0.0, dist * math.cos(theta)])
    nl = -centre / np.linalg.norm(centre)
    area = math.pi * radius * radius
    lrad = power / (math.pi * area)
    u = np.cross(nl, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(nl, u)
    v = cam / np.linalg.norm(cam)
    return dict(centre=centre, nl=nl, lrad=lrad, u=u, w=w, v=v, radius=radius, theta=theta)


def integral_at(fn, g, q, n):
    h = 2.0 / n
    gg = (np.arange(n) + 0.5) * h - 1.0
    a, b = np.meshgrid(gg, gg, indexing="ij")
    mask = a * a + b * b <= 1.0
    a, b = a[mask], b[mask]
    p = g["centre"][None, :] + g["radius"] * (a[:, None] * g["u"][None, :] + b[:, None] * g["w"][None, :])
    diff = p - np.array([q[0], q[1], 0.0])
    d = np.linalg.norm(diff, axis=1)
    ld = diff / d[:, None]
    cos_r = np.clip(ld[:, 2], 0, None)
    cos_l = np.clip((-ld) @ g["nl"], 0, None)
    wgt = g["radius"] ** 2 * h * h * (cos_l / d ** 2) * g["lrad"] * cos_r
    return fn(ld, wgt, g)


def metal(base, rough):
    a = max(rough, 0.02) ** 2
    a2 = a * a
    base = np.asarray(base, dtype=float)

    def fn(ld, wgt, g):
        hv = g["v"][None, :] + ld
        hv /= np.linalg.norm(hv, axis=1)[:, None]
        nh = np.clip(hv[:, 2], 0, None)
        vh = np.clip(hv @ g["v"], 0, None)
        nl = np.clip(ld[:, 2], 1e-8, None)
        nv = g["v"][2]
        dd = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        lam = lambda c: 0.5 * (np.sqrt(1 + a2 * (1 / (c * c) - 1)) - 1)
        g2 = 1.0 / (1 + lam(nv) + lam(nl))
        f = base[None, :] + (1 - base[None, :]) * ((1 - vh) ** 5)[:, None]
        e = lut_e(rough, nv)
        eavg = lut1(EAVG, rough)
        fss = base + (1 - base) / 21.0
        fms = fss * eavg / (1 - fss * (1 - eavg))
        m = 1 + fms * (1 - e) / e
        return ((dd * g2 / (4 * nv * nl))[:, None] * f * m[None, :] * wgt[:, None]).sum(axis=0)

    return fn


def patch_mean(fn, g, grid):
    pix = 0.2 / 16.0
    x_step = pix / math.cos(g["theta"])
    pts = [((i - 3.5) * x_step, (j - 3.5) * pix) for i in range(8) for j in range(8)]
    return np.mean([integral_at(fn, g, q, grid) for q in pts], axis=0)


if __name__ == "__main__":
    radius = float(sys.argv[1]) / 2.0
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    g = setup(radius)
    print("metal_r024", patch_mean(metal(np.array([0.055, 0.15, 0.24]), 0.24), g, grid).tolist())
