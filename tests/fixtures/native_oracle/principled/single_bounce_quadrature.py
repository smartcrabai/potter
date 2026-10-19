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


GGX_GEN_SCHLICK_IOR_S = table("GGX_GEN_SCHLICK_IOR_S").reshape(16, 16, 16)  # [z][y][x]


def lut3(t, x, y, z):
    """lookup_table_read_3D: x = roughness (fastest), y = cos(N, V), z = sqrt(|(ior-1)/(ior+1)|)."""
    def axis(a, n):
        a = min(max(a, 0.0), 1.0) * (n - 1)
        i = min(int(math.floor(a)), n - 1)
        return i, min(i + 1, n - 1), a - i

    def plane(zi):
        xi0, xi1, fx = axis(x, 16)
        yi0, yi1, fy = axis(y, 16)
        def at(yi, xi):
            return t[zi, yi, xi]
        r0 = (1 - fx) * at(yi0, xi0) + fx * at(yi0, xi1)
        r1 = (1 - fx) * at(yi1, xi0) + fx * at(yi1, xi1)
        return r0 if fy == 0 else (1 - fy) * r0 + fy * r1

    zi0, zi1, fz = axis(z, 16)
    p0 = plane(zi0)
    if fz == 0:
        return p0
    return (1 - fz) * p0 + fz * plane(zi1)


def fresnel_dielectric(cos_i, eta):
    ci = abs(cos_i)
    t = eta * eta - (1 - ci * ci)
    if t <= 0:
        return 1.0
    ct = -math.sqrt(t) / eta
    rs = (ci + eta * ct) / (ci - eta * ct)
    rp = (ct + eta * ci) / (eta * ci - ct)
    return 0.5 * (rs * rs + rp * rp)


def dielectric_white(base=0.8, rough=0.24, eta=1.5):
    """Principled white dielectric, specular IOR level 0.5: a dielectric layer over a Lambert base.

    Mirrors crates/potter-core/src/render/principled_bsdf.rs: the diffuse weight is
    (1 - albedo * darkening) with albedo from ggx_gen_schlick_ior_s and darkening = gain * E.
    """
    f0 = ((eta - 1) / (eta + 1)) ** 2
    fss_scalar = (eta - 1) / (4.08567 + 1.00071 * eta)
    fss_remap = min(max((fss_scalar - f0) / (1 - f0), 0.0), 1.0)
    fss = f0 + (1 - f0) * fss_remap
    z = math.sqrt(abs((eta - 1) / (eta + 1)))
    a = max(rough, 0.02) ** 2
    a2 = a * a

    def fn(ld, wgt, g):
        hv = g["v"][None, :] + ld
        hv /= np.linalg.norm(hv, axis=1)[:, None]
        nh = np.clip(hv[:, 2], 0, None)
        vh = np.clip(hv @ g["v"], 0, None)
        nl = np.clip(ld[:, 2], 1e-8, None)
        nv = g["v"][2]
        e = lut_e(rough, nv)
        eavg = lut1(EAVG, rough)
        fms = fss * eavg / (1 - fss * (1 - eavg))
        gain = 1 + fms * (1 - e) / e
        dark = gain * e
        albedo = f0 + (1 - f0) * lut3(GGX_GEN_SCHLICK_IOR_S, rough, nv, z)
        diffuse_w = min(max(1 - albedo * dark, 0.0), 1.0)
        dd = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        lam = lambda c: 0.5 * (np.sqrt(1 + a2 * (1 / (c * c) - 1)) - 1)
        g2 = 1.0 / (1 + lam(nv) + lam(nl))
        real_f0 = f0
        remap = np.clip((np.array([fresnel_dielectric(x, eta) for x in vh]) - real_f0) / (1 - real_f0), 0, 1)
        fres = f0 + (1 - f0) * remap
        spec = gain * fres * (dd * g2 / (4 * nv * nl))
        return (spec + diffuse_w * base / math.pi) @ np.ones(1) * np.ones(3) * wgt.sum() if False else \
            np.array([((spec + diffuse_w * base / math.pi) * wgt).sum()] * 3)

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
    print("dielectric_white", patch_mean(dielectric_white(), g, grid).tolist())
