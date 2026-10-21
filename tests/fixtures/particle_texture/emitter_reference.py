"""Float32 reference of Blender 5.2.2 particle emitter placement, checked against Blender in headless mode.

It reproduces `distribute_particles` with `distribute_from_faces_exec` and `distribute_from_volume_exec`
(FACE, VOLUME), `distribute_grid` (GRID, with the watertight ray test), `psys_frand`, and the jitter
tables, all with float32 steps. It needs Blender's numpy, so run it inside Blender with a script that
imports this file. The Blender values it is checked against are `particles[p].location` in
`blender_expected.json` from `generate.py`, whose cases give the meshes and settings.
"""
import math
import os

import numpy as np

F = np.float32
FLT_EPSILON = F(1.1920928955078125e-07)
PSYS_RND_DIST_SKIP = 3


class Rng:
    """BLI RandomNumberGenerator (rand.cc / BLI_rand.hh)."""
    HASH = None

    def __init__(self, state):
        self.x = state & 0xFFFFFFFFFFFF

    @staticmethod
    def seed(seed):
        return Rng(((seed & 0xFFFFFFFF) << 16) | 0x330E)

    @staticmethod
    def seed_random(seed):
        h = HASH_UCHAR_512
        r = Rng.seed((seed + h[seed & 255]) & 0xFFFFFFFF)
        v = r.get_uint32()
        r = Rng.seed((v + h[v & 255]) & 0xFFFFFFFF)
        v = r.get_uint32()
        return Rng.seed((v + h[v & 255]) & 0xFFFFFFFF)

    def step(self):
        self.x = (0x5DEECE66D * self.x + 0xB) & 0x0000FFFFFFFFFFFF

    def get_uint32(self):
        self.step()
        return (self.x >> 17) & 0xFFFFFFFF

    def get_int32(self):
        self.step()
        return (self.x >> 17) & 0xFFFFFFFF  # always < 2^31

    def get_float(self):
        return F(F(self.get_int32()) / F(2147483648.0))

    def get_double(self):
        return float(self.get_int32()) / 2147483648.0

    def skip(self, n):
        for _ in range(n):
            self.step()


def load_hash(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "crates", "potter-core", "src", "blender_rng.rs")):
    text = open(path).read()
    body = text.split("HASH_UCHAR_512: [u8; 512] = [", 1)[1].split("];", 1)[0]
    values = []
    for token in body.replace("\n", " ").split(","):
        token = token.strip()
        if token:
            values.append(int(token, 16) if token.lower().startswith("0x") else int(token))
    assert len(values) == 512, len(values)
    return values


HASH_UCHAR_512 = load_hash()


def radical_inverse(n):
    u = 0.0
    p = 0.5
    while n:
        if n & 1:
            u += p
        p *= 0.5
        n >>= 1
    return u




def add_newell(n, prev, curr):
    n[0] = F(n[0] + F(F(prev[1] - curr[1]) * F(prev[2] + curr[2])))
    n[1] = F(n[1] + F(F(prev[2] - curr[2]) * F(prev[0] + curr[0])))
    n[2] = F(n[2] + F(F(prev[0] - curr[0]) * F(prev[1] + curr[1])))


def len3(v):
    return F(np.sqrt(F(F(v[0] * v[0] + v[1] * v[1]) + F(v[2] * v[2]))))


def normalize3(v):
    d = F(F(v[0] * v[0] + v[1] * v[1]) + F(v[2] * v[2]))
    if d > F(1.0e-35):
        d = F(np.sqrt(d))
        m = F(F(1.0) / d)
        return [F(v[0] * m), F(v[1] * m), F(v[2] * m)], d
    return [F(0.0), F(0.0), F(0.0)], F(0.0)


def cross(a, b):
    return [F(F(a[1] * b[2]) - F(a[2] * b[1])),
            F(F(a[2] * b[0]) - F(a[0] * b[2])),
            F(F(a[0] * b[1]) - F(a[1] * b[0]))]


def sub(a, b):
    return [F(a[0] - b[0]), F(a[1] - b[1]), F(a[2] - b[2])]


def dot(a, b):
    return F(F(F(a[0] * b[0]) + F(a[1] * b[1])) + F(a[2] * b[2]))


def normal_tri(v1, v2, v3):
    n1 = sub(v1, v2)
    n2 = sub(v2, v3)
    n = cross(n1, n2)
    return normalize3(n)[0]


def normal_quad(v1, v2, v3, v4):
    n1 = sub(v1, v3)
    n2 = sub(v2, v4)
    n = cross(n1, n2)
    return normalize3(n)[0]


def area_poly(verts):
    n = [F(0.0), F(0.0), F(0.0)]
    count = len(verts)
    prev = verts[count - 1]
    for i in range(count):
        curr = verts[i]
        add_newell(n, prev, curr)
        prev = curr
    return F(len3(n) * F(0.5))


def area_tri(v1, v2, v3):
    n = cross(sub(v1, v2), sub(v2, v3))
    return F(len3(n) * F(0.5))


def area_quad(v1, v2, v3, v4):
    return area_poly([v1, v2, v3, v4])


def interp_v3(vs, w):
    out = []
    for c in range(3):
        acc = F(0.0)
        for v, wi in zip(vs, w):
            acc = F(acc + F(v[c] * wi))
        out.append(acc)
    return out


def isect_ray_tri(origin, direction, v0, v1, v2):
    eps = F(0.00000001)
    e1 = sub(v1, v0)
    e2 = sub(v2, v0)
    p = cross(direction, e2)
    a = dot(e1, p)
    if a > -eps and a < eps:
        return None
    f = F(F(1.0) / a)
    s = sub(origin, v0)
    u = F(f * dot(s, p))
    if u < F(0.0) or u > F(1.0):
        return None
    q = cross(s, e1)
    v = F(f * dot(direction, q))
    if v < F(0.0) or F(u + v) > F(1.0):
        return None
    lam = F(f * dot(e2, q))
    if lam < F(0.0):
        return None
    return lam


def dist_sq_to_segment(p, l1, l2):
    u = sub(l2, l1)
    h = sub(p, l1)
    if u[0] == 0 and u[1] == 0 and u[2] == 0:
        lam = F(0.0)
        cp = list(l1)
    else:
        lam = F(dot(u, h) / dot(u, u))
        cp = [F(l1[c] + F(u[c] * lam)) for c in range(3)]
    if lam <= F(0.0):
        closest = list(l1)
    elif lam >= F(1.0):
        closest = list(l2)
    else:
        closest = cp
    d = sub(closest, p)
    return dot(d, d)


def mean_value_half_tan(d_curr, d_next):
    cr = cross(d_curr[0], d_next[0])
    area = len3(cr)
    if area != F(0.0):
        dt = dot(d_curr[0], d_next[0])
        ln = F(d_curr[1] * d_next[1])
        result = F(F(ln - dt) / area)
        if np.isfinite(result):
            return result
    return F(0.0)


def interp_weights_poly(n, v, co):
    max_value = F(0.0)
    for i in range(n):
        for c in range(3):
            max_value = max(max_value, F(abs(F(v[i][c] - co[c]))))
    eps = F(F(16.0) * FLT_EPSILON * max_value)
    eps_sq = F(eps * eps)
    w = [F(0.0)] * n
    totweight = F(0.0)
    ix_flag = 0  # 1 point, 2 segment
    i_curr = n - 1
    i_next = 0
    v_curr = v[i_curr]
    v_next = v[i_next]

    def dir_len(a, b):
        d = sub(a, b)
        return (d, len3(d))

    d_curr = dir_len(v[n - 2], co)
    d_next = dir_len(v_curr, co)
    ht_prev = mean_value_half_tan(d_curr, d_next)
    while i_next < n:
        if d_next[1] < eps:
            ix_flag = 1
            break
        if dist_sq_to_segment(co, v_curr, v_next) < eps_sq:
            ix_flag = 2
            break
        d_curr = d_next
        d_next = dir_len(v_next, co)
        ht = mean_value_half_tan(d_curr, d_next)
        w[i_curr] = F(F(ht_prev + ht) / d_curr[1])
        totweight = F(totweight + w[i_curr])
        i_curr = i_next
        i_next += 1
        v_curr = v_next
        if i_next < n:
            v_next = v[i_next]
        ht_prev = ht
    if ix_flag:
        w = [F(0.0)] * n
        if ix_flag == 1:
            w[i_curr] = F(1.0)
        else:
            u = sub(v_next, v_curr)
            den = dot(u, u)
            fac = F(dot(u, sub(co, v_curr)) / den) if den > F(0.0) else F(0.0)
            fac = min(max(fac, F(0.0)), F(1.0))
            w[i_curr] = F(F(1.0) - fac)
            w[i_next] = fac
    else:
        if totweight != F(0.0):
            w = [F(x / totweight) for x in w]
    return w


def uv_to_w(u, v, quad):
    u = F(u)
    v = F(v)
    if not quad:
        if F(u + v) > F(1.0):
            v = F(F(1.0) - v)
        else:
            u = F(F(1.0) - u)
    vert = [[F(0), F(0), F(0)], [F(1), F(0), F(0)], [F(1), F(1), F(0)]]
    if quad:
        vert.append([F(0), F(1), F(0)])
    co = [u, v, F(0)]
    if quad:
        return interp_weights_poly(4, vert, co)
    w = interp_weights_poly(3, vert, co)
    return w + [F(0)]


def binary_search(s, n, value):
    low = 0
    high = n - 1
    if high == low:
        return low
    if s[low] >= value:
        return low
    if s[high - 1] < value:
        return high
    while low < high:
        mid = (low + high) // 2
        if s[mid] >= value and s[mid - 1] < value:
            return mid
        if s[mid] > value:
            high = mid - 1
        else:
            low = mid + 1
    return low


def init_mv_jit(jit, num, seed2, amount):
    if num == 0:
        return
    rad1 = F(F(1.0) / F(np.sqrt(F(num))))
    rad2 = F(F(1.0) / F(num))
    rad3 = F(F(np.sqrt(F(num))) / F(num))
    rng = Rng.seed((31415926 + num + seed2) & 0xFFFFFFFF)
    x = F(0.0)
    num2 = 2 * num
    for i in range(0, num2, 2):
        jit[i] = F(x + F(F(F(amount) * rad1) * F(F(0.5) - rng.get_float())))
        jit[i + 1] = F(F(F(i) / F(F(2.0) * F(num))) + F(F(F(amount) * rad1) * F(F(0.5) - rng.get_float())))
        jit[i] = F(jit[i] - F(np.floor(jit[i])))
        jit[i + 1] = F(jit[i + 1] - F(np.floor(jit[i + 1])))
        x = F(x + rad3)
        x = F(x - F(np.floor(x)))
    jit2 = [[F(0.0), F(0.0)] for _ in range(num)]
    pairs = [[jit[2 * k], jit[2 * k + 1]] for k in range(num)]
    for _ in range(4):
        jitterate1(pairs, jit2, num, rad1)
        jitterate1(pairs, jit2, num, rad1)
        jitterate2(pairs, jit2, num, rad2)
    for k in range(num):
        jit[2 * k] = pairs[k][0]
        jit[2 * k + 1] = pairs[k][1]


def jitterate1(jit1, jit2, num, radius1):
    r = radius1
    for i in range(num - 1, -1, -1):
        dvecx = F(0.0)
        dvecy = F(0.0)
        x = jit1[i][0]
        y = jit1[i][1]
        for j in range(num - 1, -1, -1):
            if i != j:
                vecx = F(F(jit1[j][0] - x) - F(1.0))
                vecy = F(F(jit1[j][1] - y) - F(1.0))
                for _k in range(3):
                    def check(vx, vy, dx, dy):
                        if abs(vx) < r and abs(vy) < r:
                            ln = F(np.sqrt(F(F(vx * vx) + F(vy * vy))))
                            if ln > F(0.0) and ln < r:
                                ln = F(ln / r)
                                dx = F(dx + F(vx / ln))
                                dy = F(dy + F(vy / ln))
                        return dx, dy
                    dvecx, dvecy = check(vecx, vecy, dvecx, dvecy)
                    vecx = F(vecx + F(1.0))
                    dvecx, dvecy = check(vecx, vecy, dvecx, dvecy)
                    vecx = F(vecx + F(1.0))
                    dvecx, dvecy = check(vecx, vecy, dvecx, dvecy)
                    vecx = F(vecx - F(2.0))
                    vecy = F(vecy + F(1.0))
        x = F(x - F(dvecx / F(18.0)))
        y = F(y - F(dvecy / F(18.0)))
        x = F(x - F(np.floor(x)))
        y = F(y - F(np.floor(y)))
        jit2[i] = [x, y]
    for k in range(num):
        jit1[k] = list(jit2[k])


def jitterate2(jit1, jit2, num, radius2):
    r = radius2
    for i in range(num - 1, -1, -1):
        dvecx = F(0.0)
        dvecy = F(0.0)
        x = jit1[i][0]
        y = jit1[i][1]
        for j in range(num - 1, -1, -1):
            if i != j:
                vecx = F(F(jit1[j][0] - x) - F(1.0))
                vecy = F(F(jit1[j][1] - y) - F(1.0))
                if abs(vecx) < r:
                    dvecx = F(dvecx + F(vecx * r))
                vecx = F(vecx + F(1.0))
                if abs(vecx) < r:
                    dvecx = F(dvecx + F(vecx * r))
                vecx = F(vecx + F(1.0))
                if abs(vecx) < r:
                    dvecx = F(dvecx + F(vecx * r))
                if abs(vecy) < r:
                    dvecy = F(dvecy + F(vecy * r))
                vecy = F(vecy + F(1.0))
                if abs(vecy) < r:
                    dvecy = F(dvecy + F(vecy * r))
                vecy = F(vecy + F(1.0))
                if abs(vecy) < r:
                    dvecy = F(dvecy + F(vecy * r))
        x = F(x - F(dvecx / F(2.0)))
        y = F(y - F(dvecy / F(2.0)))
        x = F(x - F(np.floor(x)))
        y = F(y - F(np.floor(y)))
        jit2[i] = [x, y]
    for k in range(num):
        jit1[k] = list(jit2[k])


def hammersley_create(out, n, seed, amount):
    rng = Rng.seed((31415926 + n + seed) & 0xFFFFFFFF)
    ofs0 = rng.get_double() + float(amount)
    ofs1 = rng.get_double() + float(amount)
    for k in range(n):
        t = radical_inverse(k)
        out[2 * k] = F(math.fmod(float(k) / float(n) + ofs0, 1.0))
        out[2 * k + 1] = F(math.fmod(t + ofs1, 1.0))


def array_randomize(pairs, num, seed):
    if num <= 1:
        return
    rng = Rng.seed(seed)
    i = num
    while i > 0:
        i -= 1
        j = rng.get_uint32() % num
        if i != j:
            pairs[i], pairs[j] = pairs[j], pairs[i]


def face_tables(polys, positions, smooth_flags, normals, count, seed, from_volume, distr,
                even, use_random, jitfac, userjit):
    """Returns per-particle (element, fuv(4), foffset) following distribute_particles + exec."""
    # Tessfaces: (vertex indices (4 entries, v4=-1 for tri), smooth flag, source poly)
    tess = []
    for pi, poly in enumerate(polys):
        if len(poly) == 3:
            tess.append(([poly[0], poly[1], poly[2], -1], smooth_flags[pi]))
        elif len(poly) == 4:
            tess.append(([poly[0], poly[1], poly[2], poly[3]], smooth_flags[pi]))
        else:
            raise ValueError("n-gons not in this reference")
    totelem = len(tess)
    totpart = count
    P = lambda vi: positions[vi]
    weight = [F(0.0)] * totelem
    if even:
        cur_list = []
        totarea = F(0.0)
        for t, _s in tess:
            v = [P(x) for x in t if x >= 0]
            if t[3] >= 0:
                cur = area_quad(P(t[0]), P(t[1]), P(t[2]), P(t[3]))
            else:
                cur = area_tri(P(t[0]), P(t[1]), P(t[2]))
            cur_list.append(cur)
            totarea = F(totarea + cur)
        for i in range(totelem):
            weight[i] = F(cur_list[i] / totarea)
    else:
        mn = F(F(1.0) / F(min(totelem, totpart)))
        weight = [mn] * totelem
    totweight = F(0.0)
    totmapped = 0
    for w in weight:
        if w > F(0.0):
            totmapped += 1
            totweight = F(totweight + w)
    inv = F(F(1.0) / totweight)
    element_sum = []
    element_map = []
    i = 0
    while i < totelem and weight[i] == F(0.0):
        i += 1
    element_sum.append(F(weight[i] * inv))
    element_map.append(i)
    i += 1
    for k in range(i, totelem):
        if weight[k] > F(0.0):
            nxt = F(element_sum[-1] + F(weight[k] * inv))
            if nxt > element_sum[-1]:
                element_sum.append(nxt)
                element_map.append(k)
    totmapped = len(element_sum)
    particle_element = [0] * totpart
    jitter_offset = [F(0.0)] * totelem
    if use_random:
        rng = Rng.seed_random((31415926 + seed) & 0xFFFFFFFF)
        for p in range(totpart):
            pos = F(F(rng.get_float()) * element_sum[totmapped - 1])
            eidx = binary_search(element_sum, totmapped, pos)
            particle_element[p] = element_map[eidx]
            jitter_offset[particle_element[p]] = pos
    else:
        step = 0.5 if totpart < 2 else 1.0 / float(totpart)
        pos = 0.5 / float(totmapped) if totpart < totmapped else step * 0.5
        i = 0
        for p in range(totpart):
            while i < totmapped - 1 and pos > float(element_sum[i]):
                i += 1
            particle_element[p] = element_map[i]
            jitter_offset[particle_element[p]] = F(pos)
            pos += step

    jitlevel = 1
    jit = None
    if distr == "JIT":
        jitlevel = userjit
        if jitlevel == 0:
            jitlevel = totpart // totelem
            if even:
                jitlevel *= 2
            jitlevel = max(jitlevel, 3)
        jit = [F(0.0)] * (2 + 2 * jitlevel)
        if jitlevel < 25:
            init_mv_jit(jit, jitlevel, seed, F(jitfac))
        else:
            hammersley_create(jit, jitlevel + 1, seed, F(jitfac))
        pairs = [[jit[2 * k], jit[2 * k + 1]] for k in range(jitlevel)]
        array_randomize(pairs, jitlevel, seed)
        for k in range(jitlevel):
            jit[2 * k] = pairs[k][0]
            jit[2 * k + 1] = pairs[k][1]

    rng = Rng.seed((31415926 + seed) & 0xFFFFFFFF)
    out = []
    for p in range(totpart):
        el = particle_element[p]
        t, _s = tess[el]
        quad = t[3] >= 0
        rskip = PSYS_RND_DIST_SKIP
        if distr == "JIT":
            if jitlevel == 1:
                fuv = uv_to_w(0.5, 0.5, quad) if quad else uv_to_w(1.0 / 3.0, 1.0 / 3.0, quad)
            else:
                offset = F(math.fmod(F(jitter_offset[el] + F(p)), F(jitlevel)))
                idx = int(offset)
                fuv = uv_to_w(jit[2 * idx], jit[2 * idx + 1], quad)
        else:
            ru = rng.get_float()
            rv = rng.get_float()
            rskip -= 2
            fuv = uv_to_w(ru, rv, quad)
        foffset = F(0.0)
        if from_volume:
            vs = [P(x) for x in t if x >= 0]
            # psys_interpolate_face
            if quad:
                co = interp_v3([P(t[0]), P(t[1]), P(t[2]), P(t[3])], fuv[:4])
                if _s:
                    nor = interp_v3([normals[t[0]], normals[t[1]], normals[t[2]], normals[t[3]]], fuv[:4])
                else:
                    nor = normal_quad(P(t[0]), P(t[1]), P(t[2]), P(t[3]))
            else:
                co = interp_v3([P(t[0]), P(t[1]), P(t[2])], fuv[:3])
                if _s:
                    nor = interp_v3([normals[t[0]], normals[t[1]], normals[t[2]]], fuv[:3])
                else:
                    nor = normal_tri(P(t[0]), P(t[1]), P(t[2]))
            nhat, _ = normalize3(nor)
            nor = [F(-c) for c in nhat]
            min_d = F(3.4028235e38)
            intersect = False
            for i2, (tt, _s2) in enumerate(tess):
                if i2 == el:
                    continue
                lam = isect_ray_tri(co, nor, P(tt[1]), P(tt[2]), P(tt[0]))
                if lam is not None and lam < min_d:
                    min_d = lam
                    foffset = F(lam * F(0.5))
                    intersect = True
                if tt[3] >= 0:
                    lam = isect_ray_tri(co, nor, P(tt[3]), P(tt[0]), P(tt[2]))
                    if lam is not None and lam < min_d:
                        min_d = lam
                        foffset = F(lam * F(0.5))
                        intersect = True
            if not intersect:
                foffset = F(0.0)
            else:
                if distr == "JIT":
                    foffset = F(foffset * jit[p % (2 * jitlevel)])
                else:
                    foffset = F(foffset * rng.get_float())
                    rskip -= 1
            rng.skip(rskip)
            out.append((el, fuv, foffset, co, nhat))
        else:
            rng.skip(rskip)
            out.append((el, fuv, foffset, None, None))
    return out, tess


def birth_position(out_entry, tess, positions, normals):
    el, fuv, foffset, co, nhat = out_entry
    t, smooth = tess[el]
    quad = t[3] >= 0
    if co is None:
        vs = [positions[x] for x in t[:4 if quad else 3]]
        return interp_v3(vs, fuv[:4 if quad else 3])
    # psys_particle_on_dm (VOLUME): vec + normalize(nor) * (-foffset); nhat is that normalized normal.
    return [F(co[c] + F(nhat[c] * F(-foffset))) for c in range(3)]


# ---- PART_DISTR_GRID (distribute_grid, particle_distribute.cc) and psys_frand (BKE_particle.h) ----

_PSYS = None


def psys_table():
    global _PSYS
    if _PSYS is None:
        rng = Rng.seed_random(5831)
        base, offset, mult = [], [], []
        for _ in range(1024):
            base.append(rng.get_float())
            offset.append(rng.get_uint32())
            mult.append(rng.get_uint32())
        _PSYS = (base, offset, mult)
    return _PSYS


def psys_frand(system_seed, seed):
    base, offset, mult = psys_table()
    slot = system_seed % 1024
    idx = (offset[slot] + seed * mult[slot]) & 0xFFFFFFFF
    return base[idx % 1024]


def axis_dominant(v):
    x, y, z = abs(v[0]), abs(v[1]), abs(v[2])
    if x > y:
        return 0 if x > z else 2
    return 1 if y > z else 2


def bits(f):
    import struct
    return struct.unpack('<I', struct.pack('<f', float(f)))[0]


def from_bits(b):
    import struct
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def watertight_precalc(d):
    kz = axis_dominant(d)
    kx = kz + 1 if kz != 2 else 0
    ky = kx + 1 if kx != 2 else 0
    if d[kz] < F(0.0):
        kx, ky = ky, kx
    inv = F(F(1.0) / d[kz])
    return (kx, ky, kz, F(d[kx] * inv), F(d[ky] * inv), inv)


def watertight(origin, pre, v0, v1, v2):
    kx, ky, kz, sx, sy, sz = pre
    a = [F(v0[c] - origin[c]) for c in range(3)]
    b = [F(v1[c] - origin[c]) for c in range(3)]
    c = [F(v2[c] - origin[c]) for c in range(3)]
    ax = F(a[kx] - F(sx * a[kz])); ay = F(a[ky] - F(sy * a[kz]))
    bx = F(b[kx] - F(sx * b[kz])); by = F(b[ky] - F(sy * b[kz]))
    cx = F(c[kx] - F(sx * c[kz])); cy = F(c[ky] - F(sy * c[kz]))
    u = F(F(cx * by) - F(cy * bx))
    v = F(F(ax * cy) - F(ay * cx))
    w = F(F(bx * ay) - F(by * ax))
    if (u < F(0.0) or v < F(0.0) or w < F(0.0)) and (u > F(0.0) or v > F(0.0) or w > F(0.0)):
        return None
    det = F(F(u + v) + w)
    if det == F(0.0) or not np.isfinite(det):
        return None
    sign_det = bits(det) & 0x80000000
    t = F(F(F(F(u * a[kz]) + F(v * b[kz])) + F(w * c[kz])) * sz)
    sign_t = from_bits(bits(t) ^ sign_det)
    if sign_t < F(0.0):
        return None
    inv_det = F(F(1.0) / det)
    return F(t * inv_det)


def cint(x):
    """C int(ceil(x)) with saturation for the non-finite cases."""
    x = float(x)
    if math.isnan(x):
        return 0
    if math.isinf(x):
        return 2**31 - 1 if x > 0 else -2**31
    return int(math.ceil(x))


def distribute_grid(positions, tess, res, grid_rand, hexagonal, invert, volume, seed):
    """Returns per grid index p: (unexist flag, fuv) with Blender's exact float32 steps."""
    totvert = len(positions)
    mn = [F(3.4028235e38)] * 3
    mx = [F(-3.4028235e38)] * 3
    for i in range(1, totvert):
        for c in range(3):
            mn[c] = F(min(mn[c], positions[i][c]))
            mx[c] = F(max(mx[c], positions[i][c]))
    delta = [F(mx[c] - mn[c]) for c in range(3)]
    axis = axis_dominant(delta)
    d = F(delta[axis] / F(res))
    size = [0, 0, 0]
    size[axis] = res
    size[(axis + 1) % 3] = cint(F(delta[(axis + 1) % 3] / d))
    size[(axis + 2) % 3] = cint(F(delta[(axis + 2) % 3] / d))
    size[(axis + 1) % 3] = min(size[(axis + 1) % 3], res)
    size[(axis + 2) % 3] = min(size[(axis + 2) % 3], res)
    size = [max(s, 1) for s in size]
    half = lambda c: F(d / F(2.0)) if d < delta[c] else F(delta[c] / F(2.0))
    for c in range(3):
        mn[c] = F(mn[c] + half(c))
    N = res * res * res
    fuv = [None] * N
    unexist = [True] * N
    hair = [0] * N
    for i in range(res):
        for j in range(res):
            for k in range(res):
                p = i * res * res + j * res + k
                fuv[p] = [F(mn[0] + F(F(i) * d)), F(mn[1] + F(F(j) * d)), F(mn[2] + F(F(k) * d))]
    amax = 1 if volume else 3
    for a in range(amax):
        if a == 0:
            a0mul, a1mul, a2mul = res * res, res, 1
        elif a == 1:
            a0mul, a1mul, a2mul = res, 1, res * res
        else:
            a0mul, a1mul, a2mul = 1, res * res, res
        for a1 in range(size[(a + 1) % 3]):
            for a2 in range(size[(a + 2) % 3]):
                base = a1 * a1mul + a2 * a2mul
                co1 = list(fuv[base])
                co1[a] = F(co1[a] - half(a))
                co2 = list(co1)
                co2[a] = F(co2[a] + F(delta[a] + F(F(0.001) * d)))
                co1[a] = F(co1[a] - F(F(0.001) * d))
                ray = sub(co2, co1)
                pre = watertight_precalc(ray)
                for tt in tess:
                    v1, v2, v3 = positions[tt[0]], positions[tt[1]], positions[tt[2]]
                    lam1 = watertight(co1, pre, v1, v2, v3)
                    pa1 = pa2 = None
                    if lam1 is not None:
                        k1 = int(F(lam1 * F(size[a])))
                        if k1 < size[a]:
                            pa1 = base + k1 * a0mul
                    if tt[3] >= 0 and (lam1 is None or volume):
                        v4 = positions[tt[3]]
                        lam2 = watertight(co1, pre, v1, v3, v4)
                        if lam2 is not None:
                            k2 = int(F(lam2 * F(size[a])))
                            if k2 < size[a]:
                                pa2 = base + k2 * a0mul
                    if pa1 is not None:
                        if volume:
                            hair[pa1] += 1
                        else:
                            unexist[pa1] = False
                    if pa2 is not None and pa2 != pa1:
                        if volume:
                            hair[pa2] += 1
                        else:
                            unexist[pa2] = False
                if volume:
                    inn = hair[base] % 2
                    if inn:
                        hair[base] += 1
                    for i in range(size[0]):
                        idx = base + i * a0mul
                        if inn or hair[idx] % 2:
                            unexist[idx] = False
                        inn = (inn + hair[idx]) % 2
    if hexagonal:
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    p = i * res * res + j * res + k
                    if j % 2:
                        fuv[p][0] = F(fuv[p][0] + F(d / F(2.0)))
                    if k % 2:
                        fuv[p][0] = F(fuv[p][0] + F(d / F(2.0)))
                        fuv[p][1] = F(fuv[p][1] + F(d / F(2.0)))
    if invert:
        for i in range(size[0]):
            for j in range(size[1]):
                base = res * (i * res + j)
                for k in range(size[2]):
                    unexist[base + k] = not unexist[base + k]
    if grid_rand > F(0.0):
        rfac = F(d * grid_rand)
        for p in range(N):
            if unexist[p]:
                continue
            fuv[p][0] = F(fuv[p][0] + F(rfac * F(F(psys_frand(seed, p + 31)) - F(0.5))))
            fuv[p][1] = F(fuv[p][1] + F(rfac * F(F(psys_frand(seed, p + 32)) - F(0.5))))
            fuv[p][2] = F(fuv[p][2] + F(rfac * F(F(psys_frand(seed, p + 33)) - F(0.5))))
    return unexist, fuv


def sub(a, b):
    return [F(a[0] - b[0]), F(a[1] - b[1]), F(a[2] - b[2])]
