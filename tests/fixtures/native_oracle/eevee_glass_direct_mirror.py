"""Python mirror of direct light through glass, as EEVEE 5.2.2 evaluates it, checked against `eevee_glass_direct.py`.

The mirror is the specification of Potter's realtime direct glass (`render/eevee_ltc.rs` and `render/eevee_transmission.rs`
for the light and the closures, `render/scene.rs` for the shadows). It evaluates each interior pixel of the spheres from
the oracle's JSON and reports the per-channel deviation from EEVEE's pixel.

Sources (Blender v5.2.2):
- `draw/engines/eevee/shaders/eevee_ltc_lib.bsl.hh`: `evaluate_quad`, `evaluate_disk`, `solve_cubic`,
  `edge_integral_vec`, `tangent_basis`, `diffuse_sphere_integral`.
- `draw/engines/eevee/shaders/eevee_ltc_lut_lib.bsl.hh`: the matrix LUT coordinates `(roughness, sqrt(1 - cos))`, `unpack`.
- `draw/engines/eevee/shaders/eevee_light_lib.bsl.hh`: `light_ltc`, `light_shape_corners`, `light_attenuation_facing`,
  `light_attenuation_common`, `light_attenuation_surface`, `light_influence_attenuation`, `light_sphere_disk_radius`.
- `draw/engines/eevee/shaders/eevee_light_eval.bsl.hh`: `eval_single_closure`, `light_eval_single`.
- `draw/engines/eevee/eevee_light.cc`: `shape_parameters_set` (the influence radius and the emitter sizes),
  `shape_radiance_get`.
- `blenkernel/intern/light.cc`: `BKE_light_power` (the energy times `2^exposure`).
- `draw/engines/eevee/shaders/eevee_bxdf_microfacet.bsl.hh`: `bxdf_ggx_dominant_direction_transmission`,
  `bxdf_ggx_perceived_roughness_transmission`, `bxdf_ggx_light_transmission`, `bxdf_ggx_light_reflection`.
- `draw/engines/eevee/shaders/eevee_bxdf_lut_lib.bsl.hh` and `eevee_nodetree_lib.bsl.hh`: `bsdf_lut` with multiscatter.
- `draw/engines/eevee/shaders/eevee_thickness_lib.bsl.hh`: the sphere model's exit normal.
- `draw/engines/eevee/shaders/eevee_shadow_tracing.bsl.hh`: `shadow_eval`, with the transmission bias (`N_bias`, and the
  receiver moved through the object by its thickness toward the light when the light is behind it).
- `draw/engines/eevee/shaders/eevee_deferred_combine.bsl.hh`: a transmission with a thickness takes its colour squared.
- `gpu/shaders/material/gpu_shader_material_principled.glsl`: the `MAT_REFRACTION` closures.

The tables are read from `crates/potter-core/src/render/eevee_transmission_tables.f32le`, the asset the Rust code
includes, so both implementations read the same numbers.

What the mirror models, and what it does not:
- The spheres are Blender's 64 x 32 UV spheres, with angle-weighted smooth normals; a camera ray hits the triangle the
  pixel centre falls in. The analytic sphere is off by up to 0.49 at the clear sphere's rim, where the transmitted lobe
  crosses the light's edge, so the tessellation is part of the model.
- The shadows are exact rays against the triangles; EEVEE traces a shadow map with jitter. The transmission ray is cast
  from the thickness-amended point, which is what the Rust code does too.
- EEVEE's film filter is off (`filter_size` 0 in the oracle), so each pixel is one sample at its centre. Supersampling
  the pixel makes the mirror worse (its rim value moves by 0.25), so EEVEE is point-sampled.

The world is black in the oracle, so the mirror is direct light only: the glass reflection (a GGX reflection closure whose
colour is the glass split's reflectance) and the transmission (the two-event colour, with the LTC transmission lobe). The
Principled's diffuse and dielectric layers take `(1 - transmission) = 0`.

Usage: python3 eevee_glass_direct_mirror.py <oracle.json> [--table <f32le>] [--dump <out.json>]
"""

from __future__ import annotations

import json
import math
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE.parents[2] / "crates" / "potter-core" / "src" / "render" / "eevee_transmission_tables.f32le"

TABLE_SIZE = 64
LAYER_COUNT = 16
BTDF_OFFSET = TABLE_SIZE * TABLE_SIZE * 2
BSDF_OFFSET = BTDF_OFFSET + LAYER_COUNT * TABLE_SIZE * TABLE_SIZE
LTC_OFFSET = BSDF_OFFSET + LAYER_COUNT * TABLE_SIZE * TABLE_SIZE * 3
DISK_OFFSET = LTC_OFFSET + TABLE_SIZE * TABLE_SIZE * 4
ASSET_LENGTH = DISK_OFFSET + TABLE_SIZE * TABLE_SIZE

# EEVEE's light shapes the mirror covers: `LIGHT_RECT`, `LIGHT_OMNI_DISK` (a point light with Soft Falloff) and
# `LIGHT_OMNI_SPHERE` (a point light without it).
RECT = "rect"
OMNI_DISK = "omni_disk"
OMNI_SPHERE = "omni_sphere"

# `eevee.light_threshold`, the scene setting that sets each local light's influence radius (`eevee_light.cc`).
DEFAULT_LIGHT_THRESHOLD = 0.01
# `UTIL_TEX` filtering and the attenuation threshold are not modelled: the oracle's lights are bright enough for both.
FACING_SOFTNESS = 0.1


# ---------------------------------------------------------------------------------------------------------------------
# Vectors and 3x3 matrices (rows).


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


def mat_vec(m, v):
    return tuple(dot(row, v) for row in m)


def transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_mul(a, b):
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def from_columns(c0, c1, c2):
    """GLSL `float3x3(c0, c1, c2)`: the columns, as rows of the matrix."""
    return [[c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], [c0[2], c1[2], c2[2]]]


def make_orthonormal_basis(n):
    """`draw_math_geom_lib.glsl` `make_orthonormal_basis`: the tangent and bitangent of `n`."""
    up = (0.0, 0.0, 1.0) if abs(n[2]) < 0.99999 else (1.0, 0.0, 0.0)
    t = normalize(cross(up, n))
    return t, cross(n, t)


def from_up_axis(up):
    """`gpu_shader_math_matrix_construct_lib.glsl` `from_up_axis`: columns T1, T2 and `up`."""
    z_sign = 1.0 if up[2] >= 0.0 else -1.0
    a = -1.0 / (z_sign + up[2])
    b = up[0] * up[1] * a
    c0 = (1.0 + z_sign * up[0] * up[0] * a, z_sign * b, -z_sign * up[0])
    c1 = (b, z_sign + up[1] * up[1] * a, -up[1])
    return from_columns(c0, c1, up)


def tangent_basis(n, v):
    """`eevee::ltc::detail::tangent_basis`: columns T1, T2 and N, with T1 along the view's tangent."""
    nv = dot(n, v)
    if nv > 0.999999:
        return from_up_axis(n)
    t1 = normalize(sub(v, mul(n, nv)))
    t2 = cross(n, t1)
    return from_columns(t1, t2, n)


# ---------------------------------------------------------------------------------------------------------------------
# The asset: EEVEE's LUTs, as the Rust code reads them.


class Tables:
    def __init__(self, path: Path):
        values = array("f")
        with path.open("rb") as handle:
            values.frombytes(handle.read())
        if len(values) != ASSET_LENGTH:
            raise SystemExit(f"{path}: {len(values)} values, expected {ASSET_LENGTH}")
        self.values = values

    def brdf(self, x, y, channel):
        return self.values[(y * TABLE_SIZE + x) * 2 + channel]

    def btdf(self, layer, x, y):
        return self.values[BTDF_OFFSET + (layer * TABLE_SIZE + y) * TABLE_SIZE + x]

    def ltc(self, x, y, channel):
        return self.values[LTC_OFFSET + (y * TABLE_SIZE + x) * 4 + channel]

    def disk(self, x, y):
        return self.values[DISK_OFFSET + y * TABLE_SIZE + x]


TABLES: Tables | None = None


def texel_weights(coordinate):
    """`UTIL_TEX_UV_SCALE` and `UTIL_TEX_UV_BIAS`: texel `i` sits at coordinate `i / 63`."""
    position = saturate(coordinate) * 63.0
    low = min(int(math.floor(position)), 63)
    high = min(low + 1, 63)
    return low, high, position - low


def blend(a, b, t):
    return a + (b - a) * t


def bilinear(texel, u, v):
    x0, x1, fx = texel_weights(u)
    y0, y1, fy = texel_weights(v)
    return blend(blend(texel(x0, y0), texel(x1, y0), fx), blend(texel(x0, y1), texel(x1, y1), fx), fy)


def layer_weights(roughness):
    layer = saturate(roughness) * LAYER_COUNT
    low = min(int(math.floor(layer)), LAYER_COUNT - 1)
    high = min(low + 1, LAYER_COUNT - 1)
    return low, high, layer - low


def ltc_matrix(cos_theta, roughness):
    """`eevee::lut::ltc::sample_utility_tx`: the packed inverse matrix `(m11, m13, m31, m33)` at `(roughness, sqrt(1 - cos))`."""
    v = math.sqrt(saturate(1.0 - cos_theta))
    return tuple(bilinear(lambda x, y, c=c: TABLES.ltc(x, y, c), roughness, v) for c in range(4))


def unpack(packed):
    """`eevee::lut::ltc::unpack`: `float3x3(float3(v.x, 0, v.y), float3(0, 1, 0), float3(v.z, 0, v.w))`."""
    x, y, z, w = packed
    return from_columns((x, 0.0, y), (0.0, 1.0, 0.0), (z, 0.0, w))


def diffuse_sphere_integral(avg_dir_z, form_factor):
    """`eevee::ltc::detail::diffuse_sphere_integral`: the tabulated horizon-clipped sphere integral."""
    return bilinear(TABLES.disk, avg_dir_z * 0.5 + 0.5, form_factor)


# ---------------------------------------------------------------------------------------------------------------------
# The LTC form factors.


def solve_cubic(c0, c1, c2, c3):
    """`eevee::ltc::detail::solve_cubic`: the three roots, ordered as the GLSL orders them."""
    x = c0 / c3
    y = c1 / c3 / 3.0
    z = c2 / c3 / 3.0
    w = c3
    A = w
    B = z
    C = y
    D = x
    delta = (y - z * z, x - y * z, z * x - y * y)
    discr = 4.0 * delta[0] * delta[2] - delta[1] * delta[1]
    sqrt_discr = math.sqrt(min(max(discr, 0.0), 3.4028234663852886e38))

    # Algorithm A.
    C_a = delta[0]
    D_a = -2.0 * B * delta[0] + delta[1]
    theta = math.atan2(sqrt_discr, -D_a) / 3.0
    two_sqrt_C_a = 2.0 * math.sqrt(max(-C_a, 0.0))
    x_1a = two_sqrt_C_a * math.cos(theta)
    x_3a = two_sqrt_C_a * math.cos(theta + (2.0 / 3.0) * math.pi)
    xl = x_1a if (x_1a + x_3a) > 2.0 * B else x_3a
    xlc = (xl - B, A)

    # Algorithm D.
    C_d = delta[2]
    D_d = -D * delta[1] + 2.0 * C * delta[2]
    theta = math.atan2(D * sqrt_discr, -D_d) / 3.0
    two_sqrt_C_d = 2.0 * math.sqrt(max(-C_d, 0.0))
    x_1d = two_sqrt_C_d * math.cos(theta)
    x_3d = two_sqrt_C_d * math.cos(theta + (2.0 / 3.0) * math.pi)
    xs = x_1d if (x_1d + x_3d) < 2.0 * C else x_3d
    xsc = (-D, xs + C)

    E = xlc[1] * xsc[1]
    F = -xlc[0] * xsc[1] - xlc[1] * xsc[0]
    G = xlc[0] * xsc[0]
    xmc = (C * F - B * G, -B * F + C * E)
    root = [xsc[0] / xsc[1], xmc[0] / xmc[1], xlc[0] / xlc[1]]
    if root[0] < root[1] and root[0] < root[2]:
        root = [root[1], root[0], root[2]]
    elif root[2] < root[0] and root[2] < root[1]:
        root = [root[0], root[2], root[1]]
    return root


def edge_integral_vec(v1, v2):
    """`eevee::ltc::detail::edge_integral_vec`, the edge integral of Hill and Heitz."""
    x = dot(v1, v2)
    y = abs(x)
    a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y
    b = 3.4175940 + (4.1616724 + y) * y
    v = a / b
    if x > 0.0:
        theta_sintheta = v
    else:
        theta_sintheta = 0.5 / math.sqrt(max(1.0 - x * x, 1.0e-7)) - v
    return mul(cross(v1, v2), theta_sintheta)


def evaluate_quad(corners, n, v, minv):
    """`eevee::ltc::evaluate_quad`: the LTC form factor of a rectangle whose corners are world-space vectors."""
    minv = mat_mul(minv, transpose(tangent_basis(n, v)))
    corners = [normalize(mat_vec(minv, c)) for c in corners]
    avg = (0.0, 0.0, 0.0)
    for i in range(4):
        avg = add(avg, edge_integral_vec(corners[i], corners[(i + 1) % 4]))
    # `form_factor_inv = inversesqrt(dot(avg, avg))`, so `saturate(1 / form_factor_inv)` is the length.
    avg_length = length(avg)
    if avg_length == 0.0:
        return 0.0
    form_factor = saturate(avg_length)
    return form_factor * diffuse_sphere_integral(avg[2] / avg_length, form_factor)


def evaluate_disk(n, v, minv, points):
    """`eevee::ltc::evaluate_disk`: the LTC form factor of a disk; `points` are three world-space bounding vectors."""
    r = transpose(tangent_basis(n, v))
    l0, l1, l2 = (mat_vec(r, p) for p in points)
    c = mul(add(l0, l2), 0.5)
    v1 = mul(sub(l1, l2), 0.5)
    v2 = mul(sub(l1, l0), 0.5)
    c = mat_vec(minv, c)
    v1 = mat_vec(minv, v1)
    v2 = mat_vec(minv, v2)

    d11 = dot(v1, v1)
    d22 = dot(v2, v2)
    d12 = dot(v1, v2)
    if abs(d12) / math.sqrt(d11 * d22) > 0.0007:
        tr = d11 + d22
        det = math.sqrt(-d12 * d12 + d11 * d22)
        u = 0.5 * math.sqrt(tr - 2.0 * det)
        w = 0.5 * math.sqrt(tr + 2.0 * det)
        e_max = (u + w) ** 2
        e_min = (u - w) ** 2
        if d11 > d22:
            v1_ = add(mul(v1, d12), mul(v2, e_max - d11))
            v2_ = add(mul(v1, d12), mul(v2, e_min - d11))
        else:
            v1_ = add(mul(v2, d12), mul(v1, e_max - d22))
            v2_ = add(mul(v2, d12), mul(v1, e_min - d22))
        a = 1.0 / e_max
        inv_b = e_min
        v1 = normalize(v1_)
        v2 = normalize(v2_)
    else:
        a = 1.0 / d11
        inv_b = d22
        v1 = mul(v1, math.sqrt(a))
        v2 = mul(v2, 1.0 / math.sqrt(inv_b))

    v3 = normalize(cross(v1, v2))
    if dot(c, v3) < 0.0:
        v3 = mul(v3, -1.0)
    L = dot(v3, c)
    inv_L = 1.0 / L
    x0 = dot(v1, c) * inv_L
    y0 = dot(v2, c) * inv_L

    ab = a * inv_b
    inv_b *= inv_L * inv_L
    t = 1.0 + x0 * x0
    c0 = ab * inv_b
    c1 = ab * (t + y0 * y0) - c0 - inv_b
    c2 = inv_b - ab * t - (1.0 + y0 * y0)
    e1, e2, e3 = solve_cubic(c0, c1, c2, 1.0)

    avg_dir = normalize(mat_vec(from_columns(v1, v2, v3), (ab * x0 / (ab - e2), y0 / (1.0 - e2), 1.0)))
    L1 = 1.0 / math.sqrt(-e3 / e2)
    L2 = 1.0 / math.sqrt(-e1 / e2)
    form_factor = saturate(L1 * L2 / math.sqrt((1.0 + L1 * L1) * (1.0 + L2 * L2)))
    return form_factor * diffuse_sphere_integral(avg_dir[2], form_factor)


# ---------------------------------------------------------------------------------------------------------------------
# Lights, as EEVEE's `Light` holds them.


class Light:
    def __init__(self, oracle):
        spec = oracle["light"]
        self.kind = {"AREA": RECT, "POINT": OMNI_DISK if spec["soft_falloff"] else OMNI_SPHERE}[spec["type"]]
        self.position = tuple(spec["location"])
        rotation = euler_matrix(spec["rotation"])
        self.x_axis = (rotation[0][0], rotation[1][0], rotation[2][0])
        self.y_axis = (rotation[0][1], rotation[1][1], rotation[2][1])
        self.z_axis = (rotation[0][2], rotation[1][2], rotation[2][2])
        self.energy = spec["energy"]
        self.color = tuple(spec["color"])
        self.threshold = spec["light_threshold"]
        if self.kind == RECT:
            # `shape_parameters_set`: the half extents, scaled by the object scale (one here).
            self.size = (spec["area_size"] / 2.0, spec["area_size"] / 2.0)
            self.shape_radius = math.hypot(*self.size)
        else:
            self.shape_radius = max(spec["radius"], 0.001)

    def radiance(self):
        """`Light::shape_radiance_get` times the light colour and the energy (`BKE_light_power`)."""
        if self.kind == RECT:
            area = self.size[0] * self.size[1] * 4.0
            shape = 1.0 / (math.pi * area)
        else:
            shape = 1.0 / (math.pi * 4.0 * math.pi * self.shape_radius ** 2)
        return tuple(channel * self.energy * shape for channel in self.color)

    def influence(self, distance):
        """`light_influence_attenuation`: the fade to zero at the radius where the power meets `light_threshold`."""
        max_power = max(self.color) * abs(self.energy / 100.0)
        radius_squared = max_power / self.threshold
        factor = saturate(1.0 - (distance * distance / radius_squared) ** 2)
        return factor * factor

    def common(self, l_dir):
        """`light_attenuation_common`: an area light emits along its own -Z, so the lit side is where `L · z > 0`."""
        if self.kind == RECT:
            return 1.0 if dot(l_dir, self.z_axis) > 0.0 else 0.0
        return 1.0

    def facing(self, l_dir, distance, ng, is_transmission):
        """`light_attenuation_facing`: fades a light near or below the horizon of `ng` (a transmission uses `-ng`)."""
        sin_solid_angle = self.shape_radius / distance
        sin_light_angle = dot(l_dir, mul(ng, -1.0) if is_transmission else ng)
        d = sin_solid_angle + (-sin_light_angle if is_transmission else sin_light_angle)
        return saturate((d + FACING_SOFTNESS) * 10.0)

    def corners(self, point):
        """`light_shape_corners`: the emitter's corner vectors from `point`, for the LTC evaluation."""
        to_centre = sub(self.position, point)
        if self.kind == RECT:
            hx, hy = self.size
            v0 = add(mul(self.x_axis, hx), mul(self.y_axis, -hy))
            v1 = add(mul(self.x_axis, hx), mul(self.y_axis, hy))
            return [add(v0, to_centre), add(v1, to_centre), add(mul(v0, -1.0), to_centre), add(mul(v1, -1.0), to_centre)]
        distance = length(to_centre)
        l_dir = mul(to_centre, 1.0 / distance)
        px, py = make_orthonormal_basis(l_dir)
        size = self.shape_radius
        if self.kind == OMNI_SPHERE:
            size = self.shape_radius / math.sqrt(max(1e-8, 1.0 - (self.shape_radius / distance) ** 2))
        v0 = add(mul(px, -size), mul(py, -size))
        v1 = add(mul(px, size), mul(py, -size))
        return [add(v0, to_centre), add(v1, to_centre), add(mul(v0, -1.0), to_centre)]

    def ltc(self, n, v, packed, point):
        """`light_ltc`: the form factor of this light for the normal `n`, the view `v` and the packed matrix."""
        minv = unpack(packed)
        corners = self.corners(point)
        if self.kind == RECT:
            return evaluate_quad(corners, n, v, minv)
        return evaluate_disk(n, v, minv, corners)


def euler_matrix(rotation):
    """Blender's XYZ Euler rotation (`rotation_euler`), as rows."""
    ax, ay, az = rotation
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    return mat_mul(rz, mat_mul(ry, rx))


# ---------------------------------------------------------------------------------------------------------------------
# The closures: EEVEE's `bsdf_lut` for a Principled Transmission of IOR above one, and the transmission light.


def glass_split(ior, roughness, n_dot_v, f0, transmission_tint):
    """`bsdf_lut` for IOR above one with the Principled's F90 of one: `(reflectance, transmittance)`."""
    amplitude = (ior - 1.0) / (ior + 1.0)
    real_f0 = amplitude * amplitude
    f90 = saturate(2.33 / 0.33 * amplitude)
    x_low, x_high, fx = texel_weights(roughness)
    y_low, y_high, fy = texel_weights(math.sqrt(1.0 - saturate(n_dot_v)))

    def brdf(channel):
        return blend(
            blend(TABLES.brdf(x_low, y_low, channel), TABLES.brdf(x_high, y_low, channel), fx),
            blend(TABLES.brdf(x_low, y_high, channel), TABLES.brdf(x_high, y_high, channel), fx),
            fy,
        )

    scale = brdf(0)
    bias = brdf(1)
    tx_low, tx_high, tfx = texel_weights(math.sqrt(amplitude))
    l0, l1, fl = layer_weights(roughness)

    def btdf_layer(layer):
        return blend(
            blend(TABLES.btdf(layer, tx_low, y_low), TABLES.btdf(layer, tx_high, y_low), tfx),
            blend(TABLES.btdf(layer, tx_low, y_high), TABLES.btdf(layer, tx_high, y_high), tfx),
            fy,
        )

    transmission = blend(btdf_layer(l0), btdf_layer(l1), fl)
    reflectance = f0 * scale + f90 * bias
    transmittance = (1.0 - f0) * transmission * transmission_tint
    energy = real_f0 * scale + bias + (1.0 - real_f0) * transmission
    average = f0 + (f90 - f0) / 21.0 if transmission_tint <= 0.0 else transmission_tint
    scaling = 1.0 / (1.0 - (1.0 - energy) * average)
    return reflectance * scaling, transmittance * scaling


def perceived_roughness_transmission(roughness, ior):
    """`bxdf_ggx_perceived_roughness_transmission` for IOR above one."""
    return roughness * math.sqrt(1.0 - 1.0 / ior)


def dominant_mix(roughness):
    m = roughness * roughness
    smoothness = 1.0 - m
    return smoothness * (math.sqrt(smoothness) + m)


def refract(incident, n, eta):
    """GLSL `refract`, zero under total internal reflection."""
    cos_i = dot(n, incident)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k < 0.0:
        return (0.0, 0.0, 0.0)
    return sub(mul(incident, eta), mul(n, eta * cos_i + math.sqrt(k)))


def dominant_direction_transmission(n, v, ior, roughness):
    """`bxdf_ggx_dominant_direction_transmission`: `refract(-v, n, 1 / ior)` mixed toward `-n`."""
    refracted = refract(mul(v, -1.0), n, 1.0 / ior)
    mixed = add(mul(n, -1.0), mul(sub(refracted, mul(n, -1.0)), dominant_mix(roughness)))
    return normalize(mixed)


def sphere_exit_normal(n, entry):
    """`ThicknessIsect sphere_intersect`: the exit normal of a sphere of the thickness's diameter tangent at the point."""
    cos_alpha = dot(entry, mul(n, -1.0))
    return normalize(add(n, mul(entry, cos_alpha * 2.0)))


def transmission_light(n, v, ior, roughness, thickness):
    """`bxdf_ggx_light_transmission`: the normal the transmission LTC integrates around, and its packed matrix.

    Without a thickness the lobe is `refract(-v, n, 1 / ior)` at `-n`. With one, the point is amended to the sphere's exit
    (`bxdf_ggx_context_amend_transmission`), and the matrix samples the exit's refraction at `ior`.
    """
    perceived = perceived_roughness_transmission(roughness, ior)
    if thickness == 0.0:
        cl_n = n
        refracted = refract(mul(v, -1.0), cl_n, 1.0 / ior)
    else:
        entry = dominant_direction_transmission(n, v, ior, perceived)
        cl_n = mul(sphere_exit_normal(n, entry), -1.0)
        refracted = refract(entry, cl_n, ior)
    cos_theta = dot(mul(cl_n, -1.0), refracted)
    return mul(cl_n, -1.0), ltc_matrix(cos_theta, perceived)


# ---------------------------------------------------------------------------------------------------------------------
# The geometry: the oracle's UV spheres, tessellated as Blender's `primitive_uv_sphere_add` (64 segments, 32 rings).
# EEVEE shades the triangles' interpolated smooth normals and shadows them, so the mirror does too.

SEGMENTS = 64
RINGS = 32


def uv_sphere(centre, radius):
    """Vertices and triangles of a Z-up UV sphere with its poles on the axis."""
    vertices = [(0.0, 0.0, 1.0)]
    for j in range(1, RINGS):
        theta = math.pi * j / RINGS
        for i in range(SEGMENTS):
            phi = 2.0 * math.pi * i / SEGMENTS
            vertices.append((math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)))
    vertices.append((0.0, 0.0, -1.0))
    bottom = len(vertices) - 1

    def ring(j, i):
        return 1 + (j - 1) * SEGMENTS + (i % SEGMENTS)

    triangles = []
    for i in range(SEGMENTS):
        triangles.append((0, ring(1, i), ring(1, i + 1)))
    for j in range(1, RINGS - 1):
        for i in range(SEGMENTS):
            a, b = ring(j, i), ring(j, i + 1)
            c, d = ring(j + 1, i), ring(j + 1, i + 1)
            triangles.append((a, c, d))
            triangles.append((a, d, b))
    for i in range(SEGMENTS):
        triangles.append((ring(RINGS - 1, i), bottom, ring(RINGS - 1, i + 1)))
    world = [add(mul(v, radius), centre) for v in vertices]
    return world, triangles


def face_normal(a, b, c):
    return normalize(cross(sub(b, a), sub(c, a)))


def angle_weighted_normals(vertices, triangles):
    """Blender's smooth vertex normals: the face normals, weighted by the corner angles at each vertex."""
    accumulated = [(0.0, 0.0, 0.0)] * len(vertices)
    for tri in triangles:
        corners = [vertices[k] for k in tri]
        normal = face_normal(*corners)
        for index in range(3):
            p = corners[index]
            q = normalize(sub(corners[(index + 1) % 3], p))
            r = normalize(sub(corners[(index + 2) % 3], p))
            angle = math.acos(max(-1.0, min(1.0, dot(q, r))))
            accumulated[tri[index]] = add(accumulated[tri[index]], mul(normal, angle))
    return [normalize(n) for n in accumulated]


def ray_triangle(origin, direction, a, b, c):
    """Möller-Trumbore: the distance and barycentrics of a hit, or `None`."""
    e1 = sub(b, a)
    e2 = sub(c, a)
    p = cross(direction, e2)
    det = dot(e1, p)
    if abs(det) < 1.0e-14:
        return None
    inv = 1.0 / det
    s = sub(origin, a)
    u = dot(s, p) * inv
    if u < -1.0e-9 or u > 1.0 + 1.0e-9:
        return None
    q = cross(s, e1)
    v = dot(direction, q) * inv
    if v < -1.0e-9 or u + v > 1.0 + 1.0e-9:
        return None
    t = dot(e2, q) * inv
    return (t, u, v)


class Scene:
    """Every sphere of the oracle, as triangles with their smooth normals."""

    def __init__(self, oracle):
        self.meshes = []
        for name, sphere in oracle["spheres"].items():
            vertices, triangles = uv_sphere(tuple(sphere["centre"]), 1.0)
            self.meshes.append((tuple(sphere["centre"]), vertices, triangles, angle_weighted_normals(vertices, triangles)))

    def nearest(self, origin, direction):
        """The first triangle a ray hits: its distance, the mesh's index, the triangle and the barycentrics."""
        best = None
        for index, (_, vertices, triangles, _) in enumerate(self.meshes):
            for tri in triangles:
                hit = ray_triangle(origin, direction, *(vertices[k] for k in tri))
                if hit and hit[0] > 1.0e-7 and (best is None or hit[0] < best[0]):
                    best = (hit[0], index, tri, hit[1], hit[2])
        return best

    def blocked(self, origin, direction, distance):
        for _, vertices, triangles, _ in self.meshes:
            for tri in triangles:
                hit = ray_triangle(origin, direction, *(vertices[k] for k in tri))
                if hit and SHADOW_EPSILON < hit[0] < distance:
                    return True
        return False


# ---------------------------------------------------------------------------------------------------------------------
# Shadows against the tessellated spheres.

SHADOW_EPSILON = 1.0e-5


def reflect(incident, n):
    return sub(incident, mul(n, 2.0 * dot(n, incident)))


def shadow_visibility(scene, point, ng, light_position, thickness, is_transmission):
    """`shadow_eval` for a punctual light: a ray from the (biased) point toward the light's centre.

    A transmission whose light is behind the surface (`!facing`) moves the point through the object by its thickness
    toward the light, capped before the light, and biases along `reflect(Ng, L)`. Otherwise it biases along `Ng`.
    """
    to_light = sub(light_position, point)
    distance = length(to_light)
    l_dir = mul(to_light, 1.0 / distance)
    facing = dot(ng, l_dir) > 0.0
    n_bias = reflect(ng, l_dir) if (is_transmission and not facing) else ng
    p = point
    if is_transmission and not facing:
        p = add(p, mul(l_dir, min(thickness, distance - 0.01)))
    p = add(p, mul(n_bias, 10.0 * SHADOW_EPSILON))
    to_light = sub(light_position, p)
    remaining = length(to_light)
    return 0.0 if scene.blocked(p, mul(to_light, 1.0 / remaining), remaining) else 1.0


# ---------------------------------------------------------------------------------------------------------------------
# The pixel: the glass closures of one shading point, summed as EEVEE sums them.


def shade(point, normal, geometric, light, glass, scene):
    """The direct glass light at a shading point seen along `V = (0, 0, 1)` (EEVEE's orthographic view).

    `normal` is the interpolated shading normal; `geometric` is the face normal, which biases the shadow rays.
    """
    view = (0.0, 0.0, 1.0)
    ior = glass["ior"]
    roughness = glass["roughness"]
    thickness = glass["thickness"]
    n_dot_v = dot(normal, view)
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    reflectance, transmittance = glass_split(ior, roughness, n_dot_v, f0, 1.0)
    to_light = sub(light.position, point)
    distance = length(to_light)
    l_dir = mul(to_light, 1.0 / distance)
    influence = light.influence(distance)
    radiance = light.radiance()
    total = [0.0, 0.0, 0.0]

    # The reflection: the glass part of the specular closure, a GGX reflection LTC at `(N · V, roughness)`.
    att = influence * light.common(l_dir) * light.facing(l_dir, distance, normal, False)
    if att > 0.0:
        vis = shadow_visibility(scene, point, geometric, light.position, 0.0, False)
        ltc = light.ltc(normal, view, ltc_matrix(n_dot_v, roughness), point)
        for c in range(3):
            total[c] += reflectance * radiance[c] * att * vis * ltc

    # The transmission: the two-event colour, with the LTC transmission lobe on the lobe's own normal.
    light_n, packed = transmission_light(normal, view, ior, roughness, thickness)
    att = influence * light.common(l_dir) * light.facing(l_dir, distance, light_n, True)
    if att > 0.0:
        vis = shadow_visibility(scene, point, geometric, light.position, thickness, True)
        ltc = light.ltc(light_n, view, packed, point)
        colour = transmittance * transmittance if thickness > 0.0 else transmittance
        for c in range(3):
            total[c] += colour * radiance[c] * att * vis * ltc
    return total


def predict(oracle):
    """The predicted RGB of every interior sample of each sphere, in the oracle's order.

    Each sample is a camera ray down -Z (EEVEE's orthographic view) through the sample's world position, which hits the
    tessellated sphere it belongs to. The shading point is the hit's interpolated normal, with the face normal for the
    shadow bias.
    """
    light = Light(oracle)
    thickness = oracle["object_thickness"]
    scene = Scene(oracle)
    predictions = {}
    for name, sphere in oracle["spheres"].items():
        glass = {"ior": oracle["ior"], "roughness": sphere["roughness"], "thickness": thickness}
        rows = []
        for sample in sphere["samples"]:
            origin = (sample["world"][0], sample["world"][1], 10.0)
            hit = scene.nearest(origin, (0.0, 0.0, -1.0))
            if hit is None:
                rows.append([0.0, 0.0, 0.0])
                continue
            t, index, tri, u, v = hit
            _, vertices, _, normals = scene.meshes[index]
            a, b, c = tri
            point = add(origin, mul((0.0, 0.0, -1.0), t))
            shading = normalize(add(add(mul(normals[a], 1.0 - u - v), mul(normals[b], u)), mul(normals[c], v)))
            geometric = face_normal(vertices[a], vertices[b], vertices[c])
            if dot(geometric, sub(point, scene.meshes[index][0])) < 0.0:
                geometric = mul(geometric, -1.0)
            rows.append(shade(point, shading, geometric, light, glass, scene))
        predictions[name] = rows
    return predictions


def deviations(oracle, predictions):
    report = {}
    for name, sphere in oracle["spheres"].items():
        worst = 0.0
        worst_at = None
        total = 0.0
        for sample, predicted in zip(sphere["samples"], predictions[name]):
            error = max(abs(p - w) for p, w in zip(predicted, sample["rgb"]))
            total += error
            if error > worst:
                worst = error
                worst_at = sample["world"]
        report[name] = {
            "max": worst,
            "mean": total / len(sphere["samples"]),
            "at": worst_at,
            "samples": len(sphere["samples"]),
        }
    return report


def main(argv):
    global TABLES
    if len(argv) < 2:
        raise SystemExit(__doc__)
    table_path = Path(argv[argv.index("--table") + 1]) if "--table" in argv else DEFAULT_TABLE
    TABLES = Tables(table_path)
    oracle = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    predictions = predict(oracle)
    report = deviations(oracle, predictions)
    print(json.dumps({"scene": oracle["scene"], "deviation": report}, indent=2))
    if "--dump" in argv:
        Path(argv[argv.index("--dump") + 1]).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
