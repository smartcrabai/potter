//! Diagonal choice for quads under Blender's `MOD_TRIANGULATE_QUAD_BEAUTY` (the `split_24` branch
//! of `BM_face_triangulate` in `bmesh_polygon.cc`), plus `is_quad_convex_v3`.

type Vec3 = [f32; 3];
type Mat3 = [[f32; 3]; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `cross_v3_v3v3`.
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `cross_tri_v3` (`math_geom.cc`): unnormalized normal of the corner triangle `v1, v2, v3`.
fn cross_tri(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
    cross(sub(v1, v2), sub(v2, v3))
}

/// `normalize_v3`: scales by the reciprocal length; returns the old length, or zeroes the vector
/// and returns 0 when the squared length is not above `1e-35`.
fn normalize_in_place(n: &mut Vec3) -> f32 {
    let d = dot(*n, *n);
    if d > 1.0e-35 {
        let len = d.sqrt();
        let scale = 1.0 / len;
        *n = [n[0] * scale, n[1] * scale, n[2] * scale];
        len
    } else {
        *n = [0.0; 3];
        0.0
    }
}

/// `project_plane_v3_v3v3` (`math_vector.cc`).
fn project_plane(p: Vec3, plane: Vec3) -> Vec3 {
    let mul = dot(p, plane) / dot(plane, plane);
    let f = -mul;
    [
        p[0] + plane[0] * f,
        p[1] + plane[1] * f,
        p[2] + plane[2] * f,
    ]
}

/// `is_quad_flip_v3` (`math_geom.cc`) as `(bit 0, bit 1)`: bit 0 compares the turns at corners 2
/// and 4, bit 1 the turns at corners 1 and 3. A set bit means those two corners turn oppositely.
fn is_quad_flip(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> (bool, bool) {
    let d12 = sub(v1, v2);
    let d23 = sub(v2, v3);
    let d34 = sub(v3, v4);
    let d41 = sub(v4, v1);

    let bit0 = dot(cross(d12, d23), cross(d34, d41)) < 0.0;
    let bit1 = dot(cross(d23, d34), cross(d41, d12)) < 0.0;
    (bit0, bit1)
}

/// `split_24` for a quad `v1..v4` in face order, as `BM_face_triangulate` computes it with
/// `MOD_TRIANGULATE_QUAD_BEAUTY`: a flip bit decides outright, otherwise the beauty cost decides.
pub(super) fn quad_split_24(v1: [f32; 3], v2: [f32; 3], v3: [f32; 3], v4: [f32; 3]) -> bool {
    let (bit0, bit1) = is_quad_flip(v1, v2, v3, v4);
    if bit0 {
        return true;
    }
    if bit1 {
        return false;
    }
    rotate_beauty_cost(v1, v2, v3, v4) > 0.0
}

/// `BM_verts_calc_rotate_beauty` with `flag = 0` and `method = 0`, which calls
/// `BLI_polyfill_edge_calc_rotate_beauty__area` with `lock_degenerate = false`. The `v1 == v3`
/// pointer check is omitted because the caller always passes four distinct quad corners.
fn rotate_beauty_cost(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> f32 {
    const EPS: f32 = 1e-5;

    let no_a = cross_tri(v2, v3, v4);
    let no_b = cross_tri(v2, v4, v1);
    let mut no = add(no_a, no_b);
    let no_scale = normalize_in_place(&mut no);
    if no_scale == 0.0 {
        return f32::MAX;
    }

    let m = axis_dominant_matrix(no);
    let v1_xy = mul_v2_m3v3(&m, v1);
    let v2_xy = mul_v2_m3v3(&m, v2);
    let v3_xy = mul_v2_m3v3(&m, v3);
    let v4_xy = mul_v2_m3v3(&m, v4);

    // Both faces must not be opposite-wound; one degenerate face is allowed.
    let sign_sum = signum_i_ex(cross_tri_v2(v2_xy, v3_xy, v4_xy) / no_scale, EPS)
        + signum_i_ex(cross_tri_v2(v2_xy, v4_xy, v1_xy) / no_scale, EPS);
    if sign_sum == 0 {
        return f32::MAX;
    }

    quad_rotate_cost_2d(v1_xy, v2_xy, v3_xy, v4_xy)
}

/// `BLI_polyfill_beautify_quad_rotate_calc_ex` with `lock_degenerate = false` and no area output.
fn quad_rotate_cost_2d(v1: [f32; 2], v2: [f32; 2], v3: [f32; 2], v4: [f32; 2]) -> f32 {
    const EPS_ZERO_AREA: f32 = 1e-12;

    let area_2x_234 = cross_tri_v2(v2, v3, v4);
    let area_2x_241 = cross_tri_v2(v2, v4, v1);
    let area_2x_123 = cross_tri_v2(v1, v2, v3);
    let area_2x_134 = cross_tri_v2(v1, v3, v4);

    if (area_2x_123 >= 0.0) != (area_2x_134 >= 0.0) {
        return f32::MAX;
    }
    if area_2x_123.abs() <= EPS_ZERO_AREA || area_2x_134.abs() <= EPS_ZERO_AREA {
        return f32::MAX;
    }

    if (area_2x_234 >= 0.0) != (area_2x_241 >= 0.0) {
        return -f32::MAX;
    }
    if area_2x_234.abs() <= EPS_ZERO_AREA || area_2x_241.abs() <= EPS_ZERO_AREA {
        return -f32::MAX;
    }

    let len_12 = len_v2v2(v1, v2);
    let len_23 = len_v2v2(v2, v3);
    let len_34 = len_v2v2(v3, v4);
    let len_41 = len_v2v2(v4, v1);
    let len_13 = len_v2v2(v1, v3);
    let len_24 = len_v2v2(v2, v4);

    let fac_24 = (area_2x_234.abs() / (len_23 + len_34 + len_24))
        + (area_2x_241.abs() / (len_41 + len_12 + len_24));
    let fac_13 = (area_2x_123.abs() / (len_12 + len_23 + len_13))
        + (area_2x_134.abs() / (len_34 + len_41 + len_13));

    fac_24 - fac_13
}

/// `is_quad_convex_v3` (`math_geom.cc`): projects the four corners onto the plane spanned by the
/// diagonals and checks that every corner turns the same way as that plane's normal.
pub(super) fn is_quad_convex(v1: [f32; 3], v2: [f32; 3], v3: [f32; 3], v4: [f32; 3]) -> bool {
    let plane = cross(sub(v1, v3), sub(v2, v4));

    let eps_sq = 1e-8_f32 * 1e-8_f32;
    if dot(plane, plane) < eps_sq {
        return false;
    }

    let proj = [v1, v2, v3, v4].map(|p| project_plane(p, plane));
    let dirs = [
        sub(proj[0], proj[3]),
        sub(proj[1], proj[0]),
        sub(proj[2], proj[1]),
        sub(proj[3], proj[2]),
    ];

    let turns_with_plane = |a: Vec3, b: Vec3| dot(plane, cross(a, b)) > 0.0;
    turns_with_plane(dirs[0], dirs[1])
        && turns_with_plane(dirs[1], dirs[2])
        && turns_with_plane(dirs[2], dirs[3])
        && turns_with_plane(dirs[3], dirs[0])
}

/// `cross_tri_v2` (`math_geom_inline.cc`).
fn cross_tri_v2(v1: [f32; 2], v2: [f32; 2], v3: [f32; 2]) -> f32 {
    (v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0])
}

/// `len_v2v2` (`math_vector_inline.cc`).
fn len_v2v2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    (x * x + y * y).sqrt()
}

/// `signum_i_ex` (`math_base_inline.cc`).
fn signum_i_ex(a: f32, eps: f32) -> i32 {
    if a > eps {
        1
    } else if a < -eps {
        -1
    } else {
        0
    }
}

/// `ortho_basis_v3v3_v3` (`math_vector.cc`).
fn ortho_basis(n: Vec3) -> (Vec3, Vec3) {
    let f = n[0] * n[0] + n[1] * n[1];
    if f > f32::EPSILON {
        let d = 1.0 / f.sqrt();
        let r1 = [n[1] * d, -n[0] * d, 0.0];
        let r2 = [-n[2] * r1[1], n[2] * r1[0], n[0] * r1[1] - n[1] * r1[0]];
        (r1, r2)
    } else {
        let r1 = [if n[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0];
        (r1, [0.0, 1.0, 0.0])
    }
}

/// `axis_dominant_v3_to_m3` (`math_geom.cc`): rotation whose third row is `normal`, stored as
/// `M[col][row]` and transposed.
fn axis_dominant_matrix(normal: Vec3) -> Mat3 {
    let (r0, r1) = ortho_basis(normal);
    let mut m = [r0, r1, normal];
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let t = m[i][j];
        m[i][j] = m[j][i];
        m[j][i] = t;
    }
    m
}

/// `mul_v2_m3v3` (`math_matrix_c.cc`), with `M[col][row]` indexing.
fn mul_v2_m3v3(m: &Mat3, a: Vec3) -> [f32; 2] {
    [
        m[0][0] * a[0] + m[1][0] * a[1] + m[2][0] * a[2],
        m[0][1] * a[0] + m[1][1] * a[1] + m[2][1] * a[2],
    ]
}
