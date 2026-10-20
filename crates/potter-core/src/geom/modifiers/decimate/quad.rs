//! `is_quad_convex_v3` (`math_geom.cc`) for Decimate's COLLAPSE. The diagonal choice for quads under
//! `MOD_TRIANGULATE_QUAD_BEAUTY` is the shared `geom::triangulate::quad_beauty_split_24`.

type Vec3 = [f32; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
