//! Float32 normal, centre and projection helpers mirroring what Blender's Decimate COLLAPSE sees.
//!
//! `BKE_mesh_to_bmesh_ex` (with `calc_face_normal` and `calc_vert_normal`) recomputes each face
//! normal with `BM_face_normal_update` but copies each vertex normal from `Mesh::vert_normals()`.
//! The Mesh vertex normals come from `normals_calc_verts`, which uses the Mesh face normals
//! (`normal_calc_ngon`), not the `BMesh` ones.

type Vec3 = [f32; 3];
type Mat3 = [[f32; 3]; 3];

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

/// `math::normalize` (`BLI_math_vector.hh`): divides each component by the length.
fn normalize_divide(v: Vec3) -> Vec3 {
    let len_sq = dot(v, v);
    if len_sq > 1.0e-35 {
        let len = len_sq.sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0; 3]
    }
}

/// `normal_tri_v3`.
fn normal_tri(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
    let mut n = cross(sub(v1, v2), sub(v2, v3));
    normalize_in_place(&mut n);
    n
}

/// `normal_quad_v3`: the cross product of the two diagonals.
fn normal_quad(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> Vec3 {
    let mut n = cross(sub(v1, v3), sub(v2, v4));
    normalize_in_place(&mut n);
    n
}

/// Newell sum over the edges `(prev, curr)`, starting with `(last, first)`. This is the order used
/// by both `bm_face_calc_poly_normal` and `normal_calc_ngon`.
fn newell_sum(points: &[Vec3]) -> Vec3 {
    let mut n = [0.0f32; 3];
    let mut prev = *points.last().unwrap_or(&[0.0; 3]);
    for &curr in points {
        n[0] += (prev[1] - curr[1]) * (prev[2] + curr[2]);
        n[1] += (prev[2] - curr[2]) * (prev[0] + curr[0]);
        n[2] += (prev[0] - curr[0]) * (prev[1] + curr[1]);
        prev = curr;
    }
    n
}

/// Face normal that `BKE_mesh_to_bmesh_ex` gives `f->no`: `BM_face_normal_update` →
/// `BM_face_calc_normal`. Triangles use `normal_tri_v3` and quads use `normal_quad_v3`. Larger
/// faces use Newell's method with `normalize_v3` and no `+Z` fallback, so a degenerate face
/// keeps a zero normal.
pub(super) fn face_normal(points: &[[f32; 3]]) -> [f32; 3] {
    match *points {
        [v1, v2, v3] => normal_tri(v1, v2, v3),
        [v1, v2, v3, v4] => normal_quad(v1, v2, v3, v4),
        _ => {
            let mut n = newell_sum(points);
            normalize_in_place(&mut n);
            n
        }
    }
}

/// `normal_calc_ngon` (`mesh_normals.cc`): the Mesh face normal used for vertex normals. It is
/// Newell's method for every face size, with `+Z` for a zero result.
fn mesh_face_normal(points: &[Vec3]) -> Vec3 {
    let mut n = newell_sum(points);
    if normalize_in_place(&mut n) == 0.0 {
        n[2] = 1.0;
    }
    n
}

/// `safe_acos_approx` (`BLI_math_base.hh`): a polynomial approximation of `acos` with clamping.
#[expect(
    clippy::excessive_precision,
    reason = "Blender's literal, kept bit-exact"
)]
fn safe_acos_approx(x: f32) -> f32 {
    let f = x.abs();
    let m = if f < 1.0 { 1.0 - (1.0 - f) } else { 1.0 };
    let a = (1.0 - m).sqrt()
        * (std::f32::consts::FRAC_PI_2
            + m * (-0.213_300_989_f32 + m * (0.077_980_478_f32 + m * -0.021_640_95_f32)));
    if x < 0.0 { std::f32::consts::PI - a } else { a }
}

/// Vertex normals as `normals_calc_verts` (`mesh_normals.cc`) produces them, which
/// `BKE_mesh_to_bmesh_ex` copies into `v->no`. Each vertex sums its incident Mesh face normals
/// (ascending face index) weighted by `safe_acos_approx` of the angle between the two edges.
/// Isolated vertices keep the normalized position.
pub(super) fn vertex_normals(points: &[[f32; 3]], faces: &[Vec<usize>]) -> Vec<[f32; 3]> {
    let face_normals: Vec<Vec3> = faces
        .iter()
        .map(|face| {
            let corners: Vec<Vec3> = face.iter().map(|&v| points[v]).collect();
            mesh_face_normal(&corners)
        })
        .collect();

    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (fi, face) in faces.iter().enumerate() {
        for &v in face {
            vert_faces[v].push(fi);
        }
    }

    vert_faces
        .iter()
        .enumerate()
        .map(|(v, incident)| {
            let p = points[v];
            if incident.is_empty() {
                return normalize_divide(p);
            }
            let mut acc = [0.0f32; 3];
            for &fi in incident {
                let face = &faces[fi];
                let Some(k) = face.iter().position(|&c| c == v) else {
                    continue;
                };
                let len = face.len();
                let prev = points[face[(k + len - 1) % len]];
                let next = points[face[(k + 1) % len]];
                let dir_prev = normalize_divide(sub(prev, p));
                let dir_next = normalize_divide(sub(next, p));
                let factor = safe_acos_approx(dot(dir_prev, dir_next));
                for (a, n) in acc.iter_mut().zip(face_normals[fi]) {
                    *a += n * factor;
                }
            }
            normalize_divide(acc)
        })
        .collect()
}

/// `BM_face_calc_center_median`: sums the corners in loop order, then scales by `1.0f / len`.
pub(super) fn face_center_median(points: &[[f32; 3]]) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for p in points {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    let scale = 1.0 / points.len() as f32;
    [c[0] * scale, c[1] * scale, c[2] * scale]
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

/// `axis_dominant_v3_to_m3` body (`math_geom.cc`) given the third row, which is already negated for
/// the `_negate` variant.
fn axis_dominant_matrix(third: Vec3) -> Mat3 {
    let (r0, r1) = ortho_basis(third);
    let mut m = [r0, r1, third];
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

/// `axis_dominant_v3_to_m3_negate` followed by `mul_v2_m3v3` for every point.
pub(super) fn project_axis_dominant(normal: [f32; 3], points: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let m = axis_dominant_matrix([-normal[0], -normal[1], -normal[2]]);
    points.iter().map(|&p| mul_v2_m3v3(&m, p)).collect()
}
