//! Blender's f32 arithmetic for the particle chains that place Explode fragments: the birth
//! velocity normalisation (`particle_system.cc`, `psys_get_birth_coords`), the velocity orientation
//! (`vec_to_quat`, `math_rotation_c.cc`), and the fragment transform (`MOD_explode.cc`, lines
//! 1000-1017). Each function performs Blender's operations in Blender's order, so on a platform
//! whose `sinf`, `cosf`, `acosf`, and `atan2f` match, the result equals Blender's bits.
//!
//! Quaternions use Blender's `(w, x, y, z)` order. Matrices are Blender's `M[column][row]`, which is
//! glam's column-major layout. Potter's f64 values are rounded to f32 on entry.

use glam::DMat4;

pub(crate) type Vec3f = [f32; 3];
pub(crate) type Quatf = [f32; 4];
pub(crate) type Mat4f = [[f32; 4]; 4];

/// Blender's `dot_v3v3`: `x*x + y*y + z*z`, in that order.
fn dot3(a: Vec3f, b: Vec3f) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `normalize_v3` (`math_vector_inline.cc`): scales by `1 / sqrt(d)` when `d > 1e-35`, else zeroes.
pub(crate) fn normalize3(a: Vec3f) -> Vec3f {
    let d = dot3(a, a);
    if d > 1.0e-35 {
        let mul = 1.0 / d.sqrt();
        [a[0] * mul, a[1] * mul, a[2] * mul]
    } else {
        [0.0; 3]
    }
}

/// `mul_v3_fl`.
pub(crate) fn scale3(a: Vec3f, factor: f32) -> Vec3f {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

/// `sub_v3_v3v3`.
fn sub3(a: Vec3f, b: Vec3f) -> Vec3f {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `add_v3_v3`.
fn add3(a: Vec3f, b: Vec3f) -> Vec3f {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `mul_qt_qtqt` (`math_rotation_c.cc`): the Hamilton product `a * b`.
pub(crate) fn mul_qt_qtqt(a: Quatf, b: Quatf) -> Quatf {
    let t0 = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    let t1 = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    let t2 = a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3];
    let q3 = a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1];
    [t0, t1, t2, q3]
}

/// `sub_qt_qtqt`: multiplies by `(-b0, b1, b2, b3)`, which is `-conj(b)`, so the result is
/// `-(a * conj(b))`. The sandwich in `mul_qt_v3` cancels the sign.
pub(crate) fn sub_qt_qtqt(a: Quatf, b: Quatf) -> Quatf {
    mul_qt_qtqt(a, [-b[0], b[1], b[2], b[3]])
}

/// `mul_qt_v3` (`math_rotation_c.cc:77`): the sandwich `q * r * conj(q)` in two stages, each
/// written as Blender writes it.
pub(crate) fn mul_qt_v3(q: Quatf, r: Vec3f) -> Vec3f {
    let t0 = -q[1] * r[0] - q[2] * r[1] - q[3] * r[2];
    let t1 = q[0] * r[0] + q[2] * r[2] - q[3] * r[1];
    let t2 = q[0] * r[1] + q[3] * r[0] - q[1] * r[2];
    let r2 = q[0] * r[2] + q[1] * r[1] - q[2] * r[0];
    let r = [t1, t2, r2];
    [
        t0 * -q[1] + r[0] * q[0] - r[1] * q[3] + r[2] * q[2],
        t0 * -q[2] + r[1] * q[0] - r[2] * q[1] + r[0] * q[3],
        t0 * -q[3] + r[2] * q[0] - r[0] * q[2] + r[1] * q[1],
    ]
}

/// `mul_m4_v3` (`math_matrix_c.cc`): transforms a point by a 4x4 matrix in Blender's order.
pub(crate) fn mul_m4_v3(m: &Mat4f, r: Vec3f) -> Vec3f {
    let x = r[0];
    let y = r[1];
    [
        x * m[0][0] + y * m[1][0] + m[2][0] * r[2] + m[3][0],
        x * m[0][1] + y * m[1][1] + m[2][1] * r[2] + m[3][1],
        x * m[0][2] + y * m[1][2] + m[2][2] * r[2] + m[3][2],
    ]
}

/// The row Blender's `quat_to_mat3` stores as `m[2]`, which `vec_to_quat` reads. Blender computes
/// these in double and rounds each result to float.
fn quat_to_mat3_row2(q: Quatf) -> [f32; 3] {
    let s2 = std::f64::consts::SQRT_2;
    let q0 = s2 * f64::from(q[0]);
    let q1 = s2 * f64::from(q[1]);
    let q2 = s2 * f64::from(q[2]);
    let q3 = s2 * f64::from(q[3]);
    let qda = q0 * q1;
    let qdb = q0 * q2;
    let qaa = q1 * q1;
    let qac = q1 * q3;
    let qbb = q2 * q2;
    let qbc = q2 * q3;
    [
        (qdb + qac) as f32,
        (-qda + qbc) as f32,
        (1.0 - qaa - qbb) as f32,
    ]
}

/// `vec_to_quat(q, vec, OB_POSX, OB_POSZ)` (`math_rotation_c.cc`): the rotation that takes local +X
/// onto `-vec`, twisted so local +Z stays as upright as possible. The result is not renormalised.
pub(crate) fn vec_to_quat_posx_posz(vec: Vec3f) -> Quatf {
    const EPSILON: f32 = 1.0e-4;
    let len = dot3(vec, vec).sqrt();
    if len.total_cmp(&0.0).is_eq() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let tvec = [-vec[0], -vec[1], -vec[2]];
    let mut nor = [0.0, -tvec[2], tvec[1]];
    if tvec[1].abs() + tvec[2].abs() < EPSILON {
        nor[1] = 1.0;
    }
    let co = tvec[0] / len;
    let nor = normalize3(nor);
    // `safe_acosf` clamps its argument, then calls `acosf`.
    let angle = co.clamp(-1.0, 1.0).acos();
    // `axis_angle_normalized_to_quat`.
    let phi = 0.5 * angle;
    let si = phi.sin();
    let co_half = phi.cos();
    let q = [co_half, nor[0] * si, nor[1] * si, nor[2] * si];
    // Axis 0 with an up flag other than 1 twists by `-0.5 * atan2(fp[1], fp[2])`.
    let fp = quat_to_mat3_row2(q);
    let twist = -0.5 * fp[1].atan2(fp[2]);
    let si2 = twist.sin() / len;
    let q2 = [twist.cos(), tvec[0] * si2, tvec[1] * si2, tvec[2] * si2];
    mul_qt_qtqt(q2, q)
}

/// The birth velocity of a vertex particle: `normalize(normal) * normal_factor` in f32
/// (`psys_get_birth_coords`). With no tangent, random, or object velocity terms, Blender adds
/// nothing else.
pub(crate) fn vertex_birth_velocity(normal: Vec3f, normal_factor: f32) -> Vec3f {
    scale3(normalize3(normal), normal_factor)
}

/// The orientation Blender's `PART_ROT_VEL` gives a particle moving at `velocity`. Blender passes
/// `-velocity` to `vec_to_quat`, which negates it again, so the arithmetic runs on `velocity`.
pub(crate) fn velocity_orientation(velocity: Vec3f) -> Quatf {
    vec_to_quat_posx_posz(scale3(velocity, -1.0))
}

/// One Explode fragment vertex (`MOD_explode.cc` lines 1000-1017): the mesh point goes to world
/// space, is shifted by the particle's birth location, rotated by `state * conj(birth)`, scaled by
/// the particle size when `use_size` is set, moved to the particle's current location, and
/// returned to object space.
/// `birth` and `state` are the particle's `(location, rotation)` at birth and now.
pub(crate) fn explode_fragment(
    local: Vec3f,
    (to_world, to_local): (&Mat4f, &Mat4f),
    (birth_co, birth_rotation): (Vec3f, Quatf),
    (state_co, state_rotation): (Vec3f, Quatf),
    size: Option<f32>,
) -> Vec3f {
    let world = mul_m4_v3(to_world, local);
    let relative = sub3(world, birth_co);
    let rotation = sub_qt_qtqt(state_rotation, birth_rotation);
    let mut moved = mul_qt_v3(rotation, relative);
    if let Some(size) = size {
        moved = scale3(moved, size);
    }
    mul_m4_v3(to_local, add3(moved, state_co))
}

/// Blender's `M[column][row]` for a glam matrix, rounded to f32.
pub(crate) fn mat4(matrix: DMat4) -> Mat4f {
    matrix
        .to_cols_array_2d()
        .map(|column| column.map(|value| value as f32))
}

/// A point rounded to f32.
pub(crate) fn vec3(point: [f64; 3]) -> Vec3f {
    point.map(|value| value as f32)
}

/// Potter's `[x, y, z, w]` quaternion as Blender's f32 `(w, x, y, z)`.
pub(crate) fn quat_from_xyzw(quat: [f64; 4]) -> Quatf {
    [quat[3], quat[0], quat[1], quat[2]].map(|value| value as f32)
}

/// Blender's f32 `(w, x, y, z)` as Potter's `[x, y, z, w]`.
pub(crate) fn quat_to_xyzw(quat: Quatf) -> [f64; 4] {
    [quat[1], quat[2], quat[3], quat[0]].map(f64::from)
}

#[cfg(test)]
mod tests {
    use super::{quat_to_xyzw, velocity_orientation, vertex_birth_velocity};

    // Blender 5.2.2 values (`particle_system.cc` particle states, `vec_to_quat` stored rotations) for
    // the explode edge-cut fixture: a box emitter with two vertex particles and normal_factor 0.8.
    // The corner normal is Blender's vertex normal for the corner at (-1, 1, 1).
    const CORNER_NORMAL: [f32; 3] = [-0.577_350_26, 0.577_350_26, 0.577_350_26];
    const NORMAL_FACTOR: f32 = 0.8;
    const VELOCITY_A: [f32; 3] = [-0.461_880_26, 0.461_880_26, 0.461_880_26];
    const VELOCITY_B: [f32; 3] = [-0.461_880_26, -0.461_880_26, 0.461_880_26];
    const ROTATION_A: [f32; 4] = [0.364_705_26, 0.279_848_04, -0.115_916_95, 0.880_476_2];
    const ROTATION_B: [f32; 4] = [0.364_705_26, -0.279_848_04, -0.115_916_95, -0.880_476_2];

    #[test]
    fn birth_velocity_matches_blender_bits() {
        assert_eq!(
            vertex_birth_velocity(CORNER_NORMAL, NORMAL_FACTOR),
            VELOCITY_A
        );
        // The corner at (-1, -1, 1) has the normal with its y component flipped.
        let flipped = [CORNER_NORMAL[0], -CORNER_NORMAL[1], CORNER_NORMAL[2]];
        assert_eq!(vertex_birth_velocity(flipped, NORMAL_FACTOR), VELOCITY_B);
    }

    /// Per-component bound, derived from one-ulp perturbations of each libm call in `vec_to_quat`
    /// (acos, sin and cos of phi, atan2, sin and cos of the twist). The worst first-order sums are
    /// 2.0, 3.25, 2.62 and 3.0 ulp(1); adding 0.5 ulp(1) for the final f32 rounding gives 4 ulp(1),
    /// which is 2^-21 at magnitude 1. Blender's stored bits come from macOS libm; the Linux CI
    /// libm may differ by up to this bound, and macOS is the parity target (checked on the lane).
    const ORIENTATION_LIBM_BOUND: f32 = 4.0 * (1.0 / 8_388_608.0);

    fn assert_within_libm_bound(actual: [f32; 4], expected: [f32; 4]) {
        for (component, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= ORIENTATION_LIBM_BOUND,
                "component {component}: {got} differs from {want} beyond the libm bound"
            );
        }
    }

    #[test]
    fn velocity_orientation_matches_blender_stored_rotation_within_libm_bound() {
        assert_within_libm_bound(velocity_orientation(VELOCITY_A), ROTATION_A);
        assert_within_libm_bound(velocity_orientation(VELOCITY_B), ROTATION_B);
    }

    #[test]
    fn quat_to_xyzw_reorders_blender_wxyz_to_potter_xyzw() {
        assert_eq!(
            quat_to_xyzw(ROTATION_A),
            [
                f64::from(ROTATION_A[1]),
                f64::from(ROTATION_A[2]),
                f64::from(ROTATION_A[3]),
                f64::from(ROTATION_A[0]),
            ]
        );
    }
}
