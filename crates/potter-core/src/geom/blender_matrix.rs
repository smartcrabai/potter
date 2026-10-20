//! Blender's f32 operand transform for EXACT Boolean operands.
//!
//! Blender computes an operand's transform into the subject's space as
//! `world_to_object(subject) * object_to_world(operand)` in f32 (`MOD_boolean.cc:396-440`). This
//! module replays that f32 pipeline bit for bit. It applies only where the evaluated world matrix
//! is exactly the composition of the node's own transform, parent and parent inverse; every other
//! case returns `None` and the caller keeps its f64 path.
//!
//! Verified against Blender 5.2.2 (arm64): object matrices, subject inverses and products match
//! bit for bit on random and regression scenes.

use std::collections::BTreeMap;

use glam::{DMat4, DQuat, EulerRot};

use crate::model::{Id, ParentType, SceneDoc, Transform};

/// Blender's `float[4][4]` object matrix as stored: `matrix[column][row]`.
type BlenderMat4 = [[f32; 4]; 4];

/// Blender's `float[3][3]` matrix as stored: `matrix[column][row]`.
type BlenderMat3 = [[f32; 3]; 3];

/// Blender's rotation orders, with the axis table and parity of `rotOrders` in
/// `source/blender/blenlib/intern/math_rotation_c.cc`.
#[derive(Clone, Copy)]
enum BlenderEulerOrder {
    Xyz,
    Xzy,
    Yxz,
    Yzx,
    Zxy,
    Zyx,
}

impl BlenderEulerOrder {
    fn from_mode(mode: &str) -> Option<Self> {
        Some(match mode {
            "XYZ" => Self::Xyz,
            "XZY" => Self::Xzy,
            "YXZ" => Self::Yxz,
            "YZX" => Self::Yzx,
            "ZXY" => Self::Zxy,
            "ZYX" => Self::Zyx,
            _ => return None,
        })
    }

    /// Axis indices `i, j, k` and the parity flag of the order.
    fn axes(self) -> ([usize; 3], bool) {
        match self {
            Self::Xyz => ([0, 1, 2], false),
            Self::Xzy => ([0, 2, 1], true),
            Self::Yxz => ([1, 0, 2], true),
            Self::Yzx => ([1, 2, 0], false),
            Self::Zxy => ([2, 0, 1], false),
            Self::Zyx => ([2, 1, 0], true),
        }
    }

    /// Blender's angles recovered from a quaternion. `exchange/blend.rs` (`blender_rotation`)
    /// builds the stored quaternion with glam's extrinsic order and the angles permuted: glam
    /// angle `t` holds Blender angle `slots[t]`. Undoing that permutation restores Blender's
    /// angles.
    fn angles_from_quaternion(self, quaternion: DQuat) -> [f32; 3] {
        let (order, slots): (EulerRot, [usize; 3]) = match self {
            Self::Xyz => (EulerRot::XYZEx, [0, 1, 2]),
            Self::Xzy => (EulerRot::XZYEx, [0, 2, 1]),
            Self::Yxz => (EulerRot::YXZEx, [1, 0, 2]),
            Self::Yzx => (EulerRot::YZXEx, [1, 2, 0]),
            Self::Zxy => (EulerRot::ZXYEx, [2, 0, 1]),
            Self::Zyx => (EulerRot::ZYXEx, [2, 1, 0]),
        };
        let (first, second, third) = quaternion.to_euler(order);
        let mut angles = [0.0_f32; 3];
        for (slot, value) in slots.into_iter().zip([first, second, third]) {
            angles[slot] = value as f32;
        }
        angles
    }
}

/// A Blender object rotation: Euler angles in their order, or a quaternion.
#[derive(Clone, Copy)]
enum BlenderRotation {
    Euler(BlenderEulerOrder, [f32; 3]),
    Quaternion([f32; 4]),
}

/// Blender's `eulO_to_mat3` (`math_rotation_c.cc:1710`): double-precision sines and cosines, each
/// element rounded to f32.
fn blender_euler_to_mat3(angles: [f32; 3], order: BlenderEulerOrder) -> BlenderMat3 {
    let ([i, j, k], parity) = order.axes();
    let sign = if parity { -1.0 } else { 1.0 };
    let ti = sign * f64::from(angles[i]);
    let tj = sign * f64::from(angles[j]);
    let th = sign * f64::from(angles[k]);
    let (ci, cj, ch) = (ti.cos(), tj.cos(), th.cos());
    let (si, sj, sh) = (ti.sin(), tj.sin(), th.sin());
    let cc = ci * ch;
    let cs = ci * sh;
    let sc = si * ch;
    let ss = si * sh;
    let mut matrix = [[0.0_f32; 3]; 3];
    matrix[i][i] = (cj * ch) as f32;
    matrix[j][i] = (sj * sc - cs) as f32;
    matrix[k][i] = (sj * cc + ss) as f32;
    matrix[i][j] = (cj * sh) as f32;
    matrix[j][j] = (sj * ss + cc) as f32;
    matrix[k][j] = (sj * cs - sc) as f32;
    matrix[i][k] = (-sj) as f32;
    matrix[j][k] = (cj * si) as f32;
    matrix[k][k] = (cj * ci) as f32;
    matrix
}

/// Blender's `normalize_qt_qt` (`math_rotation_c.cc:472`, `dot_qtqt` at `:110`) and `quat_to_mat3`
/// (`math_rotation_c.cc:198`): the quaternion is normalized in f32, then converted in double
/// precision and rounded to f32.
fn blender_quat_to_mat3(quaternion: [f32; 4]) -> BlenderMat3 {
    let dot = quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3];
    let length = dot.sqrt();
    let q = if length == 0.0 {
        [0.0, 1.0, 0.0, 0.0]
    } else {
        let inverse = 1.0 / length;
        quaternion.map(|value| value * inverse)
    };
    let root2 = std::f64::consts::SQRT_2;
    let q0 = root2 * f64::from(q[0]);
    let q1 = root2 * f64::from(q[1]);
    let q2 = root2 * f64::from(q[2]);
    let q3 = root2 * f64::from(q[3]);
    let qda = q0 * q1;
    let qdb = q0 * q2;
    let qdc = q0 * q3;
    let qaa = q1 * q1;
    let qab = q1 * q2;
    let qac = q1 * q3;
    let qbb = q2 * q2;
    let qbc = q2 * q3;
    let qcc = q3 * q3;
    let mut matrix = [[0.0_f32; 3]; 3];
    matrix[0][0] = (1.0 - qbb - qcc) as f32;
    matrix[0][1] = (qdc + qab) as f32;
    matrix[0][2] = (-qdb + qac) as f32;
    matrix[1][0] = (-qdc + qab) as f32;
    matrix[1][1] = (1.0 - qaa - qcc) as f32;
    matrix[1][2] = (qda + qbc) as f32;
    matrix[2][0] = (qdb + qac) as f32;
    matrix[2][1] = (-qda + qbc) as f32;
    matrix[2][2] = (1.0 - qaa - qbb) as f32;
    matrix
}

/// Blender `mul_m3_m3m3` (`math_matrix_c.cc:319`): `result[j][k]` sums `b[j][0] * a[0][k]`,
/// `b[j][1] * a[1][k]` and `b[j][2] * a[2][k]` in index order.
fn blender_mul_m3(a: &BlenderMat3, b: &BlenderMat3) -> BlenderMat3 {
    let mut result = [[0.0_f32; 3]; 3];
    for (j, column) in result.iter_mut().enumerate() {
        for (k, value) in column.iter_mut().enumerate() {
            *value = b[j][0] * a[0][k] + b[j][1] * a[1][k] + b[j][2] * a[2][k];
        }
    }
    result
}

/// Blender `mul_m4_m4m4` (`math_matrix_c.cc:217`, scalar branch) and `float4x4 operator*`
/// (`blenlib/intern/math_matrix.cc:27`): `result[j][k]` sums `b[j][0] * a[0][k]`, `b[j][1] * a[1][k]`,
/// `b[j][2] * a[2][k]` and `b[j][3] * a[3][k]`. The four products are summed in pairs,
/// `(p0 + p1) + (p2 + p3)`: the association the Blender arm64 build produces, checked bit for bit
/// against the parent-inverse products of `BKE_object_where_is_calc`.
pub(crate) fn blender_mul_m4(a: &BlenderMat4, b: &BlenderMat4) -> BlenderMat4 {
    let mut result = [[0.0_f32; 4]; 4];
    for (j, column) in result.iter_mut().enumerate() {
        for (k, value) in column.iter_mut().enumerate() {
            *value =
                (b[j][0] * a[0][k] + b[j][1] * a[1][k]) + (b[j][2] * a[2][k] + b[j][3] * a[3][k]);
        }
    }
    result
}

/// Blender's `BKE_object_to_mat4` (`object.cc:3121`) for an object without delta transforms: the
/// rotation (`BKE_object_rot_to_mat3`, `object.cc:2902`) times the scale (`size_to_mat3`, as in
/// `BKE_object_scale_to_mat3`, `object.cc:2895`), with the location as the fourth column. The
/// delta rotation and delta scale are identities, so their products are exact and omitted.
fn blender_object_matrix(
    translation: [f64; 3],
    rotation: BlenderRotation,
    scale: [f64; 3],
) -> BlenderMat4 {
    let rotation = match rotation {
        BlenderRotation::Euler(order, angles) => blender_euler_to_mat3(angles, order),
        BlenderRotation::Quaternion(quaternion) => blender_quat_to_mat3(quaternion),
    };
    let size = [
        [scale[0] as f32, 0.0, 0.0],
        [0.0, scale[1] as f32, 0.0],
        [0.0, 0.0, scale[2] as f32],
    ];
    let linear = blender_mul_m3(&rotation, &size);
    let mut matrix = [[0.0_f32; 4]; 4];
    for (column, values) in linear.iter().enumerate() {
        matrix[column][..3].copy_from_slice(values);
    }
    for (row, value) in translation.iter().enumerate() {
        matrix[3][row] = *value as f32;
    }
    matrix[3][3] = 1.0;
    matrix
}

/// The rotation of `transform` as Blender stores it. Axis-angle has no replica here: `None`.
fn blender_rotation(transform: &Transform) -> Option<BlenderRotation> {
    let mode = transform.rotation_mode.to_ascii_uppercase();
    if mode == "QUATERNION" {
        return Some(BlenderRotation::Quaternion(
            transform.rotation.map(|value| value as f32),
        ));
    }
    let order = BlenderEulerOrder::from_mode(&mode)?;
    let angles = order.angles_from_quaternion(transform.rotation_quat());
    Some(BlenderRotation::Euler(order, angles))
}

/// The f32 object matrix of a node's own transform. It must agree with the f64 transform within
/// f32 rounding, which rejects a replica that describes a different transform (`None`).
fn blender_transform_matrix(transform: &Transform) -> Option<BlenderMat4> {
    let rotation = blender_rotation(transform)?;
    let matrix = blender_object_matrix(transform.translation, rotation, transform.scale);
    same_matrix(dmat4_from_blender(&matrix), transform.matrix(), 1e-5).then_some(matrix)
}

/// Blender's f32 `object_to_world` of a Boolean subject or operand. `BKE_object_where_is_calc`
/// (`object.cc:3445-3459`) composes the parent matrix, the parent inverse and the object's own
/// matrix in f32, in that order. The replica is used only where the evaluated world matrix is the
/// f64 composition of the same parts (no animation, drivers, constraints or bone parents) and the
/// f32 replica agrees with the f64 transform. Otherwise `None`.
fn object_to_world(
    doc: &SceneDoc,
    node_id: &Id,
    world_matrices: &BTreeMap<Id, DMat4>,
    depth: usize,
) -> Option<BlenderMat4> {
    const MAX_PARENT_DEPTH: usize = 64;
    let node = doc.nodes.get(node_id)?;
    if depth > MAX_PARENT_DEPTH || node.parent_type != ParentType::Object {
        return None;
    }
    let parent_inverse = node
        .parent_inverse
        .map_or(DMat4::IDENTITY, |values| DMat4::from_cols_array(&values));
    let world = *world_matrices.get(node_id)?;
    let parent_space = match &node.parent {
        Some(parent_id) => Some(*world_matrices.get(parent_id)?),
        None => None,
    };
    // A current captured `matrix_basis` is the local matrix the rig composes (`rig.rs`), in f32.
    // Without it the local matrix is the f64 TRS, compared at f64 precision.
    let captured_basis = node.captured_matrix_basis(&node.transform).ok()?;
    let (expected, own, captured) = if let Some(basis) = captured_basis {
        let local = blender_matrix_from_f64(DMat4::from_cols_array(&basis));
        let composed = match parent_space {
            Some(parent) => blender_mul_m4(
                &blender_mul_m4(
                    &blender_matrix_from_f64(parent),
                    &blender_matrix_from_f64(parent_inverse),
                ),
                &local,
            ),
            None => local,
        };
        (dmat4_from_blender(&composed), local, true)
    } else {
        let local = node.transform.matrix();
        let expected = match parent_space {
            Some(parent) => parent * parent_inverse * local,
            None => parent_inverse * local,
        };
        if !same_matrix(expected, world, 1e-9) {
            return None;
        }
        (world, blender_transform_matrix(&node.transform)?, false)
    };
    // The captured composition is the rig's own f32 arithmetic on the same operands, so it must be
    // bit-identical to the evaluated world matrix; any constraint or animation breaks that.
    if !same_matrix(expected, world, 0.0) {
        return None;
    }
    match &node.parent {
        Some(parent_id) => {
            let parent = object_to_world(doc, parent_id, world_matrices, depth + 1)?;
            // `mul_m4_m4m4(tmat, totmat, ob->parentinv)`, then `mul_m4_m4m4(r_obmat, tmat, locmat)`.
            let inverse = blender_matrix_from_f64(parent_inverse);
            Some(blender_mul_m4(&blender_mul_m4(&parent, &inverse), &own))
        }
        // Blender ignores the parent inverse of an unparented object, so keep the f64 path if one
        // is set. A captured root is composed without it, as the rig does.
        None => (parent_inverse == DMat4::IDENTITY || captured).then_some(own),
    }
}

unsafe extern "C" {
    /// Writes `inverse` and returns `true` when `matrix` is invertible; returns `false` otherwise.
    fn potter_blender_inverse_m4(matrix: *const f32, inverse: *mut f32) -> bool;
}

/// Blender `invert_m4_m4` (`math_matrix_c.cc:1150`) through `EIG_invert_m4_m4`
/// (`intern/eigen/intern/matrix.cc:29`): the native Eigen `Matrix4f` inverse of
/// `native/blender_inverse_m4.cc`, built for the target so its packets match Blender's build on that
/// target. `None` when Eigen's threshold-zero invertibility test fails.
pub(crate) fn blender_invert_m4(matrix: &[[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inverse = [[0.0; 4]; 4];
    // SAFETY: both pointers address 16 contiguous `f32`s in column-major order, the layout of
    // Eigen's `Matrix4f`. The native call writes `inverse` only when it returns `true`.
    let invertible =
        unsafe { potter_blender_inverse_m4(matrix.as_ptr().cast(), inverse.as_mut_ptr().cast()) };
    invertible.then_some(inverse)
}

/// Quantizes an f64 matrix to Blender's f32 `float[4][4]`.
pub(crate) fn blender_matrix_from_f64(matrix: DMat4) -> BlenderMat4 {
    std::array::from_fn(|column| std::array::from_fn(|row| matrix.col(column)[row] as f32))
}

/// Blender's f32 matrix as an f64 matrix; every entry is exact.
pub(crate) fn dmat4_from_blender(matrix: &BlenderMat4) -> DMat4 {
    let columns: [f64; 16] = std::array::from_fn(|index| f64::from(matrix[index / 4][index % 4]));
    DMat4::from_cols_array(&columns)
}

/// Whether two matrices agree entry by entry within `tolerance`, relative to the larger entry and
/// absolute below 1.
fn same_matrix(a: DMat4, b: DMat4, tolerance: f64) -> bool {
    a.to_cols_array()
        .into_iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs())))
}

/// Blender's f32 operand transform into the subject's space: `world_to_object(subject) *
/// object_to_world(operand)` (`MOD_boolean.cc:440`, `non_float_boolean_mesh`), returned as a
/// `DMat4` whose entries are exact f32 values. `None` when the subject or the operand has no exact
/// replica, or the subject is not invertible; the caller then keeps its f64 path.
pub(crate) fn operand_to_subject(
    doc: &SceneDoc,
    subject_id: &Id,
    object_id: &Id,
    world_matrices: &BTreeMap<Id, DMat4>,
) -> Option<DMat4> {
    let subject = object_to_world(doc, subject_id, world_matrices, 0)?;
    let inverse = blender_invert_m4(&subject)?;
    let object = object_to_world(doc, object_id, world_matrices, 0)?;
    Some(dmat4_from_blender(&blender_mul_m4(&inverse, &object)))
}

/// Blender `is_negative_m4`: sign of the upper 3x3 determinant after the f32 matrix boundary.
pub(crate) fn is_negative_m4(matrix: DMat4) -> bool {
    let matrix = matrix.to_cols_array().map(|value| value as f32);
    let determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[4] * (matrix[1] * matrix[10] - matrix[2] * matrix[9])
        + matrix[8] * (matrix[1] * matrix[6] - matrix[2] * matrix[5]);
    determinant < 0.0
}

/// Blender `mul_m4_v3` on an f32 `float[4][4]` stored column-major: transforms `point` as a
/// position, with the translation in the fourth column.
pub(crate) fn blender_transform_point_f32(transform: &[f32; 16], point: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::new(
        point.x * transform[0] + point.y * transform[4] + transform[8] * point.z + transform[12],
        point.x * transform[1] + point.y * transform[5] + transform[9] * point.z + transform[13],
        point.x * transform[2] + point.y * transform[6] + transform[10] * point.z + transform[14],
    )
}
