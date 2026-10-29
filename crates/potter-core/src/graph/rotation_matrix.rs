use serde_json::{Map, Value, json};

use crate::error::{PotError, Result};

type Vec3 = [f32; 3];
type Quat = [f32; 4];
type Mat3 = [[f32; 3]; 3];
type Mat4 = [[f32; 4]; 4];

const IDENTITY3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const IDENTITY4: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const EULER_C_GIMBAL_EPSILON: f32 = 0.0000375;
const EULER_MATH_GIMBAL_EPSILON: f32 = 16.0 * f32::EPSILON;

fn invalid(message: impl Into<String>) -> PotError {
    PotError::invalid_argument(message)
}

fn normalized_eq(a: &str, b: &str) -> bool {
    a.bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(|byte| byte.to_ascii_lowercase())
        .eq(b
            .bytes()
            .filter(|byte| byte.is_ascii_alphanumeric())
            .map(|byte| byte.to_ascii_lowercase()))
}

fn input<'a>(inputs: &'a [(&str, Value)], name: &str, aliases: &[&str]) -> Result<&'a Value> {
    for candidate in std::iter::once(name).chain(aliases.iter().copied()) {
        if let Some((_, value)) = inputs
            .iter()
            .find(|(input_name, _)| *input_name == candidate)
        {
            return Ok(value);
        }
    }
    inputs
        .iter()
        .find(|(candidate, _)| {
            std::iter::once(name)
                .chain(aliases.iter().copied())
                .any(|expected| normalized_eq(candidate, expected))
        })
        .map(|(_, value)| value)
        .ok_or_else(|| invalid(format!("missing input socket {name:?}")))
}

fn property<'a>(properties: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    properties.get(name).or_else(|| {
        properties
            .iter()
            .find(|(candidate, _)| normalized_eq(candidate, name))
            .map(|(_, value)| value)
    })
}

fn number(value: &Value, name: &str) -> Result<f32> {
    let value = value
        .as_f64()
        .ok_or_else(|| invalid(format!("{name} must be a JSON number")))?;
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid(format!(
            "{name} must be finite and representable as float32"
        )));
    }
    Ok(value)
}

fn number_input(inputs: &[(&str, Value)], name: &str) -> Result<f32> {
    number(input(inputs, name, &[])?, name)
}

fn vector3(value: &Value, name: &str) -> Result<Vec3> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 3)
        .ok_or_else(|| invalid(format!("{name} must be a length-3 JSON array")))?;
    Ok([
        number(&components[0], name)?,
        number(&components[1], name)?,
        number(&components[2], name)?,
    ])
}

fn vector_input(inputs: &[(&str, Value)], name: &str) -> Result<Vec3> {
    vector3(input(inputs, name, &[])?, name)
}

fn quaternion(value: &Value, name: &str) -> Result<Quat> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)
        .ok_or_else(|| invalid(format!("{name} must be a length-4 xyzw JSON array")))?;
    let xyzw = [
        number(&components[0], name)?,
        number(&components[1], name)?,
        number(&components[2], name)?,
        number(&components[3], name)?,
    ];
    normalize_quaternion([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], name)
}

fn matrix4(value: &Value, name: &str) -> Result<Mat4> {
    if value.is_null() {
        return Ok(IDENTITY4);
    }
    let rows = value
        .as_array()
        .filter(|rows| rows.len() == 4)
        .ok_or_else(|| invalid(format!("{name} must be a nested 4x4 JSON array or null")))?;
    let mut matrix = [[0.0; 4]; 4];
    for (row_index, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .filter(|row| row.len() == 4)
            .ok_or_else(|| invalid(format!("{name} row {row_index} must have four values")))?;
        for (column_index, value) in row.iter().enumerate() {
            matrix[row_index][column_index] = number(value, name)?;
        }
    }
    Ok(matrix)
}

fn matrix_input(inputs: &[(&str, Value)], name: &str, aliases: &[&str]) -> Result<Mat4> {
    matrix4(input(inputs, name, aliases)?, name)
}

fn output_is(output_socket: &str, expected: &str) -> bool {
    normalized_eq(output_socket, expected)
}

fn require_output(output_socket: &str, expected: &[&str]) -> Result<()> {
    if expected.iter().any(|name| output_is(output_socket, name)) {
        Ok(())
    } else {
        Err(invalid(format!("unknown output socket {output_socket:?}")))
    }
}

fn finite_result(value: f32, name: &str) -> Result<Value> {
    if value.is_finite() {
        Ok(json!(value))
    } else {
        Err(invalid(format!("{name} result is not finite")))
    }
}

fn vector_result(value: Vec3, name: &str) -> Result<Value> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(json!(value))
    } else {
        Err(invalid(format!("{name} result is not finite")))
    }
}

fn quaternion_result(value: Quat, name: &str) -> Result<Value> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(json!([value[1], value[2], value[3], value[0]]))
    } else {
        Err(invalid(format!("{name} result is not finite")))
    }
}

fn matrix_result(value: Mat4, name: &str) -> Result<Value> {
    if value
        .iter()
        .flatten()
        .all(|component| component.is_finite())
    {
        Ok(json!(value))
    } else {
        Err(invalid(format!("{name} result is not finite")))
    }
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn cross_high_precision(a: Vec3, b: Vec3) -> Vec3 {
    [
        (f64::from(a[1]) * f64::from(b[2]) - f64::from(a[2]) * f64::from(b[1])) as f32,
        (f64::from(a[2]) * f64::from(b[0]) - f64::from(a[0]) * f64::from(b[2])) as f32,
        (f64::from(a[0]) * f64::from(b[1]) - f64::from(a[1]) * f64::from(b[0])) as f32,
    ]
}

fn normalized_angle(a: Vec3, b: Vec3) -> f32 {
    if dot(a, b) >= 0.0 {
        2.0 * (length([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) * 0.5)
            .clamp(0.0, 1.0)
            .asin()
    } else {
        let opposite_b = [-b[0], -b[1], -b[2]];
        std::f32::consts::PI
            - 2.0
                * (length([
                    a[0] - opposite_b[0],
                    a[1] - opposite_b[1],
                    a[2] - opposite_b[2],
                ]) * 0.5)
                    .clamp(0.0, 1.0)
                    .asin()
    }
}

fn length(value: Vec3) -> f32 {
    dot(value, value).sqrt()
}

fn normalize_vector(value: Vec3) -> Vec3 {
    let length = length(value);
    if length == 0.0 {
        [0.0; 3]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn normalize_quaternion(value: Quat, name: &str) -> Result<Quat> {
    let length_squared = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>();
    if length_squared == 0.0 {
        return Ok([1.0, 0.0, 0.0, 0.0]);
    }
    let length = length_squared.sqrt();
    if !length.is_finite() {
        return Err(invalid(format!("{name} quaternion cannot be normalized")));
    }
    Ok(value.map(|component| component / length))
}

fn euler_quaternion(euler: Vec3) -> Result<Quat> {
    let [x, y, z] = euler.map(|angle| angle * 0.5);
    let (si, ci) = x.sin_cos();
    let (sj, cj) = y.sin_cos();
    let (sh, ch) = z.sin_cos();
    let cc = ci * ch;
    let cs = ci * sh;
    let sc = si * ch;
    let ss = si * sh;
    normalize_quaternion(
        [
            cj * cc + sj * ss,
            cj * sc - sj * cs,
            cj * ss + sj * cc,
            cj * cs - sj * sc,
        ],
        "Euler rotation",
    )
}

fn decode_rotation(value: &Value, name: &str) -> Result<Quat> {
    let values = value.as_array().ok_or_else(|| {
        invalid(format!(
            "{name} must be a length-3 Euler or length-4 quaternion array"
        ))
    })?;
    match values.len() {
        3 => euler_quaternion(vector3(value, name)?),
        4 => quaternion(value, name),
        _ => Err(invalid(format!(
            "{name} must be a length-3 Euler or length-4 quaternion array"
        ))),
    }
}

fn rotation_input(inputs: &[(&str, Value)], name: &str) -> Result<Quat> {
    decode_rotation(input(inputs, name, &[])?, name)
}

fn quaternion_multiply(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
    ]
}

fn quaternion_matrix3(quaternion: Quat) -> Mat3 {
    let [w, x, y, z] = quaternion;
    let q0 = std::f64::consts::SQRT_2 * f64::from(w);
    let q1 = std::f64::consts::SQRT_2 * f64::from(x);
    let q2 = std::f64::consts::SQRT_2 * f64::from(y);
    let q3 = std::f64::consts::SQRT_2 * f64::from(z);
    let qda = q0 * q1;
    let qdb = q0 * q2;
    let qdc = q0 * q3;
    let qaa = q1 * q1;
    let qab = q1 * q2;
    let qac = q1 * q3;
    let qbb = q2 * q2;
    let qbc = q2 * q3;
    let qcc = q3 * q3;
    [
        [
            (1.0 - qbb - qcc) as f32,
            (-qdc + qab) as f32,
            (qdb + qac) as f32,
        ],
        [
            (qdc + qab) as f32,
            (1.0 - qaa - qcc) as f32,
            (-qda + qbc) as f32,
        ],
        [
            (-qdb + qac) as f32,
            (qda + qbc) as f32,
            (1.0 - qaa - qbb) as f32,
        ],
    ]
}

fn euler_matrix3(euler: Vec3) -> Mat3 {
    let [x, y, z] = euler.map(f64::from);
    let ci = x.cos();
    let cj = y.cos();
    let ch = z.cos();
    let si = x.sin();
    let sj = y.sin();
    let sh = z.sin();
    let cc = ci * ch;
    let cs = ci * sh;
    let sc = si * ch;
    let ss = si * sh;
    [
        [
            (cj * ch) as f32,
            (sj * sc - cs) as f32,
            (sj * cc + ss) as f32,
        ],
        [
            (cj * sh) as f32,
            (sj * ss + cc) as f32,
            (sj * cs - sc) as f32,
        ],
        [(-sj) as f32, (cj * si) as f32, (cj * ci) as f32],
    ]
}

fn axis_angle_matrix3(axis: Vec3, angle: f32) -> Mat3 {
    let axis = normalize_vector(axis);
    if axis == [0.0; 3] {
        return IDENTITY3;
    }
    let [x, y, z] = axis;
    let sine = angle.sin();
    let cosine = angle.cos();
    let one_minus_cosine = 1.0 - cosine;
    let nsi = [x * sine, y * sine, z * sine];
    let n00 = (x * x) * one_minus_cosine;
    let n01 = (x * y) * one_minus_cosine;
    let n11 = (y * y) * one_minus_cosine;
    let n02 = (x * z) * one_minus_cosine;
    let n12 = (y * z) * one_minus_cosine;
    let n22 = (z * z) * one_minus_cosine;
    [
        [n00 + cosine, n01 - nsi[2], n02 + nsi[1]],
        [n01 + nsi[2], n11 + cosine, n12 - nsi[0]],
        [n02 - nsi[1], n12 + nsi[0], n22 + cosine],
    ]
}

fn mat3_multiply(a: Mat3, b: Mat3) -> Mat3 {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] =
                a[row][0] * b[0][column] + a[row][1] * b[1][column] + a[row][2] * b[2][column];
        }
    }
    result
}

fn mat3_vector(matrix: Mat3, vector: Vec3) -> Vec3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn normalize_matrix3_columns(matrix: Mat3) -> (Mat3, Vec3) {
    let mut normalized = matrix;
    let mut scales = [0.0; 3];
    for column in 0..3 {
        scales[column] = (matrix[0][column] * matrix[0][column]
            + matrix[1][column] * matrix[1][column]
            + matrix[2][column] * matrix[2][column])
            .sqrt();
        if scales[column] != 0.0 {
            for row in 0..3 {
                normalized[row][column] /= scales[column];
            }
        }
    }
    (normalized, scales)
}

fn matrix3_euler(matrix: Mat3, gimbal_epsilon: f32) -> Vec3 {
    let (matrix, _) = normalize_matrix3_columns(matrix);
    let cy = matrix[0][0].hypot(matrix[1][0]);
    let (first, second) = if cy > gimbal_epsilon {
        (
            [
                matrix[2][1].atan2(matrix[2][2]),
                (-matrix[2][0]).atan2(cy),
                matrix[1][0].atan2(matrix[0][0]),
            ],
            [
                (-matrix[2][1]).atan2(-matrix[2][2]),
                (-matrix[2][0]).atan2(-cy),
                (-matrix[1][0]).atan2(-matrix[0][0]),
            ],
        )
    } else {
        let euler = [
            (-matrix[1][2]).atan2(matrix[1][1]),
            (-matrix[2][0]).atan2(cy),
            0.0,
        ];
        (euler, euler)
    };
    let first_length = first.iter().map(|value| value.abs()).sum::<f32>();
    let second_length = second.iter().map(|value| value.abs()).sum::<f32>();
    if first_length > second_length {
        second
    } else {
        first
    }
}

fn matrix3_quaternion(matrix: Mat3) -> Result<Quat> {
    let determinant = determinant3(matrix);
    if !determinant.is_finite() {
        return Ok([1.0, 0.0, 0.0, 0.0]);
    }
    let matrix = if determinant < 0.0 {
        matrix.map(|row| row.map(|component| -component))
    } else {
        matrix
    };
    let [m00, m01, m02] = matrix[0];
    let [m10, m11, m12] = matrix[1];
    let [m20, m21, m22] = matrix[2];
    let mut q = if m22 < 0.0 {
        if m00 > m11 {
            let trace = 1.0 + m00 - m11 - m22;
            let mut s = 2.0 * trace.sqrt();
            if m12 > m21 {
                s = -s;
            }
            let mut q = [0.0; 4];
            q[1] = 0.25 * s;
            let reciprocal = 1.0 / s;
            q[0] = (m21 - m12) * reciprocal;
            q[2] = (m01 + m10) * reciprocal;
            q[3] = (m02 + m20) * reciprocal;
            if trace == 1.0 && q[0] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
                q[1] = 1.0;
            }
            q
        } else {
            let trace = 1.0 - m00 + m11 - m22;
            let mut s = 2.0 * trace.sqrt();
            if m20 > m02 {
                s = -s;
            }
            let mut q = [0.0; 4];
            q[2] = 0.25 * s;
            let reciprocal = 1.0 / s;
            q[0] = (m02 - m20) * reciprocal;
            q[1] = (m01 + m10) * reciprocal;
            q[3] = (m12 + m21) * reciprocal;
            if trace == 1.0 && q[0] == 0.0 && q[1] == 0.0 && q[3] == 0.0 {
                q[2] = 1.0;
            }
            q
        }
    } else if m00 < -m11 {
        let trace = 1.0 - m00 - m11 + m22;
        let mut s = 2.0 * trace.sqrt();
        if m01 > m10 {
            s = -s;
        }
        let mut q = [0.0; 4];
        q[3] = 0.25 * s;
        let reciprocal = 1.0 / s;
        q[0] = (m10 - m01) * reciprocal;
        q[1] = (m02 + m20) * reciprocal;
        q[2] = (m12 + m21) * reciprocal;
        if trace == 1.0 && q[0] == 0.0 && q[1] == 0.0 && q[2] == 0.0 {
            q[3] = 1.0;
        }
        q
    } else {
        let trace = 1.0 + m00 + m11 + m22;
        let s = 2.0 * trace.sqrt();
        let mut q = [0.0; 4];
        q[0] = 0.25 * s;
        let reciprocal = 1.0 / s;
        q[1] = (m21 - m12) * reciprocal;
        q[2] = (m02 - m20) * reciprocal;
        q[3] = (m10 - m01) * reciprocal;
        if trace == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
            q[0] = 1.0;
        }
        q
    };
    if q[0] < 0.0 {
        q = q.map(|component| -component);
    }
    let length_squared = q.iter().map(|component| component * component).sum::<f32>();
    if (length_squared - 1.0).abs() >= 0.0002 {
        let reciprocal = 1.0 / length_squared.sqrt();
        q = q.map(|component| component * reciprocal);
    }
    normalize_quaternion(q, "matrix rotation")
}

fn determinant3(matrix: Mat3) -> f32 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn axis_vector(axis: usize) -> Vec3 {
    let mut vector = [0.0; 3];
    vector[axis] = 1.0;
    vector
}

fn orthogonal_vector(vector: Vec3) -> Vec3 {
    if vector[0] != -vector[1] {
        [-vector[1], vector[0], 0.0]
    } else if vector[0] != -vector[2] {
        [-vector[2], 0.0, vector[0]]
    } else {
        [0.0, -vector[2], vector[1]]
    }
}

fn axis_property(properties: &Map<String, Value>, name: &str, default: usize) -> Result<usize> {
    let Some(value) = property(properties, name) else {
        return Ok(default);
    };
    if let Some(value) = value.as_str() {
        match value.to_ascii_uppercase().as_str() {
            "X" => Ok(0),
            "Y" => Ok(1),
            "Z" => Ok(2),
            _ => Err(invalid(format!(
                "property {name:?} has unknown axis {value:?}"
            ))),
        }
    } else if let Some(value) = value.as_i64() {
        usize::try_from(value)
            .ok()
            .filter(|value| *value < 3)
            .ok_or_else(|| invalid(format!("property {name:?} must be axis X, Y, or Z")))
    } else {
        Err(invalid(format!(
            "property {name:?} must be an axis name or integer"
        )))
    }
}

fn pivot_property(properties: &Map<String, Value>, default: usize) -> Result<usize> {
    let Some(value) = property(properties, "pivot_axis") else {
        return Ok(default);
    };
    if let Some(value) = value.as_str() {
        match value.to_ascii_uppercase().as_str() {
            "AUTO" => Ok(0),
            "X" => Ok(1),
            "Y" => Ok(2),
            "Z" => Ok(3),
            _ => Err(invalid(format!(
                "property \"pivot_axis\" has unknown value {value:?}"
            ))),
        }
    } else if let Some(value) = value.as_i64() {
        usize::try_from(value)
            .ok()
            .filter(|value| *value < 4)
            .ok_or_else(|| invalid("property \"pivot_axis\" must be AUTO, X, Y, or Z"))
    } else {
        Err(invalid(
            "property \"pivot_axis\" must be an enum name or integer",
        ))
    }
}

fn enum_property(
    properties: &Map<String, Value>,
    name: &str,
    default: usize,
    names: &[&str],
) -> Result<usize> {
    let Some(value) = property(properties, name) else {
        return Ok(default);
    };
    if let Some(value) = value.as_str() {
        let value = value.to_ascii_uppercase();
        names
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(&value))
            .ok_or_else(|| invalid(format!("property {name:?} has unknown value {value:?}")))
    } else if let Some(value) = value.as_i64() {
        usize::try_from(value)
            .ok()
            .filter(|value| *value < names.len())
            .ok_or_else(|| invalid(format!("property {name:?} is outside its enum range")))
    } else {
        Err(invalid(format!(
            "property {name:?} must be an enum name or integer"
        )))
    }
}

fn matrix3_from4(matrix: Mat4) -> Mat3 {
    [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ]
}

fn matrix4_from3(matrix: Mat3) -> Mat4 {
    let mut result = IDENTITY4;
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = matrix[row][column];
        }
    }
    result
}

fn matrix_multiply(a: Mat4, b: Mat4) -> Mat4 {
    let mut result = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = a[row][0] * b[0][column]
                + a[row][1] * b[1][column]
                + a[row][2] * b[2][column]
                + a[row][3] * b[3][column];
        }
    }
    result
}

fn matrix_determinant(matrix: Mat4) -> f32 {
    let mut work = matrix;
    let mut determinant = 1.0;
    for column in 0..4 {
        let pivot =
            (column..4).max_by(|a, b| work[*a][column].abs().total_cmp(&work[*b][column].abs()));
        let Some(pivot) = pivot else {
            return 0.0;
        };
        if work[pivot][column] == 0.0 {
            return 0.0;
        }
        if pivot != column {
            work.swap(pivot, column);
            determinant = -determinant;
        }
        let diagonal = work[column][column];
        determinant *= diagonal;
        for row in (column + 1)..4 {
            let factor = work[row][column] / diagonal;
            for col in (column + 1)..4 {
                work[row][col] -= factor * work[column][col];
            }
        }
    }
    determinant
}

fn matrix_inverse(matrix: Mat4) -> Option<Mat4> {
    let mut left = matrix;
    let mut right = IDENTITY4;
    for column in 0..4 {
        let pivot =
            (column..4).max_by(|a, b| left[*a][column].abs().total_cmp(&left[*b][column].abs()))?;
        if left[pivot][column] == 0.0 {
            return None;
        }
        left.swap(pivot, column);
        right.swap(pivot, column);
        let diagonal = left[column][column];
        for col in 0..4 {
            left[column][col] /= diagonal;
            right[column][col] /= diagonal;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let factor = left[row][column];
            for col in 0..4 {
                left[row][col] -= factor * left[column][col];
                right[row][col] -= factor * right[column][col];
            }
        }
    }
    Some(right)
}

fn matrix_component(name: &str) -> Option<(usize, usize)> {
    let mut characters = name
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(|byte| byte.to_ascii_lowercase());
    for expected in b"column" {
        if characters.next()? != *expected {
            return None;
        }
    }
    let column = characters.next()?.checked_sub(b'1')?;
    if column >= 4 {
        return None;
    }
    for expected in b"row" {
        if characters.next()? != *expected {
            return None;
        }
    }
    let row = characters.next()?.checked_sub(b'1')?;
    if row >= 4 || characters.next().is_some() {
        return None;
    }
    Some((usize::from(column), usize::from(row)))
}

fn parse_matrix_component(output_socket: &str) -> Result<(usize, usize)> {
    matrix_component(output_socket)
        .ok_or_else(|| invalid(format!("unknown matrix component output {output_socket:?}")))
}

fn matrix_component_input(inputs: &[(&str, Value)], column: usize, row: usize) -> Result<f32> {
    let value = inputs
        .iter()
        .find(|(name, _)| matrix_component(name) == Some((column, row)))
        .map(|(_, value)| value)
        .ok_or_else(|| {
            invalid(format!(
                "missing input socket Column {} Row {}",
                column + 1,
                row + 1
            ))
        })?;
    number(value, "matrix component")
}

fn axis_angle_quaternion(axis: Vec3, angle: f32) -> Result<Quat> {
    let axis = normalize_vector(axis);
    if axis == [0.0; 3] {
        return Ok([1.0, 0.0, 0.0, 0.0]);
    }
    let half_angle = angle * 0.5;
    let sine = half_angle.sin();
    normalize_quaternion(
        [
            half_angle.cos(),
            axis[0] * sine,
            axis[1] * sine,
            axis[2] * sine,
        ],
        "axis-angle rotation",
    )
}

fn signed_angle_on_axis(from: Vec3, to: Vec3, axis: Vec3) -> f32 {
    let from_projected = normalize_vector([
        from[0] - axis[0] * dot(from, axis),
        from[1] - axis[1] * dot(from, axis),
        from[2] - axis[2] * dot(from, axis),
    ]);
    let to_projected = normalize_vector([
        to[0] - axis[0] * dot(to, axis),
        to[1] - axis[1] * dot(to, axis),
        to[2] - axis[2] * dot(to, axis),
    ]);
    let mut angle = normalized_angle(from_projected, to_projected);
    if dot(cross(to_projected, from_projected), axis) < 0.0 {
        angle = std::f32::consts::TAU - angle;
    }
    angle
}

fn align_rotation(
    rotation: Quat,
    axis: usize,
    pivot: usize,
    vector: Vec3,
    factor: f32,
) -> Result<Quat> {
    if pivot > 0 && axis == pivot - 1 {
        return Ok(rotation);
    }
    if vector == [0.0; 3] {
        return Ok(rotation);
    }
    let old_axis = mat3_vector(quaternion_matrix3(rotation), axis_vector(axis));
    let new_axis = normalize_vector(vector);
    let pivot_axis = if pivot == 0 {
        let mut rotation_axis = cross_high_precision(old_axis, new_axis);
        if rotation_axis == [0.0; 3] {
            rotation_axis = cross_high_precision(old_axis, [1.0, 0.0, 0.0]);
            if rotation_axis == [0.0; 3] {
                rotation_axis = cross_high_precision(old_axis, [0.0, 1.0, 0.0]);
            }
        }
        normalize_vector(rotation_axis)
    } else {
        mat3_vector(quaternion_matrix3(rotation), axis_vector(pivot - 1))
    };
    let full_angle = if pivot == 0 {
        normalized_angle(old_axis, new_axis)
    } else {
        let mut angle = signed_angle_on_axis(vector, old_axis, pivot_axis);
        if angle > std::f32::consts::PI {
            angle -= std::f32::consts::TAU;
        }
        angle
    };
    let adjustment = axis_angle_quaternion(pivot_axis, factor * full_angle)?;
    normalize_quaternion(
        quaternion_multiply(adjustment, rotation),
        "aligned rotation",
    )
}

fn align_euler(euler: Vec3, axis: usize, pivot: usize, vector: Vec3, factor: f32) -> Vec3 {
    if (pivot > 0 && axis == pivot - 1) || vector == [0.0; 3] {
        return euler;
    }
    let old_matrix = euler_matrix3(euler);
    let old_axis = mat3_vector(old_matrix, axis_vector(axis));
    let new_axis = normalize_vector(vector);
    let pivot_axis = if pivot == 0 {
        let mut rotation_axis = cross_high_precision(old_axis, new_axis);
        if rotation_axis == [0.0; 3] {
            rotation_axis = cross_high_precision(old_axis, [1.0, 0.0, 0.0]);
            if rotation_axis == [0.0; 3] {
                rotation_axis = cross_high_precision(old_axis, [0.0, 1.0, 0.0]);
            }
        }
        normalize_vector(rotation_axis)
    } else {
        mat3_vector(old_matrix, axis_vector(pivot - 1))
    };
    let full_angle = if pivot == 0 {
        normalized_angle(old_axis, new_axis)
    } else {
        let mut angle = signed_angle_on_axis(vector, old_axis, pivot_axis);
        if angle > std::f32::consts::PI {
            angle -= std::f32::consts::TAU;
        }
        angle
    };
    matrix3_euler(
        mat3_multiply(
            axis_angle_matrix3(pivot_axis, factor * full_angle),
            old_matrix,
        ),
        EULER_C_GIMBAL_EPSILON,
    )
}
fn svd3(matrix: Mat3) -> (Mat3, Vec3, Mat3) {
    let mut columns = [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ];
    let mut right = IDENTITY3;
    for _ in 0..32 {
        let mut changed = false;
        for p in 0..2 {
            for q in (p + 1)..3 {
                let alpha = dot(columns[p], columns[p]);
                let beta = dot(columns[q], columns[q]);
                let gamma = dot(columns[p], columns[q]);
                if gamma == 0.0 || gamma.abs() <= 1.0e-7 * (alpha * beta).sqrt() {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta < 0.0 { -1.0 } else { 1.0 };
                let tangent = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;
                let old_p = columns[p];
                let old_q = columns[q];
                for row in 0..3 {
                    columns[p][row] = cosine * old_p[row] - sine * old_q[row];
                    columns[q][row] = sine * old_p[row] + cosine * old_q[row];
                    let old_vp = right[row][p];
                    let old_vq = right[row][q];
                    right[row][p] = cosine * old_vp - sine * old_vq;
                    right[row][q] = sine * old_vp + cosine * old_vq;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut singular = columns.map(length);
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| singular[*b].total_cmp(&singular[*a]));
    let original_columns = columns;
    let original_right = right;
    let original_singular = singular;
    for index in 0..3 {
        columns[index] = original_columns[order[index]];
        singular[index] = original_singular[order[index]];
        for row in 0..3 {
            right[row][index] = original_right[row][order[index]];
        }
    }
    let mut left = [[0.0; 3]; 3];
    let mut defined = [false; 3];
    for column in 0..3 {
        if singular[column] > 0.0 {
            for row in 0..3 {
                left[row][column] = columns[column][row] / singular[column];
            }
            defined[column] = true;
        }
    }
    for column in 0..3 {
        if defined[column] {
            continue;
        }
        let candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut best = candidates[0];
        let mut best_length = -1.0;
        for candidate in candidates {
            let mut projected = candidate;
            for prior in 0..3 {
                if defined[prior] {
                    let basis = [left[0][prior], left[1][prior], left[2][prior]];
                    let projection = dot(projected, basis);
                    for component in 0..3 {
                        projected[component] -= projection * basis[component];
                    }
                }
            }
            let projected_length = length(projected);
            if projected_length > best_length {
                best = projected;
                best_length = projected_length;
            }
        }
        let best = normalize_vector(best);
        for row in 0..3 {
            left[row][column] = best[row];
        }
        defined[column] = true;
    }
    (left, singular, right)
}

fn svd_output(matrix: Mat4, output_socket: &str) -> Result<Value> {
    let matrix3 = matrix3_from4(matrix);
    let (left, singular, right) = svd3(matrix3);
    if output_is(output_socket, "U") {
        matrix_result(matrix4_from3(left), "Matrix SVD U")
    } else if output_is(output_socket, "S") {
        vector_result(singular, "Matrix SVD singular values")
    } else if output_is(output_socket, "V") {
        matrix_result(matrix4_from3(right), "Matrix SVD V")
    } else {
        Err(invalid(format!(
            "unknown Matrix SVD output {output_socket:?}"
        )))
    }
}

pub(super) fn evaluate(
    node_type: &str,
    properties: &Map<String, Value>,
    inputs: &[(&str, Value)],
    output_socket: &str,
) -> Result<Value> {
    match node_type {
        "FunctionNodeAlignEulerToVector" => {
            require_output(output_socket, &["Rotation"])?;
            let rotation = vector_input(inputs, "Rotation")?;
            let factor = number_input(inputs, "Factor")?;
            let vector = vector_input(inputs, "Vector")?;
            let axis = axis_property(properties, "axis", 0)?;
            let pivot = pivot_property(properties, 0)?;
            vector_result(
                align_euler(rotation, axis, pivot, vector, factor),
                "aligned Euler",
            )
        }
        "FunctionNodeAlignRotationToVector" => {
            require_output(output_socket, &["Rotation"])?;
            let rotation = rotation_input(inputs, "Rotation")?;
            let factor = number_input(inputs, "Factor")?;
            let vector = vector_input(inputs, "Vector")?;
            let axis = axis_property(properties, "axis", 2)?;
            let pivot = pivot_property(properties, 0)?;
            quaternion_result(
                align_rotation(rotation, axis, pivot, vector, factor)?,
                "aligned rotation",
            )
        }
        "FunctionNodeAxesToRotation" => {
            require_output(output_socket, &["Rotation"])?;
            let primary_axis = axis_property(properties, "primary_axis", 2)?;
            let secondary_axis = axis_property(properties, "secondary_axis", 0)?;
            if primary_axis == secondary_axis {
                return quaternion_result([1.0, 0.0, 0.0, 0.0], "axes rotation");
            }
            let mut primary = normalize_vector(vector_input(inputs, "Primary Axis")?);
            let mut secondary = vector_input(inputs, "Secondary Axis")?;
            let mut tertiary;
            let primary_nonzero = primary != [0.0; 3];
            let secondary_nonzero = secondary != [0.0; 3];
            if primary_nonzero && secondary_nonzero {
                tertiary = cross(primary, secondary);
                if tertiary == [0.0; 3] {
                    tertiary = orthogonal_vector(primary);
                }
                tertiary = normalize_vector(tertiary);
                secondary = cross(tertiary, primary);
            } else if primary_nonzero {
                secondary = normalize_vector(orthogonal_vector(primary));
                tertiary = cross(primary, secondary);
            } else if secondary_nonzero {
                secondary = normalize_vector(secondary);
                primary = normalize_vector(orthogonal_vector(secondary));
                tertiary = cross(primary, secondary);
            } else {
                return quaternion_result([1.0, 0.0, 0.0, 0.0], "axes rotation");
            }
            let tertiary_factor = if (secondary_axis + 1) % 3 == primary_axis {
                -1.0
            } else {
                1.0
            };
            let mut matrix = IDENTITY3;
            for row in 0..3 {
                matrix[row][primary_axis] = primary[row];
                matrix[row][secondary_axis] = secondary[row];
                let tertiary_axis = 3 - primary_axis - secondary_axis;
                matrix[row][tertiary_axis] = tertiary[row] * tertiary_factor;
            }
            quaternion_result(matrix3_quaternion(matrix)?, "axes rotation")
        }
        "FunctionNodeAxisAngleToRotation" => {
            require_output(output_socket, &["Rotation"])?;
            quaternion_result(
                axis_angle_quaternion(
                    vector_input(inputs, "Axis")?,
                    number_input(inputs, "Angle")?,
                )?,
                "axis-angle rotation",
            )
        }
        "FunctionNodeCombineMatrix" => {
            require_output(output_socket, &["Matrix"])?;
            let mut matrix = [[0.0; 4]; 4];
            for column in 0..4 {
                for row in 0..4 {
                    matrix[row][column] = matrix_component_input(inputs, column, row)?;
                }
            }
            matrix_result(matrix, "combined matrix")
        }
        "FunctionNodeCombineTransform" => {
            require_output(output_socket, &["Transform"])?;
            let translation = vector_input(inputs, "Translation")?;
            let rotation = rotation_input(inputs, "Rotation")?;
            let scale = vector_input(inputs, "Scale")?;
            let mut matrix = IDENTITY4;
            let basis = quaternion_matrix3(rotation);
            for row in 0..3 {
                for column in 0..3 {
                    matrix[row][column] = basis[row][column] * scale[column];
                }
                matrix[row][3] = translation[row];
            }
            matrix_result(matrix, "combined transform")
        }
        "FunctionNodeEulerToRotation" => {
            require_output(output_socket, &["Rotation"])?;
            quaternion_result(
                euler_quaternion(vector_input(inputs, "Euler")?)?,
                "Euler rotation",
            )
        }
        "FunctionNodeInputRotation" => {
            require_output(output_socket, &["Rotation"])?;
            let euler = match property(properties, "rotation_euler") {
                Some(value) => vector3(value, "rotation_euler")?,
                None => [0.0; 3],
            };
            quaternion_result(euler_quaternion(euler)?, "input rotation")
        }
        "FunctionNodeInvertMatrix" => {
            let matrix = matrix_input(inputs, "Matrix", &[])?;
            if output_is(output_socket, "Matrix") {
                matrix_result(
                    matrix_inverse(matrix).unwrap_or(IDENTITY4),
                    "inverse matrix",
                )
            } else if output_is(output_socket, "Invertible") {
                Ok(json!(matrix_inverse(matrix).is_some()))
            } else {
                Err(invalid(format!(
                    "unknown Invert Matrix output {output_socket:?}"
                )))
            }
        }
        "FunctionNodeInvertRotation" => {
            require_output(output_socket, &["Rotation"])?;
            let rotation = rotation_input(inputs, "Rotation")?;
            quaternion_result(
                [rotation[0], -rotation[1], -rotation[2], -rotation[3]],
                "inverse rotation",
            )
        }
        "FunctionNodeMatrixDeterminant" => {
            require_output(output_socket, &["Determinant"])?;
            finite_result(
                matrix_determinant(matrix_input(inputs, "Matrix", &[])?),
                "matrix determinant",
            )
        }
        "FunctionNodeMatrixMultiply" => {
            require_output(output_socket, &["Matrix"])?;
            let a = matrix_input(inputs, "Matrix", &[])?;
            let b = matrix_input(inputs, "Matrix_001", &["Matrix 2", "Matrix.001"])?;
            matrix_result(matrix_multiply(a, b), "matrix product")
        }
        "FunctionNodeMatrixSVD" => svd_output(matrix_input(inputs, "Matrix", &[])?, output_socket),
        "FunctionNodeProjectPoint" => {
            require_output(output_socket, &["Vector"])?;
            let point = vector_input(inputs, "Vector")?;
            let matrix = matrix_input(inputs, "Transform", &[])?;
            let projected = [
                matrix[0][0] * point[0]
                    + matrix[0][1] * point[1]
                    + matrix[0][2] * point[2]
                    + matrix[0][3],
                matrix[1][0] * point[0]
                    + matrix[1][1] * point[1]
                    + matrix[1][2] * point[2]
                    + matrix[1][3],
                matrix[2][0] * point[0]
                    + matrix[2][1] * point[1]
                    + matrix[2][2] * point[2]
                    + matrix[2][3],
            ];
            let w = matrix[3][0] * point[0]
                + matrix[3][1] * point[1]
                + matrix[3][2] * point[2]
                + matrix[3][3];
            vector_result(
                projected.map(|component| component / w.abs()),
                "projected point",
            )
        }
        "FunctionNodeQuaternionToRotation" => {
            require_output(output_socket, &["Rotation"])?;
            let w = number_input(inputs, "W")?;
            let x = number_input(inputs, "X")?;
            let y = number_input(inputs, "Y")?;
            let z = number_input(inputs, "Z")?;
            quaternion_result(
                normalize_quaternion([w, x, y, z], "quaternion inputs")?,
                "quaternion rotation",
            )
        }
        "FunctionNodeRotateEuler" => {
            require_output(output_socket, &["Rotation"])?;
            let input_rotation = vector_input(inputs, "Rotation")?;
            let rotation_type =
                enum_property(properties, "rotation_type", 0, &["EULER", "AXIS_ANGLE"])?;
            let space = enum_property(properties, "space", 0, &["OBJECT", "LOCAL"])?;
            let rotate_by = if rotation_type == 0 {
                euler_matrix3(vector_input(inputs, "Rotate By")?)
            } else {
                axis_angle_matrix3(
                    vector_input(inputs, "Axis")?,
                    number_input(inputs, "Angle")?,
                )
            };
            let input_matrix = euler_matrix3(input_rotation);
            let result = if space == 0 {
                mat3_multiply(rotate_by, input_matrix)
            } else {
                mat3_multiply(input_matrix, rotate_by)
            };
            vector_result(
                matrix3_euler(result, EULER_C_GIMBAL_EPSILON),
                "rotated Euler",
            )
        }
        "FunctionNodeRotateRotation" => {
            require_output(output_socket, &["Rotation"])?;
            let rotation = rotation_input(inputs, "Rotation")?;
            let rotate_by = rotation_input(inputs, "Rotate By")?;
            let space = enum_property(properties, "rotation_space", 0, &["GLOBAL", "LOCAL"])?;
            let result = if space == 0 {
                quaternion_multiply(rotate_by, rotation)
            } else {
                quaternion_multiply(rotation, rotate_by)
            };
            quaternion_result(
                normalize_quaternion(result, "rotated rotation")?,
                "rotated rotation",
            )
        }
        "FunctionNodeRotateVector" => {
            require_output(output_socket, &["Vector"])?;
            let vector = vector_input(inputs, "Vector")?;
            let rotation = rotation_input(inputs, "Rotation")?;
            vector_result(
                mat3_vector(quaternion_matrix3(rotation), vector),
                "rotated vector",
            )
        }
        "FunctionNodeRotationToAxisAngle" => {
            let rotation = rotation_input(inputs, "Rotation")?;
            let vector = [rotation[1], rotation[2], rotation[3]];
            let sine_half_angle = length(vector);
            let (axis, angle) = if sine_half_angle < 0.0005 {
                ([0.0, if rotation[0] < 0.0 { -1.0 } else { 1.0 }, 0.0], 0.0)
            } else {
                (
                    vector.map(|component| component / sine_half_angle),
                    2.0 * sine_half_angle.atan2(rotation[0]),
                )
            };
            if output_is(output_socket, "Axis") {
                vector_result(axis, "rotation axis")
            } else if output_is(output_socket, "Angle") {
                finite_result(angle, "rotation angle")
            } else {
                Err(invalid(format!(
                    "unknown Rotation to Axis Angle output {output_socket:?}"
                )))
            }
        }
        "FunctionNodeRotationToEuler" => {
            require_output(output_socket, &["Euler"])?;
            let rotation = rotation_input(inputs, "Rotation")?;
            vector_result(
                matrix3_euler(quaternion_matrix3(rotation), EULER_MATH_GIMBAL_EPSILON),
                "rotation Euler",
            )
        }
        "FunctionNodeRotationToQuaternion" => {
            let rotation = rotation_input(inputs, "Rotation")?;
            let index = if output_is(output_socket, "W") {
                0
            } else if output_is(output_socket, "X") {
                1
            } else if output_is(output_socket, "Y") {
                2
            } else if output_is(output_socket, "Z") {
                3
            } else {
                return Err(invalid(format!(
                    "unknown Rotation to Quaternion output {output_socket:?}"
                )));
            };
            finite_result(rotation[index], "quaternion component")
        }
        "FunctionNodeSeparateMatrix" => {
            let matrix = matrix_input(inputs, "Matrix", &[])?;
            let (column, row) = parse_matrix_component(output_socket)?;
            finite_result(matrix[row][column], "matrix component")
        }
        "FunctionNodeSeparateTransform" => {
            let matrix = matrix_input(inputs, "Transform", &[])?;
            if output_is(output_socket, "Translation") {
                vector_result(
                    [matrix[0][3], matrix[1][3], matrix[2][3]],
                    "transform translation",
                )
            } else if output_is(output_socket, "Scale") {
                let (_, scale) = normalize_matrix3_columns(matrix3_from4(matrix));
                vector_result(scale, "transform scale")
            } else if output_is(output_socket, "Rotation") {
                let (normalized, _) = normalize_matrix3_columns(matrix3_from4(matrix));
                let euler = matrix3_euler(normalized, EULER_MATH_GIMBAL_EPSILON);
                quaternion_result(euler_quaternion(euler)?, "transform rotation")
            } else {
                Err(invalid(format!(
                    "unknown Separate Transform output {output_socket:?}"
                )))
            }
        }
        "FunctionNodeTransformDirection" => {
            require_output(output_socket, &["Direction"])?;
            let direction = vector_input(inputs, "Direction")?;
            let matrix = matrix_input(inputs, "Transform", &[])?;
            vector_result(
                mat3_vector(matrix3_from4(matrix), direction),
                "transformed direction",
            )
        }
        "FunctionNodeTransformPoint" => {
            require_output(output_socket, &["Vector"])?;
            let point = vector_input(inputs, "Vector")?;
            let matrix = matrix_input(inputs, "Transform", &[])?;
            let linear = mat3_vector(matrix3_from4(matrix), point);
            vector_result(
                [
                    linear[0] + matrix[0][3],
                    linear[1] + matrix[1][3],
                    linear[2] + matrix[2][3],
                ],
                "transformed point",
            )
        }
        "FunctionNodeTransposeMatrix" => {
            require_output(output_socket, &["Matrix"])?;
            let matrix = matrix_input(inputs, "Matrix", &[])?;
            let mut transposed = [[0.0; 4]; 4];
            for row in 0..4 {
                for column in 0..4 {
                    transposed[row][column] = matrix[column][row];
                }
            }
            matrix_result(transposed, "transposed matrix")
        }
        _ => Err(invalid(format!(
            "unsupported rotation/matrix node type {node_type:?}"
        ))),
    }
}
