// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/octree.cpp (pseudoInverse, solve_least_squares, mass_point, minimize),
//   SPDX-FileCopyrightText: 2011-2023 Blender Authors, GPL-2.0-or-later.

//! Dual-contouring vertex placement: mass point and quadric-error minimizer.
//!
//! Accumulation is float32 as in Blender. Blender solves the pseudo-inverse with Eigen's
//! `JacobiSVD`; the normal matrix is symmetric positive semi-definite, so its singular vectors are
//! its eigenvectors and the pseudo-inverse is `sum v v^T / lambda` over eigenvalues above the
//! tolerance. The eigendecomposition here runs in float64 and is rounded to float32 once, which is
//! at least as accurate as Eigen's float32 SVD and differs from it only by float32 rounding.

/// Accumulates the mean of the active crossing points (`mass_point`).
pub(super) fn mass_point(mass: &mut [f32; 3], points: &[[f32; 3]; 12], parity: &[bool; 12]) {
    let mut count = 0_u32;
    for (point, active) in points.iter().zip(parity) {
        if *active {
            for axis in 0..3 {
                mass[axis] += point[axis];
            }
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    for value in mass.iter_mut() {
        *value /= count as f32;
    }
}

/// Builds the normal equations of the active crossings and solves for the QEF minimizer
/// (`minimize`). `mass` receives the crossing mean. `rvalue` is left unchanged when no crossing is
/// active.
pub(super) fn minimize(
    rvalue: &mut [f32; 3],
    mass: &mut [f32; 3],
    points: &[[f32; 3]; 12],
    normals: &[[f32; 3]; 12],
    parity: &[bool; 12],
) {
    let mut ata = [0.0_f32; 6];
    let mut atb = [0.0_f32; 3];
    let mut count = 0_u32;
    for index in 0..12 {
        if !parity[index] {
            continue;
        }
        let normal = normals[index];
        let point = points[index];
        ata[0] += normal[0] * normal[0];
        ata[1] += normal[0] * normal[1];
        ata[2] += normal[0] * normal[2];
        ata[3] += normal[1] * normal[1];
        ata[4] += normal[1] * normal[2];
        ata[5] += normal[2] * normal[2];

        let projection = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
        atb[0] += normal[0] * projection;
        atb[1] += normal[1] * projection;
        atb[2] += normal[2] * projection;

        for axis in 0..3 {
            mass[axis] += point[axis];
        }
        count += 1;
    }
    if count == 0 {
        return;
    }
    for value in mass.iter_mut() {
        *value /= count as f32;
    }
    *rvalue = solve_least_squares(ata, atb, *mass);
}

/// Solves the QEF with Blender's float32 largest-pivot Jacobi eigensolver.
fn solve_least_squares(half_a: [f32; 6], b: [f32; 3], midpoint: [f32; 3]) -> [f32; 3] {
    let mut matrix = [
        [half_a[0], half_a[1], half_a[2]],
        [half_a[1], half_a[3], half_a[4]],
        [half_a[2], half_a[4], half_a[5]],
    ];
    let mut residual = [0.0_f32; 3];
    for (row_index, value) in residual.iter_mut().enumerate() {
        let row = matrix[row_index];
        *value =
            b[row_index] - (row[0] * midpoint[0] + row[1] * midpoint[1] + row[2] * midpoint[2]);
    }

    let mut eigenvectors = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let mut pivot = (0, 1);
        for (left, right) in [(0, 2), (1, 2)] {
            if matrix[left][right].abs() > matrix[pivot.0][pivot.1].abs() {
                pivot = (left, right);
            }
        }
        let (left, right) = pivot;
        let off_diagonal = matrix[left][right];
        if off_diagonal.abs() <= 1.0e-7 {
            break;
        }
        let tau = (matrix[right][right] - matrix[left][left]) / (2.0 * off_diagonal);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + tau.hypot(1.0))
        } else {
            -1.0 / (-tau + tau.hypot(1.0))
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let left_diagonal = matrix[left][left];
        let right_diagonal = matrix[right][right];
        matrix[left][left] = left_diagonal - tangent * off_diagonal;
        matrix[right][right] = right_diagonal + tangent * off_diagonal;
        matrix[left][right] = 0.0;
        matrix[right][left] = 0.0;

        let mut left_values = [0.0_f32; 3];
        let mut right_values = [0.0_f32; 3];
        for (index, row) in matrix.iter_mut().enumerate() {
            if index == left || index == right {
                continue;
            }
            let left_value = row[left];
            let right_value = row[right];
            let updated_left = cosine * left_value - sine * right_value;
            let updated_right = sine * left_value + cosine * right_value;
            row[left] = updated_left;
            row[right] = updated_right;
            left_values[index] = updated_left;
            right_values[index] = updated_right;
        }
        for (index, (value, updated)) in matrix[left].iter_mut().zip(left_values).enumerate() {
            if index != left && index != right {
                *value = updated;
            }
        }
        for (index, (value, updated)) in matrix[right].iter_mut().zip(right_values).enumerate() {
            if index != left && index != right {
                *value = updated;
            }
        }
        for row in &mut eigenvectors {
            let left_value = row[left];
            let right_value = row[right];
            row[left] = cosine * left_value - sine * right_value;
            row[right] = sine * left_value + cosine * right_value;
        }
    }

    let mut minimizer = midpoint;
    for (axis, row) in matrix.iter().enumerate() {
        let eigenvalue = row[axis];
        if eigenvalue.abs() <= 0.1 {
            continue;
        }
        let projection = (eigenvectors[0][axis] * residual[0]
            + eigenvectors[1][axis] * residual[1]
            + eigenvectors[2][axis] * residual[2])
            / eigenvalue;
        for (coordinate, value) in minimizer.iter_mut().enumerate() {
            *value += eigenvectors[coordinate][axis] * projection;
        }
    }
    minimizer
}
