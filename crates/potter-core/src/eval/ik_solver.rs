// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/iksolver/intern/IK_QJacobianSolver.cpp, intern/iksolver/intern/IK_QJacobian.cpp, intern/iksolver/intern/IK_QSegment.cpp, intern/iksolver/intern/IK_Math.h, SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved., GPL-2.0-or-later.

//! Blender legacy `QJacobian` pseudo-inverse updates.
//!
//! The incoming Jacobian contains unweighted geometric derivatives. `QJacobian` scales each `DoF`
//! column by `sqrt(weight)` before decomposition, then scales the resulting update by `weight`.
//! `norm_weights` are used by the surrounding solver to measure convergence; they do not enter
//! either inverse in `IK_QJacobian::InvertSDLS` or `InvertDLS`.

const SINGULAR_EPSILON: f64 = 1.0e-10;
const SDLS_MAX_ANGLE: f64 = std::f64::consts::FRAC_PI_4;
const DLS_MAX_ANGLE: f64 = 0.1;
const SDLS_STEP_FACTOR: f64 = 0.80;

struct ThinSvd {
    /// Left singular vectors, stored row-major with singular vectors in columns.
    u: Vec<f64>,
    /// Singular values in descending order.
    singular_values: Vec<f64>,
    /// Right singular vectors, stored row-major with singular vectors in columns.
    v: Vec<f64>,
    task_rows: usize,
    dofs: usize,
    modes: usize,
}

impl ThinSvd {
    fn u(&self, row: usize, mode: usize) -> f64 {
        self.u[row * self.modes + mode]
    }

    fn v(&self, row: usize, mode: usize) -> f64 {
        self.v[row * self.modes + mode]
    }
}

/// Compute a deterministic thin SVD using cyclic one-sided Jacobi rotations.
///
/// The taller orientation is used so the rotation matrix is only square in the smaller
/// dimension. This avoids forming a Gram matrix, which would square the condition number and
/// lose singular directions close to `QJacobian`'s absolute `1e-10` cutoff.
fn thin_svd(jacobian: &[Vec<f64>], weights: &[f64]) -> ThinSvd {
    let task_rows = jacobian.len();
    let dofs = weights.len();
    let transposed = task_rows < dofs;
    let (rows, columns) = if transposed {
        (dofs, task_rows)
    } else {
        (task_rows, dofs)
    };

    // `work` is the tall weighted Jacobian, or its transpose. Column scaling is applied before
    // the transpose so it always represents J * diag(sqrt(weight)).
    let mut work = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            work[row * columns + column] = if transposed {
                jacobian[column][row] * weights[row].sqrt()
            } else {
                jacobian[row][column] * weights[column].sqrt()
            };
        }
    }

    let mut right = vec![0.0; columns * columns];
    for i in 0..columns {
        right[i * columns + i] = 1.0;
    }

    // Cyclic one-sided Jacobi: after convergence the columns of `work` are orthogonal and
    // `right` contains the corresponding right singular vectors.
    let relative_tolerance = 8.0 * f64::EPSILON * (rows as f64).sqrt();
    let max_sweeps = 32 + 4 * columns;
    for _ in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..columns {
            for q in (p + 1)..columns {
                let mut pp = 0.0;
                let mut qq = 0.0;
                let mut pq = 0.0;
                for row in 0..rows {
                    let a = work[row * columns + p];
                    let b = work[row * columns + q];
                    pp += a * a;
                    qq += b * b;
                    pq += a * b;
                }

                let scale = pp.sqrt() * qq.sqrt();
                if scale == 0.0 || pq.abs() <= relative_tolerance * scale {
                    continue;
                }

                // This rotation diagonalizes [[pp, pq], [pq, qq]] and makes the two columns
                // orthogonal. atan2 keeps the choice deterministic when the norms are equal.
                let angle = 0.5 * (2.0 * pq).atan2(qq - pp);
                let cosine = angle.cos();
                let sine = angle.sin();
                for row in 0..rows {
                    let index_p = row * columns + p;
                    let index_q = row * columns + q;
                    let a = work[index_p];
                    let b = work[index_q];
                    work[index_p] = cosine * a - sine * b;
                    work[index_q] = sine * a + cosine * b;
                }
                for row in 0..columns {
                    let index_p = row * columns + p;
                    let index_q = row * columns + q;
                    let a = right[index_p];
                    let b = right[index_q];
                    right[index_p] = cosine * a - sine * b;
                    right[index_q] = sine * a + cosine * b;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    let mut unsorted = Vec::with_capacity(columns);
    for column in 0..columns {
        let mut norm_squared = 0.0;
        for row in 0..rows {
            let value = work[row * columns + column];
            norm_squared += value * value;
        }
        unsorted.push((norm_squared.sqrt(), column));
    }
    unsorted.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let modes = columns;
    let mut singular_values = Vec::with_capacity(modes);
    let mut u = vec![0.0; task_rows * modes];
    let mut v = vec![0.0; dofs * modes];
    for (mode, &(sigma, source_column)) in unsorted.iter().enumerate() {
        singular_values.push(sigma);
        if sigma == 0.0 {
            continue;
        }
        if transposed {
            // For the SVD of J^T, swap its left and right singular vectors to obtain J's.
            for row in 0..task_rows {
                u[row * modes + mode] = right[row * columns + source_column];
            }
            for row in 0..dofs {
                v[row * modes + mode] = work[row * columns + source_column] / sigma;
            }
        } else {
            for row in 0..task_rows {
                u[row * modes + mode] = work[row * columns + source_column] / sigma;
            }
            for row in 0..dofs {
                v[row * modes + mode] = right[row * columns + source_column];
            }
        }
    }

    ThinSvd {
        u,
        singular_values,
        v,
        task_rows,
        dofs,
        modes,
    }
}

fn weighted_jacobian_column_norms(jacobian: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let mut norms = vec![0.0; weights.len()];
    for column in 0..weights.len() {
        let scale = weights[column].sqrt();
        for task_row in (0..jacobian.len()).step_by(3) {
            let x = jacobian[task_row][column] * scale;
            let y = jacobian[task_row + 1][column] * scale;
            let z = jacobian[task_row + 2][column] * scale;
            norms[column] += (x * x + y * y + z * z).sqrt();
        }
    }
    norms
}

/// Compute Blender's default selectively damped least-squares `QJacobian` angle update.
///
/// Rows are task-space derivatives in groups of three. The weighted matrix is
/// `Jw = J * diag(sqrt(weights))`. For each singular mode `(u, sigma, v)`, the raw modal
/// coefficient is `alpha = dot(u, beta) / sigma`; SDLS derives a task-space norm `N`, a
/// Jacobian/DoF norm `M`, and a modal maximum before applying `gamma = min(pi/4, (pi/4)*N/M)`
/// and `damp = min(1, gamma/max_dtheta)`. The accumulated mode is multiplied by `0.80`,
/// per-DoF damping is saturated at `min(damp / weight, 1)`, and the result is finally scaled by
/// `weight`. Blender's final global attenuation is applied if any axis exceeds pi/4.
///
/// `norm_weights` are accepted for parity with the caller's task/DoF data. In Blender they only
/// weight `AngleUpdateNorm` (the outer convergence check), not the pseudo-inverse update.
pub(crate) fn solve_sdls_update(
    jacobian: &[Vec<f64>],
    beta: &[f64],
    weights: &[f64],
    _norm_weights: &[f64],
) -> Vec<f64> {
    let mut update = vec![0.0; weights.len()];
    if jacobian.is_empty() || weights.is_empty() {
        return update;
    }

    let svd = thin_svd(jacobian, weights);
    let column_norms = weighted_jacobian_column_norms(jacobian, weights);

    for mode in 0..svd.modes {
        let sigma = svd.singular_values[mode];
        if sigma <= SINGULAR_EPSILON {
            continue;
        }

        let mut alpha = 0.0;
        let mut task_norm = 0.0;
        for task_row in (0..svd.task_rows).step_by(3) {
            let x = svd.u(task_row, mode);
            let y = svd.u(task_row + 1, mode);
            let z = svd.u(task_row + 2, mode);
            alpha += x * beta[task_row] + y * beta[task_row + 1] + z * beta[task_row + 2];
            task_norm += (x * x + y * y + z * z).sqrt();
        }
        alpha /= sigma;

        let mut jacobian_norm = 0.0;
        let mut max_axis_delta = 0.0_f64;
        for dof in 0..svd.dofs {
            let right = svd.v(dof, mode);
            jacobian_norm += right.abs() * column_norms[dof];
            max_axis_delta = max_axis_delta.max((right * alpha).abs() * weights[dof].sqrt());
        }
        jacobian_norm /= sigma;

        let mut gamma = SDLS_MAX_ANGLE;
        if task_norm < jacobian_norm {
            gamma *= task_norm / jacobian_norm;
        }
        let damping = if gamma < max_axis_delta {
            gamma / max_axis_delta
        } else {
            1.0
        };

        for (dof, output) in update.iter_mut().enumerate() {
            let dof_damping = if weights[dof] > 0.0 {
                (damping / weights[dof]).min(1.0)
            } else {
                // A zero-weight axis is removed by the final multiplication, without dividing by
                // zero during Blender's axis-weight saturation step.
                1.0
            };
            *output += SDLS_STEP_FACTOR * dof_damping * svd.v(dof, mode) * alpha;
        }
    }

    let mut max_angle = 0.0_f64;
    for (dof, output) in update.iter_mut().enumerate() {
        *output *= weights[dof];
        max_angle = max_angle.max(output.abs());
    }
    if max_angle > SDLS_MAX_ANGLE {
        let attenuation = SDLS_MAX_ANGLE / (SDLS_MAX_ANGLE + max_angle);
        for output in &mut update {
            *output *= attenuation;
        }
    }
    update
}

/// Compute Blender `QJacobian`'s damped least-squares update (the non-default inverse mode).
///
/// This uses the same `J * diag(sqrt(weights))` decomposition and final `DoF` weighting as SDLS.
/// The common damping value is chosen from the smallest retained singular value and the task
/// displacement length, following `IK_QJacobian::InvertDLS`.
pub(crate) fn solve_dls_update(
    jacobian: &[Vec<f64>],
    beta: &[f64],
    weights: &[f64],
    _norm_weights: &[f64],
) -> Vec<f64> {
    let mut update = vec![0.0; weights.len()];
    if jacobian.is_empty() || weights.is_empty() {
        return update;
    }

    let svd = thin_svd(jacobian, weights);
    let mut beta_length_squared = 0.0;
    for value in beta {
        beta_length_squared += value * value;
    }
    let distance = beta_length_squared.sqrt() / DLS_MAX_ANGLE;
    let mut minimum_singular = f64::MAX;
    for &sigma in &svd.singular_values {
        if sigma > SINGULAR_EPSILON && sigma < minimum_singular {
            minimum_singular = sigma;
        }
    }

    let lambda_root = if minimum_singular <= distance / 2.0 {
        distance / 2.0
    } else if minimum_singular < distance {
        (minimum_singular * (distance - minimum_singular)).sqrt()
    } else {
        0.0
    };
    let lambda = (lambda_root * lambda_root).min(10.0);

    for mode in 0..svd.modes {
        let sigma = svd.singular_values[mode];
        if sigma <= SINGULAR_EPSILON {
            continue;
        }
        let mut projection = 0.0;
        for (task_row, beta_value) in beta.iter().enumerate() {
            projection += svd.u(task_row, mode) * beta_value;
        }
        let coefficient = projection * sigma / (sigma * sigma + lambda);
        for (dof, output) in update.iter_mut().enumerate() {
            *output += svd.v(dof, mode) * coefficient;
        }
    }
    for (dof, output) in update.iter_mut().enumerate() {
        *output *= weights[dof];
    }
    update
}

/// One Blender `QJacobian` bone segment, in solver-root coordinates.
pub(crate) struct LegacySegment {
    pub start: glam::DVec3,
    pub rest_basis: glam::DMat3,
    pub basis: glam::DMat3,
    pub length: f64,
    pub locked: [bool; 3],
    pub use_limits: [bool; 3],
    pub limits_min: [f64; 3],
    pub limits_max: [f64; 3],
    pub stiffness: [f64; 3],
    pub ik_stretch: f64,
}

/// A Blender IK goal, expressed in the selected solver chain's root frame.
#[derive(Clone, Copy)]
pub(crate) struct LegacyGoal {
    pub position: glam::DVec3,
    pub rotation: glam::DMat3,
    pub pole: Option<glam::DVec3>,
    pub pole_angle: f64,
    pub use_location: bool,
    pub use_rotation: bool,
    pub use_stretch: bool,
    pub position_weight: f64,
    pub orientation_weight: f64,
    pub influence: f64,
    /// Blender's legacy solver defaults to selectively damped least squares.
    pub use_sdls: bool,
}

pub(crate) struct LegacyResult {
    pub basis_changes: Vec<glam::DMat3>,
    pub stretch_ratios: Vec<f64>,
}

#[derive(Clone, Copy)]
enum LegacyJoint {
    Null,
    Revolute {
        axis: usize,
        angle: f64,
        new_angle: f64,
        limit: Option<(f64, f64)>,
    },
    Swing {
        new_basis: glam::DMat3,
        limit_x: bool,
        limit_z: bool,
        min: [f64; 2],
        max: [f64; 2],
    },
    Spherical {
        new_basis: glam::DMat3,
        limit_x: bool,
        limit_y: bool,
        limit_z: bool,
        min: [f64; 2],
        max: [f64; 2],
        min_y: f64,
        max_y: f64,
        locked_ax: f64,
        locked_ay: f64,
        locked_az: f64,
    },
    Elbow {
        axis: usize,
        angle: f64,
        twist: f64,
        new_angle: f64,
        new_twist: f64,
        cos_twist: f64,
        sin_twist: f64,
        limit: Option<(f64, f64)>,
        twist_limit: Option<(f64, f64)>,
    },
    Translate {
        new_translation_y: f64,
        min_y: f64,
        max_y: f64,
    },
}

struct LegacyNode {
    parent: Option<usize>,
    start: glam::DVec3,
    rest_basis: glam::DMat3,
    basis: glam::DMat3,
    original_basis: glam::DMat3,
    translation_y: f64,
    original_length: f64,
    max_extension: f64,
    joint: LegacyJoint,
    locked: [bool; 3],
    dof_start: usize,
    global_start: glam::DVec3,
    global_end: glam::DVec3,
    global_basis: glam::DMat3,
}

impl LegacyNode {
    fn dof_count(&self) -> usize {
        match self.joint {
            LegacyJoint::Null => 0,
            LegacyJoint::Revolute { .. } | LegacyJoint::Translate { .. } => 1,
            LegacyJoint::Swing { .. } | LegacyJoint::Elbow { .. } => 2,
            LegacyJoint::Spherical { .. } => 3,
        }
    }

    fn is_translation(&self) -> bool {
        matches!(self.joint, LegacyJoint::Translate { .. })
    }

    fn dof_axis(&self, dof: usize) -> usize {
        match self.joint {
            LegacyJoint::Revolute { axis, .. } => axis,
            LegacyJoint::Swing { .. } => [0, 2][dof],
            LegacyJoint::Spherical { .. } => dof,
            LegacyJoint::Elbow { axis, .. } => [axis, 1][dof],
            LegacyJoint::Translate { .. } => 1,
            LegacyJoint::Null => unreachable!("null segments have no degrees of freedom"),
        }
    }

    fn axis(&self, dof: usize) -> glam::DVec3 {
        use glam::DVec3;

        match self.joint {
            LegacyJoint::Revolute { axis, .. } => matrix_column(self.global_basis, axis),
            LegacyJoint::Swing { .. } => matrix_column(self.global_basis, [0, 2][dof]),
            LegacyJoint::Spherical { .. } => matrix_column(self.global_basis, dof),
            LegacyJoint::Elbow {
                axis,
                cos_twist,
                sin_twist,
                ..
            } if dof == 0 => {
                let local = if axis == 0 {
                    DVec3::new(cos_twist, 0.0, sin_twist)
                } else {
                    DVec3::new(-sin_twist, 0.0, cos_twist)
                };
                self.global_basis * local
            }
            LegacyJoint::Elbow { .. } | LegacyJoint::Translate { .. } => self.global_basis.y_axis,
            LegacyJoint::Null => unreachable!("null segments have no degrees of freedom"),
        }
    }

    fn update_angle(&mut self, update: &[f64]) -> ([bool; 3], [f64; 3], bool) {
        use glam::DVec3;

        let mut clamp = [false; 3];
        let mut delta = [0.0; 3];
        let mut was_clamped = false;

        match &mut self.joint {
            LegacyJoint::Null => {}
            LegacyJoint::Revolute {
                angle,
                new_angle,
                limit,
                ..
            } => {
                if self.locked[0] {
                    return (clamp, delta, false);
                }
                *new_angle = *angle + update[self.dof_start];
                let Some((min, max)) = *limit else {
                    return (clamp, delta, false);
                };
                if *new_angle > max {
                    delta[0] = max - *angle;
                } else if *new_angle < min {
                    delta[0] = min - *angle;
                } else {
                    return (clamp, delta, false);
                }
                clamp[0] = true;
                *new_angle = *angle + delta[0];
                was_clamped = true;
            }
            LegacyJoint::Swing {
                new_basis,
                limit_x,
                limit_z,
                min,
                max,
            } => {
                if self.locked[0] && self.locked[1] {
                    return (clamp, delta, false);
                }

                let dq = DVec3::new(update[self.dof_start], 0.0, update[self.dof_start + 1]);
                *new_basis = self.basis * rodrigues(dq);
                remove_twist(new_basis);
                if !*limit_x && !*limit_z {
                    return (clamp, delta, false);
                }

                let parameters = spherical_range_parameters(*new_basis);
                // This mirrors IK_QSwingSegment::UpdateAngle: the single-axis branches clamp
                // their zero-initialized parameter, while the two-axis branch uses both values.
                let mut ax = 0.0;
                let mut az = 0.0;
                if *limit_x && *limit_z {
                    ax = parameters.x;
                    az = parameters.z;
                    if ellipse_clamp(&mut ax, &mut az, min, max) {
                        clamp[0] = true;
                        clamp[1] = true;
                    }
                } else if *limit_x {
                    if ax < min[0] {
                        ax = min[0];
                        clamp[0] = true;
                    } else if ax > max[0] {
                        ax = max[0];
                        clamp[0] = true;
                    }
                } else if *limit_z {
                    if az < min[1] {
                        az = min[1];
                        clamp[1] = true;
                    } else if az > max[1] {
                        az = max[1];
                        clamp[1] = true;
                    }
                }

                if !clamp[0] && !clamp[1] {
                    return (clamp, delta, false);
                }
                *new_basis = compute_swing_matrix(ax, az);
                let rotation_delta = matrix_to_axis_angle(self.basis.transpose() * *new_basis);
                delta[0] = rotation_delta.x;
                delta[1] = rotation_delta.z;
                was_clamped = true;
            }
            LegacyJoint::Spherical {
                new_basis,
                limit_x,
                limit_y,
                limit_z,
                min,
                max,
                min_y,
                max_y,
                locked_ax,
                locked_ay,
                locked_az,
            } => {
                if self.locked[0] && self.locked[1] && self.locked[2] {
                    return (clamp, delta, false);
                }
                let dq = DVec3::new(
                    update[self.dof_start],
                    update[self.dof_start + 1],
                    update[self.dof_start + 2],
                );
                *new_basis = self.basis * rodrigues(dq);
                if !*limit_x && !*limit_y && !*limit_z {
                    return (clamp, delta, false);
                }

                let parameters = spherical_range_parameters(*new_basis);
                let mut ax = parameters.x;
                let mut ay = parameters.y;
                let mut az = parameters.z;
                if self.locked[0] {
                    ax = *locked_ax;
                }
                if self.locked[1] {
                    ay = *locked_ay;
                }
                if self.locked[2] {
                    az = *locked_az;
                }

                if *limit_y {
                    if ay > *max_y {
                        ay = *max_y;
                        clamp[1] = true;
                    } else if ay < *min_y {
                        ay = *min_y;
                        clamp[1] = true;
                    }
                }
                if *limit_x && *limit_z {
                    if ellipse_clamp(&mut ax, &mut az, min, max) {
                        clamp[0] = true;
                        clamp[2] = true;
                    }
                } else if *limit_x {
                    if ax < min[0] {
                        ax = min[0];
                        clamp[0] = true;
                    } else if ax > max[0] {
                        ax = max[0];
                        clamp[0] = true;
                    }
                } else if *limit_z {
                    if az < min[1] {
                        az = min[1];
                        clamp[2] = true;
                    } else if az > max[1] {
                        az = max[1];
                        clamp[2] = true;
                    }
                }

                if !clamp[0] && !clamp[1] && !clamp[2] {
                    if self.locked.iter().any(|locked| *locked) {
                        *new_basis = compute_swing_matrix(ax, az) * rotation_matrix(ay, 1);
                    }
                    return (clamp, delta, false);
                }
                *new_basis = compute_swing_matrix(ax, az) * rotation_matrix(ay, 1);
                delta = matrix_to_axis_angle(self.basis.transpose() * *new_basis).to_array();
                if !self.locked[0] && !self.locked[2] && (clamp[0] || clamp[2]) {
                    *locked_ax = ax;
                    *locked_az = az;
                }
                if !self.locked[1] && clamp[1] {
                    *locked_ay = ay;
                }
                was_clamped = true;
            }
            LegacyJoint::Elbow {
                angle,
                twist,
                new_angle,
                new_twist,
                limit,
                twist_limit,
                ..
            } => {
                if self.locked[0] && self.locked[1] {
                    return (clamp, delta, false);
                }
                clamp[0] = false;
                clamp[1] = false;
                if !self.locked[0] {
                    *new_angle = *angle + update[self.dof_start];
                    if let Some((min, max)) = *limit {
                        if *new_angle > max {
                            delta[0] = max - *angle;
                            *new_angle = max;
                            clamp[0] = true;
                        } else if *new_angle < min {
                            delta[0] = min - *angle;
                            *new_angle = min;
                            clamp[0] = true;
                        }
                    }
                }
                if !self.locked[1] {
                    *new_twist = *twist + update[self.dof_start + 1];
                    if let Some((min, max)) = *twist_limit {
                        if *new_twist > max {
                            delta[1] = max - *twist;
                            *new_twist = max;
                            clamp[1] = true;
                        } else if *new_twist < min {
                            delta[1] = min - *twist;
                            *new_twist = min;
                            clamp[1] = true;
                        }
                    }
                }
                was_clamped = clamp[0] || clamp[1];
            }
            LegacyJoint::Translate {
                new_translation_y,
                min_y,
                max_y,
            } => {
                if self.locked[0] {
                    return (clamp, delta, false);
                }
                *new_translation_y = self.translation_y + update[self.dof_start];
                if *new_translation_y > *max_y {
                    delta[0] = *max_y - self.translation_y;
                    *new_translation_y = *max_y;
                    clamp[0] = true;
                    was_clamped = true;
                } else if *new_translation_y < *min_y {
                    delta[0] = *min_y - self.translation_y;
                    *new_translation_y = *min_y;
                    clamp[0] = true;
                    was_clamped = true;
                }
            }
        }
        (clamp, delta, was_clamped)
    }

    fn lock_dof(
        &mut self,
        dof: usize,
        delta: [f64; 3],
        jacobian: &mut [Vec<f64>],
        beta: &mut [f64],
        dof_weights: &[f64],
    ) {
        let mut locked_dofs = [false; 3];
        match self.joint {
            LegacyJoint::Null => return,
            LegacyJoint::Revolute { .. } | LegacyJoint::Translate { .. } => {
                self.locked[0] = true;
                locked_dofs[0] = true;
            }
            LegacyJoint::Swing { .. } => {
                self.locked[0] = true;
                self.locked[1] = true;
                locked_dofs[0] = true;
                locked_dofs[1] = true;
            }
            LegacyJoint::Spherical { .. } if dof == 1 => {
                self.locked[1] = true;
                locked_dofs[1] = true;
            }
            LegacyJoint::Spherical { .. } => {
                self.locked[0] = true;
                self.locked[2] = true;
                locked_dofs[0] = true;
                locked_dofs[2] = true;
            }
            LegacyJoint::Elbow { .. } => {
                self.locked[dof] = true;
                locked_dofs[dof] = true;
            }
        }

        for local_dof in 0..self.dof_count() {
            if !locked_dofs[local_dof] {
                continue;
            }
            let column = self.dof_start + local_dof;
            let locked_delta = delta[local_dof];
            let sqrt_weight = dof_weights[column].sqrt();
            for (row, values) in jacobian.iter_mut().enumerate() {
                beta[row] -= values[column] * sqrt_weight * locked_delta;
                values[column] = 0.0;
            }
        }
    }

    fn unlock(&mut self) {
        self.locked = [false; 3];
    }

    fn apply_update(&mut self) {
        match &mut self.joint {
            LegacyJoint::Null => {}
            LegacyJoint::Revolute {
                axis,
                angle,
                new_angle,
                ..
            } => {
                *angle = *new_angle;
                self.basis = rotation_matrix(*angle, *axis);
            }
            LegacyJoint::Swing { new_basis, .. } | LegacyJoint::Spherical { new_basis, .. } => {
                self.basis = *new_basis;
            }
            LegacyJoint::Elbow {
                axis,
                angle,
                twist,
                new_angle,
                new_twist,
                cos_twist,
                sin_twist,
                ..
            } => {
                *angle = *new_angle;
                *twist = *new_twist;
                *cos_twist = twist.cos();
                *sin_twist = twist.sin();
                self.basis = rotation_matrix(*angle, *axis) * rotation_matrix(*twist, 1);
            }
            LegacyJoint::Translate {
                new_translation_y, ..
            } => {
                self.translation_y = *new_translation_y;
            }
        }
    }

    fn prepend_basis(&mut self, matrix: glam::DMat3) {
        self.basis = self.rest_basis.inverse() * matrix * self.rest_basis * self.basis;
    }
}

fn matrix_column(matrix: glam::DMat3, axis: usize) -> glam::DVec3 {
    match axis {
        0 => matrix.x_axis,
        1 => matrix.y_axis,
        _ => matrix.z_axis,
    }
}

fn rotation_matrix(angle: f64, axis: usize) -> glam::DMat3 {
    use glam::{DMat3, DVec3};

    let sine = angle.sin();
    let cosine = angle.cos();
    match axis {
        0 => DMat3::from_cols(
            DVec3::X,
            DVec3::new(0.0, cosine, sine),
            DVec3::new(0.0, -sine, cosine),
        ),
        1 => DMat3::from_cols(
            DVec3::new(cosine, 0.0, -sine),
            DVec3::Y,
            DVec3::new(sine, 0.0, cosine),
        ),
        _ => DMat3::from_cols(
            DVec3::new(cosine, sine, 0.0),
            DVec3::new(-sine, cosine, 0.0),
            DVec3::Z,
        ),
    }
}

fn rodrigues(delta: glam::DVec3) -> glam::DMat3 {
    use glam::{DMat3, DVec3};

    let theta = delta.length();
    if theta < 1.0e-20 {
        return DMat3::IDENTITY;
    }
    let axis = delta / theta;
    let sine = theta.sin();
    let cosine = theta.cos();
    let one_minus_cosine = 1.0 - cosine;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    let rows = [
        [
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ],
        [
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ],
        [
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ],
    ];
    DMat3::from_cols(
        DVec3::new(rows[0][0], rows[1][0], rows[2][0]),
        DVec3::new(rows[0][1], rows[1][1], rows[2][1]),
        DVec3::new(rows[0][2], rows[1][2], rows[2][2]),
    )
}

fn compute_twist(matrix: glam::DMat3) -> f64 {
    let qw = matrix.x_axis.x + matrix.y_axis.y + matrix.z_axis.z + 1.0;
    let qy = matrix.z_axis.x - matrix.x_axis.z;
    2.0 * qy.atan2(qw)
}

fn remove_twist(matrix: &mut glam::DMat3) {
    *matrix = *matrix * rotation_matrix(-compute_twist(*matrix), 1);
}

fn spherical_range_parameters(matrix: glam::DMat3) -> glam::DVec3 {
    let twist = compute_twist(matrix);
    let denominator = 2.0 * (1.0 + matrix.y_axis.y);
    if denominator.abs() < 1.0e-20 {
        return glam::DVec3::new(0.0, twist, 1.0);
    }
    let inverse = 1.0 / denominator.sqrt();
    glam::DVec3::new(-matrix.y_axis.z * inverse, twist, matrix.y_axis.x * inverse)
}

fn compute_swing_matrix(ax: f64, az: f64) -> glam::DMat3 {
    let sine_squared = ax * ax + az * az;
    let cosine = (1.0 - sine_squared).max(0.0).sqrt();
    glam::DMat3::from_quat(glam::DQuat::from_xyzw(ax, 0.0, az, -cosine))
}

fn matrix_to_axis_angle(matrix: glam::DMat3) -> glam::DVec3 {
    let delta = glam::DVec3::new(
        matrix.y_axis.z - matrix.z_axis.y,
        matrix.z_axis.x - matrix.x_axis.z,
        matrix.x_axis.y - matrix.y_axis.x,
    );
    let cosine =
        ((matrix.x_axis.x + matrix.y_axis.y + matrix.z_axis.z - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let length = delta.length();
    if length < 1.0e-20 {
        glam::DVec3::ZERO
    } else {
        delta * (angle / length)
    }
}

fn euler_angle_from_matrix(matrix: glam::DMat3, axis: usize) -> f64 {
    let t = (matrix.x_axis.x * matrix.x_axis.x + matrix.y_axis.x * matrix.y_axis.x).sqrt();
    if t > 16.0e-20 {
        match axis {
            0 => -matrix.z_axis.y.atan2(matrix.z_axis.z),
            1 => (-matrix.z_axis.x).atan2(t),
            _ => -matrix.y_axis.x.atan2(matrix.x_axis.x),
        }
    } else {
        match axis {
            0 => (-matrix.y_axis.z).atan2(matrix.y_axis.y),
            1 => (-matrix.z_axis.x).atan2(t),
            _ => 0.0,
        }
    }
}

fn ellipse_clamp(ax: &mut f64, az: &mut f64, min: &[f64; 2], max: &[f64; 2]) -> bool {
    let (mut x, x_limit) = if *ax < 0.0 {
        (-*ax, -min[0])
    } else {
        (*ax, max[0])
    };
    let (mut z, z_limit) = if *az < 0.0 {
        (-*az, -min[1])
    } else {
        (*az, max[1])
    };

    if x_limit.abs() < 1.0e-20 || z_limit.abs() < 1.0e-20 {
        if x <= x_limit && z <= z_limit {
            return false;
        }
        x = x.min(x_limit);
        z = z.min(z_limit);
    } else {
        let inverse_x = 1.0 / (x_limit * x_limit);
        let inverse_z = 1.0 / (z_limit * z_limit);
        if x * x * inverse_x + z * z * inverse_z <= 1.0 {
            return false;
        }
        if x.abs() < 1.0e-20 {
            x = 0.0;
            z = z_limit;
        } else {
            let ratio = z / x;
            x = (1.0 / (inverse_x + inverse_z * ratio * ratio)).sqrt();
            z = ratio * x;
        }
    }

    *ax = if *ax < 0.0 { -x } else { x };
    *az = if *az < 0.0 { -z } else { z };
    true
}

fn clamped_angle_limit(segment: &LegacySegment, axis: usize) -> Option<(f64, f64)> {
    if !segment.use_limits[axis] || segment.limits_min[axis] > segment.limits_max[axis] {
        return None;
    }
    Some((
        segment.limits_min[axis].clamp(-std::f64::consts::PI, std::f64::consts::PI),
        segment.limits_max[axis].clamp(-std::f64::consts::PI, std::f64::consts::PI),
    ))
}

fn swing_limit_range(
    segment: &LegacySegment,
    axis: usize,
    min: &mut [f64; 2],
    max: &mut [f64; 2],
) -> bool {
    let Some((angle_min, angle_max)) = clamped_angle_limit(segment, axis) else {
        return false;
    };
    let index = usize::from(axis != 0);
    min[index] = -(angle_max * 0.5).sin();
    max[index] = -(angle_min * 0.5).sin();
    true
}

fn set_joint_limits(joint: &mut LegacyJoint, segment: &LegacySegment) {
    match joint {
        LegacyJoint::Null | LegacyJoint::Translate { .. } => {}
        LegacyJoint::Revolute { axis, limit, .. } => {
            *limit = clamped_angle_limit(segment, *axis);
        }
        LegacyJoint::Swing {
            limit_x,
            limit_z,
            min,
            max,
            ..
        } => {
            *limit_x = swing_limit_range(segment, 0, min, max);
            *limit_z = swing_limit_range(segment, 2, min, max);
        }
        LegacyJoint::Spherical {
            limit_x,
            limit_y,
            limit_z,
            min,
            max,
            min_y,
            max_y,
            ..
        } => {
            *limit_x = swing_limit_range(segment, 0, min, max);
            *limit_z = swing_limit_range(segment, 2, min, max);
            if let Some((low, high)) = clamped_angle_limit(segment, 1) {
                *min_y = low;
                *max_y = high;
                *limit_y = true;
            }
        }
        LegacyJoint::Elbow {
            axis,
            limit,
            twist_limit,
            ..
        } => {
            *limit = clamped_angle_limit(segment, *axis);
            *twist_limit = clamped_angle_limit(segment, 1);
        }
    }
}

fn stiffness_weight(stiffness: f64) -> f64 {
    if stiffness < 0.0 {
        1.0
    } else {
        1.0 - stiffness.min(0.99)
    }
}

fn make_rotation_joint(segment: &LegacySegment, axes: &[usize]) -> (LegacyJoint, glam::DMat3) {
    use glam::DMat3;

    let original = segment.basis;
    let (mut joint, basis) = match axes {
        [0, 1, 2] => (
            LegacyJoint::Spherical {
                new_basis: original,
                limit_x: false,
                limit_y: false,
                limit_z: false,
                min: [0.0; 2],
                max: [0.0; 2],
                min_y: 0.0,
                max_y: 0.0,
                locked_ax: 0.0,
                locked_ay: 0.0,
                locked_az: 0.0,
            },
            original,
        ),
        [0, 2] => {
            let mut basis = original;
            remove_twist(&mut basis);
            (
                LegacyJoint::Swing {
                    new_basis: basis,
                    limit_x: false,
                    limit_z: false,
                    min: [0.0; 2],
                    max: [0.0; 2],
                },
                basis,
            )
        }
        [axis] => {
            let angle = if *axis == 1 {
                compute_twist(original)
            } else {
                euler_angle_from_matrix(original, *axis)
            };
            (
                LegacyJoint::Revolute {
                    axis: *axis,
                    angle,
                    new_angle: angle,
                    limit: None,
                },
                rotation_matrix(angle, *axis),
            )
        }
        [first, second] => {
            let axis = if *first == 0 && *second == 1 { 0 } else { 2 };
            let twist = compute_twist(original);
            let angle = euler_angle_from_matrix(original, axis);
            let basis = rotation_matrix(angle, axis) * rotation_matrix(twist, 1);
            (
                LegacyJoint::Elbow {
                    axis,
                    angle,
                    twist,
                    new_angle: angle,
                    new_twist: twist,
                    cos_twist: 1.0,
                    sin_twist: 0.0,
                    limit: None,
                    twist_limit: None,
                },
                basis,
            )
        }
        _ => (LegacyJoint::Null, DMat3::IDENTITY),
    };
    set_joint_limits(&mut joint, segment);
    (joint, basis)
}

fn update_transforms(nodes: &mut [LegacyNode], root_basis: glam::DMat3) {
    use glam::DVec3;

    for index in 0..nodes.len() {
        let (parent_position, parent_basis) = if let Some(parent) = nodes[index].parent {
            (nodes[parent].global_end, nodes[parent].global_basis)
        } else {
            (DVec3::ZERO, root_basis)
        };
        nodes[index].global_start = parent_position + parent_basis * nodes[index].start;
        nodes[index].global_basis = parent_basis * nodes[index].rest_basis * nodes[index].basis;
        nodes[index].global_end = nodes[index].global_start
            + nodes[index].global_basis * DVec3::new(0.0, nodes[index].translation_y, 0.0);
    }
}

fn ik_normalize(vector: glam::DVec3) -> glam::DVec3 {
    let length = vector.length();
    if length < 1.0e-20 {
        glam::DVec3::ZERO
    } else {
        vector / length
    }
}

fn look_at_rows(direction: glam::DVec3, up: glam::DVec3) -> glam::DMat3 {
    let row0 = ik_normalize(direction.cross(up));
    let row1 = row0.cross(direction);
    let row2 = -direction;
    glam::DMat3::from_cols(row0, row1, row2).transpose()
}

fn constrain_pole(
    nodes: &mut [LegacyNode],
    goal: glam::DVec3,
    pole: glam::DVec3,
    pole_angle: f64,
    root_basis: &mut glam::DMat3,
) {
    update_transforms(nodes, *root_basis);
    let root_position = nodes[0].global_start;
    let end_position = nodes[nodes.len() - 1].global_end;
    let root_orientation = nodes[0].global_basis;
    let direction = ik_normalize(end_position - root_position);
    let up =
        root_orientation.x_axis * pole_angle.cos() + root_orientation.z_axis * pole_angle.sin();
    let actual_frame = look_at_rows(direction, up);
    let pole_direction = ik_normalize(goal - root_position);
    let pole_up = ik_normalize(pole - root_position);
    let pole_frame = look_at_rows(pole_direction, pole_up);
    *root_basis = pole_frame.transpose() * actual_frame * *root_basis;
}

fn matrix_slerp(from: glam::DMat3, to: glam::DMat3, factor: f64) -> glam::DMat3 {
    glam::DMat3::from_quat(glam::DQuat::from_mat3(&from).slerp(glam::DQuat::from_mat3(&to), factor))
}

/// Solve a legacy Blender `QJacobian` chain using the existing selectively-damped inverse.
#[expect(
    clippy::float_cmp,
    reason = "exact Blender influence endpoints control unblended solver results"
)]
pub(crate) fn solve_legacy(
    segments: &[LegacySegment],
    goal: LegacyGoal,
    max_iterations: usize,
) -> LegacyResult {
    use glam::{DMat3, DVec3};

    if segments.is_empty() {
        return LegacyResult {
            basis_changes: Vec::new(),
            stretch_ratios: Vec::new(),
        };
    }

    let position_task = goal.use_location && goal.position_weight > 0.0;
    let orientation_task = goal.use_rotation && goal.orientation_weight > 0.0;
    if !position_task && !orientation_task {
        return LegacyResult {
            basis_changes: vec![DMat3::IDENTITY; segments.len()],
            stretch_ratios: vec![1.0; segments.len()],
        };
    }

    let total_extension: f64 = segments
        .iter()
        .map(|segment| segment.start.length() + segment.length)
        .sum();
    let scale = if total_extension == 0.0 {
        1.0
    } else {
        f64::from(total_extension.recip() as f32)
    };
    let use_pole = goal.pole.is_some();
    let mut nodes = Vec::with_capacity(segments.len() * 2);
    let mut segment_nodes = Vec::with_capacity(segments.len());
    let mut stretch_nodes = Vec::with_capacity(segments.len());
    let mut dof_weights = Vec::with_capacity(segments.len() * 4);
    let mut norm_weights = Vec::with_capacity(segments.len() * 4);
    let mut parent = None;

    for segment in segments {
        let mut active_axes = [0; 3];
        let mut active_count = 0;
        for axis in 0..3 {
            if !segment.locked[axis] {
                active_axes[active_count] = axis;
                active_count += 1;
            }
        }
        let active_axes = &active_axes[..active_count];
        let stretch = goal.use_stretch && segment.ik_stretch > 0.0;
        let original_length = segment.length * scale;
        let segment_node;
        let stretch_node;
        if !active_axes.is_empty() {
            let (joint, basis) = make_rotation_joint(segment, active_axes);
            let dof_start = dof_weights.len();
            let node_index = nodes.len();
            nodes.push(LegacyNode {
                parent,
                start: segment.start * scale,
                rest_basis: segment.rest_basis,
                basis,
                original_basis: segment.basis,
                translation_y: if stretch { 0.0 } else { original_length },
                original_length: if stretch { 0.0 } else { original_length },
                max_extension: segment.start.length() * scale
                    + if stretch { 0.0 } else { original_length },
                joint,
                locked: [false; 3],
                dof_start,
                global_start: DVec3::ZERO,
                global_end: DVec3::ZERO,
                global_basis: DMat3::IDENTITY,
            });
            for local_dof in 0..nodes[node_index].dof_count() {
                let axis = nodes[node_index].dof_axis(local_dof);
                dof_weights.push(stiffness_weight(segment.stiffness[axis]));
                norm_weights.push(1.0);
            }
            segment_node = node_index;
            parent = Some(node_index);

            if stretch {
                let dof_start = dof_weights.len();
                let node_index = nodes.len();
                nodes.push(LegacyNode {
                    parent,
                    start: DVec3::ZERO,
                    rest_basis: DMat3::IDENTITY,
                    basis: DMat3::IDENTITY,
                    original_basis: DMat3::IDENTITY,
                    translation_y: original_length,
                    original_length,
                    max_extension: original_length,
                    joint: LegacyJoint::Translate {
                        new_translation_y: original_length,
                        min_y: 0.001,
                        max_y: 1.0e10 * scale.powi(3),
                    },
                    locked: [false; 3],
                    dof_start,
                    global_start: DVec3::ZERO,
                    global_end: DVec3::ZERO,
                    global_basis: DMat3::IDENTITY,
                });
                let stretch_stiffness = 1.0 - segment.ik_stretch * segment.ik_stretch;
                dof_weights.push(stiffness_weight(stretch_stiffness));
                norm_weights.push(100.0);
                parent = Some(node_index);
                stretch_node = Some(node_index);
            } else {
                stretch_node = None;
            }
        } else if stretch {
            let dof_start = dof_weights.len();
            let node_index = nodes.len();
            nodes.push(LegacyNode {
                parent,
                start: segment.start * scale,
                rest_basis: segment.rest_basis,
                basis: segment.basis,
                original_basis: segment.basis,
                translation_y: original_length,
                original_length,
                max_extension: segment.start.length() * scale + original_length,
                joint: LegacyJoint::Translate {
                    new_translation_y: original_length,
                    min_y: 0.001,
                    max_y: 1.0e10 * scale.powi(3),
                },
                locked: [false; 3],
                dof_start,
                global_start: DVec3::ZERO,
                global_end: DVec3::ZERO,
                global_basis: DMat3::IDENTITY,
            });
            let stretch_stiffness = 1.0 - segment.ik_stretch * segment.ik_stretch;
            dof_weights.push(stiffness_weight(stretch_stiffness));
            norm_weights.push(100.0);
            segment_node = node_index;
            stretch_node = Some(node_index);
            parent = Some(node_index);
        } else {
            let node_index = nodes.len();
            nodes.push(LegacyNode {
                parent,
                start: segment.start * scale,
                rest_basis: segment.rest_basis,
                basis: DMat3::IDENTITY,
                original_basis: segment.basis,
                translation_y: original_length,
                original_length,
                max_extension: segment.start.length() * scale + original_length,
                joint: LegacyJoint::Null,
                locked: [false; 3],
                dof_start: dof_weights.len(),
                global_start: DVec3::ZERO,
                global_end: DVec3::ZERO,
                global_basis: DMat3::IDENTITY,
            });
            segment_node = node_index;
            stretch_node = None;
            parent = Some(node_index);
        }
        segment_nodes.push(segment_node);
        stretch_nodes.push(stretch_node);
    }

    let dof_count = dof_weights.len();
    let row_count = (if position_task { 3 } else { 0 }) + if orientation_task { 3 } else { 0 };
    let mut root_basis = DMat3::IDENTITY;
    update_transforms(&mut nodes, root_basis);
    let initial_position = nodes[nodes.len() - 1].global_end;
    let initial_rotation = nodes[nodes.len() - 1].global_basis;
    let mut goal_position = goal.position * scale;
    let mut goal_rotation = goal.rotation;
    if !use_pole && goal.influence != 1.0 {
        goal_position = initial_position.lerp(goal_position, goal.influence);
        goal_rotation = matrix_slerp(initial_rotation, goal_rotation, goal.influence);
    }
    // Break the rank-deficient straight-chain branch when the goal is behind the chain.
    if !use_pole && position_task && !nodes.is_empty() {
        let root = &nodes[0];
        let root_axis = (root.global_basis * DVec3::Y).normalize_or_zero();
        let chain_axis = (initial_position - root.global_start).normalize_or_zero();
        if chain_axis.dot(root_axis) > 1.0 - 1.0e-10
            && (goal_position - root.global_start).dot(root_axis) < 0.0
            && root.dof_count() > 0
            && !root.is_translation()
        {
            let root = &mut nodes[0];
            root.basis *= rotation_matrix(1.0e-6, root.dof_axis(0));
        }
    }
    let position_weight = if position_task {
        goal.position_weight
    } else {
        0.0
    };
    let orientation_weight = if orientation_task {
        goal.orientation_weight
    } else {
        0.0
    };
    let task_weight_sum = position_weight + orientation_weight;
    // Blender's IK_QTask::Weight stores m_weight squared, while task Jacobians and betas
    // are multiplied by m_weight itself.
    let position_task_weight = if position_task {
        (position_weight / task_weight_sum).sqrt()
    } else {
        0.0
    };
    let orientation_task_weight = if orientation_task {
        (orientation_weight / task_weight_sum).sqrt()
    } else {
        0.0
    };

    let mut pole_pre_rotation = false;
    if dof_count > 0 && task_weight_sum.abs() >= 1.0e-20 {
        if position_task && let Some(pole) = goal.pole {
            constrain_pole(
                &mut nodes,
                goal.position * scale,
                pole * scale,
                goal.pole_angle,
                &mut root_basis,
            );
            pole_pre_rotation = true;
        }
        let clamp_length =
            nodes.iter().map(|node| node.max_extension).sum::<f64>() / (2.0 * nodes.len() as f64);
        let mut jacobian = vec![vec![0.0; dof_count]; row_count];
        let mut beta = vec![0.0; row_count];
        for iteration in 0..max_iterations {
            update_transforms(&mut nodes, root_basis);
            for values in &mut jacobian {
                values.fill(0.0);
            }
            beta.fill(0.0);
            let tip_position = nodes[nodes.len() - 1].global_end;
            let mut row = 0;

            if position_task {
                let mut displacement = goal_position - tip_position;
                let distance = displacement.length();
                if distance > clamp_length {
                    displacement *= clamp_length / distance;
                }
                for axis in 0..3 {
                    beta[row + axis] = displacement[axis] * position_task_weight;
                }
                for node in nodes.iter().rev() {
                    let p = node.global_start - tip_position;
                    for local_dof in 0..node.dof_count() {
                        let column = node.dof_start + local_dof;
                        let direction = node.axis(local_dof);
                        let derivative = if node.is_translation() {
                            direction
                        } else {
                            p.cross(direction)
                        } * position_task_weight;
                        for axis in 0..3 {
                            jacobian[row + axis][column] = derivative[axis];
                        }
                    }
                }
                row += 3;
            }
            if orientation_task {
                let rotation = nodes[nodes.len() - 1].global_basis;
                let error_matrix = rotation * goal_rotation.transpose();
                let orientation_error = glam::DVec3::new(
                    -0.5 * (error_matrix.y_axis.z - error_matrix.z_axis.y),
                    -0.5 * (error_matrix.z_axis.x - error_matrix.x_axis.z),
                    -0.5 * (error_matrix.x_axis.y - error_matrix.y_axis.x),
                );
                for axis in 0..3 {
                    beta[row + axis] = orientation_error[axis] * orientation_task_weight;
                }
                for node in nodes.iter().rev() {
                    for local_dof in 0..node.dof_count() {
                        if node.is_translation() {
                            continue;
                        }
                        let column = node.dof_start + local_dof;
                        let derivative = node.axis(local_dof) * orientation_task_weight;
                        for axis in 0..3 {
                            jacobian[row + axis][column] = derivative[axis];
                        }
                    }
                }
            }

            let mut norm = 0.0_f64;
            let final_update;
            loop {
                let update = if goal.use_sdls {
                    solve_sdls_update(&jacobian, &beta, &dof_weights, &norm_weights)
                } else {
                    solve_dls_update(&jacobian, &beta, &dof_weights, &norm_weights)
                };
                let mut locked_any = false;
                let mut minimum_violation = 1.0e10_f64;
                let mut minimum = None;

                for (node_index, node) in nodes.iter_mut().enumerate() {
                    let (clamp, delta, was_clamped) = node.update_angle(&update);
                    if !was_clamped {
                        continue;
                    }
                    for dof in 0..node.dof_count() {
                        if !clamp[dof] || node.locked[dof] {
                            continue;
                        }
                        let violation = delta[dof].abs();
                        if violation < 1.0e-20 {
                            node.lock_dof(dof, delta, &mut jacobian, &mut beta, &dof_weights);
                            locked_any = true;
                        } else if violation < minimum_violation {
                            minimum_violation = violation;
                            minimum = Some((node_index, dof, delta));
                        }
                    }
                }
                if let Some((node_index, dof, delta)) = minimum {
                    nodes[node_index].lock_dof(dof, delta, &mut jacobian, &mut beta, &dof_weights);
                    locked_any = true;
                    norm = norm.max(minimum_violation);
                }
                if !locked_any {
                    final_update = update;
                    for node in &mut nodes {
                        node.unlock();
                        node.apply_update();
                    }
                    break;
                }
            }

            for dof in 0..dof_count {
                norm = norm.max((final_update[dof] * norm_weights[dof]).abs());
            }
            if norm < 1.0e-3 && iteration > 10 {
                break;
            }
        }
    }

    if pole_pre_rotation {
        nodes[segment_nodes[0]].prepend_basis(root_basis);
    }

    let mut basis_changes = Vec::with_capacity(segments.len());
    let mut stretch_ratios = Vec::with_capacity(segments.len());
    let mut parent_stretch = 1.0;
    for (index, _) in segments.iter().enumerate() {
        let node = &nodes[segment_nodes[index]];
        let basis_change = node.original_basis.transpose() * node.basis;
        let absolute_stretch = if let Some(stretch_node) = stretch_nodes[index] {
            let node = &nodes[stretch_node];
            if node.original_length == 0.0 {
                1.0
            } else {
                node.translation_y / node.original_length
            }
        } else {
            1.0
        };
        let local_stretch = if parent_stretch == 0.0 {
            absolute_stretch
        } else {
            absolute_stretch / parent_stretch
        };
        parent_stretch = absolute_stretch;
        basis_changes.push(basis_change);
        stretch_ratios.push(local_stretch);
    }

    LegacyResult {
        basis_changes,
        stretch_ratios,
    }
}

#[cfg(test)]
mod tests {
    use super::{SDLS_MAX_ANGLE, solve_dls_update, solve_sdls_update};

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
    }

    #[test]
    fn sdls_matches_known_rank_one_and_diagonal_updates() {
        let rank_one = vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        let update = solve_sdls_update(&rank_one, &[0.2, 0.0, 0.0], &[1.0, 1.0], &[1.0, 1.0]);
        close(update[0], 0.16);
        close(update[1], 0.0);
        // With fewer task rows than DoFs, Blender decomposes the transposed Jacobian. This
        // rank-one example has J = [1, 2, 0, 0] in its only active task axis.
        let wide_rank_one = vec![
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let update = solve_sdls_update(
            &wide_rank_one,
            &[0.2, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
        );
        close(update[0], 0.032);
        close(update[1], 0.064);
        close(update[2], 0.0);
        close(update[3], 0.0);

        let zero = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(
            solve_sdls_update(&zero, &[1.0, 2.0, 3.0], &[1.0, 1.0], &[1.0, 1.0]),
            vec![0.0, 0.0]
        );

        let diagonal = vec![
            vec![2.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let update = solve_sdls_update(
            &diagonal,
            &[0.3, 0.0, 0.0, 0.2, 0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 100.0],
        );
        close(update[0], 0.12);
        close(update[1], 0.16);
    }

    #[test]
    fn sdls_saturates_dof_weight_and_applies_global_angle_attenuation() {
        let jacobian = vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        let weighted = solve_sdls_update(&jacobian, &[1.0, 0.0, 0.0], &[0.25, 1.0], &[1.0, 1.0]);
        // damp / weight exceeds one, so Blender saturates this DoF factor at one.
        close(weighted[0], 0.4);
        close(weighted[1], 0.0);

        let capped = solve_sdls_update(&jacobian, &[100.0, 0.0, 0.0], &[0.25, 1.0], &[1.0, 1.0]);
        let before_global_cap = 1.6 * SDLS_MAX_ANGLE;
        let expected = before_global_cap * SDLS_MAX_ANGLE / (SDLS_MAX_ANGLE + before_global_cap);
        close(capped[0], expected);
        close(capped[1], 0.0);
        assert!(capped[0] < SDLS_MAX_ANGLE);
    }

    #[test]
    fn dls_matches_unregularized_diagonal_solution() {
        let diagonal = vec![
            vec![2.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        // The smallest singular value (1) exceeds ||beta|| / 0.1, so lambda is zero.
        let update = solve_dls_update(
            &diagonal,
            &[0.01, 0.0, 0.0, 0.02, 0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
        );
        close(update[0], 0.005);
        close(update[1], 0.02);

        let scalar = vec![vec![1.0], vec![0.0], vec![0.0]];
        let regularized = solve_dls_update(&scalar, &[1.0, 0.0, 0.0], &[1.0], &[1.0]);
        // d = 10, so lambda is capped at 10 and sigma / (sigma^2 + lambda) = 1/11.
        close(regularized[0], 1.0 / 11.0);
    }
}
