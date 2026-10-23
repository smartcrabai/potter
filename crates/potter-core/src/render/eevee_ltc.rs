//! EEVEE 5.2.2's linearly transformed cosine (LTC) light evaluation, for the glass closures of realtime Beauty.
//!
//! EEVEE integrates a GGX closure against a light's shape by mapping the lobe to a clamped cosine with the inverse
//! matrix that the LUT gives, and integrating that over the emitter's polygon or disk. Sources (Blender v5.2.2):
//! - `draw/engines/eevee/shaders/eevee_ltc_lib.bsl.hh`: `evaluate_quad`, `evaluate_disk`, `solve_cubic`,
//!   `edge_integral_vec`, `tangent_basis`, `diffuse_sphere_integral`.
//! - `draw/engines/eevee/shaders/eevee_ltc_lut_lib.bsl.hh`: the matrix's LUT coordinates `(roughness, sqrt(1 - cos θ))`
//!   and `unpack`.
//! - `draw/engines/eevee/shaders/eevee_light_lib.bsl.hh`: `light_ltc`, `light_shape_corners`,
//!   `light_sphere_disk_radius`.
//! - `draw/intern/shaders/draw_math_geom_lib.glsl` (`make_orthonormal_basis`) and
//!   `gpu/shaders/common/gpu_shader_math_matrix_construct_lib.glsl` (`from_up_axis`).
//!
//! GLSL matrices are column-major, as `glam::DMat3` is, so `float3x3(a, b, c)` is `DMat3::from_cols(a, b, c)`.
//! Blender's `ltc_mag_ggx` is not read by 5.2.2 and is not here: the matrices are normalised by construction.

use std::f64::consts::TAU;

use glam::{DMat3, DVec3};

use super::{
    eevee_transmission::{blend, texel_weights},
    eevee_transmission_tables::{disk_integral, ltc_matrix},
};

/// The packed inverse LTC matrix `(m11, m13, m31, m33)`, as `eevee::lut::ltc::unpack` reads it.
pub(crate) type PackedMatrix = [f64; 4];

/// The matrix of the identity: a plain clamped cosine (`eevee::lut::ltc::identity`).
#[cfg(test)]
const IDENTITY: PackedMatrix = [1.0, 0.0, 0.0, 1.0];

/// EEVEE's `sample_utility_tx` for the GGX matrix: the bilinear sample at `(roughness, sqrt(1 - cos θ))`.
pub(crate) fn sample_matrix(cos_theta: f64, roughness: f64) -> PackedMatrix {
    let (x0, x1, fx) = texel_weights(roughness);
    let (y0, y1, fy) = texel_weights((1.0 - cos_theta.clamp(0.0, 1.0)).sqrt());
    std::array::from_fn(|channel| {
        blend(
            blend(ltc_matrix(x0, y0, channel), ltc_matrix(x1, y0, channel), fx),
            blend(ltc_matrix(x0, y1, channel), ltc_matrix(x1, y1, channel), fx),
            fy,
        )
    })
}

/// `eevee::lut::ltc::unpack`: `float3x3(float3(v.x, 0, v.y), float3(0, 1, 0), float3(v.z, 0, v.w))`.
fn unpack(packed: PackedMatrix) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(packed[0], 0.0, packed[1]),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(packed[2], 0.0, packed[3]),
    )
}

/// `make_orthonormal_basis`: the tangent and bitangent of the unit vector `normal`.
fn orthonormal_basis(normal: DVec3) -> (DVec3, DVec3) {
    let up = if normal.z.abs() < 0.999_99 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let tangent = up.cross(normal).normalize_or_zero();
    (tangent, normal.cross(tangent))
}

/// `from_up_axis`: a basis whose third column is `up`, from Duff et al. (2017).
fn from_up_axis(up: DVec3) -> DMat3 {
    let z_sign = if up.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (z_sign + up.z);
    let b = up.x * up.y * a;
    DMat3::from_cols(
        DVec3::new(1.0 + z_sign * up.x * up.x * a, z_sign * b, -z_sign * up.x),
        DVec3::new(b, z_sign + up.y * up.y * a, -up.y),
        up,
    )
}

/// `tangent_basis`: the columns `T1`, `T2`, `N`, with `T1` along the view's tangent component.
fn tangent_basis(normal: DVec3, view: DVec3) -> DMat3 {
    let normal_view = normal.dot(view);
    if normal_view.abs() > 0.999_999 {
        return from_up_axis(normal);
    }
    let t1 = (view - normal * normal_view).normalize_or_zero();
    DMat3::from_cols(t1, normal.cross(t1), normal)
}

/// `diffuse_sphere_integral`: the tabulated clipped sphere integral at `avg_dir_z` and the form factor.
fn diffuse_sphere_integral(avg_dir_z: f64, form_factor: f64) -> f64 {
    let (x0, x1, fx) = texel_weights(avg_dir_z * 0.5 + 0.5);
    let (y0, y1, fy) = texel_weights(form_factor);
    blend(
        blend(disk_integral(x0, y0), disk_integral(x1, y0), fx),
        blend(disk_integral(x0, y1), disk_integral(x1, y1), fx),
        fy,
    )
}

/// `solve_cubic`: the three roots of `c0 + c1 x + c2 x² + c3 x³`, in the order EEVEE's extended Cardano solver gives.
fn solve_cubic(coefficients: [f64; 4]) -> [f64; 3] {
    let w = coefficients[3];
    let x = coefficients[0] / w;
    let y = coefficients[1] / w / 3.0;
    let z = coefficients[2] / w / 3.0;
    let a = w;
    let b = z;
    let c = y;
    let d = x;

    // The Hessian and the discriminant.
    let delta = DVec3::new(y - z * z, x - y * z, z * x - y * y);
    let discriminant = 4.0 * delta.x * delta.z - delta.y * delta.y;
    let sqrt_discriminant = discriminant.max(0.0).sqrt();

    // Algorithm A.
    let c_a = delta.x;
    let d_a = -2.0 * b * delta.x + delta.y;
    let theta_a = sqrt_discriminant.atan2(-d_a) / 3.0;
    let two_sqrt_c_a = 2.0 * (-c_a).max(0.0).sqrt();
    let x_1a = two_sqrt_c_a * theta_a.cos();
    let x_3a = two_sqrt_c_a * (theta_a + TAU / 3.0).cos();
    let xl = if x_1a + x_3a > 2.0 * b { x_1a } else { x_3a };
    let xlc = (xl - b, a);

    // Algorithm D.
    let c_d = delta.z;
    let d_d = -d * delta.y + 2.0 * c * delta.z;
    let theta_d = (d * sqrt_discriminant).atan2(-d_d) / 3.0;
    let two_sqrt_c_d = 2.0 * (-c_d).max(0.0).sqrt();
    let x_1d = two_sqrt_c_d * theta_d.cos();
    let x_3d = two_sqrt_c_d * (theta_d + TAU / 3.0).cos();
    let xs = if x_1d + x_3d < 2.0 * c { x_1d } else { x_3d };
    let xsc = (-d, xs + c);

    let e = xlc.1 * xsc.1;
    let f = -xlc.0 * xsc.1 - xlc.1 * xsc.0;
    let g = xlc.0 * xsc.0;
    let xmc = (c * f - b * g, -b * f + c * e);
    let mut root = [xsc.0 / xsc.1, xmc.0 / xmc.1, xlc.0 / xlc.1];
    if root[0] < root[1] && root[0] < root[2] {
        root = [root[1], root[0], root[2]];
    } else if root[2] < root[0] && root[2] < root[1] {
        root = [root[0], root[2], root[1]];
    }
    root
}

/// `edge_integral_vec`: the edge integral of Hill and Heitz's rectangle formula, for the arc between `v1` and `v2`.
fn edge_integral_vec(v1: DVec3, v2: DVec3) -> DVec3 {
    let x = v1.dot(v2);
    let y = x.abs();
    let a = 0.854_398_5 + (0.496_515_5 + 0.014_520_6 * y) * y;
    let b = 3.417_594_0 + (4.161_672_4 + y) * y;
    let v = a / b;
    let theta_sin_theta = if x > 0.0 {
        v
    } else {
        0.5 / (1.0 - x * x).max(1.0e-7).sqrt() - v
    };
    v1.cross(v2) * theta_sin_theta
}

/// `evaluate_quad`: the LTC form factor of a rectangle, from its four corners in world space (relative to the point).
pub(crate) fn evaluate_quad(corners: [DVec3; 4], normal: DVec3, view: DVec3, minv: DMat3) -> f64 {
    let minv = minv * tangent_basis(normal, view).transpose();
    let corners = corners.map(|corner| (minv * corner).normalize_or_zero());
    let mut average = DVec3::ZERO;
    for index in 0..4 {
        average += edge_integral_vec(corners[index], corners[(index + 1) % 4]);
    }
    // The sphere of the same solid angle: `form_factor_inv` is `inversesqrt(dot(avg, avg))`.
    let length = average.length();
    if length <= 0.0 {
        return 0.0;
    }
    let form_factor = length.clamp(0.0, 1.0);
    form_factor * diffuse_sphere_integral(average.z / length, form_factor)
}

/// `evaluate_disk`: the LTC form factor of a disk or an ellipse, from three of its bounding points in world space
/// (relative to the point), as `light_shape_corners` gives them.
pub(crate) fn evaluate_disk(normal: DVec3, view: DVec3, minv: DMat3, points: [DVec3; 3]) -> f64 {
    let rotation = tangent_basis(normal, view).transpose();
    let l0 = rotation * points[0];
    let l1 = rotation * points[1];
    let l2 = rotation * points[2];

    // The ellipse's centre and its two half axes, transformed by the inverse matrix.
    let centre = minv * (0.5 * (l0 + l2));
    let mut v1 = minv * (0.5 * (l1 - l2));
    let mut v2 = minv * (0.5 * (l1 - l0));

    // The ellipse's eigen-decomposition.
    let d11 = v1.dot(v1);
    let d22 = v2.dot(v2);
    let d12 = v1.dot(v2);
    let a;
    let mut inv_b;
    if d12.abs() / (d11 * d22).sqrt() > 0.000_7 {
        let trace = d11 + d22;
        let det = (-d12 * d12 + d11 * d22).sqrt();
        let u = 0.5 * (trace - 2.0 * det).sqrt();
        let w = 0.5 * (trace + 2.0 * det).sqrt();
        let e_max = (u + w) * (u + w);
        let e_min = (u - w) * (u - w);
        let (v1_, v2_) = if d11 > d22 {
            (d12 * v1 + (e_max - d11) * v2, d12 * v1 + (e_min - d11) * v2)
        } else {
            (d12 * v2 + (e_max - d22) * v1, d12 * v2 + (e_min - d22) * v1)
        };
        a = 1.0 / e_max;
        inv_b = e_min;
        v1 = v1_.normalize_or_zero();
        v2 = v2_.normalize_or_zero();
    } else {
        a = 1.0 / d11;
        inv_b = d22;
        v1 *= a.sqrt();
        v2 *= 1.0 / inv_b.sqrt();
    }

    // The front-facing ellipse with the same solid angle.
    let mut v3 = v1.cross(v2).normalize_or_zero();
    if centre.dot(v3) < 0.0 {
        v3 = -v3;
    }
    let l = v3.dot(centre);
    let inv_l = 1.0 / l;
    let x0 = v1.dot(centre) * inv_l;
    let y0 = v2.dot(centre) * inv_l;

    let ab = a * inv_b;
    inv_b *= inv_l * inv_l;
    let t = 1.0 + x0 * x0;
    let c0 = ab * inv_b;
    let c1 = ab * (t + y0 * y0) - c0 - inv_b;
    let c2 = inv_b - ab * t - (1.0 + y0 * y0);
    let [e1, e2, e3] = solve_cubic([c0, c1, c2, 1.0]);

    let average = (DMat3::from_cols(v1, v2, v3)
        * DVec3::new(ab * x0 / (ab - e2), y0 / (1.0 - e2), 1.0))
    .normalize_or_zero();
    let axis1 = 1.0 / (-e3 / e2).sqrt();
    let axis2 = 1.0 / (-e1 / e2).sqrt();
    let form_factor =
        (axis1 * axis2 / ((1.0 + axis1 * axis1) * (1.0 + axis2 * axis2)).sqrt()).clamp(0.0, 1.0);
    form_factor * diffuse_sphere_integral(average.z, form_factor)
}

/// The emitter of a light, as the LTC integrates it: the shape EEVEE builds from the light's type and size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Emitter {
    /// A disk of `radius` facing the receiver (`LIGHT_OMNI_DISK`, `LIGHT_SPOT_DISK`, and a sun's disk).
    Disk { radius: f64 },
    /// A sphere of `radius`, seen as the disk of the same solid angle (`LIGHT_OMNI_SPHERE`, `LIGHT_SPOT_SPHERE`).
    Sphere { radius: f64 },
    /// A rectangle (`LIGHT_RECT`) or an ellipse (`LIGHT_ELLIPSE`) with unit axes and half extents.
    Area {
        axis_u: DVec3,
        axis_v: DVec3,
        half_u: f64,
        half_v: f64,
        ellipse: bool,
    },
}

impl Emitter {
    /// `light_shape_radius`: the radius that EEVEE's facing attenuation uses (an area's is its half extents' length).
    pub(crate) fn shape_radius(self) -> f64 {
        match self {
            Self::Disk { radius } | Self::Sphere { radius } => radius,
            Self::Area { half_u, half_v, .. } => half_u.hypot(half_v),
        }
    }

    /// `light_ltc`: the LTC form factor of this emitter for a lobe with normal `normal`, viewed along `view`, through
    /// the packed inverse matrix `packed`. `to_centre` runs from the shading point to the emitter's centre, and its
    /// length is the distance (a sun's is one).
    pub(crate) fn form_factor(
        self,
        normal: DVec3,
        view: DVec3,
        packed: PackedMatrix,
        to_centre: DVec3,
    ) -> f64 {
        let minv = unpack(packed);
        match self {
            Self::Area {
                axis_u,
                axis_v,
                half_u,
                half_v,
                ellipse: false,
            } => {
                let u = axis_u * half_u;
                let v = axis_v * half_v;
                // `light_shape_corners`: the rectangle's corners, in the order of `evaluate_quad`.
                let corners = [
                    u - v + to_centre,
                    u + v + to_centre,
                    -(u - v) + to_centre,
                    -(u + v) + to_centre,
                ];
                evaluate_quad(corners, normal, view, minv)
            }
            Self::Area {
                axis_u,
                axis_v,
                half_u,
                half_v,
                ellipse: true,
            } => {
                let u = axis_u * half_u;
                let v = axis_v * half_v;
                let first = -u - v;
                let points = [first + to_centre, (u - v) + to_centre, -first + to_centre];
                evaluate_disk(normal, view, minv, points)
            }
            Self::Disk { radius } => {
                evaluate_disk(normal, view, minv, disk_points(radius, to_centre))
            }
            Self::Sphere { radius } => {
                let distance = to_centre.length();
                // `light_ltc` takes the whole hemisphere when the receiver is inside the sphere.
                if distance < radius {
                    return 1.0;
                }
                let size = radius / (1.0 - (radius / distance).powi(2)).max(1.0e-8).sqrt();
                evaluate_disk(normal, view, minv, disk_points(size, to_centre))
            }
        }
    }
}

/// `light_shape_corners` for a disk of `size` facing the receiver: three bounding points, relative to the point.
fn disk_points(size: f64, to_centre: DVec3) -> [DVec3; 3] {
    let (tangent, bitangent) = orthonormal_basis(to_centre.normalize_or_zero());
    let first = -tangent * size - bitangent * size;
    [
        first + to_centre,
        (tangent * size - bitangent * size) + to_centre,
        -first + to_centre,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LTC of the identity matrix is a clamped cosine lobe, so a small disk facing the receiver integrates to
    /// `r² / d²`: the solid angle `π r² / d²` times the lobe's `cos / π`.
    #[test]
    fn identity_disk_on_axis_is_its_solid_angle_over_pi() -> Result<(), String> {
        let (radius, distance) = (0.1, 1.0);
        let emitter = Emitter::Disk { radius };
        let form = emitter.form_factor(DVec3::Z, DVec3::Z, IDENTITY, DVec3::Z * distance);
        let expected = radius * radius / (distance * distance);
        if (form - expected).abs() > 0.05 * expected {
            return Err(format!("disk form factor {form} is not {expected}"));
        }
        Ok(())
    }

    /// A rectangle that covers the hemisphere integrates to one: the lobe is normalised.
    #[test]
    fn identity_rectangle_covering_the_hemisphere_is_one() -> Result<(), String> {
        let emitter = Emitter::Area {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u: 1.0e4,
            half_v: 1.0e4,
            ellipse: false,
        };
        let form = emitter.form_factor(DVec3::Z, DVec3::Z, IDENTITY, DVec3::Z * 1.0e-3);
        if (form - 1.0).abs() > 0.02 {
            return Err(format!(
                "a hemisphere-covering rectangle integrates to {form}"
            ));
        }
        Ok(())
    }

    /// An ellipse inscribed in a square has `π/4` of its area, so its form factor is below the square's and above
    /// three quarters of it on axis.
    #[test]
    fn inscribed_ellipse_is_below_its_square_and_above_three_quarters_of_it() -> Result<(), String>
    {
        let (half, distance) = (0.05, 1.0);
        let area = |ellipse| Emitter::Area {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u: half,
            half_v: half,
            ellipse,
        };
        let to_centre = DVec3::Z * distance;
        let square = area(false).form_factor(DVec3::Z, DVec3::Z, IDENTITY, to_centre);
        let ellipse = area(true).form_factor(DVec3::Z, DVec3::Z, IDENTITY, to_centre);
        if !(ellipse < square && ellipse > 0.75 * square) {
            return Err(format!(
                "ellipse {ellipse} and square {square} do not order as expected"
            ));
        }
        Ok(())
    }

    #[test]
    fn sample_matrix_at_normal_incidence_and_zero_roughness_is_eevees_texel() -> Result<(), String>
    {
        let packed = sample_matrix(1.0, 0.0);
        if (packed[0] - 1.0).abs() > 1.0e-6 || packed[1].abs() > 1.0e-6 {
            return Err(format!("the matrix at (0, 0) is {packed:?}"));
        }
        Ok(())
    }
}
