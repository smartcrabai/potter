//! Exact rational arithmetic for the Mesh Arrangements port.
//!
//! `Rat`, `Rat2`, and `Rat3` mirror Blender's `mpq_class`, `mpq2`, and `mpq3` (GMP). Blender
//! decides most plane-side tests in double precision first, using a rigorous error bound
//! (`filter_plane_side`), and falls back to exact arithmetic only when the bound is inconclusive.
//! The same two-stage structure is kept here.

use glam::DVec3;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

/// Exact rational scalar (Blender `mpq_class`).
pub type Rat = BigRational;
/// Exact rational 2-vector (Blender `mpq2`).
pub type Rat2 = [Rat; 2];
/// Exact rational 3-vector (Blender `mpq3`).
pub type Rat3 = [Rat; 3];

/// Exact value of a finite `f64`. Every `f32` converts exactly through `f64`.
///
/// Callers pass finite values only (input meshes are validated upstream).
#[must_use]
pub fn rat_from_f64(value: f64) -> Rat {
    debug_assert!(value.is_finite(), "exact arithmetic needs finite inputs");
    Rat::from_float(value).unwrap_or_else(Rat::zero)
}

/// Nearest double to a rational, used only for float filters and output positions.
#[must_use]
pub fn rat_to_f64(value: &Rat) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

/// Exact rational 3-vector from finite doubles.
#[must_use]
pub fn rat3_from_dvec3(value: DVec3) -> Rat3 {
    [
        rat_from_f64(value.x),
        rat_from_f64(value.y),
        rat_from_f64(value.z),
    ]
}

/// Sign of a rational: -1, 0, or 1.
#[must_use]
pub fn sgn(value: &Rat) -> i8 {
    if value.is_zero() {
        0
    } else if value.is_negative() {
        -1
    } else {
        1
    }
}

#[must_use]
pub fn sub3(a: &Rat3, b: &Rat3) -> Rat3 {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}

#[must_use]
pub fn add3(a: &Rat3, b: &Rat3) -> Rat3 {
    [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
}

/// `scale * a`.
#[must_use]
pub fn scale3(scale: &Rat, a: &Rat3) -> Rat3 {
    [scale * &a[0], scale * &a[1], scale * &a[2]]
}

#[must_use]
pub fn dot3(a: &Rat3, b: &Rat3) -> Rat {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

#[must_use]
pub fn cross3(a: &Rat3, b: &Rat3) -> Rat3 {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}

/// Exact orientation of `c` relative to the directed line `a -> b`: sign of
/// `cross(b - a, c - a)`. Positive means `c` is left of the line (CCW turn).
#[must_use]
pub fn orient2d(a: &Rat2, b: &Rat2, c: &Rat2) -> i8 {
    let bx = &b[0] - &a[0];
    let by = &b[1] - &a[1];
    let cx = &c[0] - &a[0];
    let cy = &c[1] - &a[1];
    sgn(&(bx * cy - by * cx))
}

/// Exact sign of `dot(p - plane_p, plane_no)`: which side of the plane `p` is on.
#[must_use]
pub fn plane_side_exact(p: &Rat3, plane_p: &Rat3, plane_no: &Rat3) -> i8 {
    sgn(&dot3(&sub3(p, plane_p), plane_no))
}

/// Blender `supremum_dot_cross`: bound on `dot(cross(a, b), cross(a, b))` evaluated in double
/// precision with absolute values, used to derive the error bound of a float filter.
#[must_use]
pub fn supremum_dot_cross(a: DVec3, b: DVec3) -> f64 {
    let (aa, ab) = (a.abs(), b.abs());
    let c = DVec3::new(
        aa.y * ab.z + aa.z * ab.y,
        aa.z * ab.x + aa.x * ab.z,
        aa.x * ab.y + aa.y * ab.x,
    );
    c.dot(c)
}

/// Blender's `index_plane_side`: error-analysis index of the plane-side expression.
const INDEX_PLANE_SIDE: f64 = (3 + 2 * 15) as f64;

/// Double-precision side test of `p` against the plane through `plane_p` with normal `plane_no`.
///
/// Returns 1 or -1 when the sign is certain under Blender's error bound, and 0 when the answer
/// is undecided and an exact test is needed. Blender's `filter_plane_side`.
#[must_use]
pub fn filter_plane_side(p: DVec3, plane_p: DVec3, plane_no: DVec3) -> i8 {
    let d = (p - plane_p).dot(plane_no);
    if d == 0.0 {
        return 0;
    }
    let supremum = (p.abs() + plane_p.abs()).dot(plane_no.abs());
    let err_bound = supremum * INDEX_PLANE_SIDE * f64::EPSILON;
    if d.abs() > err_bound {
        if d > 0.0 { 1 } else { -1 }
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rat {
        Rat::from_integer(n.into())
    }

    fn r2(x: i64, y: i64) -> Rat2 {
        [r(x), r(y)]
    }

    #[test]
    fn orient2d_signs() {
        let a = r2(0, 0);
        let b = r2(1, 0);
        assert_eq!(orient2d(&a, &b, &r2(0, 1)), 1);
        assert_eq!(orient2d(&a, &b, &r2(0, -1)), -1);
        assert_eq!(orient2d(&a, &b, &r2(5, 0)), 0);
    }

    #[test]
    fn orient2d_sees_offsets_below_double_resolution() {
        // 1 + 2^-80 rounds to 1.0 in f64; the exact predicate must still see the offset.
        let eps = rat_from_f64(2.0_f64.powi(-80));
        let a = r2(0, 0);
        let b = r2(1, 0);
        let above = [r(1) + &eps, r(1)];
        let below = [r(1) + &eps, -r(1)];
        assert_eq!(orient2d(&a, &b, &above), 1);
        assert_eq!(orient2d(&a, &b, &below), -1);
    }

    #[test]
    fn plane_side_exact_matches_hand_computed() {
        let p0 = [r(0), r(0), r(0)];
        let n = [r(0), r(0), r(1)];
        assert_eq!(plane_side_exact(&[r(3), r(-2), r(5)], &p0, &n), 1);
        assert_eq!(plane_side_exact(&[r(3), r(-2), r(-5)], &p0, &n), -1);
        assert_eq!(plane_side_exact(&[r(3), r(-2), r(0)], &p0, &n), 0);
    }

    #[test]
    fn filter_decides_only_when_certain() {
        let plane_p = DVec3::ZERO;
        let plane_no = DVec3::Z;
        assert_eq!(
            filter_plane_side(DVec3::new(1.0, 0.0, 1.0), plane_p, plane_no),
            1
        );
        assert_eq!(
            filter_plane_side(DVec3::new(1.0, 0.0, -1.0), plane_p, plane_no),
            -1
        );
        assert_eq!(
            filter_plane_side(DVec3::new(1.0, 0.0, 0.0), plane_p, plane_no),
            0
        );
    }

    #[test]
    fn cross_and_dot() {
        let x = [r(1), r(0), r(0)];
        let y = [r(0), r(1), r(0)];
        let z = cross3(&x, &y);
        assert_eq!(z, [r(0), r(0), r(1)]);
        assert_eq!(dot3(&z, &z), r(1));
    }

    #[test]
    fn f32_inputs_convert_exactly() {
        let value = 0.1_f32;
        assert_eq!(
            rat_to_f64(&rat_from_f64(f64::from(value))),
            f64::from(value)
        );
    }
}
