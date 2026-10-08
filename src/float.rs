use std::cmp::Ordering;

pub(crate) fn equal_f64(left: f64, right: f64) -> bool {
    left.partial_cmp(&right) == Some(Ordering::Equal)
}

pub(crate) fn equal_f32(left: f32, right: f32) -> bool {
    left.partial_cmp(&right) == Some(Ordering::Equal)
}

pub(crate) fn equal_f64_array<const N: usize>(left: &[f64; N], right: &[f64; N]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| equal_f64(*left, *right))
}

#[cfg(test)]
mod tests {
    use super::{equal_f32, equal_f64, equal_f64_array};

    #[test]
    fn scalar_equality_preserves_ieee_comparison_boundaries() {
        assert!(equal_f64(2.5, 2.5));
        assert!(!equal_f64(2.5, 2.0));
        assert!(equal_f64(-0.0, 0.0));
        assert!(equal_f64(f64::INFINITY, f64::INFINITY));
        assert!(!equal_f64(f64::NAN, f64::NAN));

        assert!(equal_f32(2.5, 2.5));
        assert!(!equal_f32(2.5, 2.0));
        assert!(equal_f32(-0.0, 0.0));
        assert!(!equal_f32(f32::NAN, f32::NAN));
    }

    #[test]
    fn array_equality_requires_every_element_to_match() {
        assert!(equal_f64_array(&[1.0, -0.0], &[1.0, 0.0]));
        assert!(!equal_f64_array(&[1.0, 2.0], &[1.0, 2.5]));
        assert!(!equal_f64_array(&[f64::NAN], &[f64::NAN]));
        assert!(equal_f64_array::<0>(&[], &[]));
    }
}

#[cfg(kani)]
mod kani_verification {
    use super::{equal_f32, equal_f64, equal_f64_array};

    #[kani::proof]
    fn f64_equality_is_ieee_equality() {
        let left: f64 = kani::any();
        let right: f64 = kani::any();
        assert!(equal_f64(left, right) == (left == right));
        assert!(equal_f64(left, right) == equal_f64(right, left));
        assert!(!equal_f64(f64::NAN, right));
        assert!(equal_f64(-0.0, 0.0));
    }

    #[kani::proof]
    fn f32_equality_is_ieee_equality() {
        let left: f32 = kani::any();
        let right: f32 = kani::any();
        assert!(equal_f32(left, right) == (left == right));
        assert!(!equal_f32(left, f32::NAN));
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn f64_array_equality_requires_every_component_to_match() {
        let left: [f64; 3] = kani::any();
        let right: [f64; 3] = kani::any();
        let expected = left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
        assert!(equal_f64_array(&left, &right) == expected);
    }
}
