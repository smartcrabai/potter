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
