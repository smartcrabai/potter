/// EEVEE stores and re-accumulates its combined film in RGBA16F. The per-sample
/// accumulation order can differ by one representable half-float value.
pub fn assert_eevee_film_close(actual: f64, expected: f64, context: &str) {
    let magnitude = expected.abs();
    let mut ulp = 1.0 / 16_777_216.0;
    while magnitude >= ulp * 2048.0 {
        ulp *= 2.0;
    }
    assert!(
        (actual - expected).abs() <= ulp,
        "{context}: Blender {expected:?}, Potter {actual:?} (tolerance {ulp})"
    );
}
