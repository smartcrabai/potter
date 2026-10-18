//! Custom-normal code tolerance shared by the subdivision and native normal oracle tests.
//!
//! Blender 5.2.2 does not store the same codes on every platform. Its linux-x64 and macOS arm64
//! builds differ by exactly one step in the second (beta) component of 11 of the 2,656 loop codes
//! in the custom-normal subdivision cases, and never in the first (alpha) component. Potter's
//! output differs from each platform's codes in the same way. `OpenSubdiv` is built on arm64 with
//! clang's default `-ffp-contract=on` (`build_files/build_environment/cmake/options.cmake` passes no
//! `-ffp-contract=off` to the deps), while the x86-64 baseline has no FMA to contract, so the
//! float rounding of the beta computation depends on the platform.
//!
//! The contract is therefore: alpha must match exactly, beta may differ by one step, and Auto
//! (`null`) matches only Auto.

/// The one code step that beta may differ by across platforms.
pub const BETA_STEP: i64 = 1;

/// Whether a stored code matches Blender's under the platform-variance contract.
///
/// `None` is Blender's Auto code. Auto matches only Auto. An authored pair matches when its alpha
/// component is equal and its beta component differs by at most [`BETA_STEP`].
pub fn code_within_beta_step(actual: Option<[i64; 2]>, expected: Option<[i64; 2]>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(a), Some(e)) => a[0] == e[0] && (a[1] - e[1]).abs() <= BETA_STEP,
        _ => false,
    }
}
