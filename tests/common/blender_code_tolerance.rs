//! Custom-normal code tolerance shared by the subdivision and native normal oracle tests.
//!
//! Blender 5.2.2 does not store the same codes on every platform. On the 147-case corpus, which had
//! 11 custom-normal cases (2,656 loops), its linux-x64 and macOS arm64 builds differed by exactly one
//! step in the second (beta) component of 11 loop codes, and never in the first (alpha) component.
//! The three cases added since have not been compared across platforms. Potter's output differs from
//! each platform's codes in the same way. `OpenSubdiv` is
//! built on arm64 with clang's default `-ffp-contract=on` (`build_files/build_environment/cmake/`
//! `options.cmake` passes no `-ffp-contract=off` to the deps), while the x86-64 baseline has no FMA
//! to contract, so the float rounding of the beta computation depends on the platform.
//!
//! The contract is therefore: alpha must match exactly, beta may differ by one step, and Auto
//! (`null`) matches only Auto. Callers handle Auto themselves.

/// The one code step that beta may differ by across platforms.
const BETA_STEP: i64 = 1;

/// Whether an authored code pair matches Blender's under the platform-variance contract: alpha
/// must be equal and beta may differ by at most [`BETA_STEP`].
pub fn code_within_beta_step(actual: [i64; 2], expected: [i64; 2]) -> bool {
    actual[0] == expected[0] && (actual[1] - expected[1]).abs() <= BETA_STEP
}
