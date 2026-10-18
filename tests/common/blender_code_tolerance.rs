//! Custom-normal code tolerance shared by the subdivision and native normal oracle tests.
//!
//! Blender 5.2.2 does not store the same codes on every platform. On Linux x86-64 and macOS arm64
//! the subdivided positions differ by one to a few ulp, and 11 of the 2,656 loop codes in the
//! custom-normal subdivision cases differ, always by one step in the second component (Blender's
//! linux-x64 build against its macOS arm64 build). OpenSubdiv is built on arm64 with clang's default
//! `-ffp-contract=on` (`build_files/build_environment/cmake/options.cmake` passes no
//! `-ffp-contract=off` for the deps), so FMA contraction changes its float rounding there, and the
//! x86-64 baseline has no FMA to contract. The contract for codes is therefore one code step per
//! component, with Auto status (`null`) matching exactly.

/// One Blender code step: the 16-bit code scale is `SHRT_MAX`, so a step is one unit of the code.
pub const CODE_STEP: i64 = 1;

/// Whether an authored code pair matches Blender's within [`CODE_STEP`] per component.
///
/// `None` is Blender's Auto code. Auto matches only Auto: a corner that Blender leaves automatic
/// must not become an authored code, and the reverse is equally a status change.
pub fn code_within_step(actual: Option<[i64; 2]>, expected: Option<[i64; 2]>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(a), Some(e)) => (a[0] - e[0]).abs() <= CODE_STEP && (a[1] - e[1]).abs() <= CODE_STEP,
        _ => false,
    }
}
