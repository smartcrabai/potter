use std::{env, path::PathBuf};

/// Finds Blender via `POTTER_BLENDER`, `PATH`, then the macOS app bundle.
///
/// Panics instead of returning `None` when `POTTER_REQUIRE_BLENDER` is non-empty, so CI cannot
/// silently skip Blender-gated tests.
pub(super) fn discover(mut usable: impl FnMut(PathBuf) -> Option<PathBuf>) -> Option<PathBuf> {
    let found = if let Some(path) = env::var_os("POTTER_BLENDER") {
        usable(path.into())
    } else {
        env::var_os("PATH")
            .and_then(|paths| {
                env::split_paths(&paths)
                    .map(|directory| directory.join("blender"))
                    .find_map(&mut usable)
            })
            .or_else(|| usable("/Applications/Blender.app/Contents/MacOS/Blender".into()))
    };
    assert!(
        found.is_some()
            || env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
        "POTTER_REQUIRE_BLENDER is set but no usable Blender executable was found"
    );
    found
}
