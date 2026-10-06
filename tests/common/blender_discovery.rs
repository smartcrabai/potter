use std::{env, path::PathBuf};

pub(super) fn discover(mut usable: impl FnMut(PathBuf) -> Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(path.into());
    }
    let found = env::split_paths(&env::var_os("PATH")?)
        .map(|directory| directory.join("blender"))
        .find_map(&mut usable);
    found.or_else(|| usable("/Applications/Blender.app/Contents/MacOS/Blender".into()))
}
