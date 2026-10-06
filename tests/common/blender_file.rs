use std::path::PathBuf;

#[path = "blender_discovery.rs"]
mod discovery;

pub fn blender_executable() -> Option<PathBuf> {
    discovery::discover(|path| path.is_file().then_some(path))
}
