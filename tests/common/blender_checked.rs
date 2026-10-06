use std::{path::PathBuf, process::Command};

use crate::process::run_guarded;

#[path = "blender_discovery.rs"]
mod discovery;

pub fn blender_executable() -> Option<PathBuf> {
    discovery::discover(|path| {
        if !path.is_file() {
            return None;
        }
        let mut command = Command::new(&path);
        command.arg("--version");
        run_guarded(command)
            .ok()
            .filter(|output| output.status.success())
            .map(|_| path)
    })
}
