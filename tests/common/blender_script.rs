use std::{error::Error, fs, path::Path, process::Command};

pub fn run_blender_script(
    blender: &Path,
    script_name: &str,
    script: &str,
    root: &Path,
    failure_message: &str,
) -> Result<(), Box<dyn Error>> {
    let script_path = root.join(script_name);
    fs::write(&script_path, script)?;
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .arg("--")
        .arg(root)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "{failure_message}: stdout={stdout} stderr={stderr}"
    );
    Ok(())
}
