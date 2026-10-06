use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

pub fn run_blender_script(
    blender: &Path,
    script_name: &str,
    script: &str,
    root: &Path,
    trailing_args: &[&Path],
    failure_message: &str,
    run: fn(Command) -> Result<Output, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let script_path = root.join(script_name);
    fs::write(&script_path, script)?;
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .arg("--")
        .arg(root);
    for arg in trailing_args {
        command.arg(arg);
    }
    let output = run(command)?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "{failure_message}: stdout={stdout} stderr={stderr}",
    );
    Ok(())
}
