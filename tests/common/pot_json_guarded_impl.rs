use std::{
    error::Error,
    process::{Command, Output},
};

use serde_json::Value;

pub(super) fn pot_json_with_runner(
    arguments: &[&str],
    run: fn(Command) -> Result<Output, Box<dyn Error>>,
) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    let output = run(command)?;
    assert!(
        output.status.success(),
        "pot command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
