use std::{error::Error, process::Command};

use serde_json::Value;

pub fn pot_json(arguments: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(arguments)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
