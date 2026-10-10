use std::{error::Error, fs, process::Command};

use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

#[test]
fn workflow_help_lists_refine() -> Result<(), Box<dyn Error>> {
    let output = pot().args(["workflow", "--help"]).output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout
            .lines()
            .any(|line| line.trim_start().starts_with("refine ")),
        "{stdout}"
    );
    Ok(())
}

#[test]
fn refine_requires_an_image() -> Result<(), Box<dyn Error>> {
    let output = pot().args(["workflow", "refine"]).output()?;
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8(output.stderr)?.contains("--image"));
    Ok(())
}

#[test]
fn refine_checks_every_image_before_creating_output() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let first = directory.path().join("front.png");
    fs::write(&first, b"png bytes are not decoded locally")?;
    let second = directory.path().join("side.bmp");
    let out = directory.path().join("out");

    let output = pot()
        .args(["workflow", "refine", "-i"])
        .arg(&first)
        .arg("--image")
        .arg(&second)
        .arg("-o")
        .arg(&out)
        .output()?;

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("side.bmp"), "{stderr}");
    assert!(!out.exists());
    Ok(())
}

#[test]
fn refine_errors_are_envelopes_with_json() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let image = directory.path().join("side.bmp");

    let output = pot()
        .args(["workflow", "refine", "--json", "-i"])
        .arg(&image)
        .output()?;

    assert_eq!(output.status.code(), Some(1));
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["ok"], false);
    assert_eq!(envelope["command"], "workflow refine");
    assert_eq!(envelope["error"]["code"], "INTERNAL_ERROR");
    assert!(
        envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("side.bmp"))
    );
    Ok(())
}
