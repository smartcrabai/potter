use std::{
    error::Error,
    process::{Command, Output},
};

use crate::process::run_guarded_with_separator;

pub fn run_guarded_escaped_newlines(command: Command) -> Result<Output, Box<dyn Error>> {
    run_guarded_with_separator(command, "\\n")
}
