use std::error::Error;

use serde_json::Value;

use crate::process_escaped::run_guarded_escaped_newlines;

#[path = "pot_json_guarded_impl.rs"]
mod implementation;

pub fn pot_json_escaped_newlines(arguments: &[&str]) -> Result<Value, Box<dyn Error>> {
    implementation::pot_json_with_runner(arguments, run_guarded_escaped_newlines)
}
