use std::error::Error;

use serde_json::Value;

use crate::process::run_guarded;

#[path = "pot_json_guarded_impl.rs"]
mod implementation;

pub fn pot_json(arguments: &[&str]) -> Result<Value, Box<dyn Error>> {
    implementation::pot_json_with_runner(arguments, run_guarded)
}
