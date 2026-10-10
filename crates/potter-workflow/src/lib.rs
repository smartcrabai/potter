//! Agent workflows behind `pot workflow`, driven by jcode agents.

use potter_core::{
    error::{ErrorCode, PotError},
    response::Envelope,
};
use serde_json::Value;

mod potter_bridge;
mod refine;

/// One variant per workflow. Leave variants undocumented: the workflow's `Args` doc comment
/// supplies both its summary in `pot workflow -h` and its `--help` text.
#[derive(Debug, clap::Subcommand)]
pub enum Workflow {
    Refine(refine::Args),
}

/// What a workflow reports on stdout when it ends without an error.
struct Finished {
    summary: Value,
    reached_goal: bool,
}

/// Runs a workflow and returns the process exit code: 0 on success, 2 when the workflow finished
/// without reaching its goal, 1 on error. With `json`, stdout is one `pot` response envelope, as for
/// every other `pot` command; otherwise it is the bare summary and errors go to stderr.
#[must_use]
pub fn run(workflow: Workflow, json: bool) -> i32 {
    let (command, result) = match workflow {
        Workflow::Refine(args) => ("workflow refine", refine::run(args)),
    };
    match result {
        Ok(finished) => {
            if json {
                let _ = Envelope::success(command, None, finished.summary).emit(true);
            } else {
                println!("{}", finished.summary);
            }
            if finished.reached_goal { 0 } else { 2 }
        }
        Err(error) if json => Envelope::failure(
            Some(command.to_owned()),
            PotError::new(ErrorCode::InternalError, format!("{error:#}")),
        )
        .emit(true),
        Err(error) => {
            eprintln!("error: {error:#}");
            1
        }
    }
}
