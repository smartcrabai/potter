use std::ffi::OsString;

use clap::Parser;
use potter_core::{
    cli::CliCommand,
    error::{ErrorCode, PotError},
    response::Envelope,
};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "pot",
    version,
    about = "Headless, JSON-driven scene editor",
    long_about = "Create, inspect, edit, validate, render, and exchange Blender-compatible scenes through typed JSON."
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
    #[arg(long, global = true, help = "Print one JSON response envelope.")]
    json: bool,
}

fn main() {
    std::process::exit(run(std::env::args_os()));
}

fn run<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let arguments: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let json_output = arguments.iter().any(|arg| arg == "--json");
    match Cli::try_parse_from(arguments) {
        Ok(cli) => cli.command.execute().emit(cli.json),
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            i32::from(error.print().is_err())
        }
        Err(error) if json_output => Envelope::failure(
            None,
            PotError::with_details(
                ErrorCode::InvalidArgument,
                error.to_string(),
                json!({ "kind": format!("{:?}", error.kind()) }),
            ),
        )
        .emit(true),
        Err(error) => {
            let code = error.exit_code();
            if error.print().is_err() { 1 } else { code }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn help_and_version_succeed_even_with_json_flag() {
        assert_eq!(run(["pot", "--help", "--json"]), 0);
        assert_eq!(run(["pot", "--version", "--json"]), 0);
    }

    #[test]
    fn unknown_flag_is_invalid_argument_when_json_requested() {
        assert_eq!(run(["pot", "init", "scene", "--unknown", "--json"]), 2);
    }
}
