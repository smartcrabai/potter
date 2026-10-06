use std::{
    error::Error,
    io::{self, Read},
    process::{Command, Output, Stdio},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

fn collect_output(reader: JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other("child output reader panicked"))?
}

pub fn run_guarded(command: Command) -> Result<Output, Box<dyn Error>> {
    run_guarded_with_separator(command, "\n")
}

pub(crate) fn run_guarded_with_separator(
    mut command: Command,
    separator: &str,
) -> Result<Output, Box<dyn Error>> {
    let command_line = format!("{command:?}");
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdout = child.stdout.take().ok_or("missing child stdout")?;
    let mut stderr = child.stderr.take().ok_or("missing child stderr")?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let deadline = Instant::now() + CHILD_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let stdout = collect_output(stdout_reader)?;
            let stderr = collect_output(stderr_reader)?;
            return Err(format!(
                "child process exceeded {}s: {command_line}{separator}stdout={}{separator}stderr={}",
                CHILD_TIMEOUT.as_secs(),
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr),
            )
            .into());
        }
        thread::sleep(Duration::from_millis(50));
    };
    Ok(Output {
        status,
        stdout: collect_output(stdout_reader)?,
        stderr: collect_output(stderr_reader)?,
    })
}
