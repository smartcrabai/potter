//! Blender side of the third-party VRM import tests.
//!
//! Pins the VRM Add-on archive from `tests/vrm-consumer/blender/vrm-addon.lock.json`, caches it under
//! `target/vrm-addon/`, installs it into an isolated `BLENDER_USER_RESOURCES`, and runs
//! `tests/vrm-import/make_third_party.py` to write a VRM through the add-on's own exporter.
//!
//! A missing Blender, a missing `curl`, a failed download, or a failed install is a skip (`Gate::Skip`).
//! A sha256 mismatch is a hard failure.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::Value;
use sha2::Digest;
use tempfile::TempDir;

use crate::blender_checked::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const LOCK_PATH: &str = "tests/vrm-consumer/blender/vrm-addon.lock.json";
const GENERATOR_PATH: &str = "tests/vrm-import/make_third_party.py";
const ROUND_TRIP_PATH: &str = "tests/vrm-import/addon_round_trip.py";
const ADDON_URL_PREFIX: &str =
    "https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/";
const GENERATED_MARKER: &str = "POTTER_VRM_GENERATED";

/// A Blender executable and an isolated user-resources directory with the pinned add-on installed.
pub struct Generator {
    blender: PathBuf,
    user_resources: TempDir,
}

pub enum Gate {
    Ready(Generator),
    Skip(String),
}

/// Finds Blender and installs the pinned add-on. Under `POTTER_REQUIRE_BLENDER` a skip is a failure, so
/// the Blender-gated tests cannot pass without running.
pub fn setup() -> TestResult<Gate> {
    match setup_or_skip()? {
        Gate::Skip(reason)
            if std::env::var_os("POTTER_REQUIRE_BLENDER")
                .is_some_and(|value| !value.is_empty()) =>
        {
            Err(
                format!("POTTER_REQUIRE_BLENDER is set but the VRM Add-on gate skipped: {reason}")
                    .into(),
            )
        }
        gate => Ok(gate),
    }
}

/// Finds Blender and installs the pinned add-on, or explains why this environment cannot run it.
fn setup_or_skip() -> TestResult<Gate> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(blender) = blender_executable() else {
        return Ok(Gate::Skip(
            "no executable Blender was found (POTTER_BLENDER, PATH, or the macOS app bundle)"
                .to_owned(),
        ));
    };
    let lock: Value = serde_json::from_slice(&fs::read(root.join(LOCK_PATH))?)?;
    let url = lock["url"]
        .as_str()
        .ok_or("vrm-addon.lock.json has no url")?;
    let expected = lock["sha256"]
        .as_str()
        .ok_or("vrm-addon.lock.json has no sha256")?;
    if !url.starts_with(ADDON_URL_PREFIX) {
        return Err(format!("lock url {url} is not a release asset of the VRM Add-on").into());
    }
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("vrm-addon.lock.json sha256 must be 64 lowercase hex digits".into());
    }
    let archive = match fetch_archive(&root, url, expected)? {
        Ok(archive) => archive,
        Err(reason) => return Ok(Gate::Skip(reason)),
    };
    match install(&blender, &archive)? {
        Ok(user_resources) => Ok(Gate::Ready(Generator {
            blender,
            user_resources,
        })),
        Err(reason) => Ok(Gate::Skip(reason)),
    }
}

impl Generator {
    /// Writes one VRM from the pinned add-on. `spec_version` is `"1.0"` (VRM 1.0) or `"0.0"` (VRM 0.x).
    pub fn generate(&self, output: &Path, spec_version: &str) -> TestResult {
        self.run(GENERATOR_PATH, &[output.as_os_str(), spec_version.as_ref()])
    }

    /// Imports `input` with the pinned add-on and exports it again, unchanged, to `output`.
    pub fn round_trip(&self, input: &Path, output: &Path) -> TestResult {
        self.run(ROUND_TRIP_PATH, &[input.as_os_str(), output.as_os_str()])
    }

    fn run(&self, script: &str, args: &[&std::ffi::OsStr]) -> TestResult {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let result = Command::new(&self.blender)
            .args([
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
            ])
            .arg(root.join(script))
            .arg("--")
            .args(args)
            .env("BLENDER_USER_RESOURCES", self.user_resources.path())
            .output()?;
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        if !result.status.success() || !stdout.contains(GENERATED_MARKER) {
            return Err(format!(
                "Blender script {script} failed ({}): stdout={stdout} stderr={stderr}",
                result.status
            )
            .into());
        }
        Ok(())
    }
}

/// Returns the verified cached or freshly downloaded archive. `Err(String)` is a skip reason;
/// the outer `Err` is a hard failure (sha256 mismatch or I/O).
fn fetch_archive(root: &Path, url: &str, expected: &str) -> TestResult<Result<PathBuf, String>> {
    let name = url.rsplit('/').next().unwrap_or_default();
    let directory = root.join("target/vrm-addon");
    let archive = directory.join(name);
    if archive.is_file() {
        verify_sha256(&archive, expected)?;
        return Ok(Ok(archive));
    }
    fs::create_dir_all(&directory)?;
    // A unique partial file per call keeps parallel tests from writing the same download.
    let partial = tempfile::Builder::new()
        .suffix(".part")
        .tempfile_in(&directory)?;
    let download = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "120",
            "--output",
        ])
        .arg(partial.path())
        .arg(url)
        .output();
    match download {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            return Ok(Err(format!(
                "VRM Add-on download failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Err(error) => {
            return Ok(Err(format!(
                "VRM Add-on download could not start curl: {error}"
            )));
        }
    }
    verify_sha256(partial.path(), expected)?;
    partial.persist(&archive).map_err(|error| error.error)?;
    Ok(Ok(archive))
}

fn verify_sha256(path: &Path, expected: &str) -> TestResult {
    let actual = hex::encode(sha2::Sha256::digest(fs::read(path)?));
    if actual != expected {
        return Err(format!(
            "sha256 mismatch for {}: lock pins {expected}, got {actual}; delete the file and rerun",
            path.display()
        )
        .into());
    }
    Ok(())
}

/// Installs the archive into a fresh user-resources directory. `Err(String)` is a skip reason.
fn install(blender: &Path, archive: &Path) -> TestResult<Result<TempDir, String>> {
    let user_resources = tempfile::tempdir()?;
    let output = Command::new(blender)
        .args([
            "-b",
            "--factory-startup",
            "--command",
            "extension",
            "install-file",
            "-r",
            "user_default",
            "-e",
        ])
        .arg(archive)
        .env("BLENDER_USER_RESOURCES", user_resources.path())
        .output()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // The installed tree is the result. Concurrent Blender installs can exit 0 without printing the
    // `STATUS Installed "vrm"` line (it is lost while the tree is complete), so the line is not required.
    // A real failure (for example a truncated zip) is reported by Blender as an `ERROR` line, also with exit 0.
    let manifest = user_resources
        .path()
        .join("extensions/user_default/vrm/blender_manifest.toml");
    if output.status.success() && !text.contains("ERROR") && manifest.is_file() {
        return Ok(Ok(user_resources));
    }
    let tail: String = text
        .chars()
        .rev()
        .take(1000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Ok(Err(format!(
        "VRM Add-on install failed ({}): {tail}",
        output.status
    )))
}
