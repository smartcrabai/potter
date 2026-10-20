use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

fn main() -> io::Result<()> {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not set CARGO_MANIFEST_DIR",
        )
    })?);
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compiler = Command::new(rustc).arg("-vV").output()?;
    assert!(
        compiler.status.success(),
        "running RUSTC -vV failed: {}",
        String::from_utf8_lossy(&compiler.stderr)
    );

    let mut hasher = Sha256::new();
    hasher.update(b"potter-core-build-identity-v1\0");
    hash_field(&mut hasher, "metadata/rustc -vV", &compiler.stdout);
    let target =
        env::var("TARGET").map_err(|error| io::Error::new(io::ErrorKind::NotFound, error))?;
    hash_field(&mut hasher, "metadata/TARGET", target.as_bytes());
    for name in [
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_CFG_TARGET_FEATURE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        let value = match env::var_os(name) {
            Some(value) => value.into_string().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Cargo build metadata {name} is not UTF-8"),
                )
            })?,
            None => String::new(),
        };
        hash_field(&mut hasher, &format!("metadata/{name}"), value.as_bytes());
    }

    let mut features = Vec::new();
    for (name, value) in env::vars_os() {
        let Some(name) = name.as_os_str().to_str() else {
            continue;
        };
        if !name.starts_with("CARGO_FEATURE_") {
            continue;
        }
        let name = name.to_owned();
        let value = value.into_string().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Cargo feature {name} is not UTF-8"),
            )
        })?;
        println!("cargo:rerun-if-env-changed={name}");
        features.push((name, value));
    }
    features.sort_unstable();
    for (name, value) in features {
        hash_field(&mut hasher, &format!("metadata/{name}"), value.as_bytes());
    }

    let mut files = Vec::new();
    let mut watched = BTreeSet::new();
    for directory in ["src", "native", "vendor"] {
        let root = manifest_dir.join(directory);
        collect_tree(
            &root,
            &root,
            &format!("package/{directory}"),
            &mut files,
            &mut watched,
        )?;
    }
    for name in ["build.rs", "Cargo.toml", "Cargo.lock"] {
        add_file(
            &manifest_dir.join(name),
            &format!("package/{name}"),
            &mut files,
            &mut watched,
        )?;
    }
    if let Some(workspace_dir) = manifest_dir.parent().and_then(Path::parent) {
        for name in ["Cargo.toml", "Cargo.lock"] {
            add_file(
                &workspace_dir.join(name),
                &format!("workspace/{name}"),
                &mut files,
                &mut watched,
            )?;
        }
    }

    files.sort_unstable_by(|left: &(String, PathBuf), right| left.0.cmp(&right.0));
    for (relative_path, path) in files {
        hash_field(&mut hasher, &relative_path, &fs::read(path)?);
    }
    for path in watched {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let mut identity = String::with_capacity(64);
    for byte in hasher.finalize() {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        identity.push(char::from(HEX[usize::from(byte >> 4)]));
        identity.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    println!("cargo:rustc-env=POTTER_BUILD_IDENTITY={identity}");
    Ok(())
}

fn collect_tree(
    source_root: &Path,
    directory: &Path,
    logical_root: &str,
    files: &mut Vec<(String, PathBuf)>,
    watched: &mut BTreeSet<PathBuf>,
) -> io::Result<()> {
    watched.insert(directory.to_path_buf());
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(directory)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        watched.insert(path.clone());
        let file_type = fs::symlink_metadata(&path)?.file_type();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if entry.file_name() != "target" {
                collect_tree(source_root, &path, logical_root, files, watched)?;
            }
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(source_root)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let relative = relative
                .components()
                .map(|component| {
                    component.as_os_str().to_str().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 source path")
                    })
                })
                .collect::<io::Result<Vec<_>>>()?
                .join("/");
            files.push((format!("{logical_root}/{relative}"), path));
        }
    }
    Ok(())
}

fn add_file(
    path: &Path,
    logical_path: &str,
    files: &mut Vec<(String, PathBuf)>,
    watched: &mut BTreeSet<PathBuf>,
) -> io::Result<()> {
    watched.insert(path.to_path_buf());
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => {
            files.push((logical_path.to_owned(), path.to_path_buf()));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    Ok(())
}

fn hash_field(hasher: &mut Sha256, name: &str, value: &[u8]) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}
