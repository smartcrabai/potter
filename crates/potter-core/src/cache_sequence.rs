use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
};

/// Refuses a USD cache file (binary `PXR-USDC`, ASCII `#usda`, or a `PK` zip package for `.usdz`). Mesh Cache
/// and Mesh Sequence Cache read only Alembic (Ogawa) and PC2/MDD; USD is a known gap, so it fails with an
/// explicit unsupported-feature error instead of an Ogawa or header error.
///
/// # Errors
///
/// Returns an unsupported-feature error naming the resource when the bytes are USD.
pub(crate) fn reject_usd_cache(bytes: &[u8], resource: &str) -> Result<()> {
    let is_usd = bytes.starts_with(b"PXR-USDC")
        || bytes.starts_with(b"#usda")
        || bytes.starts_with(b"PK\x03\x04");
    if is_usd {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!(
                "USD cache resource `{resource}` is not supported; export the cache as Alembic"
            ),
            json!({"feature_id":"cache.usd","resource":resource}),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reject_usd_cache;
    use crate::error::ErrorCode;

    #[test]
    fn usd_cache_bytes_are_refused_with_an_unsupported_feature() {
        for bytes in [&b"PXR-USDC\x00\x00"[..], b"#usda 1.0\n", b"PK\x03\x04zip"] {
            let error = reject_usd_cache(bytes, "cache").expect_err("USD must be refused");
            assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        }
        assert!(reject_usd_cache(b"Ogawa\xff\x00\x01\x00", "cache").is_ok());
        assert!(reject_usd_cache(b"POINTCACHE2\0", "cache").is_ok());
    }
}

pub(crate) fn read_cache_resource_record(
    resource: &Value,
    resource_id: &str,
    project_root: Option<&Path>,
) -> Result<(Vec<u8>, String)> {
    let uri = resource
        .get("uri")
        .and_then(Value::as_str)
        .filter(|uri| !uri.trim().is_empty())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "Alembic cache resource has no file URI",
                json!({"resource_id":resource_id}),
            )
        })?;
    if resource
        .get("kind")
        .and_then(Value::as_str)
        .is_none_or(|kind| kind.trim().is_empty())
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "Alembic cache resource has no kind",
            json!({"resource_id":resource_id}),
        ));
    }
    let packed = resource
        .get("packed")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "Alembic cache resource has no packed state",
                json!({"resource_id":resource_id}),
            )
        })?;
    let expected_hash = resource
        .get("hash")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "Alembic cache resource has no expected content hash",
                json!({"resource_id":resource_id}),
            )
        })?;
    let bytes = if packed {
        let values = resource
            .get("bytes")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "packed Alembic cache resource has no byte payload",
                    json!({"resource_id":resource_id}),
                )
            })?;
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_u64()
                    .and_then(|byte| u8::try_from(byte).ok())
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "packed Alembic cache byte is outside 0..255",
                            json!({"resource_id":resource_id,"index":index}),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        if resource.get("bytes").is_some() {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "linked Alembic cache resource contains packed bytes",
                json!({"resource_id":resource_id}),
            ));
        }
        let path_text = uri.strip_prefix("file://").unwrap_or(uri);
        let path = Path::new(path_text);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(root) = project_root {
            root.join(path)
        } else {
            return Err(PotError::with_details(
                ErrorCode::DependencyMissing,
                "relative Alembic cache resource cannot be resolved without a project root",
                json!({"resource_id":resource_id,"uri":uri}),
            ));
        };
        fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    format!("Alembic cache file is missing: {}", path.display()),
                    json!({"resource_id":resource_id,"uri":uri}),
                )
            } else {
                PotError::io(&error)
            }
        })?
    };
    let actual_hash = hash::sha256(&bytes);
    if actual_hash != expected_hash {
        return Err(PotError::with_details(
            ErrorCode::AssetChanged,
            format!("Alembic cache resource `{resource_id}` changed after registration"),
            json!({
                "resource_id":resource_id,
                "uri":uri,
                "expected_hash":expected_hash,
                "actual_hash":actual_hash,
            }),
        ));
    }
    Ok((bytes, actual_hash))
}

/// Create the ordinary binary-resource record used for one cache sequence frame.
pub(crate) fn cache_sequence_resource_record(
    uri: &str,
    kind: &str,
    hash: &str,
    packed_bytes: Option<&[u8]>,
) -> Value {
    let mut record = json!({
        "uri": uri,
        "kind": kind,
        "hash": hash,
        "packed": packed_bytes.is_some(),
    });
    if let Some(bytes) = packed_bytes {
        record["bytes"] = json!(bytes);
    }
    record
}

pub(crate) const MAX_CACHE_SEQUENCE_FRAMES: i64 = 1_000_000;

struct CacheSequencePattern<'a> {
    path: &'a Path,
    prefix: &'a str,
    suffix: &'a str,
    width: usize,
}

impl CacheSequencePattern<'_> {
    fn filename(&self, frame: i32) -> String {
        let number = if frame < 0 {
            format!(
                "-{:0width$}",
                frame.unsigned_abs(),
                width = self.width.saturating_sub(1)
            )
        } else {
            format!("{frame:0width$}", width = self.width)
        };
        format!("{}{number}{}", self.prefix, self.suffix)
    }

    fn path_for_frame(&self, frame: i32) -> Option<PathBuf> {
        self.path
            .parent()
            .map(|parent| parent.join(self.filename(frame)))
    }

    fn frame_for_filename(&self, filename: &str) -> Option<i32> {
        let digits_end = filename.len().checked_sub(self.suffix.len())?;
        if !filename.starts_with(self.prefix)
            || !filename.ends_with(self.suffix)
            || digits_end < self.prefix.len()
        {
            return None;
        }
        let digits = &filename[self.prefix.len()..digits_end];
        if digits.len() != self.width {
            return None;
        }
        digits.parse().ok()
    }
}

fn cache_sequence_pattern(path: &Path) -> Option<CacheSequencePattern<'_>> {
    let filename = path.file_name()?.to_str()?;
    let stem_end = filename.rfind('.').unwrap_or(filename.len());
    let stem = &filename[..stem_end];
    let hash_end = stem.rfind('#');
    let (start, end) = if let Some(hash_end) = hash_end {
        let start = stem[..=hash_end]
            .rfind(|character| character != '#')
            .map_or(0, |index| index + 1);
        (start, hash_end + 1)
    } else {
        let end = stem.len();
        let start = stem
            .as_bytes()
            .iter()
            .rposition(|character| !character.is_ascii_digit())
            .map_or(0, |index| index + 1);
        (start, end)
    };
    if start == end {
        return None;
    }
    Some(CacheSequencePattern {
        path,
        prefix: &filename[..start],
        suffix: &filename[end..],
        width: end - start,
    })
}

#[derive(Default)]
pub(crate) struct CacheSequenceManifest {
    pub hashes: BTreeMap<String, String>,
    pub files: BTreeMap<i32, PathBuf>,
}

/// Resolve one frame from a `CacheFile` sequence, rejecting a symlink outside its pattern directory.
pub(crate) fn cache_sequence_frame_path(path: &Path, frame: i32) -> Result<Option<PathBuf>> {
    let Some(pattern) = cache_sequence_pattern(path) else {
        return Ok(None);
    };
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = match fs::canonicalize(parent) {
        Ok(directory) => directory,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    let frame_path = pattern.path_for_frame(frame).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "Alembic sequence path has no parent directory",
            json!({"path":path,"frame":frame}),
        )
    })?;
    let canonical = match fs::canonicalize(&frame_path) {
        Ok(canonical) => canonical,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    if canonical.parent() != Some(directory.as_path()) || !canonical.is_file() {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "Alembic sequence path escapes its canonical source directory",
            json!({"path":frame_path,"canonical_path":canonical,"frame":frame}),
        ));
    }
    Ok(Some(canonical))
}

/// Resolve an output filename from a numbered `CacheFile` path, rejecting symlink destinations.
pub(crate) fn cache_sequence_output_path(path: &Path, frame: i32) -> Result<Option<PathBuf>> {
    let Some(pattern) = cache_sequence_pattern(path) else {
        return Ok(None);
    };
    let frame_path = pattern.path_for_frame(frame).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "Alembic sequence path has no parent directory",
            json!({"path":path,"frame":frame}),
        )
    })?;
    match fs::symlink_metadata(&frame_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "Alembic sequence output path is a symlink",
            json!({"path":frame_path,"frame":frame}),
        )),
        Ok(_) => Ok(Some(frame_path)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Some(frame_path)),
        Err(error) => Err(PotError::io(&error)),
    }
}

/// Build the shared frame-to-content-hash manifest for linked Alembic sequences.
///
/// `None` scans the explicit file pattern for all frames present; `Some` requires every frame in
/// the inclusive range to exist. Files reached through a symlink must resolve within the sequence
/// directory.
pub(crate) fn cache_sequence_manifest(
    path: &Path,
    frame_range: Option<(i32, i32)>,
) -> Result<CacheSequenceManifest> {
    let Some(pattern) = cache_sequence_pattern(path) else {
        if frame_range.is_some() {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "Alembic sequence path must contain a frame-number pattern",
                json!({"path":path}),
            ));
        }
        return Ok(CacheSequenceManifest::default());
    };
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = match fs::canonicalize(parent) {
        Ok(directory) => directory,
        Err(error) if frame_range.is_none() && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CacheSequenceManifest::default());
        }
        Err(error) => return Err(PotError::io(&error)),
    };
    let mut frames = BTreeMap::new();
    if let Some((start, end)) = frame_range {
        let count = i64::from(end) - i64::from(start) + 1;
        if count <= 0 {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "Alembic sequence frame range must be ascending and inclusive",
                json!({"start":start,"end":end}),
            ));
        }
        if count > MAX_CACHE_SEQUENCE_FRAMES {
            return Err(PotError::with_details(
                ErrorCode::LimitExceeded,
                "Alembic sequence frame range exceeds the resource limit",
                json!({"start":start,"end":end,"limit":MAX_CACHE_SEQUENCE_FRAMES}),
            ));
        }
        for frame in start..=end {
            frames.insert(
                frame,
                pattern.path_for_frame(frame).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InvalidArgument,
                        "Alembic sequence path has no parent directory",
                        json!({"path":path,"frame":frame}),
                    )
                })?,
            );
        }
    } else {
        for entry in fs::read_dir(&directory).map_err(|error| PotError::io(&error))? {
            let entry = entry.map_err(|error| PotError::io(&error))?;
            let filename = entry.file_name();
            let Some(filename) = filename.to_str() else {
                continue;
            };
            if let Some(frame) = pattern.frame_for_filename(filename) {
                frames.insert(frame, entry.path());
            }
        }
    }
    let mut manifest = CacheSequenceManifest::default();
    for (frame, frame_path) in frames {
        let canonical = match fs::canonicalize(&frame_path) {
            Ok(canonical) => canonical,
            Err(error) if frame_range.is_none() && error.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(PotError::with_details(
                    ErrorCode::FileNotFound,
                    format!("Alembic sequence file is missing: {}", frame_path.display()),
                    json!({"path":frame_path,"frame":frame}),
                ));
            }
            Err(error) => return Err(PotError::io(&error)),
        };
        if canonical.parent() != Some(directory.as_path()) || !canonical.is_file() {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "Alembic sequence path escapes its canonical source directory",
                json!({"path":frame_path,"canonical_path":canonical,"frame":frame}),
            ));
        }
        let digest = hash::sha256(&fs::read(&canonical).map_err(|error| PotError::io(&error))?);
        manifest.hashes.insert(frame.to_string(), digest);
        manifest.files.insert(frame, canonical);
    }
    Ok(manifest)
}
