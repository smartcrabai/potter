use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    model::{HistoryState, SceneDoc},
    response::SceneInfo,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRecord {
    pub record_version: u32,
    pub parent: Option<String>,
    pub kind: String,
    pub revision: u64,
    pub scene_hash: String,
    pub operations: Value,
    pub changes: Value,
    pub target: Option<String>,
    pub snapshot: SceneDoc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryDirection {
    Undo,
    Redo,
}

pub struct Project {
    path: PathBuf,
    lock: Option<File>,
    doc: SceneDoc,
}

impl Project {
    pub fn init(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        fs::create_dir_all(path).map_err(|error| PotError::io(&error))?;
        if fs::read_dir(path)
            .map_err(|error| PotError::io(&error))?
            .next()
            .is_some()
        {
            return Err(PotError::new(
                ErrorCode::OutputExists,
                "scene directory is not empty",
            ));
        }
        fs::create_dir_all(path.join(".potter/tmp")).map_err(|error| PotError::io(&error))?;
        for category in ["data", "assets", "compat", "history"] {
            fs::create_dir_all(path.join(category)).map_err(|error| PotError::io(&error))?;
        }
        let lock = Some(Self::lock_file(path)?);
        let doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mut project = Self {
            path: canonical_path(path)?,
            lock,
            doc,
        };
        project.save_initial()?;
        Ok(project)
    }

    /// Opens one committed scene snapshot without acquiring the writer lock.
    ///
    /// `scene.json` is atomically replaced and referenced blobs are immutable, so
    /// reading the document once fixes a consistent snapshot for the full operation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path.as_ref(), false)
    }

    /// Opens one scene snapshot while holding the non-blocking exclusive writer lock.
    pub fn open_exclusive(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path.as_ref(), true)
    }

    fn open_inner(path: &Path, exclusive: bool) -> Result<Self> {
        if !path.join("scene.json").is_file() {
            return Err(PotError::new(
                ErrorCode::SceneNotFound,
                format!("scene not found: {}", path.display()),
            ));
        }
        let canonical = canonical_path(path)?;
        let lock = if exclusive {
            Some(Self::lock_file(&canonical)?)
        } else {
            None
        };
        let doc = Self::read_snapshot(&canonical)?;
        Ok(Self {
            path: canonical,
            lock,
            doc,
        })
    }

    fn read_snapshot(path: &Path) -> Result<SceneDoc> {
        let bytes = fs::read(path.join("scene.json")).map_err(|error| PotError::io(&error))?;
        let doc: SceneDoc = serde_json::from_slice(&bytes).map_err(|error| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                error.to_string(),
                json!({ "line": error.line(), "column": error.column() }),
            )
        })?;
        doc.validate()?;
        Ok(doc)
    }

    fn lock_file(path: &Path) -> Result<File> {
        let lock_path = path.join(".potter/lock");
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(|error| PotError::io(&error))?;
        match file.try_lock_exclusive() {
            Ok(true) => Ok(file),
            Ok(false) => Err(PotError::new(ErrorCode::SceneBusy, "scene is busy")),
            Err(error) => Err(PotError::io(&error)),
        }
    }

    fn save_initial(&mut self) -> Result<()> {
        let canonical = canonical_scene(&self.doc)?;
        let scene_hash = hash::sha256(&canonical);
        let mut snapshot = self.doc.clone();
        snapshot.history = HistoryState::default();
        let record = HistoryRecord {
            record_version: 1,
            parent: None,
            kind: "init".to_owned(),
            revision: 0,
            scene_hash: scene_hash.clone(),
            operations: json!([]),
            changes: json!({}),
            target: None,
            snapshot,
        };
        let record_hash = write_history_blob(&self.path, &record)?;
        self.doc.history.head = Some(record_hash);
        write_scene_atomic(&self.path, &self.doc)?;
        Ok(())
    }

    #[must_use]
    pub fn doc(&self) -> &SceneDoc {
        &self.doc
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn info(&self) -> Result<SceneInfo> {
        let value = serde_json::to_value(&self.doc).map_err(|error| internal_json(&error))?;
        Ok(SceneInfo {
            id: self.doc.scene_id.clone(),
            path: self.path.to_string_lossy().into_owned(),
            revision: self.doc.revision,
            hash: hash::sha256(&hash::canonicalize(&value)?),
        })
    }

    /// Commits scene metadata and its newly referenced immutable image assets.
    ///
    /// Asset blobs are verified and atomically staged before the scene snapshot is
    /// replaced. A failed scene commit may leave an unreferenced immutable blob.
    pub fn commit_with_assets(
        &mut self,
        candidate: SceneDoc,
        operations: Value,
        changes: Value,
        asset_blobs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<SceneInfo> {
        let (candidate, record) =
            self.prepare_candidate_record(candidate, operations, changes, "apply")?;
        self.stage_asset_blobs(&candidate, asset_blobs)?;
        self.commit_prepared(candidate, &record)
    }

    pub fn commit_import(
        &mut self,
        candidate: SceneDoc,
        operations: Value,
        changes: Value,
    ) -> Result<SceneInfo> {
        self.commit_with_kind("import", candidate, operations, changes)
    }
    pub fn commit_import_with_assets(
        &mut self,
        candidate: SceneDoc,
        operations: Value,
        changes: Value,
        asset_blobs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<SceneInfo> {
        let (candidate, record) =
            self.prepare_candidate_record(candidate, operations, changes, "import")?;
        self.stage_asset_blobs(&candidate, asset_blobs)?;
        self.commit_prepared(candidate, &record)
    }

    fn stage_asset_blobs(
        &self,
        candidate: &SceneDoc,
        asset_blobs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<()> {
        for (digest, bytes) in asset_blobs {
            let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "staged asset key is not a sha256 digest",
                )
            })?;
            if hex.len() != 64
                || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
                || hash::sha256(bytes) != *digest
            {
                return Err(PotError::with_details(
                    ErrorCode::InternalError,
                    "staged asset bytes do not match their content hash",
                    json!({"hash":digest}),
                ));
            }
            let prefix = format!("assets/sha256/{hex}/");
            let has_resource_alias = candidate.resources.values().any(|resource| {
                resource.get("hash").and_then(Value::as_str) == Some(digest.as_str())
                    && resource
                        .get("uri")
                        .and_then(Value::as_str)
                        .is_some_and(|uri| uri.strip_prefix(&prefix).is_some())
            });
            let image_uses_blob = candidate.images.values().any(|image| {
                image.blob.as_deref() == Some(digest.as_str())
                    || image.tiles.iter().any(|tile| tile.blob == *digest)
            });
            if !has_resource_alias || image_uses_blob {
                let destination = self
                    .path
                    .join("assets")
                    .join("sha256")
                    .join(hex)
                    .join("blob");
                if destination.is_file() {
                    let existing = fs::read(&destination).map_err(|error| PotError::io(&error))?;
                    if hash::sha256(&existing) != *digest {
                        return Err(PotError::with_details(
                            ErrorCode::ValidationFailed,
                            "existing immutable image blob is corrupted",
                            json!({"hash":digest,"path":destination}),
                        ));
                    }
                } else {
                    write_blob_atomic(&self.path, &destination, bytes)?;
                }
            }
            for resource in candidate.resources.values() {
                if resource.get("hash").and_then(Value::as_str) != Some(digest.as_str()) {
                    continue;
                }
                let Some(uri) = resource.get("uri").and_then(Value::as_str) else {
                    continue;
                };
                let Some(filename) = uri.strip_prefix(&prefix) else {
                    continue;
                };
                let relative = Path::new(filename);
                if relative.components().count() != 1
                    || relative.file_name().and_then(|name| name.to_str()) != Some(filename)
                {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "resource asset URI must name one file inside its content-addressed directory",
                    ));
                }
                let destination = self.path.join(uri);
                if destination.is_file() {
                    let existing = fs::read(&destination).map_err(|error| PotError::io(&error))?;
                    if hash::sha256(&existing) != *digest {
                        return Err(PotError::with_details(
                            ErrorCode::ValidationFailed,
                            "existing resource asset file is corrupted",
                            json!({"hash":digest,"path":destination}),
                        ));
                    }
                } else {
                    write_blob_atomic(&self.path, &destination, bytes)?;
                }
            }
        }
        Ok(())
    }

    pub fn prepare_commit_candidate(
        &self,
        candidate: SceneDoc,
        operations: Value,
        changes: Value,
    ) -> Result<SceneDoc> {
        self.prepare_candidate_record(candidate, operations, changes, "apply")
            .map(|(candidate, _record)| candidate)
    }

    fn prepare_candidate_record(
        &self,
        mut candidate: SceneDoc,
        operations: Value,
        changes: Value,
        kind: &str,
    ) -> Result<(SceneDoc, HistoryRecord)> {
        self.check_next_revision()?;
        candidate.revision = self.doc.revision + 1;
        let old_head = self.doc.history.head.clone();
        let scene_hash = hash::sha256(&canonical_scene(&candidate)?);
        let mut snapshot = candidate.clone();
        snapshot.history = HistoryState::default();
        let record = HistoryRecord {
            record_version: 1,
            parent: old_head.clone(),
            kind: kind.to_owned(),
            revision: candidate.revision,
            scene_hash,
            operations,
            changes,
            target: None,
            snapshot,
        };
        let record_hash = history_record_hash(&record)?;
        candidate.history = self.doc.history.clone();
        if let Some(head) = old_head {
            candidate.history.undo.push(head);
        }
        candidate.history.redo.clear();
        candidate.history.head = Some(record_hash);
        candidate.validate()?;
        Ok((candidate, record))
    }

    fn commit_with_kind(
        &mut self,
        kind: &str,
        candidate: SceneDoc,
        operations: Value,
        changes: Value,
    ) -> Result<SceneInfo> {
        let (candidate, record) =
            self.prepare_candidate_record(candidate, operations, changes, kind)?;
        self.commit_prepared(candidate, &record)
    }

    fn commit_prepared(
        &mut self,
        candidate: SceneDoc,
        record: &HistoryRecord,
    ) -> Result<SceneInfo> {
        let expected_head = candidate.history.head.clone();
        let record_hash = write_history_blob(&self.path, record)?;
        if expected_head.as_deref() != Some(record_hash.as_str()) {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "prepared history hash changed before commit",
            ));
        }
        write_scene_atomic(&self.path, &candidate)?;
        self.doc = candidate;
        self.info()
    }

    pub fn write_compat_blob(&self, name: &str, bytes: &[u8]) -> Result<String> {
        let basename = Path::new(name)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty() && *value != "." && *value != "..")
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "compatibility blob name is invalid",
                )
            })?;
        let digest = hash::sha256(bytes);
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "invalid content hash"))?;
        let destination = self
            .path
            .join("compat")
            .join("sha256")
            .join(hex)
            .join(basename);
        write_blob_atomic(&self.path, &destination, bytes)?;
        Ok(digest)
    }

    pub fn restore(
        &mut self,
        mut candidate: SceneDoc,
        kind: &str,
        target: String,
        operations: Value,
        changes: Value,
    ) -> Result<SceneInfo> {
        self.check_next_revision()?;
        candidate.revision = self.doc.revision + 1;
        let old_head = self.doc.history.head.clone();
        let scene_hash = hash::sha256(&canonical_scene(&candidate)?);
        let mut snapshot = candidate.clone();
        snapshot.history = HistoryState::default();
        let record = HistoryRecord {
            record_version: 1,
            parent: old_head.clone(),
            kind: kind.to_owned(),
            revision: candidate.revision,
            scene_hash,
            operations,
            changes,
            target: Some(target),
            snapshot,
        };
        let record_hash = write_history_blob(&self.path, &record)?;
        candidate.history.head = Some(record_hash);
        candidate.validate()?;
        write_scene_atomic(&self.path, &candidate)?;
        self.doc = candidate;
        self.info()
    }

    pub fn history_records(&self) -> Result<Vec<(String, HistoryRecord)>> {
        let mut records = Vec::new();
        let mut next = self.doc.history.head.clone();
        while let Some(record_hash) = next.take() {
            let record = read_history_blob(&self.path, &record_hash)?;
            next.clone_from(&record.parent);
            records.push((record_hash, record));
        }
        Ok(records)
    }

    pub fn history_target(
        &self,
        direction: HistoryDirection,
        steps: usize,
    ) -> Result<(SceneDoc, String)> {
        let history = match direction {
            HistoryDirection::Undo => &self.doc.history.undo,
            HistoryDirection::Redo => &self.doc.history.redo,
        };
        if steps == 0 || steps > history.len() {
            let message = match direction {
                HistoryDirection::Undo => "requested undo history is unavailable",
                HistoryDirection::Redo => "requested redo history is unavailable",
            };
            return Err(PotError::new(ErrorCode::TargetNotFound, message));
        }
        let target = history[history.len() - steps].clone();
        let record = read_history_blob(&self.path, &target)?;
        Ok((record.snapshot, target))
    }

    pub fn commit_history(
        &mut self,
        direction: HistoryDirection,
        mut candidate: SceneDoc,
        target: String,
        steps: usize,
        changes: Value,
    ) -> Result<SceneInfo> {
        let current = self
            .doc
            .history
            .head
            .clone()
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "history head is missing"))?;
        candidate.history = self.doc.history.clone();
        let (source, destination, kind) = match direction {
            HistoryDirection::Undo => (
                &mut candidate.history.undo,
                &mut candidate.history.redo,
                "undo",
            ),
            HistoryDirection::Redo => (
                &mut candidate.history.redo,
                &mut candidate.history.undo,
                "redo",
            ),
        };
        let start = source.len() - steps;
        let removed = source.split_off(start);
        destination.push(current);
        destination.extend(removed.into_iter().skip(1).rev());
        self.restore(candidate, kind, target, json!([]), changes)
    }

    fn check_next_revision(&self) -> Result<()> {
        if self.doc.revision >= (1_u64 << 53) - 1 {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "revision limit reached",
            ));
        }
        Ok(())
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        if let Some(lock) = &self.lock {
            let _ = lock.unlock();
        }
    }
}

fn canonical_path(path: &Path) -> Result<PathBuf> {
    fs::canonicalize(path).map_err(|error| PotError::io(&error))
}

fn canonical_scene(doc: &SceneDoc) -> Result<Vec<u8>> {
    let value = serde_json::to_value(doc).map_err(|error| internal_json(&error))?;
    hash::canonicalize(&value)
}

fn internal_json(error: &serde_json::Error) -> PotError {
    PotError::with_details(
        ErrorCode::InternalError,
        PotError::internal_json(error).message,
        json!({ "line": error.line(), "column": error.column() }),
    )
}

fn blob_path(root: &Path, category: &str, digest: &str) -> Result<PathBuf> {
    let hex = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "invalid content hash"))?;
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "invalid content hash",
        ));
    }
    Ok(root
        .join(category)
        .join("sha256")
        .join(hex)
        .join("blob.json"))
}

fn history_record_hash(record: &HistoryRecord) -> Result<String> {
    let value = serde_json::to_value(record).map_err(|error| internal_json(&error))?;
    Ok(hash::sha256(&hash::canonicalize(&value)?))
}

fn write_history_blob(root: &Path, record: &HistoryRecord) -> Result<String> {
    let value = serde_json::to_value(record).map_err(|error| internal_json(&error))?;
    let bytes = hash::canonicalize(&value)?;
    let digest = hash::sha256(&bytes);
    let destination = blob_path(root, "history", &digest)?;
    write_blob_atomic(root, &destination, &bytes)?;
    Ok(digest)
}

fn read_history_blob(root: &Path, digest: &str) -> Result<HistoryRecord> {
    let path = blob_path(root, "history", digest)?;
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    serde_json::from_slice(&bytes).map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            error.to_string(),
            json!({ "line": error.line(), "column": error.column() }),
        )
    })
}

fn write_blob_atomic(root: &Path, destination: &Path, bytes: &[u8]) -> Result<()> {
    if destination.is_file() {
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "blob path has no parent"))?;
    fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
    write_atomic_bytes(root, destination, bytes, "")
}

fn write_scene_atomic(root: &Path, doc: &SceneDoc) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(doc).map_err(|error| internal_json(&error))?;
    write_atomic_bytes(root, &root.join("scene.json"), &bytes, "scene-")
}

fn write_atomic_bytes(
    root: &Path,
    destination: &Path,
    bytes: &[u8],
    temp_prefix: &str,
) -> Result<()> {
    let temp = root
        .join(".potter/tmp")
        .join(format!("{temp_prefix}{}.tmp", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| PotError::io(&error))?;
    file.write_all(bytes)
        .map_err(|error| PotError::io(&error))?;
    file.sync_all().map_err(|error| PotError::io(&error))?;
    drop(file);
    fs::rename(&temp, destination).map_err(|error| {
        let _ = fs::remove_file(&temp);
        PotError::io(&error)
    })
}
