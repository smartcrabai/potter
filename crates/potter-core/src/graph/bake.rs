use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(any(unix, windows))]
use cap_fs_ext::OpenOptionsExt as CapOpenOptionsExt;
use cap_fs_ext::OpenOptionsFollowExt;
use cap_primitives::fs as cap_fs;
#[cfg(not(unix))]
use cap_std::{ambient_authority, fs::Dir as CapDir};
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    DELETE, FILE_DISPOSITION_INFO, FILE_READ_ATTRIBUTES, FILE_READ_DATA, FileDispositionInfo,
    SYNCHRONIZE, SetFileInformationByHandle,
};

#[cfg(unix)]
use rustix::fs::{RenameFlags, renameat_with};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::Id,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeBakeRecord {
    pub storage: String,
    pub kind: String,
    pub version: u32,
    pub frames: Vec<NodeBakeFrame>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeBakeFrame {
    pub frame: f64,
    pub values: BTreeMap<String, NodeBakeValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NodeBakeValue {
    Geometry {
        mesh: Mesh,
        #[serde(default)]
        instance_paths: Vec<Vec<Id>>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        material_refs: Vec<Id>,
    },
    Scalar {
        value: Value,
    },
    Field {
        attribute_name: String,
    },
}
// Matches the pinned Bake operator's 2 GiB packed-blob ceiling.
pub(crate) const MAX_NODE_BAKE_PACKED_BYTES: usize = 2 * 1024 * 1024 * 1024;
// Keep one serialized frame small; evaluation charges up to 32x JSON decode memory.
const MAX_NODE_BAKE_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_NODE_BAKE_FRAME_COUNT: usize = 100_001;
// Conservatively accounts for map/vector node overhead before decoding another frame.
pub(crate) const NODE_BAKE_JSON_MEMORY_MULTIPLIER: usize = 32;

fn bake_attribute_element_count(mesh: &Mesh, domain: &str) -> Option<usize> {
    match domain {
        "POINT" => Some(mesh.vertices.len()),
        "EDGE" => Some(mesh.edges.len()),
        "FACE" => Some(mesh.faces.len()),
        "CORNER" => mesh.faces.iter().try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        }),
        _ => None,
    }
}

pub(crate) fn validate_node_bake_resource(value: &Value) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "inline node-bake resource is invalid",
        )
    };
    if crate::model::migrate::inline_resource_kind(value)? != Some("node_bake") {
        return Err(invalid());
    }
    if value
        .get("frames")
        .and_then(Value::as_array)
        .is_none_or(|frames| frames.len() > MAX_NODE_BAKE_FRAME_COUNT)
        || estimate_json_value_heap(value) > MAX_NODE_BAKE_PACKED_BYTES
    {
        return Err(invalid());
    }
    let record: NodeBakeRecord = serde_json::from_value(value.clone()).map_err(|_| invalid())?;
    validate_node_bake_record(&record)
}

pub(crate) fn validate_node_bake_record(record: &NodeBakeRecord) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "inline node-bake resource is invalid",
        )
    };
    if record.storage != "inline"
        || record.kind != "node_bake"
        || record.version != 1
        || record.frames.is_empty()
        || record.frames.len() > MAX_NODE_BAKE_FRAME_COUNT
    {
        return Err(invalid());
    }
    let socket_ids = record.frames[0].values.keys().collect::<Vec<_>>();
    if socket_ids.is_empty() {
        return Err(invalid());
    }
    let mut previous = None;
    let mut retained_bytes = record
        .frames
        .capacity()
        .saturating_mul(std::mem::size_of::<NodeBakeFrame>());
    for frame in &record.frames {
        if previous.is_some_and(|previous| frame.frame <= previous)
            || frame.values.len() != socket_ids.len()
            || socket_ids
                .iter()
                .any(|socket| !frame.values.contains_key(*socket))
        {
            return Err(invalid());
        }
        validate_node_bake_frame(frame).map_err(|_| invalid())?;
        previous = Some(frame.frame);
        retained_bytes = retained_bytes.saturating_add(node_bake_frame_memory_bytes(frame));
        if retained_bytes > MAX_NODE_BAKE_PACKED_BYTES {
            return Err(invalid());
        }
    }
    Ok(())
}

pub(crate) fn validate_node_bake_frame(frame: &NodeBakeFrame) -> Result<()> {
    let invalid = || PotError::new(ErrorCode::SceneInvalid, "node-bake frame is invalid");
    if !frame.frame.is_finite() || frame.values.is_empty() {
        return Err(invalid());
    }
    for baked in frame.values.values() {
        match baked {
            NodeBakeValue::Geometry {
                mesh,
                instance_paths,
                material_refs,
            } => {
                if mesh.validate().is_err()
                    || instance_paths
                        .iter()
                        .flatten()
                        .chain(material_refs)
                        .any(|id| Id::new(id.to_string()).is_err())
                {
                    return Err(invalid());
                }
            }
            NodeBakeValue::Scalar { value } => {
                if !valid_value(value) {
                    return Err(invalid());
                }
            }
            NodeBakeValue::Field { attribute_name } => {
                let has_attribute = !attribute_name.is_empty()
                    && frame.values.values().any(|value| {
                        let NodeBakeValue::Geometry { mesh, .. } = value else {
                            return false;
                        };
                        let Some(attribute) = mesh.attributes.get(attribute_name) else {
                            return false;
                        };
                        let Some(domain) = attribute.get("domain").and_then(Value::as_str) else {
                            return false;
                        };
                        let Some(expected_count) = bake_attribute_element_count(mesh, domain)
                        else {
                            return false;
                        };
                        attribute
                            .get("data")
                            .and_then(Value::as_array)
                            .is_some_and(|values| {
                                values.len() == expected_count && values.iter().all(valid_value)
                            })
                    });
                if !has_attribute {
                    return Err(invalid());
                }
            }
        }
    }
    Ok(())
}

/// Keys of a settings-only bake entry: Blender bake settings with no baked data.
const NODE_BAKE_SETTINGS_KEYS: [&str; 7] = [
    "bake_mode",
    "bake_target",
    "use_custom_simulation_frame_range",
    "frame_start",
    "frame_end",
    "use_custom_path",
    "directory",
];

pub(crate) fn validate_node_bake_config(value: &Value) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake modifier settings are invalid",
        )
    };
    let config = value.as_object().ok_or_else(invalid)?;
    if !config.contains_key("storage") {
        // A settings-only entry carries no baked data, so it has no payload keys.
        if config
            .keys()
            .any(|key| !NODE_BAKE_SETTINGS_KEYS.contains(&key.as_str()))
        {
            return Err(invalid());
        }
        return validate_node_bake_settings(value);
    }
    let allowed = [
        "storage",
        "resource_id",
        "frames",
        "bake_mode",
        "bake_target",
        "use_custom_simulation_frame_range",
        "frame_start",
        "frame_end",
        "use_custom_path",
        "directory",
        "cache_directory",
    ];
    if config.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid());
    }
    validate_node_bake_settings(value)?;

    match config.get("storage").and_then(Value::as_str) {
        Some("packed") => {
            let resource_id = config
                .get("resource_id")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            Id::new(resource_id.to_owned()).map_err(|_| invalid())?;
            if config.contains_key("frames") {
                return Err(invalid());
            }
        }
        Some("disk") => {
            if config.contains_key("resource_id") {
                return Err(invalid());
            }
            let frames = config
                .get("frames")
                .and_then(Value::as_array)
                .filter(|frames| !frames.is_empty() && frames.len() <= MAX_NODE_BAKE_FRAME_COUNT)
                .ok_or_else(invalid)?;
            let mut previous = None;
            for frame in frames {
                let frame = frame.as_object().ok_or_else(invalid)?;
                if frame.len() != 2
                    || frame
                        .keys()
                        .any(|key| !matches!(key.as_str(), "frame" | "digest"))
                {
                    return Err(invalid());
                }
                let frame_value = frame
                    .get("frame")
                    .and_then(Value::as_f64)
                    .filter(|frame| frame.is_finite())
                    .ok_or_else(invalid)?;
                if !frame
                    .get("digest")
                    .and_then(Value::as_str)
                    .is_some_and(valid_sha256_digest)
                {
                    return Err(invalid());
                }
                if previous.is_some_and(|previous| frame_value <= previous) {
                    return Err(invalid());
                }
                previous = Some(frame_value);
            }
        }
        _ => return Err(invalid()),
    }
    Ok(())
}

pub(crate) fn validate_node_bake_settings(value: &Value) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake modifier settings are invalid",
        )
    };
    let config = value.as_object().ok_or_else(invalid)?;
    match config.get("bake_mode").and_then(Value::as_str) {
        None | Some("STILL") | Some("ANIMATION") => {}
        _ => return Err(invalid()),
    }
    match config.get("bake_target").and_then(Value::as_str) {
        None | Some("INHERIT") | Some("PACKED") | Some("DISK") => {}
        _ => return Err(invalid()),
    }
    let custom_range = match config.get("use_custom_simulation_frame_range") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(invalid()),
    };
    // Blender frame bounds are ints (rna_modifier.cc:8163-8170). An end before the start is not
    // clamped by Blender; `fix_frame_range` collapses it to the start frame.
    for key in ["frame_start", "frame_end"] {
        if let Some(frame) = config.get(key) {
            frame
                .as_i64()
                .and_then(|frame| i32::try_from(frame).ok())
                .ok_or_else(invalid)?;
        }
    }
    if custom_range && !(config.contains_key("frame_start") && config.contains_key("frame_end")) {
        return Err(invalid());
    }
    if config
        .get("use_custom_path")
        .is_some_and(|value| !value.is_boolean())
        || ["directory", "cache_directory"]
            .into_iter()
            .any(|key| config.get(key).is_some_and(|value| !value.is_string()))
    {
        return Err(invalid());
    }
    Ok(())
}

fn valid_sha256_digest(digest: &str) -> bool {
    digest.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn valid_value(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) | Value::Number(_) => true,
        Value::Array(values) => values.iter().all(valid_value),
        Value::Object(values) => values.values().all(valid_value),
    }
}

fn estimate_json_value_heap(value: &Value) -> usize {
    std::mem::size_of::<Value>().saturating_add(estimate_json_value_dynamic(value))
}

fn estimate_json_value_dynamic(value: &Value) -> usize {
    use std::mem::size_of;

    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => 0,
        Value::String(text) => text.len().saturating_add(16),
        Value::Array(values) => values.iter().fold(
            values.capacity().saturating_mul(size_of::<Value>()),
            |size, child| size.saturating_add(estimate_json_value_dynamic(child)),
        ),
        Value::Object(values) => values.iter().fold(0_usize, |size, (key, child)| {
            size.saturating_add(size_of::<(String, Value)>() + 64)
                .saturating_add(key.len().saturating_add(16))
                .saturating_add(estimate_json_value_dynamic(child))
        }),
    }
}

pub(crate) fn node_bake_frame_memory_bytes(frame: &NodeBakeFrame) -> usize {
    use std::mem::size_of;

    frame
        .values
        .iter()
        .fold(size_of::<NodeBakeFrame>(), |size, (socket, value)| {
            let value_bytes = match value {
                NodeBakeValue::Geometry {
                    mesh,
                    instance_paths,
                    material_refs,
                } => {
                    let mesh_bytes = mesh
                        .vertices
                        .capacity()
                        .saturating_mul(size_of::<crate::geom::Vertex>())
                        .saturating_add(
                            mesh.edges
                                .capacity()
                                .saturating_mul(size_of::<crate::geom::Edge>()),
                        )
                        .saturating_add(
                            mesh.faces
                                .capacity()
                                .saturating_mul(size_of::<crate::geom::Face>()),
                        )
                        .saturating_add(mesh.faces.iter().fold(0_usize, |total, face| {
                            total.saturating_add(
                                face.vertices.capacity().saturating_mul(size_of::<u32>()),
                            )
                        }))
                        .saturating_add(mesh.attributes.iter().fold(
                            0_usize,
                            |total, (key, value)| {
                                total
                                    .saturating_add(size_of::<(String, Value)>() + 64)
                                    .saturating_add(key.len().saturating_add(16))
                                    .saturating_add(estimate_json_value_dynamic(value))
                            },
                        ));
                    let paths_bytes = instance_paths.iter().fold(
                        instance_paths
                            .capacity()
                            .saturating_mul(size_of::<Vec<Id>>()),
                        |total, path| {
                            total.saturating_add(
                                path.capacity()
                                    .saturating_mul(size_of::<Id>())
                                    .saturating_add(path.iter().fold(0_usize, |bytes, id| {
                                        bytes.saturating_add(id.as_str().len().saturating_add(16))
                                    })),
                            )
                        },
                    );
                    mesh_bytes.saturating_add(paths_bytes).saturating_add(
                        material_refs
                            .capacity()
                            .saturating_mul(size_of::<Id>())
                            .saturating_add(material_refs.iter().fold(0_usize, |total, id| {
                                total.saturating_add(id.as_str().len().saturating_add(16))
                            })),
                    )
                }
                NodeBakeValue::Scalar { value } => estimate_json_value_dynamic(value),
                NodeBakeValue::Field { attribute_name } => attribute_name.len().saturating_add(16),
            };
            size.saturating_add(size_of::<(String, NodeBakeValue)>() + 64)
                .saturating_add(socket.len().saturating_add(16))
                .saturating_add(value_bytes)
        })
}
static BAKE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(any(unix, windows))]
struct StagedPayload {
    directory: PathBuf,
    filename: String,
    backup_name: String,
    temporary_name: Option<String>,
    kind: StagedPayloadKind,
    backup_created: bool,
    destination_published: bool,
}

#[cfg(any(unix, windows))]
#[derive(Clone)]
enum StagedPayloadKind {
    Write {
        digest: String,
        prior_digest: Option<String>,
    },
    Delete {
        digest: String,
        expected_frame: f64,
    },
}

#[cfg(any(unix, windows))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransactionState {
    Staging,
    Published,
    RolledBack,
    Committed,
    NeedsRecovery,
}

#[cfg(any(unix, windows))]
pub struct NodeBakePayloadTransaction {
    root: File,
    directories: BTreeMap<PathBuf, File>,
    targets: BTreeSet<(PathBuf, String)>,
    operations: Vec<StagedPayload>,
    state: TransactionState,
}

#[cfg(any(unix, windows))]
impl NodeBakePayloadTransaction {
    pub fn new(project_root: &Path) -> Result<Self> {
        let root = open_project_root(project_root)?;
        let mut directories = BTreeMap::new();
        directories.insert(
            PathBuf::new(),
            root.try_clone().map_err(|error| PotError::io(&error))?,
        );
        Ok(Self {
            root,
            directories,
            targets: BTreeSet::new(),
            operations: Vec::new(),
            state: TransactionState::Staging,
        })
    }

    pub fn stage_write(
        &mut self,
        directory: &str,
        object_id: &Id,
        modifier_id: &Id,
        node_id: &str,
        digest: &str,
        bytes: &[u8],
        expected_frame: f64,
        replace_existing: bool,
    ) -> Result<bool> {
        self.require_staging()?;
        if bytes.len() > MAX_NODE_BAKE_PAYLOAD_BYTES {
            return Err(payload_limit_error());
        }
        validate_payload_frame(bytes, digest, expected_frame, ErrorCode::InvalidArgument)?;
        let relative = node_bake_relative_path(directory, object_id, modifier_id, node_id, digest)?;
        let (directory_key, filename) = payload_parent_and_name(&relative)?;
        if self
            .targets
            .contains(&(directory_key.clone(), filename.clone()))
        {
            return Err(PotError::invalid_operation(
                "node-bake payload is already staged in this transaction",
            ));
        }
        self.ensure_directory(&directory_key)?;
        let directory_handle = self.directories.get(&directory_key).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "node-bake directory handle is missing",
            )
        })?;
        let prior_bytes =
            read_regular_file(directory_handle, &filename, MAX_NODE_BAKE_PAYLOAD_BYTES)?;
        let had_prior = prior_bytes.is_some();
        let prior_digest = if let Some(current) = prior_bytes {
            let current_digest = crate::hash::sha256(&current);
            if current_digest == digest {
                return Ok(false);
            }
            if !replace_existing {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "existing node-bake file has invalid content",
                ));
            }
            Some(current_digest)
        } else {
            None
        };
        let (temporary_name, mut temporary) = create_temporary(directory_handle)?;
        if let Err(error) = temporary
            .write_all(bytes)
            .and_then(|()| temporary.sync_all())
        {
            let _ = remove_file_at(directory_handle, &temporary_name);
            return Err(PotError::io(&error));
        }
        let backup_name = unique_backup_name();
        self.targets
            .insert((directory_key.clone(), filename.clone()));
        self.operations.push(StagedPayload {
            directory: directory_key,
            filename,
            backup_name,
            temporary_name: Some(temporary_name),
            kind: StagedPayloadKind::Write {
                digest: digest.to_owned(),
                prior_digest,
            },
            backup_created: false,
            destination_published: false,
        });
        Ok(!had_prior)
    }

    pub fn stage_delete(
        &mut self,
        directory: &str,
        object_id: &Id,
        modifier_id: &Id,
        node_id: &str,
        digest: &str,
        expected_frame: f64,
    ) -> Result<bool> {
        self.require_staging()?;
        let relative = node_bake_relative_path(directory, object_id, modifier_id, node_id, digest)?;
        let (directory_key, filename) = payload_parent_and_name(&relative)?;
        if self
            .targets
            .contains(&(directory_key.clone(), filename.clone()))
        {
            return Err(PotError::invalid_operation(
                "node-bake payload is already staged in this transaction",
            ));
        }
        if self.ensure_directory_if_present(&directory_key)?.is_none() {
            return Ok(false);
        }
        let directory_handle = self.directories.get(&directory_key).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "node-bake directory handle is missing",
            )
        })?;
        let Some(bytes) =
            read_regular_file(directory_handle, &filename, MAX_NODE_BAKE_PAYLOAD_BYTES)?
        else {
            return Ok(false);
        };
        validate_payload_frame(&bytes, digest, expected_frame, ErrorCode::SceneInvalid)?;
        self.targets
            .insert((directory_key.clone(), filename.clone()));
        self.operations.push(StagedPayload {
            directory: directory_key,
            filename,
            backup_name: unique_backup_name(),
            temporary_name: None,
            kind: StagedPayloadKind::Delete {
                digest: digest.to_owned(),
                expected_frame,
            },
            backup_created: false,
            destination_published: false,
        });
        Ok(true)
    }

    pub fn publish(&mut self) -> Result<()> {
        self.require_staging()?;
        for index in 0..self.operations.len() {
            if let Err(error) = self.publish_one(index) {
                let rollback = self.rollback_internal();
                self.state = if rollback.is_ok() {
                    TransactionState::RolledBack
                } else {
                    TransactionState::NeedsRecovery
                };
                return match rollback {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(PotError::with_details(
                        rollback_error.code,
                        "node-bake publication failed and rollback was incomplete",
                        serde_json::json!({
                            "cause":error.message,
                            "rollback":rollback_error.message,
                        }),
                    )),
                };
            }
        }
        self.state = TransactionState::Published;
        Ok(())
    }

    /// Finalizes after publication; failed backup cleanup leaves recoverable hidden copies.
    pub fn commit(mut self) -> Result<()> {
        if self.state != TransactionState::Published {
            if self.operations.is_empty() && self.state == TransactionState::Staging {
                self.state = TransactionState::Committed;
                return Ok(());
            }
            return Err(PotError::invalid_operation(
                "node-bake transaction must be published before commit",
            ));
        }
        self.state = TransactionState::Committed;
        for operation in &mut self.operations {
            if !operation.backup_created {
                continue;
            }
            let Some(directory) = self.directories.get(&operation.directory) else {
                continue;
            };
            if remove_file_at(directory, &operation.backup_name).is_ok() {
                operation.backup_created = false;
                let _ = sync_directory(directory);
            }
        }
        Ok(())
    }

    pub fn rollback(mut self) -> Result<()> {
        if self.state == TransactionState::Committed {
            return Err(PotError::invalid_operation(
                "a committed node-bake transaction cannot be rolled back",
            ));
        }
        let result = self.rollback_internal();
        self.state = if result.is_ok() {
            TransactionState::RolledBack
        } else {
            TransactionState::NeedsRecovery
        };
        result
    }

    fn require_staging(&self) -> Result<()> {
        if self.state == TransactionState::Staging {
            Ok(())
        } else {
            Err(PotError::invalid_operation(
                "node-bake transaction is no longer staging",
            ))
        }
    }

    fn ensure_directory(&mut self, relative: &Path) -> Result<()> {
        if self.directories.contains_key(relative) {
            return Ok(());
        }
        let directory = open_directory_chain(&self.root, relative, true)?.ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "node-bake directory creation failed",
            )
        })?;
        self.directories.insert(relative.to_path_buf(), directory);
        Ok(())
    }

    fn ensure_directory_if_present(&mut self, relative: &Path) -> Result<Option<()>> {
        if self.directories.contains_key(relative) {
            return Ok(Some(()));
        }
        let Some(directory) = open_directory_chain(&self.root, relative, false)? else {
            return Ok(None);
        };
        self.directories.insert(relative.to_path_buf(), directory);
        Ok(Some(()))
    }

    fn publish_one(&mut self, index: usize) -> Result<()> {
        let operation = &self.operations[index];
        let directory_key = operation.directory.clone();
        let filename = operation.filename.clone();
        let backup_name = operation.backup_name.clone();
        let temporary_name = operation.temporary_name.clone();
        let kind = operation.kind.clone();
        let directory = self.directories.get(&directory_key).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "node-bake directory handle is missing",
            )
        })?;
        match kind {
            StagedPayloadKind::Write {
                digest,
                prior_digest,
            } => {
                if let Some(prior_digest) = prior_digest {
                    move_to_backup(directory, &filename, &backup_name, &prior_digest)?;
                    self.operations[index].backup_created = true;
                    let backup =
                        read_regular_file(directory, &backup_name, MAX_NODE_BAKE_PAYLOAD_BYTES)?
                            .ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::SceneInvalid,
                                    "node-bake replacement backup disappeared",
                                )
                            })?;
                    if crate::hash::sha256(&backup) != prior_digest {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "node-bake replacement target changed before publication",
                        ));
                    }
                }
                let temporary_name = temporary_name.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "node-bake temporary file is missing",
                    )
                })?;
                let temp_remains = publish_temporary(directory, &temporary_name, &filename)?;
                self.operations[index].destination_published = true;
                if temp_remains {
                    remove_file_at(directory, &temporary_name)?;
                }
                self.operations[index].temporary_name = None;
                let published =
                    read_regular_file(directory, &filename, MAX_NODE_BAKE_PAYLOAD_BYTES)?
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "published node-bake payload disappeared",
                            )
                        })?;
                if crate::hash::sha256(&published) != digest {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "published node-bake payload digest does not match",
                    ));
                }
            }
            StagedPayloadKind::Delete {
                digest,
                expected_frame,
            } => {
                move_to_backup(directory, &filename, &backup_name, &digest)?;
                self.operations[index].backup_created = true;
                let backup =
                    read_regular_file(directory, &backup_name, MAX_NODE_BAKE_PAYLOAD_BYTES)?
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "node-bake deletion backup disappeared",
                            )
                        })?;
                validate_payload_frame(&backup, &digest, expected_frame, ErrorCode::SceneInvalid)?;
            }
        }
        sync_directory(directory)
    }

    fn rollback_internal(&mut self) -> Result<()> {
        let mut first_error = None;
        let directories = &self.directories;
        for operation in self.operations.iter_mut().rev() {
            let Some(directory) = directories.get(&operation.directory) else {
                first_error.get_or_insert_with(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "node-bake directory handle is missing",
                    )
                });
                continue;
            };
            if operation.destination_published {
                let expected_digest = match &operation.kind {
                    StagedPayloadKind::Write { digest, .. } => digest,
                    StagedPayloadKind::Delete { .. } => {
                        first_error.get_or_insert_with(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "delete operation unexpectedly published a destination",
                            )
                        });
                        continue;
                    }
                };
                match read_regular_file(directory, &operation.filename, MAX_NODE_BAKE_PAYLOAD_BYTES)
                {
                    Ok(Some(bytes)) if crate::hash::sha256(&bytes) == *expected_digest => {
                        if let Err(error) = remove_file_at(directory, &operation.filename) {
                            first_error.get_or_insert(error);
                            continue;
                        }
                        operation.destination_published = false;
                    }
                    Ok(None) => operation.destination_published = false,
                    Ok(Some(_)) => {
                        first_error.get_or_insert_with(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "published node-bake payload changed before rollback",
                            )
                        });
                        continue;
                    }
                    Err(error) => {
                        first_error.get_or_insert(error);
                        continue;
                    }
                }
            }
            if operation.backup_created {
                if let Err(error) =
                    restore_backup(directory, &operation.backup_name, &operation.filename)
                {
                    first_error.get_or_insert(error);
                    continue;
                }
                operation.backup_created = false;
            }
            if let Some(temporary) = operation.temporary_name.take()
                && let Err(error) = remove_file_at(directory, &temporary)
            {
                first_error.get_or_insert(error);
            }
            if let Err(error) = sync_directory(directory) {
                first_error.get_or_insert(error);
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

#[cfg(any(unix, windows))]
impl Drop for NodeBakePayloadTransaction {
    fn drop(&mut self) {
        if !matches!(
            self.state,
            TransactionState::Committed | TransactionState::RolledBack
        ) {
            if self.rollback_internal().is_ok() {
                self.state = TransactionState::RolledBack;
            } else {
                self.state = TransactionState::NeedsRecovery;
            }
        }
    }
}

#[cfg(any(unix, windows))]
#[cfg(test)]
pub(crate) fn read_node_bake_payload(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
) -> Result<Option<Vec<u8>>> {
    read_node_bake_payload_limited(
        project_root,
        bake_directory,
        object_id,
        modifier_id,
        node_id,
        digest,
        MAX_NODE_BAKE_PAYLOAD_BYTES,
    )
}

#[cfg(any(unix, windows))]
pub(crate) fn read_node_bake_payload_limited(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    let relative =
        node_bake_relative_path(bake_directory, object_id, modifier_id, node_id, digest)?;
    let (directory_key, filename) = payload_parent_and_name(&relative)?;
    let root = open_project_root(project_root)?;
    let Some(directory) = open_directory_chain(&root, &directory_key, false)? else {
        return Ok(None);
    };
    let Some(bytes) = read_regular_file(
        &directory,
        &filename,
        max_bytes.min(MAX_NODE_BAKE_PAYLOAD_BYTES),
    )?
    else {
        return Ok(None);
    };
    if crate::hash::sha256(&bytes) != digest {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "node-bake payload digest does not match",
            serde_json::json!({"digest":digest}),
        ));
    }
    Ok(Some(bytes))
}

fn open_project_root(project_root: &Path) -> Result<File> {
    // cap-std's Linux ambient open uses O_PATH, which rejects fsync with EBADF; a plain read-only
    // directory descriptor is needed for the directory sync after each rename.
    #[cfg(unix)]
    {
        let root = File::open(project_root).map_err(|error| PotError::io(&error))?;
        if !root
            .metadata()
            .map_err(|error| PotError::io(&error))?
            .is_dir()
        {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "node-bake project root is not a directory",
            ));
        }
        Ok(root)
    }
    #[cfg(not(unix))]
    {
        CapDir::open_ambient_dir(project_root, ambient_authority())
            .map(CapDir::into_std_file)
            .map_err(|error| PotError::io(&error))
    }
}

fn open_directory_chain(root: &File, relative: &Path, create: bool) -> Result<Option<File>> {
    let mut current = root.try_clone().map_err(|error| PotError::io(&error))?;
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "node-bake directory is not project-relative",
            ));
        };
        match open_directory_at(&current, name) {
            Ok(directory) => current = directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => {
                return Ok(None);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match cap_fs::create_dir(&current, Path::new(name), &cap_fs::DirOptions::new()) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(PotError::io(&error)),
                }
                current = open_directory_at(&current, name).map_err(directory_open_error)?;
            }
            Err(error) => return Err(directory_open_error(error)),
        }
    }
    Ok(Some(current))
}

fn open_directory_at(parent: &File, name: &std::ffi::OsStr) -> std::io::Result<File> {
    let relative = Path::new(name);
    let metadata = cap_fs::stat(parent, relative, cap_fs::FollowSymlinks::No)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotADirectory,
            "node-bake path component is a symlink or non-directory",
        ));
    }
    // cap-primitives opens directories with O_PATH on Linux when no read or write is requested,
    // and fsync on an O_PATH descriptor fails with EBADF. Open a real O_RDONLY|O_DIRECTORY
    // descriptor so the directory sync after each publish works; O_NOFOLLOW keeps the open
    // symlink-safe, as cap-std's nofollow open did.
    #[cfg(unix)]
    {
        let descriptor = rustix::fs::openat(
            parent,
            name,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )?;
        Ok(File::from(descriptor))
    }
    #[cfg(not(unix))]
    {
        cap_fs::open_dir_nofollow(&parent.try_clone()?, relative)
    }
}

fn directory_open_error(error: std::io::Error) -> PotError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::NotADirectory
            | std::io::ErrorKind::InvalidInput
            | std::io::ErrorKind::PermissionDenied
    ) {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake path contains a symlink or non-directory",
        )
    } else {
        storage_error(error)
    }
}

fn storage_error(error: std::io::Error) -> PotError {
    PotError::io(&error)
}

fn open_regular_file(directory: &File, filename: &str) -> Result<Option<File>> {
    let relative = Path::new(filename);
    let metadata = match cap_fs::stat(directory, relative, cap_fs::FollowSymlinks::No) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake payload is not a regular file",
        ));
    }
    let mut options = cap_fs::OpenOptions::new();
    options.read(true).follow(cap_fs::FollowSymlinks::No);
    #[cfg(unix)]
    options.custom_flags(rustix::fs::OFlags::NONBLOCK.bits() as i32);
    let file = cap_fs::open(directory, relative, &options).map_err(|error| {
        if matches!(
            error.kind(),
            std::io::ErrorKind::NotADirectory | std::io::ErrorKind::InvalidInput
        ) {
            PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake payload cannot be a symlink",
            )
        } else {
            PotError::io(&error)
        }
    })?;
    if !file
        .metadata()
        .map_err(|error| PotError::io(&error))?
        .is_file()
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake payload is not a regular file",
        ));
    }
    Ok(Some(file))
}

fn read_regular_file(
    directory: &File,
    filename: &str,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    let Some(mut file) = open_regular_file(directory, filename)? else {
        return Ok(None);
    };
    read_file_handle(&mut file, max_bytes).map(Some)
}

fn read_file_handle(file: &mut File, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = file.metadata().map_err(|error| PotError::io(&error))?;
    let file_size = usize::try_from(metadata.len()).map_err(|_| payload_limit_error())?;
    if file_size > max_bytes {
        return Err(payload_limit_error());
    }
    let mut bytes = Vec::with_capacity(file_size);
    let read_limit = u64::try_from(max_bytes.saturating_add(1)).unwrap_or(u64::MAX);
    (&mut *file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| PotError::io(&error))?;
    if bytes.len() > max_bytes {
        return Err(payload_limit_error());
    }
    Ok(bytes)
}

fn create_temporary(directory: &File) -> Result<(String, File)> {
    loop {
        let sequence = BAKE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!(".node-bake-{}-{sequence}.tmp", std::process::id());
        let mut options = cap_fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(cap_fs::FollowSymlinks::No);
        #[cfg(unix)]
        options.mode(0o600);
        match cap_fs::open(directory, Path::new(&name), &options) {
            Ok(file) => return Ok((name, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PotError::io(&error)),
        }
    }
}

fn unique_backup_name() -> String {
    format!(".node-bake-{}.bak", uuid::Uuid::new_v4())
}

#[cfg(unix)]
fn rename_noreplace(directory: &File, from: &str, to: &str) -> Result<()> {
    renameat_with(directory, from, directory, to, RenameFlags::NOREPLACE)
        .map_err(|error| storage_error(std::io::Error::from(error)))
}

#[cfg(unix)]
fn move_to_backup(directory: &File, from: &str, to: &str, _expected_digest: &str) -> Result<()> {
    rename_noreplace(directory, from, to)
}

#[cfg(windows)]
fn move_to_backup(directory: &File, from: &str, to: &str, expected_digest: &str) -> Result<()> {
    let mut source = open_deletable_file(directory, from)?.ok_or_else(|| {
        PotError::io(&std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "node-bake payload disappeared before backup",
        ))
    })?;
    let source_bytes = read_file_handle(&mut source, MAX_NODE_BAKE_PAYLOAD_BYTES)?;
    if crate::hash::sha256(&source_bytes) != expected_digest {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake payload changed before backup",
        ));
    }
    cap_fs::hard_link(directory, Path::new(from), directory, Path::new(to))
        .map_err(|error| PotError::io(&error))?;
    let backup = read_regular_file(directory, to, MAX_NODE_BAKE_PAYLOAD_BYTES)?
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "node-bake backup disappeared"))?;
    if crate::hash::sha256(&backup) != expected_digest {
        let _ = remove_file_at(directory, to);
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake payload changed while creating its backup",
        ));
    }
    if let Err(error) = delete_open_file(&source) {
        let _ = remove_file_at(directory, to);
        return Err(error);
    }
    Ok(())
}

#[cfg(windows)]
fn open_deletable_file(directory: &File, filename: &str) -> Result<Option<File>> {
    let relative = Path::new(filename);
    let metadata = match cap_fs::stat(directory, relative, cap_fs::FollowSymlinks::No) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake payload is not a regular file",
        ));
    }
    let mut options = cap_fs::OpenOptions::new();
    options
        .access_mode(DELETE | FILE_READ_ATTRIBUTES | FILE_READ_DATA | SYNCHRONIZE)
        .follow(cap_fs::FollowSymlinks::No);
    cap_fs::open(directory, relative, &options)
        .map(Some)
        .map_err(|error| PotError::io(&error))
}

#[cfg(windows)]
fn delete_open_file(file: &File) -> Result<()> {
    let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
    // SAFETY: `file` owns a live handle, and `disposition` is a correctly sized input record.
    let succeeded = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfo,
            (&disposition as *const FILE_DISPOSITION_INFO).cast(),
            u32::try_from(std::mem::size_of::<FILE_DISPOSITION_INFO>()).unwrap_or(u32::MAX),
        )
    };
    if succeeded == 0 {
        Err(PotError::io(&std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn restore_backup(directory: &File, backup: &str, filename: &str) -> Result<()> {
    rename_noreplace(directory, backup, filename)
}

#[cfg(windows)]
fn restore_backup(directory: &File, backup: &str, filename: &str) -> Result<()> {
    cap_fs::hard_link(directory, Path::new(backup), directory, Path::new(filename))
        .map_err(|error| PotError::io(&error))?;
    let _ = remove_file_at(directory, backup);
    Ok(())
}

#[cfg(unix)]
fn publish_temporary(directory: &File, temporary: &str, filename: &str) -> Result<bool> {
    rename_noreplace(directory, temporary, filename)?;
    Ok(false)
}

#[cfg(windows)]
fn publish_temporary(directory: &File, temporary: &str, filename: &str) -> Result<bool> {
    cap_fs::hard_link(
        directory,
        Path::new(temporary),
        directory,
        Path::new(filename),
    )
    .map_err(|error| PotError::io(&error))?;
    Ok(true)
}

#[cfg(unix)]
fn remove_file_at(directory: &File, filename: &str) -> Result<()> {
    match cap_fs::remove_file(directory, Path::new(filename)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PotError::io(&error)),
    }
}

#[cfg(windows)]
fn remove_file_at(directory: &File, filename: &str) -> Result<()> {
    if let Some(file) = open_deletable_file(directory, filename)? {
        delete_open_file(&file)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(directory: &File) -> Result<()> {
    directory.sync_all().map_err(|error| PotError::io(&error))
}

#[cfg(windows)]
fn sync_directory(_directory: &File) -> Result<()> {
    Ok(())
}

fn validate_payload_frame(
    bytes: &[u8],
    digest: &str,
    expected_frame: f64,
    digest_error: ErrorCode,
) -> Result<()> {
    if bytes.len() > MAX_NODE_BAKE_PAYLOAD_BYTES {
        return Err(payload_limit_error());
    }
    if crate::hash::sha256(bytes) != digest {
        return Err(PotError::with_details(
            digest_error,
            "node-bake payload digest does not match",
            serde_json::json!({"digest":digest}),
        ));
    }
    if !expected_frame.is_finite() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "expected bake frame is invalid",
        ));
    }
    let decode_budget = bytes
        .len()
        .saturating_mul(NODE_BAKE_JSON_MEMORY_MULTIPLIER + 1);
    if decode_budget > MAX_NODE_BAKE_PACKED_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "node-bake frame exceeds the decoded memory limit",
        ));
    }
    let frame = serde_json::from_slice::<NodeBakeFrame>(bytes).map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "node-bake payload is not a valid frame",
            serde_json::json!({"cause":error.to_string(),"digest":digest}),
        )
    })?;
    if node_bake_frame_memory_bytes(&frame) > MAX_NODE_BAKE_PACKED_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "node-bake frame exceeds the decoded memory limit",
        ));
    }
    if frame.frame != expected_frame {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake frame does not match its manifest",
        ));
    }
    validate_node_bake_frame(&frame)
}

fn payload_limit_error() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "node-bake payload exceeds the configured size limit",
    )
}

fn node_bake_relative_path(
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
) -> Result<PathBuf> {
    if !valid_sha256_digest(digest) {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake digest must be a 64-digit sha256 value",
        ));
    }
    let digest = digest.strip_prefix("sha256:").ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake digest must be sha256",
        )
    })?;
    let configured_directory = if bake_directory.is_empty() {
        ".potter/bakes"
    } else {
        bake_directory.strip_prefix("//").unwrap_or(bake_directory)
    };
    if configured_directory.contains(char::from(92)) || configured_directory.contains(':') {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake directory contains an invalid path separator",
        ));
    }
    let relative_directory = Path::new(configured_directory);
    if relative_directory.is_absolute()
        || relative_directory
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake directory must be project-relative without dot or parent segments",
        ));
    }
    let mut relative = relative_directory.to_path_buf();
    relative.push(safe_path_segment(object_id.as_str())?);
    relative.push(safe_path_segment(modifier_id.as_str())?);
    relative.push(safe_path_segment(node_id)?);
    relative.push(format!("sha256-{digest}.json"));
    Ok(relative)
}

fn payload_parent_and_name(relative: &Path) -> Result<(PathBuf, String)> {
    let parent = relative
        .parent()
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "invalid bake file path"))?;
    let filename = relative
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "invalid bake file name"))?;
    Ok((parent.to_path_buf(), filename.to_owned()))
}

fn safe_path_segment(value: &str) -> Result<&str> {
    if value.is_empty()
        || value.contains('/')
        || value.contains(char::from(92))
        || value.contains(':')
        || value == "."
        || value == ".."
    {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake path identifier is invalid",
        ));
    }
    Ok(value)
}

#[cfg(not(any(unix, windows)))]
pub struct NodeBakePayloadTransaction;

#[cfg(not(any(unix, windows)))]
impl NodeBakePayloadTransaction {
    pub fn new(_project_root: &Path) -> Result<Self> {
        Err(storage_platform_error())
    }

    pub fn stage_write(
        &mut self,
        _directory: &str,
        _object_id: &Id,
        _modifier_id: &Id,
        _node_id: &str,
        _digest: &str,
        _bytes: &[u8],
        _expected_frame: f64,
        _replace_existing: bool,
    ) -> Result<bool> {
        Err(storage_platform_error())
    }

    pub fn stage_delete(
        &mut self,
        _directory: &str,
        _object_id: &Id,
        _modifier_id: &Id,
        _node_id: &str,
        _digest: &str,
        _expected_frame: f64,
    ) -> Result<bool> {
        Err(storage_platform_error())
    }

    pub fn publish(&mut self) -> Result<()> {
        Err(storage_platform_error())
    }

    pub fn commit(self) -> Result<()> {
        Err(storage_platform_error())
    }

    pub fn rollback(self) -> Result<()> {
        Err(storage_platform_error())
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn read_node_bake_payload(
    _project_root: &Path,
    _bake_directory: &str,
    _object_id: &Id,
    _modifier_id: &Id,
    _node_id: &str,
    _digest: &str,
) -> Result<Option<Vec<u8>>> {
    Err(storage_platform_error())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn read_node_bake_payload_limited(
    _project_root: &Path,
    _bake_directory: &str,
    _object_id: &Id,
    _modifier_id: &Id,
    _node_id: &str,
    _digest: &str,
    _max_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    Err(storage_platform_error())
}

#[cfg(not(any(unix, windows)))]
fn storage_platform_error() -> PotError {
    PotError::new(
        ErrorCode::UnsupportedFeature,
        "node-bake disk storage requires no-follow directory-handle operations",
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "node-bake resource and path tests")]

    use std::{fs, os::unix::fs::symlink};

    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        MAX_NODE_BAKE_PAYLOAD_BYTES, NodeBakeFrame, NodeBakePayloadTransaction, NodeBakeRecord,
        NodeBakeValue, read_node_bake_payload, validate_node_bake_config,
        validate_node_bake_resource,
    };
    use crate::{error::ErrorCode, model::Id};

    fn frame(frame: f64, value: serde_json::Value) -> NodeBakeFrame {
        NodeBakeFrame {
            frame,
            values: [("Item_0".to_owned(), NodeBakeValue::Scalar { value })].into(),
        }
    }

    #[test]
    fn inline_node_bake_resource_validation_preserves_values_and_rejects_other_kinds() {
        let record = NodeBakeRecord {
            storage: "inline".to_owned(),
            kind: "node_bake".to_owned(),
            version: 1,
            frames: vec![frame(1.0, json!("surface_material"))],
        };
        let resource = serde_json::to_value(&record).unwrap();
        validate_node_bake_resource(&resource).unwrap();
        assert_eq!(
            serde_json::from_value::<NodeBakeRecord>(resource).unwrap(),
            record
        );

        let legacy_texture = json!({
            "storage":"inline",
            "kind":"legacy_texture",
            "texture_type":"NATIVE_VALUE_NOISE",
            "settings":{}
        });
        assert_eq!(
            validate_node_bake_resource(&legacy_texture)
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
        let incomplete_frame = json!({
            "storage":"inline",
            "kind":"node_bake",
            "version":1,
            "frames":[
                {"frame":1,"values":{"Item_0":{"type":"scalar","value":0.5}}},
                {"frame":2,"values":{}}
            ]
        });
        assert_eq!(
            validate_node_bake_resource(&incomplete_frame)
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
    }

    #[test]
    fn node_bake_manifests_validate_storage_and_digest_shape() {
        validate_node_bake_config(&json!({
            "storage":"packed",
            "resource_id":"bake_payload",
            "bake_mode":"ANIMATION",
            "bake_target":"INHERIT",
            "use_custom_simulation_frame_range":true,
            "frame_start":2,
            "frame_end":4
        }))
        .unwrap();
        validate_node_bake_config(&json!({
            "storage":"disk",
            "frames":[{"frame":1,"digest":format!("sha256:{}", "0".repeat(64))}],
            "cache_directory":"//bakes"
        }))
        .unwrap();
        for invalid in [
            json!({"storage":"packed","resource_id":"bake_payload","frames":[]}),
            json!({"storage":"disk","frames":[{"frame":1,"digest":"not-a-digest"}]}),
            json!({"storage":"disk","frames":[{"frame":2,"digest":format!("sha256:{}", "0".repeat(64))},{"frame":1,"digest":format!("sha256:{}", "1".repeat(64))}]}),
            json!({"storage":"disk","frames":[{"frame":1,"digest":format!("sha256:{}", "0".repeat(64))}],"untrusted":true}),
        ] {
            assert_eq!(
                validate_node_bake_config(&invalid).unwrap_err().code,
                ErrorCode::SceneInvalid
            );
        }
    }

    #[test]
    fn disk_payloads_are_bounded_digest_checked_and_transactional() {
        let project = tempdir().unwrap();
        let object_id = Id::new("object").unwrap();
        let modifier_id = Id::new("modifier").unwrap();
        let payload = serde_json::to_vec(&frame(1.0, json!("surface_material"))).unwrap();
        let digest = crate::hash::sha256(&payload);

        let mut invalid = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            invalid
                .stage_write(
                    "",
                    &object_id,
                    &modifier_id,
                    "bake",
                    "sha256:invalid",
                    &payload,
                    1.0,
                    false,
                )
                .unwrap_err()
                .code,
            ErrorCode::InvalidArgument
        );
        drop(invalid);
        assert!(!project.path().join(".potter/bakes").exists());

        let mut transaction = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            transaction
                .stage_write(
                    "//cache-a",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &payload,
                    1.0,
                    false,
                )
                .unwrap()
        );
        transaction.publish().unwrap();
        transaction.commit().unwrap();
        let mut unchanged = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            !unchanged
                .stage_write(
                    "//cache-a",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &payload,
                    1.0,
                    false,
                )
                .unwrap()
        );
        unchanged.commit().unwrap();
        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache-a",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
            )
            .unwrap()
            .unwrap(),
            payload
        );

        let root = fs::canonicalize(project.path()).unwrap();
        let path = root.join(
            super::node_bake_relative_path("//cache-a", &object_id, &modifier_id, "bake", &digest)
                .unwrap(),
        );
        let files = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(files, vec![path.file_name().unwrap().to_os_string()]);

        fs::write(&path, b"corrupt").unwrap();
        let mut refused = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            refused
                .stage_write(
                    "//cache-a",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &payload,
                    1.0,
                    false,
                )
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
        drop(refused);
        assert_eq!(fs::read(&path).unwrap(), b"corrupt");
        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache-a",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
            )
            .unwrap_err()
            .code,
            ErrorCode::SceneInvalid
        );

        let mut rolled_back = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            !rolled_back
                .stage_write(
                    "//cache-a",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &payload,
                    1.0,
                    true,
                )
                .unwrap()
        );
        rolled_back.publish().unwrap();
        rolled_back.rollback().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"corrupt");

        let mut replacement = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            !replacement
                .stage_write(
                    "//cache-a",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &payload,
                    1.0,
                    true,
                )
                .unwrap()
        );
        replacement.publish().unwrap();
        replacement.commit().unwrap();
        assert_eq!(fs::read(&path).unwrap(), payload);

        let mut wrong_frame = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            wrong_frame
                .stage_delete("//cache-a", &object_id, &modifier_id, "bake", &digest, 2.0,)
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
        drop(wrong_frame);
        let mut restored_delete = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            restored_delete
                .stage_delete("//cache-a", &object_id, &modifier_id, "bake", &digest, 1.0,)
                .unwrap()
        );
        restored_delete.publish().unwrap();
        restored_delete.rollback().unwrap();
        assert_eq!(fs::read(&path).unwrap(), payload);
        let mut deletion = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert!(
            deletion
                .stage_delete("//cache-a", &object_id, &modifier_id, "bake", &digest, 1.0,)
                .unwrap()
        );
        deletion.publish().unwrap();
        deletion.commit().unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn sparse_oversized_payload_is_rejected_before_reading_contents() {
        let project = tempdir().unwrap();
        let object_id = Id::new("object").unwrap();
        let modifier_id = Id::new("modifier").unwrap();
        let payload = serde_json::to_vec(&frame(1.0, json!("oversized"))).unwrap();
        let digest = crate::hash::sha256(&payload);
        let root = fs::canonicalize(project.path()).unwrap();
        let path = root.join(
            super::node_bake_relative_path("//cache", &object_id, &modifier_id, "bake", &digest)
                .unwrap(),
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::File::create(&path)
            .unwrap()
            .set_len(u64::try_from(MAX_NODE_BAKE_PAYLOAD_BYTES).unwrap() + 1)
            .unwrap();

        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
            )
            .unwrap_err()
            .code,
            ErrorCode::LimitExceeded
        );
    }

    #[test]
    fn disk_payload_paths_reject_parent_traversal_and_symlink_escape() {
        let project = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let object_id = Id::new("object").unwrap();
        let modifier_id = Id::new("modifier").unwrap();
        let bytes = serde_json::to_vec(&frame(1.0, json!("outside must stay empty"))).unwrap();
        let digest = crate::hash::sha256(&bytes);

        let mut traversal = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            traversal
                .stage_write(
                    "../outside",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &bytes,
                    1.0,
                    false,
                )
                .unwrap_err()
                .code,
            ErrorCode::InvalidArgument
        );
        drop(traversal);
        symlink(outside.path(), project.path().join("cache")).unwrap();
        let mut symlink_write = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            symlink_write
                .stage_write(
                    "//cache",
                    &object_id,
                    &modifier_id,
                    "bake",
                    &digest,
                    &bytes,
                    1.0,
                    false,
                )
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
            )
            .unwrap_err()
            .code,
            ErrorCode::SceneInvalid
        );
        let mut symlink_delete = NodeBakePayloadTransaction::new(project.path()).unwrap();
        assert_eq!(
            symlink_delete
                .stage_delete("//cache", &object_id, &modifier_id, "bake", &digest, 1.0,)
                .unwrap_err()
                .code,
            ErrorCode::SceneInvalid
        );
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
    }

    #[test]
    fn settings_only_bake_entries_are_valid_without_payload() {
        let digest = format!("sha256:{}", "0".repeat(64));
        let custom_range = json!({
            "bake_mode": "ANIMATION",
            "bake_target": "DISK",
            "use_custom_simulation_frame_range": true,
            "frame_start": 4,
            "frame_end": 2,
            "use_custom_path": false,
            "directory": ""
        });
        assert!(validate_node_bake_config(&custom_range).is_ok());
        assert!(validate_node_bake_config(&json!({"bake_mode": "STILL"})).is_ok());
        let packed = json!({"storage": "packed", "resource_id": "bake_a"});
        assert!(validate_node_bake_config(&packed).is_ok());
        let with_resource = json!({"bake_mode": "STILL", "resource_id": "bake_a"});
        assert!(validate_node_bake_config(&with_resource).is_err());
        let with_frames = json!({"frames": [{"frame": 1.0, "digest": digest}]});
        assert!(validate_node_bake_config(&with_frames).is_err());
        let float_start = json!({
            "use_custom_simulation_frame_range": true,
            "frame_start": 1.5,
            "frame_end": 4
        });
        assert!(validate_node_bake_config(&float_start).is_err());
        let missing_end = json!({"use_custom_simulation_frame_range": true, "frame_start": 1});
        assert!(validate_node_bake_config(&missing_end).is_err());
    }
}
