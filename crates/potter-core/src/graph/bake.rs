use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

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
        domain: String,
        values: Vec<Value>,
    },
}

pub(crate) fn validate_node_bake_resource(value: &Value) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "inline node-bake resource is invalid",
        )
    };
    let record: NodeBakeRecord = serde_json::from_value(value.clone()).map_err(|_| invalid())?;
    if record.storage != "inline"
        || record.kind != "node_bake"
        || record.version != 1
        || record.frames.is_empty()
    {
        return Err(invalid());
    }
    let mut previous = None;
    for frame in &record.frames {
        if !frame.frame.is_finite() || previous.is_some_and(|previous| frame.frame <= previous) {
            return Err(invalid());
        }
        previous = Some(frame.frame);
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
                NodeBakeValue::Field { domain, values } => {
                    if !matches!(domain.as_str(), "POINT" | "EDGE" | "FACE" | "CORNER")
                        || values.iter().any(|value| !valid_value(value))
                    {
                        return Err(invalid());
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_node_bake_config(value: &Value) -> Result<()> {
    let invalid = || {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node-bake modifier settings are invalid",
        )
    };
    let config = value.as_object().ok_or_else(invalid)?;
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
    let frame_start = config
        .get("frame_start")
        .map(|value| {
            value
                .as_f64()
                .filter(|frame| frame.is_finite())
                .ok_or_else(invalid)
        })
        .transpose()?;
    let frame_end = config
        .get("frame_end")
        .map(|value| {
            value
                .as_f64()
                .filter(|frame| frame.is_finite())
                .ok_or_else(invalid)
        })
        .transpose()?;
    if custom_range && !matches!((frame_start, frame_end), (Some(start), Some(end)) if start <= end)
    {
        return Err(invalid());
    }
    for key in ["use_custom_path"] {
        if config.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(invalid());
        }
    }
    for key in ["directory", "cache_directory"] {
        if config.get(key).is_some_and(|value| !value.is_string()) {
            return Err(invalid());
        }
    }

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
                .filter(|frames| !frames.is_empty() && frames.len() <= 100_001)
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
static BAKE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn read_node_bake_payload(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
) -> Result<Option<Vec<u8>>> {
    let (root, relative) = node_bake_relative_path(
        project_root,
        bake_directory,
        object_id,
        modifier_id,
        node_id,
        digest,
    )?;
    let Some(path) = checked_existing_path(&root, &relative)? else {
        return Ok(None);
    };
    let mut file = File::open(path).map_err(|error| PotError::io(&error))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| PotError::io(&error))?;
    if crate::hash::sha256(&bytes) != digest {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "node-bake payload digest does not match",
            serde_json::json!({"digest":digest}),
        ));
    }
    Ok(Some(bytes))
}

pub(crate) fn delete_node_bake_payload(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
) -> Result<()> {
    let (root, relative) = node_bake_relative_path(
        project_root,
        bake_directory,
        object_id,
        modifier_id,
        node_id,
        digest,
    )?;
    if let Some(path) = checked_existing_path(&root, &relative)? {
        fs::remove_file(path).map_err(|error| PotError::io(&error))?;
    }
    Ok(())
}

pub(crate) fn write_node_bake_payload(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
    bytes: &[u8],
) -> Result<()> {
    if crate::hash::sha256(bytes) != digest {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "node-bake payload digest does not match its bytes",
        ));
    }
    let (root, relative) = node_bake_relative_path(
        project_root,
        bake_directory,
        object_id,
        modifier_id,
        node_id,
        digest,
    )?;
    let parent = relative
        .parent()
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "invalid bake file path"))?;
    let directory = ensure_directory_chain(&root, parent)?;
    let filename = relative
        .file_name()
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "invalid bake file name"))?;
    let destination = directory.join(filename);
    if let Some(existing) = checked_existing_path(&root, &relative)? {
        let current = fs::read(existing).map_err(|error| PotError::io(&error))?;
        if crate::hash::sha256(&current) != digest {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "existing node-bake file has invalid content",
            ));
        }
        return Ok(());
    }
    let (temporary, mut file) = create_temporary(&directory)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(PotError::io(&error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, &destination) {
        let _ = fs::remove_file(&temporary);
        return Err(PotError::io(&error));
    }
    if let Ok(directory_file) = File::open(&directory) {
        let _ = directory_file.sync_all();
    }
    Ok(())
}

fn node_bake_relative_path(
    project_root: &Path,
    bake_directory: &str,
    object_id: &Id,
    modifier_id: &Id,
    node_id: &str,
    digest: &str,
) -> Result<(PathBuf, PathBuf)> {
    let root = fs::canonicalize(project_root).map_err(|error| PotError::io(&error))?;
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
    let safe_node = safe_path_segment(node_id)?;
    let mut relative = relative_directory.to_path_buf();
    relative.push(safe_path_segment(object_id.as_str())?);
    relative.push(safe_path_segment(modifier_id.as_str())?);
    relative.push(safe_node);
    relative.push(format!("sha256-{digest}.json"));
    Ok((root, relative))
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

fn checked_existing_path(root: &Path, relative: &Path) -> Result<Option<PathBuf>> {
    let mut current = root.to_path_buf();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        let has_next = components.peek().is_some();
        let Component::Normal(name) = component else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "node-bake file path is not project-relative",
            ));
        };
        current.push(name);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(PotError::io(&error)),
        };
        if metadata.file_type().is_symlink() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake path cannot traverse a symlink",
            ));
        }
        let canonical = fs::canonicalize(&current).map_err(|error| PotError::io(&error))?;
        if !canonical.starts_with(root) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake path escapes the project directory",
            ));
        }
        if has_next && !metadata.is_dir() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake parent path is not a directory",
            ));
        }
        if !has_next && !metadata.is_file() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake payload is not a regular file",
            ));
        }
    }
    Ok(Some(current))
}

fn ensure_directory_chain(root: &Path, relative: &Path) -> Result<PathBuf> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "node-bake directory is not project-relative",
            ));
        };
        current.push(name);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "node-bake directory contains a symlink or non-directory",
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(PotError::io(&error)),
                }
            }
            Err(error) => return Err(PotError::io(&error)),
        }
        let canonical = fs::canonicalize(&current).map_err(|error| PotError::io(&error))?;
        if !canonical.starts_with(root) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "node-bake directory escapes the project directory",
            ));
        }
        current = canonical;
    }
    Ok(current)
}

fn create_temporary(directory: &Path) -> Result<(PathBuf, File)> {
    loop {
        let sequence = BAKE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".node-bake-{}-{sequence}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PotError::io(&error)),
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "node-bake resource and path tests")]

    use std::{fs, os::unix::fs::symlink};

    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        NodeBakeFrame, NodeBakeRecord, NodeBakeValue, delete_node_bake_payload,
        read_node_bake_payload, validate_node_bake_config, validate_node_bake_resource,
        write_node_bake_payload,
    };
    use crate::{error::ErrorCode, model::Id};

    fn frame(frame: f64, value: serde_json::Value) -> NodeBakeFrame {
        NodeBakeFrame {
            frame,
            values: [("Item_0".to_owned(), NodeBakeValue::Scalar { value })].into(),
        }
    }

    #[test]
    fn inline_resource_validation_rejects_binary_assets_and_preserves_material_values() {
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

        let binary_asset = json!({
            "uri":"images/sha256-deadbeef",
            "hash":"sha256:deadbeef",
            "kind":"packed_image",
            "packed":true,
            "bytes":[1,2,3]
        });
        assert_eq!(
            validate_node_bake_resource(&binary_asset).unwrap_err().code,
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
    fn disk_payloads_are_digest_checked_atomic_and_deletable() {
        let project = tempdir().unwrap();
        let object_id = Id::new("object").unwrap();
        let modifier_id = Id::new("modifier").unwrap();
        let payload = b"node bake payload";
        let digest = crate::hash::sha256(payload);

        assert_eq!(
            write_node_bake_payload(
                project.path(),
                "",
                &object_id,
                &modifier_id,
                "bake",
                "sha256:invalid",
                payload,
            )
            .unwrap_err()
            .code,
            ErrorCode::InvalidArgument
        );
        assert!(!project.path().join(".potter/bakes").exists());

        write_node_bake_payload(
            project.path(),
            "//cache-a",
            &object_id,
            &modifier_id,
            "bake",
            &digest,
            payload,
        )
        .unwrap();
        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache-a",
                &object_id,
                &modifier_id,
                "bake",
                &digest
            )
            .unwrap()
            .unwrap(),
            payload.to_vec()
        );

        let (root, relative) = super::node_bake_relative_path(
            project.path(),
            "//cache-a",
            &object_id,
            &modifier_id,
            "bake",
            &digest,
        )
        .unwrap();
        let path = root.join(relative);
        let files = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(files, vec![path.file_name().unwrap().to_os_string()]);

        fs::write(&path, b"corrupt").unwrap();
        assert_eq!(
            read_node_bake_payload(
                project.path(),
                "//cache-a",
                &object_id,
                &modifier_id,
                "bake",
                &digest
            )
            .unwrap_err()
            .code,
            ErrorCode::SceneInvalid
        );
        delete_node_bake_payload(
            project.path(),
            "//cache-a",
            &object_id,
            &modifier_id,
            "bake",
            &digest,
        )
        .unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn disk_payload_paths_reject_parent_traversal_and_symlink_escape() {
        let project = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let object_id = Id::new("object").unwrap();
        let modifier_id = Id::new("modifier").unwrap();
        let bytes = b"outside must stay empty";
        let digest = crate::hash::sha256(bytes);

        assert_eq!(
            write_node_bake_payload(
                project.path(),
                "../outside",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
                bytes,
            )
            .unwrap_err()
            .code,
            ErrorCode::InvalidArgument
        );
        symlink(outside.path(), project.path().join("cache")).unwrap();
        assert_eq!(
            write_node_bake_payload(
                project.path(),
                "//cache",
                &object_id,
                &modifier_id,
                "bake",
                &digest,
                bytes,
            )
            .unwrap_err()
            .code,
            ErrorCode::SceneInvalid
        );
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
    }
}
