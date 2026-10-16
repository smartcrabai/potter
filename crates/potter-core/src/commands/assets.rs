use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Map, Value, json};

use crate::{
    cli::AssetsArgs,
    error::{ErrorCode, PotError, Result},
    hash,
    model::{ImageSource, LibraryKind, migrate::inline_resource_kind},
    response::SceneInfo,
    store::Project,
};

use super::util::with_load_warnings;

#[expect(
    clippy::too_many_lines,
    reason = "one command constructs a single complete asset verification report"
)]
pub fn run(args: AssetsArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let mut assets = Vec::new();
    for (id, value) in &project.doc().resources {
        if inline_resource_kind(value)?.is_some() {
            continue;
        }
        if is_sequence_resource(value) {
            append_sequence_assets(project.path(), id.as_str(), value, args.check, &mut assets)?;
            continue;
        }
        let object = value.as_object();
        let uri = string_field(object, &["uri", "path", "source"]);
        let expected = string_field(object, &["hash", "expected_hash"]);
        let packed = bool_field(object, &["packed", "is_packed"]).unwrap_or(false);
        let path = uri
            .as_deref()
            .and_then(|uri| resolve_uri(project.path(), uri));
        let (missing, changed) = if packed {
            packed_resource_status(value, expected.as_deref(), args.check)?
        } else {
            file_status(path.as_deref(), expected.as_deref(), args.check)?
        };
        assets.push(json!({
            "uri":uri,"hash":expected,"kind":string_field(object, &["kind", "type"]).unwrap_or_else(|| "resource".to_owned()),
            "owner":string_field(object, &["owner", "owner_id"]).unwrap_or_else(|| id.to_string()),
            "packed":packed,"missing":missing,"changed":changed
        }));
    }
    for (id, library) in &project.doc().libraries {
        let resource = if let Some(resource_id) = &library.resource {
            let resource = project.doc().resources.get(resource_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "library refers to a missing packed resource",
                    json!({"library_id":id,"resource_id":resource_id}),
                )
            })?;
            (!inline_resource_kind(resource)?.is_some()).then_some(resource)
        } else {
            None
        };
        let (missing, changed, packed) = if let Some(resource) = resource {
            let packed =
                bool_field(resource.as_object(), &["packed", "is_packed"]).unwrap_or(false);
            let status = if is_sequence_resource(resource) {
                let resource_id = library.resource.as_ref().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "library sequence resource is missing",
                    )
                })?;
                let mut found = false;
                let mut missing = false;
                let mut changed = false;
                for frame in assets
                    .iter()
                    .filter(|asset| asset["owner"] == resource_id.as_str())
                {
                    found = true;
                    missing |= frame["missing"] == true;
                    changed |= frame["changed"] == true;
                }
                (if found { missing } else { true }, changed)
            } else if packed {
                packed_resource_status(resource, Some(&library.hash), args.check)?
            } else {
                let uri = string_field(resource.as_object(), &["uri", "path", "source"]);
                file_status(
                    uri.as_deref()
                        .and_then(|uri| resolve_uri(project.path(), uri))
                        .as_deref(),
                    Some(&library.hash),
                    args.check,
                )?
            };
            (status.0, status.1, packed)
        } else if library.kind == LibraryKind::Blend {
            let path = resolve_uri(project.path(), &library.resolved_path);
            let (missing, changed) = file_status(path.as_deref(), Some(&library.hash), args.check)?;
            (missing, changed, false)
        } else {
            let (missing, changed) =
                library_status(Some(&library.uri), Some(&library.hash), args.check)?;
            (missing, changed, false)
        };
        assets.push(json!({
            "uri":library.uri,
            "resolved_path":library.resolved_path,
            "hash":library.hash,
            "kind":"library",
            "library_kind":library.kind,
            "library_name":library.name,
            "owner":id,
            "library_id":id,
            "resource":library.resource,
            "packed":packed,
            "missing":missing,
            "changed":changed
        }));
    }
    for (id, image) in &project.doc().images {
        let (source, uri, expected, path) = match image.source {
            ImageSource::File => {
                let uri = image.source_path.as_deref();
                let path = uri.and_then(|value| resolve_uri(project.path(), value));
                ("file", uri, image.source_hash.as_deref(), path)
            }
            ImageSource::Packed => ("packed", image.blob.as_deref(), image.blob.as_deref(), None),
            ImageSource::Generated => (
                "generated",
                image.blob.as_deref(),
                image.blob.as_deref(),
                None,
            ),
        };
        let (missing, changed) =
            if source == "packed" || (source == "generated" && image.blob.is_some()) {
                image_blob_status(project.path(), expected, args.check)?
            } else if source == "file" && path.is_none() {
                (true, false)
            } else {
                file_status(path.as_deref(), expected, args.check)?
            };
        assets.push(json!({"id":id,"name":image.name,"source":source,"uri":uri,"hash":expected,"kind":"image","owner":id,"packed":source == "packed","missing":missing,"changed":changed}));
    }
    let mut blobs = Vec::new();
    for category in ["assets", "compat"] {
        let root = project.path().join(category);
        collect_blobs(&root, project.path(), category, args.check, &mut blobs)?;
    }
    let scene = project.info()?;
    let mut resource_hashes = BTreeSet::new();
    for resource in project.doc().resources.values() {
        if inline_resource_kind(resource)?.is_some() {
            continue;
        }
        if is_sequence_resource(resource) {
            if let Some(frames) = resource.get("sequence_frames").and_then(Value::as_object) {
                for leaf in frames.values() {
                    collect_resource_hash(leaf, &mut resource_hashes);
                }
            }
        } else {
            collect_resource_hash(resource, &mut resource_hashes);
        }
    }
    blobs.retain(|blob| {
        let uri = blob["uri"].as_str().unwrap_or_default();
        let hash = blob["hash"].as_str();
        !(uri.ends_with("/blob") && hash.is_some_and(|hash| resource_hashes.contains(hash)))
    });
    assets.extend(blobs);
    assets.sort_by(|left, right| left["uri"].as_str().cmp(&right["uri"].as_str()));
    let mut seen_uris = BTreeSet::new();
    assets.retain(|asset| {
        if asset["kind"] == "library" {
            return true;
        }
        asset["uri"]
            .as_str()
            .is_none_or(|uri| seen_uris.insert(uri.to_owned()))
    });
    let warnings = with_load_warnings(&[], &project)?;
    Ok((
        Some(scene),
        json!({"assets":assets,"checked":args.check,"summary":{"count":assets.len(),"missing":assets.iter().filter(|asset| asset["missing"] == true).count(),"changed":assets.iter().filter(|asset| asset["changed"] == true).count()},"warnings":warnings}),
    ))
}
fn is_sequence_resource(value: &Value) -> bool {
    value.get("sequence_frames").is_some()
        || value.get("sequence_start").is_some()
        || value.get("sequence_end").is_some()
}
fn append_sequence_assets(
    root: &Path,
    id: &str,
    container: &Value,
    check: bool,
    output: &mut Vec<Value>,
) -> Result<()> {
    let invalid = || {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "Alembic sequence resource is malformed or incomplete",
            json!({"resource_id":id}),
        )
    };
    if container.get("kind").and_then(Value::as_str) != Some("alembic") {
        return Err(invalid());
    }
    let packed = container
        .get("packed")
        .and_then(Value::as_bool)
        .ok_or_else(invalid)?;
    let start = container
        .get("sequence_start")
        .and_then(Value::as_i64)
        .and_then(|frame| i32::try_from(frame).ok())
        .ok_or_else(invalid)?;
    let end = container
        .get("sequence_end")
        .and_then(Value::as_i64)
        .and_then(|frame| i32::try_from(frame).ok())
        .filter(|end| *end >= start)
        .ok_or_else(invalid)?;
    let frames = container
        .get("sequence_frames")
        .and_then(Value::as_object)
        .ok_or_else(invalid)?;
    let expected_count = i64::from(end) - i64::from(start) + 1;
    if packed && i64::try_from(frames.len()).ok() != Some(expected_count) {
        return Err(invalid());
    }
    let mut ordered = Vec::with_capacity(frames.len());
    for (key, leaf) in frames {
        let frame = key
            .parse::<i32>()
            .ok()
            .filter(|frame| (*frame >= start) && (*frame <= end))
            .ok_or_else(invalid)?;
        if inline_resource_kind(leaf)?.is_some() {
            return Err(invalid());
        }
        let leaf_packed = leaf
            .get("packed")
            .and_then(Value::as_bool)
            .ok_or_else(invalid)?;
        if leaf
            .get("uri")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || leaf
                .get("kind")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || leaf
                .get("hash")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || (leaf_packed && leaf.get("bytes").and_then(Value::as_array).is_none())
            || (packed && !leaf_packed)
        {
            return Err(invalid());
        }
        ordered.push((frame, leaf));
    }
    ordered.sort_by_key(|(frame, _)| *frame);
    if ordered.windows(2).any(|window| window[0].0 == window[1].0) {
        return Err(invalid());
    }
    for (frame, leaf) in ordered {
        let uri = leaf["uri"].as_str().ok_or_else(invalid)?;
        let expected = leaf["hash"].as_str().ok_or_else(invalid)?;
        let kind = leaf["kind"].as_str().ok_or_else(invalid)?;
        let leaf_packed = leaf["packed"].as_bool().ok_or_else(invalid)?;
        let (missing, changed) = if leaf_packed {
            packed_resource_status(leaf, Some(expected), check)?
        } else {
            let path = resolve_uri(root, uri);
            file_status(path.as_deref(), Some(expected), check)?
        };
        output.push(json!({
            "uri":uri,
            "hash":expected,
            "kind":kind,
            "owner":id,
            "packed":leaf_packed,
            "missing":missing,
            "changed":changed,
            "sequence_frame":frame
        }));
    }
    Ok(())
}
fn collect_resource_hash<'a>(resource: &'a Value, hashes: &mut BTreeSet<&'a str>) {
    if resource
        .get("uri")
        .and_then(Value::as_str)
        .is_some_and(|uri| uri.starts_with("assets/"))
        && let Some(digest) = resource.get("hash").and_then(Value::as_str)
    {
        hashes.insert(digest);
    }
}

fn string_field(object: Option<&Map<String, Value>>, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        object
            .and_then(|object| object.get(*name))
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

fn bool_field(object: Option<&Map<String, Value>>, names: &[&str]) -> Option<bool> {
    names.iter().find_map(|name| {
        object
            .and_then(|object| object.get(*name))
            .and_then(Value::as_bool)
    })
}

fn resolve_uri(root: &Path, uri: &str) -> Option<PathBuf> {
    if let Some(path) = uri.strip_prefix("file://") {
        return Some(PathBuf::from(path));
    }
    if uri.contains("://") {
        return None;
    }
    let path = PathBuf::from(uri);
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

fn file_status(path: Option<&Path>, expected: Option<&str>, rehash: bool) -> Result<(bool, bool)> {
    let Some(path) = path else {
        return Ok((false, false));
    };
    if !path.is_file() {
        return Ok((true, false));
    }
    if !rehash {
        return Ok((false, false));
    }
    let bytes = fs::read(path).map_err(|error| {
        PotError::with_details(ErrorCode::IoError, error.to_string(), json!({"path":path}))
    })?;
    let changed = expected.is_some_and(|expected| normalize_hash(expected) != hash::sha256(&bytes));
    Ok((false, changed))
}
fn sequence_resource_status(
    resource: &Value,
    resource_id: &crate::model::Id,
    project_root: &Path,
    check: bool,
) -> Result<(bool, bool)> {
    let invalid = |message: &str| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            message,
            json!({"resource_id":resource_id}),
        )
    };
    let kind = resource
        .get("kind")
        .and_then(Value::as_str)
        .filter(|kind| !kind.trim().is_empty())
        .ok_or_else(|| invalid("Alembic sequence has no kind"))?;
    if resource
        .get("uri")
        .and_then(Value::as_str)
        .is_none_or(|uri| uri.trim().is_empty())
    {
        return Err(invalid("Alembic sequence has no URI"));
    }
    let packed = resource
        .get("packed")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("Alembic sequence has no packed state"))?;
    let frames = resource
        .get("sequence_frames")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Alembic sequence has invalid frame records"))?;
    let range = match (
        resource.get("sequence_start").and_then(Value::as_i64),
        resource.get("sequence_end").and_then(Value::as_i64),
    ) {
        (Some(start), Some(end)) => {
            let start = i32::try_from(start)
                .map_err(|_| invalid("Alembic sequence start is outside int32 range"))?;
            let end = i32::try_from(end)
                .map_err(|_| invalid("Alembic sequence end is outside int32 range"))?;
            if end < start {
                return Err(invalid("Alembic sequence frame range is not ascending"));
            }
            let count = i64::from(end) - i64::from(start) + 1;
            if count > crate::cache_sequence::MAX_CACHE_SEQUENCE_FRAMES {
                return Err(PotError::with_details(
                    ErrorCode::LimitExceeded,
                    "Alembic sequence frame range exceeds the resource limit",
                    json!({"resource_id":resource_id,"frame_count":count}),
                ));
            }
            Some((start, end, count))
        }
        (None, None) if !packed => None,
        _ => {
            return Err(invalid(
                "Alembic sequence has an invalid inclusive frame range",
            ));
        }
    };
    let frame_count = i64::try_from(frames.len())
        .map_err(|_| invalid("Alembic sequence has too many frame records"))?;
    if frame_count > crate::cache_sequence::MAX_CACHE_SEQUENCE_FRAMES {
        return Err(PotError::with_details(
            ErrorCode::LimitExceeded,
            "Alembic sequence has too many frame records",
            json!({"resource_id":resource_id,"frame_count":frame_count}),
        ));
    }
    let mut missing = frames.is_empty();
    for frame_key in frames.keys() {
        let frame = frame_key
            .parse::<i32>()
            .ok()
            .filter(|frame| frame.to_string() == frame_key.as_str())
            .ok_or_else(|| invalid("Alembic sequence frame key is not a canonical int32"))?;
        if range.is_some_and(|(start, end, _)| frame < start || frame > end) {
            return Err(invalid(
                "Alembic sequence frame is outside its inclusive range",
            ));
        }
    }
    if let Some((_, _, expected_count)) = range {
        if packed && frame_count != expected_count {
            return Err(invalid(
                "packed Alembic sequence does not contain every frame in its range",
            ));
        }
        if !packed && frame_count < expected_count {
            missing = true;
        }
    }
    let mut changed = false;
    for (frame, record) in frames {
        let record_kind = record.get("kind").and_then(Value::as_str);
        let record_uri = record
            .get("uri")
            .and_then(Value::as_str)
            .filter(|uri| !uri.trim().is_empty());
        let record_packed = record.get("packed").and_then(Value::as_bool);
        let has_bytes = record.get("bytes").and_then(Value::as_array).is_some();
        let hash = record.get("hash").and_then(Value::as_str);
        if record_kind != Some(kind)
            || record_uri.is_none()
            || record_packed != Some(packed)
            || hash.is_none()
            || (packed && !has_bytes)
            || (!packed && record.get("bytes").is_some())
        {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "Alembic sequence frame is not a standard resource record",
                json!({"resource_id":resource_id,"frame":frame}),
            ));
        }
        if !check {
            continue;
        }
        match crate::cache_sequence::read_cache_resource_record(
            record,
            &format!("{resource_id}:{frame}"),
            Some(project_root),
        ) {
            Ok(_) => {}
            Err(error) if error.code == ErrorCode::AssetChanged => changed = true,
            Err(error) if error.code == ErrorCode::DependencyMissing => missing = true,
            Err(error) => return Err(error),
        }
    }
    Ok((missing, changed))
}
fn packed_resource_status(
    value: &Value,
    expected: Option<&str>,
    check: bool,
) -> Result<(bool, bool)> {
    if !check {
        return Ok((false, false));
    }
    let Some(values) = value.get("bytes").and_then(Value::as_array) else {
        return Ok((true, false));
    };
    let mut bytes = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let byte = value
            .as_u64()
            .filter(|number| *number <= u8::MAX.into())
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "packed resource byte is outside 0..255",
                    json!({"index":index}),
                )
            })?;
        bytes.push(u8::try_from(byte).map_err(|_| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "packed resource byte is outside 0..255",
            )
        })?);
    }
    let Some(expected) = expected else {
        return Ok((false, true));
    };
    Ok((false, normalize_hash(expected) != hash::sha256(&bytes)))
}

fn library_status(uri: Option<&str>, expected: Option<&str>, check: bool) -> Result<(bool, bool)> {
    if !check {
        return Ok((false, false));
    }
    let Some(uri) = uri else {
        return Ok((true, false));
    };
    match crate::library::linked_library_hash(uri) {
        Ok(actual) => Ok((
            false,
            expected.is_none_or(|value| normalize_hash(value) != actual),
        )),
        Err(error) if error.code == ErrorCode::FileNotFound => Ok((true, false)),
        Err(error) => Err(error),
    }
}

fn image_blob_status(root: &Path, expected: Option<&str>, check: bool) -> Result<(bool, bool)> {
    if !check {
        return Ok((false, false));
    }
    let Some(expected) = expected else {
        return Ok((true, false));
    };
    let digest = normalize_hash(expected);
    let Some(hex) = digest
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
    else {
        return file_status(resolve_uri(root, expected).as_deref(), Some(expected), true);
    };
    let directory = root.join("assets").join("sha256").join(hex);
    if !directory.is_dir() {
        return Ok((true, false));
    }
    let mut found_file = false;
    for entry in fs::read_dir(&directory).map_err(|error| PotError::io(&error))? {
        let path = entry.map_err(|error| PotError::io(&error))?.path();
        if !path.is_file() {
            continue;
        }
        found_file = true;
        let bytes = fs::read(&path).map_err(|error| {
            PotError::with_details(ErrorCode::IoError, error.to_string(), json!({"path":path}))
        })?;
        if hash::sha256(&bytes) == digest {
            return Ok((false, false));
        }
    }
    Ok((!found_file, found_file))
}

fn normalize_hash(value: &str) -> String {
    if value.starts_with("sha256:") {
        value.to_owned()
    } else {
        format!("sha256:{value}")
    }
}

fn collect_blobs(
    root: &Path,
    scene_root: &Path,
    category: &str,
    check: bool,
    output: &mut Vec<Value>,
) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root).map_err(|error| PotError::io(&error))? {
        let entry = entry.map_err(|error| PotError::io(&error))?;
        let path = entry.path();
        if path.is_dir() {
            collect_blobs(&path, scene_root, category, check, output)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let relative = path
            .strip_prefix(scene_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let digest = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .filter(|value| value.len() == 64)
            .map(|hex| format!("sha256:{hex}"));
        let expected = digest.as_deref();
        let (missing, changed) = file_status(Some(&path), expected, check)?;
        output.push(json!({"uri":relative,"hash":expected,"kind":if category == "compat" {"compatibility"} else {"asset"},"owner":Value::Null,"packed":true,"missing":missing,"changed":changed}));
    }
    Ok(())
}
