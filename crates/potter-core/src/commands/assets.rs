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
    model::{ImageSource, LibraryKind},
    response::SceneInfo,
    store::Project,
};

#[expect(
    clippy::too_many_lines,
    reason = "one command constructs a single complete asset verification report"
)]
pub fn run(args: AssetsArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let mut assets = Vec::new();
    for (id, value) in &project.doc().resources {
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
        let (missing, changed, packed) = if let Some(resource_id) = &library.resource {
            let resource = project.doc().resources.get(resource_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "library refers to a missing packed resource",
                    json!({"library_id":id,"resource_id":resource_id}),
                )
            })?;
            let packed =
                bool_field(resource.as_object(), &["packed", "is_packed"]).unwrap_or(false);
            let status = if packed {
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
    let resource_hashes = project
        .doc()
        .resources
        .values()
        .filter(|resource| {
            resource
                .get("uri")
                .and_then(Value::as_str)
                .is_some_and(|uri| uri.starts_with("assets/"))
        })
        .filter_map(|resource| resource.get("hash").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
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
    Ok((
        Some(scene),
        json!({"assets":assets,"checked":args.check,"summary":{"count":assets.len(),"missing":assets.iter().filter(|asset| asset["missing"] == true).count(),"changed":assets.iter().filter(|asset| asset["changed"] == true).count()}}),
    ))
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
