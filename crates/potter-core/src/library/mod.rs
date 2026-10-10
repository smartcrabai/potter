use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    model::{Id, LibraryKind, SceneDoc},
};

pub(crate) const REGISTRIES: &[&str] = &[
    "nodes",
    "data_blocks",
    "materials",
    "collections",
    "actions",
    "node_groups",
    "worlds",
    "resources",
];

fn project_paths(uri: &str) -> (PathBuf, PathBuf) {
    let path = PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri));
    let is_directory = path.is_dir() || path.extension().is_none();
    let scene_path = if is_directory {
        path.join("scene.json")
    } else if path.file_name().is_some_and(|name| name == "scene.json") {
        path.clone()
    } else {
        path.parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
            .join("scene.json")
    };
    let project = if is_directory {
        path
    } else {
        scene_path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    };
    (project, scene_path)
}

pub(crate) fn linked_library_hash(uri: &str) -> Result<String> {
    let (_, scene_path) = project_paths(uri);
    if !scene_path.is_file() {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            "linked Potter project does not contain scene.json",
            json!({"uri":uri,"path":scene_path}),
        ));
    }
    let bytes = fs::read(&scene_path).map_err(|error| PotError::io(&error))?;
    Ok(hash::sha256(&bytes))
}

pub(crate) fn load_project(uri: &str) -> Result<(PathBuf, SceneDoc, String)> {
    let (project, scene_path) = project_paths(uri);
    if !scene_path.is_file() {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            "linked Potter project does not contain scene.json",
            json!({"uri":uri,"path":scene_path}),
        ));
    }
    let bytes = fs::read(&scene_path).map_err(|error| PotError::io(&error))?;
    let doc: SceneDoc = serde_json::from_slice(&bytes).map_err(|error| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            error.to_string(),
            json!({"path":scene_path,"line":error.line(),"column":error.column()}),
        )
    })?;
    doc.validate()?;
    Ok((project, doc, hash::sha256(&bytes)))
}

pub fn validate_linked_assets(doc: &SceneDoc, project_root: Option<&Path>) -> Result<()> {
    for (id, library) in &doc.libraries {
        let actual = match library.kind {
            LibraryKind::PotterProject => linked_library_hash(&library.uri),
            LibraryKind::Blend => {
                linked_blend_bytes(doc, library, project_root).map(|bytes| hash::sha256(&bytes))
            }
        }
        .map_err(|error| {
            if error.code == ErrorCode::FileNotFound {
                PotError::with_details(
                    ErrorCode::AssetChanged,
                    "linked library is missing",
                    json!({
                        "library_id":id,
                        "uri":library.uri,
                        "expected_hash":library.hash,
                        "status":"missing"
                    }),
                )
            } else {
                error
            }
        })?;
        if !hash_matches(&library.hash, &actual) {
            return Err(PotError::with_details(
                ErrorCode::AssetChanged,
                "linked library content changed; reload or relocate the library before evaluation",
                json!({
                    "library_id":id,
                    "uri":library.uri,
                    "expected_hash":library.hash,
                    "actual_hash":actual,
                    "status":"changed"
                }),
            ));
        }
    }
    Ok(())
}

fn linked_blend_bytes(
    doc: &SceneDoc,
    library: &crate::model::Library,
    project_root: Option<&Path>,
) -> Result<Vec<u8>> {
    let resource = library
        .resource
        .as_ref()
        .and_then(|resource_id| doc.resources.get(resource_id));
    if resource
        .is_some_and(|resource| resource.get("packed").and_then(Value::as_bool) == Some(true))
    {
        let values = resource
            .and_then(|resource| resource.get("bytes"))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "packed Blender library resource has no byte payload",
                )
            })?;
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_u64()
                    .and_then(|byte| u8::try_from(byte).ok())
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "packed Blender library byte is outside 0..255",
                            json!({"resource_id":library.resource,"index":index}),
                        )
                    })
            })
            .collect();
    }
    let uri = resource
        .and_then(|resource| resource.get("uri"))
        .and_then(Value::as_str)
        .filter(|uri| !uri.is_empty())
        .unwrap_or({
            if library.resolved_path.is_empty() {
                library.uri.as_str()
            } else {
                library.resolved_path.as_str()
            }
        });
    let path_text = uri
        .strip_prefix("file://")
        .unwrap_or(uri)
        .strip_prefix("//")
        .unwrap_or_else(|| uri.strip_prefix("file://").unwrap_or(uri));
    let path = Path::new(path_text);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(root) = project_root {
        root.join(path)
    } else {
        return Err(PotError::with_details(
            ErrorCode::DependencyMissing,
            "relative Blender library resource cannot be resolved without a project root",
            json!({"library_uri":library.uri,"resource_uri":uri}),
        ));
    };
    fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                "linked Blender library is missing",
                json!({"uri":library.uri,"path":path}),
            )
        } else {
            PotError::io(&error)
        }
    })
}

pub(crate) fn hash_matches(expected: &str, actual: &str) -> bool {
    expected.strip_prefix("sha256:").unwrap_or(expected)
        == actual.strip_prefix("sha256:").unwrap_or(actual)
}

pub(crate) fn registry_entry(doc: &SceneDoc, name: &str, id: &Id) -> Result<Option<Value>> {
    fn serialize_entry<T: Serialize>(entry: Option<&T>) -> Result<Option<Value>> {
        entry
            .map(|value| {
                serde_json::to_value(value)
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
            })
            .transpose()
    }
    if !REGISTRIES.contains(&name) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("unsupported library registry `{name}`"),
        ));
    }
    match name {
        "nodes" => serialize_entry(doc.nodes.get(id)),
        "data_blocks" => serialize_entry(doc.data_blocks.get(id)),
        "materials" => serialize_entry(doc.materials.get(id)),
        "collections" => serialize_entry(doc.collections.get(id)),
        "actions" => serialize_entry(doc.actions.get(id)),
        "node_groups" => serialize_entry(doc.node_groups.get(id)),
        "worlds" => serialize_entry(doc.worlds.get(id)),
        "resources" => Ok(doc.resources.get(id).cloned()),
        _ => Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("unsupported library registry `{name}`"),
        )),
    }
}

pub(crate) fn namespaced_id(library: &str, source: &Id) -> Result<Id> {
    let prefix = format!("{library}__");
    let source_text = source.as_str();
    let available = 64_usize.saturating_sub(prefix.len());
    let mut suffix = source_text.chars().take(available).collect::<String>();
    if suffix.is_empty() {
        suffix.push('x');
    }
    Id::new(format!("{prefix}{suffix}"))
}

pub(crate) fn remap_references(value: &mut Value, ids: &BTreeMap<(String, String), String>) {
    fn visit(value: &mut Value, field: Option<&str>, ids: &BTreeMap<(String, String), String>) {
        match value {
            Value::String(text) => {
                let registry = match field {
                    Some("parent" | "objects" | "camera" | "node") => Some("nodes"),
                    Some("root_collection" | "children") => Some("collections"),
                    Some("data") => Some("data_blocks"),
                    Some("materials") => Some("materials"),
                    Some("action") => Some("actions"),
                    Some("world") => Some("worlds"),
                    Some("node_group" | "node_tree") => Some("node_groups"),
                    _ => None,
                };
                if let Some(mapped) =
                    registry.and_then(|registry| ids.get(&(registry.to_owned(), text.clone())))
                {
                    *text = mapped.clone();
                }
            }
            Value::Array(values) => values.iter_mut().for_each(|value| visit(value, field, ids)),
            Value::Object(values) => values
                .iter_mut()
                .for_each(|(field, value)| visit(value, Some(field), ids)),
            _ => {}
        }
    }
    visit(value, None, ids);
}

pub(crate) fn linked_ids_mut(doc: &mut SceneDoc) -> Result<&mut Map<String, Value>> {
    let slot = doc
        .compatibility
        .entry("linked_ids".to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    slot.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "compatibility linked_ids must be an object",
        )
    })
}

pub(crate) fn linked_key(registry: &str, id: &Id) -> String {
    format!("{registry}:{id}")
}
