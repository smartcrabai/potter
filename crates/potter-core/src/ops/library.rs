use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::NodeGroup,
    hash,
    library::{
        self, hash_matches, linked_key, load_project, namespaced_id, registry_entry,
        remap_references,
    },
    model::{
        Action, Collection, DataBlock, Id, Library, LibraryKind, LibraryOverride,
        LibraryOverrideProperty, LibraryStatus, Material, Modifier, Node, World,
    },
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_id, read_string};

const TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::Strict {
    object_message: "target must be an object",
    shape_message: "target must contain only id",
    id_message: "target id must be a string",
    require_id: true,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "library.link" => link(engine, operation),
        "library.append" => append(engine, operation),
        "library.override" => override_node(engine, operation),
        "library.register" => register(engine, operation),
        "library.reload" => reload(engine, operation),
        "library.relocate" => relocate(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid library operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn link(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "uri", "items"],
        &["id", "uri", "items"],
    )?;
    let library_id = read_id(engine, operation, "id")?;
    if engine.doc.libraries.contains_key(&library_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("library `{library_id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let uri = read_string(engine, operation, "uri")?;
    let (project_path, source, source_hash) = load_project(&uri)?;
    let items = parse_items(engine, operation)?;
    let (mappings, imported) =
        copy_items(engine, &source, &items, Some(library_id.as_str()), true)?;
    let mut source_items = BTreeMap::new();
    let mut linked_ids = BTreeMap::<String, Vec<Id>>::new();
    for ((registry, source_id), mapped) in &mappings {
        let mapped_id = Id::new(mapped.as_str())?;
        source_items.insert(linked_key(registry, &mapped_id), source_id.clone());
        linked_ids
            .entry(registry.clone())
            .or_default()
            .push(mapped_id);
    }
    let record = Library {
        name: library_id.to_string(),
        kind: LibraryKind::PotterProject,
        uri: uri.clone(),
        resolved_path: project_path.to_string_lossy().into_owned(),
        resource: None,
        hash: source_hash,
        status: LibraryStatus::Ok,
        linked_ids,
        overrides: Vec::new(),
        items: source_items,
        source_project: Some(project_path.to_string_lossy().into_owned()),
    };
    engine.doc.libraries.insert(library_id.clone(), record);
    engine.mark("libraries", &library_id, ChangeKind::Created);
    for (registry, id, value) in imported {
        insert_entry(engine, &registry, &id, value, ChangeKind::Created)?;
    }
    for ((registry, source_id), mapped) in mappings {
        let source_id = Id::new(source_id)?;
        let library_name = registry_entry(&source, &registry, &source_id)?
            .and_then(|value| value.get("name").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_else(|| source_id.to_string());
        register_linked(
            engine,
            &registry,
            &Id::new(mapped)?,
            &library_id,
            &source_id,
            &uri,
            &library_name,
        )?;
    }
    Ok(true)
}

#[expect(
    clippy::too_many_lines,
    reason = "registration validates and commits one complete linked-ID declaration"
)]
fn register(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "kind",
            "uri",
            "resolved_path",
            "resource",
            "hash",
            "items",
        ],
        &["id", "name", "kind", "uri", "resolved_path", "items"],
    )?;
    let library_id = read_id(engine, operation, "id")?;
    if engine.doc.libraries.contains_key(&library_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("library `{library_id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let kind = read_string(engine, operation, "kind")?;
    if kind != "blend" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "library.register currently registers Blender libraries only",
            &operation_pointer(engine.operation_index, "kind"),
        ));
    }
    let name = read_string(engine, operation, "name")?;
    let uri = read_string(engine, operation, "uri")?;
    let resolved_input = read_string(engine, operation, "resolved_path")?;
    let resolved_path = resolve_library_path(engine, &resolved_input)
        .to_string_lossy()
        .into_owned();
    let resource = operation
        .get("resource")
        .filter(|value| !value.is_null())
        .map(|_| read_id(engine, operation, "resource"))
        .transpose()?;
    if resource
        .as_ref()
        .is_some_and(|resource| !engine.doc.resources.contains_key(resource))
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "library resource does not exist",
            &operation_pointer(engine.operation_index, "resource"),
        ));
    }
    let items = operation
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "items must be an array",
                &operation_pointer(engine.operation_index, "items"),
            )
        })?;
    if items.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "items must contain at least one linked item",
            &operation_pointer(engine.operation_index, "items"),
        ));
    }

    let mut linked_ids = BTreeMap::<String, Vec<Id>>::new();
    let mut source_items = BTreeMap::new();
    let mut registrations = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let pointer = format!("items/{index}");
        let object = item.as_object().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "library item must be an object",
                &operation_pointer(engine.operation_index, &pointer),
            )
        })?;
        if object
            .keys()
            .any(|key| !["registry", "id", "name", "source_id"].contains(&key.as_str()))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "unknown library item field",
                &operation_pointer(engine.operation_index, &pointer),
            ));
        }
        let registry = object
            .get("registry")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "library item registry must be a string",
                    &operation_pointer(engine.operation_index, &format!("{pointer}/registry")),
                )
            })?
            .to_owned();
        if !library::REGISTRIES.contains(&registry.as_str()) {
            return Err(engine.error(
                ErrorCode::UnsupportedFeature,
                format!("library linking `{registry}` is not supported"),
                &operation_pointer(engine.operation_index, &format!("{pointer}/registry")),
            ));
        }
        let raw_id = object.get("id").and_then(Value::as_str).ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "library item id must be a string",
                &operation_pointer(engine.operation_index, &format!("{pointer}/id")),
            )
        })?;
        let item_id = parse_id(
            engine,
            raw_id,
            &operation_pointer(engine.operation_index, &format!("{pointer}/id")),
        )?;
        if !target_exists(engine, &registry, &item_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("linked `{registry}` item `{item_id}` does not exist"),
                &operation_pointer(engine.operation_index, &format!("{pointer}/id")),
            ));
        }
        if engine
            .doc
            .compatibility
            .get("linked_ids")
            .and_then(Value::as_object)
            .is_some_and(|linked| linked.contains_key(&linked_key(&registry, &item_id)))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("`{registry}` item `{item_id}` is already linked"),
                &operation_pointer(engine.operation_index, &format!("{pointer}/id")),
            ));
        }
        let library_name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "library item name must be a string",
                    &operation_pointer(engine.operation_index, &format!("{pointer}/name")),
                )
            })?
            .to_owned();
        let source_id = object
            .get("source_id")
            .and_then(Value::as_str)
            .map(|source_id| {
                parse_id(
                    engine,
                    source_id,
                    &operation_pointer(engine.operation_index, &format!("{pointer}/source_id")),
                )
            })
            .transpose()?
            .unwrap_or_else(|| item_id.clone());
        let key = linked_key(&registry, &item_id);
        if source_items.insert(key, source_id.to_string()).is_some() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "library items must not contain duplicate registry IDs",
                &operation_pointer(engine.operation_index, &pointer),
            ));
        }
        linked_ids
            .entry(registry.clone())
            .or_default()
            .push(item_id.clone());
        registrations.push((registry, item_id, source_id, library_name));
    }

    let expected_hash = operation
        .get("hash")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let (hash_value, status) = match fs::read(&resolved_path) {
        Ok(bytes) => {
            let actual = hash::sha256(&bytes);
            let matches = expected_hash.as_deref().is_none_or(|expected| {
                expected.strip_prefix("sha256:").unwrap_or(expected)
                    == actual.strip_prefix("sha256:").unwrap_or(&actual)
            });
            let status = if matches {
                LibraryStatus::Ok
            } else {
                LibraryStatus::Changed
            };
            (expected_hash.unwrap_or(actual), status)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            (expected_hash.unwrap_or_default(), LibraryStatus::Missing)
        }
        Err(error) => return Err(PotError::io(&error)),
    };
    let record = Library {
        name,
        kind: LibraryKind::Blend,
        uri: uri.clone(),
        resolved_path: resolved_path.clone(),
        resource,
        hash: hash_value,
        status,
        linked_ids,
        overrides: Vec::new(),
        items: source_items,
        source_project: None,
    };
    engine.doc.libraries.insert(library_id.clone(), record);
    engine.mark("libraries", &library_id, ChangeKind::Created);
    for (registry, item_id, source_id, library_name) in registrations {
        register_linked(
            engine,
            &registry,
            &item_id,
            &library_id,
            &source_id,
            &uri,
            &library_name,
        )?;
    }
    Ok(true)
}

fn append(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "library_id", "uri", "items"],
        &["items"],
    )?;
    let (source, uri) = if operation.contains_key("library_id") {
        let library_id = read_id(engine, operation, "library_id")?;
        let record = engine.doc.libraries.get(&library_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("library `{library_id}` does not exist"),
                &operation_pointer(engine.operation_index, "library_id"),
            )
        })?;
        let uri = record.uri.clone();
        (load_project(&uri)?.1, uri)
    } else {
        let uri = read_string(engine, operation, "uri")?;
        (load_project(&uri)?.1, uri)
    };
    let items = parse_items(engine, operation)?;
    let (mappings, imported) = copy_items(engine, &source, &items, None, false)?;
    for ((registry, old), new) in &mappings {
        let old_id = Id::new(old.clone())?;
        let new_id = Id::new(new.clone())?;
        engine.map_id(registry, &old_id, &new_id);
    }
    for (registry, id, value) in imported {
        insert_entry(engine, &registry, &id, value, ChangeKind::Created)?;
    }
    let _ = uri;
    Ok(true)
}

#[expect(
    clippy::too_many_lines,
    reason = "override creation applies all requested properties and records the reference atomically"
)]
fn override_node(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "operations"],
        &["target", "id", "operations"],
    )?;
    let target = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )?;
    let local_id = read_id(engine, operation, "id")?;
    if engine.doc.nodes.contains_key(&local_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("node `{local_id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let marker = engine
        .doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
        .and_then(|linked| linked.get(&linked_key("nodes", &target)))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "library.override target must be a linked node",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    let library_id = marker
        .get("library_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "linked node is missing its library ID",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    let library_id = Id::new(library_id.to_owned())?;
    let source_id = marker.get("source_id").cloned().unwrap_or(Value::Null);
    let mut node = engine.doc.nodes.get(&target).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{target}` does not exist"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let operations = operation
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "operations must be an array",
                &operation_pointer(engine.operation_index, "operations"),
            )
        })?;
    let mut applied = Vec::with_capacity(operations.len());
    let mut properties = Vec::with_capacity(operations.len());
    for (index, action) in operations.iter().enumerate() {
        let action = action.as_object().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "override operation must be an object",
                &operation_pointer(engine.operation_index, &format!("operations/{index}")),
            )
        })?;
        properties.push(apply_override_action(engine, &mut node, action, index)?);
        applied.push(Value::Object(action.clone()));
    }
    node.properties.remove("library");
    node.properties.remove("library_name");
    node.properties.remove("editable");
    node.properties.remove("source");
    node.properties.insert(
        "library_override".to_owned(),
        json!({
            "library_id":library_id,
            "reference_id":target,
            "source_id":source_id,
            "operations":applied
        }),
    );
    engine.doc.nodes.insert(local_id.clone(), node);
    engine.mark("nodes", &local_id, ChangeKind::Created);
    let library = engine.doc.libraries.get_mut(&library_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "linked node refers to a missing library",
            json!({"library_id":library_id,"node_id":target}),
        )
    })?;
    library.overrides.push(LibraryOverride {
        registry: "nodes".to_owned(),
        id: local_id,
        reference_id: target,
        properties,
    });
    engine.mark("libraries", &library_id, ChangeKind::Updated);
    Ok(true)
}

#[expect(
    clippy::too_many_lines,
    reason = "override actions validate and apply the complete supported operation set"
)]
fn apply_override_action(
    engine: &Engine<'_>,
    node: &mut Node,
    action: &Map<String, Value>,
    index: usize,
) -> Result<LibraryOverrideProperty> {
    let pointer = |field: &str| {
        operation_pointer(
            engine.operation_index,
            &format!("operations/{index}/{field}"),
        )
    };
    if action
        .keys()
        .any(|key| !["op", "path", "value"].contains(&key.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "unknown library override operation field",
            &pointer(""),
        ));
    }
    let raw_operation = action.get("op").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "override operation must specify op",
            &pointer("op"),
        )
    })?;
    let operation = match raw_operation {
        "set" | "replace" => "replace",
        "insert_after" => "insert_after",
        "delete" => "delete",
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "override operation must be replace, insert_after, or delete",
                &pointer("op"),
            ));
        }
    };
    let path = action.get("path").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "override path must be a string",
            &pointer("path"),
        )
    })?;
    let value = action.get("value").cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "override operation requires value",
            &pointer("value"),
        )
    })?;
    match operation {
        "replace" => apply_override_replace(engine, node, path, &value, &pointer("value"))?,
        "insert_after" => {
            let modifier_id = path.strip_prefix("modifiers.").ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "insert_after only supports modifier stack entries",
                    &pointer("path"),
                )
            })?;
            let reference_id = Id::new(modifier_id.to_owned()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.message, &pointer("path"))
            })?;
            let position = node
                .modifiers
                .iter()
                .position(|modifier| modifier.id == reference_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "insert_after reference modifier does not exist",
                        &pointer("path"),
                    )
                })?;
            let modifier: Modifier = serde_json::from_value(value.clone()).map_err(|error| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    error.to_string(),
                    &pointer("value"),
                )
            })?;
            if node
                .modifiers
                .iter()
                .any(|existing| existing.id == modifier.id)
            {
                return Err(engine.error(
                    ErrorCode::IdExists,
                    format!("modifier `{}` already exists", modifier.id),
                    &pointer("value"),
                ));
            }
            node.modifiers.insert(position + 1, modifier);
        }
        "delete" => {
            let modifier_id = path.strip_prefix("modifiers.").ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "delete only supports modifier stack entries",
                    &pointer("path"),
                )
            })?;
            let modifier_id = Id::new(modifier_id.to_owned()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.message, &pointer("path"))
            })?;
            let position = node
                .modifiers
                .iter()
                .position(|modifier| modifier.id == modifier_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "delete reference modifier does not exist",
                        &pointer("path"),
                    )
                })?;
            node.modifiers.remove(position);
        }
        _ => {
            return Err(engine.error(
                ErrorCode::InternalError,
                "invalid library override operation dispatch",
                &pointer("op"),
            ));
        }
    }
    Ok(LibraryOverrideProperty {
        path: path.to_owned(),
        operation: operation.to_owned(),
        value,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the replacement whitelist is kept together to make overridable paths auditable"
)]
fn apply_override_replace(
    engine: &Engine<'_>,
    node: &mut Node,
    path: &str,
    value: &Value,
    pointer: &str,
) -> Result<()> {
    match path {
        "name" => {
            value
                .as_str()
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "name override must be a string",
                        pointer,
                    )
                })?
                .clone_into(&mut node.name);
            Ok(())
        }
        "transform" => {
            node.transform = serde_json::from_value(value.clone()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.to_string(), pointer)
            })?;
            Ok(())
        }
        "transform.translation" => {
            node.transform.translation =
                serde_json::from_value(value.clone()).map_err(|error| {
                    engine.error(ErrorCode::InvalidOperation, error.to_string(), pointer)
                })?;
            Ok(())
        }
        "transform.rotation" => {
            node.transform.rotation = serde_json::from_value(value.clone()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.to_string(), pointer)
            })?;
            Ok(())
        }
        "transform.scale" => {
            node.transform.scale = serde_json::from_value(value.clone()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.to_string(), pointer)
            })?;
            Ok(())
        }
        "visible" => {
            node.visible = value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "visible override must be a boolean",
                    pointer,
                )
            })?;
            Ok(())
        }
        "render_visible" => {
            node.render_visible = value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "render_visible override must be a boolean",
                    pointer,
                )
            })?;
            Ok(())
        }
        path if path.starts_with("properties.") => {
            let key = &path["properties.".len()..];
            if key.is_empty()
                || [
                    "library",
                    "library_name",
                    "editable",
                    "source",
                    "library_override",
                ]
                .contains(&key)
            {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "linked metadata cannot be overridden",
                    pointer,
                ));
            }
            node.properties.insert(key.to_owned(), value.clone());
            Ok(())
        }
        path if path.starts_with("modifiers.") => {
            let mut parts = path.split('.');
            let _ = parts.next();
            let modifier_id = parts.next().unwrap_or_default();
            let param = match (parts.next(), parts.next()) {
                (Some("params"), Some(param)) if parts.next().is_none() => param,
                (Some(param), None) => param,
                _ => "",
            };
            let modifier_id = Id::new(modifier_id.to_owned()).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.message, pointer)
            })?;
            let modifier = node
                .modifiers
                .iter_mut()
                .find(|modifier| modifier.id == modifier_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "modifier override target does not exist",
                        pointer,
                    )
                })?;
            if param.is_empty() || !modifier.params.contains_key(param) {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "modifier property is not declared overridable",
                    pointer,
                ));
            }
            modifier.params.insert(param.to_owned(), value.clone());
            Ok(())
        }
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "property is not overridable",
            pointer,
        )),
    }
}

fn reload(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id"], &["id"])?;
    let id = read_id(engine, operation, "id")?;
    let mut record = engine.doc.libraries.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("library `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    if record.kind == LibraryKind::Blend {
        let path_text = if record.resolved_path.is_empty() {
            &record.uri
        } else {
            &record.resolved_path
        };
        let path = resolve_library_path(engine, path_text);
        let bytes = fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                PotError::with_details(
                    ErrorCode::FileNotFound,
                    "linked Blender library is missing",
                    json!({"library_id":id,"uri":record.uri,"path":path}),
                )
            } else {
                PotError::io(&error)
            }
        })?;
        record.hash = hash::sha256(&bytes);
        record.status = LibraryStatus::Ok;
        engine.doc.libraries.insert(id.clone(), record);
        engine.mark("libraries", &id, ChangeKind::Updated);
        return Ok(true);
    }

    let (_path, source, digest) = load_project(&record.uri)?;
    let mut id_map = BTreeMap::new();
    let mut replacements = Vec::new();
    for (key, source_id) in &record.items {
        let Some((registry_name, local_text)) = key.split_once(':') else {
            continue;
        };
        let local_id = Id::new(local_text)?;
        let source_id_id = Id::new(source_id.clone())?;
        let source_value =
            registry_entry(&source, registry_name, &source_id_id)?.ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!(
                        "source item `{source_id}` was removed; library reload cannot resolve it"
                    ),
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?;
        id_map.insert(
            (registry_name.to_owned(), source_id.clone()),
            local_id.to_string(),
        );
        replacements.push((
            registry_name.to_owned(),
            local_id,
            source_value,
            source_id.clone(),
        ));
    }
    for (registry_name, local_id, mut value, source_id) in replacements {
        remap_references(&mut value, &id_map);
        let library_name = value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(&source_id)
            .to_owned();
        set_link_markers(&mut value, &registry_name, &id, &source_id, &library_name);
        update_entry(engine, &registry_name, &local_id, value)?;
    }
    record.hash = digest;
    record.status = LibraryStatus::Ok;
    engine.doc.libraries.insert(id.clone(), record);
    engine.mark("libraries", &id, ChangeKind::Updated);
    Ok(true)
}

fn relocate(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "uri", "resolved_path"],
        &["id", "uri"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let uri = read_string(engine, operation, "uri")?;
    let existing = engine.doc.libraries.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("library `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    let (resolved_path, status) = if existing.kind == LibraryKind::Blend {
        let path_text = operation
            .get("resolved_path")
            .and_then(Value::as_str)
            .unwrap_or(&uri);
        let path = resolve_library_path(engine, path_text);
        let bytes = fs::read(&path).map_err(|error| PotError::io(&error))?;
        let status = if hash_matches(&existing.hash, &hash::sha256(&bytes)) {
            LibraryStatus::Ok
        } else {
            LibraryStatus::Changed
        };
        (path.to_string_lossy().into_owned(), status)
    } else {
        let (project_path, _, digest) = load_project(&uri)?;
        let status = if hash_matches(&existing.hash, &digest) {
            LibraryStatus::Ok
        } else {
            LibraryStatus::Changed
        };
        (project_path.to_string_lossy().into_owned(), status)
    };
    let record = engine.doc.libraries.get_mut(&id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "library disappeared during relocation",
        )
    })?;
    record.status = status;
    record.uri = uri;
    record.resolved_path = resolved_path;
    engine.mark("libraries", &id, ChangeKind::Updated);
    Ok(true)
}

fn resolve_library_path(engine: &Engine<'_>, value: &str) -> PathBuf {
    let path = PathBuf::from(value.strip_prefix("file://").unwrap_or(value));
    if path.is_absolute() {
        path
    } else {
        engine
            .asset_root
            .map_or(path.clone(), |root| root.join(path))
    }
}

fn parse_items(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Vec<(String, Id)>> {
    let items = operation
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "items must be an array",
                &operation_pointer(engine.operation_index, "items"),
            )
        })?;
    if items.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "items must contain at least one item",
            &operation_pointer(engine.operation_index, "items"),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let object = item.as_object().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "library item must be an object",
                    &operation_pointer(engine.operation_index, &format!("items/{index}")),
                )
            })?;
            if object
                .keys()
                .any(|key| !["registry", "id"].contains(&key.as_str()))
            {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "unknown library item field",
                    &operation_pointer(engine.operation_index, &format!("items/{index}")),
                ));
            }
            let registry = object
                .get("registry")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "library item registry must be a string",
                        &operation_pointer(
                            engine.operation_index,
                            &format!("items/{index}/registry"),
                        ),
                    )
                })?
                .to_owned();
            if !library::REGISTRIES.contains(&registry.as_str()) {
                let mut error = engine.error(
                    ErrorCode::UnsupportedFeature,
                    format!("library linking `{registry}` is not supported"),
                    &operation_pointer(engine.operation_index, &format!("items/{index}/registry")),
                );
                if let Some(details) = error.details.as_object_mut() {
                    details.insert(
                        "feature_id".to_owned(),
                        Value::String("asset.library.other_registries".to_owned()),
                    );
                }
                return Err(error);
            }
            let id_text = object.get("id").and_then(Value::as_str).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "library item id must be a string",
                    &operation_pointer(engine.operation_index, &format!("items/{index}/id")),
                )
            })?;
            Ok((
                registry,
                parse_id(
                    engine,
                    id_text,
                    &operation_pointer(engine.operation_index, &format!("items/{index}/id")),
                )?,
            ))
        })
        .collect()
}

type LibraryItem = (String, Id, Value);
type LibraryCopy = (BTreeMap<(String, String), String>, Vec<LibraryItem>);

fn copy_items(
    engine: &mut Engine<'_>,
    source: &crate::model::SceneDoc,
    items: &[(String, Id)],
    namespace: Option<&str>,
    linked: bool,
) -> Result<LibraryCopy> {
    let mut queue = items.to_vec();
    let mut selected = BTreeMap::<String, BTreeMap<Id, Value>>::new();
    while let Some((registry_name, source_id)) = queue.pop() {
        if selected
            .get(&registry_name)
            .is_some_and(|entries| entries.contains_key(&source_id))
        {
            continue;
        }
        let value = registry_entry(source, &registry_name, &source_id)?.ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("source {registry_name} item `{source_id}` does not exist"),
                &operation_pointer(engine.operation_index, "items"),
            )
        })?;
        match registry_name.as_str() {
            "nodes" => {
                for (field, registry_name) in [
                    ("data", "data_blocks"),
                    ("parent", "nodes"),
                    ("action", "actions"),
                ] {
                    if let Some(reference) = value.get(field).and_then(Value::as_str) {
                        queue.push((registry_name.to_owned(), Id::new(reference)?));
                    }
                }
                if let Some(materials) = value.get("materials").and_then(Value::as_array) {
                    for material in materials.iter().filter_map(Value::as_str) {
                        queue.push(("materials".to_owned(), Id::new(material)?));
                    }
                }
            }
            "collections" => {
                for (field, registry_name) in [("children", "collections"), ("objects", "nodes")] {
                    if let Some(references) = value.get(field).and_then(Value::as_array) {
                        for reference in references.iter().filter_map(Value::as_str) {
                            queue.push((registry_name.to_owned(), Id::new(reference)?));
                        }
                    }
                }
            }
            "materials" => {
                if let Some(node_tree) = value.get("node_tree").and_then(Value::as_str) {
                    queue.push(("node_groups".to_owned(), Id::new(node_tree)?));
                }
            }
            "actions" => {
                if let Some(slots) = value.get("slots").and_then(Value::as_array) {
                    for node in slots
                        .iter()
                        .filter_map(|slot| slot.get("node").and_then(Value::as_str))
                    {
                        queue.push(("nodes".to_owned(), Id::new(node)?));
                    }
                }
            }
            _ => {}
        }
        selected
            .entry(registry_name)
            .or_default()
            .insert(source_id, value);
    }
    let mut mappings = BTreeMap::new();
    let mut reserved = BTreeSet::new();
    for (registry_name, entries) in &selected {
        for source_id in entries.keys() {
            let target = if let Some(namespace) = namespace {
                namespaced_id(namespace, source_id)?
            } else {
                unique_copy_id(engine, source_id, registry_name, &reserved)?
            };
            if target_exists(engine, registry_name, &target)
                || !reserved.insert((registry_name.clone(), target.clone()))
            {
                return Err(engine.error(
                    ErrorCode::IdExists,
                    format!("copied ID `{target}` already exists in `{registry_name}`"),
                    &operation_pointer(engine.operation_index, "items"),
                ));
            }
            mappings.insert(
                (registry_name.clone(), source_id.to_string()),
                target.to_string(),
            );
        }
    }
    let by_id: BTreeMap<(String, String), String> = mappings
        .iter()
        .map(|(key, new)| (key.clone(), new.clone()))
        .collect();
    let mut imported = Vec::new();
    for (registry_name, entries) in selected {
        for (source_id, mut value) in entries {
            remap_references(&mut value, &by_id);
            let local_text = mappings
                .get(&(registry_name.clone(), source_id.to_string()))
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InternalError,
                        "library ID mapping is missing",
                        &operation_pointer(engine.operation_index, "items"),
                    )
                })?;
            let local_id = Id::new(local_text.clone())?;
            if linked {
                let library_name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| source_id.as_str())
                    .to_owned();
                set_link_markers(
                    &mut value,
                    &registry_name,
                    &Id::new(namespace.ok_or_else(|| {
                        engine.error(
                            ErrorCode::InternalError,
                            "library namespace missing",
                            &operation_pointer(engine.operation_index, "id"),
                        )
                    })?)?,
                    source_id.as_str(),
                    &library_name,
                );
            }
            imported.push((registry_name.clone(), local_id, value));
        }
    }
    Ok((mappings, imported))
}

fn set_link_markers(
    value: &mut Value,
    registry: &str,
    library_id: &Id,
    source_id: &str,
    library_name: &str,
) {
    if registry == "nodes"
        && let Some(object) = value.as_object_mut()
    {
        let properties = object.entry("properties").or_insert_with(|| json!({}));
        if let Some(properties) = properties.as_object_mut() {
            properties.insert("library".to_owned(), Value::String(library_id.to_string()));
            properties.insert(
                "library_name".to_owned(),
                Value::String(library_name.to_owned()),
            );
            properties.insert("editable".to_owned(), Value::Bool(false));
            properties.insert("source".to_owned(), Value::String(source_id.to_owned()));
        }
    }
}

fn unique_copy_id(
    engine: &Engine<'_>,
    source: &Id,
    registry_name: &str,
    reserved: &BTreeSet<(String, Id)>,
) -> Result<Id> {
    if !target_exists(engine, registry_name, source)
        && !reserved.contains(&(registry_name.to_owned(), source.clone()))
    {
        return Ok(source.clone());
    }
    for ordinal in 1_u32..u32::MAX {
        let suffix = format!("_copy{ordinal}");
        let base = source.as_str();
        let clipped = base
            .chars()
            .take(64_usize.saturating_sub(suffix.len()))
            .collect::<String>();
        let candidate = Id::new(format!("{clipped}{suffix}"))?;
        if !target_exists(engine, registry_name, &candidate)
            && !reserved.contains(&(registry_name.to_owned(), candidate.clone()))
        {
            return Ok(candidate);
        }
    }
    Err(PotError::new(
        ErrorCode::LimitExceeded,
        "no unique ID available for append",
    ))
}

fn target_exists(engine: &Engine<'_>, registry: &str, id: &Id) -> bool {
    match registry {
        "nodes" => engine.doc.nodes.contains_key(id),
        "data_blocks" => engine.doc.data_blocks.contains_key(id),
        "materials" => engine.doc.materials.contains_key(id),
        "collections" => engine.doc.collections.contains_key(id),
        "actions" => engine.doc.actions.contains_key(id),
        "node_groups" => engine.doc.node_groups.contains_key(id),
        "worlds" => engine.doc.worlds.contains_key(id),
        "resources" => engine.doc.resources.contains_key(id),
        _ => false,
    }
}

fn insert_entry(
    engine: &mut Engine<'_>,
    registry: &str,
    id: &Id,
    value: Value,
    change: ChangeKind,
) -> Result<()> {
    match registry {
        "nodes" => {
            let node: Node = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.nodes.insert(id.clone(), node);
        }
        "data_blocks" => {
            let block: DataBlock = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.data_blocks.insert(id.clone(), block);
        }
        "materials" => {
            let material: Material = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.materials.insert(id.clone(), material);
        }
        "collections" => {
            let collection: Collection = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.collections.insert(id.clone(), collection);
        }
        "actions" => {
            let action: Action = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.actions.insert(id.clone(), action);
        }
        "node_groups" => {
            let node_group: NodeGroup = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.node_groups.insert(id.clone(), node_group);
        }
        "worlds" => {
            let world: World = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.worlds.insert(id.clone(), world);
        }
        "resources" => {
            engine.doc.resources.insert(id.clone(), value);
        }
        _ => {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!("library insertion for `{registry}` is not supported"),
            ));
        }
    }
    engine.mark(registry, id, change);
    Ok(())
}

fn update_entry(engine: &mut Engine<'_>, registry: &str, id: &Id, value: Value) -> Result<()> {
    match registry {
        "nodes" => {
            let node: Node = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.nodes.insert(id.clone(), node);
        }
        "data_blocks" => {
            let block: DataBlock = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.data_blocks.insert(id.clone(), block);
        }
        "materials" => {
            let material: Material = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.materials.insert(id.clone(), material);
        }
        "collections" => {
            let collection: Collection = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.collections.insert(id.clone(), collection);
        }
        "actions" => {
            let action: Action = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.actions.insert(id.clone(), action);
        }
        "node_groups" => {
            let node_group: NodeGroup = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.node_groups.insert(id.clone(), node_group);
        }
        "worlds" => {
            let world: World = serde_json::from_value(value)
                .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.to_string()))?;
            engine.doc.worlds.insert(id.clone(), world);
        }
        "resources" => {
            engine.doc.resources.insert(id.clone(), value);
        }
        _ => {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!("library update for `{registry}` is not supported"),
            ));
        }
    }
    engine.mark(registry, id, ChangeKind::Updated);
    Ok(())
}

pub(crate) fn ensure_editable(engine: &Engine<'_>, registry: &str, id: &Id) -> Result<()> {
    let key = linked_key(registry, id);
    let marker = engine
        .doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
        .and_then(|linked| linked.get(&key));
    if let Some(marker) = marker {
        let lib = marker
            .get("library_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let mut error = engine.error(
            ErrorCode::InvalidOperation,
            format!("linked data `{id}` is read-only; create a library override to edit it"),
            &operation_pointer(engine.operation_index, "target"),
        );
        error.details =
            json!({"reason":"linked data is read-only","editable":false,"library_id":lib});
        return Err(error);
    }
    Ok(())
}

pub(crate) fn ensure_operation_editable(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<()> {
    fn scan(value: &Value, linked: &Map<String, Value>) -> Option<(String, Value)> {
        match value {
            Value::String(text) => linked
                .iter()
                .find(|(key, _)| key.rsplit_once(':').is_some_and(|(_, id)| id == text))
                .map(|(key, marker)| (key.clone(), marker.clone())),
            Value::Array(values) => values.iter().find_map(|value| scan(value, linked)),
            Value::Object(values) => values.values().find_map(|value| scan(value, linked)),
            _ => None,
        }
    }
    if matches!(
        operation.get("op").and_then(Value::as_str),
        Some(
            "library.link"
                | "library.append"
                | "library.override"
                | "library.register"
                | "library.reload"
                | "library.relocate"
        )
    ) {
        return Ok(());
    }
    let Some(linked) = engine
        .doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    if let Some((key, marker)) = scan(&Value::Object(operation.clone()), linked) {
        let library_id = marker
            .get("library_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let id = key.rsplit_once(':').map_or(key.as_str(), |(_, id)| id);
        let mut error = engine.error(
            ErrorCode::InvalidOperation,
            format!("linked data `{id}` is read-only; create a library override to edit it"),
            &operation_pointer(engine.operation_index, "target"),
        );
        error.details =
            json!({"reason":"linked data is read-only","editable":false,"library_id":library_id});
        return Err(error);
    }
    Ok(())
}

fn register_linked(
    engine: &mut Engine<'_>,
    registry: &str,
    id: &Id,
    library_id: &Id,
    source_id: &Id,
    uri: &str,
    library_name: &str,
) -> Result<()> {
    library::linked_ids_mut(engine.doc)?.insert(
        linked_key(registry, id),
        json!({
            "registry":registry,
            "library_id":library_id,
            "library_name":library_name,
            "source_id":source_id,
            "uri":uri,
            "editable":false
        }),
    );
    if registry == "nodes"
        && let Some(node) = engine.doc.nodes.get_mut(id)
    {
        node.properties
            .insert("library".to_owned(), Value::String(library_id.to_string()));
        node.properties.insert(
            "library_name".to_owned(),
            Value::String(library_name.to_owned()),
        );
        node.properties
            .insert("editable".to_owned(), Value::Bool(false));
        node.properties
            .insert("source".to_owned(), Value::String(source_id.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "fixed valid library IDs in pure invariant test"
    )]

    use super::*;

    #[test]
    fn namespace_never_exceeds_persistent_id_limit() {
        let id = Id::new("x".repeat(64)).unwrap();
        assert!(namespaced_id("library", &id).unwrap().as_str().len() <= 64);
    }
}
