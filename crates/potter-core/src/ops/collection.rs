use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{Collection, Id, Node, SceneDoc, Transform},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, pointer_escape, read_bool, read_id,
    read_string, resolve_collection_targets,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "collection.create" => create(engine, operation),
        "collection.instance_create" => instance_create(engine, operation),
        "collection.update" => update(engine, operation),
        "collection.delete" => delete(engine, operation),
        "collection.link" => link(engine, operation, true),
        "collection.parent" => parent(engine, operation),
        "collection.unlink" => link(engine, operation, false),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid collection operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "name", "parent"], &["id"])?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.collections.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("collection ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    let parent = if operation.contains_key("parent") {
        Some(read_id(engine, operation, "parent")?)
    } else {
        let scene = engine
            .doc
            .scenes
            .get(&engine.doc.active_scene)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "active scene is missing",
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?;
        Some(scene.root_collection.clone())
    };
    if let Some(parent) = &parent
        && !engine.doc.collections.contains_key(parent)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("parent collection `{parent}` was not found"),
            &operation_pointer(
                engine.operation_index,
                if operation.contains_key("parent") {
                    "parent"
                } else {
                    "id"
                },
            ),
        ));
    }
    engine.doc.collections.insert(
        id.clone(),
        Collection {
            name,
            children: Vec::new(),
            objects: Vec::new(),
        },
    );
    engine.mark("collections", &id, ChangeKind::Created);
    if let Some(parent) = parent
        && let Some(parent_collection) = engine.doc.collections.get_mut(&parent)
        && !parent_collection.children.contains(&id)
    {
        parent_collection.children.push(id.clone());
        parent_collection.children.sort();
        engine.mark("collections", &parent, ChangeKind::Updated);
    }
    Ok(true)
}

fn instance_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "collection", "name", "transform"],
        &["id", "collection"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.nodes.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("node ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let target = read_id(engine, operation, "collection")?;
    if !engine.doc.collections.contains_key(&target) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("collection `{target}` was not found"),
            &operation_pointer(engine.operation_index, "collection"),
        ));
    }
    let scene = engine
        .doc
        .scenes
        .get(&engine.doc.active_scene)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "active scene is missing",
                &operation_pointer(engine.operation_index, "id"),
            )
        })?;
    let root = scene.root_collection.clone();
    if !engine.doc.collections.contains_key(&root) {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "scene root collection is missing",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    if collection_reaches(engine.doc, &target, &root) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "collection instance would create a cycle",
            &operation_pointer(engine.operation_index, "collection"),
        ));
    }
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    let transform = if let Some(value) = operation.get("transform") {
        let parsed = serde_json::from_value::<Transform>(value.clone()).map_err(|_| {
            engine.error(
                ErrorCode::InvalidOperation,
                "transform must be a valid transform object",
                &operation_pointer(engine.operation_index, "transform"),
            )
        })?;
        if !parsed.translation.iter().all(|value| value.is_finite())
            || !parsed.rotation.iter().all(|value| value.is_finite())
            || !parsed.scale.iter().all(|value| value.is_finite())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "transform values must be finite",
                &operation_pointer(engine.operation_index, "transform"),
            ));
        }
        parsed
    } else {
        Transform::default()
    };
    let mut node = Node {
        name,
        kind: "collection_instance".to_owned(),
        transform,
        ..Node::default()
    };
    node.properties.insert(
        "instance_collection".to_owned(),
        Value::String(target.to_string()),
    );
    engine.doc.nodes.insert(id.clone(), node);
    engine.mark("nodes", &id, ChangeKind::Created);
    if let Some(collection) = engine.doc.collections.get_mut(&root)
        && !collection.objects.contains(&id)
    {
        collection.objects.push(id.clone());
        collection.objects.sort();
        engine.mark("collections", &root, ChangeKind::Updated);
    }
    Ok(true)
}

fn collection_reaches(doc: &SceneDoc, start: &Id, target: &Id) -> bool {
    let mut pending = vec![start.clone()];
    let mut visited = BTreeSet::new();
    while let Some(current) = pending.pop() {
        if current == *target {
            return true;
        }
        if !visited.insert(current.clone()) {
            continue;
        }
        if let Some(collection) = doc.collections.get(&current) {
            pending.extend(collection.children.iter().cloned());
            for object_id in &collection.objects {
                if let Some(instance_target) = doc
                    .nodes
                    .get(object_id)
                    .filter(|node| node.kind == "collection_instance")
                    .and_then(|node| node.properties.get("instance_collection"))
                    .and_then(Value::as_str)
                    .and_then(|value| value.parse::<Id>().ok())
                {
                    pending.push(instance_target);
                }
            }
        }
    }
    false
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set", "scene_id", "view_layer"],
        &["target", "set"],
    )?;
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let targets = resolve_collection_targets(engine, target, true)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    for key in set.keys() {
        if !matches!(key.as_str(), "name" | "exclude") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown collection set field `{key}`"),
                &format!(
                    "{}/{}",
                    operation_pointer(engine.operation_index, "set"),
                    pointer_escape(key)
                ),
            ));
        }
    }
    let name = match set.get("name") {
        Some(Value::String(value)) => Some(value),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "name must be a string",
                &operation_pointer(engine.operation_index, "set/name"),
            ));
        }
        None => None,
    };
    let exclude = match set.get("exclude") {
        Some(Value::Bool(value)) => Some(*value),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "exclude must be a boolean",
                &operation_pointer(engine.operation_index, "set/exclude"),
            ));
        }
        None => None,
    };
    if exclude.is_none()
        && (operation.contains_key("view_layer") || operation.contains_key("scene_id"))
    {
        let field = if operation.contains_key("view_layer") {
            "view_layer"
        } else {
            "scene_id"
        };
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} is only valid when updating exclude"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    let mut changed = false;
    if let Some(name) = name {
        for id in &targets {
            if let Some(collection) = engine.doc.collections.get_mut(id)
                && collection.name != *name
            {
                name.clone_into(&mut collection.name);
                engine.mark("collections", id, ChangeKind::Updated);
                changed = true;
            }
        }
    }
    if let Some(exclude) = exclude {
        let scene_id = if operation.contains_key("scene_id") {
            read_id(engine, operation, "scene_id")?
        } else {
            engine.doc.active_scene.clone()
        };
        let scene_pointer = if operation.contains_key("scene_id") {
            "scene_id"
        } else {
            "view_layer"
        };
        let scene = engine.doc.scenes.get(&scene_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("scene `{scene_id}` was not found"),
                &operation_pointer(engine.operation_index, scene_pointer),
            )
        })?;
        let selected_view_layer = if operation.contains_key("view_layer") {
            read_id(engine, operation, "view_layer")?
        } else {
            scene.view_layers.keys().next().cloned().ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    "scene has no view layers",
                    &operation_pointer(engine.operation_index, "view_layer"),
                )
            })?
        };
        let missing_scene = engine.error(
            ErrorCode::TargetNotFound,
            format!("scene `{scene_id}` was not found"),
            &operation_pointer(engine.operation_index, scene_pointer),
        );
        let missing_view_layer = engine.error(
            ErrorCode::TargetNotFound,
            format!("view layer `{selected_view_layer}` was not found"),
            &operation_pointer(engine.operation_index, "view_layer"),
        );
        let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
        let view_layer = scene
            .view_layers
            .get_mut(&selected_view_layer)
            .ok_or(missing_view_layer)?;
        let mut exclusion_changed = false;
        for id in &targets {
            let exists = view_layer.excluded_collections.contains(id);
            if exclude && !exists {
                view_layer.excluded_collections.push(id.clone());
                view_layer.excluded_collections.sort();
                exclusion_changed = true;
            } else if !exclude && exists {
                view_layer.excluded_collections.retain(|item| item != id);
                exclusion_changed = true;
            }
        }
        if exclusion_changed {
            engine.mark("scenes", &scene_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn parent(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "parent"],
        &["target", "parent"],
    )?;
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut targets = resolve_collection_targets(engine, target, false)?;
    let Some(target_id) = targets.pop() else {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "target collection was not found",
            &operation_pointer(engine.operation_index, "target"),
        ));
    };
    let parent_id = match operation.get("parent") {
        Some(Value::Null) => None,
        Some(Value::String(_)) => Some(read_id(engine, operation, "parent")?),
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "parent must be a collection ID or null",
                &operation_pointer(engine.operation_index, "parent"),
            ));
        }
    };
    if let Some(parent_id) = &parent_id {
        if !engine.doc.collections.contains_key(parent_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("parent collection `{parent_id}` was not found"),
                &operation_pointer(engine.operation_index, "parent"),
            ));
        }
        if collection_reaches(engine.doc, &target_id, parent_id) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "collection parent would create a cycle",
                &operation_pointer(engine.operation_index, "parent"),
            ));
        }
    }
    let parents = engine
        .doc
        .collections
        .iter()
        .filter(|(id, collection)| {
            collection.children.contains(&target_id) && parent_id.as_ref() != Some(*id)
        })
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    let new_parent_needs_link = parent_id.as_ref().is_some_and(|parent_id| {
        engine
            .doc
            .collections
            .get(parent_id)
            .is_some_and(|collection| !collection.children.contains(&target_id))
    });
    if parents.is_empty() && !new_parent_needs_link {
        return Ok(false);
    }
    for id in parents {
        if let Some(collection) = engine.doc.collections.get_mut(&id) {
            collection.children.retain(|child| child != &target_id);
            engine.mark("collections", &id, ChangeKind::Updated);
        }
    }
    if new_parent_needs_link
        && let Some(parent_id) = parent_id
        && let Some(collection) = engine.doc.collections.get_mut(&parent_id)
    {
        collection.children.push(target_id);
        collection.children.sort();
        engine.mark("collections", &parent_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "unlink"], &["target"])?;
    let targets = resolve_collection_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?,
        true,
    )?;
    let unlink = read_bool(engine, operation, "unlink", false)?;
    for id in &targets {
        if engine
            .doc
            .scenes
            .values()
            .any(|scene| scene.root_collection == *id)
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("collection `{id}` is a scene root and cannot be deleted"),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        if let Some(collection) = engine.doc.collections.get(id)
            && (!collection.children.is_empty() || !collection.objects.is_empty())
            && !unlink
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("collection `{id}` is not empty; set unlink:true to detach its contents"),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
    }
    for id in targets {
        let mut changed_parents = BTreeSet::new();
        for (parent_id, collection) in &mut engine.doc.collections {
            if collection.children.contains(&id) {
                collection.children.retain(|child| child != &id);
                changed_parents.insert(parent_id.clone());
            }
        }
        for parent_id in changed_parents {
            engine.mark("collections", &parent_id, ChangeKind::Updated);
        }
        let mut changed_scenes = BTreeSet::new();
        for (scene_id, scene) in &mut engine.doc.scenes {
            for view_layer in scene.view_layers.values_mut() {
                let old_len = view_layer.excluded_collections.len();
                view_layer
                    .excluded_collections
                    .retain(|collection| collection != &id);
                if view_layer.excluded_collections.len() != old_len {
                    changed_scenes.insert(scene_id.clone());
                }
            }
        }
        for scene_id in changed_scenes {
            engine.mark("scenes", &scene_id, ChangeKind::Updated);
        }
        if engine.doc.collections.remove(&id).is_some() {
            engine.mark("collections", &id, ChangeKind::Deleted);
        }
    }
    Ok(true)
}

fn link(engine: &mut Engine<'_>, operation: &Map<String, Value>, add: bool) -> Result<bool> {
    let allowed = ["op", "collection", "object", "child"];
    check_fields(engine, operation, &allowed, &["collection"])?;
    let collection_id = read_id(engine, operation, "collection")?;
    if !engine.doc.collections.contains_key(&collection_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("collection `{collection_id}` was not found"),
            &operation_pointer(engine.operation_index, "collection"),
        ));
    }
    let has_object = operation.contains_key("object");
    let has_child = operation.contains_key("child");
    if has_object == has_child {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "collection link requires exactly one of object or child",
            &operation_pointer(engine.operation_index, "object"),
        ));
    }
    if has_object {
        let object_id = read_id(engine, operation, "object")?;
        if !engine.doc.nodes.contains_key(&object_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{object_id}` was not found"),
                &operation_pointer(engine.operation_index, "object"),
            ));
        }
        if add
            && !engine
                .doc
                .collections
                .get(&collection_id)
                .is_some_and(|collection| collection.objects.contains(&object_id))
            && let Some(instance_target) = engine
                .doc
                .nodes
                .get(&object_id)
                .filter(|node| node.kind == "collection_instance")
                .and_then(|node| node.properties.get("instance_collection"))
                .and_then(Value::as_str)
                .and_then(|value| value.parse::<Id>().ok())
            && collection_reaches(engine.doc, &instance_target, &collection_id)
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "collection link would create an instance cycle",
                &operation_pointer(engine.operation_index, "object"),
            ));
        }
        let changed = if let Some(collection) = engine.doc.collections.get_mut(&collection_id) {
            let exists = collection.objects.contains(&object_id);
            if add && !exists {
                collection.objects.push(object_id);
                collection.objects.sort();
                true
            } else if !add && exists {
                collection.objects.retain(|item| item != &object_id);
                true
            } else {
                false
            }
        } else {
            false
        };
        if changed {
            engine.mark("collections", &collection_id, ChangeKind::Updated);
        }
        return Ok(changed);
    }
    let child_id = read_id(engine, operation, "child")?;
    if !engine.doc.collections.contains_key(&child_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("child collection `{child_id}` was not found"),
            &operation_pointer(engine.operation_index, "child"),
        ));
    }
    if add && collection_reaches(engine.doc, &child_id, &collection_id) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "collection link would create a cycle",
            &operation_pointer(engine.operation_index, "child"),
        ));
    }
    let changed = if let Some(collection) = engine.doc.collections.get_mut(&collection_id) {
        let exists = collection.children.contains(&child_id);
        if add && !exists {
            collection.children.push(child_id);
            collection.children.sort();
            true
        } else if !add && exists {
            collection.children.retain(|item| item != &child_id);
            true
        } else {
            false
        }
    } else {
        false
    };
    if changed {
        engine.mark("collections", &collection_id, ChangeKind::Updated);
    }
    Ok(changed)
}
