use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    graph::GraphKind,
};

use super::{ChangeKind, Engine, check_fields, graph, operation_pointer, parse_id};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "compositor.enable" => enable(engine, operation),
        "compositor.node_add" => graph_operation(engine, operation, "graph.node_add"),
        "compositor.node_update" => graph_operation(engine, operation, "graph.node_update"),
        "compositor.node_remove" => graph_operation(engine, operation, "graph.node_remove"),
        "compositor.link" => graph_operation(engine, operation, "graph.link"),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid compositor operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn enable(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let target = operation
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "compositor target must be an object",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    check_fields(engine, target, &["id"], &["id"])?;
    let text = target.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "compositor target requires a scene ID",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let scene_id = parse_id(
        engine,
        text,
        &operation_pointer(engine.operation_index, "target/id"),
    )?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "compositor set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    check_fields(engine, set, &["enabled", "graph"], &["enabled"])?;
    let enabled = set.get("enabled").and_then(Value::as_bool).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "enabled must be a boolean",
            &operation_pointer(engine.operation_index, "set/enabled"),
        )
    })?;
    let graph = set
        .get("graph")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "graph must be a compositor graph ID or null",
                    &operation_pointer(engine.operation_index, "set/graph"),
                )
            })?;
            let id = parse_id(
                engine,
                text,
                &operation_pointer(engine.operation_index, "set/graph"),
            )?;
            let group = engine.doc.node_groups.get(&id).ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!("compositor graph `{id}` was not found"),
                    &operation_pointer(engine.operation_index, "set/graph"),
                )
            })?;
            if group.kind != GraphKind::Compositor {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "graph must have compositor kind",
                    &operation_pointer(engine.operation_index, "set/graph"),
                ));
            }
            Ok(Some(id))
        })
        .transpose()?;
    let missing_scene = engine.error(
        ErrorCode::TargetNotFound,
        format!("scene `{scene_id}` was not found"),
        &operation_pointer(engine.operation_index, "target/id"),
    );
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
    let changed = scene.use_compositing != enabled
        || graph
            .as_ref()
            .is_some_and(|value| scene.compositor != *value);
    if changed {
        scene.use_compositing = enabled;
        if let Some(graph) = graph {
            scene.compositor = graph;
        }
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn graph_operation(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    graph_operation: &str,
) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "graph",
            "id",
            "type",
            "name",
            "location",
            "properties",
            "inputs",
            "node_id",
            "from_node",
            "from_socket",
            "to_node",
            "to_socket",
        ],
        &["graph"],
    )?;
    let graph_text = operation
        .get("graph")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "compositor graph ID must be a string",
                &operation_pointer(engine.operation_index, "graph"),
            )
        })?;
    let graph_id = parse_id(
        engine,
        graph_text,
        &operation_pointer(engine.operation_index, "graph"),
    )?;
    let group = engine.doc.node_groups.get(&graph_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("compositor graph `{graph_id}` was not found"),
            &operation_pointer(engine.operation_index, "graph"),
        )
    })?;
    if group.kind != GraphKind::Compositor {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "compositor operation requires a compositor graph",
            &operation_pointer(engine.operation_index, "graph"),
        ));
    }
    let mut mapped = operation.clone();
    mapped.insert("op".to_owned(), Value::String(graph_operation.to_owned()));
    graph::apply(engine, graph_operation, &mapped)
}
