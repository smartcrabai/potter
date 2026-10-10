use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom,
    model::{CurveData, CurveFillMode, CurvePoint, DataBlock, Id, SurfaceData, TextObjectData},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_id, read_string,
    resolve_node_targets,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "curve.create" => create_curve(engine, operation),
        "curve.update" => update_curve(engine, operation),
        "curve.point_add" => add_curve_point(engine, operation),
        "curve.point_update" => update_curve_point(engine, operation),
        "curve.point_delete" => delete_curve_point(engine, operation),
        "curve.convert" => convert_curve(engine, operation),
        "surface.create" => create_surface(engine, operation),
        "text_object.create" => create_text_object(engine, operation),
        "text_object.update" => update_text_object(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid curve/surface/text operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "splines",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "fill_mode",
            "collection",
            "transform",
            "parent",
            "parent_inverse",
            "materials",
            "visible",
            "render_visible",
            "selectable",
        ],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_curve")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let curve = decode_selected::<CurveData>(
        engine,
        operation,
        &[
            "splines",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "fill_mode",
        ],
        "",
    )?;
    validate_curve(engine, &curve, "")?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "curve".to_owned(),
            mesh: None,
            curve: Some(curve),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "curve", &id, &data_id)
}

fn update_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "splines",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "fill_mode",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated: Value = serialize(&curve)?;
    merge_values(&mut updated, set);
    let updated: CurveData = decode_value(engine, updated, "set")?;
    validate_curve(engine, &updated, "set")?;
    if updated == curve {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    block.curve = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn add_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "index", "point"],
        &["target", "spline_index", "point"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point = decode_value::<CurvePoint>(
        engine,
        operation.get("point").cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing required field `point`",
                &operation_pointer(engine.operation_index, "point"),
            )
        })?,
        "point",
    )?;
    let index = operation
        .get("index")
        .map(|_| read_index(engine, operation, "index"))
        .transpose()?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let spline = updated.splines.get_mut(spline_index).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "spline_index is out of range",
            &operation_pointer(engine.operation_index, "spline_index"),
        )
    })?;
    let insertion_index = index.unwrap_or(spline.points.len());
    if insertion_index > spline.points.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "index is out of range",
            &operation_pointer(engine.operation_index, "index"),
        ));
    }
    spline.points.insert(insertion_index, point);
    validate_curve(engine, &updated, "point")?;
    store_curve(engine, &data_id, updated)
}

fn update_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "point_index", "set"],
        &["target", "spline_index", "point_index", "set"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point_index = read_index(engine, operation, "point_index")?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "co",
            "handle_left",
            "handle_right",
            "handle_type",
            "weight",
            "radius",
            "tilt",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let point = updated
        .splines
        .get_mut(spline_index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "spline_index is out of range",
                &operation_pointer(engine.operation_index, "spline_index"),
            )
        })?
        .points
        .get_mut(point_index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "point_index is out of range",
                &operation_pointer(engine.operation_index, "point_index"),
            )
        })?;
    let mut point_value = serialize(point)?;
    merge_values(&mut point_value, set);
    let updated_point: CurvePoint = decode_value(engine, point_value, "set")?;
    let changed = *point != updated_point;
    if !changed {
        return Ok(false);
    }
    *point = updated_point;
    validate_curve(engine, &updated, "set")?;
    store_curve(engine, &data_id, updated)
}

fn delete_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "point_index"],
        &["target", "spline_index", "point_index"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point_index = read_index(engine, operation, "point_index")?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let spline = updated.splines.get_mut(spline_index).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "spline_index is out of range",
            &operation_pointer(engine.operation_index, "spline_index"),
        )
    })?;
    if point_index >= spline.points.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "point_index is out of range",
            &operation_pointer(engine.operation_index, "point_index"),
        ));
    }
    spline.points.remove(point_index);
    validate_curve(engine, &updated, "target")?;
    store_curve(engine, &data_id, updated)
}

fn convert_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "to"],
        &["target", "to"],
    )?;
    let destination = read_string(engine, operation, "to")?;
    if destination != "mesh" {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("curve conversion to `{destination}` is not supported"),
            json!({
                "operation_index": engine.operation_index,
                "pointer": operation_pointer(engine.operation_index, "to"),
                "feature_id": "curve.convert.destination",
                "destination": destination,
            }),
        ));
    }
    let (node_id, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.as_ref())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    reject_taper_object(engine, curve, "target")?;
    reject_multiple_fill_contours(engine, curve, "target")?;
    let mesh = geom::curve::evaluate_curve(curve).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("curve geometry is invalid: {error}"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    *block = DataBlock {
        data_type: "mesh".to_owned(),
        mesh: Some(mesh),
        ..DataBlock::default()
    };
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "curve node disappeared"))?;
    "mesh".clone_into(&mut node.kind);
    node.primitive = None;
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    record_curve_loss(engine, &data_id)?;
    Ok(true)
}

fn create_surface(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "points",
            "order_u",
            "order_v",
            "resolution",
            "cyclic_u",
            "cyclic_v",
            "use_endpoint_u",
            "use_endpoint_v",
            "collection",
            "transform",
            "parent",
            "parent_inverse",
            "materials",
            "visible",
            "render_visible",
            "selectable",
        ],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_surface")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let surface = decode_selected::<SurfaceData>(
        engine,
        operation,
        &[
            "points",
            "order_u",
            "order_v",
            "resolution",
            "cyclic_u",
            "cyclic_v",
            "use_endpoint_u",
            "use_endpoint_v",
        ],
        "",
    )?;
    geom::curve::evaluate_surface(&surface).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("surface geometry is invalid: {error}"),
            &operation_pointer(engine.operation_index, "points"),
        )
    })?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "surface".to_owned(),
            mesh: None,
            surface: Some(surface),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "surface", &id, &data_id)
}

fn create_text_object(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "body",
            "font",
            "size",
            "align_x",
            "align_y",
            "extrude",
            "bevel_depth",
            "collection",
            "transform",
            "parent",
            "parent_inverse",
            "materials",
            "visible",
            "render_visible",
            "selectable",
        ],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_text")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let text = decode_selected::<TextObjectData>(
        engine,
        operation,
        &[
            "body",
            "font",
            "font_name",
            "size",
            "align_x",
            "align_y",
            "extrude",
            "bevel_depth",
            "character_spacing",
            "word_spacing",
            "line_spacing",
            "shear",
            "offset_x",
            "offset_y",
            "small_caps_scale",
        ],
        "",
    )?;
    geom::text::evaluate_text(&text).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("text geometry is invalid: {error}"),
            &operation_pointer(engine.operation_index, "body"),
        )
    })?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "text".to_owned(),
            mesh: None,
            text: Some(text),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "text", &id, &data_id)
}

fn update_text_object(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "body",
            "font",
            "font_name",
            "size",
            "align_x",
            "align_y",
            "extrude",
            "bevel_depth",
            "character_spacing",
            "word_spacing",
            "line_spacing",
            "shear",
            "offset_x",
            "offset_y",
            "small_caps_scale",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "text", "text")?;
    let text = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.text.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "text", "target"))?;
    let mut updated: Value = serialize(&text)?;
    merge_values(&mut updated, set);
    let updated: TextObjectData = decode_value(engine, updated, "set")?;
    geom::text::evaluate_text(&updated).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("text geometry is invalid: {error}"),
            &operation_pointer(engine.operation_index, "set"),
        )
    })?;
    if updated == text {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "text data block disappeared")
        })?;
    block.text = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn create_node(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    id: &Id,
    data_id: &Id,
) -> Result<bool> {
    let mut node_operation = Map::new();
    node_operation.insert("op".to_owned(), Value::String("node.create".to_owned()));
    node_operation.insert("kind".to_owned(), Value::String(kind.to_owned()));
    node_operation.insert("id".to_owned(), Value::String(id.to_string()));
    node_operation.insert("data".to_owned(), Value::String(data_id.to_string()));
    for field in [
        "name",
        "collection",
        "transform",
        "parent",
        "parent_inverse",
        "materials",
        "visible",
        "render_visible",
        "selectable",
    ] {
        if let Some(value) = operation.get(field) {
            node_operation.insert(field.to_owned(), value.clone());
        }
    }
    super::node::apply(engine, "node.create", &node_operation)
}

fn create_data_id(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    node_id: &Id,
    suffix: &str,
) -> Result<Id> {
    if operation.contains_key("data_id") {
        let value = operation
            .get("data_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "data_id must be a string",
                    &operation_pointer(engine.operation_index, "data_id"),
                )
            })?;
        parse_id(
            engine,
            value,
            &operation_pointer(engine.operation_index, "data_id"),
        )
    } else {
        let generated = format!("{node_id}{suffix}");
        parse_id(
            engine,
            &generated,
            &operation_pointer(engine.operation_index, "id"),
        )
    }
}

fn ensure_data_id_available(engine: &Engine<'_>, data_id: &Id, field: &str) -> Result<()> {
    if engine.doc.data_blocks.contains_key(data_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("data-block ID `{data_id}` already exists"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(())
}

fn insert_data_block(engine: &mut Engine<'_>, id: &Id, data: DataBlock) {
    engine.doc.data_blocks.insert(id.clone(), data);
    engine.mark("data_blocks", id, ChangeKind::Created);
}

fn typed_node_data_id(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    data_type: &str,
) -> Result<(Id, Id)> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut targets = resolve_node_targets(engine, target, false)?;
    let node_id = targets.pop().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let node = engine.doc.nodes.get(&node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if node.kind != kind {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("target node must be a {kind}"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let data_id = node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node has no data block"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node data block `{data_id}` is missing"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if block.data_type != data_type {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node must reference a {data_type} Data-Block"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok((node_id, data_id))
}

fn store_curve(engine: &mut Engine<'_>, data_id: &Id, curve: CurveData) -> Result<bool> {
    let block =
        engine.doc.data_blocks.get_mut(data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    if block.curve.as_ref() == Some(&curve) {
        return Ok(false);
    }
    block.curve = Some(curve);
    engine.mark("data_blocks", data_id, ChangeKind::Updated);
    Ok(true)
}

fn record_curve_loss(engine: &mut Engine<'_>, data_id: &Id) -> Result<()> {
    let existed = engine.doc.compatibility.contains_key("losses");
    let slot = engine
        .doc
        .compatibility
        .entry("losses".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let losses = slot.as_array_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "compatibility losses must be an array",
        )
    })?;
    losses.push(json!({
        "feature_id": "curve.production_data",
        "data_id": data_id,
        "reason": "Converting a curve to a mesh discards its editable spline and curve settings.",
        "suggestion": "Keep the curve object when its parametric production data must remain editable.",
    }));
    let compatibility_id = Id::new("losses")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?;
    engine.mark(
        "compatibility",
        &compatibility_id,
        if existed {
            ChangeKind::Updated
        } else {
            ChangeKind::Created
        },
    );
    Ok(())
}

fn read_set<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
    operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })
}

fn check_curve_set_fields(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    super::check_set_fields_by(engine, values, allowed, |engine, key| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown field `{key}`"),
            &operation_pointer(engine.operation_index, &format!("{prefix}/{key}")),
        )
    })
}

fn read_index(engine: &Engine<'_>, operation: &Map<String, Value>, field: &str) -> Result<usize> {
    let pointer = operation_pointer(engine.operation_index, field);
    operation
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a non-negative integer"),
                &pointer,
            )
        })
}

fn decode_selected<T: DeserializeOwned>(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    fields: &[&str],
    pointer: &str,
) -> Result<T> {
    let mut values = Map::new();
    for field in fields {
        if let Some(value) = operation.get(*field) {
            values.insert((*field).to_owned(), value.clone());
        }
    }
    decode_value(engine, Value::Object(values), pointer)
}

fn decode_value<T: DeserializeOwned>(
    engine: &Engine<'_>,
    value: Value,
    pointer: &str,
) -> Result<T> {
    serde_json::from_value(value).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid curve, surface, or text data: {error}"),
            &operation_pointer(engine.operation_index, pointer),
        )
    })
}

fn serialize<T: serde::Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
}

fn merge_values(destination: &mut Value, source: &Map<String, Value>) {
    if let Some(destination) = destination.as_object_mut() {
        for (key, value) in source {
            destination.insert(key.clone(), value.clone());
        }
    }
}

fn reject_taper_object(engine: &Engine<'_>, curve: &CurveData, pointer: &str) -> Result<()> {
    let Some(taper) = curve.taper.as_ref() else {
        return Ok(());
    };
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "curve taper-object evaluation is not supported",
        json!({
            "operation_index": engine.operation_index,
            "pointer": operation_pointer(engine.operation_index, pointer),
            "feature_id": "curve.taper_object",
            "taper": taper,
        }),
    ))
}

fn reject_multiple_fill_contours(
    engine: &Engine<'_>,
    curve: &CurveData,
    pointer: &str,
) -> Result<()> {
    if curve.fill_mode == CurveFillMode::None || curve.splines.len() <= 1 {
        return Ok(());
    }
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "multiple 2D fill contours are not supported",
        json!({
            "operation_index": engine.operation_index,
            "pointer": operation_pointer(engine.operation_index, pointer),
            "feature_id": "curve.fill.multiple_contours",
        }),
    ))
}

fn validate_curve(engine: &Engine<'_>, curve: &CurveData, pointer: &str) -> Result<()> {
    reject_taper_object(engine, curve, pointer)?;
    reject_multiple_fill_contours(engine, curve, pointer)?;
    geom::curve::evaluate_curve(curve)
        .map(|_| ())
        .map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("curve geometry is invalid: {error}"),
                &operation_pointer(engine.operation_index, pointer),
            )
        })
}
