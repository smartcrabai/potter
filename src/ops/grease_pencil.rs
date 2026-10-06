use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{
        GreasePencilData, GreasePencilFrame, GreasePencilLayer, GreasePencilPoint,
        GreasePencilStroke, Id,
    },
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_bool, read_id, read_string,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "grease_pencil.layer_create" => layer_create(engine, operation),
        "grease_pencil.layer_update" => layer_update(engine, operation),
        "grease_pencil.layer_delete" => layer_delete(engine, operation),
        "grease_pencil.frame_add" => frame_add(engine, operation),
        "grease_pencil.stroke_add" => stroke_add(engine, operation),
        "grease_pencil.stroke_update" => stroke_update(engine, operation),
        "grease_pencil.stroke_delete" => stroke_delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported grease pencil operation `{name}`"),
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn layer_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "name", "opacity", "visible"],
        &["target", "id", "name"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let name = read_string(engine, operation, "name")?;
    let opacity = optional_opacity(engine, operation.get("opacity"), "opacity", 1.0)?;
    let visible = read_bool(engine, operation, "visible", true)?;
    let data = gp_data(engine, &data_id)?;
    if data.layers.iter().any(|layer| layer.id == id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("grease pencil layer `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let layer = GreasePencilLayer {
        id,
        name,
        opacity,
        visible,
        frames: Vec::new(),
    };
    gp_data_mut(engine, &data_id)?.layers.push(layer);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn layer_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "layer", "set"],
        &["target", "layer", "set"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let set = set_object(engine, operation)?;
    check_grease_pencil_set_fields(engine, set, &["name", "opacity", "visible"])?;
    let index = layer_index(engine, &data_id, &layer_id)?;
    let name = set
        .get("name")
        .map(|_| read_string_at(engine, set, "name", "set/name"))
        .transpose()?;
    let opacity = set
        .get("opacity")
        .map(|value| optional_opacity(engine, Some(value), "set/opacity", 1.0))
        .transpose()?;
    let visible = set
        .get("visible")
        .map(|value| read_bool_at(engine, value, "set/visible"))
        .transpose()?;
    let layer = &mut gp_data_mut(engine, &data_id)?.layers[index];
    let changed = name.as_ref().is_some_and(|value| value != &layer.name)
        || opacity.is_some_and(|value| !crate::float::equal_f64(value, layer.opacity))
        || visible.is_some_and(|value| value != layer.visible);
    if !changed {
        return Ok(false);
    }
    if let Some(name) = name {
        layer.name = name;
    }
    if let Some(opacity) = opacity {
        layer.opacity = opacity;
    }
    if let Some(visible) = visible {
        layer.visible = visible;
    }
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn layer_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "layer"],
        &["target", "layer"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let index = layer_index(engine, &data_id, &layer_id)?;
    gp_data_mut(engine, &data_id)?.layers.remove(index);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn frame_add(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "layer", "frame"],
        &["target", "layer", "frame"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let frame = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let layer_index = layer_index(engine, &data_id, &layer_id)?;
    let data = gp_data(engine, &data_id)?;
    let layer = &data.layers[layer_index];
    let insert_at = layer
        .frames
        .partition_point(|candidate| candidate.frame < frame);
    if layer
        .frames
        .get(insert_at)
        .is_some_and(|candidate| crate::float::equal_f64(candidate.frame, frame))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("grease pencil frame {frame} already exists on layer `{layer_id}`"),
            &operation_pointer(engine.operation_index, "frame"),
        ));
    }
    gp_data_mut(engine, &data_id)?.layers[layer_index]
        .frames
        .insert(
            insert_at,
            GreasePencilFrame {
                frame,
                strokes: Vec::new(),
            },
        );
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn stroke_add(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "target", "layer", "frame", "id", "points", "material", "cyclic", "fill",
        ],
        &["target", "layer", "frame", "id", "points"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let frame_number = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let id = read_id(engine, operation, "id")?;
    let points = parse_points(engine, operation.get("points"), "points")?;
    let material = parse_material(engine, operation.get("material"), "material")?;
    let cyclic = read_bool(engine, operation, "cyclic", false)?;
    let fill = parse_fill(engine, operation.get("fill"), "fill")?;
    let layer_index = layer_index(engine, &data_id, &layer_id)?;
    let frame_index = frame_index(engine, &data_id, layer_index, frame_number)?;
    let frame = &gp_data(engine, &data_id)?.layers[layer_index].frames[frame_index];
    if frame.strokes.iter().any(|stroke| stroke.id == id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("grease pencil stroke `{id}` already exists in frame {frame_number}"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    gp_data_mut(engine, &data_id)?.layers[layer_index].frames[frame_index]
        .strokes
        .push(GreasePencilStroke {
            id,
            points,
            material,
            cyclic,
            fill,
        });
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn stroke_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "layer", "frame", "stroke", "set"],
        &["target", "layer", "frame", "stroke", "set"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let frame_number = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let stroke_id = read_id(engine, operation, "stroke")?;
    let set = set_object(engine, operation)?;
    check_grease_pencil_set_fields(engine, set, &["points", "material", "cyclic", "fill"])?;
    let points = set
        .get("points")
        .map(|value| parse_points(engine, Some(value), "set/points"))
        .transpose()?;
    let material = set
        .get("material")
        .map(|value| parse_material(engine, Some(value), "set/material"))
        .transpose()?;
    let cyclic = set
        .get("cyclic")
        .map(|value| read_bool_at(engine, value, "set/cyclic"))
        .transpose()?;
    let fill = set
        .get("fill")
        .map(|value| parse_fill(engine, Some(value), "set/fill"))
        .transpose()?;
    let layer_index = layer_index(engine, &data_id, &layer_id)?;
    let frame_index = frame_index(engine, &data_id, layer_index, frame_number)?;
    let stroke_index = stroke_index(engine, &data_id, layer_index, frame_index, &stroke_id)?;
    let stroke = &mut gp_data_mut(engine, &data_id)?.layers[layer_index].frames[frame_index]
        .strokes[stroke_index];
    let changed = points.as_ref().is_some_and(|value| value != &stroke.points)
        || material
            .as_ref()
            .is_some_and(|value| value != &stroke.material)
        || cyclic.is_some_and(|value| value != stroke.cyclic)
        || fill.as_ref().is_some_and(|value| value != &stroke.fill);
    if !changed {
        return Ok(false);
    }
    if let Some(points) = points {
        stroke.points = points;
    }
    if let Some(material) = material {
        stroke.material = material;
    }
    if let Some(cyclic) = cyclic {
        stroke.cyclic = cyclic;
    }
    if let Some(fill) = fill {
        stroke.fill = fill;
    }
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn stroke_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "layer", "frame", "stroke"],
        &["target", "layer", "frame", "stroke"],
    )?;
    let data_id = target_data_id(engine, operation)?;
    let layer_id = read_id(engine, operation, "layer")?;
    let frame_number = super::finite_number(
        engine,
        operation.get("frame"),
        "frame",
        super::FiniteNumberMessage::Field,
    )?;
    let stroke_id = read_id(engine, operation, "stroke")?;
    let layer_index = layer_index(engine, &data_id, &layer_id)?;
    let frame_index = frame_index(engine, &data_id, layer_index, frame_number)?;
    let stroke_index = stroke_index(engine, &data_id, layer_index, frame_index, &stroke_id)?;
    gp_data_mut(engine, &data_id)?.layers[layer_index].frames[frame_index]
        .strokes
        .remove(stroke_index);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn target_data_id(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let node_id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        super::TargetIdPolicy::Strict {
            object_message: "target must be an object containing id",
            shape_message: "target must contain only id",
            id_message: "target.id must be a string",
            require_id: true,
        },
    )?;
    let node = engine.doc.nodes.get(&node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    if node.kind != "grease_pencil" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node must have kind grease_pencil",
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    let data_id = node.data.as_ref().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "grease pencil node must reference grease pencil data",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let data = engine.doc.data_blocks.get(data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("data block `{data_id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    if data.data_type != "grease_pencil" || data.grease_pencil.is_none() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target node must reference typed grease pencil data",
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok(data_id.clone())
}

fn gp_data<'a>(engine: &'a Engine<'_>, data_id: &Id) -> Result<&'a GreasePencilData> {
    engine
        .doc
        .data_blocks
        .get(data_id)
        .and_then(|block| block.grease_pencil.as_ref())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "data block has no typed grease pencil payload",
                &operation_pointer(engine.operation_index, "target/id"),
            )
        })
}

fn gp_data_mut<'a>(engine: &'a mut Engine<'_>, data_id: &Id) -> Result<&'a mut GreasePencilData> {
    let operation_index = engine.operation_index;
    engine
        .doc
        .data_blocks
        .get_mut(data_id)
        .and_then(|block| block.grease_pencil.as_mut())
        .ok_or_else(|| missing_payload_error(operation_index))
}

fn missing_payload_error(operation_index: usize) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidOperation,
        "data block has no typed grease pencil payload",
        json!({
            "operation_index": operation_index,
            "pointer": operation_pointer(operation_index, "target/id"),
        }),
    )
}

fn layer_index(engine: &Engine<'_>, data_id: &Id, layer_id: &Id) -> Result<usize> {
    gp_data(engine, data_id)?
        .layers
        .iter()
        .position(|layer| &layer.id == layer_id)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("grease pencil layer `{layer_id}` was not found"),
                &operation_pointer(engine.operation_index, "layer"),
            )
        })
}

fn frame_index(engine: &Engine<'_>, data_id: &Id, layer_index: usize, frame: f64) -> Result<usize> {
    gp_data(engine, data_id)?
        .layers
        .get(layer_index)
        .and_then(|layer| {
            layer
                .frames
                .iter()
                .position(|candidate| crate::float::equal_f64(candidate.frame, frame))
        })
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("grease pencil frame {frame} was not found"),
                &operation_pointer(engine.operation_index, "frame"),
            )
        })
}

fn stroke_index(
    engine: &Engine<'_>,
    data_id: &Id,
    layer_index: usize,
    frame_index: usize,
    stroke_id: &Id,
) -> Result<usize> {
    gp_data(engine, data_id)?
        .layers
        .get(layer_index)
        .and_then(|layer| layer.frames.get(frame_index))
        .and_then(|frame| {
            frame
                .strokes
                .iter()
                .position(|stroke| &stroke.id == stroke_id)
        })
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("grease pencil stroke `{stroke_id}` was not found"),
                &operation_pointer(engine.operation_index, "stroke"),
            )
        })
}

fn set_object<'a>(
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

fn check_grease_pencil_set_fields(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    super::check_set_fields_by(engine, set, allowed, |engine, field| {
        let escaped = super::pointer_escape(field);
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown grease pencil set field `{field}`"),
            &operation_pointer(engine.operation_index, &format!("set/{escaped}")),
        )
    })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    Ok(())
}

fn optional_opacity(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
    default: f64,
) -> Result<f64> {
    let opacity = match value {
        Some(value) => super::finite_number(
            engine,
            Some(value),
            field,
            super::FiniteNumberMessage::Field,
        )?,
        None => default,
    };
    if !(0.0..=1.0).contains(&opacity) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "opacity must be between 0 and 1",
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(opacity)
}

fn read_string_at(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
    field: &str,
) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a string"),
                &operation_pointer(engine.operation_index, field),
            )
        })
}

fn read_bool_at(engine: &Engine<'_>, value: &Value, field: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a boolean"),
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn parse_points(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
) -> Result<Vec<GreasePencilPoint>> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be an array"),
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    if values.len() < 2 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "a stroke must contain at least two points",
            &operation_pointer(engine.operation_index, field),
        ));
    }
    let mut points = Vec::with_capacity(values.len());
    let mut previous_time = None;
    for (index, value) in values.iter().enumerate() {
        let point_field = format!("{field}/{index}");
        let point = value.as_object().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "point must be an object",
                &operation_pointer(engine.operation_index, &point_field),
            )
        })?;
        check_nested_fields(
            engine,
            point,
            &["position", "pressure", "radius", "opacity", "time"],
            &["position", "pressure", "radius", "opacity", "time"],
            &point_field,
        )?;
        let position = parse_position(
            engine,
            point.get("position"),
            &format!("{point_field}/position"),
        )?;
        let pressure = super::finite_number(
            engine,
            point.get("pressure"),
            &format!("{point_field}/pressure"),
            super::FiniteNumberMessage::Field,
        )?;
        let radius = super::finite_number(
            engine,
            point.get("radius"),
            &format!("{point_field}/radius"),
            super::FiniteNumberMessage::Field,
        )?;
        let opacity = super::finite_number(
            engine,
            point.get("opacity"),
            &format!("{point_field}/opacity"),
            super::FiniteNumberMessage::Field,
        )?;
        let time = super::finite_number(
            engine,
            point.get("time"),
            &format!("{point_field}/time"),
            super::FiniteNumberMessage::Field,
        )?;
        if !(0.0..=1.0).contains(&pressure) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "point pressure must be between 0 and 1",
                &operation_pointer(engine.operation_index, &format!("{point_field}/pressure")),
            ));
        }
        if radius < 0.0 {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "point radius must be non-negative",
                &operation_pointer(engine.operation_index, &format!("{point_field}/radius")),
            ));
        }
        if !(0.0..=1.0).contains(&opacity) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "point opacity must be between 0 and 1",
                &operation_pointer(engine.operation_index, &format!("{point_field}/opacity")),
            ));
        }
        if previous_time.is_some_and(|previous| time < previous) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "point times must be nondecreasing",
                &operation_pointer(engine.operation_index, &format!("{point_field}/time")),
            ));
        }
        previous_time = Some(time);
        points.push(GreasePencilPoint {
            position,
            pressure,
            radius,
            opacity,
            time,
        });
    }
    Ok(points)
}

fn parse_position(engine: &Engine<'_>, value: Option<&Value>, field: &str) -> Result<[f64; 3]> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be an array of three numbers"),
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    if values.len() != 3 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain three numbers"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    let x = super::finite_number(
        engine,
        values.first(),
        &format!("{field}/0"),
        super::FiniteNumberMessage::Field,
    )?;
    let y = super::finite_number(
        engine,
        values.get(1),
        &format!("{field}/1"),
        super::FiniteNumberMessage::Field,
    )?;
    let z = super::finite_number(
        engine,
        values.get(2),
        &format!("{field}/2"),
        super::FiniteNumberMessage::Field,
    )?;
    Ok([x, y, z])
}

fn parse_material(engine: &Engine<'_>, value: Option<&Value>, field: &str) -> Result<Option<Id>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let text = value.as_str().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a material ID or null"),
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    let id = parse_id(
        engine,
        text,
        &operation_pointer(engine.operation_index, field),
    )?;
    if !engine.doc.materials.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("material `{id}` was not found"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(Some(id))
}

fn parse_fill(engine: &Engine<'_>, value: Option<&Value>, field: &str) -> Result<Option<[f64; 4]>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let values = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be an array of four numbers or null"),
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    if values.len() != 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain four numbers"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(Some([
        super::finite_number(
            engine,
            values.first(),
            &format!("{field}/0"),
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            values.get(1),
            &format!("{field}/1"),
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            values.get(2),
            &format!("{field}/2"),
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            values.get(3),
            &format!("{field}/3"),
            super::FiniteNumberMessage::Field,
        )?,
    ]))
}

fn check_nested_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
    required: &[&str],
    prefix: &str,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            let escaped = key.replace('~', "~0").replace('/', "~1");
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown grease pencil field `{key}`"),
                &operation_pointer(engine.operation_index, &format!("{prefix}/{escaped}")),
            ));
        }
    }
    for key in required {
        if !object.contains_key(*key) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("missing required field `{key}`"),
                &operation_pointer(engine.operation_index, &format!("{prefix}/{key}")),
            ));
        }
    }
    Ok(())
}
