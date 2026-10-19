use serde_json::{Map, Value};

use crate::{
    color::{ViewTransform, curve::CurveMapping},
    error::{ErrorCode, Result},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "color.update" => color_update(engine, operation),
        "render.passes_update" => passes_update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid color operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn color_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let scene_id = scene_target(engine, operation)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "color set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    check_fields(
        engine,
        set,
        &[
            "display_device",
            "view_transform",
            "look",
            "exposure",
            "gamma",
            "curve",
        ],
        &[],
    )?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "color set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let display_device = set
        .get("display_device")
        .map(|value| {
            let name = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "display_device must be a string",
                    &operation_pointer(engine.operation_index, "set/display_device"),
                )
            })?;
            if name != "sRGB" {
                return Err(engine.error(
                    ErrorCode::UnsupportedFeature,
                    "only the sRGB display device is implemented",
                    &operation_pointer(engine.operation_index, "set/display_device"),
                ));
            }
            Ok(name.to_owned())
        })
        .transpose()?;
    let view_transform = set
        .get("view_transform")
        .map(|value| {
            let name = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "view_transform must be a string",
                    &operation_pointer(engine.operation_index, "set/view_transform"),
                )
            })?;
            match name {
                "standard" => Ok(ViewTransform::Standard),
                "agx" => Ok(ViewTransform::AgX),
                "filmic" => Ok(ViewTransform::Filmic),
                "raw" => Ok(ViewTransform::Raw),
                "false_color" => Ok(ViewTransform::FalseColor),
                _ => Err(engine.error(
                    ErrorCode::InvalidOperation,
                    format!("unsupported view transform `{name}`"),
                    &operation_pointer(engine.operation_index, "set/view_transform"),
                )),
            }
        })
        .transpose()?;
    let look = set
        .get("look")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "look must be a string",
                    &operation_pointer(engine.operation_index, "set/look"),
                )
            })
        })
        .transpose()?;
    let exposure = numeric_setting(engine, set, "exposure")?;
    let gamma = numeric_setting(engine, set, "gamma")?;
    if gamma.is_some_and(|value| value <= 0.0) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "gamma must be greater than zero",
            &operation_pointer(engine.operation_index, "set/gamma"),
        ));
    }
    let curve = set
        .get("curve")
        .map(|value| parse_curve(engine, value))
        .transpose()?;

    let changed = {
        let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
            crate::error::PotError::new(
                ErrorCode::InternalError,
                "scene disappeared during color update",
            )
        })?;
        let before = scene.color_management.clone();
        if let Some(value) = display_device {
            scene.color_management.display_device = value;
        }
        if let Some(value) = view_transform {
            scene.color_management.view_transform = value;
        }
        if let Some(value) = look {
            scene.color_management.look = value;
        }
        if let Some(value) = exposure {
            scene.color_management.exposure = value;
        }
        if let Some(value) = gamma {
            scene.color_management.gamma = value;
        }
        if let Some(value) = curve {
            scene.color_management.curve = value;
        }
        // Rejects an unknown look, a look that does not apply to the resulting view, or an unsupported
        // display. The batch is atomic, so a rejected update leaves the document unchanged.
        scene.color_management.display_encoder()?;
        scene.color_management != before
    };
    if changed {
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn passes_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let scene_id = scene_target(engine, operation)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "render pass set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    check_fields(engine, set, &["passes"], &["passes"])?;
    let values = set.get("passes").and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "passes must be an array of pass names",
            &operation_pointer(engine.operation_index, "set/passes"),
        )
    })?;
    if values.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "at least one render pass is required",
            &operation_pointer(engine.operation_index, "set/passes"),
        ));
    }
    let supported = [
        "combined",
        "z",
        "normal",
        "albedo",
        "emission",
        "object_index",
        "cryptomatte",
        "ao",
    ];
    let mut passes = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let name = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "render pass name must be a string",
                &operation_pointer(engine.operation_index, &format!("set/passes/{index}")),
            )
        })?;
        if !supported.contains(&name) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported render pass `{name}`"),
                &operation_pointer(engine.operation_index, &format!("set/passes/{index}")),
            ));
        }
        if passes.iter().any(|previous| previous == name) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("duplicate render pass `{name}`"),
                &operation_pointer(engine.operation_index, &format!("set/passes/{index}")),
            ));
        }
        passes.push(name.to_owned());
    }
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
        crate::error::PotError::new(
            ErrorCode::InternalError,
            "scene disappeared during pass update",
        )
    })?;
    let changed = scene.render.passes != passes;
    if changed {
        scene.render.passes = passes;
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn scene_target(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<crate::model::Id> {
    let target = operation
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "operation target must be an object",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    check_fields(engine, target, &["id"], &["id"])?;
    let text = target.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target requires a scene ID",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let id = parse_id(
        engine,
        text,
        &operation_pointer(engine.operation_index, "target/id"),
    )?;
    if !engine.doc.scenes.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("scene `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok(id)
}

fn numeric_setting(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
) -> Result<Option<f64>> {
    set.get(key)
        .map(|value| {
            let number = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        format!("{key} must be a finite number"),
                        &operation_pointer(engine.operation_index, &format!("set/{key}")),
                    )
                })?;
            Ok(number)
        })
        .transpose()
}

/// Parses `set.curve`: `null` removes the view curve, and an object is a full view `CurveMapping`
/// (see [`CurveMapping::from_json`]).
fn parse_curve(engine: &Engine<'_>, value: &Value) -> Result<Option<CurveMapping>> {
    if value.is_null() {
        return Ok(None);
    }
    CurveMapping::from_json(value).map(Some).map_err(|message| {
        engine.error(
            ErrorCode::InvalidOperation,
            message,
            &operation_pointer(engine.operation_index, "set/curve"),
        )
    })
}
