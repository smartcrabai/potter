use serde_json::{Map, Value};

use crate::{
    color::ViewTransform,
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
            let look = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "look must be a string",
                    &operation_pointer(engine.operation_index, "set/look"),
                )
            })?;
            let supported = matches!(
                look,
                "none"
                    | "None"
                    | "AgX - None"
                    | "high_contrast"
                    | "High Contrast"
                    | "AgX - High Contrast"
                    | "AgX - Punchy"
                    | "medium_high_contrast"
                    | "Medium High Contrast"
                    | "AgX - Medium High Contrast"
                    | "medium_low_contrast"
                    | "Medium Low Contrast"
                    | "AgX - Medium Low Contrast"
                    | "low_contrast"
                    | "Low Contrast"
                    | "AgX - Low Contrast"
                    | "very_low_contrast"
                    | "Very Low Contrast"
                    | "AgX - Very Low Contrast"
                    | "very_high_contrast"
                    | "Very High Contrast"
                    | "AgX - Very High Contrast"
            );
            if !supported {
                return Err(engine.error(
                    ErrorCode::UnsupportedFeature,
                    format!("color look `{look}` is not supported"),
                    &operation_pointer(engine.operation_index, "set/look"),
                ));
            }
            Ok::<String, crate::error::PotError>(look.to_owned())
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

fn parse_curve(engine: &Engine<'_>, value: &Value) -> Result<Vec<[f64; 2]>> {
    let points = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "curve must be an array of [x,y] points",
            &operation_pointer(engine.operation_index, "set/curve"),
        )
    })?;
    if points.len() < 2 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "curve requires at least two points",
            &operation_pointer(engine.operation_index, "set/curve"),
        ));
    }
    let mut curve = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let pair = point
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "curve points must have two coordinates",
                    &operation_pointer(engine.operation_index, &format!("set/curve/{index}")),
                )
            })?;
        let x = pair[0]
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "curve x coordinate must be finite",
                    &operation_pointer(engine.operation_index, &format!("set/curve/{index}/0")),
                )
            })?;
        let y = pair[1]
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "curve y coordinate must be finite",
                    &operation_pointer(engine.operation_index, &format!("set/curve/{index}/1")),
                )
            })?;
        if curve
            .last()
            .is_some_and(|previous: &[f64; 2]| x <= previous[0])
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "curve x coordinates must be strictly increasing",
                &operation_pointer(engine.operation_index, &format!("set/curve/{index}/0")),
            ));
        }
        curve.push([x, y]);
    }
    Ok(curve)
}
