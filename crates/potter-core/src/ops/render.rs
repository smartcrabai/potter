use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::PixelFilterType,
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape};

const SET_FIELDS: &[&str] = &[
    "resolution_x",
    "resolution_y",
    "resolution_percentage",
    "samples",
    "seed",
    "max_bounces",
    "transparent_max_bounces",
    "pixel_filter_type",
    "filter_width",
    "film_transparent",
    "engine",
    "use_sequencer",
    "audio_codec",
    "motion_blur",
    "shutter",
    "motion_blur_samples",
    "dicing_rate",
    "preview_dicing_rate",
    "max_subdivisions",
    "offscreen_dicing_scale",
    "dicing_camera",
];

pub(super) fn apply(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing render target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let target = target.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "render target must be an object",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    for key in target.keys() {
        if key != "id" {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown render target field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("target/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let id_text = target.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "render target requires a scene ID",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let scene_id = parse_id(
        engine,
        id_text,
        &operation_pointer(engine.operation_index, "target/id"),
    )?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "render set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "render set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    for key in set.keys() {
        if !SET_FIELDS.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown render setting `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    if !engine.doc.scenes.contains_key(&scene_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("scene `{scene_id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    let resolution_x = optional_u32(engine, set, "resolution_x", 1, 16_384)?;
    let resolution_y = optional_u32(engine, set, "resolution_y", 1, 16_384)?;
    let resolution_percentage = optional_u32(engine, set, "resolution_percentage", 1, 100)?;
    let samples = optional_u32(engine, set, "samples", 1, 1_000_000)?;
    let seed = optional_u32(engine, set, "seed", 0, u32::MAX)?;
    let max_bounces = optional_u32(engine, set, "max_bounces", 0, 1024)?;
    let transparent_max_bounces = optional_u32(engine, set, "transparent_max_bounces", 0, 1024)?;
    let pixel_filter_type = set
        .get("pixel_filter_type")
        .map(|value| {
            let name = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "pixel_filter_type must be a string",
                    &operation_pointer(engine.operation_index, "set/pixel_filter_type"),
                )
            })?;
            match name {
                "BOX" => Ok(PixelFilterType::Box),
                "GAUSSIAN" => Ok(PixelFilterType::Gaussian),
                "BLACKMAN_HARRIS" => Ok(PixelFilterType::BlackmanHarris),
                _ => Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "pixel_filter_type must be BOX, GAUSSIAN, or BLACKMAN_HARRIS",
                    &operation_pointer(engine.operation_index, "set/pixel_filter_type"),
                )),
            }
        })
        .transpose()?;
    let filter_width = optional_f64(engine, set, "filter_width", 0.01, 10.0)?;
    let motion_blur_samples = optional_u32(engine, set, "motion_blur_samples", 1, 64)?;
    let dicing_rate = optional_f64(engine, set, "dicing_rate", 0.1, 1000.0)?;
    let preview_dicing_rate = optional_f64(engine, set, "preview_dicing_rate", 0.1, 1000.0)?;
    let max_subdivisions = optional_u32(engine, set, "max_subdivisions", 0, 16)?;
    let offscreen_dicing_scale =
        optional_f64(engine, set, "offscreen_dicing_scale", 1.0, f32::MAX as f64)?;
    let dicing_camera = set
        .get("dicing_camera")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            let camera_id = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "dicing_camera must be null or a camera ID",
                    &operation_pointer(engine.operation_index, "set/dicing_camera"),
                )
            })?;
            let camera_id = parse_id(
                engine,
                camera_id,
                &operation_pointer(engine.operation_index, "set/dicing_camera"),
            )?;
            if !engine.doc.is_camera_object(&camera_id) {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "dicing_camera must reference a camera object",
                    &operation_pointer(engine.operation_index, "set/dicing_camera"),
                ));
            }
            Ok(Some(camera_id))
        })
        .transpose()?;
    let motion_blur = set
        .get("motion_blur")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "motion_blur must be a boolean",
                    &operation_pointer(engine.operation_index, "set/motion_blur"),
                )
            })
        })
        .transpose()?;
    let shutter = set
        .get("shutter")
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite() && *number > 0.0 && *number <= 2.0)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "shutter must be finite and in (0, 2]",
                        &operation_pointer(engine.operation_index, "set/shutter"),
                    )
                })
        })
        .transpose()?;
    let use_sequencer = set
        .get("use_sequencer")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "use_sequencer must be a boolean",
                    &operation_pointer(engine.operation_index, "set/use_sequencer"),
                )
            })
        })
        .transpose()?;
    let film_transparent = set
        .get("film_transparent")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "film_transparent must be a boolean",
                    &operation_pointer(engine.operation_index, "set/film_transparent"),
                )
            })
        })
        .transpose()?;
    let render_engine = set
        .get("engine")
        .map(|value| {
            let engine_name = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "engine must be a string",
                    &operation_pointer(engine.operation_index, "set/engine"),
                )
            })?;
            if !matches!(engine_name, "path" | "realtime") {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "engine must be path or realtime",
                    &operation_pointer(engine.operation_index, "set/engine"),
                ));
            }
            Ok(engine_name.to_owned())
        })
        .transpose()?;
    let audio_codec = set
        .get("audio_codec")
        .map(|value| {
            let codec = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "audio_codec must be a string",
                    &operation_pointer(engine.operation_index, "set/audio_codec"),
                )
            })?;
            if !matches!(codec, "wav" | "flac") {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "audio_codec must be wav or flac",
                    &operation_pointer(engine.operation_index, "set/audio_codec"),
                ));
            }
            Ok(codec.to_owned())
        })
        .transpose()?;
    let changed = {
        let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
            crate::error::PotError::new(
                ErrorCode::InternalError,
                "scene disappeared during render update",
            )
        })?;
        let before = scene.render.clone();
        if let Some(value) = resolution_x {
            scene.render.resolution_x = value;
        }
        if let Some(value) = resolution_y {
            scene.render.resolution_y = value;
        }
        if let Some(value) = resolution_percentage {
            scene.render.resolution_percentage = value;
        }
        if let Some(value) = samples {
            scene.render.samples = value;
        }
        if let Some(value) = seed {
            scene.render.seed = value;
        }
        if let Some(value) = max_bounces {
            scene.render.max_bounces = value;
        }
        if let Some(value) = transparent_max_bounces {
            scene.render.transparent_max_bounces = value;
        }
        if let Some(value) = pixel_filter_type {
            scene.render.pixel_filter_type = value;
        }
        if let Some(value) = filter_width {
            scene.render.filter_width = value;
        }
        if let Some(value) = film_transparent {
            scene.render.film_transparent = value;
        }
        if let Some(value) = render_engine {
            scene.render.engine = value;
        }
        if let Some(value) = use_sequencer {
            scene.render.use_sequencer = value;
        }
        if let Some(value) = audio_codec {
            scene.render.audio_codec = value;
        }
        if let Some(value) = motion_blur {
            scene.render.motion_blur = value;
        }
        if let Some(value) = shutter {
            scene.render.shutter = value;
        }
        if let Some(value) = motion_blur_samples {
            scene.render.motion_blur_samples = value;
        }
        if let Some(value) = dicing_rate {
            scene.render.dicing_rate = value;
        }
        if let Some(value) = preview_dicing_rate {
            scene.render.preview_dicing_rate = value;
        }
        if let Some(value) = max_subdivisions {
            scene.render.max_subdivisions = value;
        }
        if let Some(value) = offscreen_dicing_scale {
            scene.render.offscreen_dicing_scale = value;
        }
        if let Some(value) = dicing_camera {
            scene.render.dicing_camera = value;
        }
        scene.render != before
    };
    if changed {
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn optional_u32(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
    minimum: u32,
    maximum: u32,
) -> Result<Option<u32>> {
    set.get(key)
        .map(|value| {
            let number = value.as_u64().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{key} must be a non-negative integer"),
                    &operation_pointer(engine.operation_index, &format!("set/{key}")),
                )
            })?;
            let number = u32::try_from(number).map_err(|_| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{key} exceeds the supported integer range"),
                    &operation_pointer(engine.operation_index, &format!("set/{key}")),
                )
            })?;
            if !(minimum..=maximum).contains(&number) {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{key} must be in {minimum}..={maximum}"),
                    &operation_pointer(engine.operation_index, &format!("set/{key}")),
                ));
            }
            Ok(number)
        })
        .transpose()
}
fn optional_f64(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
    minimum: f64,
    maximum: f64,
) -> Result<Option<f64>> {
    set.get(key)
        .map(|value| {
            let number = value
                .as_f64()
                .filter(|number| number.is_finite() && (minimum..=maximum).contains(number));
            number.ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{key} must be finite and in {minimum}..={maximum}"),
                    &operation_pointer(engine.operation_index, &format!("set/{key}")),
                )
            })
        })
        .transpose()
}
