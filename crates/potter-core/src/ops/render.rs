use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{
        CyclesCurveShape, EEVEE_GI_CUBEMAP_RESOLUTIONS, EEVEE_RAYTRACING_RESOLUTION_SCALES,
        EeveeFastGiMethod, EeveeRayTracingMethod, PixelFilterType,
    },
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape};

const SET_FIELDS: &[&str] = &[
    "resolution_x",
    "resolution_y",
    "resolution_percentage",
    "samples",
    "seed",
    "max_bounces",
    "volume_bounces",
    "transparent_max_bounces",
    "pixel_filter_type",
    "filter_width",
    "filter_glossy",
    "film_transparent",
    "filter_size",
    "engine",
    "use_sequencer",
    "audio_codec",
    "motion_blur",
    "shutter",
    "motion_blur_samples",
    "dicing_rate",
    "preview_dicing_rate",
    "max_subdivisions",
    "cycles_curves_shape",
    "cycles_curves_subdivisions",
    "eevee_clamp_surface_indirect",
    "eevee_gi_cubemap_resolution",
    "eevee_use_raytracing",
    "eevee_use_fast_gi",
    "eevee_raytracing_method",
    "eevee_raytracing_resolution_scale",
    "eevee_raytracing_screen_trace_quality",
    "eevee_raytracing_screen_trace_thickness",
    "eevee_raytracing_trace_max_roughness",
    "eevee_raytracing_use_backface_hit",
    "eevee_raytracing_backface_radiance_scale",
    "eevee_raytracing_use_denoise",
    "eevee_raytracing_denoise_spatial",
    "eevee_raytracing_denoise_temporal",
    "eevee_raytracing_denoise_bilateral",
    "eevee_fast_gi_bias",
    "eevee_fast_gi_resolution",
    "eevee_fast_gi_step_count",
    "eevee_fast_gi_ray_count",
    "eevee_fast_gi_quality",
    "eevee_fast_gi_distance",
    "eevee_fast_gi_thickness_near",
    "eevee_fast_gi_method",
    "offscreen_dicing_scale",
    "eevee_use_volumetric_shadows",
    "eevee_volumetric_tile_size",
    "eevee_volumetric_samples",
    "eevee_volumetric_sample_distribution",
    "eevee_use_volume_custom_range",
    "eevee_volumetric_start",
    "eevee_volumetric_end",
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
    let volume_bounces = optional_u32(engine, set, "volume_bounces", 0, 1024)?;
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
    let filter_width = optional_f64(
        engine,
        set,
        "filter_width",
        crate::model::FILTER_WIDTH_MIN,
        crate::model::FILTER_WIDTH_MAX,
    )?;
    // Blender's `cycles.blur_glossy` ranges over 0 to 10 (`intern/cycles/blender/addon/properties.py`, `blur_glossy`).
    let filter_glossy = optional_f64(engine, set, "filter_glossy", 0.0, 10.0)?;
    let motion_blur_samples = optional_u32(engine, set, "motion_blur_samples", 1, 64)?;
    let dicing_rate = optional_f64(engine, set, "dicing_rate", 0.1, 1000.0)?;
    let filter_size = optional_f64(engine, set, "filter_size", 0.0, 500.0)?;
    let preview_dicing_rate = optional_f64(engine, set, "preview_dicing_rate", 0.1, 1000.0)?;
    let max_subdivisions = optional_u32(engine, set, "max_subdivisions", 0, 16)?;
    let cycles_curves_shape = set
        .get("cycles_curves_shape")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            let name = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "cycles_curves_shape must be null or a snake_case enum value",
                    &operation_pointer(engine.operation_index, "set/cycles_curves_shape"),
                )
            })?;
            match name {
                "ribbons" => Ok(Some(CyclesCurveShape::Ribbons)),
                "thick" => Ok(Some(CyclesCurveShape::Thick)),
                "thick_linear" => Ok(Some(CyclesCurveShape::ThickLinear)),
                _ => Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "cycles_curves_shape must be ribbons, thick, or thick_linear",
                    &operation_pointer(engine.operation_index, "set/cycles_curves_shape"),
                )),
            }
        })
        .transpose()?;
    let cycles_curves_subdivisions = set
        .get("cycles_curves_subdivisions")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            let subdivisions = value.as_u64().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "cycles_curves_subdivisions must be null or a non-negative integer",
                    &operation_pointer(engine.operation_index, "set/cycles_curves_subdivisions"),
                )
            })?;
            let subdivisions = u32::try_from(subdivisions).map_err(|_| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "cycles_curves_subdivisions exceeds the supported integer range",
                    &operation_pointer(engine.operation_index, "set/cycles_curves_subdivisions"),
                )
            })?;
            if subdivisions > 24 {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "cycles_curves_subdivisions must be in 0..=24",
                    &operation_pointer(engine.operation_index, "set/cycles_curves_subdivisions"),
                ));
            }
            Ok(Some(subdivisions))
        })
        .transpose()?;
    let eevee_gi_cubemap_resolution = set
        .get("eevee_gi_cubemap_resolution")
        .map(|value| {
            if value.is_null() {
                return Ok::<Option<u32>, crate::error::PotError>(None);
            }
            let resolution = value
                .as_u64()
                .and_then(|resolution| u32::try_from(resolution).ok())
                .filter(|resolution| EEVEE_GI_CUBEMAP_RESOLUTIONS.contains(resolution))
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "eevee_gi_cubemap_resolution must be null or one of 128, 256, 512, 1024, 2048, or 4096",
                        &operation_pointer(
                            engine.operation_index,
                            "set/eevee_gi_cubemap_resolution",
                        ),
                    )
                })?;
            Ok(Some(resolution))
        })
        .transpose()?;
    let eevee_use_raytracing = optional_nullable_bool(engine, set, "eevee_use_raytracing")?;
    let eevee_use_fast_gi = optional_nullable_bool(engine, set, "eevee_use_fast_gi")?;
    let eevee_raytracing_method = set
        .get("eevee_raytracing_method")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            match value.as_str() {
                Some("probe") => Ok(Some(EeveeRayTracingMethod::Probe)),
                Some("screen") => Ok(Some(EeveeRayTracingMethod::Screen)),
                _ => Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "eevee_raytracing_method must be null, probe, or screen",
                    &operation_pointer(engine.operation_index, "set/eevee_raytracing_method"),
                )),
            }
        })
        .transpose()?;
    let eevee_raytracing_resolution_scale =
        optional_nullable_u32(engine, set, "eevee_raytracing_resolution_scale", 1, 16)?;
    if matches!(
        eevee_raytracing_resolution_scale,
        OptionalUpdate::Set(value) if !EEVEE_RAYTRACING_RESOLUTION_SCALES.contains(&value)
    ) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "eevee_raytracing_resolution_scale must be 1, 2, 4, 8, or 16",
            &operation_pointer(
                engine.operation_index,
                "set/eevee_raytracing_resolution_scale",
            ),
        ));
    }
    let eevee_raytracing_screen_trace_quality = optional_nullable_f64(
        engine,
        set,
        "eevee_raytracing_screen_trace_quality",
        0.0,
        1.0,
    )?;
    let eevee_raytracing_screen_trace_thickness = optional_nullable_f64(
        engine,
        set,
        "eevee_raytracing_screen_trace_thickness",
        1e-6,
        f32::MAX as f64,
    )?;
    let eevee_raytracing_trace_max_roughness = optional_nullable_f64(
        engine,
        set,
        "eevee_raytracing_trace_max_roughness",
        0.0,
        1.0,
    )?;
    let eevee_raytracing_use_backface_hit =
        optional_nullable_bool(engine, set, "eevee_raytracing_use_backface_hit")?;
    let eevee_raytracing_backface_radiance_scale = optional_nullable_f64(
        engine,
        set,
        "eevee_raytracing_backface_radiance_scale",
        0.0,
        1.0,
    )?;
    let eevee_raytracing_use_denoise =
        optional_nullable_bool(engine, set, "eevee_raytracing_use_denoise")?;
    let eevee_raytracing_denoise_spatial =
        optional_nullable_bool(engine, set, "eevee_raytracing_denoise_spatial")?;
    let eevee_raytracing_denoise_temporal =
        optional_nullable_bool(engine, set, "eevee_raytracing_denoise_temporal")?;
    let eevee_raytracing_denoise_bilateral =
        optional_nullable_bool(engine, set, "eevee_raytracing_denoise_bilateral")?;
    let eevee_fast_gi_bias = optional_nullable_f64(engine, set, "eevee_fast_gi_bias", 0.0, 1.0)?;
    let eevee_fast_gi_resolution =
        optional_nullable_u32(engine, set, "eevee_fast_gi_resolution", 1, 16)?;
    if matches!(
        eevee_fast_gi_resolution,
        OptionalUpdate::Set(value) if !EEVEE_RAYTRACING_RESOLUTION_SCALES.contains(&value)
    ) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "eevee_fast_gi_resolution must be 1, 2, 4, 8, or 16",
            &operation_pointer(engine.operation_index, "set/eevee_fast_gi_resolution"),
        ));
    }
    let eevee_fast_gi_step_count =
        optional_nullable_u32(engine, set, "eevee_fast_gi_step_count", 1, 64)?;
    let eevee_fast_gi_ray_count =
        optional_nullable_u32(engine, set, "eevee_fast_gi_ray_count", 1, 16)?;
    let eevee_fast_gi_quality =
        optional_nullable_f64(engine, set, "eevee_fast_gi_quality", 0.0, 1.0)?;
    let eevee_fast_gi_distance =
        optional_nullable_f64(engine, set, "eevee_fast_gi_distance", 0.0, 100_000.0)?;
    let eevee_fast_gi_thickness_near =
        optional_nullable_f64(engine, set, "eevee_fast_gi_thickness_near", 0.0, 100_000.0)?;
    let eevee_fast_gi_method = set
        .get("eevee_fast_gi_method")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            match value.as_str() {
                Some("ambient_occlusion_only") => {
                    Ok(Some(EeveeFastGiMethod::AmbientOcclusionOnly))
                }
                Some("global_illumination") => {
                    Ok(Some(EeveeFastGiMethod::GlobalIllumination))
                }
                _ => Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "eevee_fast_gi_method must be null, ambient_occlusion_only, or global_illumination",
                    &operation_pointer(engine.operation_index, "set/eevee_fast_gi_method"),
                )),
            }
        })
        .transpose()?;
    let eevee_clamp_surface_indirect = optional_nullable_f64(
        engine,
        set,
        "eevee_clamp_surface_indirect",
        0.0,
        f32::MAX as f64,
    )?;
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
    let eevee_use_volumetric_shadows = set
        .get("eevee_use_volumetric_shadows")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "eevee_use_volumetric_shadows must be a boolean",
                    &operation_pointer(engine.operation_index, "set/eevee_use_volumetric_shadows"),
                )
            })
        })
        .transpose()?;
    let eevee_volumetric_tile_size =
        optional_u32(engine, set, "eevee_volumetric_tile_size", 1, 16)?;
    if eevee_volumetric_tile_size.is_some_and(|value| !matches!(value, 1 | 2 | 4 | 8 | 16)) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "eevee_volumetric_tile_size must be 1, 2, 4, 8, or 16",
            &operation_pointer(engine.operation_index, "set/eevee_volumetric_tile_size"),
        ));
    }
    let eevee_volumetric_samples = optional_u32(engine, set, "eevee_volumetric_samples", 1, 256)?;
    let eevee_volumetric_sample_distribution = optional_f64(
        engine,
        set,
        "eevee_volumetric_sample_distribution",
        0.0,
        1.0,
    )?;
    let eevee_use_volume_custom_range = set
        .get("eevee_use_volume_custom_range")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "eevee_use_volume_custom_range must be a boolean",
                    &operation_pointer(engine.operation_index, "set/eevee_use_volume_custom_range"),
                )
            })
        })
        .transpose()?;
    let eevee_volumetric_start = optional_positive_f64(engine, set, "eevee_volumetric_start")?;
    let eevee_volumetric_end = optional_positive_f64(engine, set, "eevee_volumetric_end")?;
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
        if let Some(value) = volume_bounces {
            scene.render.volume_bounces = value;
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
        if let Some(value) = filter_glossy {
            scene.render.filter_glossy = value;
        }
        if let Some(value) = film_transparent {
            scene.render.film_transparent = value;
        }
        if let Some(value) = eevee_use_volumetric_shadows {
            scene.render.eevee_use_volumetric_shadows = value;
        }
        if let Some(value) = eevee_volumetric_tile_size {
            scene.render.eevee_volumetric_tile_size = value;
        }
        if let Some(value) = eevee_volumetric_samples {
            scene.render.eevee_volumetric_samples = value;
        }
        if let Some(value) = eevee_volumetric_sample_distribution {
            scene.render.eevee_volumetric_sample_distribution = value;
        }
        if let Some(value) = eevee_use_volume_custom_range {
            scene.render.eevee_use_volume_custom_range = value;
        }
        if let Some(value) = eevee_volumetric_start {
            scene.render.eevee_volumetric_start = value;
        }
        if let Some(value) = eevee_volumetric_end {
            scene.render.eevee_volumetric_end = value;
        }
        if let Some(value) = filter_size {
            scene.render.filter_size = value;
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
        if let Some(value) = cycles_curves_shape {
            scene.render.cycles_curves_shape = value;
        }
        if let Some(value) = cycles_curves_subdivisions {
            scene.render.cycles_curves_subdivisions = value;
        }
        if let Some(value) = eevee_gi_cubemap_resolution {
            scene.render.eevee_gi_cubemap_resolution = value;
        }
        eevee_use_raytracing.apply_to(&mut scene.render.eevee_use_raytracing);
        eevee_use_fast_gi.apply_to(&mut scene.render.eevee_use_fast_gi);
        if let Some(value) = eevee_raytracing_method {
            scene.render.eevee_raytracing_method = value;
        }
        eevee_raytracing_resolution_scale
            .apply_to(&mut scene.render.eevee_raytracing_resolution_scale);
        eevee_raytracing_screen_trace_quality
            .apply_to(&mut scene.render.eevee_raytracing_screen_trace_quality);
        eevee_raytracing_screen_trace_thickness
            .apply_to(&mut scene.render.eevee_raytracing_screen_trace_thickness);
        eevee_raytracing_trace_max_roughness
            .apply_to(&mut scene.render.eevee_raytracing_trace_max_roughness);
        eevee_raytracing_use_backface_hit
            .apply_to(&mut scene.render.eevee_raytracing_use_backface_hit);
        eevee_raytracing_backface_radiance_scale
            .apply_to(&mut scene.render.eevee_raytracing_backface_radiance_scale);
        eevee_raytracing_use_denoise.apply_to(&mut scene.render.eevee_raytracing_use_denoise);
        eevee_raytracing_denoise_spatial
            .apply_to(&mut scene.render.eevee_raytracing_denoise_spatial);
        eevee_raytracing_denoise_temporal
            .apply_to(&mut scene.render.eevee_raytracing_denoise_temporal);
        eevee_raytracing_denoise_bilateral
            .apply_to(&mut scene.render.eevee_raytracing_denoise_bilateral);
        eevee_fast_gi_bias.apply_to(&mut scene.render.eevee_fast_gi_bias);
        eevee_fast_gi_resolution.apply_to(&mut scene.render.eevee_fast_gi_resolution);
        eevee_fast_gi_step_count.apply_to(&mut scene.render.eevee_fast_gi_step_count);
        eevee_fast_gi_ray_count.apply_to(&mut scene.render.eevee_fast_gi_ray_count);
        eevee_fast_gi_quality.apply_to(&mut scene.render.eevee_fast_gi_quality);
        eevee_fast_gi_distance.apply_to(&mut scene.render.eevee_fast_gi_distance);
        eevee_fast_gi_thickness_near.apply_to(&mut scene.render.eevee_fast_gi_thickness_near);
        eevee_clamp_surface_indirect.apply_to(&mut scene.render.eevee_clamp_surface_indirect);
        if let Some(value) = eevee_fast_gi_method {
            scene.render.eevee_fast_gi_method = value;
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

#[derive(Clone, Copy)]
enum OptionalUpdate<T> {
    Unchanged,
    Clear,
    Set(T),
}

impl<T> OptionalUpdate<T> {
    fn apply_to(self, target: &mut Option<T>) {
        match self {
            Self::Unchanged => {}
            Self::Clear => *target = None,
            Self::Set(value) => *target = Some(value),
        }
    }
}

fn optional_nullable_bool(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
) -> Result<OptionalUpdate<bool>> {
    let Some(value) = set.get(key) else {
        return Ok(OptionalUpdate::Unchanged);
    };
    if value.is_null() {
        return Ok(OptionalUpdate::Clear);
    }
    value.as_bool().map(OptionalUpdate::Set).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} must be a boolean or null"),
            &operation_pointer(engine.operation_index, &format!("set/{key}")),
        )
    })
}

fn optional_nullable_f64(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
    minimum: f64,
    maximum: f64,
) -> Result<OptionalUpdate<f64>> {
    let Some(value) = set.get(key) else {
        return Ok(OptionalUpdate::Unchanged);
    };
    if value.is_null() {
        return Ok(OptionalUpdate::Clear);
    }
    value
        .as_f64()
        .filter(|number| number.is_finite() && (minimum..=maximum).contains(number))
        .map(OptionalUpdate::Set)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be finite, null, or in {minimum}..={maximum}"),
                &operation_pointer(engine.operation_index, &format!("set/{key}")),
            )
        })
}

fn optional_nullable_u32(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
    minimum: u32,
    maximum: u32,
) -> Result<OptionalUpdate<u32>> {
    let Some(value) = set.get(key) else {
        return Ok(OptionalUpdate::Unchanged);
    };
    if value.is_null() {
        return Ok(OptionalUpdate::Clear);
    }
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| (minimum..=maximum).contains(number))
        .map(OptionalUpdate::Set)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be null or an integer in {minimum}..={maximum}"),
                &operation_pointer(engine.operation_index, &format!("set/{key}")),
            )
        })
}

fn optional_positive_f64(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    key: &str,
) -> Result<Option<f64>> {
    set.get(key)
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite() && *number > 0.0)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        format!("{key} must be a finite positive number"),
                        &operation_pointer(engine.operation_index, &format!("set/{key}")),
                    )
                })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        error::ErrorCode,
        model::{CyclesCurveShape, EeveeFastGiMethod, EeveeRayTracingMethod, Id, SceneDoc},
        ops::apply_batch,
    };

    #[test]
    fn cycles_curve_render_settings_can_be_set_cleared_and_validated() -> crate::error::Result<()> {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let updated = apply_batch(
            &doc,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {
                        "cycles_curves_shape": "ribbons",
                        "cycles_curves_subdivisions": 2
                    }
                }]
            }),
        )?;
        let scene_id = Id::from_static("scene_main");
        let render = &updated.doc.scenes[&scene_id].render;
        assert_eq!(render.cycles_curves_shape, Some(CyclesCurveShape::Ribbons));
        assert_eq!(render.cycles_curves_subdivisions, Some(2));

        let cleared = apply_batch(
            &updated.doc,
            &json!({
                "schema_version": 1,
                "base_revision": updated.doc.revision,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {
                        "cycles_curves_shape": null,
                        "cycles_curves_subdivisions": null
                    }
                }]
            }),
        )?;
        let render = &cleared.doc.scenes[&scene_id].render;
        assert_eq!(render.cycles_curves_shape, None);
        assert_eq!(render.cycles_curves_subdivisions, None);

        assert!(
            matches!(
                apply_batch(
                    &doc,
                    &json!({
                        "schema_version": 1,
                        "base_revision": 0,
                        "operations": [{
                            "op": "render.update",
                            "target": {"id": "scene_main"},
                            "set": {"cycles_curves_subdivisions": 25}
                        }]
                    }),
                ),
                Err(error) if error.code == ErrorCode::InvalidOperation
            ),
            "subdivisions beyond Blender's RNA range must be rejected"
        );
        Ok(())
    }
    #[test]
    fn eevee_probe_resolution_render_update_sets_clears_and_validates() -> crate::error::Result<()>
    {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let updated = apply_batch(
            &doc,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {"eevee_gi_cubemap_resolution": 1024}
                }]
            }),
        )?;
        let scene_id = Id::from_static("scene_main");
        assert_eq!(
            updated.doc.scenes[&scene_id]
                .render
                .eevee_gi_cubemap_resolution,
            Some(1024)
        );

        let cleared = apply_batch(
            &updated.doc,
            &json!({
                "schema_version": 1,
                "base_revision": updated.doc.revision,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {"eevee_gi_cubemap_resolution": null}
                }]
            }),
        )?;
        assert_eq!(
            cleared.doc.scenes[&scene_id]
                .render
                .eevee_gi_cubemap_resolution,
            None
        );

        assert!(
            matches!(
                apply_batch(
                    &doc,
                    &json!({
                        "schema_version": 1,
                        "base_revision": 0,
                        "operations": [{
                            "op": "render.update",
                            "target": {"id": "scene_main"},
                            "set": {"eevee_gi_cubemap_resolution": 2049}
                        }]
                    }),
                ),
                Err(error) if error.code == ErrorCode::InvalidOperation
            ),
            "unsupported Blender enum values must be rejected"
        );
        Ok(())
    }
    #[test]
    fn eevee_raytracing_render_update_sets_clears_and_validates() -> crate::error::Result<()> {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let scene_id = Id::from_static("scene_main");
        for (value, expected) in [
            (json!(true), Some(true)),
            (json!(false), Some(false)),
            (json!(null), None),
        ] {
            let updated = apply_batch(
                &doc,
                &json!({
                    "schema_version": 1,
                    "base_revision": 0,
                    "operations": [{
                        "op": "render.update",
                        "target": {"id": "scene_main"},
                        "set": {"eevee_use_raytracing": value}
                    }]
                }),
            )?;
            assert_eq!(
                updated.doc.scenes[&scene_id].render.eevee_use_raytracing,
                expected
            );
        }

        assert!(
            matches!(
                apply_batch(
                    &doc,
                    &json!({
                        "schema_version": 1,
                        "base_revision": 0,
                        "operations": [{
                            "op": "render.update",
                            "target": {"id": "scene_main"},
                            "set": {"eevee_use_raytracing": 1}
                        }]
                    }),
                ),
                Err(error) if error.code == ErrorCode::InvalidOperation
            ),
            "non-boolean raytracing values must be rejected"
        );
        Ok(())
    }
    #[test]
    fn eevee_raytracing_and_fast_gi_render_update_sets_clears_and_validates()
    -> crate::error::Result<()> {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let updated = apply_batch(
            &doc,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {
                        "eevee_use_raytracing": true,
                        "eevee_use_fast_gi": false,
                        "eevee_raytracing_method": "screen",
                        "eevee_raytracing_resolution_scale": 4,
                        "eevee_raytracing_screen_trace_quality": 0.4,
                        "eevee_raytracing_screen_trace_thickness": 0.25,
                        "eevee_raytracing_trace_max_roughness": 0.75,
                        "eevee_raytracing_use_backface_hit": false,
                        "eevee_raytracing_backface_radiance_scale": 0.4,
                        "eevee_raytracing_use_denoise": true,
                        "eevee_raytracing_denoise_spatial": false,
                        "eevee_raytracing_denoise_temporal": true,
                        "eevee_raytracing_denoise_bilateral": false,
                        "eevee_fast_gi_bias": 0.2,
                        "eevee_fast_gi_resolution": 8,
                        "eevee_fast_gi_step_count": 32,
                        "eevee_fast_gi_ray_count": 5,
                        "eevee_fast_gi_quality": 0.7,
                        "eevee_fast_gi_distance": 200.0,
                        "eevee_fast_gi_thickness_near": 0.3,
                        "eevee_fast_gi_method": "ambient_occlusion_only"
                    }
                }]
            }),
        )?;
        let render = &updated.doc.scenes[&Id::from_static("scene_main")].render;
        assert_eq!(render.eevee_use_raytracing, Some(true));
        assert_eq!(render.eevee_use_fast_gi, Some(false));
        assert_eq!(
            render.eevee_raytracing_method,
            Some(EeveeRayTracingMethod::Screen)
        );
        assert_eq!(render.eevee_raytracing_resolution_scale, Some(4));
        assert_eq!(
            render.eevee_fast_gi_method,
            Some(EeveeFastGiMethod::AmbientOcclusionOnly)
        );
        assert_eq!(render.eevee_fast_gi_ray_count, Some(5));
        let omitted = apply_batch(
            &updated.doc,
            &json!({
                "schema_version": 1,
                "base_revision": updated.doc.revision,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {"eevee_fast_gi_method": "global_illumination"}
                }]
            }),
        )?;
        let render = &omitted.doc.scenes[&Id::from_static("scene_main")].render;
        assert_eq!(render.eevee_use_raytracing, Some(true));
        assert_eq!(render.eevee_raytracing_resolution_scale, Some(4));
        assert_eq!(render.eevee_fast_gi_bias, Some(0.2));

        let cleared = apply_batch(
            &updated.doc,
            &json!({
                "schema_version": 1,
                "base_revision": updated.doc.revision,
                "operations": [{
                    "op": "render.update",
                    "target": {"id": "scene_main"},
                    "set": {
                        "eevee_use_raytracing": null,
                        "eevee_raytracing_resolution_scale": null,
                        "eevee_fast_gi_method": null
                    }
                }]
            }),
        )?;
        let render = &cleared.doc.scenes[&Id::from_static("scene_main")].render;
        assert_eq!(render.eevee_use_raytracing, None);
        assert_eq!(render.eevee_raytracing_resolution_scale, None);
        assert_eq!(render.eevee_fast_gi_method, None);

        assert!(
            matches!(
                apply_batch(
                    &doc,
                    &json!({
                        "schema_version": 1,
                        "base_revision": 0,
                        "operations": [{
                            "op": "render.update",
                            "target": {"id": "scene_main"},
                            "set": {"eevee_raytracing_resolution_scale": 3}
                        }]
                    }),
                ),
                Err(error) if error.code == ErrorCode::InvalidOperation
            ),
            "unsupported ray-tracing resolutions must be rejected"
        );
        Ok(())
    }
}
