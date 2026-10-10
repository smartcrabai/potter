use serde_json::Value;

use crate::{
    cli::{RenderArgs, RenderDevice, RenderEngine, RenderFormat},
    commands::util::evaluation_context,
    error::{ErrorCode, PotError, Result},
    model::Id,
    render::render_sequence,
    response::SceneInfo,
    store::Project,
};

const MAX_FRAMES: usize = 1_000_000;

pub fn run(args: RenderArgs) -> Result<(Option<SceneInfo>, Value)> {
    if args.device == RenderDevice::Gpu {
        return Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "GPU rendering is not supported by this CPU build",
        ));
    }
    let format = match args.format {
        RenderFormat::Png => "png",
        RenderFormat::Exr | RenderFormat::Openexr => "exr",
        _ => {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                "this output format is not supported by the CPU renderer",
            ));
        }
    };
    let project = Project::open(args.scene)?;
    let doc = project.doc();
    let context = evaluation_context(
        args.context.scene_id.as_deref(),
        args.context.view_layer.as_deref(),
        args.context.frame,
    )?;
    let scene_id = context.scene_id.as_ref().unwrap_or(&doc.active_scene);
    let scene = doc.scenes.get(scene_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "render scene does not exist",
            serde_json::json!({ "scene_id": scene_id }),
        )
    })?;
    let camera_id = args
        .camera
        .as_deref()
        .map(|value| Id::new(value.to_owned()))
        .transpose()?
        .or_else(|| scene.camera.clone());
    let uses_sequencer = scene.render.use_sequencer && !scene.sequencer.strips.is_empty();
    if camera_id.is_none() && !uses_sequencer {
        return Err(PotError::new(
            ErrorCode::TargetNotFound,
            "render scene has no camera",
        ));
    }
    let frames = args
        .frames
        .as_deref()
        .map(parse_frames)
        .transpose()?
        .unwrap_or_else(|| vec![context.frame.unwrap_or(scene.frame_current)]);
    let settings = &scene.render;
    if settings.resolution_x == 0
        || settings.resolution_y == 0
        || settings.resolution_x > 16_384
        || settings.resolution_y > 16_384
        || !(1..=100).contains(&settings.resolution_percentage)
        || settings.samples == 0
        || settings.samples > 1_000_000
        || settings.max_bounces > 1024
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "scene render settings are outside supported limits",
        ));
    }
    let engine = match args.engine {
        Some(RenderEngine::Path) => "path",
        Some(RenderEngine::Realtime) => "realtime",
        None => settings.engine.as_str(),
    };
    if !matches!(engine, "path" | "realtime") {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "scene render engine must be path or realtime",
        ));
    }
    let width =
        (u64::from(settings.resolution_x) * u64::from(settings.resolution_percentage) / 100).max(1);
    let height =
        (u64::from(settings.resolution_y) * u64::from(settings.resolution_percentage) / 100).max(1);
    let width = u32::try_from(width)
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "render width exceeds limits"))?;
    let height = u32::try_from(height)
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "render height exceeds limits"))?;
    let samples = if engine == "realtime" {
        1
    } else {
        settings.samples
    };
    let seed = if engine == "realtime" {
        0
    } else {
        settings.seed
    };
    let result = render_sequence(
        doc,
        &context,
        project.path(),
        camera_id.as_ref().map(Id::as_str),
        &frames,
        format,
        &args.out,
        args.overwrite,
        engine,
        width,
        height,
        samples,
        seed,
        settings.film_transparent,
    )?;
    Ok((Some(project.info()?), result))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "frame indices are capped before conversion to f64"
)]
fn parse_frames(value: &str) -> Result<Vec<f64>> {
    let parts: Vec<&str> = value.split(':').collect();
    if !(2..=3).contains(&parts.len()) || parts.iter().any(|part| part.is_empty()) {
        return Err(PotError::invalid_argument(
            "frames must use start:end[:step]",
        ));
    }
    let parse = |part: &str| {
        part.parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .ok_or_else(|| PotError::invalid_argument("frame range values must be finite numbers"))
    };
    let start = parse(parts[0])?;
    let end = parse(parts[1])?;
    let step = if parts.len() == 3 {
        parse(parts[2])?
    } else {
        1.0
    };
    let count = frame_range_capacity(start, end, step)?;
    let mut frames = Vec::with_capacity(count);
    for index in 0..count {
        let frame = start + index as f64 * step;
        if frame <= end + step.abs() * 1.0e-12 {
            frames.push(frame.min(end));
        }
    }
    let tolerance = step.abs() * 1.0e-12;
    if frames.last().is_some_and(|last| end - *last > tolerance) {
        if frames.len() == MAX_FRAMES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "frame range is too large",
            ));
        }
        frames.push(end);
    }
    if frames.is_empty() || frames.len() > MAX_FRAMES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "frame range is too large",
        ));
    }
    Ok(frames)
}

fn frame_range_order_is_valid(start: f64, end: f64, step: f64) -> bool {
    start <= end && step > 0.0
}
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the finite frame span is checked and capped before conversion"
)]
fn checked_frame_range_capacity(start: f64, end: f64, step: f64) -> Option<usize> {
    let interval_count = ((end - start) / step).floor();
    if !interval_count.is_finite() || interval_count >= MAX_FRAMES as f64 {
        return None;
    }
    usize::try_from(interval_count as u64)
        .ok()
        .and_then(|count| count.checked_add(1))
}

fn frame_range_capacity(start: f64, end: f64, step: f64) -> Result<usize> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(PotError::invalid_argument(
            "frame range values must be finite numbers",
        ));
    }
    if !frame_range_order_is_valid(start, end, step) {
        return Err(PotError::invalid_argument(
            "frame range requires start <= end and a positive step",
        ));
    }
    checked_frame_range_capacity(start, end, step)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "frame range is too large"))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "frame parser unit tests")]

    use super::parse_frames;

    #[test]
    fn frame_range_is_inclusive_with_fractional_step() {
        let frames = parse_frames("1:2:0.5").unwrap();
        assert_eq!(frames.len(), 3);
        assert!((frames[0] - 1.0).abs() < 1.0e-10);
        assert!((frames[1] - 1.5).abs() < 1.0e-10);
        assert!((frames[2] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn frame_range_includes_an_unaligned_end() {
        let frames = parse_frames("1:4:3").unwrap();
        assert_eq!(frames.len(), 2);
        assert!((frames[0] - 1.0).abs() < 1.0e-10);
        assert!((frames[1] - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn frame_range_rejects_invalid_and_descending_ranges() {
        assert!(parse_frames("1:2:0").is_err());
        assert!(parse_frames("2:1").is_err());
        assert!(parse_frames("1::2").is_err());
    }
}

#[cfg(kani)]
mod kani_verification {
    use super::{checked_frame_range_capacity, frame_range_order_is_valid};

    #[kani::proof]
    fn small_integer_frame_range_capacity_matches_inclusive_steps() {
        let start: u8 = kani::any();
        let end: u8 = kani::any();
        let step: u8 = kani::any();
        kani::assume(start <= end && end <= 15);
        kani::assume(step > 0 && step <= 15);
        let capacity =
            match checked_frame_range_capacity(f64::from(start), f64::from(end), f64::from(step)) {
                Some(capacity) => capacity,
                None => {
                    kani::assert(false, "bounded frame-range inputs must be accepted");
                    return;
                }
            };
        let distance = usize::from(end - start);
        kani::assert(
            capacity == distance / usize::from(step) + 1,
            "inclusive interval capacity is floor(distance / step) plus one",
        );
        kani::assert(
            (1..=16).contains(&capacity),
            "bounded integer ranges have at most sixteen frame slots",
        );
    }

    #[kani::proof]
    fn invalid_frame_ranges_are_rejected() {
        let start: u8 = kani::any();
        let end: u8 = kani::any();
        let step: u8 = kani::any();
        kani::assume(step == 0 || start > end);
        kani::assert(
            !frame_range_order_is_valid(f64::from(start), f64::from(end), f64::from(step)),
            "zero steps and descending ranges are rejected",
        );
    }
}
