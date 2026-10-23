use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};

use crate::{
    compositor::{self, Image as CompositorImage},
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, EvaluationPurpose, EvaluationSession, Snapshot},
    hash,
    image::{ImageData, ImageInterpolation, Precision, sample, storage},
    media::{AudioBuffer, decode_wav, encode_flac, encode_wav},
    model::{Id, ImageColorspace, Scene, SceneDoc},
    sequencer::{
        AudioTrack, EffectType, RgbaFrame, SequencerColorSpace, Strip, StripBlendType, StripType,
        TransitionType, apply_blend_effect, apply_effect, apply_modifiers, blend_frames, mix_audio,
        transition,
    },
};

use super::preview::{encode_png, render_camera_frame};

const MAX_FRAMES: usize = 1_000_000;

fn frame_filename(frame: f64, extension: &str) -> Result<String> {
    let mut filename = String::with_capacity(16 + extension.len());
    filename.push_str("frame_");
    let numeric_start = filename.len();
    std::fmt::Write::write_fmt(&mut filename, format_args!("{frame}")).map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "render frame filename formatting failed",
        )
    })?;
    let numeric = &filename[numeric_start..];
    let sign_offset = usize::from(numeric.starts_with('-'));
    if !numeric.contains('e') && !numeric.contains('E') {
        let integer_end = numeric.find('.').unwrap_or(numeric.len());
        let padding = 4_usize.saturating_sub(integer_end - sign_offset);
        if padding > 0 {
            filename.insert_str(numeric_start + sign_offset, &"000"[..padding]);
        }
    }
    filename.push('.');
    filename.push_str(extension);
    Ok(filename)
}

fn frame_output_names(frames: &[f64], extension: &str) -> Result<Vec<String>> {
    let mut previous_frame = None;
    let mut occurrence = 0;
    frames
        .iter()
        .map(|frame| {
            if previous_frame == Some(frame.to_bits()) {
                occurrence += 1;
            } else {
                previous_frame = Some(frame.to_bits());
                occurrence = 1;
            }
            let mut filename = frame_filename(*frame, extension)?;
            if occurrence > 1
                && let Some(separator) = filename.rfind('.')
            {
                filename.insert_str(separator, &format!("__{occurrence}"));
            }
            Ok(filename)
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "render sequence receives the resolved CLI and scene settings"
)]
pub fn render_sequence(
    doc: &SceneDoc,
    context: &EvaluationContext,
    project_root: &Path,
    cache_directory: &Path,
    camera_id: Option<&str>,
    frames: &[f64],
    format: &str,
    output: &Path,
    overwrite: bool,
    engine: &str,
    width: u32,
    height: u32,
    samples: u32,
    seed: u32,
    film_transparent: bool,
) -> Result<Value> {
    if frames.is_empty()
        || frames.len() > MAX_FRAMES
        || frames.iter().any(|frame| !frame.is_finite())
        || frames.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "render frames must be finite, ordered, and contain between 1 and 1000000 entries",
            json!({ "frame_count": frames.len() }),
        ));
    }
    if !matches!(format, "png" | "exr") {
        return Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            format!("render format {format} is not supported by the CPU renderer"),
        ));
    }
    if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
        return Err(PotError::invalid_argument(
            "render dimensions must be between 1 and 16384 pixels",
        ));
    }
    if samples == 0 || samples > 1_000_000 {
        return Err(PotError::invalid_argument(
            "render samples must be between 1 and 1000000",
        ));
    }
    let scene_id = context.scene_id.as_ref().unwrap_or(&doc.active_scene);
    let scene = doc
        .scenes
        .get(scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let uses_sequencer = scene.render.use_sequencer && !scene.sequencer.strips.is_empty();
    if !uses_sequencer && camera_id.is_none() {
        return Err(PotError::new(
            ErrorCode::TargetNotFound,
            "render scene has no camera",
        ));
    }
    let audio_filename = (uses_sequencer
        && matches!(scene.render.audio_codec.as_str(), "wav" | "flac"))
    .then(|| format!("audio.{}", scene.render.audio_codec));
    let extension = format;
    let expected_names = frame_output_names(frames, extension)?;
    let manifest_path = output.join("render.manifest.json");
    if output.exists() && !overwrite {
        let conflict = expected_names.iter().any(|name| output.join(name).exists())
            || audio_filename
                .as_ref()
                .is_some_and(|name| output.join(name).exists())
            || manifest_path.exists();
        if conflict {
            return Err(PotError::with_details(
                ErrorCode::OutputExists,
                "render output already exists",
                json!({ "output": output }),
            ));
        }
    }
    fs::create_dir_all(output).map_err(|error| PotError::io(&error))?;
    let staging = output.join(format!(".potter-render-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging).map_err(|error| PotError::io(&error))?;
    let result = (|| -> Result<Value> {
        let mut rendered_frames = Vec::with_capacity(frames.len());
        let mut scene_hash = None;
        let mut warnings = Vec::new();
        let mut evaluation_session = EvaluationSession::default();
        for (index, frame) in frames.iter().copied().enumerate() {
            let mut frame_context = context.clone();
            frame_context.frame = Some(frame);
            let filename = expected_names.get(index).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "frame output name is missing")
            })?;
            let staged_image = staging.join(filename);
            let snapshot = if uses_sequencer {
                let snapshot = Snapshot::evaluate_with_cache_session(
                    doc,
                    &frame_context,
                    Some(project_root),
                    Some(cache_directory),
                    &mut evaluation_session,
                )?;
                let composed = render_vse_frame(
                    doc,
                    scene,
                    frame,
                    width,
                    height,
                    project_root,
                    cache_directory,
                    &mut evaluation_session,
                    engine,
                    samples,
                    seed,
                    film_transparent,
                    &mut warnings,
                )?;
                if format == "png" {
                    let image = CompositorImage::new(width, height, composed.pixels)?;
                    let rgba = compositor::encode_associated_image_display(
                        &image,
                        &scene.color_management,
                    )?;
                    let bytes = encode_png(&rgba, width, height)?;
                    fs::write(&staged_image, bytes).map_err(|error| PotError::io(&error))?;
                } else {
                    validate_vse_exr_passes(&scene.render.passes)?;
                    write_exr(&staged_image, &composed.pixels, width, height)?;
                }
                snapshot
            } else {
                let camera_id = camera_id.ok_or_else(|| {
                    PotError::new(ErrorCode::TargetNotFound, "render scene has no camera")
                })?;
                let (snapshot, raster) = if scene.render.motion_blur {
                    render_motion_blurred_frame(
                        doc,
                        &frame_context,
                        &mut evaluation_session,
                        project_root,
                        cache_directory,
                        camera_id,
                        width,
                        height,
                        engine,
                        film_transparent,
                        samples,
                        seed,
                        scene.render.shutter,
                        scene.render.motion_blur_samples,
                        &mut warnings,
                    )?
                } else {
                    let (snapshot, raster, frame_warnings) = render_camera_frame(
                        doc,
                        &frame_context,
                        &mut evaluation_session,
                        project_root,
                        cache_directory,
                        camera_id,
                        width,
                        height,
                        "beauty",
                        engine,
                        film_transparent,
                        samples,
                        seed,
                    )?;
                    push_new_warnings(&mut warnings, frame_warnings);
                    (snapshot, raster)
                };
                if format == "png" {
                    let bytes = encode_png(&raster.rgba, width, height)?;
                    fs::write(&staged_image, &bytes).map_err(|error| PotError::io(&error))?;
                } else {
                    compositor::write_multilayer_exr(
                        &staged_image,
                        &raster,
                        width,
                        height,
                        &scene.render.passes,
                    )?;
                }
                snapshot
            };
            let _ = scene_hash.get_or_insert_with(|| snapshot.scene_hash.clone());
            rendered_frames.push(json!({
                "frame": frame,
                "path": filename,
                "width": width,
                "height": height,
                "format": format,
                "hash": file_hash(&staged_image)?,
                "evaluation_hash": snapshot.evaluation_hash,
                "samples": samples,
                "seed": seed,
                "motion_blur_samples": if scene.render.motion_blur {
                    scene.render.motion_blur_samples
                } else {
                    1
                },
                "shutter": if scene.render.motion_blur {
                    scene.render.shutter
                } else {
                    0.0
                },
            }));
        }
        let audio_info = if let Some(filename) = &audio_filename {
            let audio = render_vse_audio(scene, project_root, frames)?;
            let bytes = match scene.render.audio_codec.as_str() {
                "wav" => encode_wav(&audio)?,
                "flac" => encode_flac(&audio)?,
                _ => {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "sequencer audio codec must be wav or flac",
                    ));
                }
            };
            let staged_audio = staging.join(filename);
            fs::write(&staged_audio, bytes).map_err(|error| PotError::io(&error))?;
            Some(json!({
                "path": filename,
                "format": scene.render.audio_codec,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "hash": file_hash(&staged_audio)?,
            }))
        } else {
            None
        };
        let manifest = json!({
            "scene_id": scene_id,
            "revision": doc.revision,
            "scene_hash": scene_hash,
            "engine": engine,
            "implementation": super::preview::ENGINE,
            "profile": doc.profile,
            "format": format,
            "visibility": super::preview::visibility_name(context.purpose),
            "camera": camera_id,
            "resolution": { "width": width, "height": height },
            "resolution_percentage": scene.render.resolution_percentage,
            "use_sequencer": scene.render.use_sequencer,
            "film_transparent": film_transparent,
            "fps": { "fps": scene.fps, "fps_base": scene.fps_base },
            "color_space": if format == "exr" { "linear" } else { "srgb" },
            "view_transform": "Scene color management",
            "ocio": Value::Null,
            "samples": samples,
            "seed": seed,
            "motion_blur": {
                "enabled": scene.render.motion_blur,
                "shutter": scene.render.shutter,
                "samples": scene.render.motion_blur_samples,
            },
            "audio": audio_info,
            "frames": rendered_frames,
        });
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            PotError::new(
                ErrorCode::InternalError,
                format!("render manifest serialization failed: {error}"),
            )
        })?;
        fs::write(staging.join("render.manifest.json"), manifest_bytes)
            .map_err(|error| PotError::io(&error))?;
        for name in &expected_names {
            fs::rename(staging.join(name), output.join(name))
                .map_err(|error| PotError::io(&error))?;
        }
        if let Some(name) = &audio_filename {
            fs::rename(staging.join(name), output.join(name))
                .map_err(|error| PotError::io(&error))?;
        }
        fs::rename(staging.join("render.manifest.json"), &manifest_path)
            .map_err(|error| PotError::io(&error))?;
        let absolute_manifest = manifest_path
            .canonicalize()
            .map_err(|error| PotError::io(&error))?;
        let absolute_audio = audio_filename
            .as_ref()
            .map(|name| {
                output
                    .join(name)
                    .canonicalize()
                    .map_err(|error| PotError::io(&error))
            })
            .transpose()?;
        let mut output_frames = Vec::with_capacity(rendered_frames.len());
        for (name, frame) in expected_names.iter().zip(&rendered_frames) {
            let path = output
                .join(name)
                .canonicalize()
                .map_err(|error| PotError::io(&error))?;
            let mut frame = frame.clone();
            frame["path"] = json!(path.display().to_string());
            output_frames.push(frame);
        }
        Ok(json!({
            "format": format,
            "engine": engine,
            "camera": camera_id,
            "frame_count": frames.len(),
            "samples": samples,
            "seed": seed,
            "motion_blur": {
                "enabled": scene.render.motion_blur,
                "shutter": scene.render.shutter,
                "samples": scene.render.motion_blur_samples,
            },
            "audio": absolute_audio.map(|path| path.display().to_string()),
            "frames": output_frames,
            "manifest": absolute_manifest.display().to_string(),
            "warnings": warnings,
        }))
    })();
    let _ = fs::remove_dir_all(&staging);
    result
}
#[expect(
    clippy::too_many_arguments,
    reason = "motion blur samples explicit sequence and render settings"
)]
fn render_motion_blurred_frame(
    doc: &SceneDoc,
    context: &EvaluationContext,
    session: &mut EvaluationSession,
    project_root: &Path,
    cache_directory: &Path,
    camera_id: &str,
    width: u32,
    height: u32,
    engine: &str,
    film_transparent: bool,
    samples: u32,
    seed: u32,
    shutter: f64,
    shutter_samples: u32,
    warnings: &mut Vec<Value>,
) -> Result<(Snapshot, crate::render::raster::RasterOutput)> {
    if !shutter.is_finite()
        || shutter <= 0.0
        || shutter > 2.0
        || !(1..=64).contains(&shutter_samples)
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "motion-blur shutter settings are outside supported limits",
        ));
    }
    let scene_id = context.scene_id.as_ref().unwrap_or(&doc.active_scene);
    let scene = doc
        .scenes
        .get(scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "render scene does not exist"))?;
    let center_frame = context.frame.unwrap_or(scene.frame_current);
    let mut center_context = context.clone();
    center_context.frame = Some(center_frame);
    let center_snapshot = Snapshot::evaluate_with_cache_session(
        doc,
        &center_context,
        Some(project_root),
        Some(cache_directory),
        session,
    )?;
    let mut accumulated = None;
    let center_sample = shutter_samples / 2;
    for sample_index in 0..shutter_samples {
        let offset = (f64::from(sample_index) + 0.5) / f64::from(shutter_samples) - 0.5;
        let mut sample_context = context.clone();
        sample_context.frame = Some(center_frame + offset * shutter);
        let (_, rendered, sample_warnings) = render_camera_frame(
            doc,
            &sample_context,
            session,
            project_root,
            cache_directory,
            camera_id,
            width,
            height,
            "beauty",
            engine,
            film_transparent,
            // EEVEE divides its sample count across the motion-blur steps, `divide_ceil_u(sample_count, steps)` in
            // `Sampling::init` (`eevee_sampling.cc`). Each step is a full film pass, so the realtime total stays `samples`.
            if engine == "realtime" {
                samples.div_ceil(shutter_samples)
            } else {
                samples
            },
            seed,
        )?;
        push_new_warnings(warnings, sample_warnings);
        let Some(output) = accumulated.as_mut() else {
            accumulated = Some(rendered);
            continue;
        };
        let previous_count = u16::try_from(sample_index).unwrap_or(u16::MAX);
        let divisor_count = previous_count.saturating_add(1);
        let previous = f32::from(previous_count);
        let divisor = f32::from(divisor_count);
        for (destination, source) in output.rgba.iter_mut().zip(rendered.rgba.iter()) {
            let numerator = u16::from(*destination) * previous_count + u16::from(*source);
            *destination = u8::try_from(numerator / divisor_count).unwrap_or(u8::MAX);
        }
        for (destination, source) in output
            .linear_rgba
            .iter_mut()
            .zip(rendered.linear_rgba.iter())
        {
            for channel in 0..4 {
                destination[channel] =
                    (destination[channel] * previous + source[channel]) / divisor;
            }
        }
        for (destination, source) in output
            .ambient_occlusion
            .iter_mut()
            .zip(rendered.ambient_occlusion.iter())
        {
            *destination = (*destination * previous + *source) / divisor;
        }
        if sample_index == center_sample {
            output.ids = rendered.ids;
            output.depths = rendered.depths;
            output.elements = rendered.elements;
            output.normals = rendered.normals;
            output.albedo = rendered.albedo;
            output.emission = rendered.emission;
        }
    }
    let raster = accumulated.ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "motion blur did not produce a frame sample",
        )
    })?;
    Ok((center_snapshot, raster))
}

/// Appends warnings not already recorded, so a warning repeated on every frame is reported once.
fn push_new_warnings(warnings: &mut Vec<Value>, new: Vec<Value>) {
    for warning in new {
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }
}

fn validate_vse_exr_passes(passes: &[String]) -> Result<()> {
    for pass in passes {
        if pass != "combined" {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("VSE EXR pass `{pass}` is not available"),
                json!({"feature_id":format!("render.sequencer.pass.{pass}")}),
            ));
        }
    }
    Ok(())
}

fn associate_straight_frame(mut frame: RgbaFrame) -> Result<RgbaFrame> {
    for pixel in &mut frame.pixels {
        let alpha = pixel[3];
        for channel in &mut pixel[..3] {
            *channel *= alpha;
        }
    }
    RgbaFrame::new(frame.width, frame.height, frame.pixels)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FramePrecision {
    Byte(ImageColorspace),
    Float,
}

impl FramePrecision {
    fn from_image(precision: Precision, colorspace: ImageColorspace) -> Self {
        match precision {
            Precision::Byte => Self::Byte(colorspace),
            Precision::Short | Precision::Half | Precision::Float => Self::Float,
        }
    }

    fn combine(self, other: Self) -> Self {
        if matches!((self, other), (Self::Byte(_), Self::Byte(_))) {
            Self::Byte(ImageColorspace::Srgb)
        } else {
            Self::Float
        }
    }

    const fn is_byte(self) -> bool {
        matches!(self, Self::Byte(_))
    }
}

enum SequencerFrame {
    /// Straight RGBA8 in the byte buffer's native color space.
    Byte {
        width: u32,
        height: u32,
        colorspace: ImageColorspace,
        pixels: Vec<[u8; 4]>,
    },
    /// Associated RGB in the scene's declared VSE working space.
    Float(RgbaFrame),
}

impl SequencerFrame {
    fn from_straight_scene_linear(frame: RgbaFrame, colorspace: ImageColorspace) -> Result<Self> {
        let mut pixels = Vec::with_capacity(frame.pixels.len());
        for mut pixel in frame.pixels {
            for channel in &mut pixel[..3] {
                *channel = scene_to_color_space(*channel, colorspace)?;
            }
            pixels.push([
                byte_channel(pixel[0])?,
                byte_channel(pixel[1])?,
                byte_channel(pixel[2])?,
                byte_channel(pixel[3])?,
            ]);
        }
        Ok(Self::Byte {
            width: frame.width,
            height: frame.height,
            colorspace,
            pixels,
        })
    }

    fn from_associated_float(frame: RgbaFrame, colorspace: ImageColorspace) -> Result<Self> {
        let mut pixels = Vec::with_capacity(frame.pixels.len());
        for pixel in frame.pixels {
            let alpha = pixel[3];
            let straight: [f32; 3] = std::array::from_fn(|channel| {
                if alpha == 0.0 {
                    0.0
                } else {
                    pixel[channel] / alpha
                }
            });
            pixels.push([
                byte_channel(straight[0])?,
                byte_channel(straight[1])?,
                byte_channel(straight[2])?,
                byte_channel(alpha)?,
            ]);
        }
        Ok(Self::Byte {
            width: frame.width,
            height: frame.height,
            colorspace,
            pixels,
        })
    }

    fn from_scene_linear(frame: RgbaFrame, colorspace: SequencerColorSpace) -> Result<Self> {
        Ok(Self::Float(scene_to_sequencer(
            frame,
            colorspace.image_colorspace(),
        )?))
    }

    fn precision(&self) -> FramePrecision {
        match self {
            Self::Byte { colorspace, .. } => FramePrecision::Byte(*colorspace),
            Self::Float(_) => FramePrecision::Float,
        }
    }

    fn into_working(self, working: SequencerColorSpace) -> Result<RgbaFrame> {
        match self {
            Self::Float(frame) => Ok(frame),
            Self::Byte {
                width,
                height,
                colorspace,
                pixels,
            } => {
                let working_colorspace = working.image_colorspace();
                let mut associated = Vec::with_capacity(pixels.len());
                for pixel in pixels {
                    let alpha = byte_float(pixel[3]);
                    let mut output = [0.0; 4];
                    output[3] = alpha;
                    for channel in 0..3 {
                        let scene = color_space_to_scene(byte_float(pixel[channel]), colorspace)?;
                        output[channel] = scene_to_color_space(scene, working_colorspace)? * alpha;
                    }
                    associated.push(output);
                }
                RgbaFrame::new(width, height, associated)
            }
        }
    }

    fn into_scene_linear(self, working: SequencerColorSpace) -> Result<RgbaFrame> {
        match self {
            Self::Float(frame) => sequencer_to_scene(frame, working.image_colorspace()),
            Self::Byte {
                width,
                height,
                colorspace,
                pixels,
            } => {
                let mut associated = Vec::with_capacity(pixels.len());
                for pixel in pixels {
                    let alpha = byte_float(pixel[3]);
                    let mut output = [0.0; 4];
                    output[3] = alpha;
                    for channel in 0..3 {
                        output[channel] =
                            color_space_to_scene(byte_float(pixel[channel]), colorspace)? * alpha;
                    }
                    associated.push(output);
                }
                RgbaFrame::new(width, height, associated)
            }
        }
    }
}

fn scene_to_color_space(value: f32, colorspace: ImageColorspace) -> Result<f32> {
    if colorspace == ImageColorspace::Srgb {
        finite_f64_to_f32(crate::color::linear_to_srgb_unclamped(f64::from(value)))
    } else {
        Ok(value)
    }
}

fn color_space_to_scene(value: f32, colorspace: ImageColorspace) -> Result<f32> {
    if colorspace == ImageColorspace::Srgb {
        finite_f64_to_f32(crate::color::srgb_to_linear(f64::from(value)))
    } else {
        Ok(value)
    }
}

fn scene_to_sequencer(mut frame: RgbaFrame, colorspace: ImageColorspace) -> Result<RgbaFrame> {
    if colorspace == ImageColorspace::Srgb {
        for pixel in &mut frame.pixels {
            let alpha = pixel[3];
            for channel in &mut pixel[..3] {
                let straight = if alpha == 0.0 {
                    *channel
                } else {
                    *channel / alpha
                };
                let encoded = scene_to_color_space(straight, colorspace)?;
                *channel = if alpha == 0.0 {
                    encoded
                } else {
                    encoded * alpha
                };
            }
        }
    }
    RgbaFrame::new(frame.width, frame.height, frame.pixels)
}

fn sequencer_to_scene(mut frame: RgbaFrame, colorspace: ImageColorspace) -> Result<RgbaFrame> {
    if colorspace == ImageColorspace::Srgb {
        for pixel in &mut frame.pixels {
            let alpha = pixel[3];
            for channel in &mut pixel[..3] {
                let straight = if alpha == 0.0 {
                    *channel
                } else {
                    *channel / alpha
                };
                let linear = color_space_to_scene(straight, colorspace)?;
                *channel = if alpha == 0.0 { linear } else { linear * alpha };
            }
        }
    }
    RgbaFrame::new(frame.width, frame.height, frame.pixels)
}

fn byte_channel(value: f32) -> Result<u8> {
    let scaled = value.clamp(0.0, 1.0) * 255.0 + 0.5;
    num_traits::ToPrimitive::to_u8(&scaled).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "VSE byte channel is not representable",
        )
    })
}

fn byte_float(value: u8) -> f32 {
    f32::from(value) / 255.0
}

fn byte_to_associated(width: u32, height: u32, pixels: &[[u8; 4]]) -> Result<RgbaFrame> {
    let associated = pixels
        .iter()
        .map(|pixel| {
            let alpha = byte_float(pixel[3]);
            [
                byte_float(pixel[0]) * alpha,
                byte_float(pixel[1]) * alpha,
                byte_float(pixel[2]) * alpha,
                alpha,
            ]
        })
        .collect();
    RgbaFrame::new(width, height, associated)
}

fn byte_cross_factor(factor: f64) -> Result<u32> {
    let factor = num_traits::ToPrimitive::to_f32(&factor).ok_or_else(|| {
        PotError::invalid_argument("VSE byte transition factor is not representable")
    })?;
    num_traits::ToPrimitive::to_u32(&(256.0_f32 * factor)).ok_or_else(|| {
        PotError::invalid_argument("VSE byte transition factor is outside its supported range")
    })
}

fn byte_cross_channel(first: u8, second: u8, second_weight: u32) -> Result<u8> {
    let value = ((256 - second_weight) * u32::from(first) + second_weight * u32::from(second)) >> 8;
    u8::try_from(value).map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "native VSE byte transition channel exceeded its range",
        )
    })
}

fn byte_crossfade(
    width: u32,
    height: u32,
    first: &[[u8; 4]],
    second: &[[u8; 4]],
    factor: f64,
) -> Result<SequencerFrame> {
    if first.len() != second.len() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frames must have matching nonzero dimensions",
        ));
    }
    let weight = byte_cross_factor(factor)?;
    let pixels = first
        .iter()
        .zip(second)
        .map(|(first, second)| {
            Ok([
                byte_cross_channel(first[0], second[0], weight)?,
                byte_cross_channel(first[1], second[1], weight)?,
                byte_cross_channel(first[2], second[2], weight)?,
                byte_cross_channel(first[3], second[3], weight)?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SequencerFrame::Byte {
        width,
        height,
        colorspace: ImageColorspace::Srgb,
        pixels,
    })
}

fn byte_wipe(
    width: u32,
    height: u32,
    first: &[[u8; 4]],
    second: &[[u8; 4]],
    factor: f64,
) -> Result<SequencerFrame> {
    if first.len() != second.len() {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "frames must have matching nonzero dimensions",
        ));
    }
    let cut = num_traits::ToPrimitive::to_usize(&(f64::from(width) * factor).round())
        .ok_or_else(|| PotError::invalid_argument("VSE wipe cut is outside the platform range"))?;
    let width = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render width exceeds platform range",
        )
    })?;
    let pixels = first
        .iter()
        .zip(second)
        .enumerate()
        .map(
            |(index, (first, second))| {
                if index % width < cut { *second } else { *first }
            },
        )
        .collect();
    Ok(SequencerFrame::Byte {
        width: u32::try_from(width).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "render width exceeds platform range",
            )
        })?,
        height,
        colorspace: ImageColorspace::Srgb,
        pixels,
    })
}

fn byte_gamma_cross(
    width: u32,
    height: u32,
    first: &[[u8; 4]],
    second: &[[u8; 4]],
    factor: f64,
) -> Result<SequencerFrame> {
    let first = byte_to_associated(width, height, first)?;
    let second = byte_to_associated(width, height, second)?;
    let output = transition(&first, &second, factor, TransitionType::GammaCross)?;
    SequencerFrame::from_associated_float(output, ImageColorspace::Srgb)
}

fn byte_effect_factor(factor: f64) -> Result<i32> {
    let factor = num_traits::ToPrimitive::to_f32(&factor)
        .ok_or_else(|| PotError::invalid_argument("VSE byte effect factor is not representable"))?;
    num_traits::ToPrimitive::to_i32(&(256.0_f32 * factor)).ok_or_else(|| {
        PotError::invalid_argument("VSE byte effect factor is outside its supported range")
    })
}

fn byte_add_sub_channel(
    first: u8,
    second: u8,
    second_alpha: u8,
    factor: i32,
    add: bool,
) -> Result<u8> {
    let scaled = factor * i32::from(second_alpha);
    let delta = (scaled * i32::from(second)) >> 16;
    let value = if add {
        (i32::from(first) + delta).min(255)
    } else {
        (i32::from(first) - delta).max(0)
    };
    u8::try_from(value).map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "native VSE Add/Subtract channel exceeded its range",
        )
    })
}

fn byte_multiply_channel(first: u8, second: u8, factor: i32) -> Result<u8> {
    let first = i32::from(first);
    let second = i32::from(second);
    u8::try_from((first + ((factor * first * (second - 255)) >> 16)).clamp(0, 255)).map_err(|_| {
        PotError::new(
            ErrorCode::InternalError,
            "native VSE Multiply channel exceeded its range",
        )
    })
}

fn byte_blend_effect(
    effect: EffectType,
    first: [u8; 4],
    second: [u8; 4],
    factor: f64,
) -> Result<[u8; 4]> {
    let factor = byte_effect_factor(factor)?;
    Ok(match effect {
        EffectType::Add | EffectType::Subtract => [
            byte_add_sub_channel(
                first[0],
                second[0],
                second[3],
                factor,
                effect == EffectType::Add,
            )?,
            byte_add_sub_channel(
                first[1],
                second[1],
                second[3],
                factor,
                effect == EffectType::Add,
            )?,
            byte_add_sub_channel(
                first[2],
                second[2],
                second[3],
                factor,
                effect == EffectType::Add,
            )?,
            first[3],
        ],
        EffectType::Multiply => [
            byte_multiply_channel(first[0], second[0], factor)?,
            byte_multiply_channel(first[1], second[1], factor)?,
            byte_multiply_channel(first[2], second[2], factor)?,
            byte_multiply_channel(first[3], second[3], factor)?,
        ],
        _ => unreachable!("only byte Add/Subtract/Multiply use this kernel"),
    })
}

fn byte_effect_frame(
    effect: EffectType,
    first: SequencerFrame,
    second: SequencerFrame,
    factor: f64,
) -> Result<SequencerFrame> {
    match (first, second) {
        (
            SequencerFrame::Byte {
                width,
                height,
                pixels: first,
                ..
            },
            SequencerFrame::Byte { pixels: second, .. },
        ) if matches!(
            effect,
            EffectType::Add | EffectType::Subtract | EffectType::Multiply
        ) =>
        {
            if first.len() != second.len() {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "frames must have matching nonzero dimensions",
                ));
            }
            Ok(SequencerFrame::Byte {
                width,
                height,
                colorspace: ImageColorspace::Srgb,
                pixels: first
                    .iter()
                    .zip(&second)
                    .map(|(first, second)| byte_blend_effect(effect, *first, *second, factor))
                    .collect::<Result<Vec<_>>>()?,
            })
        }
        (first, second) => {
            let (
                SequencerFrame::Byte {
                    width,
                    height,
                    pixels: first_pixels,
                    ..
                },
                SequencerFrame::Byte {
                    pixels: second_pixels,
                    ..
                },
            ) = (first, second)
            else {
                return Err(PotError::new(
                    ErrorCode::InvalidArgument,
                    "byte effect inputs must both have byte storage",
                ));
            };
            let first = byte_to_associated(width, height, &first_pixels)?;
            let second = byte_to_associated(width, height, &second_pixels)?;
            let output = apply_blend_effect(EffectType::AlphaOver, &first, &second, factor)?;
            SequencerFrame::from_associated_float(output, ImageColorspace::Srgb)
        }
    }
}

fn divide_round_i(numerator: i32, denominator: i32) -> i32 {
    let half = denominator / 2;
    if numerator < 0 {
        -((-numerator + half) / denominator)
    } else {
        (numerator + half) / denominator
    }
}

fn byte_alpha_with_opacity(alpha: u8, opacity: f64) -> Result<u8> {
    let opacity = num_traits::ToPrimitive::to_f32(&opacity)
        .ok_or_else(|| PotError::invalid_argument("VSE byte opacity is not representable"))?;
    let scaled = f32::from(alpha) * opacity;
    num_traits::ToPrimitive::to_u8(&scaled).ok_or_else(|| {
        PotError::invalid_argument("VSE byte opacity is outside its supported range")
    })
}

fn byte_strip_blend_pixel(
    destination: [u8; 4],
    source: [u8; 4],
    blend_type: StripBlendType,
    opacity: f64,
) -> Result<[u8; 4]> {
    let alpha = byte_alpha_with_opacity(source[3], opacity)?;
    match blend_type {
        StripBlendType::Add | StripBlendType::Subtract => {
            let add = blend_type == StripBlendType::Add;
            let channel = |index| {
                let numerator = i32::from(destination[index]) * 255
                    + if add {
                        i32::from(source[index]) * i32::from(alpha)
                    } else {
                        -i32::from(source[index]) * i32::from(alpha)
                    };
                u8::try_from(divide_round_i(numerator, 255).clamp(0, 255)).map_err(|_| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "native VSE strip channel exceeded its range",
                    )
                })
            };
            Ok([channel(0)?, channel(1)?, channel(2)?, destination[3]])
        }
        StripBlendType::Multiply if alpha != 0 => {
            let channel = |index| {
                let inverse = 255 - i32::from(alpha);
                let destination = i32::from(destination[index]);
                let source = i32::from(source[index]);
                let numerator =
                    inverse * destination * 255 + i32::from(alpha) * destination * source;
                u8::try_from(divide_round_i(numerator, 255 * 255).clamp(0, 255)).map_err(|_| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "native VSE strip channel exceeded its range",
                    )
                })
            };
            Ok([channel(0)?, channel(1)?, channel(2)?, destination[3]])
        }
        StripBlendType::Multiply | StripBlendType::AlphaOver | StripBlendType::Replace => {
            Ok(destination)
        }
    }
}

fn multiply_sequencer_frame(
    frame: SequencerFrame,
    multiplier: f64,
    multiply_alpha: bool,
) -> Result<SequencerFrame> {
    match frame {
        SequencerFrame::Byte {
            width,
            height,
            colorspace,
            mut pixels,
        } => {
            let multiplier = byte_effect_factor(multiplier)?;
            for pixel in &mut pixels {
                for channel in &mut pixel[..3] {
                    *channel =
                        u8::try_from(((multiplier * i32::from(*channel)) >> 8).clamp(0, 255))
                            .map_err(|_| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "native VSE multiplier channel exceeded its range",
                                )
                            })?;
                }
                if multiply_alpha {
                    pixel[3] =
                        u8::try_from(((multiplier * i32::from(pixel[3])) >> 8).clamp(0, 255))
                            .map_err(|_| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "native VSE multiplier channel exceeded its range",
                                )
                            })?;
                }
            }
            Ok(SequencerFrame::Byte {
                width,
                height,
                colorspace,
                pixels,
            })
        }
        SequencerFrame::Float(mut frame) => {
            let multiplier = finite_f64_to_f32(multiplier)?;
            for pixel in &mut frame.pixels {
                for channel in &mut pixel[..3] {
                    *channel *= multiplier;
                }
                if multiply_alpha {
                    pixel[3] *= multiplier;
                }
            }
            Ok(SequencerFrame::Float(RgbaFrame::new(
                frame.width,
                frame.height,
                frame.pixels,
            )?))
        }
    }
}

fn preprocess_strip_frame(
    frame: SequencerFrame,
    strip: &Strip,
    working: SequencerColorSpace,
) -> Result<SequencerFrame> {
    let multiplier = strip.color_multiply
        * if strip.blend_type == StripBlendType::Replace {
            strip.opacity
        } else {
            1.0
        };
    if multiplier.to_bits() == 1.0_f64.to_bits() {
        return Ok(frame);
    }
    let needs_promote = match &frame {
        SequencerFrame::Byte { colorspace, .. } => *colorspace != working.image_colorspace(),
        SequencerFrame::Float(_) => false,
    };
    if needs_promote {
        return multiply_sequencer_frame(
            SequencerFrame::Float(frame.into_working(working)?),
            multiplier,
            strip.multiply_alpha,
        );
    }
    multiply_sequencer_frame(frame, multiplier, strip.multiply_alpha)
}

fn blend_sequence_frames(
    destination: SequencerFrame,
    source: SequencerFrame,
    blend_type: StripBlendType,
    opacity: f64,
    working: SequencerColorSpace,
) -> Result<SequencerFrame> {
    if blend_type == StripBlendType::Replace {
        return Ok(source);
    }
    let precision = destination.precision().combine(source.precision());
    if precision.is_byte() {
        let (
            SequencerFrame::Byte {
                width,
                height,
                pixels: destination_pixels,
                ..
            },
            SequencerFrame::Byte {
                pixels: source_pixels,
                ..
            },
        ) = (destination, source)
        else {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "byte sequencer composite has a non-byte input",
            ));
        };
        if destination_pixels.len() != source_pixels.len() {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "frames must have matching nonzero dimensions",
            ));
        }
        if matches!(
            blend_type,
            StripBlendType::Add | StripBlendType::Subtract | StripBlendType::Multiply
        ) {
            return Ok(SequencerFrame::Byte {
                width,
                height,
                colorspace: ImageColorspace::Srgb,
                pixels: destination_pixels
                    .iter()
                    .zip(&source_pixels)
                    .map(|(destination, source)| {
                        byte_strip_blend_pixel(*destination, *source, blend_type, opacity)
                    })
                    .collect::<Result<Vec<_>>>()?,
            });
        }
        let destination = byte_to_associated(width, height, &destination_pixels)?;
        let source = byte_to_associated(width, height, &source_pixels)?;
        let output = blend_frames(&destination, &source, blend_type, opacity)?;
        return SequencerFrame::from_associated_float(output, ImageColorspace::Srgb);
    }
    let destination = destination.into_working(working)?;
    let source = source.into_working(working)?;
    Ok(SequencerFrame::Float(blend_frames(
        &destination,
        &source,
        blend_type,
        opacity,
    )?))
}

#[expect(
    clippy::too_many_arguments,
    reason = "VSE output depends on resolved scene, render, and output settings"
)]
fn render_vse_frame(
    doc: &SceneDoc,
    scene: &Scene,
    frame: f64,
    width: u32,
    height: u32,
    root: &Path,
    cache_directory: &Path,
    session: &mut EvaluationSession,
    engine: &str,
    samples: u32,
    seed: u32,
    film_transparent: bool,
    warnings: &mut Vec<Value>,
) -> Result<RgbaFrame> {
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render height exceeds platform limits",
        )
    })?;
    let count = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "render dimensions overflow"))?;
    let background_alpha = if film_transparent { 0 } else { 255 };
    let mut composite = SequencerFrame::Byte {
        width,
        height,
        colorspace: ImageColorspace::Srgb,
        pixels: vec![[0, 0, 0, background_alpha]; count],
    };
    let mut nested_ids = BTreeSet::new();
    for strip in &scene.sequencer.strips {
        if strip.kind == StripType::Meta {
            nested_ids.extend(strip.inputs.iter().map(String::as_str));
        }
    }
    let mut consumed_ids = nested_ids.clone();
    for strip in &scene.sequencer.strips {
        if !strip.mute
            && strip.contains_frame(frame)
            && matches!(strip.kind, StripType::Transition | StripType::Effect)
        {
            consumed_ids.extend(strip.inputs.iter().map(String::as_str));
        }
    }
    let mut strips: Vec<&Strip> = scene
        .sequencer
        .strips
        .iter()
        .filter(|strip| {
            !strip.mute
                && strip.contains_frame(frame)
                && strip.kind != StripType::Sound
                && !consumed_ids.contains(strip.id.as_str())
        })
        .collect();
    strips.sort_by_key(|strip| strip.channel);
    for strip in strips {
        let mut visiting = BTreeSet::new();
        if let Some(layer) = render_strip_frame(
            doc,
            scene,
            root,
            cache_directory,
            session,
            strip,
            frame,
            width,
            height,
            engine,
            samples,
            seed,
            film_transparent,
            false,
            &mut visiting,
            warnings,
        )? {
            composite = blend_sequence_frames(
                composite,
                layer,
                strip.blend_type,
                strip.opacity,
                scene.sequencer.colorspace,
            )?;
        }
    }
    let composite = composite.into_scene_linear(scene.sequencer.colorspace)?;
    let input = CompositorImage::new(width, height, composite.pixels)?;
    let output = compositor::apply_associated_image(input, doc, scene, frame)?;
    RgbaFrame::new(output.width, output.height, output.pixels)
}

#[expect(
    clippy::too_many_arguments,
    reason = "strip rendering carries immutable scene and output context through recursive inputs"
)]
fn render_strip_frame(
    doc: &SceneDoc,
    scene: &Scene,
    root: &Path,
    cache_directory: &Path,
    session: &mut EvaluationSession,
    strip: &Strip,
    frame: f64,
    width: u32,
    height: u32,
    engine: &str,
    samples: u32,
    seed: u32,
    film_transparent: bool,
    allow_outside_range: bool,
    visiting: &mut BTreeSet<String>,
    warnings: &mut Vec<Value>,
) -> Result<Option<SequencerFrame>> {
    if strip.mute || (!allow_outside_range && !strip.contains_frame(frame)) {
        return Ok(None);
    }
    if !visiting.insert(strip.id.clone()) {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "sequencer input strips contain a cycle",
            json!({"strip_id":strip.id}),
        ));
    }
    let result = render_strip_content(
        doc,
        scene,
        root,
        cache_directory,
        session,
        strip,
        frame,
        width,
        height,
        engine,
        samples,
        seed,
        film_transparent,
        visiting,
        warnings,
    );
    visiting.remove(&strip.id);
    let Some(frame) = result? else {
        return Ok(None);
    };
    let frame = preprocess_strip_frame(frame, strip, scene.sequencer.colorspace)?;
    if strip.modifiers.is_empty() {
        return Ok(Some(frame));
    }
    let frame = frame.into_working(scene.sequencer.colorspace)?;
    Ok(Some(SequencerFrame::Float(apply_modifiers(
        &frame,
        &strip.modifiers,
    )?)))
}

#[expect(
    clippy::too_many_arguments,
    reason = "strip kinds share the resolved render and recursive graph context"
)]
fn render_strip_content(
    doc: &SceneDoc,
    scene: &Scene,
    root: &Path,
    cache_directory: &Path,
    session: &mut EvaluationSession,
    strip: &Strip,
    frame: f64,
    width: u32,
    height: u32,
    engine: &str,
    samples: u32,
    seed: u32,
    film_transparent: bool,
    visiting: &mut BTreeSet<String>,
    warnings: &mut Vec<Value>,
) -> Result<Option<SequencerFrame>> {
    let result = match strip.kind {
        StripType::Image | StripType::ImageSequence => {
            let source = strip.source.as_deref().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "image strip requires a source",
                    json!({"strip_id":strip.id}),
                )
            })?;
            Some(load_strip_image(
                doc,
                root,
                source,
                strip.source_frame(frame),
                width,
                height,
                scene.sequencer.colorspace,
            )?)
        }
        StripType::Movie => {
            return Err(PotError::with_details(
                ErrorCode::DependencyMissing,
                "movie strips require a pure-Rust movie decoder, which is not available",
                json!({"feature_id":"sequencer.strip.movie","source":strip.source}),
            ));
        }
        StripType::Sound => None,
        StripType::Scene => {
            let source = strip.source.as_deref().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "scene strip requires a scene ID source",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let scene_id = Id::new(source.to_owned())
                .map_err(|error| PotError::new(ErrorCode::InvalidArgument, error.message))?;
            let nested_scene = doc.scenes.get(&scene_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "scene strip references a missing scene",
                    json!({"scene_id":scene_id}),
                )
            })?;
            let camera_id = nested_scene
                .camera
                .as_ref()
                .map(Id::as_str)
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::TargetNotFound,
                        "scene strip source has no camera",
                        json!({"scene_id":scene_id}),
                    )
                })?;
            let nested_context = EvaluationContext {
                scene_id: Some(scene_id),
                frame: Some(strip.source_frame(frame)),
                purpose: EvaluationPurpose::Render,
                ..EvaluationContext::default()
            };
            let (_, raster, strip_warnings) = render_camera_frame(
                doc,
                &nested_context,
                session,
                root,
                cache_directory,
                camera_id,
                width,
                height,
                "beauty",
                engine,
                film_transparent,
                samples,
                seed,
            )?;
            push_new_warnings(warnings, strip_warnings);
            let associated_alpha = raster.linear_rgba_associated;
            let pixels = RgbaFrame::new(width, height, raster.linear_rgba)?;
            let pixels = if associated_alpha {
                pixels
            } else {
                associate_straight_frame(pixels)?
            };
            Some(SequencerFrame::from_scene_linear(
                pixels,
                scene.sequencer.colorspace,
            )?)
        }
        StripType::Color => {
            let width_usize = usize::try_from(width).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "render width exceeds platform limits",
                )
            })?;
            let height_usize = usize::try_from(height).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "render height exceeds platform limits",
                )
            })?;
            let count = width_usize.checked_mul(height_usize).ok_or_else(|| {
                PotError::new(ErrorCode::LimitExceeded, "render dimensions overflow")
            })?;
            let pixels = RgbaFrame::new(
                width,
                height,
                vec![[strip.color[0], strip.color[1], strip.color[2], 1.0]; count],
            )?;
            Some(SequencerFrame::from_straight_scene_linear(
                pixels,
                ImageColorspace::Srgb,
            )?)
        }
        StripType::Text => {
            let text = strip.text.as_deref().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "text strip requires text content",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let pixels = render_text_frame(text, strip.color, width, height)?;
            Some(SequencerFrame::from_straight_scene_linear(
                pixels,
                ImageColorspace::Srgb,
            )?)
        }
        StripType::Meta => Some(compose_meta_frame(
            doc,
            scene,
            root,
            cache_directory,
            session,
            strip,
            strip.source_frame(frame),
            width,
            height,
            engine,
            samples,
            seed,
            film_transparent,
            visiting,
            warnings,
        )?),
        StripType::Transition => {
            let transition_kind = strip.transition.ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "transition strip requires a transition type",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let first_id = strip.inputs.first().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "transition strip requires two input strips",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let second_id = strip.inputs.get(1).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "transition strip requires two input strips",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let first = render_strip_input(
                doc,
                scene,
                root,
                cache_directory,
                session,
                first_id,
                frame,
                width,
                height,
                engine,
                samples,
                seed,
                film_transparent,
                visiting,
                warnings,
            )?;
            let second = render_strip_input(
                doc,
                scene,
                root,
                cache_directory,
                session,
                second_id,
                frame,
                width,
                height,
                engine,
                samples,
                seed,
                film_transparent,
                visiting,
                warnings,
            )?;
            let factor = ((frame - strip.frame_start) / strip.length).clamp(0.0, 1.0);
            let precision = first.precision().combine(second.precision());
            let output = if precision.is_byte() {
                match (first, second) {
                    (
                        SequencerFrame::Byte {
                            width,
                            height,
                            pixels: first,
                            ..
                        },
                        SequencerFrame::Byte { pixels: second, .. },
                    ) => match transition_kind {
                        TransitionType::Cross => {
                            byte_crossfade(width, height, &first, &second, factor)
                        }
                        TransitionType::GammaCross => {
                            byte_gamma_cross(width, height, &first, &second, factor)
                        }
                        TransitionType::Wipe => byte_wipe(width, height, &first, &second, factor),
                    },
                    _ => Err(PotError::new(
                        ErrorCode::InternalError,
                        "byte transition has a non-byte input",
                    )),
                }
            } else {
                let first = first.into_working(scene.sequencer.colorspace)?;
                let second = second.into_working(scene.sequencer.colorspace)?;
                transition(&first, &second, factor, transition_kind).map(SequencerFrame::Float)
            };
            Some(output?)
        }
        StripType::Effect => {
            let effect = strip.effect.ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "effect strip requires an effect type",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let input_frame = if effect == EffectType::Speed {
                strip.source_frame(frame)
            } else {
                frame
            };
            let first_id = strip.inputs.first().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "effect strip requires an input strip",
                    json!({"strip_id":strip.id}),
                )
            })?;
            let first = render_strip_input(
                doc,
                scene,
                root,
                cache_directory,
                session,
                first_id,
                input_frame,
                width,
                height,
                engine,
                samples,
                seed,
                film_transparent,
                visiting,
                warnings,
            )?;
            if matches!(
                effect,
                EffectType::Add
                    | EffectType::Subtract
                    | EffectType::Multiply
                    | EffectType::AlphaOver
            ) {
                let second_id = strip.inputs.get(1).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InvalidOperation,
                        "two-input blend effect requires two input strips",
                        json!({"strip_id":strip.id}),
                    )
                })?;
                let second = render_strip_input(
                    doc,
                    scene,
                    root,
                    cache_directory,
                    session,
                    second_id,
                    input_frame,
                    width,
                    height,
                    engine,
                    samples,
                    seed,
                    film_transparent,
                    visiting,
                    warnings,
                )?;
                let precision = first.precision().combine(second.precision());
                if precision.is_byte() {
                    Some(byte_effect_frame(effect, first, second, 1.0)?)
                } else {
                    let first = first.into_working(scene.sequencer.colorspace)?;
                    let second = second.into_working(scene.sequencer.colorspace)?;
                    let output = apply_blend_effect(effect, &first, &second, 1.0)?;
                    Some(SequencerFrame::Float(output))
                }
            } else if effect == EffectType::Speed {
                Some(first)
            } else {
                match first {
                    SequencerFrame::Byte {
                        width,
                        height,
                        colorspace: _,
                        pixels,
                    } => {
                        let associated = byte_to_associated(width, height, &pixels)?;
                        let output = apply_effect(effect, associated)?;
                        Some(SequencerFrame::from_associated_float(
                            output,
                            ImageColorspace::Srgb,
                        )?)
                    }
                    SequencerFrame::Float(frame) => {
                        Some(SequencerFrame::Float(apply_effect(effect, frame)?))
                    }
                }
            }
        }
    };
    Ok(result)
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive strip input rendering carries the current output context"
)]
fn render_strip_input(
    doc: &SceneDoc,
    scene: &Scene,
    root: &Path,
    cache_directory: &Path,
    session: &mut EvaluationSession,
    strip_id: &str,
    frame: f64,
    width: u32,
    height: u32,
    engine: &str,
    samples: u32,
    seed: u32,
    film_transparent: bool,
    visiting: &mut BTreeSet<String>,
    warnings: &mut Vec<Value>,
) -> Result<SequencerFrame> {
    let strip = scene
        .sequencer
        .strips
        .iter()
        .find(|strip| strip.id == strip_id)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "sequencer input strip does not exist",
                json!({"strip_id":strip_id}),
            )
        })?;
    render_strip_frame(
        doc,
        scene,
        root,
        cache_directory,
        session,
        strip,
        frame,
        width,
        height,
        engine,
        samples,
        seed,
        film_transparent,
        true,
        visiting,
        warnings,
    )?
    .ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "sequencer input strip has no video output",
            json!({"strip_id":strip_id}),
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "meta strips evaluate their child strip graph at the selected frame"
)]
fn compose_meta_frame(
    doc: &SceneDoc,
    scene: &Scene,
    root: &Path,
    cache_directory: &Path,
    session: &mut EvaluationSession,
    meta: &Strip,
    frame: f64,
    width: u32,
    height: u32,
    engine: &str,
    samples: u32,
    seed: u32,
    film_transparent: bool,
    visiting: &mut BTreeSet<String>,
    warnings: &mut Vec<Value>,
) -> Result<SequencerFrame> {
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render height exceeds platform limits",
        )
    })?;
    let count = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "render dimensions overflow"))?;
    let mut composite = SequencerFrame::Byte {
        width,
        height,
        colorspace: ImageColorspace::Srgb,
        pixels: vec![[0, 0, 0, 0]; count],
    };
    let mut children = Vec::with_capacity(meta.inputs.len());
    for input_id in &meta.inputs {
        let child = scene
            .sequencer
            .strips
            .iter()
            .find(|strip| strip.id == *input_id)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "meta strip references a missing child strip",
                    json!({"strip_id":meta.id,"input":input_id}),
                )
            })?;
        children.push(child);
    }
    children.sort_by_key(|child| child.channel);
    for child in children {
        if let Some(layer) = render_strip_frame(
            doc,
            scene,
            root,
            cache_directory,
            session,
            child,
            frame,
            width,
            height,
            engine,
            samples,
            seed,
            film_transparent,
            true,
            visiting,
            warnings,
        )? {
            composite = blend_sequence_frames(
                composite,
                layer,
                child.blend_type,
                child.opacity,
                scene.sequencer.colorspace,
            )?;
        }
    }
    Ok(composite)
}

fn load_strip_image(
    doc: &SceneDoc,
    root: &Path,
    source: &str,
    source_frame: f64,
    width: u32,
    height: u32,
    working: SequencerColorSpace,
) -> Result<SequencerFrame> {
    if let Ok(image_id) = Id::new(source.to_owned())
        && let Some(image) = doc.images.get(&image_id)
    {
        let data = storage::load_image_data(image, root, ImageInterpolation::Linear)?;
        let precision = FramePrecision::from_image(data.storage.precision, image.colorspace);
        return match precision {
            FramePrecision::Byte(colorspace) => SequencerFrame::from_straight_scene_linear(
                resize_image_straight(&data, width, height)?,
                colorspace,
            ),
            FramePrecision::Float => {
                SequencerFrame::from_scene_linear(resize_image_data(&data, width, height)?, working)
            }
        };
    }
    let path = resolve_image_sequence_path(root, source, source_frame)?;
    let bytes = fs::read(&path).map_err(|error| PotError::io(&error))?;
    let (colorspace, image_width, image_height, pixels) =
        storage::decode_pixels_with_default_colorspace(&bytes, None)?;
    let precision = storage::source_precision(&bytes);
    let data = ImageData {
        colour: crate::image::ColourCache::default(),
        storage: crate::image::ImageStorage {
            straight_alpha: storage::default_alpha_mode(&bytes)
                == crate::model::ImageAlphaMode::Straight,
            precision,
            ..crate::image::ImageStorage::default()
        },
        width: image_width,
        height: image_height,
        srgb: storage::decodes_srgb(colorspace),
        udim: false,
        pixels,
        tiles: std::collections::BTreeMap::new(),
        interpolation: ImageInterpolation::Linear,
    };
    match FramePrecision::from_image(precision, colorspace) {
        FramePrecision::Byte(colorspace) => SequencerFrame::from_straight_scene_linear(
            resize_image_straight(&data, width, height)?,
            colorspace,
        ),
        FramePrecision::Float => {
            SequencerFrame::from_scene_linear(resize_image_data(&data, width, height)?, working)
        }
    }
}

fn resize_image_straight(data: &ImageData, width: u32, height: u32) -> Result<RgbaFrame> {
    let mut frame = resample_image(data, width, height, false)?;
    if !data.storage.straight_alpha && !data.storage.ignore_alpha && !data.storage.data {
        for pixel in &mut frame.pixels {
            let alpha = pixel[3];
            if alpha != 0.0 {
                for channel in &mut pixel[..3] {
                    *channel /= alpha;
                }
            }
        }
    }
    RgbaFrame::new(frame.width, frame.height, frame.pixels)
}

fn resize_image_data(data: &ImageData, width: u32, height: u32) -> Result<RgbaFrame> {
    if data.width == 0 || data.height == 0 || width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "image strip source and output dimensions must be non-zero",
        ));
    }
    // Straight inputs are associated after sampling; premultiplied inputs stay in their stored form,
    // including RGB at alpha zero.
    let associate = data.storage.straight_alpha && !data.storage.ignore_alpha;
    resample_image(data, width, height, associate)
}

fn resample_image(
    source: &ImageData,
    width: u32,
    height: u32,
    associate: bool,
) -> Result<RgbaFrame> {
    let count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "image strip output is too large")
        })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(count)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "image strip allocation failed"))?;
    for y in 0..height {
        for x in 0..width {
            let sampled = sample(
                source,
                [
                    (f64::from(x) + 0.5) / f64::from(width),
                    (f64::from(y) + 0.5) / f64::from(height),
                ],
                false,
            );
            let alpha = sampled[3];
            pixels.push([
                finite_f64_to_f32(if associate {
                    sampled[0] * alpha
                } else {
                    sampled[0]
                })?,
                finite_f64_to_f32(if associate {
                    sampled[1] * alpha
                } else {
                    sampled[1]
                })?,
                finite_f64_to_f32(if associate {
                    sampled[2] * alpha
                } else {
                    sampled[2]
                })?,
                finite_f64_to_f32(alpha)?,
            ]);
        }
    }
    RgbaFrame::new(width, height, pixels)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is checked against the finite f32 range before conversion"
)]
fn finite_f64_to_f32(value: f64) -> Result<f32> {
    if !value.is_finite() || value < -f64::from(f32::MAX) || value > f64::from(f32::MAX) {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            "image strip pixel cannot be represented as f32",
        ));
    }
    Ok(value as f32)
}

fn resolve_image_sequence_path(root: &Path, source: &str, frame: f64) -> Result<PathBuf> {
    let frame_number = rounded_frame(frame)?;
    let plain = frame_number.to_string();
    let mut formatted = source.replace("{frame}", &plain);
    for width in 1..=12 {
        let token = format!("{{frame:{width:02}}}");
        if formatted.contains(&token) {
            formatted = formatted.replace(&token, &format!("{frame_number:0width$}"));
        }
        let percent_token = format!("%0{width}d");
        if formatted.contains(&percent_token) {
            formatted = formatted.replace(&percent_token, &format!("{frame_number:0width$}"));
        }
    }
    if let Some(start) = formatted.find('#') {
        let end = formatted[start..]
            .find(|character| character != '#')
            .map_or(formatted.len(), |offset| start + offset);
        let padding = end - start;
        formatted.replace_range(start..end, &format!("{frame_number:0padding$}"));
    }
    let source_path = Path::new(&formatted);
    if source_path.is_absolute() {
        Ok(source_path.to_path_buf())
    } else {
        Ok(root.join(source_path))
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sequence frames are rounded and constrained to the i64 range"
)]
fn rounded_frame(frame: f64) -> Result<i64> {
    let rounded = frame.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= -(i64::MIN as f64) {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "image sequence frame exceeds the supported range",
        ));
    }
    Ok(rounded as i64)
}

fn render_text_frame(text: &str, color: [f32; 4], width: u32, height: u32) -> Result<RgbaFrame> {
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render height exceeds platform limits",
        )
    })?;
    let count = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "render dimensions overflow"))?;
    let mut pixels = vec![[0.0; 4]; count];
    let mut origin_x = 0_usize;
    let mut origin_y = 0_usize;
    for byte in text.bytes() {
        if byte == b'\n' {
            origin_x = 0;
            origin_y = origin_y.saturating_add(9);
            continue;
        }
        let glyph = glyph_rows(byte).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "built-in sequencer text rasterizer supports ASCII glyphs only",
                json!({"feature_id":"sequencer.strip.text.unicode"}),
            )
        })?;
        for (row, bits) in glyph.iter().copied().enumerate() {
            let y = origin_y.saturating_add(row);
            if y >= height_usize {
                continue;
            }
            for column in 0..5 {
                let x = origin_x.saturating_add(column);
                if x < width_usize && bits & (1 << (4 - column)) != 0 {
                    pixels[y * width_usize + x] = color;
                }
            }
        }
        origin_x = origin_x.saturating_add(6);
        if origin_x >= width_usize {
            break;
        }
    }
    RgbaFrame::new(width, height, pixels)
}

fn glyph_rows(byte: u8) -> Option<[u8; 7]> {
    let character = byte.to_ascii_uppercase();
    Some(match character {
        b' ' => [0, 0, 0, 0, 0, 0, 0],
        b'0' => [14, 17, 19, 21, 25, 17, 14],
        b'1' => [4, 12, 4, 4, 4, 4, 14],
        b'2' => [14, 17, 1, 2, 4, 8, 31],
        b'3' => [30, 1, 1, 14, 1, 1, 30],
        b'4' => [2, 6, 10, 18, 31, 2, 2],
        b'5' => [31, 16, 16, 30, 1, 1, 30],
        b'6' => [14, 16, 16, 30, 17, 17, 14],
        b'7' => [31, 1, 2, 4, 8, 8, 8],
        b'8' => [14, 17, 17, 14, 17, 17, 14],
        b'9' => [14, 17, 17, 15, 1, 1, 14],
        b'A' => [14, 17, 17, 31, 17, 17, 17],
        b'B' => [30, 17, 17, 30, 17, 17, 30],
        b'C' => [14, 17, 16, 16, 16, 17, 14],
        b'D' => [30, 17, 17, 17, 17, 17, 30],
        b'E' => [31, 16, 16, 30, 16, 16, 31],
        b'F' => [31, 16, 16, 30, 16, 16, 16],
        b'G' => [14, 17, 16, 23, 17, 17, 15],
        b'H' => [17, 17, 17, 31, 17, 17, 17],
        b'I' => [14, 4, 4, 4, 4, 4, 14],
        b'J' => [7, 2, 2, 2, 2, 18, 12],
        b'K' => [17, 18, 20, 24, 20, 18, 17],
        b'L' => [16, 16, 16, 16, 16, 16, 31],
        b'M' => [17, 27, 21, 21, 17, 17, 17],
        b'N' => [17, 25, 21, 19, 17, 17, 17],
        b'O' => [14, 17, 17, 17, 17, 17, 14],
        b'P' => [30, 17, 17, 30, 16, 16, 16],
        b'Q' => [14, 17, 17, 17, 21, 18, 13],
        b'R' => [30, 17, 17, 30, 20, 18, 17],
        b'S' => [15, 16, 16, 14, 1, 1, 30],
        b'T' => [31, 4, 4, 4, 4, 4, 4],
        b'U' => [17, 17, 17, 17, 17, 17, 14],
        b'V' => [17, 17, 17, 17, 17, 10, 4],
        b'W' => [17, 17, 17, 21, 21, 21, 10],
        b'X' => [17, 17, 10, 4, 10, 17, 17],
        b'Y' => [17, 17, 10, 4, 4, 4, 4],
        b'Z' => [31, 1, 2, 4, 8, 16, 31],
        b'.' => [0, 0, 0, 0, 0, 12, 12],
        b',' => [0, 0, 0, 0, 4, 4, 8],
        b'!' => [4, 4, 4, 4, 4, 0, 4],
        b'?' => [14, 17, 1, 2, 4, 0, 4],
        b':' => [0, 12, 12, 0, 12, 12, 0],
        b';' => [0, 12, 12, 0, 4, 4, 8],
        b'-' => [0, 0, 0, 31, 0, 0, 0],
        b'_' => [0, 0, 0, 0, 0, 0, 31],
        b'+' => [0, 4, 4, 31, 4, 4, 0],
        b'/' => [1, 2, 2, 4, 8, 8, 16],
        b'\\' => [16, 8, 8, 4, 2, 2, 1],
        b'(' => [2, 4, 8, 8, 8, 4, 2],
        b')' => [8, 4, 2, 2, 2, 4, 8],
        b'=' => [0, 31, 0, 31, 0, 0, 0],
        b'\'' => [4, 4, 2, 0, 0, 0, 0],
        b'"' => [10, 10, 5, 0, 0, 0, 0],
        b'%' => [17, 2, 4, 8, 17, 0, 0],
        b'&' => [12, 18, 20, 8, 21, 18, 13],
        b'*' => [0, 21, 14, 31, 14, 21, 0],
        _ => return None,
    })
}

fn render_vse_audio(scene: &Scene, root: &Path, frames: &[f64]) -> Result<AudioBuffer> {
    let fps_base = scene.fps_base;
    let fps = f64::from(scene.fps) / fps_base;
    if !fps_base.is_finite() || fps_base <= 0.0 || !fps.is_finite() || fps <= 0.0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "scene frame rate is invalid for sequencer audio",
        ));
    }
    let frame_start = frames
        .first()
        .copied()
        .ok_or_else(|| PotError::invalid_argument("audio render requires at least one frame"))?;
    let frame_end =
        frames.last().copied().ok_or_else(|| {
            PotError::invalid_argument("audio render requires at least one frame")
        })? + 1.0;
    let mut nested_ids = BTreeSet::new();
    for strip in &scene.sequencer.strips {
        if strip.kind == StripType::Meta {
            nested_ids.extend(strip.inputs.iter().map(String::as_str));
        }
    }
    let mut sound_strips = Vec::new();
    for strip in &scene.sequencer.strips {
        if !nested_ids.contains(strip.id.as_str())
            && matches!(strip.kind, StripType::Sound | StripType::Meta)
        {
            gather_sound_strips(scene, strip, &mut BTreeSet::new(), &mut sound_strips)?;
        }
    }
    let mut audio_sources = Vec::with_capacity(sound_strips.len());
    for strip in sound_strips {
        let source = strip.source.as_deref().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "sound strip requires an audio source path",
                json!({"strip_id":strip.id}),
            )
        })?;
        let path = resolve_source_path(root, source);
        let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
        let audio = decode_wav(&bytes)?;
        audio_sources.push((audio, strip));
    }
    let tracks: Vec<AudioTrack<'_>> = audio_sources
        .iter()
        .map(|(audio, strip)| AudioTrack {
            audio,
            frame_start: strip.frame_start,
            frame_end: Some(strip.frame_start + strip.length),
            frame_offset_start: strip.frame_offset_start,
            frame_offset_end: strip.frame_offset_end,
            retiming_keys: &strip.retiming_keys,
            volume: strip.sound_volume,
            pan: strip.sound_pan,
            pitch: strip.sound_pitch,
        })
        .collect();
    mix_audio(&tracks, frame_start, frame_end, fps, 48_000)
}

fn gather_sound_strips<'a>(
    scene: &'a Scene,
    strip: &'a Strip,
    visiting: &mut BTreeSet<String>,
    output: &mut Vec<&'a Strip>,
) -> Result<()> {
    if strip.mute {
        return Ok(());
    }
    if !visiting.insert(strip.id.clone()) {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "meta strip inputs contain a cycle",
            json!({"strip_id":strip.id}),
        ));
    }
    match strip.kind {
        StripType::Sound => output.push(strip),
        StripType::Meta => {
            for input_id in &strip.inputs {
                let child = scene
                    .sequencer
                    .strips
                    .iter()
                    .find(|candidate| candidate.id == *input_id)
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::TargetNotFound,
                            "meta strip references a missing child strip",
                            json!({"strip_id":strip.id,"input":input_id}),
                        )
                    })?;
                gather_sound_strips(scene, child, visiting, output)?;
            }
        }
        _ => {}
    }
    visiting.remove(&strip.id);
    Ok(())
}

fn resolve_source_path(root: &Path, source: &str) -> PathBuf {
    let path = Path::new(source);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn write_exr(path: &Path, rgba: &[[f32; 4]], width: u32, height: u32) -> Result<()> {
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR height exceeds platform limits",
        )
    })?;
    let expected_len = width_usize.checked_mul(height_usize).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR dimensions exceed platform limits",
        )
    })?;
    if rgba.len() != expected_len {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            "linear RGBA buffer dimensions do not match EXR output",
        ));
    }
    exr::prelude::write_rgba_file(path, width_usize, height_usize, |x, y| {
        let pixel = rgba[y * width_usize + x];
        // The VSE frame is associated; alpha-zero RGB can contain additive camera-shape radiance.
        (pixel[0], pixel[1], pixel[2], pixel[3])
    })
    .map_err(|error| {
        PotError::new(
            ErrorCode::RenderFailed,
            format!("EXR encoding failed: {error}"),
        )
    })
}

fn file_hash(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    Ok(hash::sha256(&bytes))
}
#[cfg(test)]
#[path = "sequence_tests.rs"]
mod tests;
