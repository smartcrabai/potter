// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/render/intern/texture_image.cc (imagewrap, boxsample), texture_procedural.cc
//   source/blender/blenlib/intern/noise_c.cc, source/blender/modifiers/intern/MOD_util.cc,
//   SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved., GPL-2.0-or-later.

//! Blender legacy textures: inline `legacy_texture` records in `doc.resources`, the single texture
//! resolver, and the sampler shared by every consumer.
//!
//! Coordinates are texture-space points: the caller applies the mapping (LOCAL, GLOBAL, OBJECT, or
//! UV). The result follows Blender's `BKE_texture_get_value`: `intensity` is the texture's `tin`,
//! and `rgb` is the colour for RGB results or the intensity replicated otherwise.

use std::{f64::consts::PI, path::Path};

use crate::blender_rng::BlenderRng;
use glam::DVec3;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    image::{ImageData, ImageInterpolation, load_image_data},
    model::{Id, ImageAlphaMode, ImageColorspace, SceneDoc, migrate},
};

/// Record kind of a Blender legacy texture.
pub(crate) const LEGACY_TEXTURE_KIND: &str = "legacy_texture";

/// Blender's `Blend` texture progression (`Tex::stype` with the RNA `progression` identifiers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlendProgression {
    Linear,
    Quadratic,
    Easing,
    Diagonal,
    Spherical,
    QuadraticSphere,
    Radial,
}

impl BlendProgression {
    fn from_identifier(text: &str) -> Option<Self> {
        Some(match text {
            "LINEAR" => Self::Linear,
            "QUADRATIC" => Self::Quadratic,
            "EASING" => Self::Easing,
            "DIAGONAL" => Self::Diagonal,
            "SPHERICAL" => Self::Spherical,
            "QUADRATIC_SPHERE" => Self::QuadraticSphere,
            "RADIAL" => Self::Radial,
            _ => return None,
        })
    }

    fn identifier(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::Quadratic => "QUADRATIC",
            Self::Easing => "EASING",
            Self::Diagonal => "DIAGONAL",
            Self::Spherical => "SPHERICAL",
            Self::QuadraticSphere => "QUADRATIC_SPHERE",
            Self::Radial => "RADIAL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlendTexture {
    pub(crate) progression: BlendProgression,
    pub(crate) flip_axis: bool,
}

/// A Blender legacy texture held inline in `doc.resources`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LegacyTexture {
    Blend(BlendTexture),
    ImageRecord {
        image: Option<Id>,
        settings: Map<String, Value>,
        sampler: ImageSamplerSettings,
    },
    NativeValueNoise,
    Procedural {
        texture_type: String,
        settings: Map<String, Value>,
        sampler: ProceduralSamplerSettings,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProceduralSamplerSettings {
    kind: u8,
    noise_size: f32,
    turbulence: f32,
    depth: i32,
    basis: u8,
    basis_2: u8,
    hard_noise: bool,
    cloud_color: bool,
    wood_type: u8,
    marble_type: u8,
    stucci_type: u8,
    distortion: f32,
    musgrave_type: u8,
    dimension_max: f32,
    lacunarity: f32,
    octaves: f32,
    offset: f32,
    gain: f32,
    noise_intensity: f32,
    voronoi_weights: [f32; 4],
    minkovsky_exponent: f32,
    distance_metric: u8,
    color_mode: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageExtension {
    Extend,
    Clip,
    ClipCube,
    Repeat,
    Checker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RampInterpolation {
    Linear,
    Ease,
    BSpline,
    Cardinal,
    Constant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RampColorMode {
    Rgb,
    Hsv,
    Hsl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RampHueInterpolation {
    Near,
    Far,
    Clockwise,
    CounterClockwise,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RampStop {
    position: f32,
    color: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorRamp {
    stops: [RampStop; 32],
    stop_count: u8,
    interpolation: RampInterpolation,
    color_mode: RampColorMode,
    hue_interpolation: RampHueInterpolation,
}

const fn default_ramp_stops() -> [RampStop; 32] {
    let mut stops = [RampStop {
        position: 0.0,
        color: [0.0, 0.0, 0.0, 1.0],
    }; 32];
    stops[1] = RampStop {
        position: 1.0,
        color: [1.0, 1.0, 1.0, 1.0],
    };
    stops
}

const DEFAULT_COLOR_RAMP: ColorRamp = ColorRamp {
    stops: default_ramp_stops(),
    stop_count: 2,
    interpolation: RampInterpolation::Linear,
    color_mode: RampColorMode::Rgb,
    hue_interpolation: RampHueInterpolation::Near,
};

/// Sampler-affecting Blender `Texture`/`ImageTexture` RNA values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImageSamplerSettings {
    pub(crate) extension: ImageExtension,
    pub(crate) interpolation: bool,
    pub(crate) filter_size: f32,
    pub(crate) flip_axis: bool,
    pub(crate) repeat_x: u32,
    pub(crate) repeat_y: u32,
    pub(crate) mirror_x: bool,
    pub(crate) mirror_y: bool,
    pub(crate) checker_odd: bool,
    pub(crate) checker_even: bool,
    pub(crate) checker_distance: f32,
    pub(crate) crop_min_x: f32,
    pub(crate) crop_min_y: f32,
    pub(crate) crop_max_x: f32,
    pub(crate) crop_max_y: f32,
    pub(crate) use_alpha: bool,
    pub(crate) calculate_alpha: bool,
    pub(crate) invert_alpha: bool,
    pub(crate) brightness: f32,
    pub(crate) contrast: f32,
    pub(crate) saturation: f32,
    pub(crate) factor_red: f32,
    pub(crate) factor_green: f32,
    pub(crate) factor_blue: f32,
    pub(crate) use_clamp: bool,
    source_srgb_encoded: bool,
    use_color_ramp: bool,
    color_ramp: ColorRamp,
}

pub(crate) const DEFAULT_IMAGE_SETTINGS: ImageSamplerSettings = ImageSamplerSettings {
    extension: ImageExtension::Repeat,
    interpolation: true,
    filter_size: 1.0,
    flip_axis: false,
    repeat_x: 1,
    repeat_y: 1,
    mirror_x: false,
    mirror_y: false,
    checker_odd: true,
    checker_even: false,
    checker_distance: 0.0,
    crop_min_x: 0.0,
    crop_min_y: 0.0,
    crop_max_x: 1.0,
    crop_max_y: 1.0,
    use_alpha: true,
    calculate_alpha: false,
    invert_alpha: false,
    brightness: 1.0,
    contrast: 1.0,
    saturation: 1.0,
    factor_red: 1.0,
    factor_green: 1.0,
    factor_blue: 1.0,
    use_clamp: false,
    source_srgb_encoded: false,
    use_color_ramp: false,
    color_ramp: DEFAULT_COLOR_RAMP,
};

/// A resolved image or Blender legacy texture resource, ready for shared sampling.
#[derive(Debug)]
pub struct ResolvedTexture {
    resolved: ResolvedTextureKind,
}

#[derive(Debug)]
enum ResolvedTextureKind {
    Image(ImageData),
    ImageWithSettings {
        image: ImageData,
        settings: ImageSamplerSettings,
    },
    Legacy(LegacyTexture),
}

impl ResolvedTexture {
    pub(crate) fn is_image(&self) -> bool {
        matches!(
            &self.resolved,
            ResolvedTextureKind::Image(_) | ResolvedTextureKind::ImageWithSettings { .. }
        )
    }

    pub(crate) fn source(&self) -> TextureSource<'_> {
        match &self.resolved {
            ResolvedTextureKind::Image(image) => TextureSource::Image(image),
            ResolvedTextureKind::ImageWithSettings { image, settings } => {
                TextureSource::ImageWithSettings(image, settings)
            }
            ResolvedTextureKind::Legacy(texture) => TextureSource::Legacy(texture),
        }
    }
}

/// What the sampler reads.
#[derive(Clone, Copy)]
pub(crate) enum TextureSource<'a> {
    Image(&'a ImageData),
    ImageWithSettings(&'a ImageData, &'a ImageSamplerSettings),
    Legacy(&'a LegacyTexture),
}

/// One lookup result, as Blender's `TexResult` after `BKE_texture_get_value`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextureResult {
    /// Blender's `tin`: the mean RGB for RGB results, otherwise the scalar intensity.
    pub(crate) intensity: f64,
    pub(crate) rgb: [f64; 3],
    pub(crate) alpha: f64,
    pub(crate) is_rgb: bool,
}

fn record_error(id: &str, message: impl Into<String>) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        message.into(),
        json!({"resource": id, "kind": LEGACY_TEXTURE_KIND}),
    )
}

/// Rejects settings keys that the texture type does not define.
fn known_keys(id: &str, settings: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    match settings.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(record_error(
            id,
            format!("settings key `{key}` is not defined for this texture type"),
        )),
        None => Ok(()),
    }
}

fn parse_blend(id: &str, settings: &Map<String, Value>) -> Result<BlendTexture> {
    known_keys(id, settings, &["progression", "use_flip_axis"])?;
    let progression = match settings.get("progression") {
        None => BlendProgression::Linear,
        Some(value) => value
            .as_str()
            .and_then(BlendProgression::from_identifier)
            .ok_or_else(|| record_error(id, "`progression` is not a Blend progression"))?,
    };
    let flip_axis = match settings.get("use_flip_axis") {
        None => false,
        Some(Value::String(value)) if value == "HORIZONTAL" => false,
        Some(Value::String(value)) if value == "VERTICAL" => true,
        _ => {
            return Err(record_error(
                id,
                "`use_flip_axis` must be HORIZONTAL or VERTICAL",
            ));
        }
    };
    Ok(BlendTexture {
        progression,
        flip_axis,
    })
}

const LEGACY_TEXTURE_TYPES: [&str; 12] = [
    "CLOUDS",
    "MARBLE",
    "WOOD",
    "MAGIC",
    "BLEND",
    "STUCCI",
    "NOISE",
    "DISTORTED_NOISE",
    "VORONOI",
    "MUSGRAVE",
    "NATIVE_VALUE_NOISE",
    "IMAGE",
];

fn texture_settings_keys(texture_type: &str) -> &'static [&'static str] {
    match texture_type {
        "CLOUDS" => &[
            "noise_scale",
            "noise_depth",
            "noise_basis",
            "noise_type",
            "cloud_type",
        ],
        "MARBLE" => &[
            "noise_scale",
            "turbulence",
            "noise_depth",
            "noise_type",
            "marble_type",
            "noise_basis",
            "noise_basis_2",
        ],
        "WOOD" => &[
            "noise_scale",
            "turbulence",
            "noise_basis",
            "noise_type",
            "wood_type",
            "noise_basis_2",
        ],
        "MAGIC" => &["turbulence", "noise_depth"],
        "BLEND" => &["progression", "use_flip_axis"],
        "STUCCI" => &[
            "turbulence",
            "noise_basis",
            "noise_scale",
            "noise_type",
            "stucci_type",
        ],
        "NOISE" => &[],
        "DISTORTED_NOISE" => &[
            "distortion",
            "noise_scale",
            "noise_basis",
            "noise_distortion",
        ],
        "VORONOI" => &[
            "weight_1",
            "weight_2",
            "weight_3",
            "weight_4",
            "minkovsky_exponent",
            "distance_metric",
            "color_mode",
            "noise_intensity",
            "noise_scale",
        ],
        "MUSGRAVE" => &[
            "musgrave_type",
            "dimension_max",
            "lacunarity",
            "octaves",
            "offset",
            "gain",
            "noise_intensity",
            "noise_scale",
            "noise_basis",
        ],
        "NATIVE_VALUE_NOISE" => &[],
        "IMAGE" => &[
            "use_interpolation",
            "filter_size",
            "use_flip_axis",
            "use_alpha",
            "use_calculate_alpha",
            "invert_alpha",
            "extension",
            "repeat_x",
            "repeat_y",
            "use_mirror_x",
            "use_mirror_y",
            "use_checker_odd",
            "use_checker_even",
            "checker_distance",
            "crop_min_x",
            "crop_min_y",
            "crop_max_x",
            "crop_max_y",
            "brightness",
            "contrast",
            "saturation",
            "factor_red",
            "factor_green",
            "factor_blue",
            "use_clamp",
            "use_color_ramp",
            "color_ramp",
            "image_user",
        ],
        _ => &[],
    }
}

fn enum_values(key: &str) -> Option<&'static [&'static str]> {
    match key {
        "noise_basis" | "noise_distortion" => Some(&[
            "BLENDER_ORIGINAL",
            "ORIGINAL_PERLIN",
            "IMPROVED_PERLIN",
            "VORONOI_F1",
            "VORONOI_F2",
            "VORONOI_F3",
            "VORONOI_F4",
            "VORONOI_F2_F1",
            "VORONOI_CRACKLE",
            "CELL_NOISE",
        ]),
        "noise_basis_2" => Some(&["SIN", "SAW", "TRI"]),
        "noise_type" => Some(&["SOFT_NOISE", "HARD_NOISE"]),
        "cloud_type" => Some(&["GRAYSCALE", "COLOR"]),
        "wood_type" => Some(&["BANDS", "RINGS", "BANDNOISE", "RINGNOISE"]),
        "marble_type" => Some(&["SOFT", "SHARP", "SHARPER"]),
        "musgrave_type" => Some(&[
            "MULTIFRACTAL",
            "RIDGED_MULTIFRACTAL",
            "HYBRID_MULTIFRACTAL",
            "FBM",
            "HETERO_TERRAIN",
        ]),
        "distance_metric" => Some(&[
            "DISTANCE",
            "DISTANCE_SQUARED",
            "MANHATTAN",
            "CHEBYCHEV",
            "MINKOVSKY_HALF",
            "MINKOVSKY_FOUR",
            "MINKOVSKY",
        ]),
        "color_mode" => Some(&[
            "INTENSITY",
            "POSITION",
            "POSITION_OUTLINE",
            "POSITION_OUTLINE_INTENSITY",
        ]),
        "progression" => Some(&[
            "LINEAR",
            "QUADRATIC",
            "EASING",
            "DIAGONAL",
            "SPHERICAL",
            "QUADRATIC_SPHERE",
            "RADIAL",
        ]),
        "use_flip_axis" => Some(&["HORIZONTAL", "VERTICAL"]),
        "stucci_type" => Some(&["PLASTIC", "WALL_IN", "WALL_OUT"]),
        "extension" => Some(&["EXTEND", "CLIP", "CLIP_CUBE", "REPEAT", "CHECKER"]),
        _ => None,
    }
}

fn validate_legacy_settings(
    id: &str,
    texture_type: &str,
    settings: &Map<String, Value>,
) -> Result<()> {
    known_keys(id, settings, texture_settings_keys(texture_type))?;
    if texture_type == "IMAGE" {
        return validate_image_settings(id, settings);
    }
    for (key, value) in settings {
        if let Some(values) = enum_values(key) {
            if !value.as_str().is_some_and(|value| values.contains(&value)) {
                return Err(record_error(
                    id,
                    format!("`{key}` is not a valid Blender enum value"),
                ));
            }
        } else if key == "noise_depth" {
            if !value.as_u64().is_some_and(|depth| depth <= 30) {
                return Err(record_error(
                    id,
                    "`noise_depth` must be an integer from 0 through 30",
                ));
            }
        } else if let Some(number) = value.as_f64().filter(|number| number.is_finite()) {
            if number.abs() > f64::from(f32::MAX) {
                return Err(record_error(
                    id,
                    format!("`{key}` exceeds Blender's float range"),
                ));
            }
            if key == "noise_scale" && number < 0.0001 {
                return Err(record_error(id, "`noise_scale` must be at least 0.0001"));
            }
            if key == "distortion" && !(0.0..=10.0).contains(&number) {
                return Err(record_error(id, "`distortion` must be in [0, 10]"));
            }
            if key == "minkovsky_exponent" && number <= 0.0 {
                return Err(record_error(id, "`minkovsky_exponent` must be positive"));
            }
        } else {
            return Err(record_error(id, format!("`{key}` must be a finite number")));
        }
    }
    Ok(())
}

fn validate_image_settings(id: &str, settings: &Map<String, Value>) -> Result<()> {
    for (key, value) in settings {
        let valid = match key.as_str() {
            "use_interpolation"
            | "use_flip_axis"
            | "use_alpha"
            | "use_calculate_alpha"
            | "invert_alpha"
            | "use_mirror_x"
            | "use_mirror_y"
            | "use_checker_odd"
            | "use_checker_even"
            | "use_clamp"
            | "use_color_ramp" => value.is_boolean(),
            "extension" => enum_values(key)
                .is_some_and(|values| value.as_str().is_some_and(|value| values.contains(&value))),
            "repeat_x" | "repeat_y" => value
                .as_u64()
                .is_some_and(|repeat| (1..=512).contains(&repeat)),
            "filter_size" => image_number_in_range(value, 0.1, 50.0),
            "checker_distance" => image_number_in_range(value, 0.0, 0.99),
            "crop_min_x" | "crop_min_y" | "crop_max_x" | "crop_max_y" => {
                image_number_in_range(value, -10.0, 10.0)
            }
            "brightness" | "saturation" | "factor_red" | "factor_green" | "factor_blue" => {
                image_number_in_range(value, 0.0, 2.0)
            }
            "contrast" => image_number_in_range(value, 0.0, 5.0),
            "color_ramp" => value.as_object().is_some(),
            "image_user" => value.is_object(),
            _ => false,
        };
        if !valid {
            return Err(record_error(
                id,
                format!("`{key}` is not a valid Blender IMAGE setting"),
            ));
        }
    }
    if let Some(value) = settings.get("color_ramp") {
        let ramp = value
            .as_object()
            .ok_or_else(|| record_error(id, "`color_ramp` must be an object"))?;
        parse_color_ramp(id, ramp)?;
    }
    Ok(())
}

fn image_number_in_range(value: &Value, minimum: f64, maximum: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|number| number.is_finite() && (minimum..=maximum).contains(&number))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Blender stores ColorBand positions and colors as f32"
)]
fn parse_color_ramp(id: &str, object: &Map<String, Value>) -> Result<ColorRamp> {
    if let Some(key) = object.keys().find(|key| {
        !matches!(
            key.as_str(),
            "elements" | "interpolation" | "color_mode" | "hue_interpolation"
        )
    }) {
        return Err(record_error(
            id,
            format!("color_ramp field `{key}` is not defined"),
        ));
    }
    let elements = object
        .get("elements")
        .and_then(Value::as_array)
        .filter(|elements| (1..=32).contains(&elements.len()))
        .ok_or_else(|| record_error(id, "`color_ramp.elements` must contain 1 through 32 stops"))?;
    let interpolation = match object
        .get("interpolation")
        .and_then(Value::as_str)
        .unwrap_or("LINEAR")
    {
        "LINEAR" => RampInterpolation::Linear,
        "EASE" => RampInterpolation::Ease,
        "B_SPLINE" => RampInterpolation::BSpline,
        "CARDINAL" => RampInterpolation::Cardinal,
        "CONSTANT" => RampInterpolation::Constant,
        _ => return Err(record_error(id, "color_ramp interpolation is invalid")),
    };
    let color_mode = match object
        .get("color_mode")
        .and_then(Value::as_str)
        .unwrap_or("RGB")
    {
        "RGB" => RampColorMode::Rgb,
        "HSV" => RampColorMode::Hsv,
        "HSL" => RampColorMode::Hsl,
        _ => return Err(record_error(id, "color_ramp color_mode is invalid")),
    };
    let hue_interpolation = match object
        .get("hue_interpolation")
        .and_then(Value::as_str)
        .unwrap_or("NEAR")
    {
        "NEAR" => RampHueInterpolation::Near,
        "FAR" => RampHueInterpolation::Far,
        "CW" => RampHueInterpolation::Clockwise,
        "CCW" => RampHueInterpolation::CounterClockwise,
        _ => return Err(record_error(id, "color_ramp hue_interpolation is invalid")),
    };
    let mut ramp = DEFAULT_COLOR_RAMP;
    ramp.stop_count = u8::try_from(elements.len())
        .map_err(|_| record_error(id, "color_ramp has too many stops"))?;
    ramp.interpolation = interpolation;
    ramp.color_mode = color_mode;
    ramp.hue_interpolation = hue_interpolation;
    let mut previous_position = f32::NEG_INFINITY;
    for (index, element) in elements.iter().enumerate() {
        let element = element
            .as_object()
            .ok_or_else(|| record_error(id, "color_ramp elements must be objects"))?;
        if let Some(key) = element
            .keys()
            .find(|key| !matches!(key.as_str(), "position" | "pos" | "color"))
        {
            return Err(record_error(
                id,
                format!("color_ramp element field `{key}` is not defined"),
            ));
        }
        let position = element
            .get("position")
            .or_else(|| element.get("pos"))
            .and_then(Value::as_f64)
            .filter(|position| position.is_finite() && (0.0..=1.0).contains(position))
            .map(|position| position as f32)
            .ok_or_else(|| record_error(id, "color_ramp stop position must be in [0, 1]"))?;
        if position < previous_position {
            return Err(record_error(
                id,
                "color_ramp stop positions must be ordered",
            ));
        }
        previous_position = position;
        let color = element
            .get("color")
            .and_then(Value::as_array)
            .filter(|color| color.len() == 4)
            .ok_or_else(|| record_error(id, "color_ramp stop color must be RGBA"))?;
        let mut rgba = [0.0; 4];
        for (channel, value) in rgba.iter_mut().zip(color) {
            *channel = value
                .as_f64()
                .filter(|value| value.is_finite() && value.abs() <= f64::from(f32::MAX))
                .map(|value| value as f32)
                .ok_or_else(|| record_error(id, "color_ramp stop channels must be finite"))?;
        }
        ramp.stops[index] = RampStop {
            position,
            color: rgba,
        };
    }
    Ok(ramp)
}

fn parse_image_settings(settings: &Map<String, Value>) -> Result<ImageSamplerSettings> {
    let mut parsed = DEFAULT_IMAGE_SETTINGS;
    parsed.extension = match setting_text(settings, "extension", "REPEAT") {
        "EXTEND" => ImageExtension::Extend,
        "CLIP" => ImageExtension::Clip,
        "CLIP_CUBE" => ImageExtension::ClipCube,
        "CHECKER" => ImageExtension::Checker,
        _ => ImageExtension::Repeat,
    };
    parsed.interpolation = settings
        .get("use_interpolation")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    parsed.filter_size = setting_f32(settings, "filter_size", 1.0);
    parsed.flip_axis = settings
        .get("use_flip_axis")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.repeat_x = settings
        .get("repeat_x")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(1);
    parsed.repeat_y = settings
        .get("repeat_y")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(1);
    parsed.mirror_x = settings
        .get("use_mirror_x")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.mirror_y = settings
        .get("use_mirror_y")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.checker_odd = settings
        .get("use_checker_odd")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    parsed.checker_even = settings
        .get("use_checker_even")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.checker_distance = setting_f32(settings, "checker_distance", 0.0);
    parsed.crop_min_x = setting_f32(settings, "crop_min_x", 0.0);
    parsed.crop_min_y = setting_f32(settings, "crop_min_y", 0.0);
    parsed.crop_max_x = setting_f32(settings, "crop_max_x", 1.0);
    parsed.crop_max_y = setting_f32(settings, "crop_max_y", 1.0);
    parsed.use_alpha = settings
        .get("use_alpha")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    parsed.calculate_alpha = settings
        .get("use_calculate_alpha")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.invert_alpha = settings
        .get("invert_alpha")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.brightness = setting_f32(settings, "brightness", 1.0);
    parsed.contrast = setting_f32(settings, "contrast", 1.0);
    parsed.saturation = setting_f32(settings, "saturation", 1.0);
    parsed.factor_red = setting_f32(settings, "factor_red", 1.0);
    parsed.factor_green = setting_f32(settings, "factor_green", 1.0);
    parsed.factor_blue = setting_f32(settings, "factor_blue", 1.0);
    parsed.use_clamp = settings
        .get("use_clamp")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    parsed.use_color_ramp = settings
        .get("use_color_ramp")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(ramp) = settings.get("color_ramp").and_then(Value::as_object) {
        parsed.color_ramp = parse_color_ramp("texture", ramp)?;
    }
    Ok(parsed)
}

/// Parses Blender RNA settings once into a compact sampler configuration.
fn parse_procedural_settings(
    texture_type: &str,
    settings: &Map<String, Value>,
) -> ProceduralSamplerSettings {
    let kind = match texture_type {
        "CLOUDS" => 0,
        "WOOD" => 1,
        "MARBLE" => 2,
        "MAGIC" => 3,
        "STUCCI" => 4,
        "VORONOI" => 5,
        "DISTORTED_NOISE" => 6,
        "MUSGRAVE" => 7,
        "NOISE" => 8,
        _ => 9,
    };
    let waveform = match setting_text(settings, "noise_basis_2", "SIN") {
        "SAW" => 1,
        "TRI" => 2,
        _ => 0,
    };
    let basis = if texture_type == "DISTORTED_NOISE" {
        texture_noise_basis(settings, "noise_distortion", "BLENDER_ORIGINAL")
    } else {
        texture_noise_basis(settings, "noise_basis", "BLENDER_ORIGINAL")
    };
    let basis_2 = if texture_type == "DISTORTED_NOISE" {
        texture_noise_basis(settings, "noise_basis", "BLENDER_ORIGINAL")
    } else {
        waveform
    };
    ProceduralSamplerSettings {
        kind,
        noise_size: setting_f32(settings, "noise_scale", 0.25),
        turbulence: setting_f32(settings, "turbulence", 5.0),
        depth: noise_depth(settings),
        basis,
        basis_2,
        hard_noise: hard_noise(settings),
        cloud_color: setting_text(settings, "cloud_type", "GRAYSCALE") == "COLOR",
        wood_type: match setting_text(settings, "wood_type", "BANDS") {
            "RINGS" => 1,
            "BANDNOISE" => 2,
            "RINGNOISE" => 3,
            _ => 0,
        },
        marble_type: match setting_text(settings, "marble_type", "SOFT") {
            "SHARP" => 1,
            "SHARPER" => 2,
            _ => 0,
        },
        stucci_type: match setting_text(settings, "stucci_type", "PLASTIC") {
            "WALL_IN" => 1,
            "WALL_OUT" => 2,
            _ => 0,
        },
        distortion: setting_f32(settings, "distortion", 1.0),
        musgrave_type: match setting_text(settings, "musgrave_type", "MULTIFRACTAL") {
            "RIDGED_MULTIFRACTAL" => 1,
            "HYBRID_MULTIFRACTAL" => 2,
            "FBM" => 3,
            "HETERO_TERRAIN" => 4,
            _ => 0,
        },
        dimension_max: setting_f32(settings, "dimension_max", 1.0),
        lacunarity: setting_f32(settings, "lacunarity", 2.0),
        octaves: setting_f32(settings, "octaves", 2.0),
        offset: setting_f32(settings, "offset", 1.0),
        gain: setting_f32(settings, "gain", 1.0),
        noise_intensity: setting_f32(settings, "noise_intensity", 1.0),
        voronoi_weights: [
            setting_f32(settings, "weight_1", 1.0),
            setting_f32(settings, "weight_2", 0.0),
            setting_f32(settings, "weight_3", 0.0),
            setting_f32(settings, "weight_4", 0.0),
        ],
        minkovsky_exponent: setting_f32(settings, "minkovsky_exponent", 2.5),
        distance_metric: match setting_text(settings, "distance_metric", "DISTANCE") {
            "DISTANCE_SQUARED" => 1,
            "MANHATTAN" => 2,
            "CHEBYCHEV" => 3,
            "MINKOVSKY_HALF" => 4,
            "MINKOVSKY_FOUR" => 5,
            "MINKOVSKY" => 6,
            _ => 0,
        },
        color_mode: match setting_text(settings, "color_mode", "INTENSITY") {
            "POSITION" => 1,
            "POSITION_OUTLINE" => 2,
            "POSITION_OUTLINE_INTENSITY" => 3,
            _ => 0,
        },
    }
}

/// Parses and validates an inline `legacy_texture` record.
///
/// # Errors
///
/// Returns `SCENE_INVALID` for a malformed inline legacy texture record.
pub(crate) fn parse_record(id: &str, value: &Value) -> Result<LegacyTexture> {
    if migrate::inline_resource_kind(value)? != Some(LEGACY_TEXTURE_KIND) {
        return Err(record_error(id, "expected an inline legacy_texture record"));
    }
    let object = value
        .as_object()
        .ok_or_else(|| record_error(id, "expected an inline legacy_texture record"))?;
    if let Some(key) = object.keys().find(|key| {
        !matches!(
            key.as_str(),
            "storage" | "kind" | "texture_type" | "image" | "settings"
        )
    }) {
        return Err(record_error(
            id,
            format!("record field `{key}` is not defined"),
        ));
    }
    let texture_type = object
        .get("texture_type")
        .and_then(Value::as_str)
        .ok_or_else(|| record_error(id, "texture_type must be a string"))?;
    let image = match (texture_type, object.get("image")) {
        ("IMAGE", Some(Value::String(image))) => Some(
            Id::new(image.clone())
                .map_err(|_| record_error(id, "`image` must be a valid image ID or null"))?,
        ),
        ("IMAGE", Some(Value::Null)) => None,
        ("IMAGE", Some(_)) => return Err(record_error(id, "`image` must be an image ID or null")),
        ("IMAGE", None) => return Err(record_error(id, "IMAGE texture requires an `image` field")),
        (_, Some(_)) => {
            return Err(record_error(
                id,
                "only IMAGE textures may have an image field",
            ));
        }
        (_, None) => None,
    };
    let settings = match object.get("settings") {
        None => Map::new(),
        Some(Value::Object(settings)) => settings.clone(),
        Some(_) => return Err(record_error(id, "settings must be an object")),
    };
    if !LEGACY_TEXTURE_TYPES.contains(&texture_type) {
        return Err(record_error(
            id,
            format!("unknown legacy texture type `{texture_type}`"),
        ));
    }
    validate_legacy_settings(id, texture_type, &settings)?;
    match texture_type {
        "NATIVE_VALUE_NOISE" => Ok(LegacyTexture::NativeValueNoise),
        "BLEND" => Ok(LegacyTexture::Blend(parse_blend(id, &settings)?)),
        "IMAGE" => Ok(LegacyTexture::ImageRecord {
            image,
            sampler: parse_image_settings(&settings)?,
            settings,
        }),
        _ => {
            let sampler = parse_procedural_settings(texture_type, &settings);
            Ok(LegacyTexture::Procedural {
                texture_type: texture_type.to_owned(),
                settings,
                sampler,
            })
        }
    }
}

/// Validates an inline `legacy_texture` record in `doc.resources`.
///
/// # Errors
///
/// Returns the errors of [`parse_record`].
pub(crate) fn validate_record(id: &Id, value: &Value) -> Result<()> {
    parse_record(&id.to_string(), value).map(|_| ())
}

/// Validates a record and its image reference without loading the image pixels.
pub(crate) fn validate_record_in_scene(doc: &SceneDoc, id: &Id, value: &Value) -> Result<()> {
    if let LegacyTexture::ImageRecord {
        image: Some(image), ..
    } = parse_record(&id.to_string(), value)?
    {
        if !doc.images.contains_key(&image) {
            return Err(record_error(
                &id.to_string(),
                format!("IMAGE texture references missing image `{image}`"),
            ));
        }
    }
    Ok(())
}

/// The canonical record of a texture: every setting explicit, defaults included.
pub(crate) fn record_value(texture: &LegacyTexture) -> Value {
    match texture {
        LegacyTexture::Blend(blend) => json!({
            "storage": "inline",
            "kind": LEGACY_TEXTURE_KIND,
            "texture_type": "BLEND",
            "settings": {
                "progression": blend.progression.identifier(),
                "use_flip_axis": if blend.flip_axis { "VERTICAL" } else { "HORIZONTAL" },
            },
        }),
        LegacyTexture::ImageRecord {
            image, settings, ..
        } => json!({
            "storage": "inline",
            "kind": LEGACY_TEXTURE_KIND,
            "texture_type": "IMAGE",
            "image": image.as_ref().map(ToString::to_string),
            "settings": settings,
        }),
        LegacyTexture::NativeValueNoise => json!({
            "storage": "inline",
            "kind": LEGACY_TEXTURE_KIND,
            "texture_type": "NATIVE_VALUE_NOISE",
            "settings": {},
        }),
        LegacyTexture::Procedural {
            texture_type,
            settings,
            sampler: _,
        } => json!({
            "storage": "inline",
            "kind": LEGACY_TEXTURE_KIND,
            "texture_type": texture_type,
            "settings": settings,
        }),
    }
}

fn default_settings(texture_type: &str) -> Map<String, Value> {
    let defaults = match texture_type {
        "CLOUDS" => {
            json!({"noise_scale":0.25,"noise_depth":2,"noise_basis":"BLENDER_ORIGINAL","noise_type":"SOFT_NOISE","cloud_type":"GRAYSCALE"})
        }
        "MARBLE" => {
            json!({"noise_scale":0.25,"turbulence":5.0,"noise_depth":2,"noise_type":"SOFT_NOISE","marble_type":"SOFT","noise_basis":"BLENDER_ORIGINAL","noise_basis_2":"SIN"})
        }
        "WOOD" => {
            json!({"noise_scale":0.25,"turbulence":5.0,"noise_basis":"BLENDER_ORIGINAL","noise_type":"SOFT_NOISE","wood_type":"BANDS","noise_basis_2":"SIN"})
        }
        "MAGIC" => json!({"turbulence":5.0,"noise_depth":2}),
        "BLEND" => json!({"progression":"LINEAR","use_flip_axis":"HORIZONTAL"}),
        "STUCCI" => {
            json!({"turbulence":5.0,"noise_basis":"BLENDER_ORIGINAL","noise_scale":0.25,"noise_type":"SOFT_NOISE","stucci_type":"PLASTIC"})
        }
        "DISTORTED_NOISE" => {
            json!({"distortion":1.0,"noise_scale":0.25,"noise_basis":"BLENDER_ORIGINAL","noise_distortion":"BLENDER_ORIGINAL"})
        }
        "VORONOI" => {
            json!({"weight_1":1.0,"weight_2":0.0,"weight_3":0.0,"weight_4":0.0,"minkovsky_exponent":2.5,"distance_metric":"DISTANCE","color_mode":"INTENSITY","noise_intensity":1.0,"noise_scale":0.25})
        }
        "MUSGRAVE" => {
            json!({"musgrave_type":"MULTIFRACTAL","dimension_max":1.0,"lacunarity":2.0,"octaves":2.0,"offset":1.0,"gain":1.0,"noise_intensity":1.0,"noise_scale":0.25,"noise_basis":"BLENDER_ORIGINAL"})
        }
        "IMAGE" => json!({
            "use_interpolation":true,"filter_size":1.0,"use_flip_axis":false,"use_alpha":true,
            "use_calculate_alpha":false,"invert_alpha":false,"extension":"REPEAT","repeat_x":1,
            "repeat_y":1,"use_mirror_x":false,"use_mirror_y":false,"use_checker_odd":true,
            "use_checker_even":false,"checker_distance":0.0,"crop_min_x":0.0,"crop_min_y":0.0,
            "crop_max_x":1.0,"crop_max_y":1.0,"brightness":1.0,"contrast":1.0,"saturation":1.0,
            "factor_red":1.0,"factor_green":1.0,"factor_blue":1.0,"use_clamp":false,
            "use_color_ramp":false,
            "color_ramp":{
                "elements":[
                    {"position":0.0,"color":[0.0,0.0,0.0,1.0]},
                    {"position":1.0,"color":[1.0,1.0,1.0,1.0]}
                ],
                "interpolation":"LINEAR","color_mode":"RGB","hue_interpolation":"NEAR"
            }
        }),
        _ => json!({}),
    };
    defaults.as_object().cloned().unwrap_or_default()
}
/// Migrates Potter's old Volume Displace `texture: "noise"` sentinel to one explicit typed resource.
///
/// Project loading calls this after raw-source/hash verification and before `SceneDoc::validate`.
/// Only the loaded in-memory document changes; source bytes and history remain unchanged.
///
/// # Errors
///
/// Returns `LIMIT_EXCEEDED` if all available deterministic resource IDs are occupied.
pub(crate) fn migrate_legacy_volume_noise_sentinel(doc: &mut SceneDoc) -> Result<bool> {
    let has_sentinel = doc.nodes.values().any(|node| {
        node.modifiers.iter().any(|modifier| {
            modifier.modifier_type == "volume_displace"
                && modifier.params.get("texture").and_then(Value::as_str) == Some("noise")
        })
    });
    if !has_sentinel {
        return Ok(false);
    }
    let existing =
        doc.resources
            .iter()
            .find_map(|(id, value)| match parse_record(&id.to_string(), value) {
                Ok(LegacyTexture::NativeValueNoise) => Some(id.clone()),
                _ => None,
            });
    let (resource_id, insert_record) = if let Some(id) = existing {
        (id, false)
    } else {
        let mut suffix = 0_u32;
        loop {
            let name = if suffix == 0 {
                "potter_value_noise".to_owned()
            } else {
                format!("potter_value_noise_{suffix}")
            };
            let id = Id::new(name)?;
            if !doc.resources.contains_key(&id) && !doc.images.contains_key(&id) {
                break (id, true);
            }
            suffix = suffix.checked_add(1).ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "exhausted inline legacy texture resource IDs",
                )
            })?;
        }
    };
    if insert_record {
        doc.resources.insert(
            resource_id.clone(),
            record_value(&LegacyTexture::NativeValueNoise),
        );
    }
    let texture_id = Value::String(resource_id.to_string());
    for node in doc.nodes.values_mut() {
        for modifier in &mut node.modifiers {
            if modifier.modifier_type == "volume_displace"
                && modifier.params.get("texture").and_then(Value::as_str) == Some("noise")
            {
                modifier
                    .params
                    .insert("texture".to_owned(), texture_id.clone());
            }
        }
    }
    Ok(true)
}

/// The canonical record for `texture_type` with `settings` merged over `base` (when present).
///
/// # Errors
///
/// Returns the errors of [`parse_record`] for the merged settings.
pub(crate) fn merged_record(
    id: &str,
    texture_type: &str,
    image: Option<&Id>,
    base: Option<&Map<String, Value>>,
    settings: &Map<String, Value>,
) -> Result<Value> {
    let mut merged = default_settings(texture_type);
    if let Some(base) = base {
        for (key, value) in base {
            merged.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in settings {
        merged.insert(key.clone(), value.clone());
    }
    let mut record = json!({
        "storage": "inline",
        "kind": LEGACY_TEXTURE_KIND,
        "texture_type": texture_type,
        "settings": merged,
    });
    if texture_type == "IMAGE" {
        record["image"] = image
            .map(|image| json!(image.to_string()))
            .unwrap_or(Value::Null);
    }
    let texture = parse_record(id, &record)?;
    Ok(record_value(&texture))
}

/// Resolves a texture ID: an image in `doc.images`, else an inline legacy texture record.
///
/// # Errors
///
/// Returns `TARGET_NOT_FOUND` for an unknown ID, `DEPENDENCY_MISSING` for an image without a
/// project root, and the errors of [`parse_record`] for a legacy record.
pub fn resolve_texture(
    doc: &SceneDoc,
    texture_id: &Id,
    project_root: Option<&Path>,
) -> Result<ResolvedTexture> {
    if let Some(image) = doc.images.get(texture_id) {
        let root = project_root.ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "image texture requires a project root",
                json!({"texture": texture_id}),
            )
        })?;
        let mut settings = DEFAULT_IMAGE_SETTINGS;
        settings.use_alpha &= image.alpha_mode != ImageAlphaMode::None;
        settings.source_srgb_encoded = image.colorspace == ImageColorspace::Srgb;
        let data = load_image_data(image, root, ImageInterpolation::Linear)?;
        let resolved = if settings == DEFAULT_IMAGE_SETTINGS {
            ResolvedTextureKind::Image(data)
        } else {
            ResolvedTextureKind::ImageWithSettings {
                image: data,
                settings,
            }
        };
        return Ok(ResolvedTexture { resolved });
    }
    let record = doc.resources.get(texture_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("texture `{texture_id}` was not found"),
            json!({"texture": texture_id}),
        )
    })?;
    if migrate::inline_resource_kind(record)? != Some(LEGACY_TEXTURE_KIND) {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("resource `{texture_id}` is not a texture"),
            json!({"texture": texture_id}),
        ));
    }
    let resolved = match parse_record(&texture_id.to_string(), record)? {
        LegacyTexture::ImageRecord {
            image,
            settings: _,
            mut sampler,
        } => {
            let Some(image_id) = image else {
                return Ok(ResolvedTexture {
                    resolved: ResolvedTextureKind::ImageWithSettings {
                        image: ImageData {
                            width: 0,
                            height: 0,
                            pixels: Vec::new(),
                            tiles: Default::default(),
                            interpolation: ImageInterpolation::Linear,
                        },
                        settings: sampler,
                    },
                });
            };
            let image = doc.images.get(&image_id).ok_or_else(|| {
                record_error(
                    &texture_id.to_string(),
                    format!("IMAGE texture references missing image `{image_id}`"),
                )
            })?;
            let root = project_root.ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    "image texture requires a project root",
                    json!({"texture": texture_id}),
                )
            })?;
            sampler.use_alpha &= image.alpha_mode != ImageAlphaMode::None;
            sampler.source_srgb_encoded = image.colorspace == ImageColorspace::Srgb;
            let data = load_image_data(image, root, ImageInterpolation::Linear)?;
            ResolvedTextureKind::ImageWithSettings {
                image: data,
                settings: sampler,
            }
        }
        texture => ResolvedTextureKind::Legacy(texture),
    };
    Ok(ResolvedTexture { resolved })
}

/// Samples a texture at a texture-space point (`BKE_texture_get_value`).
///
/// Coordinates are already mapped by the caller. Blender's legacy Noise type is stateful; its RNG
/// algorithm is `BLI_rng_thread_rand` but Blender seeds worker streams from `clock()`. Potter uses
/// the deterministic per-evaluation `BlenderRng` seeded from the render seed (or zero in Preview),
/// so Noise's bit extraction and call-order semantics match while cross-process clock noise does not.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for malformed texture data or coordinates, or `UNSUPPORTED_FEATURE`
/// for an unknown legacy type. `rng` is consumed only by the random Blender Noise texture.
pub(crate) fn sample(
    source: TextureSource<'_>,
    coords: DVec3,
    rng: &mut BlenderRng,
) -> Result<TextureResult> {
    match source {
        TextureSource::Image(image) => sample_image(image, coords, &DEFAULT_IMAGE_SETTINGS),
        TextureSource::ImageWithSettings(image, settings) => sample_image(image, coords, settings),
        TextureSource::Legacy(LegacyTexture::Blend(blend)) => Ok(sample_blend(*blend, coords)),
        TextureSource::Legacy(LegacyTexture::NativeValueNoise) => {
            Ok(sample_native_value_noise(coords))
        }
        TextureSource::Legacy(LegacyTexture::Procedural { sampler, .. }) => {
            sample_procedural(sampler, coords, rng)
        }
        TextureSource::Legacy(LegacyTexture::ImageRecord { .. }) => Err(
            PotError::invalid_argument("IMAGE texture records must be resolved before sampling"),
        ),
    }
}
/// The former Volume Displace `value_noise_color`, retained only as an explicit migrated native
/// texture type. This preserves old Potter scene appearance without keeping a volume-local sampler.
#[expect(
    clippy::cast_possible_truncation,
    reason = "coordinates are bounded to the exact i64 conversion range"
)]
fn sample_native_value_noise(point: DVec3) -> TextureResult {
    let rgb = if !point.is_finite() || point.abs().max_element() > 9.0e18 {
        [0.5; 3]
    } else {
        let floor = point.floor();
        let base = floor.to_array().map(|value| value as i64);
        let mut fade = (point - floor).to_array();
        for component in &mut fade {
            *component = *component * *component * (3.0 - 2.0 * *component);
        }
        std::array::from_fn(|channel| {
            let mut value = 0.0;
            for z in 0..=1_i64 {
                for y in 0..=1_i64 {
                    for x in 0..=1_i64 {
                        let cell = [base[0] + x, base[1] + y, base[2] + z];
                        let weight = (if x == 0 { 1.0 - fade[0] } else { fade[0] })
                            * (if y == 0 { 1.0 - fade[1] } else { fade[1] })
                            * (if z == 0 { 1.0 - fade[2] } else { fade[2] });
                        value += hash_native_value_noise(cell, channel) * weight;
                    }
                }
            }
            value
        })
    };
    TextureResult {
        intensity: (rgb[0] + rgb[1] + rgb[2]) / 3.0,
        rgb,
        alpha: 1.0,
        is_rgb: true,
    }
}

fn hash_native_value_noise(point: [i64; 3], channel: usize) -> f64 {
    let mut value = (point[0] as u64).wrapping_mul(0x9e37_79b9)
        ^ (point[1] as u64).wrapping_mul(0x85eb_ca6b)
        ^ (point[2] as u64).wrapping_mul(0xc2b2_ae35)
        ^ (channel as u64).wrapping_mul(0x27d4_eb2d);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value as f64) / (u64::MAX as f64)
}

/// Blender's `Blend` texture (`texture_procedural.cc`, `blend`), followed by `BRICONT`.
fn sample_blend(texture: BlendTexture, coords: DVec3) -> TextureResult {
    let (x, y) = if texture.flip_axis {
        (coords.y, coords.x)
    } else {
        (coords.x, coords.y)
    };
    let linear = (1.0 + x) / 2.0;
    let value = match texture.progression {
        BlendProgression::Linear => linear,
        BlendProgression::Quadratic => {
            if linear < 0.0 {
                0.0
            } else {
                linear * linear
            }
        }
        BlendProgression::Easing => {
            if linear <= 0.0 {
                0.0
            } else if linear >= 1.0 {
                1.0
            } else {
                let squared = linear * linear;
                3.0 * squared - 2.0 * squared * linear
            }
        }
        BlendProgression::Diagonal => (2.0 + x + y) / 4.0,
        BlendProgression::Radial => y.atan2(x) / (2.0 * PI) + 0.5,
        BlendProgression::Spherical | BlendProgression::QuadraticSphere => {
            let sphere = (1.0 - (x * x + y * y + coords.z * coords.z).sqrt()).max(0.0);
            if texture.progression == BlendProgression::QuadraticSphere {
                sphere * sphere
            } else {
                sphere
            }
        }
    };
    // BRICONT without TEX_NO_CLAMP cleared: the default texture keeps the value unclamped.
    let intensity = value;
    TextureResult {
        intensity,
        rgb: [intensity; 3],
        alpha: 1.0,
        is_rgb: false,
    }
}

/// `imagewrap` uses Blender `float` coordinates and texels; linear sRGB input is encoded back to
/// the byte values used by its no-color-management image path.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the input is checked against Blender's signed integer texel range"
)]
fn floor_to_texture_i64(value: f32) -> Option<i64> {
    let value = f64::from(value.floor());
    const UPPER: f64 = 9_223_372_036_854_775_808.0;
    const LOWER: f64 = -9_223_372_036_854_775_808.0;
    if !(LOWER..UPPER).contains(&value) {
        None
    } else {
        Some(value as i64)
    }
}

fn invalid_image_sample(message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        message,
        json!({"feature_id": "texture.image"}),
    )
}

fn empty_image_sample(settings: &ImageSamplerSettings) -> TextureResult {
    if settings.use_color_ramp {
        let ramp = evaluate_color_ramp(&settings.color_ramp, 0.0);
        let intensity = (ramp[0] + ramp[1] + ramp[2]) * (1.0 / 3.0);
        return TextureResult {
            intensity: f64::from(intensity),
            rgb: [f64::from(ramp[0]), f64::from(ramp[1]), f64::from(ramp[2])],
            alpha: f64::from(ramp[3]),
            is_rgb: true,
        };
    }
    TextureResult {
        intensity: 0.0,
        rgb: [0.0; 3],
        alpha: 0.0,
        is_rgb: true,
    }
}

fn repeated_coordinate(value: f32, repeat: u32, mirror: bool) -> Option<f32> {
    if repeat == 1 {
        return Some(value);
    }
    let repeat = f32::from(u16::try_from(repeat).ok()?);
    let original = value * repeat;
    let original_floor = floor_to_texture_i64(original)?;
    let mut mapped = original;
    if original > 1.0 {
        mapped -= original.trunc();
    } else if original < 0.0 {
        mapped += 1.0 - original.trunc();
    }
    if mirror && original_floor.rem_euclid(2) == 1 {
        mapped = 1.0 - mapped;
    }
    Some(mapped)
}

fn srgb_encoded_channel(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "ImageData channels are converted to Blender's float texel values"
)]
fn image_pixel(
    pixels: &[[f64; 4]],
    width: usize,
    row: i64,
    column: i64,
    settings: &ImageSamplerSettings,
) -> Result<[f32; 4]> {
    let row =
        usize::try_from(row).map_err(|_| invalid_image_sample("image texture row is invalid"))?;
    let column = usize::try_from(column)
        .map_err(|_| invalid_image_sample("image texture column is invalid"))?;
    let index = row
        .checked_mul(width)
        .and_then(|index| index.checked_add(column))
        .ok_or_else(|| invalid_image_sample("image texture texel index overflows"))?;
    let pixel = pixels
        .get(index)
        .copied()
        .ok_or_else(|| invalid_image_sample("image texture texel index is invalid"))?;
    let mut texel = pixel.map(|value| value as f32);
    if settings.source_srgb_encoded {
        texel[0] = srgb_encoded_channel(texel[0]);
        texel[1] = srgb_encoded_channel(texel[1]);
        texel[2] = srgb_encoded_channel(texel[2]);
    }
    Ok(texel)
}

#[derive(Clone, Copy, Default)]
struct ImageFilterRect {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

fn wrap_filter_rects_x(rects: &mut [ImageFilterRect; 8], count: &mut usize, width: f32) {
    let initial_count = *count;
    for index in 0..initial_count {
        let mut rect = rects[index];
        if rect.min_x < 0.0 {
            if rect.max_x < 0.0 {
                rect.min_x += width;
                rect.max_x += width;
            } else {
                rect.max_x = rect.max_x.min(width);
                let new_min = rect.min_x + width;
                if new_min != width && *count < rects.len() {
                    rects[*count] = ImageFilterRect {
                        min_x: new_min,
                        max_x: width,
                        min_y: rect.min_y,
                        max_y: rect.max_y,
                    };
                    *count += 1;
                }
                rect.min_x = 0.0;
            }
        } else if rect.max_x > width {
            if rect.min_x > width {
                rect.min_x -= width;
                rect.max_x -= width;
            } else {
                rect.min_x = rect.min_x.max(0.0);
                let new_max = rect.max_x - width;
                if new_max != 0.0 && *count < rects.len() {
                    rects[*count] = ImageFilterRect {
                        min_x: 0.0,
                        max_x: new_max,
                        min_y: rect.min_y,
                        max_y: rect.max_y,
                    };
                    *count += 1;
                }
                rect.max_x = width;
            }
        }
        rects[index] = rect;
    }
}

fn wrap_filter_rects_y(rects: &mut [ImageFilterRect; 8], count: &mut usize, height: f32) {
    let initial_count = *count;
    for index in 0..initial_count {
        let mut rect = rects[index];
        if rect.min_y < 0.0 {
            if rect.max_y < 0.0 {
                rect.min_y += height;
                rect.max_y += height;
            } else {
                rect.max_y = rect.max_y.min(height);
                let new_min = rect.min_y + height;
                if new_min != height && *count < rects.len() {
                    rects[*count] = ImageFilterRect {
                        min_x: rect.min_x,
                        max_x: rect.max_x,
                        min_y: new_min,
                        max_y: height,
                    };
                    *count += 1;
                }
                rect.min_y = 0.0;
            }
        } else if rect.max_y > height {
            if rect.min_y > height {
                rect.min_y -= height;
                rect.max_y -= height;
            } else {
                rect.min_y = rect.min_y.max(0.0);
                let new_max = rect.max_y - height;
                if new_max != 0.0 && *count < rects.len() {
                    rects[*count] = ImageFilterRect {
                        min_x: rect.min_x,
                        max_x: rect.max_x,
                        min_y: 0.0,
                        max_y: new_max,
                    };
                    *count += 1;
                }
                rect.max_y = height;
            }
        }
        rects[index] = rect;
    }
}

fn clip_filter_rect_x(rect: &mut ImageFilterRect, minimum: f32, maximum: f32) -> f32 {
    let original_size = rect.max_x - rect.min_x;
    rect.min_x = rect.min_x.max(minimum);
    rect.max_x = rect.max_x.min(maximum);
    if rect.min_x > rect.max_x {
        rect.min_x = rect.max_x;
        0.0
    } else if original_size != 0.0 {
        (rect.max_x - rect.min_x) / original_size
    } else {
        1.0
    }
}

fn clip_filter_rect_y(rect: &mut ImageFilterRect, minimum: f32, maximum: f32) -> f32 {
    let original_size = rect.max_y - rect.min_y;
    rect.min_y = rect.min_y.max(minimum);
    rect.max_y = rect.max_y.min(maximum);
    if rect.min_y > rect.max_y {
        rect.min_y = rect.max_y;
        0.0
    } else if original_size != 0.0 {
        (rect.max_y - rect.min_y) / original_size
    } else {
        1.0
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Blender filters integer texel indices in float space"
)]
fn sample_filter_rect(
    pixels: &[[f64; 4]],
    width: usize,
    width_px: u32,
    height_px: u32,
    rect: ImageFilterRect,
    settings: &ImageSamplerSettings,
    alpha_weighted: bool,
) -> Result<([f32; 3], f32)> {
    let width_i = i64::from(width_px);
    let height_i = i64::from(height_px);
    let start_x = floor_to_texture_i64(rect.min_x)
        .ok_or_else(|| invalid_image_sample("image texture filter coordinate is out of range"))?
        .max(0);
    let end_x = floor_to_texture_i64(rect.max_x)
        .ok_or_else(|| invalid_image_sample("image texture filter coordinate is out of range"))?
        .min(width_i - 1);
    let start_y = floor_to_texture_i64(rect.min_y)
        .ok_or_else(|| invalid_image_sample("image texture filter coordinate is out of range"))?
        .max(0);
    let end_y = floor_to_texture_i64(rect.max_y)
        .ok_or_else(|| invalid_image_sample("image texture filter coordinate is out of range"))?
        .min(height_i - 1);
    if start_x > end_x || start_y > end_y {
        return Ok(([0.0; 3], 0.0));
    }
    if start_x == end_x && start_y == end_y {
        let texel = image_pixel(pixels, width, start_y, start_x, settings)?;
        let rgb = if alpha_weighted {
            [
                texel[0] * texel[3],
                texel[1] * texel[3],
                texel[2] * texel[3],
            ]
        } else {
            [texel[0], texel[1], texel[2]]
        };
        return Ok((rgb, texel[3]));
    }
    let mut rgb_sum = [0.0_f32; 3];
    let mut alpha_sum = 0.0_f32;
    let mut total_weight = 0.0_f32;
    let mut row = start_y;
    while row <= end_y {
        let row_float = row as f32;
        let mut row_weight = 1.0;
        if start_y != end_y {
            if row == start_y {
                row_weight = 1.0 - (rect.min_y - row_float);
            }
            if row == end_y {
                row_weight = rect.max_y - row_float;
            }
        }
        if start_x == end_x {
            let texel = image_pixel(pixels, width, row, start_x, settings)?;
            let rgb = if alpha_weighted {
                [
                    texel[0] * texel[3],
                    texel[1] * texel[3],
                    texel[2] * texel[3],
                ]
            } else {
                [texel[0], texel[1], texel[2]]
            };
            for channel in 0..3 {
                rgb_sum[channel] += row_weight * rgb[channel];
            }
            if alpha_weighted {
                alpha_sum += row_weight * texel[3];
            }
            total_weight += row_weight;
        } else {
            let mut column = start_x;
            while column <= end_x {
                let column_float = column as f32;
                let mut weight = row_weight;
                if column == start_x {
                    weight *= 1.0 - (rect.min_x - column_float);
                }
                if column == end_x {
                    weight *= rect.max_x - column_float;
                }
                if weight > 0.0 {
                    let texel = image_pixel(pixels, width, row, column, settings)?;
                    for channel in 0..3 {
                        let value = if alpha_weighted {
                            texel[channel] * texel[3]
                        } else {
                            texel[channel]
                        };
                        rgb_sum[channel] += weight * value;
                    }
                    if alpha_weighted {
                        alpha_sum += weight * texel[3];
                    }
                    total_weight += weight;
                }
                if column == end_x {
                    break;
                }
                column += 1;
            }
        }
        if row == end_y {
            break;
        }
        row += 1;
    }
    if total_weight == 0.0 {
        return Ok(([0.0; 3], 0.0));
    }
    Ok((
        rgb_sum.map(|value| value / total_weight),
        if alpha_weighted {
            alpha_sum / total_weight
        } else {
            1.0
        },
    ))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Blender converts image dimensions to float before box filtering"
)]
fn filtered_image_pixels(
    pixels: &[[f64; 4]],
    width: usize,
    width_px: u32,
    height_px: u32,
    center_x: f32,
    center_y: f32,
    settings: &ImageSamplerSettings,
) -> Result<([f32; 3], f32, f32)> {
    let width_float = width_px as f32;
    let height_float = height_px as f32;
    let half_filter = 0.5 * settings.filter_size;
    let mut rects = [ImageFilterRect::default(); 8];
    rects[0] = ImageFilterRect {
        min_x: center_x * width_float - half_filter,
        min_y: center_y * height_float - half_filter,
        max_x: center_x * width_float + half_filter,
        max_y: center_y * height_float + half_filter,
    };
    let mut count = 1_usize;
    let mut alpha_clip = 1.0_f32;
    match settings.extension {
        ImageExtension::Extend => {
            rects[0].min_x = rects[0].min_x.clamp(0.0, width_float - 1.0);
            rects[0].max_x = rects[0].max_x.clamp(0.0, width_float - 1.0);
            rects[0].min_y = rects[0].min_y.clamp(0.0, height_float - 1.0);
            rects[0].max_y = rects[0].max_y.clamp(0.0, height_float - 1.0);
        }
        ImageExtension::Repeat => {
            wrap_filter_rects_x(&mut rects, &mut count, width_float);
            wrap_filter_rects_y(&mut rects, &mut count, height_float);
        }
        ImageExtension::Clip | ImageExtension::ClipCube | ImageExtension::Checker => {
            alpha_clip = clip_filter_rect_x(&mut rects[0], 0.0, width_float);
            if alpha_clip <= 0.0 {
                return Ok(([0.0; 3], 0.0, 0.0));
            }
            alpha_clip *= clip_filter_rect_y(&mut rects[0], 0.0, height_float);
            if alpha_clip <= 0.0 {
                return Ok(([0.0; 3], 0.0, 0.0));
            }
        }
    }
    let alpha_weighted = settings.use_alpha && !settings.calculate_alpha;
    if count == 1 {
        let (rgb, alpha) = sample_filter_rect(
            pixels,
            width,
            width_px,
            height_px,
            rects[0],
            settings,
            alpha_weighted,
        )?;
        return Ok((rgb, alpha, alpha_clip));
    }
    let mut rgb_sum = [0.0_f32; 3];
    let mut alpha_sum = 0.0_f32;
    let mut total_area = 0.0_f32;
    for rect in rects.into_iter().take(count) {
        let (rgb, alpha) = sample_filter_rect(
            pixels,
            width,
            width_px,
            height_px,
            rect,
            settings,
            alpha_weighted,
        )?;
        let area = (rect.max_x - rect.min_x) * (rect.max_y - rect.min_y);
        total_area += area;
        for channel in 0..3 {
            rgb_sum[channel] += area * rgb[channel];
        }
        if alpha_weighted {
            alpha_sum += area * alpha;
        }
    }
    if total_area == 0.0 {
        return Ok(([0.0; 3], 0.0, alpha_clip));
    }
    Ok((
        rgb_sum.map(|value| value / total_area),
        if alpha_weighted {
            alpha_sum / total_area
        } else {
            1.0
        },
        alpha_clip,
    ))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Blender evaluates texture coordinates and dimensions as f32"
)]
fn sample_image(
    image: &ImageData,
    coords: DVec3,
    settings: &ImageSamplerSettings,
) -> Result<TextureResult> {
    if image.width == 0 && image.height == 0 && image.pixels.is_empty() && image.tiles.is_empty() {
        return Ok(empty_image_sample(settings));
    }
    if !coords.is_finite() {
        return Err(invalid_image_sample(
            "image texture coordinates must be finite",
        ));
    }
    let (mut fx, mut fy, fz) = (coords.x as f32, coords.y as f32, coords.z as f32);
    if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
        return Err(invalid_image_sample(
            "image texture coordinates exceed Blender's float range",
        ));
    }
    fx = (fx + 1.0) / 2.0;
    fy = (fy + 1.0) / 2.0;
    if settings.extension == ImageExtension::Repeat {
        fx = repeated_coordinate(fx, settings.repeat_x, settings.mirror_x)
            .ok_or_else(|| invalid_image_sample("image repeat X coordinate is out of range"))?;
        fy = repeated_coordinate(fy, settings.repeat_y, settings.mirror_y)
            .ok_or_else(|| invalid_image_sample("image repeat Y coordinate is out of range"))?;
    }
    fx = settings.crop_min_x + fx * (settings.crop_max_x - settings.crop_min_x);
    fy = settings.crop_min_y + fy * (settings.crop_max_y - settings.crop_min_y);
    let mut width_px = image.width;
    let mut height_px = image.height;
    let mut pixels = image.pixels.as_slice();
    if !image.tiles.is_empty() {
        let u = floor_to_texture_i64(fx)
            .ok_or_else(|| invalid_image_sample("UDIM U coordinate is out of range"))?;
        let v = floor_to_texture_i64(fy)
            .ok_or_else(|| invalid_image_sample("UDIM V coordinate is out of range"))?;
        let tile_number = i64::from(1001_u32)
            .checked_add(
                v.checked_mul(10)
                    .ok_or_else(|| invalid_image_sample("UDIM tile number overflows"))?,
            )
            .and_then(|number| number.checked_add(u))
            .and_then(|number| u32::try_from(number).ok());
        let Some(tile_number) = tile_number else {
            return Ok(empty_image_sample(settings));
        };
        if let Some(tile) = image.tiles.get(&tile_number) {
            width_px = tile.width;
            height_px = tile.height;
            pixels = tile.pixels.as_slice();
        } else if tile_number != 1001 {
            return Ok(empty_image_sample(settings));
        }
        fx = fx.rem_euclid(1.0);
        fy = fy.rem_euclid(1.0);
    }
    if settings.flip_axis {
        std::mem::swap(&mut fx, &mut fy);
    }
    let width = usize::try_from(width_px)
        .map_err(|_| invalid_image_sample("image texture width exceeds addressable memory"))?;
    let height = usize::try_from(height_px)
        .map_err(|_| invalid_image_sample("image texture height exceeds addressable memory"))?;
    if width == 0 || height == 0 || width.checked_mul(height) != Some(pixels.len()) {
        return Err(invalid_image_sample(
            "image texture dimensions do not match its pixels",
        ));
    }
    if settings.extension == ImageExtension::Checker {
        let tile_x = floor_to_texture_i64(fx)
            .ok_or_else(|| invalid_image_sample("checker X coordinate is out of range"))?;
        let tile_y = floor_to_texture_i64(fy)
            .ok_or_else(|| invalid_image_sample("checker Y coordinate is out of range"))?;
        let odd = tile_x.rem_euclid(2) != tile_y.rem_euclid(2);
        if (!settings.checker_odd && !odd) || (!settings.checker_even && odd) {
            return Ok(empty_image_sample(settings));
        }
        fx -= fx.floor();
        fy -= fy.floor();
        if settings.checker_distance < 1.0 {
            fx = (fx - 0.5) / (1.0 - settings.checker_distance) + 0.5;
            fy = (fy - 0.5) / (1.0 - settings.checker_distance) + 0.5;
        }
    }
    let width_float = width_px as f32;
    let height_float = height_px as f32;
    let raw_x = floor_to_texture_i64(fx * width_float)
        .ok_or_else(|| invalid_image_sample("image texture x coordinate is out of range"))?;
    let raw_y = floor_to_texture_i64(fy * height_float)
        .ok_or_else(|| invalid_image_sample("image texture y coordinate is out of range"))?;
    let width_i = i64::from(width_px);
    let height_i = i64::from(height_px);
    let (x, y) = match settings.extension {
        ImageExtension::Clip | ImageExtension::Checker => {
            if raw_x < 0 || raw_y < 0 || raw_x >= width_i || raw_y >= height_i {
                return Ok(empty_image_sample(settings));
            }
            (raw_x, raw_y)
        }
        ImageExtension::ClipCube => {
            if raw_x < 0
                || raw_y < 0
                || raw_x >= width_i
                || raw_y >= height_i
                || !(-1.0..=1.0).contains(&fz)
            {
                return Ok(empty_image_sample(settings));
            }
            (raw_x, raw_y)
        }
        ImageExtension::Extend => (raw_x.clamp(0, width_i - 1), raw_y.clamp(0, height_i - 1)),
        ImageExtension::Repeat => (raw_x.rem_euclid(width_i), raw_y.rem_euclid(height_i)),
    };
    let alpha_active = settings.use_alpha && !settings.calculate_alpha;
    let (mut rgb, sampled_alpha, coverage) = if settings.interpolation {
        let adjusted_x = fx - (raw_x - x) as f32 / width_float;
        let adjusted_y = fy - (raw_y - y) as f32 / height_float;
        filtered_image_pixels(
            pixels, width, width_px, height_px, adjusted_x, adjusted_y, settings,
        )?
    } else {
        let texel = image_pixel(pixels, width, y, x, settings)?;
        let rgb = if alpha_active {
            [
                texel[0] * texel[3],
                texel[1] * texel[3],
                texel[2] * texel[3],
            ]
        } else {
            [texel[0], texel[1], texel[2]]
        };
        (rgb, texel[3], 1.0)
    };
    rgb = rgb.map(|value| value * coverage);
    let tint = if alpha_active {
        sampled_alpha * coverage
    } else if settings.calculate_alpha {
        rgb[0].max(rgb[1]).max(rgb[2])
    } else {
        1.0
    };
    let mut alpha = if settings.invert_alpha {
        1.0 - tint
    } else {
        tint
    };
    if alpha_active && alpha != 1.0 && alpha > 1e-4 {
        rgb = rgb.map(|value| value / alpha);
    }
    rgb[0] = settings.factor_red * ((rgb[0] - 0.5) * settings.contrast + settings.brightness - 0.5);
    rgb[1] =
        settings.factor_green * ((rgb[1] - 0.5) * settings.contrast + settings.brightness - 0.5);
    rgb[2] =
        settings.factor_blue * ((rgb[2] - 0.5) * settings.contrast + settings.brightness - 0.5);
    if settings.use_clamp {
        rgb = rgb.map(|value| value.max(0.0));
    }
    if settings.saturation != 1.0 {
        let hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2]);
        rgb = hsv_to_rgb(hsv[0], hsv[1] * settings.saturation, hsv[2]);
        if settings.saturation > 1.0 && settings.use_clamp {
            rgb = rgb.map(|value| value.max(0.0));
        }
    }
    if settings.use_color_ramp {
        let ramp = evaluate_color_ramp(&settings.color_ramp, tint);
        rgb = [ramp[0], ramp[1], ramp[2]];
        alpha = ramp[3];
    }
    let intensity = (rgb[0] + rgb[1] + rgb[2]) * (1.0 / 3.0);
    Ok(TextureResult {
        intensity: f64::from(intensity),
        rgb: rgb.map(f64::from),
        alpha: f64::from(alpha),
        is_rgb: true,
    })
}

fn evaluate_color_ramp(ramp: &ColorRamp, input: f32) -> [f32; 4] {
    let stops = &ramp.stops[..usize::from(ramp.stop_count)];
    if let Some(only) = stops.first().filter(|_| stops.len() == 1) {
        return only.color;
    }
    let first = stops[0];
    if input <= first.position
        && matches!(
            ramp.interpolation,
            RampInterpolation::Linear | RampInterpolation::Ease | RampInterpolation::Constant
        )
    {
        return first.color;
    }
    let right_index = stops
        .iter()
        .position(|stop| stop.position > input)
        .unwrap_or(stops.len());
    let (left, right, a) = if right_index == stops.len() {
        let last = stops[stops.len() - 1];
        if matches!(
            ramp.interpolation,
            RampInterpolation::Linear | RampInterpolation::Ease | RampInterpolation::Constant
        ) {
            return last.color;
        }
        (
            last,
            RampStop {
                position: 1.0,
                color: last.color,
            },
            right_index,
        )
    } else if right_index == 0 {
        (
            RampStop {
                position: 0.0,
                color: first.color,
            },
            first,
            right_index,
        )
    } else {
        (stops[right_index - 1], stops[right_index], right_index)
    };
    if ramp.interpolation == RampInterpolation::Constant {
        return left.color;
    }
    let mut factor = if right.position == left.position {
        if a == stops.len() { 1.0 } else { 0.0 }
    } else {
        (input - right.position) / (left.position - right.position)
    };
    factor = factor.clamp(0.0, 1.0);
    if ramp.color_mode == RampColorMode::Rgb {
        match ramp.interpolation {
            RampInterpolation::Linear => return lerp_rgba(right.color, left.color, factor),
            RampInterpolation::Ease => {
                factor = factor * factor * (3.0 - 2.0 * factor);
                return lerp_rgba(right.color, left.color, factor);
            }
            RampInterpolation::Constant => return left.color,
            RampInterpolation::Cardinal | RampInterpolation::BSpline => {
                let p0 = if a >= stops.len() - 1 {
                    right.color
                } else {
                    stops[a + 1].color
                };
                let p3 = if a < 2 {
                    left.color
                } else {
                    stops[a - 2].color
                };
                let t = factor;
                let t2 = t * t;
                let t3 = t2 * t;
                let weights = if ramp.interpolation == RampInterpolation::Cardinal {
                    let cardinal = 0.71;
                    [
                        -cardinal * t3 + 2.0 * cardinal * t2 - cardinal * t,
                        (2.0 - cardinal) * t3 + (cardinal - 3.0) * t2 + 1.0,
                        (cardinal - 2.0) * t3 + (3.0 - 2.0 * cardinal) * t2 + cardinal * t,
                        cardinal * t3 - cardinal * t2,
                    ]
                } else {
                    [
                        -t3 / 6.0 + 0.5 * t2 - 0.5 * t + 1.0 / 6.0,
                        0.5 * t3 - t2 + 2.0 / 3.0,
                        -0.5 * t3 + 0.5 * t2 + 0.5 * t + 1.0 / 6.0,
                        t3 / 6.0,
                    ]
                };
                let output = std::array::from_fn(|channel| {
                    weights[0] * p0[channel]
                        + weights[1] * right.color[channel]
                        + weights[2] * left.color[channel]
                        + weights[3] * p3[channel]
                });
                return output.map(|value| value.clamp(0.0, 1.0));
            }
        }
    }
    let right_rgb = [right.color[0], right.color[1], right.color[2]];
    let left_rgb = [left.color[0], left.color[1], left.color[2]];
    let right_color = match ramp.color_mode {
        RampColorMode::Hsv => rgb_to_hsv(right_rgb[0], right_rgb[1], right_rgb[2]),
        RampColorMode::Hsl => rgb_to_hsl(right_rgb[0], right_rgb[1], right_rgb[2]),
        RampColorMode::Rgb => unreachable!(),
    };
    let left_color = match ramp.color_mode {
        RampColorMode::Hsv => rgb_to_hsv(left_rgb[0], left_rgb[1], left_rgb[2]),
        RampColorMode::Hsl => rgb_to_hsl(left_rgb[0], left_rgb[1], left_rgb[2]),
        RampColorMode::Rgb => unreachable!(),
    };
    let hue = interpolate_hue(
        right_color[0],
        left_color[0],
        factor,
        ramp.hue_interpolation,
    );
    let color = [
        hue,
        right_color[1] + (left_color[1] - right_color[1]) * factor,
        right_color[2] + (left_color[2] - right_color[2]) * factor,
    ];
    let rgb = if ramp.color_mode == RampColorMode::Hsv {
        hsv_to_rgb(color[0], color[1], color[2])
    } else {
        hsl_to_rgb(color[0], color[1], color[2])
    };
    [
        rgb[0],
        rgb[1],
        rgb[2],
        right.color[3] + (left.color[3] - right.color[3]) * factor,
    ]
}

fn lerp_rgba(first: [f32; 4], second: [f32; 4], factor: f32) -> [f32; 4] {
    std::array::from_fn(|index| first[index] + (second[index] - first[index]) * factor)
}

fn interpolate_hue(first: f32, second: f32, factor: f32, mode: RampHueInterpolation) -> f32 {
    let first = first.rem_euclid(1.0);
    let second = second.rem_euclid(1.0);
    let mut from = first;
    let mut to = second;
    match mode {
        RampHueInterpolation::Near if first < second && second - first > 0.5 => from += 1.0,
        RampHueInterpolation::Near if first > second && second - first < -0.5 => to += 1.0,
        RampHueInterpolation::Far if first == second => from += 1.0,
        RampHueInterpolation::Far if first < second && second - first < 0.5 => from += 1.0,
        RampHueInterpolation::Far if first > second && second - first > -0.5 => to += 1.0,
        RampHueInterpolation::CounterClockwise if first > second => to += 1.0,
        RampHueInterpolation::Clockwise if first < second => from += 1.0,
        _ => {}
    }
    (from * (1.0 - factor) + to * factor).rem_euclid(1.0)
}

fn rgb_to_hsv(mut red: f32, mut green: f32, mut blue: f32) -> [f32; 3] {
    let mut offset = 0.0_f32;
    let mut minimum_green_blue;
    if green < blue {
        std::mem::swap(&mut green, &mut blue);
        offset = -1.0;
    }
    minimum_green_blue = blue;
    if red < green {
        std::mem::swap(&mut red, &mut green);
        offset = -2.0 / 6.0 - offset;
        minimum_green_blue = green.min(blue);
    }
    let chroma = red - minimum_green_blue;
    let hue = (offset + (green - blue) / (6.0 * chroma + 1e-20)).abs();
    let saturation = chroma / (red + 1e-20);
    [hue, saturation, red]
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let red = (hue * 6.0 - 3.0).abs() - 1.0;
    let green = 2.0 - (hue * 6.0 - 2.0).abs();
    let blue = 2.0 - (hue * 6.0 - 4.0).abs();
    let red = red.clamp(0.0, 1.0);
    let green = green.clamp(0.0, 1.0);
    let blue = blue.clamp(0.0, 1.0);
    [
        ((red - 1.0) * saturation + 1.0) * value,
        ((green - 1.0) * saturation + 1.0) * value,
        ((blue - 1.0) * saturation + 1.0) * value,
    ]
}

fn rgb_to_hsl(red: f32, green: f32, blue: f32) -> [f32; 3] {
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let lightness = ((maximum + minimum) / 2.0).min(1.0);
    if maximum == minimum {
        return [0.0, 0.0, lightness];
    }
    let saturation = if lightness > 0.5 {
        delta / (2.0 - maximum - minimum)
    } else {
        delta / (maximum + minimum)
    };
    let mut hue = if maximum == red {
        (green - blue) / delta + if green < blue { 6.0 } else { 0.0 }
    } else if maximum == green {
        (blue - red) / delta + 2.0
    } else {
        (red - green) / delta + 4.0
    };
    hue /= 6.0;
    [hue, saturation, lightness]
}

fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> [f32; 3] {
    let red = (hue * 6.0 - 3.0).abs() - 1.0;
    let green = 2.0 - (hue * 6.0 - 2.0).abs();
    let blue = 2.0 - (hue * 6.0 - 4.0).abs();
    let red = red.clamp(0.0, 1.0);
    let green = green.clamp(0.0, 1.0);
    let blue = blue.clamp(0.0, 1.0);
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    [
        (red - 0.5) * chroma + lightness,
        (green - 0.5) * chroma + lightness,
        (blue - 0.5) * chroma + lightness,
    ]
}
const fn hex_digit(byte: u8) -> u32 {
    match byte {
        b'0'..=b'9' => (byte - b'0') as u32,
        b'a'..=b'f' => (byte - b'a') as u32 + 10,
        b'A'..=b'F' => (byte - b'A') as u32 + 10,
        _ => 0,
    }
}

const fn decode_u32_table<const N: usize>(hex: &str) -> [u32; N] {
    let bytes = hex.as_bytes();
    let mut output = [0_u32; N];
    let mut index = 0;
    while index < N {
        let mut value = 0_u32;
        let mut digit = 0;
        while digit < 8 {
            value = (value << 4) | hex_digit(bytes[index * 8 + digit]);
            digit += 1;
        }
        output[index] = value;
        index += 1;
    }
    output
}

const fn decode_u8_table<const N: usize>(hex: &str) -> [u8; N] {
    let bytes = hex.as_bytes();
    let mut output = [0_u8; N];
    let mut index = 0;
    while index < N {
        output[index] =
            ((hex_digit(bytes[index * 2]) << 4) | hex_digit(bytes[index * 2 + 1])) as u8;
        index += 1;
    }
    output
}

const HASH512_HEX: &str = concat!(
    "a2a0193bf8ebaaeef31c67281ded00de952edc3f3a82354d6cba36d0f60c7932d159f4088b63892fb8b49783f28f18c7",
    "51146587482042a880b54013b2227e57bc7f6b9d864cc8db7cd5254e5a557450cdb37abbc3cbb6e2e4ecfd980b96d39e",
    "5ca164f18161e1c42472498c904b843438ab78ca1f01d79311c158a931f9446dbf339c5f0994a38506c69a1e7b461530",
    "272b1b713c5bd66f62ac4fc2c00eb123a7df47b0776905e9e6e776730ffe6e9b56ef12a537fcaed9038edd10b9cec98d",
    "da2abd68179fbed40accd2e8433d70b7027d99d80d608a042c3e92e5af5307e029a6c5e3f5f74a41266a165e522d21ad",
    "f091ffea54fa661a4539cf75a488fb5da2a0193bf8ebaaeef31c67281ded00de952edc3f3a82354d6cba36d0f60c7932",
    "d159f4088b63892fb8b49783f28f18c751146587482042a880b54013b2227e57bc7f6b9d864cc8db7cd5254e5a557450",
    "cdb37abbc3cbb6e2e4ecfd980b96d39e5ca164f18161e1c42472498c904b843438ab78ca1f01d79311c158a931f9446d",
    "bf339c5f0994a38506c69a1e7b461530272b1b713c5bd66f62ac4fc2c00eb123a7df47b0776905e9e6e776730ffe6e9b",
    "56ef12a537fcaed9038edd10b9cec98dda2abd68179fbed40accd2e8433d70b7027d99d80d608a042c3e92e5af5307e0",
    "29a6c5e3f5f74a41266a165e522d21adf091ffea54fa661a4539cf75a488fb5d",
);
const HASH512: [u8; 512] = decode_u8_table::<512>(HASH512_HEX);

const HASHPOINT_HEX: &str = concat!(
    "3f0972693cab55ef3f005eb33e5d81f13f045c793f528d223f771ee23ec12f2b3f2dc37a3f3e9a803dc827313ecaef4e",
    "3e7d22853f0559813f1d07853f0ac8a33e1656263e82cf743f4f950c3f237dfa3f7b11e83e95aa713eb744f63dea411c",
    "3f5c83343f212fb23f38f7ba3f36cf743d46db513d9af7d33f109e993e25ea253d5f17be3ed2cf533e20a99b3f633d7d",
    "3f19702e3d980fdc3e2e5d183e6737da3ec94b7b3e9a4a493d6b3fea3e9673603ee2af793e199a203ecc269a3e3d0310",
    "3f6a4a8c3f21882b3f795aaf3df0153c3f4ba7333f4363913e22cc2d3f1dc0013e806c223f6812f13ea23d4f3f2d1bc5",
    "3e705a713f2aec9e3f4acf423e8c1e9b3df474973e4be8ff3f5b51bd3f541c1d3f6695743f34b9cb3f22cbda3f7d4b7b",
    "3cdf52763e9127d43de84be43ec6b7023f6671ef3f2342ff3ee0aaa83f372a5a3d32f6623e4f80dc3ec1b3d03ee69100",
    "3ef193f73e1759ab3f682a883f0630883f48f4843d52dfd73f64b3d53f18ce1c3e8d20083ec870763f583c7d3dd311a5",
    "3e83bfb63f35590c3f010c1c3e4c4e2f3f290bfa3ec08c613d1f40a33f0802d13e5d4d833e0e00083f6861743f4ed8b2",
    "3f28da5e3f6a4ebe3ee5ed293f208b763ef5d93c3ec30ad43c9a5ca33e58e4323f11d7ed3da889c23f3082493f12b40f",
    "3f1807253e5d44133ebf0eb23ddd6c723f18753e3cb252ce3ec2b64a3f72b8523ec173543ea3af963ea168513f7909c0",
    "3f4ad2353ee40af23f5867173ebe8f2a3dc4e8b83f307cd93ed911063d640b783df18a873ea82b633f1b163c3f21bb8c",
    "3ebe8d113e5ada343d05f06f3ee4fedf3ee19e523f3a7e313e9986553e182a563f2632c23eff2bcb3d4c79f63efe0264",
    "3f1b6da83f10063a3f7d80c33f3d5a4b3dde7afa3f7a686e3e56d97b3e8451b93f6042ea3e9e57543f19c5b53f36ae4b",
    "3f13b41c3f2432ec3f5a9b283ceefc373ed612a53f14e3473c0986fd3f1703f33f295cea3f7ab51c3e8d2bb23de38d1e",
    "3ee1858c3df77a703f0587b53f25f7833e91b3d03e5168b63e1cf7013f528bb13e9a009f3e9b99913ebabc303e560e0f",
    "3f6ab5703f6676753f19c67e3f63d72c3f14e7ab3edcc0383f34a02c3f0d1eb83ed58adb3eceb8303f3257c93e95d67c",
    "3f694fad3f30e0443ea5be5e3d169f493e8498813e8c6c763e667a533f2103da3eb40aaf3d85718f3da4894c3eef3e25",
    "3e05c7473ec596853f6b583a3e81f4d33f7756753c8edc3c3ec9042e3ef5243a3f7a6a553d94df053f7b64f13dc8ad69",
    "3f3b45bf3ecd6e693ddc4da93eb2fd113ef54f7b3f335a643ef6a8113f49d67c3f34f50a3df5efa63f100c843f7b570c",
    "3aa04d123e44bb1b3ee730a53e313f073ddf266c3f0cb2313f1680173f2854263ecaf05a3e00281c3f2a9a803ec58a98",
    "3f6413123edf9af23e0330513ebd3bf33f42541e3d36ff3b3f679dd43db4549b3f1f038e3ba5614e3ee797ee3e25ebb7",
    "3efd0cc33f05fa333f3de8813f32cd9f3ee795f63f1040af3f51dcd73efbfc653b89f40a3f25ac263f0d2d773eb9da7b",
    "3e36351e3f50919f3f3a4c733f5dfbf83f7f45ae3ddd7d393daf911d3f4bc83b3f2a425b3eb9949f3ea570823d3311a5",
    "3d2d7e893ea14b9d3ea86ca93f5eefb33f2ee5473eef93793f03d0183f4f5a423f21c9603e34cf073ebb8e4c3f59c24c",
    "3f016c0d3f3fe2933f4036653ecdad753ef661f23ee0750c3f0242d03f5e33593f27929f3d6e67513f1103213dac4934",
    "3f0c6d0d3f6716b13f4791f73f0fe3083d44f1a23f5cf4b63da262cc3f36abcd3f4893323e87d0f23f2c0e063f4974d6",
    "3e127b2d3db087873f6049fd3ed1422d3e6ad0c43f21475a3e539ffd3e54488c3f35c7bd3eaef3d43e87a3ba3ceb80ed",
    "3f2114b13ef0ba843f39ea143e003fb33ea810413f4b4fd73f3dc8753efad3843e41f1063f3982ed3dbe236c3d230ec3",
    "3f5c31383e7da5553f27c3c63f76d0a23f2c2b523d365cb33f30c9003f56553b3f21ab4b3f7427e93eb478c83e93fb37",
    "3d0be2f83f5a425b3dc9a2403f4ba7ca3ee784453f537f283e3f7b183f521b093ee172ac3f6c112c3f3460243f6a5a14",
    "3f3cf9903f3bc41a3e4672fc3f6de0e73e25436c3ea318983f63852f3edca7d63f3da77e3e4630663f5f4bf93f10259e",
    "3e8c981c3eb2137f3f2a8a833ee650403f4d0dbf3f16b6d03d5634553f36fa943ed75b813f1200d23d6bb98c3e57f306",
    "3ed08b333f29b0683f6ca44c3f655d813f4673603f30486b3cd2b4d93e9a94573f4a99b73f0027743f54fce73f54192a",
    "3f2460353efd04603f06a0123ee489f43ee14ff03f435b703f215f243e6551503eaae89f3f67f1c73eff21f73e771194",
    "3edaf3a13f45ce8e3e4ad6023e66c9c13f00ff323edf89e73c87314d3f4dc0ec3e426f613e950a673eff8bdf3e1b35f4",
    "3da1db013f67950c3f35628d3e8c3c9f3d8a12f93ea25e143f6fbe663f376d443f0b396d3f709de03f539a5b3f371023",
    "3e1e94ab3f4009d53f5886e03f25ea683f0e7afa3f250ecb3e1041033f4c96353f220b913ee374153ef1a71a3edf2f98",
    "3eb836ca3e5d6e483bfa1e3f3eacaa323f5b9ce93ec2afbc3dbd798e3f4ccb083f6b43963e97f5e83dc6a22b3f5ad634",
    "3e29567e3f1187943e5e0f3d3ca795703f55e40d3f09bced3f7fef173f1ec7223e79e9083d7764f13ea5612c3e96f9ba",
    "3f7d287c3ec5141a3f3c53e33f45ff613f0cbe1f3e95f4a13f291dcd3f17f0133f1f01973e3379b73f2e9f5a3f4b63e5",
    "3f5e42293f21d4953f4ecd7d3f0278d13ef6cd4b3ae94ee43e84ac2a3eb74c0a3e8f7ea23e459ddc3e94aafc3e551b0d",
    "3f6ae9ff3dea56153f6cecfa3f7b2a063e82da123f79908e3c88398a3f0d77213f1363c33f1cec203f1dafe23f4db9b6",
    "3f45b7ae3db7cc3a3f56bc623f2271583de8e0443f417e353f13dcdf3f2ae08f3f07a3333ea6bce83f514e093ea213c2",
    "3eab914f3f3cc3d23e859b5f3f3cb0d13d4aea743f3c4c6a3f7d27f63e9925bb3e173d5c3ed5a48b3f70d8fd3f062f5a",
    "3f5ba3ca3e34a2fd3e7ab6073ef8b5673d089d283e8f85f93f402d173f417cb33f6c98e53dc484943f7666aa3e8d1959",
    "3e31e2583f7155d63f34ebab3da0b1bc3f13a3a93f4dfea83f0f2af13f668e713f259b803edd3da73f6d89df3ec43ce2",
    "3e8a39e73de9f8c23f132a7c3e1a51e33e77a8d63e8bb5bf3e4cbfb13da343b73f5e5a573f767c393f4a03e63ea3da51",
    "3f67d2633e7083563f7dc9823d7d987c3f3ba00e3f4934ad3f5fc3503d8d62393f289c1b3e9eab583cc1f42c3f4aa67f",
    "3f43c0d33f2949843ea3a0623f5463f13defcd683f6733223f7b6ac63e2582993e4638c93f2ba12b3f5b7de93b766a55",
    "3f1294793e637e6f3dea99b73ed719b53f07c88e3ef0fd3b3f0688f03f43a7eb3d2647fe3ee2f4f13f0049aa3f50efed",
    "3ed5be3c3eaa12733ee5273d3f1d55d63f0f2abf3f4e27d03e67c6323e1ec95c3f36f1fe3e24c9863f428f6d3ee82b63",
    "3dc03e643ecfeb4a3e8793ff3f386a2b3f3e4ead3ebf1d1d3ece62dc3f6973c93f65b21c3e1691213f40c7083f041eac",
    "3f3d3ff23e3337eb3d3b773a3f3c61203f4d23f63f6d7e463e76c2ad3f1784cb3f6bfd1e3f0a6d7a3e18adac3ed8a761",
    "3f4e8f6d3ecb4e333d7a43fe3f4fc6fc3d37e8043eebfd0d3f7611563f5275043f78b5cc3f5e1b8f3f460ae93e680e06",
    "3f2fb0703f6d4e093ed2bef03e8926a63da7ce963e678d0d3e9761463ebfcac93f08826b3e7395c43f2b6cce3f195d81",
    "3f0359473f60f3ba3e4e6a333f38a6db3efd759f3da359363f7749173e6eaf253d57060c3f36f9ba3f6349bf3e9e1112",
    "3f790efe3f392c9d3ee7fb483f6a1af83f343d793f52bd1b3f55b36c3f6814623f6b6ae83dce07e63e9d79073f66e7f7",
    "3ef010e93e87df5d3f629bae3f2f126f3f5e5e143da5f41b3eef04ff3e4bdd763f29d7023f4fef073e9f67903f5244b7",
    "3eb697f23f65e6de3e24a3c63e63e1433f3702f73efb06e63f7c23643f73a6373f76cf103f244cad3f50c2c63f5a4ca5",
    "3f5cb0e23d51569f3ee159e63ea57b633f0465523f101fb83dec1fc93f3e3c003f7a58923eacd9623ee1a2a53f092f5e",
    "3f759d023eb3ee3d3f3451013c315f893e064f553f13bb843eb2fb1a3f465f523eda01ef3d94e22a3f0000113e88de6e",
    "3f68df163e531fcd3e655cd93f700cfa3ea597b33df0a45d3efb09a63ef2e0e33f30a01b3e2cb8173f382bb63e417657",
    "3ea932943e87d1dd3ee47da23e300f343e345c363ef300de3e0f8bdf3de9f9cf3f67b0e63f36c02f3f0e25963f738a8f",
);
const HASHPOINT: [u32; 768] = decode_u32_table::<768>(HASHPOINT_HEX);

const HASHVECTOR_HEX: &str = concat!(
    "3eacf80e3f3737fcbf1c77ffbf71ac04bea737fcbd3ae03bbdceffe3bed53804bf674bf93f4c9e023efcfbfcbeaf13f0",
    "bf5aca043f049e023d0b007a3ed9abf3bee02ff9bf4ac7fcbeb7c8083f18d8023f37a605bf7c44073e142fe83db6dfc6",
    "bf1a0dfebeacaff7bf3951f8bee65c0c3f1832013f2a98033e555001bddcffeb3f78d9f93f1347fa3d7780343f50cdfb",
    "3e9633f0bf602e02bec453f43e9667f93eee77f73f55adfbbf5893fbbe6effe3bef56c0dbded2fe43e127807bf7b9ffd",
    "3e5197e5bf72f406be7547e03f725bfbbe8724103e3cbff4be7140143f12e6003f48c9fb3e50980bbebe97f2bf67cdff",
    "3f41ca08bf26a9f83d6280283efee0093f508ffcbe983ff7be2608073d830fd0bf7c13fdbea893fb3f5599fe3ee243fe",
    "3f342e02bf06ec07bef3d4063f74d8023e883ff73df54fbe3ef5d3fe3ef737fc3f3b7803be3ec80c3f7611fdbe500000",
    "bf3f9bfebf282bfabdb91ff7bdae2fbebe2938153f7b8801bf44fbfc3f20d003bdebffce3ec42bfa3f09a4053f4045fe",
    "3f1dd207bf2b2208bed4f3febe851bf8bf2148023f3b53f83f1229fabdb4202e3f50f3febeea80063cbe7f2c3f637c03",
    "be48e0013f7a0008bdb4cfd1be871befbf32b7febf2a5ffa3edfd007bf646c01bde70004bf1f2607be6bf7f03f3fa205",
    "3e6e00083f668c05bebc08103e639ffdbf4bd0073f100ffebf2a5e03bf3cd5fa3de9003f3f2b8df83f346df83e6e3001",
    "3f6545fa3edc0810bdead03ebdecaff7bef2cffa3f5f6c053f1f2ffd3f1aa002beff340dbf550bfa3c50fe8b3f0de3fc",
    "bf33d5fe3f34bc06bdb7a030bf3128073f2638003ea17bfabe343809bf7a5e033de51fc53dc68fd2bd4adff83f7e7807",
    "3f22a1f9bf1b4ff8bef49ff9be85c400bf1b8002bf400a073f7bb7fa3e29680e3d9b8fdebe98c3f43ecee3f8bf5d5a08",
    "bf2380023f01fa013f1402083ec6c3fc3ed2fc053f5301fc3f0e6202be55d80e3f4dea033f4747fa3f0fe0053e8f1bef",
    "bf1dd7fe3f482bfabdbb6fe33e48fffc3d5120193f7aac00bdf840183e541febbf788206be319008bec4fbfc3f6813fd",
    "3e255ffebf3925ffbf2be0053e36c7ebbbcc0379bf7be005be01101b3e21c8213f7ab6073f549a02bccb40703f0e7408",
    "bf029803bf44fa053ec4bc063f5072083e4ce810bf0b7dfabde68017bec9600fbf6995ff3f0fd8063e1be015bf5023fb",
    "3e4bb01dbe82500dbf724606bed307f23f30e400bf1803ffbed083fd3ea628023f5a89fcbf30b0083ebb73f3bf1fc600",
    "bedb50093f581a05bea503fbbe595feebe9857f3bf6e4606be8bcff63f041ffb3f4fd1fe3f6da5fc3ebe47fe3bed02cd",
    "3f492c06bef55c10bec8140ebf3474003d715f893f34f2063e3167ec3e5078103f76ac003ed82c0abf678002bd7fdf7f",
    "beba0408be3ad81b3f69e2043e820ff5bf0d63febf4b3ffbbe93ec03bf4407ffbf1315f93d6edfe73e9530063f746df8",
    "3f7235f8be9b9c0f3de3dff4bda010413ee32bfebf648a053e3a77e63f64d5fa3ed1abf3bf2e2802be5a70093f337dfa",
    "3d7fdf7f3f593006bf069004bf3a91fbbf2450053e740fedbf6182023edaf7f53e4f8013bf0851fcbe5b2fec3f51a1fd",
    "3f72f007be9c33f83da30fd03f6cea073ebfd3f63d64a05e3f5228023f10c4053da2003b3e84840e3dcc5fc0bf75f202",
    "be96d80a3f6917f83e9477ff3e0c67e03ea3dc05bf6ff9f8bf688bfc3eb0c804be71600fbf6fabfc3e0e17e33ea54003",
    "3ebbaffb3f6d0a03bdb99fe43e9e5c0cbf2fd406bf285dfb3f2b79fb3cc840e13f3df7f9bf6b01fcbd722079bec8bbf5",
    "3ebb23ff3eec9ff9bf4ed3fa3daabffd3f7f1401bc727fe53f1806083e819bf6bf4380023f799807be296feb3e17d806",
    "3f6b42033e0c981c3ebd33f43e9703f33f31ea033f27ddfd3f716e053e1c47e5be97340d3f167c07bf1925ff3f0b6406",
    "3e50afe63f2da0063f34bc06bef52807bf2938043f13e1fc3f3813fdbe2de00dbf2c7ffebe075819bf771fff3e6667f9",
    "bf7b5c07be14801f3df9d0203f4371fc3f1469f93e91b003bf64a7f83da32fcb3ee28bf4bf4bb7fabf0618053e9b8bf0",
    "3f3c26033f2d9408bbf3fd933e1ec80c3f7c7bfebd6620253e841bfc3f124e05bf476806bf2e79ffbf33bbf9be531013",
    "bf767dfebe356ffc3e507fedbef12407bf03ddfd3f375200bf3d86013f1243fe3eb54bf1bf0f6406bf533205bd99902a",
    "bf0204083f40d1fa3ed60008bf00cc033d1c40073f5d03fb3f1812063f2c6df8bee127f6bdf57ffabbb502acbf7e2403",
    "3dc94035be5ae8193f78cdfbbf1d99fe3f24d9f93ee8880e3f659807bee1d40f3d03c07fbf0e15fd3f403201beb757f7",
    "3ecc2bfa3e9c00113f5d6bfdbf3913f93e9b70043f1edbf9bf22b607bf1f37fcbeea2403be9623f23d94202e3f740a07",
    "befbb7fa3f2939fcbf1117f8be9bbff9bf127408bf42fdfc3f6877ffbecc43f63e020ff5bf060c07bf0cb806bf269df9",
    "bf6ed3fabe4ca81a3e994c013dcb9fdc3e103ff73f7c27fabca6809dbf7ff002bb17fa5d3f73e1fc3e8fcbf7bdede00d",
    "bf78cdfb3da9203a3e61a0063f28c1fd3f348e04be85740c3f3baa04be33bff93f2833f8bf0e21fb3edbe7f3bf365205",
    "3ecd03fbbf6609ff3e37a80d3f403a00be49afe23f215e073f4921ffbf11c0013e779fedbd714014beed9ff53f623fff",
    "3e04b7f6bf71f7f93e99680ebeb7340d3f685c033e607fedbf579607bee0d3f2bea037f8bea0800f3f5826033ede7408",
    "be7700043d5a1fd13f780ffe3d82cfd93f52c2053f1060033ef3dc053f5cb5febe30a8093f1ef5fdbe98a3f93f399408",
    "3e8137f4bf3fd5febf1cb407bf13d9fd3ec5ebfb3f380ffebed00c0fbf3cebff3f09e7fb3d413fd13f0e6a013f546003",
    "bf4d87f93f1663fa3dcf301fbf352dfdbf34b9fe3cd87f883ebff3f13f2f3ffb3f2006083ef1bbf13e55681fbf5b43fa",
    "bf33fc01bf14c1fdbed1d40fbed59c06bf3c7ffe3f0859fbbee55001be4fb7e9bf5ee8083db21038bf7d8bf83ddb4fe8",
    "3e3c17ec3c963fddbf7b97ff3f0f85f9bedb94103f3551f83e97ebf2bf73dc05bd89a0273e0b08143f7d82023cf7c02b",
    "bd8ce0333f279bfe3f40b0083f433a04bf0d40033eacd7f1bf51d207bed0bbf53ece1006bf3a53fcbf0cfbfcbed12c06",
    "bf5b1401bef5c4003e456ffc3f3173fbbda2602d3f3761fe3e676ff43f266e01bf39b5fa3e7c9ff93f740ffebe31b7e1",
    "bf786c01be74f7ecbd04dfce3e0ba7fcbec9e40d3f68a3f93f502ff93f0ee4083e2850163ecf1bef3f0c8e04bf3b37fc",
    "bec2afffbf1113f93f3b19f83cb580803f6ec7fc3eb82ff93c0701113f7f1e08bda8900c3f74e608bd85cfeebe914403",
    "bf3e5a043c067f0b3f2b27febf5c0e06bf0279ff3d12806bbf5a2c023eba3401bec0800fbe4ecfeabf788605be043fe6",
    "3ea0800f3f7284063d860fe4be820ff5bf073bfb3f4f65fe3f20c8043f19e2043efcfbfcbefd20083f5e4e053d1b5fa2",
    "3e8d2008bdb1001d3f7513f9be4aafff3ef0140ebf5c59fb3f6545fa3ecc7c0f3e48800f3f0f90003f5333fdbd8dc012",
    "3e9bbff9bf5977fb3edcaff73dc560423f6f8df83ead9ff5bd529fe03ed163febf693c043ea92ff53f715e07bd2da017",
    "bee79bfebf4951f83ed754093e098809bf6ef805beaa3c003f4d49f9bf1113f9be4177eabf7cea07bdd9103d3de66fd6",
    "be79f01c3f6469f9bec273f7bf6ba6053e6a97e1bea1fc093f4765fe3ea6e3f83f092407bf69b1fb3e3da7f4beba4010",
    "be3d38043f10bc06bf4dc800bc973fb83df4cfd13f7e1c04be84bc06bf6f84023e7547e0bf52dff8bee5f3f9beb11ff7",
    "beddbff03f28d5fa3f1d45fabf52a0023ec1f7f9bed8ec103f2fec03bf2cc1fdbe89abf3be7c7ffebf4d03fb3f0bb3fa",
    "bf3aac00be54c7f33f26e8083f274e01bf1c45febee5100a3efc33f8bcbf401c3f5eb3ff3f1be9fb3c14ff00bf4b05fb",
    "3f766602be8acc0cbc10ff973e88240bbba0fd3b3f76c6043f26b7fe3f0db9fabf04d9f93e8fa3fdbf57d8063eeacffa",
    "be338002bf7bb3fa3d4c00543d10bfa1bf7adbf93e48c005bf7ba2053e284018bda870113e6f57f7bf78e608bba400fc",
    "bf3f4bf9bf1c7ffe3e8558003f2545fa3f17bff93ef677f73df10fd8bf731401be94d402bf08f606bf4aba06be96b010",
    "bf208e04be5b97ff3f3fabfcbd448063bf504007bf146003be33bff93ef45bf2bf5c69f93f3d1202bed4180dbf082603",
    "3f1003ffbf343dff3eddd80ebf4144033f25e3fcbdce00083deaf0393d365fd9bf7e0c073f774bf93e7abffdbda9802c",
);
const HASHVECTOR: [u32; 768] = decode_u32_table::<768>(HASHVECTOR_HEX);

const PERLIN_PERM_HEX: &str = concat!(
    "a2a0193bf8ebaaeef31c67281ded00de952edc3f3a82354d6cba36d0f60c7932d159f4088b63892fb8b49783f28f18c7",
    "51146587482042a880b54013b2227e57bc7f6b9d864cc8db7cd5254e5a557450cdb37abbc3cbb6e2e4ecfd980b96d39e",
    "5ca164f18161e1c42472498c904b843438ab78ca1f01d79311c158a931f9446dbf339c5f0994a38506c69a1e7b461530",
    "272b1b713c5bd66f62ac4fc2c00eb123a7df47b0776905e9e6e776730ffe6e9b56ef12a537fcaed9038edd10b9cec98d",
    "da2abd68179fbed40accd2e8433d70b7027d99d80d608a042c3e92e5af5307e029a6c5e3f5f74a41266a165e522d21ad",
    "f091ffea54fa661a4539cf75a488fb5da2a0193bf8ebaaeef31c67281ded00de952edc3f3a82354d6cba36d0f60c7932",
    "d159f4088b63892fb8b49783f28f18c751146587482042a880b54013b2227e57bc7f6b9d864cc8db7cd5254e5a557450",
    "cdb37abbc3cbb6e2e4ecfd980b96d39e5ca164f18161e1c42472498c904b843438ab78ca1f01d79311c158a931f9446d",
    "bf339c5f0994a38506c69a1e7b461530272b1b713c5bd66f62ac4fc2c00eb123a7df47b0776905e9e6e776730ffe6e9b",
    "56ef12a537fcaed9038edd10b9cec98dda2abd68179fbed40accd2e8433d70b7027d99d80d608a042c3e92e5af5307e0",
    "29a6c5e3f5f74a41266a165e522d21adf091ffea54fa661a4539cf75a488fb5da2a0",
);
const PERLIN_PERM: [u8; 514] = decode_u8_table::<514>(PERLIN_PERM_HEX);

const PERLIN_GRAD_HEX: &str = concat!(
    "3eacf80e3f3737fcbf1c77ffbf71ac04bea737fcbd3ae03bbdceffe3bed53804bf674bf93f4c9e023efcfbfcbeaf13f0",
    "bf5aca043f049e023d0b007a3ed9abf3bee02ff9bf4ac7fcbeb7c8083f18d8023f37a605bf7c44073e142fe83db6dfc6",
    "bf1a0dfebeacaff7bf3951f8bee65c0c3f1832013f2a98033e555001bddcffeb3f78d9f93f1347fa3d7780343f50cdfb",
    "3e9633f0bf602e02bec453f43e9667f93eee77f73f55adfbbf5893fbbe6effe3bef56c0dbded2fe43e127807bf7b9ffd",
    "3e5197e5bf72f406be7547e03f725bfbbe8724103e3cbff4be7140143f12e6003f48c9fb3e50980bbebe97f2bf67cdff",
    "3f41ca08bf26a9f83d6280283efee0093f508ffcbe983ff7be2608073d830fd0bf7c13fdbea893fb3f5599fe3ee243fe",
    "3f342e02bf06ec07bef3d4063f74d8023e883ff73df54fbe3ef5d3fe3ef737fc3f3b7803be3ec80c3f7611fdbe500000",
    "bf3f9bfebf282bfabdb91ff7bdae2fbebe2938153f7b8801bf44fbfc3f20d003bdebffce3ec42bfa3f09a4053f4045fe",
    "3f1dd207bf2b2208bed4f3febe851bf8bf2148023f3b53f83f1229fabdb4202e3f50f3febeea80063cbe7f2c3f637c03",
    "be48e0013f7a0008bdb4cfd1be871befbf32b7febf2a5ffa3edfd007bf646c01bde70004bf1f2607be6bf7f03f3fa205",
    "3e6e00083f668c05bebc08103e639ffdbf4bd0073f100ffebf2a5e03bf3cd5fa3de9003f3f2b8df83f346df83e6e3001",
    "3f6545fa3edc0810bdead03ebdecaff7bef2cffa3f5f6c053f1f2ffd3f1aa002beff340dbf550bfa3c50fe8b3f0de3fc",
    "bf33d5fe3f34bc06bdb7a030bf3128073f2638003ea17bfabe343809bf7a5e033de51fc53dc68fd2bd4adff83f7e7807",
    "3f22a1f9bf1b4ff8bef49ff9be85c400bf1b8002bf400a073f7bb7fa3e29680e3d9b8fdebe98c3f43ecee3f8bf5d5a08",
    "bf2380023f01fa013f1402083ec6c3fc3ed2fc053f5301fc3f0e6202be55d80e3f4dea033f4747fa3f0fe0053e8f1bef",
    "bf1dd7fe3f482bfabdbb6fe33e48fffc3d5120193f7aac00bdf840183e541febbf788206be319008bec4fbfc3f6813fd",
    "3e255ffebf3925ffbf2be0053e36c7ebbbcc0379bf7be005be01101b3e21c8213f7ab6073f549a02bccb40703f0e7408",
    "bf029803bf44fa053ec4bc063f5072083e4ce810bf0b7dfabde68017bec9600fbf6995ff3f0fd8063e1be015bf5023fb",
    "3e4bb01dbe82500dbf724606bed307f23f30e400bf1803ffbed083fd3ea628023f5a89fcbf30b0083ebb73f3bf1fc600",
    "bedb50093f581a05bea503fbbe595feebe9857f3bf6e4606be8bcff63f041ffb3f4fd1fe3f6da5fc3ebe47fe3bed02cd",
    "3f492c06bef55c10bec8140ebf3474003d715f893f34f2063e3167ec3e5078103f76ac003ed82c0abf678002bd7fdf7f",
    "beba0408be3ad81b3f69e2043e820ff5bf0d63febf4b3ffbbe93ec03bf4407ffbf1315f93d6edfe73e9530063f746df8",
    "3f7235f8be9b9c0f3de3dff4bda010413ee32bfebf648a053e3a77e63f64d5fa3ed1abf3bf2e2802be5a70093f337dfa",
    "3d7fdf7f3f593006bf069004bf3a91fbbf2450053e740fedbf6182023edaf7f53e4f8013bf0851fcbe5b2fec3f51a1fd",
    "3f72f007be9c33f83da30fd03f6cea073ebfd3f63d64a05e3f5228023f10c4053da2003b3e84840e3dcc5fc0bf75f202",
    "be96d80a3f6917f83e9477ff3e0c67e03ea3dc05bf6ff9f8bf688bfc3eb0c804be71600fbf6fabfc3e0e17e33ea54003",
    "3ebbaffb3f6d0a03bdb99fe43e9e5c0cbf2fd406bf285dfb3f2b79fb3cc840e13f3df7f9bf6b01fcbd722079bec8bbf5",
    "3ebb23ff3eec9ff9bf4ed3fa3daabffd3f7f1401bc727fe53f1806083e819bf6bf4380023f799807be296feb3e17d806",
    "3f6b42033e0c981c3ebd33f43e9703f33f31ea033f27ddfd3f716e053e1c47e5be97340d3f167c07bf1925ff3f0b6406",
    "3e50afe63f2da0063f34bc06bef52807bf2938043f13e1fc3f3813fdbe2de00dbf2c7ffebe075819bf771fff3e6667f9",
    "bf7b5c07be14801f3df9d0203f4371fc3f1469f93e91b003bf64a7f83da32fcb3ee28bf4bf4bb7fabf0618053e9b8bf0",
    "3f3c26033f2d9408bbf3fd933e1ec80c3f7c7bfebd6620253e841bfc3f124e05bf476806bf2e79ffbf33bbf9be531013",
    "bf767dfebe356ffc3e507fedbef12407bf03ddfd3f375200bf3d86013f1243fe3eb54bf1bf0f6406bf533205bd99902a",
    "bf0204083f40d1fa3ed60008bf00cc033d1c40073f5d03fb3f1812063f2c6df8bee127f6bdf57ffabbb502acbf7e2403",
    "3dc94035be5ae8193f78cdfbbf1d99fe3f24d9f93ee8880e3f659807bee1d40f3d03c07fbf0e15fd3f403201beb757f7",
    "3ecc2bfa3e9c00113f5d6bfdbf3913f93e9b70043f1edbf9bf22b607bf1f37fcbeea2403be9623f23d94202e3f740a07",
    "befbb7fa3f2939fcbf1117f8be9bbff9bf127408bf42fdfc3f6877ffbecc43f63e020ff5bf060c07bf0cb806bf269df9",
    "bf6ed3fabe4ca81a3e994c013dcb9fdc3e103ff73f7c27fabca6809dbf7ff002bb17fa5d3f73e1fc3e8fcbf7bdede00d",
    "bf78cdfb3da9203a3e61a0063f28c1fd3f348e04be85740c3f3baa04be33bff93f2833f8bf0e21fb3edbe7f3bf365205",
    "3ecd03fbbf6609ff3e37a80d3f403a00be49afe23f215e073f4921ffbf11c0013e779fedbd714014beed9ff53f623fff",
    "3e04b7f6bf71f7f93e99680ebeb7340d3f685c033e607fedbf579607bee0d3f2bea037f8bea0800f3f5826033ede7408",
    "be7700043d5a1fd13f780ffe3d82cfd93f52c2053f1060033ef3dc053f5cb5febe30a8093f1ef5fdbe98a3f93f399408",
    "3e8137f4bf3fd5febf1cb407bf13d9fd3ec5ebfb3f380ffebed00c0fbf3cebff3f09e7fb3d413fd13f0e6a013f546003",
    "bf4d87f93f1663fa3dcf301fbf352dfdbf34b9fe3cd87f883ebff3f13f2f3ffb3f2006083ef1bbf13e55681fbf5b43fa",
    "bf33fc01bf14c1fdbed1d40fbed59c06bf3c7ffe3f0859fbbee55001be4fb7e9bf5ee8083db21038bf7d8bf83ddb4fe8",
    "3e3c17ec3c963fddbf7b97ff3f0f85f9bedb94103f3551f83e97ebf2bf73dc05bd89a0273e0b08143f7d82023cf7c02b",
    "bd8ce0333f279bfe3f40b0083f433a04bf0d40033eacd7f1bf51d207bed0bbf53ece1006bf3a53fcbf0cfbfcbed12c06",
    "bf5b1401bef5c4003e456ffc3f3173fbbda2602d3f3761fe3e676ff43f266e01bf39b5fa3e7c9ff93f740ffebe31b7e1",
    "bf786c01be74f7ecbd04dfce3e0ba7fcbec9e40d3f68a3f93f502ff93f0ee4083e2850163ecf1bef3f0c8e04bf3b37fc",
    "bec2afffbf1113f93f3b19f83cb580803f6ec7fc3eb82ff93c0701113f7f1e08bda8900c3f74e608bd85cfeebe914403",
    "bf3e5a043c067f0b3f2b27febf5c0e06bf0279ff3d12806bbf5a2c023eba3401bec0800fbe4ecfeabf788605be043fe6",
    "3ea0800f3f7284063d860fe4be820ff5bf073bfb3f4f65fe3f20c8043f19e2043efcfbfcbefd20083f5e4e053d1b5fa2",
    "3e8d2008bdb1001d3f7513f9be4aafff3ef0140ebf5c59fb3f6545fa3ecc7c0f3e48800f3f0f90003f5333fdbd8dc012",
    "3e9bbff9bf5977fb3edcaff73dc560423f6f8df83ead9ff5bd529fe03ed163febf693c043ea92ff53f715e07bd2da017",
    "bee79bfebf4951f83ed754093e098809bf6ef805beaa3c003f4d49f9bf1113f9be4177eabf7cea07bdd9103d3de66fd6",
    "be79f01c3f6469f9bec273f7bf6ba6053e6a97e1bea1fc093f4765fe3ea6e3f83f092407bf69b1fb3e3da7f4beba4010",
    "be3d38043f10bc06bf4dc800bc973fb83df4cfd13f7e1c04be84bc06bf6f84023e7547e0bf52dff8bee5f3f9beb11ff7",
    "beddbff03f28d5fa3f1d45fabf52a0023ec1f7f9bed8ec103f2fec03bf2cc1fdbe89abf3be7c7ffebf4d03fb3f0bb3fa",
    "bf3aac00be54c7f33f26e8083f274e01bf1c45febee5100a3efc33f8bcbf401c3f5eb3ff3f1be9fb3c14ff00bf4b05fb",
    "3f766602be8acc0cbc10ff973e88240bbba0fd3b3f76c6043f26b7fe3f0db9fabf04d9f93e8fa3fdbf57d8063eeacffa",
    "be338002bf7bb3fa3d4c00543d10bfa1bf7adbf93e48c005bf7ba2053e284018bda870113e6f57f7bf78e608bba400fc",
    "bf3f4bf9bf1c7ffe3e8558003f2545fa3f17bff93ef677f73df10fd8bf731401be94d402bf08f606bf4aba06be96b010",
    "bf208e04be5b97ff3f3fabfcbd448063bf504007bf146003be33bff93ef45bf2bf5c69f93f3d1202bed4180dbf082603",
    "3f1003ffbf343dff3eddd80ebf4144033f25e3fcbdce00083deaf0393d365fd9bf7e0c073f774bf93e7abffdbda9802c",
    "3eacf80e3f3737fcbf1c77ffbf71ac04bea737fcbd3ae03bbdceffe3bed53804bf674bf93f4c9e023efcfbfcbeaf13f0",
    "bf5aca043f049e023d0b007a3ed9abf3bee02ff9bf4ac7fcbeb7c8083f18d8023f37a605bf7c44073e142fe83db6dfc6",
    "bf1a0dfebeacaff7bf3951f8bee65c0c3f1832013f2a98033e555001bddcffeb3f78d9f93f1347fa3d7780343f50cdfb",
    "3e9633f0bf602e02bec453f43e9667f93eee77f73f55adfbbf5893fbbe6effe3bef56c0dbded2fe43e127807bf7b9ffd",
    "3e5197e5bf72f406be7547e03f725bfbbe8724103e3cbff4be7140143f12e6003f48c9fb3e50980bbebe97f2bf67cdff",
    "3f41ca08bf26a9f83d6280283efee0093f508ffcbe983ff7be2608073d830fd0bf7c13fdbea893fb3f5599fe3ee243fe",
    "3f342e02bf06ec07bef3d4063f74d8023e883ff73df54fbe3ef5d3fe3ef737fc3f3b7803be3ec80c3f7611fdbe500000",
    "bf3f9bfebf282bfabdb91ff7bdae2fbebe2938153f7b8801bf44fbfc3f20d003bdebffce3ec42bfa3f09a4053f4045fe",
    "3f1dd207bf2b2208bed4f3febe851bf8bf2148023f3b53f83f1229fabdb4202e3f50f3febeea80063cbe7f2c3f637c03",
    "be48e0013f7a0008bdb4cfd1be871befbf32b7febf2a5ffa3edfd007bf646c01bde70004bf1f2607be6bf7f03f3fa205",
    "3e6e00083f668c05bebc08103e639ffdbf4bd0073f100ffebf2a5e03bf3cd5fa3de9003f3f2b8df83f346df83e6e3001",
    "3f6545fa3edc0810bdead03ebdecaff7bef2cffa3f5f6c053f1f2ffd3f1aa002beff340dbf550bfa3c50fe8b3f0de3fc",
    "bf33d5fe3f34bc06bdb7a030bf3128073f2638003ea17bfabe343809bf7a5e033de51fc53dc68fd2bd4adff83f7e7807",
    "3f22a1f9bf1b4ff8bef49ff9be85c400bf1b8002bf400a073f7bb7fa3e29680e3d9b8fdebe98c3f43ecee3f8bf5d5a08",
    "bf2380023f01fa013f1402083ec6c3fc3ed2fc053f5301fc3f0e6202be55d80e3f4dea033f4747fa3f0fe0053e8f1bef",
    "bf1dd7fe3f482bfabdbb6fe33e48fffc3d5120193f7aac00bdf840183e541febbf788206be319008bec4fbfc3f6813fd",
    "3e255ffebf3925ffbf2be0053e36c7ebbbcc0379bf7be005be01101b3e21c8213f7ab6073f549a02bccb40703f0e7408",
    "bf029803bf44fa053ec4bc063f5072083e4ce810bf0b7dfabde68017bec9600fbf6995ff3f0fd8063e1be015bf5023fb",
    "3e4bb01dbe82500dbf724606bed307f23f30e400bf1803ffbed083fd3ea628023f5a89fcbf30b0083ebb73f3bf1fc600",
    "bedb50093f581a05bea503fbbe595feebe9857f3bf6e4606be8bcff63f041ffb3f4fd1fe3f6da5fc3ebe47fe3bed02cd",
    "3f492c06bef55c10bec8140ebf3474003d715f893f34f2063e3167ec3e5078103f76ac003ed82c0abf678002bd7fdf7f",
    "beba0408be3ad81b3f69e2043e820ff5bf0d63febf4b3ffbbe93ec03bf4407ffbf1315f93d6edfe73e9530063f746df8",
    "3f7235f8be9b9c0f3de3dff4bda010413ee32bfebf648a053e3a77e63f64d5fa3ed1abf3bf2e2802be5a70093f337dfa",
    "3d7fdf7f3f593006bf069004bf3a91fbbf2450053e740fedbf6182023edaf7f53e4f8013bf0851fcbe5b2fec3f51a1fd",
    "3f72f007be9c33f83da30fd03f6cea073ebfd3f63d64a05e3f5228023f10c4053da2003b3e84840e3dcc5fc0bf75f202",
    "be96d80a3f6917f83e9477ff3e0c67e03ea3dc05bf6ff9f8bf688bfc3eb0c804be71600fbf6fabfc3e0e17e33ea54003",
    "3ebbaffb3f6d0a03bdb99fe43e9e5c0cbf2fd406bf285dfb3f2b79fb3cc840e13f3df7f9bf6b01fcbd722079bec8bbf5",
    "3ebb23ff3eec9ff9bf4ed3fa3daabffd3f7f1401bc727fe53f1806083e819bf6bf4380023f799807be296feb3e17d806",
    "3f6b42033e0c981c3ebd33f43e9703f33f31ea033f27ddfd3f716e053e1c47e5be97340d3f167c07bf1925ff3f0b6406",
    "3e50afe63f2da0063f34bc06bef52807bf2938043f13e1fc3f3813fdbe2de00dbf2c7ffebe075819bf771fff3e6667f9",
    "bf7b5c07be14801f3df9d0203f4371fc3f1469f93e91b003bf64a7f83da32fcb3ee28bf4bf4bb7fabf0618053e9b8bf0",
    "3f3c26033f2d9408bbf3fd933e1ec80c3f7c7bfebd6620253e841bfc3f124e05bf476806bf2e79ffbf33bbf9be531013",
    "bf767dfebe356ffc3e507fedbef12407bf03ddfd3f375200bf3d86013f1243fe3eb54bf1bf0f6406bf533205bd99902a",
    "bf0204083f40d1fa3ed60008bf00cc033d1c40073f5d03fb3f1812063f2c6df8bee127f6bdf57ffabbb502acbf7e2403",
    "3dc94035be5ae8193f78cdfbbf1d99fe3f24d9f93ee8880e3f659807bee1d40f3d03c07fbf0e15fd3f403201beb757f7",
    "3ecc2bfa3e9c00113f5d6bfdbf3913f93e9b70043f1edbf9bf22b607bf1f37fcbeea2403be9623f23d94202e3f740a07",
    "befbb7fa3f2939fcbf1117f8be9bbff9bf127408bf42fdfc3f6877ffbecc43f63e020ff5bf060c07bf0cb806bf269df9",
    "bf6ed3fabe4ca81a3e994c013dcb9fdc3e103ff73f7c27fabca6809dbf7ff002bb17fa5d3f73e1fc3e8fcbf7bdede00d",
    "bf78cdfb3da9203a3e61a0063f28c1fd3f348e04be85740c3f3baa04be33bff93f2833f8bf0e21fb3edbe7f3bf365205",
    "3ecd03fbbf6609ff3e37a80d3f403a00be49afe23f215e073f4921ffbf11c0013e779fedbd714014beed9ff53f623fff",
    "3e04b7f6bf71f7f93e99680ebeb7340d3f685c033e607fedbf579607bee0d3f2bea037f8bea0800f3f5826033ede7408",
    "be7700043d5a1fd13f780ffe3d82cfd93f52c2053f1060033ef3dc053f5cb5febe30a8093f1ef5fdbe98a3f93f399408",
    "3e8137f4bf3fd5febf1cb407bf13d9fd3ec5ebfb3f380ffebed00c0fbf3cebff3f09e7fb3d413fd13f0e6a013f546003",
    "bf4d87f93f1663fa3dcf301fbf352dfdbf34b9fe3cd87f883ebff3f13f2f3ffb3f2006083ef1bbf13e55681fbf5b43fa",
    "bf33fc01bf14c1fdbed1d40fbed59c06bf3c7ffe3f0859fbbee55001be4fb7e9bf5ee8083db21038bf7d8bf83ddb4fe8",
    "3e3c17ec3c963fddbf7b97ff3f0f85f9bedb94103f3551f83e97ebf2bf73dc05bd89a0273e0b08143f7d82023cf7c02b",
    "bd8ce0333f279bfe3f40b0083f433a04bf0d40033eacd7f1bf51d207bed0bbf53ece1006bf3a53fcbf0cfbfcbed12c06",
    "bf5b1401bef5c4003e456ffc3f3173fbbda2602d3f3761fe3e676ff43f266e01bf39b5fa3e7c9ff93f740ffebe31b7e1",
    "bf786c01be74f7ecbd04dfce3e0ba7fcbec9e40d3f68a3f93f502ff93f0ee4083e2850163ecf1bef3f0c8e04bf3b37fc",
    "bec2afffbf1113f93f3b19f83cb580803f6ec7fc3eb82ff93c0701113f7f1e08bda8900c3f74e608bd85cfeebe914403",
    "bf3e5a043c067f0b3f2b27febf5c0e06bf0279ff3d12806bbf5a2c023eba3401bec0800fbe4ecfeabf788605be043fe6",
    "3ea0800f3f7284063d860fe4be820ff5bf073bfb3f4f65fe3f20c8043f19e2043efcfbfcbefd20083f5e4e053d1b5fa2",
    "3e8d2008bdb1001d3f7513f9be4aafff3ef0140ebf5c59fb3f6545fa3ecc7c0f3e48800f3f0f90003f5333fdbd8dc012",
    "3e9bbff9bf5977fb3edcaff73dc560423f6f8df83ead9ff5bd529fe03ed163febf693c043ea92ff53f715e07bd2da017",
    "bee79bfebf4951f83ed754093e098809bf6ef805beaa3c003f4d49f9bf1113f9be4177eabf7cea07bdd9103d3de66fd6",
    "be79f01c3f6469f9bec273f7bf6ba6053e6a97e1bea1fc093f4765fe3ea6e3f83f092407bf69b1fb3e3da7f4beba4010",
    "be3d38043f10bc06bf4dc800bc973fb83df4cfd13f7e1c04be84bc06bf6f84023e7547e0bf52dff8bee5f3f9beb11ff7",
    "beddbff03f28d5fa3f1d45fabf52a0023ec1f7f9bed8ec103f2fec03bf2cc1fdbe89abf3be7c7ffebf4d03fb3f0bb3fa",
    "bf3aac00be54c7f33f26e8083f274e01bf1c45febee5100a3efc33f8bcbf401c3f5eb3ff3f1be9fb3c14ff00bf4b05fb",
    "3f766602be8acc0cbc10ff973e88240bbba0fd3b3f76c6043f26b7fe3f0db9fabf04d9f93e8fa3fdbf57d8063eeacffa",
    "be338002bf7bb3fa3d4c00543d10bfa1bf7adbf93e48c005bf7ba2053e284018bda870113e6f57f7bf78e608bba400fc",
    "bf3f4bf9bf1c7ffe3e8558003f2545fa3f17bff93ef677f73df10fd8bf731401be94d402bf08f606bf4aba06be96b010",
    "bf208e04be5b97ff3f3fabfcbd448063bf504007bf146003be33bff93ef45bf2bf5c69f93f3d1202bed4180dbf082603",
    "3f1003ffbf343dff3eddd80ebf4144033f25e3fcbdce00083deaf0393d365fd9bf7e0c073f774bf93e7abffdbda9802c",
    "3eacf80e3f3737fcbf1c77ffbf71ac04bea737fcbd3ae03b",
);
const PERLIN_GRAD: [u32; 1542] = decode_u32_table::<1542>(PERLIN_GRAD_HEX);

fn table_float(table: &[u32], index: usize) -> f32 {
    table.get(index).copied().map(f32::from_bits).unwrap_or(0.0)
}

fn hash_value(index: usize) -> usize {
    HASH512.get(index).copied().map(usize::from).unwrap_or(0)
}

fn noise_hash(x: i32, y: i32, z: i32) -> usize {
    let zi = (z & 255) as usize;
    let yi = (hash_value(zi).wrapping_add(y as usize)) & 255;
    (hash_value(yi).wrapping_add(x as usize)) & 255
}

fn hash_point(x: i32, y: i32, z: i32) -> [f32; 3] {
    let index = noise_hash(x, y, z) * 3;
    [
        table_float(&HASHPOINT, index),
        table_float(&HASHPOINT, index + 1),
        table_float(&HASHPOINT, index + 2),
    ]
}

fn org_blender_noise(x: f32, y: f32, z: f32) -> f32 {
    let fx = x.floor();
    let fy = y.floor();
    let fz = z.floor();
    let (ox, oy, oz) = (x - fx, y - fy, z - fz);
    let (ix, iy, iz) = (fx as i32, fy as i32, fz as i32);
    let (jx, jy, jz) = (ox - 1.0, oy - 1.0, oz - 1.0);
    let mut cn = [ox * ox, oy * oy, oz * oz, jx * jx, jy * jy, jz * jz];
    cn[0] = 1.0 - 3.0 * cn[0] + 2.0 * cn[0] * ox;
    cn[1] = 1.0 - 3.0 * cn[1] + 2.0 * cn[1] * oy;
    cn[2] = 1.0 - 3.0 * cn[2] + 2.0 * cn[2] * oz;
    cn[3] = 1.0 - 3.0 * cn[3] - 2.0 * cn[3] * jx;
    cn[4] = 1.0 - 3.0 * cn[4] - 2.0 * cn[4] * jy;
    cn[5] = 1.0 - 3.0 * cn[5] - 2.0 * cn[5] * jz;
    let b00 = hash_value(hash_value((ix & 255) as usize).wrapping_add((iy & 255) as usize));
    let b10 = hash_value(hash_value(((ix + 1) & 255) as usize).wrapping_add((iy & 255) as usize));
    let b01 = hash_value(hash_value((ix & 255) as usize).wrapping_add(((iy + 1) & 255) as usize));
    let b11 =
        hash_value(hash_value(((ix + 1) & 255) as usize).wrapping_add(((iy + 1) & 255) as usize));
    let b20 = (iz & 255) as usize;
    let b21 = ((iz + 1) & 255) as usize;
    let mut value = 0.5_f32;
    for (cx, cy, cz, xx, yy, zz, bz, bb) in [
        (0, 1, 2, ox, oy, oz, b20, b00),
        (0, 1, 5, ox, oy, jz, b21, b00),
        (0, 4, 2, ox, jy, oz, b20, b01),
        (0, 4, 5, ox, jy, jz, b21, b01),
        (3, 1, 2, jx, oy, oz, b20, b10),
        (3, 1, 5, jx, oy, jz, b21, b10),
        (3, 4, 2, jx, jy, oz, b20, b11),
        (3, 4, 5, jx, jy, jz, b21, b11),
    ] {
        let weight = cn[cx] * cn[cy] * cn[cz];
        let h = hash_value(bz + bb) * 3;
        let gx = table_float(&HASHVECTOR, h);
        let gy = table_float(&HASHVECTOR, h + 1);
        let gz = table_float(&HASHVECTOR, h + 2);
        value += weight * (gx * xx + gy * yy + gz * zz);
    }
    value.clamp(0.0, 1.0)
}

fn lerp_noise(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

fn noise_fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn noise_gradient(hash: usize, x: f32, y: f32, z: f32) -> f32 {
    let h = (hash & 15) as i32;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

fn improved_perlin(x: f32, y: f32, z: f32) -> f32 {
    let floor = [x.floor(), y.floor(), z.floor()];
    let frac = [x - floor[0], y - floor[1], z - floor[2]];
    let xyz = [
        (floor[0] as i32 & 255) as usize,
        (floor[1] as i32 & 255) as usize,
        (floor[2] as i32 & 255) as usize,
    ];
    let (x, y, z) = (frac[0], frac[1], frac[2]);
    let (u, v, w) = (noise_fade(x), noise_fade(y), noise_fade(z));
    let a = hash_value(xyz[0]) + xyz[1];
    let aa = hash_value(a) + xyz[2];
    let ab = hash_value(a + 1) + xyz[2];
    let b = hash_value(xyz[0] + 1) + xyz[1];
    let ba = hash_value(b) + xyz[2];
    let bb = hash_value(b + 1) + xyz[2];
    lerp_noise(
        w,
        lerp_noise(
            v,
            lerp_noise(
                u,
                noise_gradient(hash_value(aa), x, y, z),
                noise_gradient(hash_value(ba), x - 1.0, y, z),
            ),
            lerp_noise(
                u,
                noise_gradient(hash_value(ab), x, y - 1.0, z),
                noise_gradient(hash_value(bb), x - 1.0, y - 1.0, z),
            ),
        ),
        lerp_noise(
            v,
            lerp_noise(
                u,
                noise_gradient(hash_value(aa + 1), x, y, z - 1.0),
                noise_gradient(hash_value(ba + 1), x - 1.0, y, z - 1.0),
            ),
            lerp_noise(
                u,
                noise_gradient(hash_value(ab + 1), x, y - 1.0, z - 1.0),
                noise_gradient(hash_value(bb + 1), x - 1.0, y - 1.0, z - 1.0),
            ),
        ),
    )
}

fn original_perlin(x: f32, y: f32, z: f32) -> f32 {
    let coord = [x, y, z];
    let mut base = [0_i32; 3];
    let mut next = [0_i32; 3];
    let mut relative0 = [0_f32; 3];
    let mut relative1 = [0_f32; 3];
    for axis in 0..3 {
        let t = coord[axis] + 10_000.0;
        base[axis] = (t as i32) & 255;
        next[axis] = (base[axis] + 1) & 255;
        relative0[axis] = t - t.floor();
        relative1[axis] = relative0[axis] - 1.0;
    }
    let mut bx0 = [0_usize; 3];
    let mut bx1 = [0_usize; 3];
    for axis in 0..3 {
        bx0[axis] = base[axis] as usize;
        bx1[axis] = next[axis] as usize;
    }
    let p = &PERLIN_PERM;
    let i = usize::from(p.get(bx0[0]).copied().unwrap_or(0));
    let j = usize::from(p.get(bx1[0]).copied().unwrap_or(0));
    let b00 = usize::from(p.get(i + bx0[1]).copied().unwrap_or(0));
    let b10 = usize::from(p.get(j + bx0[1]).copied().unwrap_or(0));
    let b01 = usize::from(p.get(i + bx1[1]).copied().unwrap_or(0));
    let b11 = usize::from(p.get(j + bx1[1]).copied().unwrap_or(0));
    let rx0 = relative0[0];
    let rx1 = relative1[0];
    let ry0 = relative0[1];
    let ry1 = relative1[1];
    let rz0 = relative0[2];
    let rz1 = relative1[2];
    let gdot = |index: usize, rx: f32, ry: f32, rz: f32| -> f32 {
        table_float(&PERLIN_GRAD, index * 3) * rx
            + table_float(&PERLIN_GRAD, index * 3 + 1) * ry
            + table_float(&PERLIN_GRAD, index * 3 + 2) * rz
    };
    let surf = |t: f32| t * t * (3.0 - 2.0 * t);
    let sx = surf(rx0);
    let sy = surf(ry0);
    let sz = surf(rz0);
    let a = lerp_noise(
        sx,
        gdot(b00 + bx0[2], rx0, ry0, rz0),
        gdot(b10 + bx0[2], rx1, ry0, rz0),
    );
    let b = lerp_noise(
        sx,
        gdot(b01 + bx0[2], rx0, ry1, rz0),
        gdot(b11 + bx0[2], rx1, ry1, rz0),
    );
    let c = lerp_noise(sy, a, b);
    let a = lerp_noise(
        sx,
        gdot(b00 + bx1[2], rx0, ry0, rz1),
        gdot(b10 + bx1[2], rx1, ry0, rz1),
    );
    let b = lerp_noise(
        sx,
        gdot(b01 + bx1[2], rx0, ry1, rz1),
        gdot(b11 + bx1[2], rx1, ry1, rz1),
    );
    1.5 * lerp_noise(sz, c, lerp_noise(sy, a, b))
}

fn voronoi_distance(dx: f32, dy: f32, dz: f32, metric: u8, exponent: f32) -> f32 {
    match metric {
        1 => dx * dx + dy * dy + dz * dz,
        2 => dx.abs() + dy.abs() + dz.abs(),
        3 => dx.abs().max(dy.abs()).max(dz.abs()),
        4 => {
            let d = dx.abs().sqrt() + dy.abs().sqrt() + dz.abs().sqrt();
            d * d
        }
        5 => {
            let x = dx * dx;
            let y = dy * dy;
            let z = dz * dz;
            (x * x + y * y + z * z).sqrt().sqrt()
        }
        6 => (dx.abs().powf(exponent) + dy.abs().powf(exponent) + dz.abs().powf(exponent))
            .powf(1.0 / exponent),
        _ => (dx * dx + dy * dy + dz * dz).sqrt(),
    }
}

fn voronoi4(x: f32, y: f32, z: f32, metric: u8, exponent: f32) -> ([f32; 4], [[f32; 3]; 4]) {
    let origin = [x.floor() as i32, y.floor() as i32, z.floor() as i32];
    let mut distances = [1.0e10_f32; 4];
    let mut points = [[0.0_f32; 3]; 4];
    for xx in origin[0].saturating_sub(1)..=origin[0].saturating_add(1) {
        for yy in origin[1].saturating_sub(1)..=origin[1].saturating_add(1) {
            for zz in origin[2].saturating_sub(1)..=origin[2].saturating_add(1) {
                let random = hash_point(xx, yy, zz);
                let point = [
                    random[0] + xx as f32,
                    random[1] + yy as f32,
                    random[2] + zz as f32,
                ];
                let distance =
                    voronoi_distance(x - point[0], y - point[1], z - point[2], metric, exponent);
                let mut slot = 0_usize;
                while slot < 4 && distance >= distances[slot] {
                    slot += 1;
                }
                if slot < 4 {
                    let mut shift = 3;
                    while shift > slot {
                        distances[shift] = distances[shift - 1];
                        points[shift] = points[shift - 1];
                        shift -= 1;
                    }
                    distances[slot] = distance;
                    points[slot] = point;
                }
            }
        }
    }
    (distances, points)
}

fn cell_noise_u(x: f32, y: f32, z: f32) -> f32 {
    let x = ((x + 0.000001) * 1.00001).floor() as i32;
    let y = ((y + 0.000001) * 1.00001).floor() as i32;
    let z = ((z + 0.000001) * 1.00001).floor() as i32;
    let mut n = (x as u32)
        .wrapping_add((y as u32).wrapping_mul(1301))
        .wrapping_add((z as u32).wrapping_mul(314159));
    n ^= n.wrapping_shl(13);
    let inner = n
        .wrapping_mul(n)
        .wrapping_mul(n)
        .wrapping_mul(15731)
        .wrapping_add(n.wrapping_mul(789221))
        .wrapping_add(1376312589);
    inner as f32 / 4_294_967_296.0
}

fn cell_noise_v3(x: f32, y: f32, z: f32) -> [f32; 3] {
    let x = ((x + 0.000_001) * 1.000_01).floor() as i32;
    let y = ((y + 0.000_001) * 1.000_01).floor() as i32;
    let z = ((z + 0.000_001) * 1.000_01).floor() as i32;
    hash_point(x, y, z)
}

fn noise_basis_unsigned(basis: u8, x: f32, y: f32, z: f32) -> f32 {
    match basis {
        0 => org_blender_noise(x, y, z),
        1 => 0.5 + 0.5 * original_perlin(x, y, z),
        2 => 0.5 + 0.5 * improved_perlin(x, y, z),
        3..=8 => {
            let distances = voronoi4(x, y, z, 0, 1.0).0;
            let first = distances.first().copied().unwrap_or(0.0);
            let second = distances.get(1).copied().unwrap_or(0.0);
            match basis {
                3 => first,
                4 => second,
                5 => distances.get(2).copied().unwrap_or(0.0),
                6 => distances.get(3).copied().unwrap_or(0.0),
                7 => second - first,
                _ => (10.0 * (second - first)).min(1.0),
            }
        }
        14 => cell_noise_u(x, y, z),
        _ => org_blender_noise(x, y, z),
    }
}

fn noise_basis_signed(basis: u8, x: f32, y: f32, z: f32) -> f32 {
    if basis == 1 {
        return original_perlin(x, y, z);
    }
    if basis == 2 {
        return improved_perlin(x, y, z);
    }
    let value = noise_basis_unsigned(basis, x, y, z);
    2.0 * value - 1.0
}

fn generic_noise(noise_size: f32, x: f32, y: f32, z: f32, hard: bool, basis: u8) -> f32 {
    let (mut x, mut y, mut z) = (x, y, z);
    if basis == 0 {
        x += 1.0;
        y += 1.0;
        z += 1.0;
    }
    if noise_size != 0.0 {
        let scale = 1.0 / noise_size;
        x *= scale;
        y *= scale;
        z *= scale;
    }
    let value = noise_basis_unsigned(basis, x, y, z);
    if hard {
        (2.0 * value - 1.0).abs()
    } else {
        value
    }
}

fn generic_turbulence(
    noise_size: f32,
    x: f32,
    y: f32,
    z: f32,
    octaves: i32,
    hard: bool,
    basis: u8,
) -> f32 {
    let (mut x, mut y, mut z) = (x, y, z);
    if basis == 0 {
        x += 1.0;
        y += 1.0;
        z += 1.0;
    }
    if noise_size != 0.0 {
        let scale = 1.0 / noise_size;
        x *= scale;
        y *= scale;
        z *= scale;
    }
    let octaves = octaves.clamp(0, 30);
    let mut sum = 0.0_f32;
    let mut amplitude = 1.0_f32;
    let mut frequency = 1.0_f32;
    for _ in 0..=octaves {
        let mut value = noise_basis_unsigned(basis, frequency * x, frequency * y, frequency * z);
        if hard {
            value = (2.0 * value - 1.0).abs();
        }
        sum += value * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    let power = 2.0_f32.powi(octaves);
    let denominator = 2.0_f32.powi(octaves + 1) - 1.0;
    sum * power / denominator
}
fn setting_f32(settings: &Map<String, Value>, key: &str, default: f32) -> f32 {
    settings
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(default)
}

fn setting_text<'a>(settings: &'a Map<String, Value>, key: &str, default: &'a str) -> &'a str {
    settings.get(key).and_then(Value::as_str).unwrap_or(default)
}

fn texture_noise_basis(settings: &Map<String, Value>, key: &str, default: &str) -> u8 {
    match setting_text(settings, key, default) {
        "ORIGINAL_PERLIN" => 1,
        "IMPROVED_PERLIN" => 2,
        "VORONOI_F1" => 3,
        "VORONOI_F2" => 4,
        "VORONOI_F3" => 5,
        "VORONOI_F4" => 6,
        "VORONOI_F2_F1" => 7,
        "VORONOI_CRACKLE" => 8,
        "CELL_NOISE" => 14,
        _ => 0,
    }
}

fn hard_noise(settings: &Map<String, Value>) -> bool {
    setting_text(settings, "noise_type", "SOFT_NOISE") == "HARD_NOISE"
}

fn noise_depth(settings: &Map<String, Value>) -> i32 {
    settings
        .get("noise_depth")
        .and_then(Value::as_i64)
        .unwrap_or(2)
        .clamp(0, 30) as i32
}

fn scalar_result(intensity: f32) -> TextureResult {
    let intensity = f64::from(intensity);
    TextureResult {
        intensity,
        rgb: [intensity; 3],
        alpha: 1.0,
        is_rgb: false,
    }
}

fn rgb_result(rgb: [f32; 3]) -> TextureResult {
    let rgb = rgb.map(f64::from);
    TextureResult {
        intensity: (rgb[0] + rgb[1] + rgb[2]) / 3.0,
        rgb,
        alpha: 1.0,
        is_rgb: true,
    }
}

fn texture_waveform(kind: u8, value: f32) -> f32 {
    match kind {
        1 => {
            let period = 2.0 * std::f32::consts::PI;
            let turns = (value / period) as i32;
            let mut wrapped = value - turns as f32 * period;
            if wrapped < 0.0 {
                wrapped += period;
            }
            wrapped / period
        }
        2 => {
            let scaled = value / (2.0 * std::f32::consts::PI);
            1.0 - 2.0 * ((scaled + 0.5).floor() - scaled).abs()
        }
        _ => 0.5 + 0.5 * value.sin(),
    }
}

fn sample_procedural(
    settings: &ProceduralSamplerSettings,
    coords: DVec3,
    rng: &mut BlenderRng,
) -> Result<TextureResult> {
    let point = [coords.x as f32, coords.y as f32, coords.z as f32];
    if point.iter().any(|value| !value.is_finite()) {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "legacy texture coordinates exceed Blender's finite float range",
            json!({"feature_id":"texture.coordinates"}),
        ));
    }
    let (x, y, z) = (point[0], point[1], point[2]);
    match settings.kind {
        0 => {
            let red = generic_turbulence(
                settings.noise_size,
                x,
                y,
                z,
                settings.depth,
                settings.hard_noise,
                settings.basis,
            );
            if settings.cloud_color {
                Ok(rgb_result([
                    red,
                    generic_turbulence(
                        settings.noise_size,
                        y,
                        x,
                        z,
                        settings.depth,
                        settings.hard_noise,
                        settings.basis,
                    ),
                    generic_turbulence(
                        settings.noise_size,
                        y,
                        z,
                        x,
                        settings.depth,
                        settings.hard_noise,
                        settings.basis,
                    ),
                ]))
            } else {
                Ok(scalar_result(red))
            }
        }
        1 => {
            let value = match settings.wood_type {
                1 | 3 => {
                    let noise = if settings.wood_type == 3 {
                        settings.turbulence
                            * generic_noise(
                                settings.noise_size,
                                x,
                                y,
                                z,
                                settings.hard_noise,
                                settings.basis,
                            )
                    } else {
                        0.0
                    };
                    texture_waveform(
                        settings.basis_2,
                        (x * x + y * y + z * z).sqrt() * 20.0 + noise,
                    )
                }
                2 => {
                    let noise = settings.turbulence
                        * generic_noise(
                            settings.noise_size,
                            x,
                            y,
                            z,
                            settings.hard_noise,
                            settings.basis,
                        );
                    texture_waveform(settings.basis_2, (x + y + z) * 10.0 + noise)
                }
                _ => texture_waveform(settings.basis_2, (x + y + z) * 10.0),
            };
            Ok(scalar_result(value))
        }
        2 => {
            let noise = 5.0 * (x + y + z)
                + settings.turbulence
                    * generic_turbulence(
                        settings.noise_size,
                        x,
                        y,
                        z,
                        settings.depth,
                        settings.hard_noise,
                        settings.basis,
                    );
            let mut value = texture_waveform(settings.basis_2, noise);
            if settings.marble_type == 1 {
                value = value.sqrt();
            } else if settings.marble_type == 2 {
                value = value.sqrt().sqrt();
            }
            Ok(scalar_result(value))
        }
        3 => {
            let mut depth = settings.depth;
            let turbulence = settings.turbulence / 5.0;
            let mut px = ((x + y + z) * 5.0).sin();
            let mut py = ((-x + y - z) * 5.0).cos();
            let mut pz = -((-x - y + z) * 5.0).cos();
            if depth > 0 {
                px *= turbulence;
                py *= turbulence;
                pz *= turbulence;
                py = -(px - py + pz).cos();
                py *= turbulence;
                depth -= 1;
                if depth > 0 {
                    px = (px - py - pz).cos() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    pz = (-px - py - pz).sin() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    px = -(-px + py - pz).cos() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    py = -(-px + py + pz).sin() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    py = -(-px + py + pz).cos() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    px = (px + py + pz).cos() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    pz = (px + py - pz).sin() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    px = -(-px - py + pz).cos() * turbulence;
                    depth -= 1;
                }
                if depth > 0 {
                    py = (px - py + pz).sin() * -turbulence;
                }
            }
            let divisor = if turbulence == 0.0 {
                1.0
            } else {
                turbulence * 2.0
            };
            Ok(rgb_result([
                0.5 - px / divisor,
                0.5 - py / divisor,
                0.5 - pz / divisor,
            ]))
        }
        4 => {
            let first = generic_noise(
                settings.noise_size,
                x,
                y,
                z,
                settings.hard_noise,
                settings.basis,
            );
            let mut offset = settings.turbulence / 200.0;
            if settings.stucci_type != 0 {
                offset *= first * first;
            }
            let mut value = generic_noise(
                settings.noise_size,
                x,
                y,
                z + offset,
                settings.hard_noise,
                settings.basis,
            );
            if settings.stucci_type == 2 {
                value = 1.0 - value;
            }
            Ok(scalar_result(value.max(0.0)))
        }
        5 => sample_voronoi(settings, point),
        6 => {
            if settings.noise_size == 0.0 {
                return Err(PotError::invalid_argument(
                    "distorted noise_scale must be nonzero",
                ));
            }
            let p = [
                x / settings.noise_size,
                y / settings.noise_size,
                z / settings.noise_size,
            ];
            let amount = settings.distortion;
            let rv = [
                noise_basis_signed(settings.basis, p[0] + 13.5, p[1] + 13.5, p[2] + 13.5) * amount,
                noise_basis_signed(settings.basis, p[0], p[1], p[2]) * amount,
                noise_basis_signed(settings.basis, p[0] - 13.5, p[1] - 13.5, p[2] - 13.5) * amount,
            ];
            Ok(scalar_result(noise_basis_signed(
                settings.basis_2,
                p[0] + rv[0],
                p[1] + rv[1],
                p[2] + rv[2],
            )))
        }
        7 => sample_musgrave(settings, point),
        8 => {
            let random = rng.get_int32() as u32;
            let mut value = (random >> 29) & 3;
            let mut shift = 29_i32;
            let mut divisor = 3.0_f32;
            for _ in 0..settings.depth {
                shift -= 2;
                value = value.wrapping_mul((random >> (shift as u32)) & 3);
                divisor *= 3.0;
            }
            Ok(scalar_result(value as f32 / divisor))
        }
        _ => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "inline IMAGE textures must reference an image ID in doc.images",
            json!({"feature_id":"texture.image_resource"}),
        )),
    }
}

fn sample_voronoi(settings: &ProceduralSamplerSettings, p: [f32; 3]) -> Result<TextureResult> {
    if settings.noise_size == 0.0 {
        return Err(PotError::invalid_argument(
            "Voronoi noise_scale must be nonzero",
        ));
    }
    let p = [
        p[0] / settings.noise_size,
        p[1] / settings.noise_size,
        p[2] / settings.noise_size,
    ];
    let (distances, points) = voronoi4(
        p[0],
        p[1],
        p[2],
        settings.distance_metric,
        settings.minkovsky_exponent,
    );
    let weights = settings.voronoi_weights;
    let absolute = weights.map(f32::abs);
    let total = absolute.iter().sum::<f32>();
    let intensity_scale = if total != 0.0 {
        settings.noise_intensity / total
    } else {
        0.0
    };
    let intensity = intensity_scale
        * (weights[0] * distances[0]
            + weights[1] * distances[1]
            + weights[2] * distances[2]
            + weights[3] * distances[3])
            .abs();
    match settings.color_mode {
        0 => Ok(scalar_result(intensity)),
        mode => {
            let mut color = [0.0_f32; 3];
            for cell in 0..4 {
                let point = points.get(cell).copied().unwrap_or([0.0; 3]);
                let c = cell_noise_v3(point[0], point[1], point[2]);
                for ch in 0..3 {
                    color[ch] += absolute[cell] * c[ch];
                }
            }
            let outline = ((distances[1] - distances[0]) * 10.0).min(1.0);
            let factor = match mode {
                2 => outline * intensity_scale,
                3 => outline * intensity,
                _ => intensity_scale,
            };
            Ok(rgb_result(color.map(|value| value * factor)))
        }
    }
}

fn sample_musgrave(settings: &ProceduralSamplerSettings, p: [f32; 3]) -> Result<TextureResult> {
    if settings.noise_size == 0.0 {
        return Err(PotError::invalid_argument(
            "Musgrave noise_scale must be nonzero",
        ));
    }
    let (x, y, z) = (
        p[0] / settings.noise_size,
        p[1] / settings.noise_size,
        p[2] / settings.noise_size,
    );
    let value = match settings.musgrave_type {
        3 => musgrave_fbm(
            x,
            y,
            z,
            settings.dimension_max,
            settings.lacunarity,
            settings.octaves,
            settings.basis,
        ),
        4 => musgrave_hetero(
            x,
            y,
            z,
            settings.dimension_max,
            settings.lacunarity,
            settings.octaves,
            settings.offset,
            settings.basis,
        ),
        2 => musgrave_hybrid(
            x,
            y,
            z,
            settings.dimension_max,
            settings.lacunarity,
            settings.octaves,
            settings.offset,
            settings.gain,
            settings.basis,
        ),
        1 => musgrave_ridged(
            x,
            y,
            z,
            settings.dimension_max,
            settings.lacunarity,
            settings.octaves,
            settings.offset,
            settings.gain,
            settings.basis,
        ),
        _ => musgrave_multi(
            x,
            y,
            z,
            settings.dimension_max,
            settings.lacunarity,
            settings.octaves,
            settings.basis,
        ),
    } * settings.noise_intensity;
    Ok(scalar_result(value))
}
fn musgrave_fbm(mut x: f32, mut y: f32, mut z: f32, h: f32, lac: f32, oct: f32, basis: u8) -> f32 {
    let mut value = 0.0;
    let mut power = 1.0;
    let octave_power = lac.powf(-h);
    for _ in 0..oct.floor().max(0.0) as usize {
        value += noise_basis_signed(basis, x, y, z) * power;
        power *= octave_power;
        x *= lac;
        y *= lac;
        z *= lac;
    }
    let remainder = oct - oct.floor();
    if remainder != 0.0 {
        value += remainder * noise_basis_signed(basis, x, y, z) * power;
    }
    value
}
fn musgrave_multi(
    mut x: f32,
    mut y: f32,
    mut z: f32,
    h: f32,
    lac: f32,
    oct: f32,
    basis: u8,
) -> f32 {
    let mut value = 1.0;
    let mut power = 1.0;
    let octave_power = lac.powf(-h);
    for _ in 0..oct.floor().max(0.0) as usize {
        value *= power * noise_basis_signed(basis, x, y, z) + 1.0;
        power *= octave_power;
        x *= lac;
        y *= lac;
        z *= lac;
    }
    let remainder = oct - oct.floor();
    if remainder != 0.0 {
        value *= remainder * noise_basis_signed(basis, x, y, z) * power + 1.0;
    }
    value
}
fn musgrave_hetero(
    mut x: f32,
    mut y: f32,
    mut z: f32,
    h: f32,
    lac: f32,
    oct: f32,
    offset: f32,
    basis: u8,
) -> f32 {
    let mut value = offset + noise_basis_signed(basis, x, y, z);
    x *= lac;
    y *= lac;
    z *= lac;
    let octave_power = lac.powf(-h);
    let mut power = octave_power;
    for _ in 1..oct.floor().max(1.0) as usize {
        value += (noise_basis_signed(basis, x, y, z) + offset) * power * value;
        power *= octave_power;
        x *= lac;
        y *= lac;
        z *= lac;
    }
    let remainder = oct - oct.floor();
    if remainder != 0.0 {
        value += remainder * (noise_basis_signed(basis, x, y, z) + offset) * power * value;
    }
    value
}
fn musgrave_hybrid(
    mut x: f32,
    mut y: f32,
    mut z: f32,
    h: f32,
    lac: f32,
    oct: f32,
    offset: f32,
    gain: f32,
    basis: u8,
) -> f32 {
    let mut result = noise_basis_signed(basis, x, y, z) + offset;
    let mut weight = gain * result;
    x *= lac;
    y *= lac;
    z *= lac;
    let octave_power = lac.powf(-h);
    let mut power = octave_power;
    for _ in 1..oct.floor().max(1.0) as usize {
        if weight <= 0.001 {
            break;
        }
        weight = weight.min(1.0);
        let signal = (noise_basis_signed(basis, x, y, z) + offset) * power;
        power *= octave_power;
        result += weight * signal;
        weight *= gain * signal;
        x *= lac;
        y *= lac;
        z *= lac;
    }
    let remainder = oct - oct.floor();
    if remainder != 0.0 {
        result += remainder * (noise_basis_signed(basis, x, y, z) + offset) * power;
    }
    result
}
fn musgrave_ridged(
    mut x: f32,
    mut y: f32,
    mut z: f32,
    h: f32,
    lac: f32,
    oct: f32,
    offset: f32,
    gain: f32,
    basis: u8,
) -> f32 {
    let mut signal = (offset - noise_basis_signed(basis, x, y, z).abs()).powi(2);
    let mut result = signal;
    let octave_power = lac.powf(-h);
    let mut power = octave_power;
    for _ in 1..oct.floor().max(1.0) as usize {
        x *= lac;
        y *= lac;
        z *= lac;
        let weight = (signal * gain).clamp(0.0, 1.0);
        signal = offset - noise_basis_signed(basis, x, y, z).abs();
        signal *= signal;
        signal *= weight;
        result += signal * power;
        power *= octave_power;
    }
    result
}
#[cfg(test)]
mod tests {
    use glam::DVec3;
    use serde_json::{Value, json};

    use super::{
        BlenderRng, DEFAULT_IMAGE_SETTINGS, ImageData, ImageInterpolation, LegacyTexture,
        ResolvedTexture, TextureSource, cell_noise_v3, parse_record, record_value, sample,
    };

    fn image_record(settings: Value) -> Value {
        json!({
            "storage":"inline",
            "kind":"legacy_texture",
            "texture_type":"IMAGE",
            "image":"shared_image",
            "settings":settings,
        })
    }

    fn image_data() -> ImageData {
        ImageData {
            width: 2,
            height: 1,
            pixels: vec![[0.2, 0.4, 0.6, 1.0], [0.8, 0.7, 0.1, 1.0]],
            tiles: Default::default(),
            interpolation: ImageInterpolation::Linear,
        }
    }

    fn assert_rgb_close(actual: [f64; 3], expected: [f64; 3]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn blender_522_cell_noise_v3_matches_hash_point_color() {
        // Blender 5.2.2 BLI_noise_cell_v3(0, 0, 0), using its HASHPNT tables.
        assert_eq!(
            cell_noise_v3(0.0, 0.0, 0.0),
            [0.982_098, 0.161_631, 0.193_576]
        );
    }

    #[test]
    fn image_records_share_pixels_but_keep_independent_sampler_settings() {
        let first = parse_record(
            "texture_a",
            &image_record(json!({"use_interpolation":false})),
        )
        .expect("valid IMAGE resource");
        let second = parse_record(
            "texture_b",
            &image_record(json!({"use_interpolation":false,"factor_red":0.0})),
        )
        .expect("valid IMAGE resource");
        let image = image_data();
        let (first_settings, second_settings) = match (&first, &second) {
            (
                LegacyTexture::ImageRecord { sampler: first, .. },
                LegacyTexture::ImageRecord {
                    sampler: second, ..
                },
            ) => (*first, *second),
            _ => panic!("IMAGE records parse as image sampler resources"),
        };
        let mut rng = BlenderRng::seed(0);
        let first_result = sample(
            TextureSource::ImageWithSettings(&image, &first_settings),
            DVec3::new(-0.5, 0.0, 0.0),
            &mut rng,
        )
        .expect("first image sample");
        let second_result = sample(
            TextureSource::ImageWithSettings(&image, &second_settings),
            DVec3::new(-0.5, 0.0, 0.0),
            &mut rng,
        )
        .expect("second image sample");
        assert_rgb_close(first_result.rgb, [0.2, 0.4, 0.6]);
        assert_rgb_close(second_result.rgb, [0.0, 0.4, 0.6]);
        let record = record_value(&first);
        assert_eq!(record["image"], "shared_image");
        assert_eq!(record["settings"], json!({"use_interpolation":false}));
    }

    #[test]
    fn image_color_ramp_samples_the_pre_color_adjustment_tint() {
        let texture = parse_record(
            "texture_ramp",
            &image_record(json!({
                "use_interpolation":false,
                "use_color_ramp":true,
                "color_ramp":{
                    "elements":[
                        {"position":0.0,"color":[0.0,0.0,0.0,1.0]},
                        {"position":1.0,"color":[1.0,1.0,1.0,1.0]}
                    ],
                    "interpolation":"LINEAR"
                }
            })),
        )
        .expect("valid IMAGE color ramp");
        let image = ImageData {
            width: 1,
            height: 1,
            pixels: vec![[0.2, 0.4, 0.6, 0.5]],
            tiles: Default::default(),
            interpolation: ImageInterpolation::Linear,
        };
        let LegacyTexture::ImageRecord { sampler, .. } = texture else {
            panic!("IMAGE record");
        };
        let mut rng = BlenderRng::seed(0);
        let result = sample(
            TextureSource::ImageWithSettings(&image, &sampler),
            DVec3::ZERO,
            &mut rng,
        )
        .expect("sample with ramp");
        assert_eq!(result.rgb, [0.5; 3]);
        assert_eq!(result.alpha, 1.0);
    }

    #[test]
    fn empty_image_texture_resolves_to_the_empty_image_sample() {
        let texture = parse_record(
            "empty_image",
            &json!({
                "storage":"inline",
                "kind":"legacy_texture",
                "texture_type":"IMAGE",
                "image":null,
                "settings":{"use_interpolation":false}
            }),
        )
        .expect("null IMAGE reference is valid");
        assert_eq!(record_value(&texture)["image"], Value::Null);
        let LegacyTexture::ImageRecord {
            image: None,
            sampler,
            ..
        } = texture
        else {
            panic!("IMAGE record with an empty image pointer");
        };
        let empty = ImageData {
            width: 0,
            height: 0,
            pixels: Vec::new(),
            tiles: Default::default(),
            interpolation: ImageInterpolation::Linear,
        };
        let mut rng = BlenderRng::seed(0);
        let result = sample(
            TextureSource::ImageWithSettings(&empty, &sampler),
            DVec3::ZERO,
            &mut rng,
        )
        .expect("empty image sample");
        assert_rgb_close(result.rgb, [0.0; 3]);
        assert_eq!(result.alpha, 0.0);
    }

    #[test]
    fn noise_consumes_one_blender_rng_value_per_sample() {
        let noise = parse_record(
            "noise",
            &json!({
                "storage":"inline",
                "kind":"legacy_texture",
                "texture_type":"NOISE",
                "settings":{}
            }),
        )
        .expect("valid NOISE resource");
        let mut actual = BlenderRng::seed(47);
        let mut expected = BlenderRng::seed(47);
        sample(TextureSource::Legacy(&noise), DVec3::ZERO, &mut actual).expect("noise sample");
        let _ = expected.get_int32();
        assert_eq!(actual.get_int32(), expected.get_int32());
    }

    #[test]
    fn resolved_direct_image_uses_default_settings() {
        let image = image_data();
        let resolved = ResolvedTexture {
            resolved: super::ResolvedTextureKind::Image(image),
        };
        let mut rng = BlenderRng::seed(0);
        let result = sample(resolved.source(), DVec3::new(-0.5, 0.0, 0.0), &mut rng)
            .expect("default direct image sample");
        assert_rgb_close(result.rgb, [0.2, 0.4, 0.6]);
    }

    #[test]
    fn blender_522_byte_image_oracle_matches_srgb_reencoding() {
        // Texture.evaluate per-channel oracle; the Non-Color byte image avoids its extra conversion.
        // The same sRGB bytes were also sampled by Blender 5.2.2 Displace via BKE_texture_get_value_ex(false).
        const BLENDER_BYTE_ORACLE: [f64; 3] = [
            0.501_960_754_394_531_2,
            0.250_980_377_197_265_6,
            0.125_490_188_598_632_8,
        ];
        const BLENDER_MODIFIER_ORACLE_INTENSITY: f64 = 0.292_810_440_063_476_56;
        let image = ImageData {
            width: 1,
            height: 1,
            pixels: vec![[
                0.215_860_500_113_899_26,
                0.051_269_458_374_043_24,
                0.014_443_843_596_092_545,
                1.0,
            ]],
            tiles: Default::default(),
            interpolation: ImageInterpolation::Linear,
        };
        let mut settings = DEFAULT_IMAGE_SETTINGS;
        settings.interpolation = false;
        settings.source_srgb_encoded = true;
        let mut rng = BlenderRng::seed(0);
        let result = sample(
            TextureSource::ImageWithSettings(&image, &settings),
            DVec3::ZERO,
            &mut rng,
        )
        .expect("sRGB-tagged byte-image sample");
        assert_rgb_close(result.rgb, BLENDER_BYTE_ORACLE);
        assert!(
            (result.intensity - BLENDER_MODIFIER_ORACLE_INTENSITY).abs() < 1e-6,
            "Blender modifier oracle intensity mismatch: {}",
            result.intensity
        );
    }

    #[test]
    fn blender_522_oracle_matches_filter_boxes_spanning_multiple_periods() {
        // Captured with Blender 5.2.2 Texture.evaluate on byte pixels [black, white], filter_size=5,
        // REPEAT extension, and a Non-Color image to avoid scene color conversion.
        const BLENDER_FILTER_ORACLE: f64 = 0.666_666_746_139_526_4;
        let image = ImageData {
            width: 2,
            height: 1,
            pixels: vec![[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            tiles: Default::default(),
            interpolation: ImageInterpolation::Linear,
        };
        let mut settings = DEFAULT_IMAGE_SETTINGS;
        settings.filter_size = 5.0;
        let mut rng = BlenderRng::seed(0);
        let result = sample(
            TextureSource::ImageWithSettings(&image, &settings),
            DVec3::new(0.5, 0.0, 0.0),
            &mut rng,
        )
        .expect("large repeated image filter");
        assert!((result.intensity - BLENDER_FILTER_ORACLE).abs() < 1e-6);
    }
}
