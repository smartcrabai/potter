use glam::{DMat4, DVec2, DVec3};
use serde_json::{Map, Value as JsonValue, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    image::{ImageData, ImageExtension, ImageInterpolation, ImageTileData},
    model::Id,
};

#[derive(Clone, Copy)]
enum ImageProjection {
    Flat,
    Box,
    Sphere,
    Tube,
}

/// Refuses non-default Texture and Color Mapping settings unsupported by shader texture nodes.
pub(crate) fn validate_texture_node_mapping(
    id: &Id,
    node_type: &str,
    properties: &Map<String, JsonValue>,
) -> Result<()> {
    if let Some(mapping) = properties.get("texture_mapping")
        && !texture_mapping_is_default(mapping)
    {
        return Err(unsupported_mapping(id, node_type, "texture_mapping"));
    }
    if let Some(mapping) = properties.get("color_mapping")
        && !color_mapping_is_default(mapping)
    {
        return Err(unsupported_mapping(id, node_type, "color_mapping"));
    }
    Ok(())
}

pub(crate) fn texture_node_mapping_is_default(properties: &Map<String, JsonValue>) -> bool {
    properties.iter().all(|(property, value)| {
        texture_node_mapping_property_is_default(property, value).unwrap_or(true)
    })
}

pub(crate) fn texture_node_mapping_property_is_default(
    property: &str,
    value: &JsonValue,
) -> Option<bool> {
    match property {
        "texture_mapping" => Some(texture_mapping_is_default(value)),
        "color_mapping" => Some(color_mapping_is_default(value)),
        _ => None,
    }
}

pub(crate) fn image_texture_interpolation(
    id: &Id,
    node_type: &str,
    properties: &Map<String, JsonValue>,
    default: ImageInterpolation,
) -> Result<ImageInterpolation> {
    let Some(value) = properties.get("interpolation") else {
        return Ok(default);
    };
    match value.as_str() {
        Some(value) if value.eq_ignore_ascii_case("LINEAR") => Ok(ImageInterpolation::Linear),
        Some(value) if value.eq_ignore_ascii_case("CLOSEST") => Ok(ImageInterpolation::Closest),
        Some(value)
            if value.eq_ignore_ascii_case("CUBIC") || value.eq_ignore_ascii_case("SMART") =>
        {
            Ok(ImageInterpolation::Cubic)
        }
        _ => Err(unsupported_option(id, node_type, "interpolation", value)),
    }
}

pub(crate) fn unsupported_realtime_environment_filtering(id: &Id, image_id: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "realtime MIRROR_BALL Environment Texture LINEAR filtering requires unavailable mip/aniso gradients",
        json!({
            "feature_id": "shader.node.ShaderNodeTexEnvironment.linear_mip_filtering",
            "node_id": id,
            "image_id": image_id,
            "projection": "MIRROR_BALL",
        }),
    )
}
/// Samples an environment image with the renderer's interpolation path. EEVEE's cubic shader is a 4-filtered-tap
/// B-spline, unlike Cycles' 4×4 cubic image interpolation.
pub(crate) fn sample_environment_texture(
    image: &ImageData,
    uv: [f64; 2],
    interpolation: ImageInterpolation,
    extensions: [ImageExtension; 2],
    path_traced: bool,
) -> [f64; 4] {
    if !path_traced && interpolation == ImageInterpolation::Cubic {
        return sample_eevee_environment_cubic(image, uv, extensions);
    }
    if !path_traced && image.udim {
        crate::image::sampling::sample_with_extension_alpha_first_tile_axes(
            image,
            uv,
            interpolation,
            extensions,
            false,
        )
    } else {
        crate::image::sampling::sample_with_extension_alpha_axes(
            image,
            uv,
            false,
            interpolation,
            extensions,
            false,
        )
    }
}

fn sample_eevee_environment_cubic(
    image: &ImageData,
    uv: [f64; 2],
    extensions: [ImageExtension; 2],
) -> [f64; 4] {
    if !uv[0].is_finite() || !uv[1].is_finite() {
        return [0.0; 4];
    }
    let (width, height) = image_dimensions(image);
    if width == 0 || height == 0 {
        return [0.0; 4];
    }
    let width = width as f32;
    let height = height as f32;
    let coordinate = [uv[0] as f32 * width, uv[1] as f32 * height];
    let center = [
        (coordinate[0] - 0.5).floor() + 0.5,
        (coordinate[1] - 0.5).floor() + 0.5,
    ];
    let (weights_x, weights_y) = (
        eevee_cubic_bspline_coefficients(coordinate[0] - center[0]),
        eevee_cubic_bspline_coefficients(coordinate[1] - center[1]),
    );
    let (sum_x0, sum_x1) = (weights_x[0] + weights_x[1], weights_x[2] + weights_x[3]);
    let (sum_y0, sum_y1) = (weights_y[0] + weights_y[1], weights_y[2] + weights_y[3]);
    let (fraction_x0, fraction_x1) = (weights_x[1] / sum_x0, weights_x[3] / sum_x1);
    let (fraction_y0, fraction_y1) = (weights_y[1] / sum_y0, weights_y[3] / sum_y1);
    let u0 = (center[0] - 1.0 + fraction_x0) / width;
    let u1 = (center[0] + 1.0 + fraction_x1) / width;
    let v0 = (center[1] - 1.0 + fraction_y0) / height;
    let v1 = (center[1] + 1.0 + fraction_y1) / height;
    let sample =
        |u: f32, v: f32| sample_eevee_linear_texture(image, [u, v], width, height, extensions);
    let sample00 = sample(u0, v0);
    let sample10 = sample(u1, v0);
    let sample01 = sample(u0, v1);
    let sample11 = sample(u1, v1);
    std::array::from_fn(|channel| {
        f64::from(
            (((sample00[channel] * sum_x0) * sum_y0 + (sample10[channel] * sum_x1) * sum_y0)
                + (sample01[channel] * sum_x0) * sum_y1)
                + (sample11[channel] * sum_x1) * sum_y1,
        )
    })
}
fn sample_eevee_linear_texture(
    image: &ImageData,
    uv: [f32; 2],
    width: f32,
    height: f32,
    extensions: [ImageExtension; 2],
) -> [f32; 4] {
    let coordinate = [uv[0] * width - 0.5, uv[1] * height - 0.5];
    if !coordinate[0].is_finite() || !coordinate[1].is_finite() {
        return [0.0; 4];
    }
    let lower = [coordinate[0].floor(), coordinate[1].floor()];
    let fraction = [coordinate[0] - lower[0], coordinate[1] - lower[1]];
    let texel = |x: f32, y: f32| {
        let uv = [f64::from((x + 0.5) / width), f64::from((y + 0.5) / height)];
        let color = if image.udim {
            crate::image::sampling::sample_with_extension_alpha_first_tile_axes(
                image,
                uv,
                ImageInterpolation::Closest,
                extensions,
                false,
            )
        } else {
            crate::image::sampling::sample_with_extension_alpha_axes(
                image,
                uv,
                false,
                ImageInterpolation::Closest,
                extensions,
                false,
            )
        };
        color.map(|channel| crate::image::sampling::round_to_half(channel) as f32)
    };
    let color00 = texel(lower[0], lower[1]);
    let color10 = texel(lower[0] + 1.0, lower[1]);
    let color01 = texel(lower[0], lower[1] + 1.0);
    let color11 = texel(lower[0] + 1.0, lower[1] + 1.0);
    let top: [f32; 4] = std::array::from_fn(|channel| {
        color00[channel] + (color10[channel] - color00[channel]) * fraction[0]
    });
    let bottom: [f32; 4] = std::array::from_fn(|channel| {
        color01[channel] + (color11[channel] - color01[channel]) * fraction[0]
    });
    std::array::from_fn(|channel| top[channel] + (bottom[channel] - top[channel]) * fraction[1])
}

fn eevee_cubic_bspline_coefficients(fraction: f32) -> [f32; 4] {
    let squared = fraction * fraction;
    let cubed = squared * fraction;
    let weight3 = cubed / 6.0;
    let weight0 = -weight3 + squared * 0.5 - fraction * 0.5 + 1.0 / 6.0;
    let weight1 = cubed * 0.5 - squared + 2.0 / 3.0;
    let weight2 = 1.0 - weight0 - weight1 - weight3;
    [weight0, weight1, weight2, weight3]
}

fn image_dimensions(image: &ImageData) -> (u32, u32) {
    let has_pixels = |tile: &ImageTileData| {
        usize::try_from(tile.width)
            .ok()
            .and_then(|width| {
                usize::try_from(tile.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .is_some_and(|count| tile.width > 0 && tile.height > 0 && count == tile.pixels.len())
    };
    if image.udim {
        image
            .tiles
            .get(&1001)
            .filter(|tile| has_pixels(tile))
            .or_else(|| image.tiles.values().find(|tile| has_pixels(tile)))
            .map_or((image.width, image.height), |tile| {
                (tile.width, tile.height)
            })
    } else {
        (image.width, image.height)
    }
}

/// Cycles' Image Texture projections and their BOX face weights.
pub(crate) fn image_texture_projection_samples(
    id: &Id,
    node_type: &str,
    properties: &Map<String, JsonValue>,
    coordinate: DVec3,
    world_normal: DVec3,
    object_to_world: DMat4,
) -> Result<([DVec2; 3], [f64; 3])> {
    let projection = match properties.get("projection") {
        None => ImageProjection::Flat,
        Some(value) => match value.as_str() {
            Some(value) if value.eq_ignore_ascii_case("FLAT") => ImageProjection::Flat,
            Some(value) if value.eq_ignore_ascii_case("BOX") => ImageProjection::Box,
            Some(value) if value.eq_ignore_ascii_case("SPHERE") => ImageProjection::Sphere,
            Some(value) if value.eq_ignore_ascii_case("TUBE") => ImageProjection::Tube,
            _ => return Err(unsupported_option(id, node_type, "projection", value)),
        },
    };
    let mut uvs = [DVec2::ZERO; 3];
    let mut weights = [0.0; 3];
    match projection {
        ImageProjection::Flat => {
            uvs[0] = coordinate.truncate();
            weights[0] = 1.0;
        }
        ImageProjection::Sphere => {
            uvs[0] = sphere_projection(coordinate);
            weights[0] = 1.0;
        }
        ImageProjection::Tube => {
            uvs[0] = tube_projection(coordinate);
            weights[0] = 1.0;
        }
        ImageProjection::Box => {
            let blend = match properties.get("projection_blend") {
                None => 0.0,
                Some(value) => value
                    .as_f64()
                    .filter(|value| (0.0..=1.0).contains(value))
                    .ok_or_else(|| unsupported_option(id, node_type, "projection_blend", value))?,
            };
            let signed_normal = object_to_world
                .transpose()
                .transform_vector3(world_normal)
                .normalize_or_zero();
            weights = box_projection_weights(signed_normal, blend);
            uvs = [
                DVec2::new(
                    if signed_normal.x < 0.0 {
                        1.0 - coordinate.y
                    } else {
                        coordinate.y
                    },
                    coordinate.z,
                ),
                DVec2::new(
                    if signed_normal.y > 0.0 {
                        1.0 - coordinate.x
                    } else {
                        coordinate.x
                    },
                    coordinate.z,
                ),
                DVec2::new(
                    if signed_normal.z > 0.0 {
                        1.0 - coordinate.y
                    } else {
                        coordinate.y
                    },
                    coordinate.x,
                ),
            ];
        }
    }
    Ok((uvs, weights))
}

fn sphere_projection(coordinate: DVec3) -> DVec2 {
    let coordinate = (coordinate - DVec3::splat(0.5)) * 2.0;
    let length_squared = coordinate.length_squared();
    if length_squared <= 0.0 {
        return DVec2::ZERO;
    }
    let u = if coordinate.x == 0.0 && coordinate.y == 0.0 {
        0.0
    } else {
        0.5 - coordinate.x.atan2(coordinate.y) / std::f64::consts::TAU
    };
    let v = 1.0
        - (coordinate.z / length_squared.sqrt())
            .clamp(-1.0, 1.0)
            .acos()
            / std::f64::consts::PI;
    DVec2::new(u, v)
}

fn tube_projection(coordinate: DVec3) -> DVec2 {
    let coordinate = (coordinate - DVec3::splat(0.5)) * 2.0;
    let radius = coordinate.x.hypot(coordinate.y);
    if radius <= 0.0 {
        return DVec2::ZERO;
    }
    let u =
        (1.0 - (coordinate.x / radius).atan2(coordinate.y / radius) / std::f64::consts::PI) * 0.5;
    DVec2::new(u, f64::midpoint(coordinate.z, 1.0))
}

fn box_projection_weights(signed_normal: DVec3, blend: f64) -> [f64; 3] {
    let mut normal = signed_normal.abs();
    normal /= normal.element_sum();
    let limit = f64::midpoint(1.0, blend);
    let mut weights = [0.0; 3];
    if normal.x > limit * (normal.x + normal.y) && normal.x > limit * (normal.x + normal.z) {
        weights[0] = 1.0;
    } else if normal.y > limit * (normal.x + normal.y) && normal.y > limit * (normal.y + normal.z) {
        weights[1] = 1.0;
    } else if normal.z > limit * (normal.x + normal.z) && normal.z > limit * (normal.y + normal.z) {
        weights[2] = 1.0;
    } else if blend > 0.0 {
        if normal.z < (1.0 - limit) * (normal.y + normal.x) {
            weights[0] =
                ((normal.x / (normal.x + normal.y) - 0.5 * (1.0 - blend)) / blend).clamp(0.0, 1.0);
            weights[1] = 1.0 - weights[0];
        } else if normal.x < (1.0 - limit) * (normal.y + normal.z) {
            weights[1] =
                ((normal.y / (normal.y + normal.z) - 0.5 * (1.0 - blend)) / blend).clamp(0.0, 1.0);
            weights[2] = 1.0 - weights[1];
        } else if normal.y < (1.0 - limit) * (normal.x + normal.z) {
            weights[0] =
                ((normal.x / (normal.x + normal.z) - 0.5 * (1.0 - blend)) / blend).clamp(0.0, 1.0);
            weights[2] = 1.0 - weights[0];
        } else {
            weights[0] = ((2.0 - limit) * normal.x + (limit - 1.0)) / (2.0 * limit - 1.0);
            weights[1] = ((2.0 - limit) * normal.y + (limit - 1.0)) / (2.0 * limit - 1.0);
            weights[2] = ((2.0 - limit) * normal.z + (limit - 1.0)) / (2.0 * limit - 1.0);
        }
    } else {
        weights[0] = 1.0;
    }
    weights
}

fn unsupported_option(id: &Id, node_type: &str, property: &str, value: &JsonValue) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("shader node `{node_type}` has unsupported `{property}` setting"),
        json!({
            "feature_id": format!("shader.node.{node_type}.{property}"),
            "node_id": id,
            "property": property,
            "value": value,
        }),
    )
}

fn texture_mapping_is_default(value: &JsonValue) -> bool {
    let Some(mapping) = value.as_object() else {
        return false;
    };
    mapping.iter().all(|(key, value)| match key.as_str() {
        "vector_type" => is_enum(value, "POINT"),
        "translation" | "rotation" | "min" => is_vector(value, [0.0; 3]),
        "scale" | "max" => is_vector(value, [1.0; 3]),
        "use_min" | "use_max" => value.as_bool() == Some(false),
        "mapping_x" => is_enum(value, "X"),
        "mapping_y" => is_enum(value, "Y"),
        "mapping_z" => is_enum(value, "Z"),
        "mapping" => is_enum(value, "FLAT"),
        _ => false,
    })
}

fn color_mapping_is_default(value: &JsonValue) -> bool {
    let Some(mapping) = value.as_object() else {
        return false;
    };
    mapping.iter().all(|(key, value)| match key.as_str() {
        "brightness" | "contrast" | "saturation" => value.as_f64() == Some(1.0),
        "blend_color" => is_vector(value, [f64::from(0.8_f32); 3]) || is_vector(value, [0.8; 3]),
        "blend_type" => is_enum(value, "MIX"),
        "blend_factor" => value.as_f64() == Some(0.0),
        "use_color_ramp" => value.as_bool() == Some(false),
        // With the ramp disabled its stored curve has no effect on Cycles' image read.
        "color_ramp" => true,
        _ => false,
    })
}

fn is_enum(value: &JsonValue, expected: &str) -> bool {
    value
        .as_str()
        .is_some_and(|value| value.eq_ignore_ascii_case(expected))
}

fn is_vector(value: &JsonValue, expected: [f64; 3]) -> bool {
    value.as_array().is_some_and(|values| {
        values.len() == expected.len()
            && values
                .iter()
                .zip(expected)
                .all(|(value, expected)| value.as_f64() == Some(expected))
    })
}

fn unsupported_mapping(id: &Id, node_type: &str, property: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "shader node `{node_type}` has unsupported `{property}` settings; re-import is required to reconstruct the mapping"
        ),
        json!({
            "feature_id": format!("shader.node.{node_type}.{property}"),
            "node_id": id,
            "property": property,
        }),
    )
}
