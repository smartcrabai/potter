use std::borrow::Cow;

use glam::{DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::volume::{VolumeData, VolumeGeneratedSource, VolumeGrid, VolumeSource},
    image::ImageData,
};

/// Resamples inline or VDB scalar volume grids using mapped texture color as a displacement vector.
pub fn displace_volume(
    volume: &VolumeData,
    params: &Map<String, Value>,
    texture_type: &str,
    image: Option<&ImageData>,
    volume_world: DMat4,
    texture_map_object_world: Option<DMat4>,
) -> Result<VolumeData> {
    let strength = number_param("volume_displace", params, "strength")?;
    if !matches!(texture_type, "NONE" | "NOISE" | "IMAGE") {
        return Err(unsupported(
            "modifier.volume_displace.texture_type",
            "only deterministic noise and image textures are supported",
        ));
    }
    if texture_type == "NONE" || strength == 0.0 {
        return Ok(volume.clone());
    }
    let texture = VolumeDisplacement::new(
        params,
        texture_type,
        image,
        strength,
        volume_world,
        texture_map_object_world,
    )?;
    match &volume.source {
        VolumeSource::Generated(_) => {
            let mut displaced = volume.clone();
            if displaced.grids.is_empty() {
                return Err(invalid(
                    "volume displacement requires at least one scalar grid",
                ));
            }
            for grid in &mut displaced.grids {
                displace_inline_grid(grid, &texture)?;
            }
            Ok(displaced)
        }
        VolumeSource::File(source) if source.format == "vdb" => {
            displace_vdb_volume(volume, &texture)
        }
        VolumeSource::File(source) => Err(unsupported(
            "modifier.volume_displace.file_format",
            &format!(
                "file-backed volume format `{}` is not supported",
                source.format
            ),
        )),
    }
}

struct VolumeDisplacement<'a> {
    texture_type: &'a str,
    image: Option<&'a ImageData>,
    /// Straight colour of `image`, converted once here rather than for every sampled voxel.
    image_pixels: Option<Cow<'a, [[f64; 4]]>>,
    mapping: DMat4,
    strength: f64,
    mid_level: [f64; 3],
    sample_radius: f64,
}

impl<'a> VolumeDisplacement<'a> {
    fn new(
        params: &Map<String, Value>,
        texture_type: &'a str,
        image: Option<&'a ImageData>,
        strength: f64,
        volume_world: DMat4,
        texture_map_object_world: Option<DMat4>,
    ) -> Result<Self> {
        if texture_type == "IMAGE" && image.is_none() {
            return Err(unsupported(
                "modifier.volume_displace.image_texture",
                "image texture source is unavailable",
            ));
        }
        let sample_radius = number_param("volume_displace", params, "texture_sample_radius")?;
        if sample_radius < 0.0 {
            return Err(invalid("texture_sample_radius must be non-negative"));
        }
        if sample_radius > 16.0 {
            return Err(limit(
                "texture_sample_radius exceeds the 16-voxel output padding limit",
            ));
        }
        let mode = string_param("volume_displace", params, "texture_map_mode")?;
        let mapping = match mode {
            "LOCAL" => DMat4::IDENTITY,
            "GLOBAL" => volume_world,
            "OBJECT" => {
                let object_world = texture_map_object_world.ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InvalidOperation,
                        "OBJECT texture mapping requires texture_map_object",
                        json!({"feature_id":"modifier.volume_displace.texture_map_object"}),
                    )
                })?;
                invert_transform(object_world)? * volume_world
            }
            _ => {
                return Err(invalid("texture_map_mode must be LOCAL, GLOBAL, or OBJECT"));
            }
        };
        if !mapping.is_finite() || mapping.determinant().abs() <= 1.0e-15 {
            return Err(invalid(
                "volume texture coordinate transform must be finite and invertible",
            ));
        }
        Ok(Self {
            texture_type,
            image,
            image_pixels: image.map(ImageData::straight_pixels),
            mapping,
            strength,
            mid_level: vector_param("volume_displace", params, "texture_mid_level")?,
            sample_radius,
        })
    }

    fn displacement_at(&self, local_position: DVec3) -> Result<DVec3> {
        let texture_position = self.mapping.transform_point3(local_position);
        if !texture_position.is_finite() {
            return Err(invalid("volume texture coordinates are not finite"));
        }
        let color = match self.texture_type {
            "NOISE" => value_noise_color(texture_position),
            "IMAGE" => {
                let image = self
                    .image
                    .ok_or_else(|| invalid("image texture source disappeared"))?;
                let pixels = self
                    .image_pixels
                    .as_deref()
                    .ok_or_else(|| invalid("image texture source disappeared"))?;
                // Blender's flat Image Texture projection maps object coordinates from
                // [-1, 1] to UVs and repeats outside the unit square.
                let uv = [
                    (texture_position.x * 0.5 + 0.5).rem_euclid(1.0),
                    (texture_position.y * 0.5 + 0.5).rem_euclid(1.0),
                ];
                let rgba = sample_repeated_image(image, pixels, uv)?;
                [rgba[0], rgba[1], rgba[2]]
            }
            _ => {
                return Err(unsupported(
                    "modifier.volume_displace.texture_type",
                    "unsupported volume displacement texture source",
                ));
            }
        };
        Ok(DVec3::new(
            color[0] - self.mid_level[0],
            color[1] - self.mid_level[1],
            color[2] - self.mid_level[2],
        ) * self.strength)
    }
}
fn sample_repeated_image(image: &ImageData, pixels: &[[f64; 4]], uv: [f64; 2]) -> Result<[f64; 4]> {
    let width = usize::try_from(image.width)
        .map_err(|_| limit("image texture width exceeds addressable memory"))?;
    let height = usize::try_from(image.height)
        .map_err(|_| limit("image texture height exceeds addressable memory"))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| limit("image texture pixel count overflows"))?;
    if width == 0 || height == 0 || image.pixels.len() != pixel_count {
        return Err(invalid("image texture dimensions do not match its pixels"));
    }
    if uv.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(invalid("image texture coordinates must be finite"));
    }
    let width_f64 = f64::from(image.width);
    let height_f64 = f64::from(image.height);
    let width_i64 = i64::from(image.width);
    let height_i64 = i64::from(image.height);
    if image.interpolation == crate::image::ImageInterpolation::Closest {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "wrapped coordinates are bounded by the u32 image dimensions"
        )]
        let x = (uv[0].rem_euclid(1.0) * width_f64).floor() as i64;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "wrapped coordinates are bounded by the u32 image dimensions"
        )]
        let y = (uv[1].rem_euclid(1.0) * height_f64).floor() as i64;
        let x = usize::try_from(x.rem_euclid(width_i64))
            .map_err(|_| invalid("image texture x coordinate is invalid"))?;
        let y = usize::try_from(y.rem_euclid(height_i64))
            .map_err(|_| invalid("image texture y coordinate is invalid"))?;
        return Ok(pixels[y * width + x]);
    }
    let x = uv[0].rem_euclid(1.0) * width_f64 - 0.5;
    let y = uv[1].rem_euclid(1.0) * height_f64 - 0.5;
    let x_floor = x.floor();
    let y_floor = y.floor();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "wrapped coordinates are bounded by the u32 image dimensions"
    )]
    let x0 = x_floor as i64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "wrapped coordinates are bounded by the u32 image dimensions"
    )]
    let y0 = y_floor as i64;
    let x1 = (x0 + 1).rem_euclid(width_i64);
    let y1 = (y0 + 1).rem_euclid(height_i64);
    let x0 = usize::try_from(x0.rem_euclid(width_i64))
        .map_err(|_| invalid("image texture x coordinate is invalid"))?;
    let y0 = usize::try_from(y0.rem_euclid(height_i64))
        .map_err(|_| invalid("image texture y coordinate is invalid"))?;
    let x1 = usize::try_from(x1).map_err(|_| invalid("image texture x coordinate is invalid"))?;
    let y1 = usize::try_from(y1).map_err(|_| invalid("image texture y coordinate is invalid"))?;
    let fraction_x = x - x_floor;
    let fraction_y = y - y_floor;
    let top: [f64; 4] = std::array::from_fn(|channel| {
        let first = pixels[y0 * width + x0][channel];
        let second = pixels[y0 * width + x1][channel];
        first + (second - first) * fraction_x
    });
    let bottom: [f64; 4] = std::array::from_fn(|channel| {
        let first = pixels[y1 * width + x0][channel];
        let second = pixels[y1 * width + x1][channel];
        first + (second - first) * fraction_x
    });
    Ok(std::array::from_fn(|channel| {
        top[channel] + (bottom[channel] - top[channel]) * fraction_y
    }))
}

fn displace_inline_grid(grid: &mut VolumeGrid, texture: &VolumeDisplacement<'_>) -> Result<()> {
    if let Some(blob) = &grid.blob_f32 {
        if grid.values.is_some() {
            return Err(invalid(
                "volume grid must not contain both values and blob_f32",
            ));
        }
        if blob.len() % 4 != 0 {
            return Err(invalid("volume blob_f32 length must be a multiple of four"));
        }
        grid.values = Some(
            blob.as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect(),
        );
        grid.blob_f32 = None;
    }
    let original = grid
        .values
        .as_ref()
        .ok_or_else(|| invalid("volume grid has no inline samples"))?
        .clone();
    let source_dims = grid.dims.map(|dimension| dimension as usize);
    let source_count = source_dims
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| invalid("volume grid sample count overflows"))?;
    if source_count == 0 || original.len() != source_count {
        return Err(invalid(
            "volume grid sample count does not match its dimensions",
        ));
    }
    if !grid.voxel_size.is_finite() || grid.voxel_size <= 0.0 || !grid.origin.is_finite() {
        return Err(invalid(
            "volume grid transform must be finite with positive voxel_size",
        ));
    }
    let dilation_radius = texture.sample_radius * texture.strength.abs() / grid.voxel_size / 2.0;
    if !dilation_radius.is_finite() || dilation_radius > 16.0 {
        return Err(limit("volume displacement dilation exceeds 16 voxels"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated dilation radius is bounded to 16"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "validated dilation radius is non-negative"
    )]
    let padding = dilation_radius.ceil() as usize;
    let padding_total = padding
        .checked_mul(2)
        .ok_or_else(|| limit("volume displacement padding overflows"))?;
    let mut output_dims = [0_usize; 3];
    for axis in 0..3 {
        output_dims[axis] = source_dims[axis]
            .checked_add(padding_total)
            .ok_or_else(|| limit("volume displacement dimensions overflow"))?;
    }
    if output_dims.contains(&0) || output_dims.iter().any(|dimension| *dimension > 512) {
        return Err(limit(
            "displaced volume grid exceeds the 512-voxel axis limit",
        ));
    }
    let output_count = output_dims
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| limit("displaced volume grid sample count overflows"))?;
    if output_count > 1_000_000 {
        return Err(limit("displaced volume grid exceeds one million samples"));
    }
    let original_origin = grid.origin;
    let voxel_size = DVec3::splat(grid.voxel_size);
    let output_origin = original_origin - voxel_size * padding as f64;
    let mut output = Vec::with_capacity(output_count);
    for z in 0..output_dims[2] {
        for y in 0..output_dims[1] {
            for x in 0..output_dims[0] {
                let local_position =
                    output_origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64);
                let displacement = texture.displacement_at(local_position)?;
                let source_position = local_position - displacement;
                let source_coordinate = (source_position - original_origin)
                    .to_array()
                    .map(|value| value / grid.voxel_size);
                let value = sample_volume_grid(&original, source_dims, source_coordinate);
                if !value.is_finite() || value > f64::from(f32::MAX) {
                    return Err(invalid("volume displacement produced a non-finite sample"));
                }
                output.push(value.max(0.0) as f32);
            }
        }
    }
    grid.dims = [
        u32::try_from(output_dims[0]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[1]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[2]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
    ];
    grid.origin = output_origin;
    grid.values = Some(output);
    grid.blob_f32 = None;
    Ok(())
}

fn displace_vdb_volume(
    volume: &VolumeData,
    texture: &VolumeDisplacement<'_>,
) -> Result<VolumeData> {
    let decoded = volume.decoded_vdb.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "file-backed VDB volume is not decoded for evaluation",
            json!({"feature_id":"volume.openvdb_evaluation.resource"}),
        )
    })?;
    if decoded.grids.len() > 1 {
        return Err(unsupported(
            "modifier.volume_displace.vdb_multiple_grids",
            "Volume Displace supports one scalar VDB density grid at a time",
        ));
    }
    let Some(grid) = decoded.density_grid() else {
        return Err(unsupported(
            "modifier.volume_displace.vdb_density_grid",
            "VDB volume has no scalar density grid to displace",
        ));
    };
    if grid.value_type.split('_').nth(1) != Some("float") {
        return Err(unsupported(
            "modifier.volume_displace.vdb_grid_type",
            "only scalar float VDB density grids can be displaced",
        ));
    }
    let Some((active_minimum, active_maximum)) = grid.active_bbox else {
        return Ok(volume.clone());
    };
    let voxel_size = grid.voxel_size().max_element();
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(invalid(
            "VDB density grid transform must be finite with positive voxel_size",
        ));
    }
    let dilation_radius =
        (texture.sample_radius * texture.strength.abs() / voxel_size / 2.0).ceil();
    if !dilation_radius.is_finite() || dilation_radius > 16.0 {
        return Err(limit("VDB displacement dilation exceeds 16 voxels"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated ceil radius is integral and bounded to 16"
    )]
    let padding = dilation_radius as i32;
    let mut output_minimum = [0_i32; 3];
    let mut output_maximum = [0_i32; 3];
    let mut dimensions = [0_usize; 3];
    for axis in 0..3 {
        output_minimum[axis] = active_minimum[axis]
            .checked_sub(padding)
            .ok_or_else(|| limit("VDB displacement minimum index overflows"))?;
        output_maximum[axis] = active_maximum[axis]
            .checked_add(padding)
            .ok_or_else(|| limit("VDB displacement maximum index overflows"))?;
        let length = i64::from(output_maximum[axis]) - i64::from(output_minimum[axis]) + 1;
        dimensions[axis] = usize::try_from(length)
            .ok()
            .filter(|dimension| *dimension > 0 && *dimension <= 512)
            .ok_or_else(|| limit("displaced VDB grid exceeds the 512-voxel axis limit"))?;
    }
    let sample_count = dimensions
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| limit("displaced VDB grid sample count overflows"))?;
    if sample_count > 1_000_000 {
        return Err(limit("displaced VDB grid exceeds one million samples"));
    }
    let index_to_world = grid.transform.index_to_world;
    let output_origin = index_to_world.transform_point3(DVec3::new(
        f64::from(output_minimum[0]),
        f64::from(output_minimum[1]),
        f64::from(output_minimum[2]),
    ));
    let mut values = Vec::with_capacity(sample_count);
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let index = [
                    output_minimum[0]
                        + i32::try_from(x)
                            .map_err(|_| limit("VDB output x index exceeds signed range"))?,
                    output_minimum[1]
                        + i32::try_from(y)
                            .map_err(|_| limit("VDB output y index exceeds signed range"))?,
                    output_minimum[2]
                        + i32::try_from(z)
                            .map_err(|_| limit("VDB output z index exceeds signed range"))?,
                ];
                let index_position = DVec3::new(
                    f64::from(index[0]),
                    f64::from(index[1]),
                    f64::from(index[2]),
                );
                let object_position = index_to_world.transform_point3(index_position);
                let displacement = texture.displacement_at(object_position)? / voxel_size;
                let source_position =
                    index_to_world.transform_point3(index_position - displacement);
                let value = grid.sample_world(source_position).max(0.0);
                if !value.is_finite() || value > f64::from(f32::MAX) {
                    return Err(invalid("VDB displacement produced a non-finite density"));
                }
                values.push(value as f32);
            }
        }
    }
    let mut displaced = volume.clone();
    displaced.source = VolumeSource::Generated(VolumeGeneratedSource {
        algorithm: "volume_displace_vdb".to_owned(),
    });
    displaced.decoded_vdb = None;
    displaced.grids = vec![VolumeGrid {
        dims: [
            u32::try_from(dimensions[0]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
            u32::try_from(dimensions[1]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
            u32::try_from(dimensions[2]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
        ],
        voxel_size,
        origin: output_origin,
        values: Some(values),
        ..VolumeGrid::default()
    }];
    Ok(displaced)
}

fn value_noise_color(point: DVec3) -> [f64; 3] {
    if !point.is_finite() || point.abs().max_element() > 9.0e18 {
        return [0.5; 3];
    }
    let floor = point.floor();
    let base = floor.to_array().map(|value| value as i64);
    let mut fade = (point - floor).to_array();
    for component in &mut fade {
        *component = *component * *component * (3.0 - 2.0 * *component);
    }
    std::array::from_fn(|channel| {
        let mut value = 0.0;
        for z in 0..=1 {
            for y in 0..=1 {
                for x in 0..=1 {
                    let cell = [base[0] + x, base[1] + y, base[2] + z];
                    let weight = (if x == 0 { 1.0 - fade[0] } else { fade[0] })
                        * (if y == 0 { 1.0 - fade[1] } else { fade[1] })
                        * (if z == 0 { 1.0 - fade[2] } else { fade[2] });
                    value += hash_value_noise(cell, channel) * weight;
                }
            }
        }
        value
    })
}

fn sample_volume_grid(samples: &[f32], dims: [usize; 3], coordinate: [f64; 3]) -> f64 {
    if coordinate.iter().any(|value| !value.is_finite())
        || (0..3).any(|axis| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "volume grid dimensions are bounded to 512 samples per axis"
            )]
            let maximum = dims[axis] as f64;
            coordinate[axis] < -1.0 || coordinate[axis] > maximum
        })
    {
        return 0.0;
    }
    let lower: [i64; 3] = coordinate.map(|value| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded finite coordinates are floored to integer voxel indices"
        )]
        {
            value.floor() as i64
        }
    });
    let fraction = std::array::from_fn::<_, 3, _>(|axis| {
        #[expect(
            clippy::cast_precision_loss,
            reason = "voxel indices are bounded to the 512-sample grid limit"
        )]
        let lower = lower[axis] as f64;
        coordinate[axis] - lower
    });
    let sample = |point: [i64; 3]| {
        if (0..3).any(|axis| {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "volume dimensions originate as u32 and fit in i64"
            )]
            let dimension = dims[axis] as i64;
            point[axis] < 0 || point[axis] >= dimension
        }) {
            return 0.0;
        }
        let [Ok(x), Ok(y), Ok(z)] = point.map(usize::try_from) else {
            return 0.0;
        };
        f64::from(samples[(z * dims[1] + y) * dims[0] + x])
    };
    let lerp = |first: f64, second: f64, amount: f64| first + (second - first) * amount;
    let x00 = lerp(
        sample(lower),
        sample([lower[0] + 1, lower[1], lower[2]]),
        fraction[0],
    );
    let x10 = lerp(
        sample([lower[0], lower[1] + 1, lower[2]]),
        sample([lower[0] + 1, lower[1] + 1, lower[2]]),
        fraction[0],
    );
    let x01 = lerp(
        sample([lower[0], lower[1], lower[2] + 1]),
        sample([lower[0] + 1, lower[1], lower[2] + 1]),
        fraction[0],
    );
    let x11 = lerp(
        sample([lower[0], lower[1] + 1, lower[2] + 1]),
        sample([lower[0] + 1, lower[1] + 1, lower[2] + 1]),
        fraction[0],
    );
    lerp(
        lerp(x00, x10, fraction[1]),
        lerp(x01, x11, fraction[1]),
        fraction[2],
    )
}

fn hash_value_noise(point: [i64; 3], channel: usize) -> f64 {
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

fn invert_transform(matrix: DMat4) -> Result<DMat4> {
    if !matrix.is_finite() || matrix.determinant().abs() <= 1.0e-15 {
        return Err(invalid(
            "texture mapping object transform must be finite and invertible",
        ));
    }
    Ok(matrix.inverse())
}

/// The Blender default for `name` on modifier type `kind`, from `param_table.json`.
fn schema_default(kind: &str, name: &str) -> Result<&'static Value> {
    crate::params::default_value(crate::params::ParameterFamily::Modifier, kind, name).ok_or_else(
        || {
            PotError::with_details(
                ErrorCode::InternalError,
                format!("modifier type `{kind}` has no schema default for parameter `{name}`"),
                json!({"modifier_type": kind, "parameter": name}),
            )
        },
    )
}

/// The modifier's own `name` value, else the table default for modifier type `kind`.
pub(super) fn value_or_default<'a>(
    kind: &str,
    params: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a Value> {
    match params.get(name) {
        Some(value) => Ok(value),
        None => schema_default(kind, name),
    }
}

pub(super) fn bool_param(kind: &str, params: &Map<String, Value>, name: &str) -> Result<bool> {
    value_or_default(kind, params, name)?
        .as_bool()
        .ok_or_else(|| invalid(&format!("{name} must be a boolean")))
}

pub(super) fn number_param(kind: &str, params: &Map<String, Value>, name: &str) -> Result<f64> {
    value_or_default(kind, params, name)?
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid(&format!("{name} must be a finite number")))
}

fn vector_param(kind: &str, params: &Map<String, Value>, name: &str) -> Result<[f64; 3]> {
    let Some(values) = value_or_default(kind, params, name)?
        .as_array()
        .filter(|values| values.len() == 3)
    else {
        return Err(invalid(&format!(
            "{name} must contain three finite numbers"
        )));
    };
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid(&format!("{name} must contain three finite numbers")))?;
    }
    Ok(result)
}

pub(super) fn string_param<'a>(
    kind: &str,
    params: &'a Map<String, Value>,
    name: &str,
) -> Result<&'a str> {
    let value = value_or_default(kind, params, name)?
        .as_str()
        .ok_or_else(|| invalid(&format!("{name} must be a string")))?;
    if value.is_empty() {
        Err(invalid(&format!("{name} must not be empty")))
    } else {
        Ok(value)
    }
}

pub(super) fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

fn unsupported(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":feature_id,"reason":reason,"status":"not_supported"}),
    )
}

#[cfg(test)]
mod tests {
    use glam::DMat4;
    use serde_json::{Map, json};

    use super::displace_volume;
    use crate::geom::volume::{VolumeData, VolumeGrid};

    #[test]
    fn volume_displace_resamples_and_rejects_unknown_textures() -> crate::error::Result<()> {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 1, 1],
                values: Some(vec![0.0, 1.0, 0.0]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let unchanged = displace_volume(&volume, &Map::new(), "NONE", None, DMat4::IDENTITY, None)?;
        assert_eq!(
            unchanged.grids[0].values.as_deref(),
            Some(&[0.0_f32, 1.0, 0.0][..])
        );

        let image = crate::image::ImageData {
            colour: crate::image::ColourCache::default(),
            storage: crate::image::ImageStorage::default(),
            width: 1,
            height: 1,
            pixels: vec![[1.0, 0.5, 0.5, 1.0]],
            tiles: std::collections::BTreeMap::new(),
            interpolation: crate::image::ImageInterpolation::Linear,
            srgb: false,
            udim: false,
        };
        let displaced = displace_volume(
            &volume,
            &Map::from_iter([
                ("texture_map_mode".into(), json!("LOCAL")),
                ("strength".into(), json!(1.0)),
                ("texture_mid_level".into(), json!([0.5, 0.5, 0.5])),
                ("texture_sample_radius".into(), json!(0.0)),
            ]),
            "IMAGE",
            Some(&image),
            DMat4::IDENTITY,
            None,
        )?;
        assert_eq!(
            displaced.grids[0].values.as_deref(),
            Some(&[0.0_f32, 0.5, 0.5][..])
        );
        let Err(error) =
            displace_volume(&volume, &Map::new(), "VORONOI", None, DMat4::IDENTITY, None)
        else {
            return Err(crate::error::PotError::new(
                crate::error::ErrorCode::InternalError,
                "unsupported texture type unexpectedly succeeded",
            ));
        };
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
        Ok(())
    }
}
