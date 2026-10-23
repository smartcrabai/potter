use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use glam::{DMat4, DVec3, IVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        vdb::{VdbGrid, VdbValue, VdbVolume},
        volume::{VolumeData, VolumeGeneratedSource, VolumeGrid, VolumeSource},
    },
};

/// Resamples inline or VDB scalar volume grids using mapped texture color as a displacement vector.
pub(crate) fn displace_volume(
    volume: &VolumeData,
    params: &Map<String, Value>,
    texture: Option<crate::geom::legacy_texture::TextureSource<'_>>,
    rng: &mut crate::blender_rng::BlenderRng,
    texture_mapping: Option<crate::model::TextureCoordinateMapping>,
) -> Result<VolumeData> {
    let strength = number_param("volume_displace", params, "strength")?;
    let Some(texture) = texture.filter(|_| strength != 0.0) else {
        return Ok(volume.clone());
    };
    let mut texture = VolumeDisplacement::new(params, texture, rng, strength, texture_mapping)?;
    match &volume.source {
        VolumeSource::Generated(source) => {
            let mut displaced = volume.clone();
            if displaced.grids.is_empty() {
                return Err(invalid(
                    "volume displacement requires at least one scalar grid",
                ));
            }
            // The output's active bounds are those of its first grid, which Volume to Mesh reads.
            let mut active_bounds = None;
            for (index, grid) in displaced.grids.iter_mut().enumerate() {
                let bounds = displace_inline_grid(grid, &mut texture, source.active_bounds)?;
                if index == 0 {
                    active_bounds = bounds;
                }
            }
            displaced.source = VolumeSource::Generated(VolumeGeneratedSource {
                algorithm: source.algorithm.clone(),
                active_bounds,
            });
            Ok(displaced)
        }
        VolumeSource::File(source) if source.format == "vdb" => {
            displace_vdb_volume(volume, &mut texture)
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

#[derive(Clone, Copy)]
enum VolumeTextureMapMode {
    Local,
    Global,
    Object,
}

struct VolumeDisplacement<'a> {
    texture: crate::geom::legacy_texture::TextureSource<'a>,
    rng: &'a mut crate::blender_rng::BlenderRng,
    mapping: crate::model::TextureCoordinateMapping,
    mapping_mode: VolumeTextureMapMode,
    strength: f64,
    mid_level: [f64; 3],
    sample_radius: f64,
}

impl<'a> VolumeDisplacement<'a> {
    fn new(
        params: &Map<String, Value>,
        texture: crate::geom::legacy_texture::TextureSource<'a>,
        rng: &'a mut crate::blender_rng::BlenderRng,
        strength: f64,
        mapping: Option<crate::model::TextureCoordinateMapping>,
    ) -> Result<Self> {
        let sample_radius = number_param("volume_displace", params, "texture_sample_radius")?;
        if sample_radius < 0.0 {
            return Err(invalid("texture_sample_radius must be non-negative"));
        }
        if sample_radius > 16.0 {
            return Err(limit(
                "texture_sample_radius exceeds the 16-voxel output padding limit",
            ));
        }
        let mapping_mode = match string_param("volume_displace", params, "texture_map_mode")? {
            "LOCAL" => VolumeTextureMapMode::Local,
            "GLOBAL" => VolumeTextureMapMode::Global,
            "OBJECT" => VolumeTextureMapMode::Object,
            _ => return Err(invalid("texture_map_mode must be LOCAL, GLOBAL, or OBJECT")),
        };
        let mapping = mapping.ok_or_else(|| invalid("volume texture mapping was not resolved"))?;
        Ok(Self {
            texture,
            rng,
            mapping,
            mapping_mode,
            strength,
            mid_level: vector_param("volume_displace", params, "texture_mid_level")?,
            sample_radius,
        })
    }

    /// Blender's `index_to_texture` (`MOD_volume_displace.cc`, `get_index_to_texture_transform`) as
    /// `OpenVDB` multiplies it: `index_to_object` for `LOCAL`, `index_to_object * object_to_world` for
    /// `GLOBAL`, and `index_to_object * object_to_world * world_to_texture` for `OBJECT`.
    fn texture_matrix(&self, index_to_object: &[f32; 16]) -> [f32; 16] {
        match self.mapping_mode {
            VolumeTextureMapMode::Local => *index_to_object,
            VolumeTextureMapMode::Global => {
                openvdb_mat4_mul(index_to_object, &self.mapping.subject_to_world_f32)
            }
            VolumeTextureMapMode::Object => match &self.mapping.world_to_mapping_f32 {
                Some(world_to_mapping) => openvdb_mat4_mul(
                    &openvdb_mat4_mul(index_to_object, &self.mapping.subject_to_world_f32),
                    world_to_mapping,
                ),
                None => *index_to_object,
            },
        }
    }

    /// Blender's `DisplaceOp::compute_displace_vector` (`MOD_volume_displace.cc`): the texture is
    /// evaluated at `index * texture_matrix` in single precision, and the displacement
    /// `(texture - mid) * scale`, with `scale = strength / max_voxel_side` as a float, is in index units.
    fn index_displacement_at(
        &mut self,
        texture_matrix: &[f32; 16],
        index: DVec3,
        max_voxel_side: f64,
    ) -> Result<DVec3> {
        let texture_position =
            crate::geom::modifiers::blender_transform_point_f32(texture_matrix, index.as_vec3());
        let texture_position = DVec3::new(
            f64::from(texture_position.x),
            f64::from(texture_position.y),
            f64::from(texture_position.z),
        );
        if !texture_position.is_finite() {
            return Err(invalid("volume texture coordinates are not finite"));
        }
        let sample = crate::geom::legacy_texture::sample(self.texture, texture_position, self.rng)?;
        let scale = f64::from((self.strength as f32) / (max_voxel_side as f32));
        Ok((DVec3::from_array(sample.rgb) - DVec3::from_array(self.mid_level)) * scale)
    }
}

/// Displaces one inline grid. `source_active` is the inclusive index box of the grid's active voxels
/// (`None`: all voxels). Returns the output's active box, inclusive.
fn displace_inline_grid(
    grid: &mut VolumeGrid,
    texture: &mut VolumeDisplacement<'_>,
    source_active: Option<([i64; 3], [i64; 3])>,
) -> Result<Option<([i64; 3], [i64; 3])>> {
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
    let original_transform = if grid.index_to_local.is_some() {
        Some(grid.checked_index_to_local_matrix()?)
    } else {
        None
    };
    let output_transform = original_transform
        .map(|matrix| matrix * DMat4::from_translation(-DVec3::splat(padding as f64)));
    let output_origin = output_transform.map_or_else(
        || original_origin - voxel_size * padding as f64,
        |matrix| matrix.transform_point3(DVec3::ZERO),
    );
    // Blender's texture position is `index * index_to_texture` with the grid's absolute index: an
    // affine grid carries its own transform and its sample index is the index; otherwise the grid's
    // transform is the voxel scale, and the absolute index is the sample index plus the origin's index.
    let (index_to_object, index_offset) = if grid.index_to_local.is_some() {
        (
            grid.checked_index_to_local_matrix()?
                .to_cols_array()
                .map(|value| value as f32),
            DVec3::ZERO,
        )
    } else {
        let voxel = grid.voxel_size as f32;
        let mut scale = [0.0_f32; 16];
        scale[0] = voxel;
        scale[5] = voxel;
        scale[10] = voxel;
        scale[15] = 1.0;
        (scale, (original_origin / grid.voxel_size).round())
    };
    let texture_matrix = texture.texture_matrix(&index_to_object);
    let mut output = Vec::with_capacity(output_count);
    for z in 0..output_dims[2] {
        for y in 0..output_dims[1] {
            for x in 0..output_dims[0] {
                // The output keeps the source grid shifted by the padding, so the source index is the
                // output index minus the padding.
                let source_index =
                    DVec3::new(x as f64, y as f64, z as f64) - DVec3::splat(padding as f64);
                let index_displacement = texture.index_displacement_at(
                    &texture_matrix,
                    source_index + index_offset,
                    grid.voxel_size,
                )?;
                let source_coordinate = source_index - index_displacement;
                let value = sample_volume_grid(&original, source_dims, source_coordinate);
                if !value.is_finite() {
                    return Err(invalid("volume displacement produced a non-finite sample"));
                }
                output.push(value);
            }
        }
    }
    // Blender dilates the active voxels by the same rounds as the padding (`dilateActiveValues` with
    // `NN_FACE_EDGE`, `rounds` = the dilation radius), so the dilated box grows by one padding on the low
    // side and two on the high side, once the output's padding offset is added to the source indices.
    let output_active = source_active.map(|(low, high)| {
        let pad = padding as i64;
        let last = output_dims.map(|dimension| dimension as i64 - 1);
        (
            std::array::from_fn(|axis| low[axis].clamp(0, last[axis])),
            std::array::from_fn(|axis| (high[axis] + 2 * pad).clamp(0, last[axis])),
        )
    });
    grid.dims = [
        u32::try_from(output_dims[0]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[1]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[2]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
    ];
    grid.origin = output_origin;
    grid.index_to_local = output_transform.map(|matrix| matrix.to_cols_array());
    grid.values = Some(output);
    grid.blob_f32 = None;
    Ok(output_active)
}

/// Blender's `MOD_volume_displace.cc` displaces every grid of the volume with the same texture
/// (`DisplaceGridOp` in a loop over `BKE_volume_num_grids`), so each decoded grid is displaced in turn.
fn displace_vdb_volume(
    volume: &VolumeData,
    texture: &mut VolumeDisplacement<'_>,
) -> Result<VolumeData> {
    let decoded = volume.decoded_vdb.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "file-backed VDB volume is not decoded for evaluation",
            json!({"feature_id":"volume.openvdb_evaluation.resource"}),
        )
    })?;
    let grids = decoded
        .grids
        .iter()
        .map(|grid| displace_vdb_grid(grid, texture))
        .collect::<Result<Vec<_>>>()?;
    let mut displaced = volume.clone();
    displaced.decoded_vdb = Some(Arc::new(VdbVolume {
        grids,
        uuid: decoded.uuid.clone(),
        library_version: decoded.library_version,
    }));
    Ok(displaced)
}

/// Upper bound on the displaced VDB samples, one million as Blender's `dilateActiveValues` budget.
const MAX_DISPLACED_SAMPLES: usize = 1_000_000;

/// Displaces one VDB grid as Blender's `DisplaceGridOp` does: the active voxels are dilated by the
/// sample radius (`dilateActiveValues`, `NN_FACE_EDGE`), then each dilated voxel takes the
/// box-sampled value of the original grid at `index - displacement`, where the displacement is
/// `(texture - mid_level) * strength / max_voxel_side` in index units. Only float and vector grids
/// are displaced.
fn displace_vdb_grid(grid: &VdbGrid, texture: &mut VolumeDisplacement<'_>) -> Result<VdbGrid> {
    let vector = match grid.value_type.split('_').nth(1) {
        Some("float") => false,
        Some("vec3s") => true,
        _ => {
            return Err(unsupported(
                "modifier.volume_displace.vdb_grid_type",
                "only float and vector VDB grids can be displaced",
            ));
        }
    };
    let max_voxel_side = grid.voxel_size().max_element();
    if !max_voxel_side.is_finite() || max_voxel_side <= 0.0 {
        return Err(invalid(
            "VDB grid transform must be finite with positive voxel_size",
        ));
    }
    let dilation_radius =
        (texture.sample_radius * texture.strength.abs() / max_voxel_side / 2.0).ceil();
    if !dilation_radius.is_finite() || dilation_radius > 16.0 {
        return Err(limit("VDB displacement dilation exceeds 16 voxels"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated ceil radius is integral and bounded to 16"
    )]
    let rounds = dilation_radius as i32;
    let domain = dilate_face_edge(grid.tree.active_seeds(MAX_DISPLACED_SAMPLES)?, rounds)?;
    // The grid's own transform is Blender's `index_to_object`; the index is absolute.
    let index_to_world = grid
        .transform
        .index_to_world
        .to_cols_array()
        .map(|value| value as f32);
    let texture_matrix = texture.texture_matrix(&index_to_world);
    let mut values = HashMap::with_capacity(domain.len());
    for point in domain {
        let index = DVec3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z));
        let source =
            index - texture.index_displacement_at(&texture_matrix, index, max_voxel_side)?;
        let value = if vector {
            VdbValue::Vec3(std::array::from_fn(|component| {
                grid.sample_index_component_box(source, component)
            }))
        } else {
            VdbValue::Float(grid.sample_index_component_box(source, 0))
        };
        values.insert(point, value);
    }
    let tree = grid.tree.with_samples(values);
    Ok(VdbGrid {
        name: grid.name.clone(),
        class: grid.class.clone(),
        value_type: grid.value_type.clone(),
        transform: grid.transform,
        active_bbox: tree.active_bbox,
        tree: Arc::new(tree),
    })
}

/// Blender's `dilateActiveValues` with `NN_FACE_EDGE`: each round adds the 18 face and edge
/// neighbours of the voxels the previous round added.
fn dilate_face_edge(seeds: Vec<IVec3>, rounds: i32) -> Result<HashSet<IVec3>> {
    let mut offsets = Vec::with_capacity(18);
    for z in -1_i32..=1 {
        for y in -1_i32..=1 {
            for x in -1_i32..=1 {
                if (1..=2).contains(&(x.abs() + y.abs() + z.abs())) {
                    offsets.push(IVec3::new(x, y, z));
                }
            }
        }
    }
    let mut domain: HashSet<IVec3> = seeds.into_iter().collect();
    let mut frontier: Vec<IVec3> = domain.iter().copied().collect();
    for _ in 0..rounds {
        let mut added = Vec::new();
        for point in &frontier {
            for offset in &offsets {
                let candidate = *point + *offset;
                if domain.insert(candidate) {
                    if domain.len() > MAX_DISPLACED_SAMPLES {
                        return Err(limit("displaced VDB grid exceeds one million samples"));
                    }
                    added.push(candidate);
                }
            }
        }
        if added.is_empty() {
            break;
        }
        frontier = added;
    }
    Ok(domain)
}

/// `OpenVDB`'s `Mat4 * Mat4` (`openvdb/math/Mat4.h`, `operator*=`) on the memory of two float
/// matrices: `result(i, j) = a(i, 0) b(0, j) + a(i, 1) b(1, j) + a(i, 2) b(2, j) + a(i, 3) b(3, j)`,
/// summed left to right in `float`. Blender's `float[4][4]` has the same memory layout.
pub(crate) fn openvdb_mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|index| {
        let (row, column) = (index / 4, index % 4);
        let mut sum = a[4 * row] * b[column];
        for inner in 1..4 {
            sum += a[4 * row + inner] * b[4 * inner + column];
        }
        sum
    })
}

/// Box-samples a dense inline grid at an index-space position, as Blender's `BoxSampler` does on the
/// grid (`openvdb/tools/Interpolation.h`), with the background 0 outside the samples.
fn sample_volume_grid(samples: &[f32], dims: [usize; 3], coordinate: DVec3) -> f32 {
    if !coordinate.is_finite() {
        return 0.0;
    }
    crate::geom::vdb::box_sample_trilinear(
        |point| {
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
            samples[(z * dims[1] + y) * dims[0] + x]
        },
        coordinate,
    )
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
    use glam::{DMat4, DVec3};
    use serde_json::{Map, json};

    use super::displace_volume;
    use crate::geom::volume::{VolumeData, VolumeGrid};

    #[test]
    fn volume_displace_resamples_with_shared_texture() -> crate::error::Result<()> {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 1, 1],
                values: Some(vec![0.0, 1.0, 0.0]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let mut rng = crate::blender_rng::BlenderRng::seed(0);
        let unchanged = displace_volume(&volume, &Map::new(), None, &mut rng, None)?;
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
            Some(crate::geom::legacy_texture::TextureSource::Image(&image)),
            &mut rng,
            Some(crate::model::TextureCoordinateMapping::for_subject_world(
                DMat4::IDENTITY,
            )),
        )?;
        assert_eq!(
            displaced.grids[0].values.as_deref(),
            Some(&[0.0_f32, 0.5, 0.5][..])
        );
        Ok(())
    }
    #[test]
    fn volume_displace_object_mapping_without_object_uses_local_coordinates()
    -> crate::error::Result<()> {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 1, 1],
                values: Some(vec![0.0, 1.0, 0.0]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let image = crate::image::ImageData {
            udim: false,
            colour: crate::image::ColourCache::default(),
            storage: crate::image::ImageStorage::default(),
            width: 1,
            height: 1,
            pixels: vec![[1.0, 0.5, 0.5, 1.0]],
            tiles: std::collections::BTreeMap::new(),
            interpolation: crate::image::ImageInterpolation::Linear,
            srgb: false,
        };
        let mut rng = crate::blender_rng::BlenderRng::seed(0);
        let settings = |mode| {
            Map::from_iter([
                ("texture_map_mode".into(), json!(mode)),
                ("strength".into(), json!(1.0)),
                ("texture_mid_level".into(), json!([0.5, 0.5, 0.5])),
                ("texture_sample_radius".into(), json!(0.0)),
            ])
        };
        let local = displace_volume(
            &volume,
            &settings("LOCAL"),
            Some(crate::geom::legacy_texture::TextureSource::Image(&image)),
            &mut rng,
            Some(crate::model::TextureCoordinateMapping::for_subject_world(
                DMat4::from_translation(DVec3::new(2.0, 3.0, 4.0)),
            )),
        )?;
        let object_without_target = displace_volume(
            &volume,
            &settings("OBJECT"),
            Some(crate::geom::legacy_texture::TextureSource::Image(&image)),
            &mut rng,
            Some(crate::model::TextureCoordinateMapping::for_subject_world(
                DMat4::from_translation(DVec3::new(2.0, 3.0, 4.0)),
            )),
        )?;
        assert_eq!(object_without_target, local);
        Ok(())
    }
}
