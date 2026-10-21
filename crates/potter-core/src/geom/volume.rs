//! Inline scalar volume grids and mesh conversion.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use glam::{DMat4, DVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, MeshError},
    model::VolumeInterpolation,
};

use super::vdb::{BoxScalar, box_sample_trilinear};

const MAX_GRID_SAMPLES: usize = 1_000_000;
const MAX_GRID_AXIS: usize = 512;
const MAX_INPUT_VERTICES: usize = 100_000;
const MAX_INPUT_TRIANGLES: usize = 100_000;
const MAX_GRID_COUNT: usize = 64;
const MAX_OUTPUT_FACES: usize = 500_000;

/// Provenance information for an in-memory generated volume.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VolumeGeneratedSource {
    /// Name of the generator or conversion that produced the samples.
    pub algorithm: String,
    /// Inclusive index-space bounds of the active voxels, for a generator that leaves some voxels
    /// inactive (Blender's `evalActiveVoxelBoundingBox`, which Volume to Mesh's `VOXEL_AMOUNT` uses).
    /// `None` means every voxel is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_bounds: Option<([i64; 3], [i64; 3])>,
}

/// File-source metadata for externally stored volume grids.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VolumeFileSource {
    /// Source file format, for example `"vdb"`.
    pub format: String,
    /// Optional stable reference to the external content.
    pub content_ref: Option<String>,
    /// Names of grids reported by the source adapter.
    pub grid_names: Vec<String>,
    /// Optional local-space bounds reported by the source adapter.
    pub bounds_min: Option<[f64; 3]>,
    pub bounds_max: Option<[f64; 3]>,
}

/// Origin of a volume's grid data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "metadata",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum VolumeSource {
    /// Samples were generated or are available inline.
    Generated(VolumeGeneratedSource),
    /// External file source; VDB content is decoded from the resource registry for evaluation.
    File(VolumeFileSource),
}

impl Default for VolumeSource {
    fn default() -> Self {
        Self::Generated(VolumeGeneratedSource::default())
    }
}

/// One scalar grid with x-fastest inline samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VolumeGrid {
    /// Sample dimensions along x, y, and z.
    pub dims: [u32; 3],
    /// Legacy uniform spacing; affine grids store their maximum voxel side here.
    pub voxel_size: f64,
    /// Volume-local position of sample `[0, 0, 0]`, used when `index_to_local` is absent.
    pub origin: DVec3,
    /// Optional column-major 4x4 matrix mapping grid index coordinates to volume-local space.
    /// When absent, the legacy `origin + voxel_size * index` transform is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_to_local: Option<[f64; 16]>,
    /// Optional inline f32 samples, laid out with x as the fastest-changing axis.
    pub values: Option<Vec<f32>>,
    /// Optional little-endian f32 byte blob with the same sample layout.
    pub blob_f32: Option<Vec<u8>>,
    /// Optional stable reference to a backing blob. Inline `values` or `blob_f32`,
    /// when present, are the decoded samples for this reference.
    pub content_ref: Option<String>,
}

impl VolumeGrid {
    pub(crate) fn checked_index_to_local_matrix(&self) -> Result<DMat4> {
        let Some(columns) = self.index_to_local else {
            return Ok(DMat4::from_translation(self.origin)
                * DMat4::from_scale(DVec3::splat(self.voxel_size)));
        };
        let matrix = DMat4::from_cols_array(&columns);
        #[expect(
            clippy::float_cmp,
            reason = "affine matrices carry an exact 1.0 homogeneous corner"
        )]
        let is_affine = matrix.x_axis.w == 0.0
            && matrix.y_axis.w == 0.0
            && matrix.z_axis.w == 0.0
            && matrix.w_axis.w == 1.0;
        let determinant = matrix.determinant();
        if !matrix.is_finite()
            || !is_affine
            || !determinant.is_finite()
            || determinant == 0.0
            || !matrix.inverse().is_finite()
        {
            return Err(invalid_volume(
                "volume grid index_to_local must be a finite, invertible affine matrix",
            ));
        }
        Ok(matrix)
    }

    pub(crate) fn local_bounds(
        &self,
        minimum_index: DVec3,
        maximum_index: DVec3,
    ) -> Result<(DVec3, DVec3)> {
        if !minimum_index.is_finite()
            || !maximum_index.is_finite()
            || minimum_index.cmpgt(maximum_index).any()
        {
            return Err(invalid_volume("volume grid index bounds are invalid"));
        }
        let index_to_local = self.checked_index_to_local_matrix()?;
        let mut minimum = DVec3::splat(f64::INFINITY);
        let mut maximum = DVec3::splat(f64::NEG_INFINITY);
        for corner in 0..8 {
            let index = DVec3::new(
                if corner & 1 == 0 {
                    minimum_index.x
                } else {
                    maximum_index.x
                },
                if corner & 2 == 0 {
                    minimum_index.y
                } else {
                    maximum_index.y
                },
                if corner & 4 == 0 {
                    minimum_index.z
                } else {
                    maximum_index.z
                },
            );
            let local = index_to_local.transform_point3(index);
            if !local.is_finite() {
                return Err(invalid_volume(
                    "volume grid transform produces non-finite positions",
                ));
            }
            minimum = minimum.min(local);
            maximum = maximum.max(local);
        }
        Ok((minimum, maximum))
    }
}

impl Default for VolumeGrid {
    fn default() -> Self {
        Self {
            dims: [1, 1, 1],
            voxel_size: 1.0,
            origin: DVec3::ZERO,
            index_to_local: None,
            values: Some(vec![0.0]),
            blob_f32: None,
            content_ref: None,
        }
    }
}

/// Volume data containing one or more named-by-position scalar grids.
/// Scalar density values are rendered as extinction coefficients in inverse meters; ray
/// distance is converted from scene units before applying Beer–Lambert transmittance.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VolumeData {
    /// Scalar grids. Sampling and mesh extraction use the first grid.
    pub grids: Vec<VolumeGrid>,
    /// Provenance; file-backed VDB data is populated in evaluation snapshots.
    pub source: VolumeSource,
    /// Decoded VDB archive attached only to an evaluation snapshot; never serialized.
    #[serde(skip)]
    pub decoded_vdb: Option<Arc<crate::geom::vdb::VdbVolume>>,
}

#[derive(Clone, Copy)]
enum GridSamples<'a> {
    Inline(&'a [f32]),
    Blob(&'a [u8]),
    F64(&'a [f64]),
}

impl GridSamples<'_> {
    fn get(self, index: usize) -> Option<f64> {
        match self {
            Self::Inline(values) => values.get(index).copied().map(f64::from),
            Self::Blob(bytes) => {
                let start = index.checked_mul(std::mem::size_of::<f32>())?;
                let end = start.checked_add(std::mem::size_of::<f32>())?;
                let sample: [u8; 4] = bytes.get(start..end)?.try_into().ok()?;
                Some(f64::from(f32::from_le_bytes(sample)))
            }
            Self::F64(values) => values.get(index).copied(),
        }
    }
}

/// Validate inline volume grids or file-source metadata.
///
/// # Errors
///
/// Returns `UnsupportedFeature` for non-VDB file sources or undecoded external sample blobs,
/// `InvalidArgument` for malformed inline grids, and `LimitExceeded` for resource bounds.
pub fn validate_data(volume: &VolumeData) -> Result<()> {
    match &volume.source {
        VolumeSource::File(source) => {
            if source.format != "vdb" {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("volume file format `{}` is preserve-only", source.format),
                    serde_json::json!({"feature_id":"volume.file_format"}),
                ));
            }
            Ok(())
        }
        VolumeSource::Generated(_) => {
            if volume.grids.len() > MAX_GRID_COUNT {
                return Err(resource_error());
            }
            if volume.grids.is_empty() {
                return Err(invalid_volume("volume must contain at least one grid"));
            }
            for grid in &volume.grids {
                validate_grid(grid)?;
            }
            Ok(())
        }
    }
}

/// Sample the first grid using deterministic trilinear interpolation.
///
/// Positions are volume-local world coordinates. Inline grids use `index_to_local` when present,
/// otherwise their stored origin and voxel size; decoded VDB grids apply their own index transform.
/// This convenience API cannot return an error, so malformed data or unresolved resources return `NaN`;
/// use [`sample_density_checked`] when the caller needs a typed error.
#[must_use]
pub fn sample_density(position: DVec3, volume: &VolumeData) -> f64 {
    sample_density_checked(position, volume).unwrap_or(f64::NAN)
}

/// Fallible form of [`sample_density`], including file-source and data validation.
///
/// # Errors
///
/// Returns `UnsupportedFeature` for preserve-only file inputs and `InvalidArgument`
/// for malformed grids or sample positions.
pub fn sample_density_checked(position: DVec3, volume: &VolumeData) -> Result<f64> {
    sample_density_checked_with(position, volume, VolumeInterpolation::Linear)
}

/// The density at `position` with the material's interpolation, as [`sample_density_checked`] samples it with the
/// trilinear default.
pub fn sample_density_checked_with(
    position: DVec3,
    volume: &VolumeData,
    interpolation: VolumeInterpolation,
) -> Result<f64> {
    ensure_evaluable_source(volume)?;
    if !position.is_finite() {
        return Err(invalid_volume("sample position must be finite"));
    }
    if matches!(&volume.source, VolumeSource::File(_)) {
        let decoded = volume
            .decoded_vdb
            .as_ref()
            .ok_or_else(vdb_resource_unavailable)?;
        let grid = decoded
            .density_grid()
            .ok_or_else(|| invalid_volume("VDB archive contains no density grid"))?;
        return Ok(match interpolation {
            VolumeInterpolation::Linear => grid.sample_world(position),
            VolumeInterpolation::Cubic => grid.sample_world_cubic(position),
        });
    }
    let grid = first_grid(volume)?;
    let values = validate_grid(grid)?;
    let coordinate = grid
        .checked_index_to_local_matrix()?
        .inverse()
        .transform_point3(position);
    sample_index_checked(coordinate, grid, &values, interpolation)
}

/// The inverse of the first inline grid's index-to-local matrix: volume-local positions to index coordinates. The renderer
/// builds it once per volume, so sampling does not invert the matrix at every sample.
pub fn local_to_index_matrix(volume: &VolumeData) -> Result<DMat4> {
    let grid = first_grid(volume)?;
    validate_grid(grid)?;
    Ok(grid.checked_index_to_local_matrix()?.inverse())
}

/// The density at an index-space coordinate of the first inline grid, with the material's interpolation.
/// `local_to_index_matrix` maps volume-local positions to these coordinates.
pub fn sample_index_density_checked_with(
    coordinate: DVec3,
    volume: &VolumeData,
    interpolation: VolumeInterpolation,
) -> Result<f64> {
    let grid = first_grid(volume)?;
    let values = validate_grid(grid)?;
    sample_index_checked(coordinate, grid, &values, interpolation)
}

/// `sample_index_density_checked_with` as a NaN where it errors.
#[must_use]
pub fn sample_index_density_with(
    coordinate: DVec3,
    volume: &VolumeData,
    interpolation: VolumeInterpolation,
) -> f64 {
    sample_index_density_checked_with(coordinate, volume, interpolation).unwrap_or(f64::NAN)
}

/// The interpolated density of an inline grid at an index-space coordinate; a non-finite coordinate samples as zero.
fn sample_index_checked(
    coordinate: DVec3,
    grid: &VolumeGrid,
    values: &GridSamples<'_>,
    interpolation: VolumeInterpolation,
) -> Result<f64> {
    if !coordinate.is_finite() {
        return Ok(0.0);
    }
    // The nodes past the grid are background, zero, and the interpolation reads them up to one node (linear) or two
    // (cubic) beyond the cell, as Cycles' sampling reads past a volume's active voxels (`scene/volume.cpp`).
    let fraction = coordinate - coordinate.floor();
    Ok(match interpolation {
        VolumeInterpolation::Linear => sample_inline_nodes(
            coordinate,
            grid,
            values,
            [
                [1.0 - fraction.x, fraction.x],
                [1.0 - fraction.y, fraction.y],
                [1.0 - fraction.z, fraction.z],
            ],
            0,
        )?,
        VolumeInterpolation::Cubic => sample_inline_nodes(
            coordinate,
            grid,
            values,
            [
                cubic_weights(fraction.x),
                cubic_weights(fraction.y),
                cubic_weights(fraction.z),
            ],
            -1,
        )?,
    })
}

/// Cycles' cubic B-spline weights for the fraction `t` of a cell, for the samples at offsets -1, 0, 1, 2
/// (`fill_cubic_weights`, `intern/cycles/kernel/util/image_3d.h`). The weights are non-negative and sum to one.
#[must_use]
pub fn cubic_weights(t: f64) -> [f64; 4] {
    [
        ((-t / 6.0 + 0.5) * t - 0.5) * t + 1.0 / 6.0,
        ((0.5 * t - 1.0) * t) * t + 2.0 / 3.0,
        ((-0.5 * t + 0.5) * t + 0.5) * t + 1.0 / 6.0,
        t * t * t / 6.0,
    ]
}

/// The inline grid's interpolation at `coordinate` from the nodes around its cell: `weights[axis][i]` is the weight of
/// the node at offset `first + i` from the cell's lower node along that axis, zero beyond the grid
/// (`kernel_image_interp_tricubic_nanovdb` reads its inactive voxels as background).
fn sample_inline_nodes<const N: usize>(
    coordinate: DVec3,
    grid: &VolumeGrid,
    values: &GridSamples<'_>,
    weights: [[f64; N]; 3],
    first: i64,
) -> Result<f64> {
    let lower = coordinate.floor();
    let base = [lower.x as i64, lower.y as i64, lower.z as i64];
    let mut result = 0.0;
    for (k, wz) in weights[2].iter().enumerate() {
        for (j, wy) in weights[1].iter().enumerate() {
            for (i, wx) in weights[0].iter().enumerate() {
                let node = [
                    base[0] + first + i as i64,
                    base[1] + first + j as i64,
                    base[2] + first + k as i64,
                ];
                result += inline_node(grid, values, node)? * wx * wy * wz;
            }
        }
    }
    Ok(result)
}

/// One node's value, zero outside the grid: the background beyond a volume's active voxels.
fn inline_node(grid: &VolumeGrid, values: &GridSamples<'_>, node: [i64; 3]) -> Result<f64> {
    if node
        .iter()
        .zip(grid.dims)
        .any(|(&index, dim)| index < 0 || index >= i64::from(dim))
    {
        return Ok(0.0);
    }
    let index = (node[2] as usize * grid.dims[1] as usize + node[1] as usize)
        * grid.dims[0] as usize
        + node[0] as usize;
    let sample = values
        .get(index)
        .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
    Ok(sample)
}

/// The density with the material's interpolation, as a `NaN` where [`sample_density_checked_with`] errors.
#[must_use]
pub fn sample_density_with(
    position: DVec3,
    volume: &VolumeData,
    interpolation: VolumeInterpolation,
) -> f64 {
    sample_density_checked_with(position, volume, interpolation).unwrap_or(f64::NAN)
}

/// Extract an iso-surface of the first scalar grid using `OpenVDB`-style dual contouring.
///
/// Each intersected cell contributes a feature point per connected contour component. Intersected
/// grid edges connect those points, while adaptivity merges regions into quads or triangles.
///
/// # Errors
///
/// Returns a typed error for unsupported file formats, unresolved VDB resources, invalid grid
/// data or iso-levels, and resource-limit violations. A field that does not cross the iso-level
/// yields an empty mesh, as `OpenVDB` and Blender do.
pub fn volume_to_mesh(volume: &VolumeData, iso_level: f64) -> Result<Mesh> {
    volume_to_mesh_for_grid(
        volume,
        iso_level,
        None,
        0.0,
        DMat4::IDENTITY,
        VolumeToMeshResolution::Grid,
    )
}

/// Extract a named volume grid with `OpenVDB`'s adaptive merging threshold (`0.0..=1.0`).
pub fn volume_to_mesh_named_adaptive(
    volume: &VolumeData,
    iso_level: f64,
    grid_name: &str,
    adaptivity: f64,
) -> Result<Mesh> {
    volume_to_mesh_for_grid(
        volume,
        iso_level,
        Some(grid_name),
        adaptivity,
        DMat4::IDENTITY,
        VolumeToMeshResolution::Grid,
    )
}

/// `volume_to_mesh_named_adaptive` with the output transform `output` applied in double precision
/// before the single rounding to `float` (`OpenVDB` `VolumeToMesh` `indexToWorld`, then `Vec3s`).
/// A Volume to Mesh modifier passes `source_world^-1 * volume_world` here.
///
/// # Errors
///
/// Returns the same errors as [`volume_to_mesh_named_adaptive`], and `InvalidArgument` when the
/// output transform is not finite.
pub fn volume_to_mesh_placed(
    volume: &VolumeData,
    iso_level: f64,
    grid_name: &str,
    adaptivity: f64,
    output: DMat4,
    use_smooth_shade: bool,
    resolution: VolumeToMeshResolution,
) -> Result<Mesh> {
    if !output.is_finite() {
        return Err(invalid_volume(
            "volume_to_mesh output transform is not finite",
        ));
    }
    let mut mesh = volume_to_mesh_for_grid(
        volume,
        iso_level,
        Some(grid_name),
        adaptivity,
        output,
        resolution,
    )?;
    // `mesh_smooth_set(*mesh, flag & VOLUME_TO_MESH_USE_SMOOTH_SHADE)` (`MOD_volume_to_mesh.cc`);
    // the extraction itself produces flat faces.
    if use_smooth_shade {
        crate::geom::normals::shade_faces_smooth(&mut mesh);
    }
    Ok(mesh)
}

/// Blender's `VolumeToMeshResolution` (`BKE_volume_to_mesh.hh`): the grid's own voxels, or the grid
/// resampled to voxels of a given size, or to a given number of voxels along its largest extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VolumeToMeshResolution {
    /// Extract at the grid's own resolution.
    Grid,
    /// Resample to voxels of this size in the output's units (`VOXEL_SIZE`).
    VoxelSize(f64),
    /// Resample to this many voxels along the grid's largest active extent (`VOXEL_AMOUNT`).
    VoxelAmount(u32),
}

/// A dense scalar grid over absolute index coordinates. `values` cover `origin + [0, dims)`, and every
/// other index holds `background`, as an `OpenVDB` grid reads its inactive voxels.
#[derive(Clone)]
struct DenseGrid<T> {
    origin: [i64; 3],
    dims: [usize; 3],
    values: Vec<T>,
    background: T,
}

impl<T: BoxScalar> DenseGrid<T> {
    fn value_at(&self, index: [i64; 3]) -> T {
        let mut local = [0_usize; 3];
        for (axis, slot) in local.iter_mut().enumerate() {
            match usize::try_from(index[axis] - self.origin[axis]) {
                Ok(offset) if offset < self.dims[axis] => *slot = offset,
                _ => return self.background,
            }
        }
        self.values[(local[2] * self.dims[1] + local[1]) * self.dims[0] + local[0]]
    }

    /// `BoxSampler` at an absolute index-space position.
    fn sample(&self, position: DVec3) -> T {
        box_sample_trilinear(|index| self.value_at(index), position)
    }
}

/// One `GridTransformer` pass for a uniform `scale` with `BoxSampler` (`openvdb/tools/GridTransformer.h`,
/// `GridResampler::transformBBox`, affine path): output index `x` samples the input at `x / scale`. The
/// sample position advances by the inverse scale per output voxel, as `OpenVDB`'s loop does.
fn scale_pass<T: BoxScalar>(input: &DenseGrid<T>, scale: f64) -> Result<DenseGrid<T>> {
    // The output box is the scaled input box (`inRMax` is exclusive), padded by `BoxSampler`'s radius of 1.
    let out_min: [i64; 3] =
        std::array::from_fn(|axis| (input.origin[axis] as f64 * scale).floor() as i64 - 1);
    let out_max: [i64; 3] = std::array::from_fn(|axis| {
        ((input.origin[axis] + input.dims[axis] as i64) as f64 * scale).ceil() as i64 + 1
    });
    let dims: [usize; 3] = std::array::from_fn(|axis| (out_max[axis] - out_min[axis] + 1) as usize);
    // The resampled grid is checked before it is allocated: a tiny voxel size makes it large.
    if dims.iter().any(|dimension| *dimension > MAX_GRID_AXIS)
        || dims
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
            .is_none_or(|count| count > MAX_GRID_SAMPLES)
    {
        return Err(resource_error());
    }
    let inverse = 1.0 / scale;
    let mut values = vec![input.background; dims.iter().product()];
    let mut start_x = out_min.map(|index| index as f64 * inverse);
    for x in 0..dims[0] {
        let mut start_y = start_x;
        for y in 0..dims[1] {
            let mut position = start_y;
            for z in 0..dims[2] {
                values[(z * dims[1] + y) * dims[0] + x] = input.sample(DVec3::from_array(position));
                position[2] += inverse;
            }
            start_y[1] += inverse;
        }
        start_x[0] += inverse;
    }
    Ok(DenseGrid {
        origin: out_min,
        dims,
        values,
        background: input.background,
    })
}

/// `GridTransformer::transformGrid` for a uniform `factor` with `BoxSampler`. While the factor is below
/// one half, the grid is halved along every axis (`mMipLevels`, `mipmap()`), and the remaining scale
/// is then applied in one pass; otherwise the factor is applied directly.
fn resample_uniform<T: BoxScalar>(input: DenseGrid<T>, factor: f64) -> Result<DenseGrid<T>> {
    if factor >= 0.5 {
        return scale_pass(&input, factor);
    }
    let halvings = (-factor.ln() / 2.0_f64.ln()).floor() as i32;
    let remainder = factor * f64::from(1_i32 << halvings);
    let mut current = input;
    for _ in 0..halvings {
        current = scale_pass(&current, 0.5)?;
    }
    // `mPostScaleTransform` is the identity when the remainder is exactly one.
    if remainder.to_bits() != 1.0_f64.to_bits() {
        current = scale_pass(&current, remainder)?;
    }
    Ok(current)
}

/// Blender's `VolumeToMeshOp::compute_resolution_factor` (`volume_to_mesh.cc`) in single precision:
/// the largest voxel size of the transformed grid over the desired voxel size. `active` is the
/// absolute index box of the grid's active voxels, inclusive. `None` when no factor exists, which
/// Blender reports as an empty mesh.
fn resolution_factor(
    index_to_world: DMat4,
    active: ([i64; 3], [i64; 3]),
    resolution: VolumeToMeshResolution,
) -> Option<f32> {
    let current = [
        index_to_world.x_axis,
        index_to_world.y_axis,
        index_to_world.z_axis,
    ]
    .iter()
    .map(|column| column.truncate().length() as f32)
    .fold(0.0_f32, f32::max);
    let desired = match resolution {
        VolumeToMeshResolution::Grid => return None,
        VolumeToMeshResolution::VoxelSize(size) => size as f32,
        VolumeToMeshResolution::VoxelAmount(amount) => {
            // `indexToWorld` of the inclusive active box: the extents of its world corners.
            let low = DVec3::from_array(active.0.map(|index| index as f64));
            let high = DVec3::from_array(active.1.map(|index| index as f64));
            let mut minimum = DVec3::splat(f64::INFINITY);
            let mut maximum = DVec3::splat(f64::NEG_INFINITY);
            for corner in 0..8_u8 {
                let index = DVec3::new(
                    if corner & 1 == 0 { low.x } else { high.x },
                    if corner & 2 == 0 { low.y } else { high.y },
                    if corner & 4 == 0 { low.z } else { high.z },
                );
                let world = index_to_world.transform_point3(index);
                minimum = minimum.min(world);
                maximum = maximum.max(world);
            }
            let max_extent = (maximum - minimum).max_element() as f32;
            if max_extent.is_nan() || max_extent <= 0.0 {
                return None;
            }
            max_extent / amount as f32
        }
    };
    if desired.is_nan() || desired <= 0.0 {
        return None;
    }
    Some(current / desired)
}

/// The inline grid of a non-`GRID` resolution: the grid resampled by `factor`, with the layout
/// `index_to_local = scale(1 / factor) * translation(origin)` that the extraction maps through
/// `output * index_to_world`, where `index_to_world` is the grid's absolute index-to-local map.
fn resample_inline_grid(
    grid: &VolumeGrid,
    values: GridSamples<'_>,
    active: ([i64; 3], [i64; 3]),
    output: DMat4,
    resolution: VolumeToMeshResolution,
) -> Result<Option<(VolumeGrid, DMat4)>> {
    // Blender indexes the grid by its absolute voxel index; a legacy grid's origin is that index's offset.
    let offset = global_grid_origin(grid);
    let index_to_world = grid.checked_index_to_local_matrix()?
        * DMat4::from_translation(-DVec3::from_array(offset.map(|index| index as f64)));
    let index_to_output = output * index_to_world;
    let world_active = (
        std::array::from_fn(|axis| active.0[axis] + offset[axis]),
        std::array::from_fn(|axis| active.1[axis] + offset[axis]),
    );
    let Some(factor) = resolution_factor(index_to_output, world_active, resolution) else {
        return Ok(None);
    };
    // Inline grids are float and read outside their samples as zero (`inline_node`, `outside_value`).
    let count = grid
        .dims
        .iter()
        .map(|dimension| *dimension as usize)
        .product();
    let mut input_values = Vec::with_capacity(count);
    for index in 0..count {
        let sample = values
            .get(index)
            .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
        input_values.push(<f32 as BoxScalar>::from_f64(sample));
    }
    let input = DenseGrid {
        origin: offset,
        dims: grid.dims.map(|dimension| dimension as usize),
        values: input_values,
        background: 0.0,
    };
    let resampled = resample_uniform(input, f64::from(factor))?;
    // The layout carries the grid's own index-to-world map, so its determinant keeps a mirror's sign.
    let layout = index_to_world * resampled_layout(factor, resampled.origin);
    let layout_grid = VolumeGrid {
        dims: grid_dimensions(resampled.dims)?,
        voxel_size: grid.voxel_size / f64::from(factor),
        origin: layout.transform_point3(DVec3::ZERO),
        index_to_local: Some(layout.to_cols_array()),
        values: Some(resampled.values),
        blob_f32: None,
        content_ref: None,
    };
    Ok(Some((layout_grid, output)))
}

/// The map from a resampled grid's index to the index of the grid it was resampled from. Grid index `g`
/// is absolute index `g + origin`, and absolute index `a` maps to `a / factor`, which is Blender's
/// `preScale(1.0f / resolution_factor)` on the grid's transform.
fn resampled_layout(factor: f32, origin: [i64; 3]) -> DMat4 {
    DMat4::from_scale(DVec3::splat(f64::from(1.0_f32 / factor)))
        * DMat4::from_translation(DVec3::from_array(origin.map(|index| index as f64)))
}

fn grid_dimensions(dims: [usize; 3]) -> Result<[u32; 3]> {
    let mut grid_dims = [0_u32; 3];
    for (slot, dimension) in grid_dims.iter_mut().zip(dims) {
        *slot = u32::try_from(dimension).map_err(|_| resource_error())?;
    }
    Ok(grid_dims)
}

/// A dense VDB grid resampled for a non-GRID resolution, with the values in double precision. A float
/// grid interpolates in `float`, as Blender's `BoxSampler` does for it.
fn resample_dense_vdb(
    origin: [i64; 3],
    dims: [usize; 3],
    values: &[f64],
    background: f64,
    single: bool,
    factor: f32,
) -> Result<(VolumeGrid, Vec<f64>)> {
    let (resampled_origin, resampled_dims, resampled_values) = if single {
        let input = DenseGrid {
            origin,
            dims,
            values: values
                .iter()
                .map(|value| <f32 as BoxScalar>::from_f64(*value))
                .collect(),
            background: <f32 as BoxScalar>::from_f64(background),
        };
        let resampled = resample_uniform(input, f64::from(factor))?;
        (
            resampled.origin,
            resampled.dims,
            resampled
                .values
                .iter()
                .map(|value| value.to_f64())
                .collect(),
        )
    } else {
        let input = DenseGrid {
            origin,
            dims,
            values: values.to_vec(),
            background,
        };
        let resampled = resample_uniform(input, f64::from(factor))?;
        (resampled.origin, resampled.dims, resampled.values)
    };
    let layout = VolumeGrid {
        dims: grid_dimensions(resampled_dims)?,
        voxel_size: 1.0 / f64::from(factor),
        origin: resampled_layout(factor, resampled_origin).transform_point3(DVec3::ZERO),
        index_to_local: Some(resampled_layout(factor, resampled_origin).to_cols_array()),
        values: None,
        blob_f32: None,
        content_ref: None,
    };
    Ok((layout, resampled_values))
}

fn volume_to_mesh_for_grid(
    volume: &VolumeData,
    iso_level: f64,
    grid_name: Option<&str>,
    adaptivity: f64,
    output: DMat4,
    resolution: VolumeToMeshResolution,
) -> Result<Mesh> {
    ensure_evaluable_source(volume)?;
    if !iso_level.is_finite() {
        return Err(invalid_volume("iso_level must be finite"));
    }
    if !adaptivity.is_finite() || !(0.0..=1.0).contains(&adaptivity) {
        return Err(invalid_volume(
            "adaptivity must be finite and between 0 and 1",
        ));
    }
    // Blender passes the threshold as a `float` (`volumeToMesh(..., const float threshold, ...)`), so
    // `OpenVDB`'s double isovalue is the single-precision value; the edge fractions use that value.
    let iso_level = f64::from(iso_level as f32);
    let adaptivity = adaptivity as f32;
    if matches!(&volume.source, VolumeSource::File(_)) {
        let decoded = volume
            .decoded_vdb
            .as_ref()
            .ok_or_else(vdb_resource_unavailable)?;
        let grid = if let Some(grid_name) = grid_name {
            decoded
                .grids
                .iter()
                .find(|grid| grid.name == grid_name)
                .ok_or_else(|| grid_not_found(grid_name))?
        } else {
            decoded
                .density_grid()
                .ok_or_else(|| invalid_volume("VDB archive contains no density grid"))?
        };
        let inside_is_low = grid.class.as_deref().is_some_and(|class| {
            matches!(
                class,
                "level set" | "Level Set" | "LevelSet" | "level_set" | "levelset"
            )
        });
        return vdb_grid_to_mesh(
            grid,
            iso_level,
            inside_is_low,
            adaptivity,
            output,
            resolution,
        );
    }
    if let Some(grid_name) = grid_name
        && (grid_name != "density" || volume.grids.is_empty())
    {
        return Err(grid_not_found(grid_name));
    }
    if volume.grids.is_empty() {
        return Ok(Mesh::default());
    }
    let grid = first_grid(volume)?;
    let values = validate_grid(grid)?;
    let dims = grid.dims.map(|dimension| dimension as usize);
    if dims.iter().any(|dimension| *dimension < 2) {
        return Err(invalid_volume(
            "surface extraction requires at least two samples on each axis",
        ));
    }
    let inside_is_low = matches!(
        &volume.source,
        VolumeSource::Generated(source) if source.algorithm == "signed_distance"
    );
    // A non-GRID resolution extracts the resampled grid, with the output transform of its voxels.
    let mut resampled_grid = None;
    let (grid, values, output) = if matches!(resolution, VolumeToMeshResolution::Grid) {
        (grid, values, output)
    } else {
        let active = match &volume.source {
            VolumeSource::Generated(source) => source.active_bounds,
            VolumeSource::File(_) => None,
        }
        .unwrap_or_else(|| {
            (
                [0_i64; 3],
                grid.dims.map(|dimension| i64::from(dimension) - 1),
            )
        });
        let Some((layout, world_output)) =
            resample_inline_grid(grid, values, active, output, resolution)?
        else {
            return Ok(Mesh::default());
        };
        let layout = resampled_grid.insert(layout);
        let layout_values = validate_grid(layout)?;
        (&*layout, layout_values, world_output)
    };
    let boundary_crosses = grid_boundary_crosses(grid, values, iso_level)?;
    let mut mesh = if boundary_crosses {
        let padded = pad_volume_grid(grid, values)?;
        let padded_values = padded
            .values
            .as_deref()
            .ok_or_else(|| invalid_volume("padded volume grid has no samples"))?;
        dual_contour_grid_to_mesh(
            &padded,
            GridSamples::Inline(padded_values),
            iso_level,
            inside_is_low,
            adaptivity,
            0.0,
            output,
        )?
    } else {
        dual_contour_grid_to_mesh(
            grid,
            values,
            iso_level,
            inside_is_low,
            adaptivity,
            0.0,
            output,
        )?
    };
    if boundary_crosses {
        let maximum_index = DVec3::new(
            f64::from(grid.dims[0] - 1),
            f64::from(grid.dims[1] - 1),
            f64::from(grid.dims[2] - 1),
        );
        let index_to_output = output * grid.checked_index_to_local_matrix()?;
        let output_to_index = index_to_output.inverse();
        if !output_to_index.is_finite() {
            return Err(invalid_volume(
                "volume grid index transform is not invertible",
            ));
        }
        grid.local_bounds(DVec3::ZERO, maximum_index)?;
        for vertex in &mut mesh.vertices {
            let index = output_to_index.transform_point3(vertex.co);
            if !index.is_finite() {
                return Err(invalid_volume("volume contour index is not finite"));
            }
            vertex.co = index_to_output.transform_point3(index.clamp(DVec3::ZERO, maximum_index));
            if !vertex.co.is_finite() {
                return Err(invalid_volume("volume contour position is not finite"));
            }
        }
    }
    round_mesh_positions_to_f32(&mut mesh);
    Ok(mesh)
}

fn grid_boundary_crosses(
    grid: &VolumeGrid,
    values: GridSamples<'_>,
    iso_level: f64,
) -> Result<bool> {
    let [nx, ny, nz] = grid.dims.map(|dimension| dimension as usize);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let first = grid_sample(values, x, y, z, nx, ny)? < iso_level;
                if x + 1 < nx
                    && (x == 0 || x + 1 == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
                    && first != (grid_sample(values, x + 1, y, z, nx, ny)? < iso_level)
                {
                    return Ok(true);
                }
                if y + 1 < ny
                    && (x == 0 || x == nx - 1 || y == 0 || y + 1 == ny - 1 || z == 0 || z == nz - 1)
                    && first != (grid_sample(values, x, y + 1, z, nx, ny)? < iso_level)
                {
                    return Ok(true);
                }
                if z + 1 < nz
                    && (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z + 1 == nz - 1)
                    && first != (grid_sample(values, x, y, z + 1, nx, ny)? < iso_level)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Extend boundary samples so dual contour faces retain the source grid's open boundary.
fn pad_volume_grid(grid: &VolumeGrid, values: GridSamples<'_>) -> Result<VolumeGrid> {
    let dims = grid.dims.map(|dimension| dimension as usize);
    let padded_dims = [
        grid.dims[0].checked_add(2).ok_or_else(resource_error)?,
        grid.dims[1].checked_add(2).ok_or_else(resource_error)?,
        grid.dims[2].checked_add(2).ok_or_else(resource_error)?,
    ];
    let padded_shape = padded_dims.map(|dimension| dimension as usize);
    let sample_count = padded_shape[0]
        .checked_mul(padded_shape[1])
        .and_then(|count| count.checked_mul(padded_shape[2]))
        .ok_or_else(resource_error)?;
    let cell_count = (padded_shape[0] - 1)
        .checked_mul(padded_shape[1] - 1)
        .and_then(|count| count.checked_mul(padded_shape[2] - 1))
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_SAMPLES {
        return Err(resource_error());
    }
    let mut padded_values = vec![0.0_f32; sample_count];
    for z in 0..padded_shape[2] {
        let source_z = z.saturating_sub(1).min(dims[2] - 1);
        for y in 0..padded_shape[1] {
            let source_y = y.saturating_sub(1).min(dims[1] - 1);
            for x in 0..padded_shape[0] {
                let source_x = x.saturating_sub(1).min(dims[0] - 1);
                let source_index = (source_z * dims[1] + source_y) * dims[0] + source_x;
                let sample = values
                    .get(source_index)
                    .ok_or_else(|| invalid_volume("volume sample index is out of range"))?
                    as f32;
                if !sample.is_finite() {
                    return Err(invalid_volume("volume sample is outside f32 range"));
                }
                let padded_index = (z * padded_shape[1] + y) * padded_shape[0] + x;
                padded_values[padded_index] = sample;
            }
        }
    }
    let output_transform = grid
        .index_to_local
        .map(|_| {
            grid.checked_index_to_local_matrix()
                .map(|transform| transform * DMat4::from_translation(-DVec3::ONE))
        })
        .transpose()?;
    let origin = output_transform.map_or_else(
        || grid.origin - DVec3::splat(grid.voxel_size),
        |transform| transform.transform_point3(DVec3::ZERO),
    );
    if !origin.is_finite() {
        return Err(invalid_volume("padded volume origin is not finite"));
    }
    Ok(VolumeGrid {
        dims: padded_dims,
        voxel_size: grid.voxel_size,
        origin,
        index_to_local: output_transform.map(|transform| transform.to_cols_array()),
        values: Some(padded_values),
        blob_f32: None,
        content_ref: None,
    })
}

const CUBE_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

const CUBE_FACES: [([usize; 4], [usize; 4]); 6] = [
    ([0, 1, 2, 3], [0, 1, 2, 3]),
    ([4, 5, 6, 7], [4, 5, 6, 7]),
    ([0, 1, 5, 4], [0, 9, 4, 8]),
    ([3, 2, 6, 7], [2, 10, 6, 11]),
    ([0, 3, 7, 4], [3, 11, 7, 8]),
    ([1, 2, 6, 5], [1, 10, 5, 9]),
];
// `OpenVDB` 13.0.0 VolumeToMesh::sAdaptable (Apache-2.0 source table).
const OPENVDB_ADAPTABLE: [u8; 256] = [
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
];

fn dual_contour_grid_to_mesh(
    grid: &VolumeGrid,
    values: GridSamples<'_>,
    iso_level: f64,
    inside_is_low: bool,
    adaptivity: f32,
    outside_value: f64,
    output: DMat4,
) -> Result<Mesh> {
    let nx = usize::try_from(grid.dims[0]).map_err(|_| resource_error())?;
    let ny = usize::try_from(grid.dims[1]).map_err(|_| resource_error())?;
    let nz = usize::try_from(grid.dims[2]).map_err(|_| resource_error())?;
    let cell_dims = [nx - 1, ny - 1, nz - 1];
    let cell_count = cell_dims[0]
        .checked_mul(cell_dims[1])
        .and_then(|count| count.checked_mul(cell_dims[2]))
        .ok_or_else(resource_error)?;
    // A dense grid has fewer cells than samples, so the sample cap bounds the cell work too.
    if cell_count > MAX_GRID_SAMPLES {
        return Err(resource_error());
    }
    let index_to_local = grid.checked_index_to_local_matrix()?;
    let index_to_output = output * index_to_local;

    let mut positions = Vec::new();
    let mut cell_edges = HashMap::<usize, [usize; 12]>::new();
    for z in 0..cell_dims[2] {
        for y in 0..cell_dims[1] {
            for x in 0..cell_dims[0] {
                let nodes = cube_node_indices(x, y, z, nx, ny);
                let mut cube_values = [0.0_f64; 8];
                for corner in 0..CUBE_CORNERS.len() {
                    cube_values[corner] = values
                        .get(nodes[corner])
                        .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
                }
                let mut crossing = [false; 12];
                let mut edge_points = [DVec3::ZERO; 12];
                let mut parent = std::array::from_fn(|edge| edge);
                let mut crossing_count = 0;
                for (edge, [first, second]) in CUBE_EDGES.iter().copied().enumerate() {
                    let first_inside = cube_values[first] < iso_level;
                    let second_inside = cube_values[second] < iso_level;
                    if first_inside == second_inside {
                        continue;
                    }
                    crossing[edge] = true;
                    crossing_count += 1;
                    let fraction = (iso_level - cube_values[first])
                        / (cube_values[second] - cube_values[first]);
                    if !fraction.is_finite() {
                        return Err(invalid_volume("iso-level interpolation is not finite"));
                    }
                    let first_offset = CUBE_CORNERS[first];
                    let second_offset = CUBE_CORNERS[second];
                    let local = DVec3::new(
                        first_offset[0] as f64,
                        first_offset[1] as f64,
                        first_offset[2] as f64,
                    )
                    .lerp(
                        DVec3::new(
                            second_offset[0] as f64,
                            second_offset[1] as f64,
                            second_offset[2] as f64,
                        ),
                        fraction.clamp(0.0, 1.0),
                    );
                    edge_points[edge] = local;
                }
                if crossing_count == 0 {
                    continue;
                }

                for (face_corners, face_edges) in CUBE_FACES {
                    let mut crossed_edges = [0_usize; 4];
                    let mut crossed_count = 0;
                    for (index, edge) in face_edges.iter().copied().enumerate() {
                        if crossing[edge] {
                            crossed_edges[crossed_count] = index;
                            crossed_count += 1;
                        }
                    }
                    match crossed_count {
                        0 => {}
                        2 => join_edge_groups(
                            &mut parent,
                            face_edges[crossed_edges[0]],
                            face_edges[crossed_edges[1]],
                        ),
                        4 => {
                            // Saddle face: the pairing keeps corner 0's side apart from the other side
                            // (`OpenVDB`'s VolumeToMesh resolves the face by its inside). Inside is low for a
                            // level set and high for a density grid, so a fog's high-density corners are
                            // separated, as Blender's 5.2.2 mesh does (`rotated_nonuniform` at 0.35: 112 points).
                            let corner_inside = if inside_is_low {
                                cube_values[face_corners[0]] < iso_level
                            } else {
                                cube_values[face_corners[0]] >= iso_level
                            };
                            let pairs = if corner_inside {
                                [
                                    (face_edges[crossed_edges[0]], face_edges[crossed_edges[3]]),
                                    (face_edges[crossed_edges[1]], face_edges[crossed_edges[2]]),
                                ]
                            } else {
                                [
                                    (face_edges[crossed_edges[0]], face_edges[crossed_edges[1]]),
                                    (face_edges[crossed_edges[2]], face_edges[crossed_edges[3]]),
                                ]
                            };
                            for (first, second) in pairs {
                                join_edge_groups(&mut parent, first, second);
                            }
                        }
                        _ => return Err(invalid_volume("invalid cube contour edge count")),
                    }
                }

                let mut grouped_points = [DVec3::ZERO; 12];
                let mut grouped_counts = [0_u8; 12];
                let mut edge_vertices = [usize::MAX; 12];
                for edge in 0..CUBE_EDGES.len() {
                    if !crossing[edge] {
                        continue;
                    }
                    let group = find_edge_group(&mut parent, edge);
                    grouped_points[group] += edge_points[edge];
                    grouped_counts[group] += 1;
                }
                for edge in 0..CUBE_EDGES.len() {
                    if !crossing[edge] {
                        continue;
                    }
                    let group = find_edge_group(&mut parent, edge);
                    if edge_vertices[edge] != usize::MAX {
                        continue;
                    }
                    if grouped_counts[group] == 0 {
                        return Err(invalid_volume("cube contour group is empty"));
                    }
                    let point = grouped_points[group] / f64::from(grouped_counts[group]);
                    let index_position = DVec3::new(x as f64, y as f64, z as f64) + point;
                    let position = index_to_output.transform_point3(index_position);
                    if !position.is_finite() || positions.len() >= u32::MAX as usize {
                        return Err(resource_error());
                    }
                    let vertex = positions.len();
                    positions.push(position);
                    for group_edge in 0..CUBE_EDGES.len() {
                        if crossing[group_edge] && find_edge_group(&mut parent, group_edge) == group
                        {
                            edge_vertices[group_edge] = vertex;
                        }
                    }
                }
                let cell = cell_linear_index(x, y, z, cell_dims);
                cell_edges.insert(cell, edge_vertices);
            }
        }
    }

    // A field with no crossing cell has no contour, which Blender's Volume to Mesh returns as an
    // empty mesh rather than an error.
    if positions.is_empty() {
        return Ok(Mesh::default());
    }
    let adaptive = adaptivity > 1.0e-7;
    if adaptive {
        merge_adaptive_regions(
            grid,
            values,
            cell_dims,
            iso_level,
            adaptivity,
            outside_value,
            &mut positions,
            &mut cell_edges,
        )?;
    }

    let mut faces = Vec::new();
    for z in 1..nz - 1 {
        for y in 1..ny - 1 {
            for x in 0..nx - 1 {
                let first = grid_sample(values, x, y, z, nx, ny)?;
                let second = grid_sample(values, x + 1, y, z, nx, ny)?;
                if (first < iso_level) == (second < iso_level) {
                    continue;
                }
                append_dual_face(
                    &cell_edges,
                    &mut faces,
                    cell_dims,
                    [
                        ((x, y - 1, z - 1), 6),
                        ((x, y, z - 1), 4),
                        ((x, y, z), 0),
                        ((x, y - 1, z), 2),
                    ],
                    first < iso_level,
                    inside_is_low,
                    adaptive,
                )?;
            }
        }
    }
    for z in 1..nz - 1 {
        for y in 0..ny - 1 {
            for x in 1..nx - 1 {
                let first = grid_sample(values, x, y, z, nx, ny)?;
                let second = grid_sample(values, x, y + 1, z, nx, ny)?;
                if (first < iso_level) == (second < iso_level) {
                    continue;
                }
                append_dual_face(
                    &cell_edges,
                    &mut faces,
                    cell_dims,
                    [
                        ((x - 1, y, z - 1), 5),
                        ((x - 1, y, z), 1),
                        ((x, y, z), 3),
                        ((x, y, z - 1), 7),
                    ],
                    first < iso_level,
                    inside_is_low,
                    adaptive,
                )?;
            }
        }
    }
    for z in 0..nz - 1 {
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let first = grid_sample(values, x, y, z, nx, ny)?;
                let second = grid_sample(values, x, y, z + 1, nx, ny)?;
                if (first < iso_level) == (second < iso_level) {
                    continue;
                }
                append_dual_face(
                    &cell_edges,
                    &mut faces,
                    cell_dims,
                    [
                        ((x - 1, y - 1, z), 10),
                        ((x, y - 1, z), 11),
                        ((x, y, z), 8),
                        ((x - 1, y, z), 9),
                    ],
                    first < iso_level,
                    inside_is_low,
                    adaptive,
                )?;
            }
        }
    }
    // `OpenVDB`'s volumeToMesh yields no faces for a field that does not cross the iso-level, and
    // Blender's Volume to Mesh keeps that empty result rather than reporting an error.
    if faces.is_empty() {
        return Ok(Mesh::default());
    }
    if index_to_local.determinant() < 0.0 {
        for face in &mut faces {
            face.reverse();
        }
    }
    Mesh::from_positions_and_faces(positions, faces).map_err(|error| mesh_output_error(&error))
}

/// Stores each position as a `float`, as Blender's `Vec3s` mesh positions are.
fn round_mesh_positions_to_f32(mesh: &mut Mesh) {
    for vertex in &mut mesh.vertices {
        vertex.co = vertex.co.as_vec3().as_dvec3();
    }
}

fn cube_node_indices(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> [usize; 8] {
    let node = |x, y, z| (z * ny + y) * nx + x;
    [
        node(x, y, z),
        node(x + 1, y, z),
        node(x + 1, y + 1, z),
        node(x, y + 1, z),
        node(x, y, z + 1),
        node(x + 1, y, z + 1),
        node(x + 1, y + 1, z + 1),
        node(x, y + 1, z + 1),
    ]
}

fn grid_sample(
    values: GridSamples<'_>,
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
) -> Result<f64> {
    let index = z
        .checked_mul(ny)
        .and_then(|row| row.checked_add(y))
        .and_then(|row| row.checked_mul(nx))
        .and_then(|row| row.checked_add(x))
        .ok_or_else(resource_error)?;
    values
        .get(index)
        .ok_or_else(|| invalid_volume("volume sample index is out of range"))
}

fn cell_linear_index(x: usize, y: usize, z: usize, dims: [usize; 3]) -> usize {
    (z * dims[1] + y) * dims[0] + x
}

fn find_edge_group(parent: &mut [usize; 12], edge: usize) -> usize {
    let mut root = edge;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = edge;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn join_edge_groups(parent: &mut [usize; 12], first: usize, second: usize) {
    let first = find_edge_group(parent, first);
    let second = find_edge_group(parent, second);
    if first != second {
        let (root, child) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        parent[child] = root;
    }
}

fn global_grid_origin(grid: &VolumeGrid) -> [i64; 3] {
    if grid.index_to_local.is_some() {
        return [0; 3];
    }
    let index_origin = grid.origin / grid.voxel_size;
    [
        index_origin.x.round() as i64,
        index_origin.y.round() as i64,
        index_origin.z.round() as i64,
    ]
}

fn global_grid_sample(
    values: GridSamples<'_>,
    dims: [usize; 3],
    origin: [i64; 3],
    coordinate: [i64; 3],
    outside_value: f64,
) -> f64 {
    let local = [
        coordinate[0] - origin[0],
        coordinate[1] - origin[1],
        coordinate[2] - origin[2],
    ];
    if local
        .iter()
        .zip(dims)
        .any(|(axis, dimension)| *axis < 0 || *axis >= dimension as i64)
    {
        return outside_value;
    }
    let [x, y, z] = local.map(|axis| axis as usize);
    values
        .get((z * dims[1] + y) * dims[0] + x)
        .unwrap_or(outside_value)
}

fn global_cell_coordinates(cell: usize, cell_dims: [usize; 3], grid_origin: [i64; 3]) -> [i64; 3] {
    let x = cell % cell_dims[0];
    let y = (cell / cell_dims[0]) % cell_dims[1];
    let z = cell / (cell_dims[0] * cell_dims[1]);
    [
        grid_origin[0] + x as i64,
        grid_origin[1] + y as i64,
        grid_origin[2] + z as i64,
    ]
}

fn adaptable_signs(signs: [bool; 8]) -> bool {
    let openvdb_order = [0, 1, 5, 4, 3, 2, 6, 7];
    let mask = openvdb_order
        .iter()
        .enumerate()
        .fold(0_usize, |mask, (bit, corner)| {
            mask | (usize::from(signs[*corner]) << bit)
        });
    OPENVDB_ADAPTABLE[mask] != 0
}

fn region_corner_signs(
    values: GridSamples<'_>,
    dims: [usize; 3],
    grid_origin: [i64; 3],
    start: [i64; 3],
    extent: i64,
    outside_value: f64,
    iso_level: f64,
) -> [bool; 8] {
    CUBE_CORNERS.map(|offset| {
        let coordinate = [
            start[0] + offset[0] as i64 * extent,
            start[1] + offset[1] as i64 * extent,
            start[2] + offset[2] as i64 * extent,
        ];
        global_grid_sample(values, dims, grid_origin, coordinate, outside_value) < iso_level
    })
}

fn region_is_non_manifold(
    values: GridSamples<'_>,
    dims: [usize; 3],
    grid_origin: [i64; 3],
    start: [i64; 3],
    extent: i64,
    outside_value: f64,
    iso_level: f64,
) -> bool {
    let signs = region_corner_signs(
        values,
        dims,
        grid_origin,
        start,
        extent,
        outside_value,
        iso_level,
    );
    if !adaptable_signs(signs) {
        return true;
    }
    let half = extent / 2;
    for [first, second] in CUBE_EDGES {
        let coordinate = [
            start[0] + (CUBE_CORNERS[first][0] as i64 + CUBE_CORNERS[second][0] as i64) * half,
            start[1] + (CUBE_CORNERS[first][1] as i64 + CUBE_CORNERS[second][1] as i64) * half,
            start[2] + (CUBE_CORNERS[first][2] as i64 + CUBE_CORNERS[second][2] as i64) * half,
        ];
        let middle =
            global_grid_sample(values, dims, grid_origin, coordinate, outside_value) < iso_level;
        if signs[first] != middle && signs[second] != middle {
            return true;
        }
    }
    for (face_corners, _) in CUBE_FACES {
        let coordinate = [
            start[0]
                + face_corners
                    .iter()
                    .map(|corner| CUBE_CORNERS[*corner][0] as i64)
                    .sum::<i64>()
                    * extent
                    / 4,
            start[1]
                + face_corners
                    .iter()
                    .map(|corner| CUBE_CORNERS[*corner][1] as i64)
                    .sum::<i64>()
                    * extent
                    / 4,
            start[2]
                + face_corners
                    .iter()
                    .map(|corner| CUBE_CORNERS[*corner][2] as i64)
                    .sum::<i64>()
                    * extent
                    / 4,
        ];
        let middle =
            global_grid_sample(values, dims, grid_origin, coordinate, outside_value) < iso_level;
        if face_corners.iter().all(|corner| signs[*corner] != middle) {
            return true;
        }
    }
    let center = [start[0] + half, start[1] + half, start[2] + half];
    let center_inside =
        global_grid_sample(values, dims, grid_origin, center, outside_value) < iso_level;
    signs.iter().all(|inside| *inside != center_inside)
}

fn grid_gradient(
    values: GridSamples<'_>,
    dims: [usize; 3],
    grid_origin: [i64; 3],
    coordinate: [i64; 3],
    outside_value: f64,
) -> Vec3 {
    let mut low = coordinate;
    let mut high = coordinate;
    low[0] -= 1;
    high[0] += 1;
    let x = (global_grid_sample(values, dims, grid_origin, high, outside_value)
        - global_grid_sample(values, dims, grid_origin, low, outside_value)) as f32;
    low = coordinate;
    high = coordinate;
    low[1] -= 1;
    high[1] += 1;
    let y = (global_grid_sample(values, dims, grid_origin, high, outside_value)
        - global_grid_sample(values, dims, grid_origin, low, outside_value)) as f32;
    low = coordinate;
    high = coordinate;
    low[2] -= 1;
    high[2] += 1;
    let z = (global_grid_sample(values, dims, grid_origin, high, outside_value)
        - global_grid_sample(values, dims, grid_origin, low, outside_value)) as f32;
    (Vec3::new(x, y, z) * 0.5).normalize_or_zero()
}

fn adaptive_region_vertices(
    cell_edges: &HashMap<usize, [usize; 12]>,
    cell_dims: [usize; 3],
    grid_origin: [i64; 3],
    sample_dims: [usize; 3],
    values: GridSamples<'_>,
    outside_value: f64,
    start: [i64; 3],
    extent: i64,
    normal_to_local: DMat4,
) -> (Vec<usize>, Vec<Vec3>, bool) {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut split_cell = false;
    for z in 0..extent {
        for y in 0..extent {
            for x in 0..extent {
                let global = [start[0] + x, start[1] + y, start[2] + z];
                let local = [
                    global[0] - grid_origin[0],
                    global[1] - grid_origin[1],
                    global[2] - grid_origin[2],
                ];
                if local
                    .iter()
                    .zip(cell_dims)
                    .any(|(axis, dimension)| *axis < 0 || *axis >= dimension as i64)
                {
                    continue;
                }
                let [cx, cy, cz] = local.map(|axis| axis as usize);
                let cell = cell_linear_index(cx, cy, cz, cell_dims);
                let Some(edges) = cell_edges.get(&cell) else {
                    continue;
                };
                let mut cell_vertices = [usize::MAX; 12];
                let mut vertex_count = 0;
                for vertex in edges.iter().copied().filter(|vertex| *vertex != usize::MAX) {
                    if !cell_vertices[..vertex_count].contains(&vertex) {
                        cell_vertices[vertex_count] = vertex;
                        vertex_count += 1;
                    }
                }
                if vertex_count > 1 {
                    split_cell = true;
                }
                vertices.extend_from_slice(&cell_vertices[..vertex_count]);
                let index_gradient =
                    grid_gradient(values, sample_dims, grid_origin, global, outside_value);
                let local_gradient = normal_to_local.transform_vector3(DVec3::new(
                    f64::from(index_gradient.x),
                    f64::from(index_gradient.y),
                    f64::from(index_gradient.z),
                ));
                normals.push(local_gradient.as_vec3().normalize_or_zero());
            }
        }
    }
    (vertices, normals, split_cell)
}

fn merge_adaptive_regions(
    grid: &VolumeGrid,
    values: GridSamples<'_>,
    cell_dims: [usize; 3],
    iso_level: f64,
    adaptivity: f32,
    outside_value: f64,
    positions: &mut Vec<DVec3>,
    cell_edges: &mut HashMap<usize, [usize; 12]>,
) -> Result<()> {
    let sample_dims = grid.dims.map(|dimension| dimension as usize);
    let grid_origin = global_grid_origin(grid);
    let normal_to_local = grid.checked_index_to_local_matrix()?.inverse().transpose();
    let mut leaves = HashSet::new();
    let mut blocked = HashSet::<([i64; 3], i64)>::new();
    for (cell, edges) in &*cell_edges {
        let coordinate = global_cell_coordinates(*cell, cell_dims, grid_origin);
        let leaf = coordinate.map(|axis| axis.div_euclid(8) * 8);
        leaves.insert(leaf);
        let group_count = edges
            .iter()
            .filter(|edge| **edge != usize::MAX)
            .copied()
            .collect::<HashSet<_>>()
            .len();
        let signs = region_corner_signs(
            values,
            sample_dims,
            grid_origin,
            coordinate,
            1,
            outside_value,
            iso_level,
        );
        if group_count > 1 || !adaptable_signs(signs) {
            let start = coordinate.map(|axis| axis.div_euclid(2) * 2);
            blocked.insert((start, 2));
        }
    }
    let mut leaves = leaves.into_iter().collect::<Vec<_>>();
    leaves.sort_unstable();
    let mut parent = (0..positions.len()).collect::<Vec<_>>();
    for dimension in [2_i64, 4, 8] {
        let step = dimension as usize;
        for leaf in &leaves {
            for z in (0..8_i64).step_by(step) {
                for y in (0..8_i64).step_by(step) {
                    for x in (0..8_i64).step_by(step) {
                        let start = [leaf[0] + x, leaf[1] + y, leaf[2] + z];
                        let (vertices, normals, split_cell) = adaptive_region_vertices(
                            cell_edges,
                            cell_dims,
                            grid_origin,
                            sample_dims,
                            values,
                            outside_value,
                            start,
                            dimension,
                            normal_to_local,
                        );
                        if vertices.is_empty() {
                            continue;
                        }
                        let region_is_blocked = blocked.contains(&(start, dimension));
                        let non_manifold = region_is_non_manifold(
                            values,
                            sample_dims,
                            grid_origin,
                            start,
                            dimension,
                            outside_value,
                            iso_level,
                        );
                        let gradients_merge = if adaptivity < 1.0e-6 {
                            false
                        } else if adaptivity >= 1.0 {
                            true
                        } else {
                            normals.iter().all(|first| {
                                normals
                                    .iter()
                                    .all(|second| 1.0_f32 - first.dot(*second) <= adaptivity)
                            })
                        };
                        if region_is_blocked || non_manifold || split_cell || !gradients_merge {
                            if dimension < 8 {
                                let parent_dimension = dimension * 2;
                                let parent_start = start.map(|axis| {
                                    axis.div_euclid(parent_dimension) * parent_dimension
                                });
                                blocked.insert((parent_start, parent_dimension));
                            }
                            continue;
                        }
                        let first = vertices[0];
                        for vertex in vertices.iter().copied().skip(1) {
                            union_vertex_regions(&mut parent, first, vertex);
                        }
                    }
                }
            }
        }
    }

    let mut root_positions = vec![DVec3::ZERO; positions.len()];
    let mut root_counts = vec![0_usize; positions.len()];
    for (index, position) in positions.iter().copied().enumerate() {
        let root = vertex_region_root(&mut parent, index);
        root_positions[root] += position;
        root_counts[root] += 1;
    }
    let mut remapped = vec![usize::MAX; positions.len()];
    let mut merged_positions = Vec::new();
    for index in 0..positions.len() {
        let root = vertex_region_root(&mut parent, index);
        if remapped[root] == usize::MAX {
            remapped[root] = merged_positions.len();
            merged_positions.push(root_positions[root] / root_counts[root] as f64);
        }
    }
    for edges in cell_edges.values_mut() {
        for vertex in edges.iter_mut().filter(|vertex| **vertex != usize::MAX) {
            let root = vertex_region_root(&mut parent, *vertex);
            *vertex = remapped[root];
        }
    }
    *positions = merged_positions;
    Ok(())
}

fn vertex_region_root(parent: &mut [usize], vertex: usize) -> usize {
    let mut root = vertex;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = vertex;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn union_vertex_regions(parent: &mut [usize], first: usize, second: usize) {
    let first = vertex_region_root(parent, first);
    let second = vertex_region_root(parent, second);
    if first != second {
        let (root, child) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        parent[child] = root;
    }
}

fn append_dual_face(
    cell_edges: &HashMap<usize, [usize; 12]>,
    faces: &mut Vec<Vec<usize>>,
    cell_dims: [usize; 3],
    ring: [((usize, usize, usize), usize); 4],
    first_is_low: bool,
    inside_is_low: bool,
    adaptive: bool,
) -> Result<()> {
    let mut quad = [usize::MAX; 4];
    for (index, ((x, y, z), edge)) in ring.into_iter().enumerate() {
        let cell = cell_linear_index(x, y, z, cell_dims);
        let Some(vertex) = cell_edges.get(&cell).and_then(|edges| edges.get(edge)) else {
            return Ok(());
        };
        if *vertex == usize::MAX {
            return Ok(());
        }
        quad[index] = *vertex;
    }
    let [a, b, c, d] = quad;
    let mut face = if a != b && a != c && a != d && b != c && b != d && c != d {
        quad.to_vec()
    } else if adaptive && a == d && a != b && a != c && b != c {
        vec![a, b, c]
    } else if adaptive && b == c && a != b && a != d && d != b {
        vec![a, b, d]
    } else if adaptive && a == b && a != c && a != d && c != d {
        vec![a, c, d]
    } else if adaptive && c == d && a != b && a != c && b != c {
        vec![a, b, c]
    } else {
        return Ok(());
    };
    if first_is_low != inside_is_low {
        face.reverse();
    }
    if faces.len() >= MAX_OUTPUT_FACES {
        return Err(resource_error());
    }
    faces.push(face);
    Ok(())
}

fn vdb_grid_to_mesh(
    grid: &crate::geom::vdb::VdbGrid,
    iso_level: f64,
    inside_is_low: bool,
    adaptivity: f32,
    output: DMat4,
    resolution: VolumeToMeshResolution,
) -> Result<Mesh> {
    if !grid.is_scalar() {
        return Ok(Mesh::default());
    }
    let Some((active_min, active_max)) = grid.active_bbox else {
        return Ok(Mesh::default());
    };
    let mut minimum = [0_i32; 3];
    let mut maximum = [0_i32; 3];
    for axis in 0..3 {
        minimum[axis] = active_min[axis]
            .checked_sub(1)
            .ok_or_else(|| invalid_volume("VDB active bounds cannot be padded"))?;
        maximum[axis] = active_max[axis]
            .checked_add(1)
            .ok_or_else(|| invalid_volume("VDB active bounds cannot be padded"))?;
    }
    let scalar_kind = grid.value_type.split('_').nth(1).unwrap_or_default();
    let iso_level = match scalar_kind {
        "bool" => 0.5,
        "float" | "double" => f64::from(iso_level as f32),
        "int32" => f64::from((iso_level as f32) as i32),
        "int64" => ((iso_level as f32) as i64) as f64,
        _ => return Ok(Mesh::default()),
    };
    let outside_value = grid
        .sample_index_scalar_linear(DVec3::new(
            f64::from(minimum[0]),
            f64::from(minimum[1]),
            f64::from(minimum[2]),
        ))
        .ok_or_else(|| invalid_volume("VDB scalar grid could not be sampled"))?;
    let mut dims = [0_u32; 3];
    for axis in 0..3 {
        let extent = i64::from(maximum[axis]) - i64::from(minimum[axis]) + 1;
        if extent < 2 || extent > i64::try_from(MAX_GRID_AXIS).unwrap_or(i64::MAX) {
            return Err(resource_error());
        }
        dims[axis] = u32::try_from(extent).map_err(|_| resource_error())?;
    }
    let sample_count = dims
        .iter()
        .try_fold(1_usize, |count, dimension| {
            count.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .filter(|count| *count <= MAX_GRID_SAMPLES)
        .ok_or_else(resource_error)?;
    let mut values = vec![0.0_f64; sample_count];
    let nx = usize::try_from(dims[0]).map_err(|_| resource_error())?;
    let ny = usize::try_from(dims[1]).map_err(|_| resource_error())?;
    let nz = usize::try_from(dims[2]).map_err(|_| resource_error())?;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let coordinate = [
                    minimum[0]
                        .checked_add(i32::try_from(x).map_err(|_| resource_error())?)
                        .ok_or_else(resource_error)?,
                    minimum[1]
                        .checked_add(i32::try_from(y).map_err(|_| resource_error())?)
                        .ok_or_else(resource_error)?,
                    minimum[2]
                        .checked_add(i32::try_from(z).map_err(|_| resource_error())?)
                        .ok_or_else(resource_error)?,
                ];
                let index = z
                    .checked_mul(ny)
                    .and_then(|row| row.checked_add(y))
                    .and_then(|row| row.checked_mul(nx))
                    .and_then(|row| row.checked_add(x))
                    .ok_or_else(resource_error)?;
                let value = grid.sample_index_scalar_linear(DVec3::new(
                    f64::from(coordinate[0]),
                    f64::from(coordinate[1]),
                    f64::from(coordinate[2]),
                ));
                let Some(value) = value else {
                    return Ok(Mesh::default());
                };
                if !value.is_finite() {
                    return Err(PotError::with_details(
                        ErrorCode::EvaluationFailed,
                        "VDB grid contains a non-finite scalar value",
                        serde_json::json!({"feature_id":"volume.openvdb_evaluation"}),
                    ));
                }
                let slot = values
                    .get_mut(index)
                    .ok_or_else(|| invalid_volume("VDB density sample index is out of range"))?;
                *slot = value;
            }
        }
    }
    let dense_grid = VolumeGrid {
        dims,
        voxel_size: 1.0,
        origin: DVec3::new(
            f64::from(minimum[0]),
            f64::from(minimum[1]),
            f64::from(minimum[2]),
        ),
        index_to_local: None,
        values: None,
        blob_f32: None,
        content_ref: None,
    };
    // The grid's voxels are absolute VDB indices, mapped to world by `index_to_world` and `output`.
    let world_output = output * grid.transform.index_to_world;
    let (layout, layout_values) = if matches!(resolution, VolumeToMeshResolution::Grid) {
        (dense_grid, values)
    } else {
        let active = (active_min.map(i64::from), active_max.map(i64::from));
        let Some(factor) = resolution_factor(world_output, active, resolution) else {
            return Ok(Mesh::default());
        };
        resample_dense_vdb(
            minimum.map(i64::from),
            dims.map(|dimension| dimension as usize),
            &values,
            outside_value,
            scalar_kind == "float",
            factor,
        )?
    };
    let mut mesh = dual_contour_grid_to_mesh(
        &layout,
        GridSamples::F64(&layout_values),
        iso_level,
        inside_is_low,
        adaptivity,
        outside_value,
        world_output,
    )?;
    for vertex in &mut mesh.vertices {
        if !vertex.co.is_finite() {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "VDB-to-mesh transform produced non-finite coordinates",
                serde_json::json!({"feature_id":"volume.openvdb_evaluation.transform"}),
            ));
        }
    }
    round_mesh_positions_to_f32(&mut mesh);
    Ok(mesh)
}

fn vdb_resource_unavailable() -> PotError {
    PotError::with_details(
        ErrorCode::DependencyMissing,
        "VDB volume data has not been resolved from its registered resource",
        serde_json::json!({"feature_id":"volume.openvdb_evaluation.resource"}),
    )
}

/// Rasterize a mesh as a Blender-compatible fog-density grid.
///
/// The dense inline representation stores integer-index samples over the source bounds plus one
/// zero-background sample outside each side. This makes the inline grid read identically through
/// [`sample_density`] and [`volume_to_mesh`] without exposing `OpenVDB`'s sparse leaf topology.
///
/// A voxel cube below `OpenVDB`'s determinant tolerance, or an empty mesh, creates no grid: the
/// result is the empty volume (see [`VolumeData`] with zero grids), as Blender's conversion does.
///
/// # Errors
///
/// Returns `InvalidArgument` for invalid conversion settings or meshes and `LimitExceeded`
/// when bounded grid or distance work is exceeded.
pub fn mesh_to_fog_volume(
    mesh: &Mesh,
    voxel_size: f64,
    interior_band_width: f64,
    density: f64,
) -> Result<VolumeData> {
    let identity: [[f32; 4]; 4] = std::array::from_fn(|column| {
        std::array::from_fn(|row| if column == row { 1.0 } else { 0.0 })
    });
    mesh_to_fog_volume_with_transform(mesh, &identity, voxel_size, interior_band_width, density)
}

/// Mesh to Volume fog for a mesh whose vertices are in object space, transformed into the volume
/// space by `mesh_to_volume` (a column-major `float4x4` of Blender's operand-to-owner transform).
///
/// Blender builds the index-space transform as `from_scale(1 / voxel) * mesh_to_volume` and
/// transforms each vertex with it in single precision (`mesh_to_volume.cc` `OpenVDBMeshAdapter`,
/// `mesh_to_density_grid_impl`), so the vertices and every distance below use the same `f32` values.
pub fn mesh_to_fog_volume_with_transform(
    mesh: &Mesh,
    mesh_to_volume: &[[f32; 4]; 4],
    voxel_size: f64,
    interior_band_width: f64,
    density: f64,
) -> Result<VolumeData> {
    if !voxel_size.is_finite()
        || !interior_band_width.is_finite()
        || interior_band_width < 0.0
        || !density.is_finite()
        || density < 0.0
    {
        return Err(invalid_volume(
            "mesh-to-volume voxel size, band width, and density must be finite and non-negative",
        ));
    }
    let voxel_size = voxel_size as f32;
    // Blender creates no grid when the voxel cube is below `OpenVDB`'s determinant tolerance
    // (`BKE_volume_voxel_size_valid`), leaving the empty volume.
    if !voxel_size.is_finite() || voxel_size <= 0.0 || f64::from(voxel_size).powi(3) < 3.0e-15 {
        return Ok(empty_fog_volume());
    }
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Ok(empty_fog_volume());
    }
    let interior_band_width = interior_band_width as f32;
    let density = density as f32;
    let (bounds_min, bounds_max, triangles) = fog_mesh_geometry(mesh, mesh_to_volume, voxel_size)?;
    // Blender's grid is the union of `OpenVDB` leaves (8 voxels per axis) that hold active voxels, so
    // the index box is rounded out to whole leaves (`BKE_volume_grid_bounds`, evalLeafBoundingBox).
    let band_minimum = bounds_min.floor() - DVec3::ONE;
    let band_maximum = bounds_max.ceil() + DVec3::ONE;
    let minimum_index = (band_minimum / 8.0).floor() * 8.0;
    let maximum_index = (band_maximum / 8.0).floor() * 8.0 + DVec3::splat(7.0);
    let dimensions_f64 = [
        maximum_index.x - minimum_index.x + 1.0,
        maximum_index.y - minimum_index.y + 1.0,
        maximum_index.z - minimum_index.z + 1.0,
    ];
    if dimensions_f64.iter().any(|dimension| {
        !dimension.is_finite() || *dimension < 1.0 || *dimension > MAX_GRID_AXIS as f64
    }) {
        return Err(resource_error());
    }
    let dims = [
        dimensions_f64[0] as usize,
        dimensions_f64[1] as usize,
        dimensions_f64[2] as usize,
    ];
    let sample_count = dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(resource_error)?;
    let cell_count = (dims[0] - 1)
        .checked_mul(dims[1] - 1)
        .and_then(|count| count.checked_mul(dims[2] - 1))
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_SAMPLES {
        return Err(resource_error());
    }
    let origin = minimum_index * f64::from(voxel_size);
    if !origin.is_finite() {
        return Err(invalid_volume("mesh-to-volume grid origin is not finite"));
    }
    let grid_dims = [
        u32::try_from(dims[0]).map_err(|_| resource_error())?,
        u32::try_from(dims[1]).map_err(|_| resource_error())?,
        u32::try_from(dims[2]).map_err(|_| resource_error())?,
    ];

    let interior_band_voxels = (interior_band_width / voxel_size).max(1.0);
    let interior_distance = interior_band_voxels * voxel_size;
    if !interior_distance.is_finite() || interior_distance <= 0.0 {
        return Err(invalid_volume("mesh-to-volume interior band is invalid"));
    }
    let (band, band_primitives) = narrow_band_squared_distances(&triangles, minimum_index, dims)?;
    let mut signed_distances = Vec::with_capacity(sample_count);
    let mut active = Vec::with_capacity(sample_count);
    let mut primitives = Vec::with_capacity(sample_count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let index_position = minimum_index + DVec3::new(x as f64, y as f64, z as f64);
                let inside = triangles.is_inside(index_position);
                let linear_index = (z * dims[1] + y) * dims[0] + x;
                let squared = band[linear_index];
                primitives.push(band_primitives[linear_index]);
                let signed = if squared.is_finite() {
                    let unsigned = squared.sqrt() * voxel_size;
                    if !unsigned.is_finite() {
                        return Err(resource_error());
                    }
                    active.push(true);
                    if inside { -unsigned } else { unsigned }
                } else {
                    // Voxels outside the narrow band keep `OpenVDB`'s background: the exterior width
                    // outside, and the interior width inside (`signedFloodFillWithValues`).
                    active.push(false);
                    if inside {
                        -interior_distance
                    } else {
                        voxel_size
                    }
                };
                signed_distances.push(signed);
            }
        }
    }
    // `OpenVDB` expands the band only when the interior band is wider than two voxels
    // (`minBandWidth = 2 * voxelSize`; the exterior band is one voxel).
    if interior_distance > 2.0 * voxel_size {
        NarrowBand {
            triangles: &triangles,
            origin: minimum_index,
            dims,
            voxel_size,
            exterior_width: voxel_size,
            interior_width: interior_distance,
        }
        .expand(&mut signed_distances, &mut active, &mut primitives);
    }
    renormalize_signed_distances(
        &mut signed_distances,
        &mut active,
        dims,
        voxel_size,
        interior_distance,
    );
    let active_bounds = active_index_bounds(&active, dims);
    let minimum_distance = signed_distances
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    if !minimum_distance.is_finite() {
        return Err(invalid_volume(
            "mesh-to-volume conversion produced no interior",
        ));
    }
    if minimum_distance >= 0.0 {
        signed_distances.fill(0.0);
        return Ok(VolumeData {
            grids: vec![VolumeGrid {
                dims: grid_dims,
                voxel_size: f64::from(voxel_size),
                origin,
                index_to_local: None,
                values: Some(signed_distances),
                blob_f32: None,
                content_ref: None,
            }],
            source: VolumeSource::Generated(VolumeGeneratedSource {
                algorithm: "mesh_to_volume_fog".to_owned(),
                active_bounds,
            }),
            decoded_vdb: None,
        });
    }
    let cutoff_distance = minimum_distance.max(-interior_distance);
    let cutoff_weight = 1.0_f32 / cutoff_distance;
    let mut values = Vec::with_capacity(sample_count);
    for signed in signed_distances {
        let mut fog = if signed < 0.0 {
            signed * cutoff_weight
        } else {
            0.0
        };
        fog *= density;
        if !fog.is_finite() {
            return Err(resource_error());
        }
        values.push(fog);
    }
    let grid = VolumeGrid {
        dims: grid_dims,
        voxel_size: f64::from(voxel_size),
        origin,
        index_to_local: None,
        values: Some(values),
        blob_f32: None,
        content_ref: None,
    };
    Ok(VolumeData {
        grids: vec![grid],
        source: VolumeSource::Generated(VolumeGeneratedSource {
            algorithm: "mesh_to_volume_fog".to_owned(),
            active_bounds,
        }),
        decoded_vdb: None,
    })
}

/// The inclusive index box of the active voxels of a dense grid with `dims`, in its linear layout
/// `(z * ny + y) * nx + x`. `None` when no voxel is active.
fn active_index_bounds(active: &[bool], dims: [usize; 3]) -> Option<([i64; 3], [i64; 3])> {
    let mut bounds: Option<([i64; 3], [i64; 3])> = None;
    for (linear, _) in active
        .iter()
        .enumerate()
        .filter(|(_, is_active)| **is_active)
    {
        let point = [
            (linear % dims[0]) as i64,
            ((linear / dims[0]) % dims[1]) as i64,
            (linear / (dims[0] * dims[1])) as i64,
        ];
        bounds = Some(match bounds {
            None => (point, point),
            Some((low, high)) => (
                std::array::from_fn(|axis| low[axis].min(point[axis])),
                std::array::from_fn(|axis| high[axis].max(point[axis])),
            ),
        });
    }
    bounds
}

/// Squared index-space distance from each voxel to the nearest triangle, and that triangle, for the
/// voxels `OpenVDB`'s voxelizer stores (`VoxelizePolygons::voxelizeTriangle` and `updateDistance`,
/// MeshToVolume.h 2156-2222).
///
/// Each triangle floods from the voxel holding its first corner through the 26-neighbourhood. A
/// flooded voxel keeps the smaller of its stored distance and the distance to this triangle; on a
/// tie it keeps the smaller triangle index. Its neighbours are flooded further only while that
/// distance is at most `0.75` in squared voxel units (`!(dist > 0.75)`), so the stored set is one
/// voxel thick around the surface. Voxels no triangle reached stay `f64::INFINITY` (`OpenVDB`'s
/// background) with `u32::MAX` as their triangle.
fn narrow_band_squared_distances(
    triangles: &TriangleBvh,
    origin: DVec3,
    dims: [usize; 3],
) -> Result<(Vec<f32>, Vec<u32>)> {
    const SPREAD_LIMIT_SQUARED: f32 = 0.75;
    let [nx, ny, _] = dims;
    let origin_coord = [origin.x as i64, origin.y as i64, origin.z as i64];
    let linear = |coord: [i64; 3]| -> Option<usize> {
        let mut local = [0_usize; 3];
        for axis in 0..3 {
            let offset = usize::try_from(coord[axis] - origin_coord[axis]).ok()?;
            if offset >= dims[axis] {
                return None;
            }
            local[axis] = offset;
        }
        Some((local[2] * ny + local[1]) * nx + local[0])
    };
    let sample_count = dims[0] * dims[1] * dims[2];
    let mut squared = vec![f32::INFINITY; sample_count];
    let mut primitive = vec![u32::MAX; sample_count];
    let mut visited = vec![0_u32; sample_count];
    for (triangle_index, triangle) in triangles.triangles.iter().enumerate() {
        let mark = u32::try_from(triangle_index + 1).map_err(|_| resource_error())?;
        let prim = u32::try_from(triangle_index).map_err(|_| resource_error())?;
        // Blender measures against `closestPointOnTriangleToPoint(prim.a, prim.c, prim.b, ...)`
        // (`updateDistance`, MeshToVolume.h), and stores the squared distance as a `float`.
        let [first, second, third] = triangle.points;
        let triangle_order = [first, third, second];
        // Keeps the nearer triangle, and the smaller triangle index on an exact tie (`updateDistance`).
        let record = |index: usize, distance: f32, squared: &mut [f32], primitive: &mut [u32]| {
            // `OpenVDB` compares exactly (`isExactlyEqual`), so the tie test is on the bit pattern.
            if distance < squared[index]
                || (distance.to_bits() == squared[index].to_bits() && prim < primitive[index])
            {
                squared[index] = distance;
                primitive[index] = prim;
            }
        };
        let corner = triangle.points[0].floor();
        let seed = [corner.x as i64, corner.y as i64, corner.z as i64];
        // `OpenVDB` stores the seed's distance and expands it whatever that distance is.
        if let Some(index) = linear(seed) {
            visited[index] = mark;
            let distance = point_triangle_distance_squared(
                DVec3::new(seed[0] as f64, seed[1] as f64, seed[2] as f64),
                triangle_order,
            ) as f32;
            record(index, distance, &mut squared, &mut primitive);
        }
        let mut pending = vec![seed];
        while let Some(current) = pending.pop() {
            for dz in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dx in -1_i64..=1 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let neighbour = [current[0] + dx, current[1] + dy, current[2] + dz];
                        let Some(index) = linear(neighbour) else {
                            continue;
                        };
                        if visited[index] == mark {
                            continue;
                        }
                        visited[index] = mark;
                        let distance = point_triangle_distance_squared(
                            DVec3::new(
                                neighbour[0] as f64,
                                neighbour[1] as f64,
                                neighbour[2] as f64,
                            ),
                            triangle_order,
                        ) as f32;
                        record(index, distance, &mut squared, &mut primitive);
                        if distance <= SPREAD_LIMIT_SQUARED {
                            pending.push(neighbour);
                        }
                    }
                }
            }
        }
    }
    Ok((squared, primitive))
}

/// `OpenVDB`'s narrow-band expansion for bands wider than two voxels (`ExpandNarrowband`,
/// `updateVoxel`, `ConstructVoxelMask`, `expandNarrowband`, MeshToVolume.h 2285-2800).
///
/// The active set grows one layer at a time from the inactive face neighbours of the active voxels
/// (the mask). A candidate takes the distance to the nearest primitive among the active voxels within
/// a manhattan limit of 5 (first layer) or 6 (second layer), `computeDistance`. It is stored when it
/// lies inside its band (`!inside && distance < exterior`, `inside && distance < interior`), with the
/// sign of its background. Its neighbours are queued only when the band has room for one more voxel
/// (`distance + voxel < band`). The second layer is the 26-neighbourhood of the updated first-layer
/// voxels within the same 8-voxel leaf, and it runs only in leaves where a first-layer voxel updated.
struct NarrowBand<'a> {
    triangles: &'a TriangleBvh,
    origin: DVec3,
    dims: [usize; 3],
    voxel_size: f32,
    exterior_width: f32,
    interior_width: f32,
}

impl NarrowBand<'_> {
    fn index(&self, coord: [i64; 3]) -> Option<usize> {
        let mut local = [0_usize; 3];
        for axis in 0..3 {
            let value = usize::try_from(coord[axis]).ok()?;
            if value >= self.dims[axis] {
                return None;
            }
            local[axis] = value;
        }
        Some((local[2] * self.dims[1] + local[1]) * self.dims[0] + local[0])
    }

    fn local(&self, index: usize) -> [usize; 3] {
        let x = index % self.dims[0];
        let y = (index / self.dims[0]) % self.dims[1];
        let z = index / (self.dims[0] * self.dims[1]);
        [x, y, z]
    }

    fn leaf(&self, index: usize) -> [usize; 3] {
        self.local(index).map(|value| value / 8)
    }

    fn neighbours_face(&self, index: usize) -> Vec<usize> {
        let [x, y, z] = self.local(index).map(|value| value as i64);
        [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        .into_iter()
        .filter_map(|(dx, dy, dz)| self.index([x + dx, y + dy, z + dz]))
        .collect()
    }

    fn neighbours_26(&self, index: usize) -> Vec<usize> {
        let [x, y, z] = self.local(index).map(|value| value as i64);
        let mut neighbours = Vec::with_capacity(26);
        for dz in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dx in -1_i64..=1 {
                    if (dx, dy, dz) != (0, 0, 0)
                        && let Some(neighbour) = self.index([x + dx, y + dy, z + dz])
                    {
                        neighbours.push(neighbour);
                    }
                }
            }
        }
        neighbours
    }

    /// `computeDistance`: the distance to the nearest primitive of the active voxels (as of the start
    /// of the round) within `limit` manhattan steps, in world units.
    fn nearest(
        &self,
        index: usize,
        limit: i64,
        active: &[bool],
        primitive: &[u32],
    ) -> Option<(f32, u32)> {
        let [x, y, z] = self.local(index).map(|value| value as i64);
        let mut primitives = BTreeSet::new();
        for dz in -limit..=limit {
            let reach_z = limit - dz.abs();
            for dy in -reach_z..=reach_z {
                let reach_y = reach_z - dy.abs();
                for dx in -reach_y..=reach_y {
                    if let Some(neighbour) = self.index([x + dx, y + dy, z + dz])
                        && active[neighbour]
                    {
                        primitives.insert(primitive[neighbour]);
                    }
                }
            }
        }
        let point = self.origin + DVec3::new(x as f64, y as f64, z as f64);
        let mut best: Option<(f64, u32)> = None;
        for prim in primitives {
            // `computeDistance` measures `closestPointOnTriangleToPoint(a, c, b, ...)`.
            let [a, b, c] = self.triangles.triangles[prim as usize].points;
            let distance = point_triangle_distance_squared(point, [a, c, b]);
            if best.is_none_or(|(current, _)| distance < current) {
                best = Some((distance, prim));
            }
        }
        best.map(|(squared, prim)| ((squared.sqrt() as f32) * self.voxel_size, prim))
    }

    /// `updateVoxel`: returns whether the voxel was updated, and whether its neighbours are queued.
    fn update(
        &self,
        index: usize,
        limit: i64,
        snapshot: &[bool],
        values: &mut [f32],
        active: &mut [bool],
        primitive: &mut [u32],
    ) -> (bool, bool) {
        let Some((distance, prim)) = self.nearest(index, limit, snapshot, primitive) else {
            return (false, false);
        };
        let inside = values[index] < 0.0;
        if !inside && distance < self.exterior_width {
            values[index] = distance;
            active[index] = true;
            primitive[index] = prim;
            return (true, distance + self.voxel_size < self.exterior_width);
        }
        if inside && distance < self.interior_width {
            values[index] = -distance;
            active[index] = true;
            primitive[index] = prim;
            return (true, distance + self.voxel_size < self.interior_width);
        }
        (false, false)
    }

    /// `expandNarrowband` over successive masks until no inactive voxel is queued.
    fn expand(&self, values: &mut [f32], active: &mut [bool], primitive: &mut [u32]) {
        let mut mask: BTreeSet<usize> = BTreeSet::new();
        for (index, &is_active) in active.iter().enumerate() {
            if is_active {
                for neighbour in self.neighbours_face(index) {
                    if !active[neighbour] {
                        mask.insert(neighbour);
                    }
                }
            }
        }
        while !mask.is_empty() {
            let snapshot = active.to_vec();
            mask.retain(|&index| !snapshot[index]);
            if mask.is_empty() {
                break;
            }
            let mut next = BTreeSet::new();
            let mut leaves: BTreeMap<[usize; 3], Vec<usize>> = BTreeMap::new();
            for &index in &mask {
                leaves.entry(self.leaf(index)).or_default().push(index);
            }
            for (leaf, members) in &leaves {
                let mut any_updated = false;
                let mut second_layer = BTreeSet::new();
                let mut updated_here = BTreeSet::new();
                for &index in members {
                    let (updated, activate) =
                        self.update(index, 5, &snapshot, values, active, primitive);
                    if updated {
                        any_updated = true;
                        updated_here.insert(index);
                    }
                    if activate {
                        for neighbour in self.neighbours_26(index) {
                            if self.leaf(neighbour) == *leaf {
                                second_layer.insert(neighbour);
                            }
                        }
                        for neighbour in self.neighbours_face(index) {
                            if self.leaf(neighbour) != *leaf {
                                next.insert(neighbour);
                            }
                        }
                    }
                }
                if any_updated {
                    for &index in &second_layer {
                        if snapshot[index] || updated_here.contains(&index) {
                            continue;
                        }
                        let (updated, activate) =
                            self.update(index, 6, &snapshot, values, active, primitive);
                        if updated && activate {
                            next.extend(self.neighbours_face(index));
                        }
                    }
                }
            }
            mask = next.into_iter().filter(|&index| !active[index]).collect();
        }
    }
}

/// `OpenVDB`'s renormalisation of the narrow band (`mesh_to_volume_internal::OffsetValues`,
/// `Renormalize`, `MinCombine`, and `InactivateValues`, MeshToVolume.h 2827-3046): every active voxel
/// moves back by 0.8 voxels, takes one Godunov step on the offset values of its neighbours, keeps the
/// smaller of the two, and is offset forward again. A voxel past either band width is set to the
/// background and made inactive, as `InactivateValues` does, so the active set is the band itself.
fn renormalize_signed_distances(
    distances: &mut [f32],
    active: &mut [bool],
    dims: [usize; 3],
    voxel_size: f32,
    interior_distance: f32,
) {
    let [nx, ny, nz] = dims;
    let offset = (0.8_f64 * f64::from(voxel_size)) as f32;
    let epsilon = 1.0e-7_f32;
    let row_stride = nx;
    let plane_stride = nx * ny;
    let mut offset_values = distances.to_vec();
    for (index, is_active) in active.iter().enumerate() {
        if *is_active {
            offset_values[index] -= offset;
        }
    }
    // A voxel on the grid boundary has no neighbours to step from, so it keeps its offset value.
    let mut renormalized = offset_values.clone();
    // Blender's `invDx = ValueType(1.0) / mVoxelSize` (`Renormalize`, MeshToVolume.h).
    let inverse_voxel = 1.0_f32 / voxel_size;
    for z in 1..nz - 1 {
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let index = (z * ny + y) * nx + x;
                if !active[index] {
                    continue;
                }
                let phi = offset_values[index];
                let up = [
                    offset_values[index + 1] - phi,
                    offset_values[index + row_stride] - phi,
                    offset_values[index + plane_stride] - phi,
                ];
                let down = [
                    phi - offset_values[index - 1],
                    phi - offset_values[index - row_stride],
                    phi - offset_values[index - plane_stride],
                ];
                let norm_squared = godunov_norm_squared(phi > 0.0, down, up);
                // `math::Sqrt(normSqGradPhi) * invDx - ValueType(1.0)`, then `S = phi0 / sqrt(phi0^2 + norm)`.
                let difference = norm_squared.sqrt() * inverse_voxel - 1.0;
                let sign = phi / (phi * phi + norm_squared).sqrt();
                renormalized[index] = phi - voxel_size * sign * difference;
            }
        }
    }
    for (index, value) in distances.iter_mut().enumerate() {
        if !active[index] {
            continue;
        }
        // `MinCombine`, then `OffsetValues(offset - epsilon)`: the offset is formed first, as Blender does.
        let shifted = *value - offset;
        let restore = offset - epsilon;
        *value = shifted.min(renormalized[index]) + restore;
        // InactivateValues clamps a band value past either width to the background. `sdfToFogVolume`
        // (`SDFVoxelsToFogVolume`) then keeps only the strictly negative values active: a value of zero or
        // more becomes an inactive zero. Blender keeps the filled interior active too, so the inside is not
        // deactivated here; it is enclosed by the surface, so the active bounds are the surface's.
        if *value < 0.0 {
            if *value <= -interior_distance {
                *value = -interior_distance;
            }
        } else {
            if *value >= voxel_size || value.is_nan() {
                *value = voxel_size;
            }
            active[index] = false;
        }
    }
}

fn godunov_norm_squared(phi_positive: bool, down: [f32; 3], up: [f32; 3]) -> f32 {
    let mut squared = 0.0;
    for axis in 0..3 {
        let derivative = if phi_positive {
            down[axis].max(0.0).max((-up[axis]).max(0.0))
        } else {
            (-down[axis]).max(0.0).max(up[axis].max(0.0))
        };
        squared += derivative * derivative;
    }
    squared
}

#[derive(Clone, Copy)]
struct DistanceTriangle {
    points: [DVec3; 3],
    minimum: DVec3,
    maximum: DVec3,
    centroid: DVec3,
}

impl DistanceTriangle {
    fn new(points: [DVec3; 3]) -> Self {
        let minimum = points[0].min(points[1]).min(points[2]);
        let maximum = points[0].max(points[1]).max(points[2]);
        Self {
            points,
            minimum,
            maximum,
            centroid: (points[0] + points[1] + points[2]) / 3.0,
        }
    }
}

struct DistanceNode {
    minimum: DVec3,
    maximum: DVec3,
    children: Option<(usize, usize)>,
    start: usize,
    end: usize,
}

struct TriangleBvh {
    triangles: Vec<DistanceTriangle>,
    indices: Vec<usize>,
    nodes: Vec<DistanceNode>,
}

impl TriangleBvh {
    fn new(triangles: Vec<DistanceTriangle>) -> Result<Self> {
        if triangles.is_empty() || triangles.len() > MAX_INPUT_TRIANGLES {
            return Err(resource_error());
        }
        let mut bvh = Self {
            indices: (0..triangles.len()).collect(),
            triangles,
            nodes: Vec::new(),
        };
        bvh.build_node(0, bvh.indices.len())?;
        Ok(bvh)
    }

    fn build_node(&mut self, start: usize, end: usize) -> Result<usize> {
        let first = *self
            .indices
            .get(start)
            .ok_or_else(|| invalid_volume("mesh distance tree has an empty range"))?;
        let mut minimum = self.triangles[first].minimum;
        let mut maximum = self.triangles[first].maximum;
        for index in &self.indices[start + 1..end] {
            minimum = minimum.min(self.triangles[*index].minimum);
            maximum = maximum.max(self.triangles[*index].maximum);
        }
        let node_index = self.nodes.len();
        self.nodes.push(DistanceNode {
            minimum,
            maximum,
            children: None,
            start,
            end,
        });
        if end - start > 8 {
            let extents = maximum - minimum;
            let axis = if extents.x >= extents.y && extents.x >= extents.z {
                0
            } else if extents.y >= extents.z {
                1
            } else {
                2
            };
            self.indices[start..end].sort_unstable_by(|first, second| {
                self.triangles[*first].centroid[axis]
                    .total_cmp(&self.triangles[*second].centroid[axis])
            });
            let middle = start + (end - start) / 2;
            let left = self.build_node(start, middle)?;
            let right = self.build_node(middle, end)?;
            self.nodes[node_index].children = Some((left, right));
        }
        Ok(node_index)
    }

    fn is_inside(&self, point: DVec3) -> bool {
        let direction = DVec3::new(1.0, 0.371_390_676_354_103_7, 0.529_173_248_214_697_2);
        self.ray_intersection_count(0, point, direction) % 2 == 1
    }

    fn ray_intersection_count(&self, node_index: usize, origin: DVec3, direction: DVec3) -> usize {
        let node = &self.nodes[node_index];
        if !ray_intersects_bounds(origin, direction, node.minimum, node.maximum) {
            return 0;
        }
        if let Some((left, right)) = node.children {
            self.ray_intersection_count(left, origin, direction)
                + self.ray_intersection_count(right, origin, direction)
        } else {
            self.indices[node.start..node.end]
                .iter()
                .filter(|triangle| {
                    ray_intersects_triangle(origin, direction, self.triangles[**triangle].points)
                })
                .count()
        }
    }
}

fn ray_intersects_bounds(origin: DVec3, direction: DVec3, minimum: DVec3, maximum: DVec3) -> bool {
    let mut near = 0.0_f64;
    let mut far = f64::INFINITY;
    for axis in 0..3 {
        let first = (minimum[axis] - origin[axis]) / direction[axis];
        let second = (maximum[axis] - origin[axis]) / direction[axis];
        near = near.max(first.min(second));
        far = far.min(first.max(second));
        if far < near {
            return false;
        }
    }
    far > 0.0
}

/// The index-space triangles of a fog mesh and their index-space bounds. Each vertex goes through
/// Blender's single-precision index transform `from_scale(1 / voxel) * mesh_to_volume`.
fn fog_mesh_geometry(
    mesh: &Mesh,
    mesh_to_volume: &[[f32; 4]; 4],
    voxel_size: f32,
) -> Result<(DVec3, DVec3, TriangleBvh)> {
    if mesh.vertices.len() > MAX_INPUT_VERTICES
        || mesh.faces.len() > MAX_INPUT_TRIANGLES
        || mesh.edges.len() > MAX_INPUT_TRIANGLES * 3
    {
        return Err(resource_error());
    }
    mesh.validate()
        .map_err(|error| invalid_mesh_input(&error))?;
    if mesh.bounds().is_none() {
        return Err(invalid_volume(
            "mesh-to-volume requires non-empty mesh geometry",
        ));
    }
    let corner_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(resource_error)?;
    if corner_count > MAX_INPUT_TRIANGLES * 3 {
        return Err(resource_error());
    }

    let triangles = mesh
        .tessellate()
        .map_err(|error| invalid_mesh_input(&error))?;
    if triangles.is_empty() || triangles.len() > MAX_INPUT_TRIANGLES {
        return Err(resource_error());
    }
    // Blender's `from_scale<float4x4>(float3(1.0f / voxel_size))` times the operand transform.
    let reciprocal = 1.0_f32 / voxel_size;
    let scale: [[f32; 4]; 4] = std::array::from_fn(|column| {
        std::array::from_fn(|row| match (column, row) {
            (0..=2, _) if column == row => reciprocal,
            (3, 3) => 1.0,
            _ => 0.0,
        })
    });
    let index_transform = crate::geom::blender_matrix::blender_mul_m4(&scale, mesh_to_volume);
    let flat: [f32; 16] = std::array::from_fn(|index| index_transform[index / 4][index % 4]);
    let vertex_positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let co = glam::Vec3::new(vertex.co.x as f32, vertex.co.y as f32, vertex.co.z as f32);
            let index = crate::geom::blender_matrix::blender_transform_point_f32(&flat, co);
            (
                vertex.id,
                DVec3::new(f64::from(index.x), f64::from(index.y), f64::from(index.z)),
            )
        })
        .collect();
    let mut surface = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        surface.push(DistanceTriangle::new([
            *vertex_positions
                .get(&triangle[0])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
            *vertex_positions
                .get(&triangle[1])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
            *vertex_positions
                .get(&triangle[2])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
        ]));
    }
    let mut bounds_min = DVec3::splat(f64::INFINITY);
    let mut bounds_max = DVec3::splat(f64::NEG_INFINITY);
    for position in vertex_positions.values() {
        bounds_min = bounds_min.min(*position);
        bounds_max = bounds_max.max(*position);
    }
    Ok((bounds_min, bounds_max, TriangleBvh::new(surface)?))
}

fn ensure_evaluable_source(volume: &VolumeData) -> Result<()> {
    if let VolumeSource::File(source) = &volume.source {
        if source.format == "vdb" {
            return if volume.decoded_vdb.is_some() {
                Ok(())
            } else {
                Err(vdb_resource_unavailable())
            };
        }
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!(
                "file-backed volume format `{}` is not supported",
                source.format
            ),
            serde_json::json!({"feature_id":"volume.file_format"}),
        ));
    }
    Ok(())
}

fn first_grid(volume: &VolumeData) -> Result<&VolumeGrid> {
    if volume.grids.len() > MAX_GRID_COUNT {
        return Err(resource_error());
    }
    volume
        .grids
        .first()
        .ok_or_else(|| invalid_volume("volume must contain at least one grid"))
}

fn validate_grid(grid: &VolumeGrid) -> Result<GridSamples<'_>> {
    let samples = validate_grid_samples(grid)?;
    let maximum_index = DVec3::new(
        f64::from(grid.dims[0] - 1),
        f64::from(grid.dims[1] - 1),
        f64::from(grid.dims[2] - 1),
    );
    grid.local_bounds(DVec3::ZERO, maximum_index)?;
    Ok(samples)
}

fn validate_grid_samples(grid: &VolumeGrid) -> Result<GridSamples<'_>> {
    if grid.dims.contains(&0) {
        return Err(invalid_volume("volume grid dimensions must be positive"));
    }
    if grid
        .dims
        .iter()
        .any(|dimension| *dimension as usize > MAX_GRID_AXIS)
    {
        return Err(resource_error());
    }
    if !grid.voxel_size.is_finite() || grid.voxel_size <= 0.0 || !grid.origin.is_finite() {
        return Err(invalid_volume(
            "volume grid transform must be finite with positive voxel_size",
        ));
    }
    let count = (grid.dims[0] as usize)
        .checked_mul(grid.dims[1] as usize)
        .and_then(|total| total.checked_mul(grid.dims[2] as usize))
        .ok_or_else(resource_error)?;
    if count > MAX_GRID_SAMPLES {
        return Err(resource_error());
    }
    let blob_size = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(resource_error)?;
    let samples = match (&grid.values, &grid.blob_f32) {
        (Some(values), None) => {
            if values.len() != count {
                return Err(invalid_volume(
                    "volume sample count does not match grid dimensions",
                ));
            }
            GridSamples::Inline(values)
        }
        (None, Some(blob)) => {
            if blob.len() != blob_size {
                return Err(invalid_volume(
                    "volume f32 blob size does not match grid dimensions",
                ));
            }
            GridSamples::Blob(blob)
        }
        (Some(_), Some(_)) => {
            return Err(invalid_volume(
                "volume grid cannot contain both inline samples and an f32 blob",
            ));
        }
        (None, None) if grid.content_ref.is_some() => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "external volume samples require decoded f32 values for evaluation",
                serde_json::json!({"feature_id":"volume.external_blob_samples"}),
            ));
        }
        (None, None) => {
            return Err(invalid_volume(
                "volume grid has no inline or blob sample values",
            ));
        }
    };
    for index in 0..count {
        let sample = samples
            .get(index)
            .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
        if !sample.is_finite() {
            return Err(invalid_volume("volume samples must be finite"));
        }
    }
    Ok(samples)
}

#[expect(
    clippy::many_single_char_names,
    reason = "point-triangle distance uses standard barycentric variables"
)]
fn point_triangle_distance_squared(point: DVec3, [a, b, c]: [DVec3; 3]) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (point - (a + v * ab)).length_squared();
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (point - (a + w * ac)).length_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (point - (b + w * (c - b))).length_squared();
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    (point - (a + ab * v + ac * w)).length_squared()
}

#[expect(
    clippy::many_single_char_names,
    reason = "ray-triangle intersection uses standard vector notation"
)]
fn ray_intersects_triangle(origin: DVec3, direction: DVec3, [a, b, c]: [DVec3; 3]) -> bool {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant.abs() <= f64::MIN_POSITIVE {
        return false;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - a;
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = offset.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    edge2.dot(q) * inverse > 0.0
}

fn invalid_volume(message: &'static str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn invalid_mesh_input(error: &MeshError) -> PotError {
    PotError::new(
        ErrorCode::InvalidArgument,
        format!("invalid mesh for volume conversion: {error}"),
    )
}

fn mesh_output_error(error: &MeshError) -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        format!("volume mesh output exceeds geometry limits: {error}"),
    )
}

fn resource_error() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "volume grid or geometry exceeds resource limits",
    )
}

/// Blender's empty volume: no grid, hence no density and no bounds.
fn empty_fog_volume() -> VolumeData {
    VolumeData {
        source: VolumeSource::Generated(VolumeGeneratedSource {
            algorithm: "mesh_to_volume_fog".to_owned(),
            active_bounds: None,
        }),
        ..VolumeData::default()
    }
}

/// Blender's `BKE_modifier_set_error` for a Volume to Mesh grid name that does not exist. It is a
/// modifier warning: the modifier yields Blender's empty mesh.
fn grid_not_found(grid_name: &str) -> PotError {
    PotError::with_details(
        ErrorCode::EvaluationFailed,
        format!("Cannot find '{grid_name}' grid"),
        serde_json::json!({
            "modifier_warning": true,
            "feature_id": "modifier.volume_to_mesh.grid_name",
        }),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::{DMat4, DVec3};
    use proptest::prelude::*;

    use crate::{
        error::ErrorCode,
        geom::{BoxParams, Mesh},
    };

    use super::{
        VolumeData, VolumeGrid, cubic_weights, mesh_to_fog_volume, sample_density,
        sample_density_checked, sample_density_with,
    };
    use crate::model::VolumeInterpolation;

    #[test]
    fn density_sampling_is_trilinear_and_zero_outside() {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [2, 2, 2],
                voxel_size: 1.0,
                origin: DVec3::ZERO,
                index_to_local: None,
                values: Some(vec![0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]),
                blob_f32: None,
                content_ref: None,
            }],
            ..VolumeData::default()
        };

        let point = DVec3::new(0.25, 0.5, 0.75);
        assert!((sample_density(point, &volume) - 3.5).abs() < 1.0e-12);
        assert_eq!(sample_density(DVec3::new(-0.01, 0.0, 0.0), &volume), 0.0);
    }

    #[test]
    fn malformed_inline_sample_count_is_rejected() {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [2, 2, 2],
                values: Some(vec![1.0]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };

        assert_eq!(
            sample_density_checked(DVec3::ZERO, &volume)
                .unwrap_err()
                .code,
            ErrorCode::InvalidArgument
        );
        assert!(sample_density(DVec3::ZERO, &volume).is_nan());
    }

    proptest! {
        #[test]
        fn trilinear_sampling_reproduces_linear_fields(
            x in 0.0_f64..=1.0,
            y in 0.0_f64..=1.0,
            z in 0.0_f64..=1.0,
        ) {
            let volume = VolumeData {
                grids: vec![VolumeGrid {
                    dims: [2, 2, 2],
                    voxel_size: 1.0,
                    origin: DVec3::ZERO,
                    index_to_local: None,
                    values: Some(vec![0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]),
                    blob_f32: None,
                    content_ref: None,
                }],
                ..VolumeData::default()
            };
            let actual = sample_density(DVec3::new(x, y, z), &volume);
            let expected = x + 2.0 * y + 3.0 * z;
            prop_assert!((actual - expected).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn coarse_mesh_to_fog_returns_a_zero_density_grid() {
        let mut mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(0.1),
        })
        .unwrap();
        for vertex in &mut mesh.vertices {
            vertex.co += DVec3::splat(0.5);
        }

        let volume = mesh_to_fog_volume(&mesh, 1.0, 0.2, 1.0).unwrap();
        let grid = volume.grids.first().unwrap();
        assert!(grid.values.as_ref().is_some_and(|values| {
            !values.is_empty() && values.iter().all(|value| value.abs() <= f32::EPSILON)
        }));
    }

    #[test]
    fn affine_grid_density_sampling_maps_volume_local_to_index() {
        let index_to_local = DMat4::from_cols_array(&[
            0.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.5, 0.25, 1.0, 0.0, 2.0, -1.0, 0.25, 1.0,
        ]);
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [2, 2, 2],
                voxel_size: 3.0,
                origin: index_to_local.transform_point3(DVec3::ZERO),
                index_to_local: Some(index_to_local.to_cols_array()),
                values: Some(vec![0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]),
                blob_f32: None,
                content_ref: None,
            }],
            ..VolumeData::default()
        };
        let point = DVec3::new(0.25, 0.5, 0.75);
        let local = index_to_local.transform_point3(point);
        assert!((sample_density(local, &volume) - 3.5).abs() < 1.0e-12);
    }

    /// The uniform cubic B-spline `B3(u)` from its truncated-power definition, `(1/6) Σ_j (-1)^j C(4, j) (u + 2 - j)_+^3`,
    /// which is not the Horner form `cubic_weights` evaluates.
    fn truncated_power_cubic(u: f64) -> f64 {
        let positive = |x: f64| x.max(0.0).powi(3);
        (positive(u + 2.0) - 4.0 * positive(u + 1.0) + 6.0 * positive(u) - 4.0 * positive(u - 1.0)
            + positive(u - 2.0))
            / 6.0
    }

    #[test]
    fn cubic_weights_are_the_truncated_power_b_spline_at_each_offset() {
        for t in [0.0, 0.125, 0.25, 0.5, 0.75, 0.9375] {
            let weights = cubic_weights(t);
            // The sample at offset k from the floor gets B3(t - k), for k = -1, 0, 1, 2.
            for (index, offset) in [-1.0, 0.0, 1.0, 2.0].into_iter().enumerate() {
                let expected = truncated_power_cubic(t - offset);
                assert!(
                    (weights[index] - expected).abs() <= 1.0e-12,
                    "t {t}, offset {offset}: {} vs {expected}",
                    weights[index]
                );
            }
        }
    }

    #[test]
    fn cubic_weights_sum_to_one_and_put_the_centroid_at_the_fraction() {
        for t in [0.0, 0.125, 0.25, 0.5, 0.75, 0.9375] {
            let weights = cubic_weights(t);
            let total: f64 = weights.iter().sum();
            let centroid: f64 = weights
                .iter()
                .zip([-1.0, 0.0, 1.0, 2.0])
                .map(|(weight, offset)| weight * offset)
                .sum();
            assert!((total - 1.0).abs() <= 1.0e-12, "t {t}: sum {total}");
            assert!(
                (centroid - t).abs() <= 1.0e-12,
                "t {t}: centroid {centroid}"
            );
        }
    }

    /// A cubic B-spline with the grid values as its coefficients reproduces any linear function of the index
    /// coordinate (its centroid is the coordinate), so the cubic sample of `1 + 0.1 x + 0.2 y + 0.05 z` is that function.
    #[test]
    fn cubic_sampling_reproduces_a_linear_field_in_the_interior() {
        let dims = [8_u32, 8, 8];
        let mut values = Vec::new();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    values.push(
                        (1.0 + 0.1 * f64::from(x) + 0.2 * f64::from(y) + 0.05 * f64::from(z))
                            as f32,
                    );
                }
            }
        }
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims,
                voxel_size: 1.0,
                origin: DVec3::ZERO,
                index_to_local: None,
                values: Some(values),
                blob_f32: None,
                content_ref: None,
            }],
            ..VolumeData::default()
        };
        for coordinate in [
            DVec3::new(3.25, 4.5, 2.75),
            DVec3::new(2.5, 2.5, 5.9),
            DVec3::new(4.0, 3.0, 3.0),
        ] {
            let expected = 1.0 + 0.1 * coordinate.x + 0.2 * coordinate.y + 0.05 * coordinate.z;
            let actual = sample_density_with(coordinate, &volume, VolumeInterpolation::Cubic);
            // f32 storage of the samples bounds the error at about 1e-7 relative.
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "{coordinate:?}: {actual} vs {expected}"
            );
        }
    }

    /// Outside the grid's neighbours the background is zero: at x = 0.5 the node at x = -1 has weight
    /// B3(1.5) = 1/48, so a constant grid of ones samples to 1 - 1/48 there and to 1 in the interior.
    #[test]
    fn cubic_sampling_of_a_constant_grid_loses_the_outside_neighbour_weight() {
        let dims = [8_u32, 8, 8];
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims,
                voxel_size: 1.0,
                origin: DVec3::ZERO,
                index_to_local: None,
                values: Some(vec![1.0; 512]),
                blob_f32: None,
                content_ref: None,
            }],
            ..VolumeData::default()
        };
        let edge = sample_density_with(
            DVec3::new(0.5, 3.5, 3.5),
            &volume,
            VolumeInterpolation::Cubic,
        );
        let interior = sample_density_with(
            DVec3::new(3.5, 3.5, 3.5),
            &volume,
            VolumeInterpolation::Cubic,
        );
        assert!((edge - (1.0 - 1.0 / 48.0)).abs() <= 1.0e-6, "edge {edge}");
        assert!((interior - 1.0).abs() <= 1.0e-6, "interior {interior}");
    }

    /// Past the last node the background is zero, and linear reads one node beyond the cell while cubic reads two. A
    /// constant grid of ones therefore samples to 1/2 at x = 7.5 for both (cubic's offsets 6 and 7 carry 1/48 + 23/48),
    /// and to 5/6 for cubic at the last node (1/6 + 2/3), while linear at the last node is exactly 1.
    #[test]
    fn inline_sampling_reads_zero_background_past_the_last_node() {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [8, 8, 8],
                voxel_size: 1.0,
                origin: DVec3::ZERO,
                index_to_local: None,
                values: Some(vec![1.0; 512]),
                blob_f32: None,
                content_ref: None,
            }],
            ..VolumeData::default()
        };
        let at = |x: f64, interpolation: VolumeInterpolation| {
            sample_density_with(DVec3::new(x, 3.5, 3.5), &volume, interpolation)
        };
        assert!((at(7.5, VolumeInterpolation::Linear) - 0.5).abs() <= 1.0e-12);
        assert!((at(7.5, VolumeInterpolation::Cubic) - 0.5).abs() <= 1.0e-12);
        assert!((at(7.0, VolumeInterpolation::Cubic) - 5.0 / 6.0).abs() <= 1.0e-12);
        assert!((at(7.0, VolumeInterpolation::Linear) - 1.0).abs() <= 1.0e-12);
    }
}
