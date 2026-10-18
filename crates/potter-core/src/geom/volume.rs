//! Inline scalar volume grids and mesh conversion.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use glam::{DMat4, DVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, MeshError},
};

use super::edge_key;

const MAX_GRID_SAMPLES: usize = 1_000_000;
const MAX_GRID_AXIS: usize = 512;
const MAX_GRID_CELLS: usize = 250_000;
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
    /// Legacy uniform voxel spacing; affine grids store their maximum voxel side here.
    pub voxel_size: f64,
    /// Volume-local position of sample `[0, 0, 0]` when `index_to_local` is absent.
    pub origin: DVec3,
    /// Optional column-major affine transform from grid indices to volume-local coordinates.
    /// When absent, the legacy origin-plus-uniform-voxel-size mapping applies.
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
            origin: DVec3::ZERO,
            voxel_size: 1.0,
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
/// Positions are volume-local coordinates. Inline grids use `index_to_local` when present,
/// otherwise the stored origin and voxel size; decoded VDB grids apply their index-to-world transform.
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
    sample_density_checked_with_inverse(position, volume, None)
}

pub(crate) fn sample_density_checked_with_inverse(
    position: DVec3,
    volume: &VolumeData,
    local_to_index: Option<DMat4>,
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
        return Ok(grid.sample_world(position));
    }
    let grid = first_grid(volume)?;
    let values = if grid.index_to_local.is_some() && local_to_index.is_some() {
        validate_grid_samples(grid)?
    } else {
        validate_grid(grid)?
    };

    let coordinate = if grid.index_to_local.is_some() {
        let local_to_index = match local_to_index {
            Some(matrix) => matrix,
            None => grid.checked_index_to_local_matrix()?.inverse(),
        };
        if !local_to_index.is_finite() {
            return Err(invalid_volume(
                "volume grid index transform is not invertible",
            ));
        }
        local_to_index.transform_point3(position)
    } else {
        (position - grid.origin) / grid.voxel_size
    };

    if !coordinate.is_finite() || coordinate.cmplt(DVec3::ZERO).any() {
        return Ok(0.0);
    }
    let maximum = DVec3::new(
        f64::from(grid.dims[0] - 1),
        f64::from(grid.dims[1] - 1),
        f64::from(grid.dims[2] - 1),
    );
    if coordinate.cmpgt(maximum).any() {
        return Ok(0.0);
    }

    let lower = [
        coordinate.x.floor() as usize,
        coordinate.y.floor() as usize,
        coordinate.z.floor() as usize,
    ];
    let upper = [
        (lower[0] + 1).min(grid.dims[0] as usize - 1),
        (lower[1] + 1).min(grid.dims[1] as usize - 1),
        (lower[2] + 1).min(grid.dims[2] as usize - 1),
    ];
    let fraction = [
        coordinate.x - lower[0] as f64,
        coordinate.y - lower[1] as f64,
        coordinate.z - lower[2] as f64,
    ];

    let mut result = 0.0;
    for z_corner in 0..2 {
        for y_corner in 0..2 {
            for x_corner in 0..2 {
                let x = if x_corner == 0 { lower[0] } else { upper[0] };
                let y = if y_corner == 0 { lower[1] } else { upper[1] };
                let z = if z_corner == 0 { lower[2] } else { upper[2] };
                let wx = if x_corner == 0 {
                    1.0 - fraction[0]
                } else {
                    fraction[0]
                };
                let wy = if y_corner == 0 {
                    1.0 - fraction[1]
                } else {
                    fraction[1]
                };
                let wz = if z_corner == 0 {
                    1.0 - fraction[2]
                } else {
                    fraction[2]
                };
                let index = z
                    .checked_mul(grid.dims[1] as usize)
                    .and_then(|row| row.checked_add(y))
                    .and_then(|row| row.checked_mul(grid.dims[0] as usize))
                    .and_then(|row| row.checked_add(x))
                    .ok_or_else(|| invalid_volume("volume sample index overflow"))?;
                let sample = values
                    .get(index)
                    .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
                result += sample * wx * wy * wz;
            }
        }
    }
    Ok(result)
}

/// Extract an iso-surface of the first scalar grid using OpenVDB-style dual contouring.
///
/// Each intersected cell contributes a feature point per connected contour component. Intersected
/// grid edges connect those points, while adaptivity merges regions into quads or triangles.
///
/// # Errors
///
/// Returns a typed error for unsupported file formats, unresolved VDB resources, invalid grid
/// data or iso-levels, resource-limit violations, and fields that do not cross the iso-level.
pub fn volume_to_mesh(volume: &VolumeData, iso_level: f64) -> Result<Mesh> {
    volume_to_mesh_for_grid(volume, iso_level, None, 0.0)
}

/// Extract a named volume grid with OpenVDB's adaptive merging threshold (`0.0..=1.0`).
pub fn volume_to_mesh_named_adaptive(
    volume: &VolumeData,
    iso_level: f64,
    grid_name: &str,
    adaptivity: f64,
) -> Result<Mesh> {
    volume_to_mesh_for_grid(volume, iso_level, Some(grid_name), adaptivity)
}

/// Extract an iso-surface from the grid named by a Blender Volume to Mesh modifier.
///
/// Generated inline volumes use the implicit Blender grid name `"density"`; file-backed VDB
/// grids are selected by their stored grid names.
///
/// # Errors
///
/// Returns a typed error for a missing grid, unsupported source, invalid grid data or iso-level,
/// resource-limit violations, and fields that do not cross the requested iso-level.
pub fn volume_to_mesh_named(volume: &VolumeData, iso_level: f64, grid_name: &str) -> Result<Mesh> {
    volume_to_mesh_named_adaptive(volume, iso_level, grid_name, 0.0)
}

fn volume_to_mesh_for_grid(
    volume: &VolumeData,
    iso_level: f64,
    grid_name: Option<&str>,
    adaptivity: f64,
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
                .ok_or_else(|| invalid_volume("volume does not contain the requested grid"))?
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
        return vdb_grid_to_mesh(grid, iso_level, inside_is_low, adaptivity);
    }
    if grid_name.is_some_and(|name| name != "density") {
        return Err(invalid_volume("volume does not contain the requested grid"));
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
        )?
    } else {
        dual_contour_grid_to_mesh(grid, values, iso_level, inside_is_low, adaptivity, 0.0)?
    };
    if boundary_crosses {
        let maximum_index = DVec3::new(
            f64::from(grid.dims[0] - 1),
            f64::from(grid.dims[1] - 1),
            f64::from(grid.dims[2] - 1),
        );
        let index_to_local = grid.checked_index_to_local_matrix()?;
        let local_to_index = index_to_local.inverse();
        if !local_to_index.is_finite() {
            return Err(invalid_volume(
                "volume grid index transform is not invertible",
            ));
        }
        grid.local_bounds(DVec3::ZERO, maximum_index)?;
        for vertex in &mut mesh.vertices {
            let index = local_to_index.transform_point3(vertex.co);
            if !index.is_finite() {
                return Err(invalid_volume("volume contour index is not finite"));
            }
            vertex.co = index_to_local.transform_point3(index.clamp(DVec3::ZERO, maximum_index));
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
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_CELLS {
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
// OpenVDB 13.0.0 VolumeToMesh::sAdaptable (Apache-2.0 source table).
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
) -> Result<Mesh> {
    let nx = usize::try_from(grid.dims[0]).map_err(|_| resource_error())?;
    let ny = usize::try_from(grid.dims[1]).map_err(|_| resource_error())?;
    let nz = usize::try_from(grid.dims[2]).map_err(|_| resource_error())?;
    let cell_dims = [nx - 1, ny - 1, nz - 1];
    let cell_count = cell_dims[0]
        .checked_mul(cell_dims[1])
        .and_then(|count| count.checked_mul(cell_dims[2]))
        .ok_or_else(resource_error)?;
    if cell_count > MAX_GRID_CELLS {
        return Err(resource_error());
    }
    let index_to_local = grid.checked_index_to_local_matrix()?;

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
                            let pairs = if f64::from(cube_values[face_corners[0]]) < iso_level {
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
                    let position = index_to_local.transform_point3(index_position);
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

    if positions.is_empty() {
        return Err(invalid_volume(
            "volume scalar field does not cross the requested iso-level",
        ));
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
                if (f64::from(first) < iso_level) == (f64::from(second) < iso_level) {
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
                    f64::from(first) < iso_level,
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
                if (f64::from(first) < iso_level) == (f64::from(second) < iso_level) {
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
                    f64::from(first) < iso_level,
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
                if (f64::from(first) < iso_level) == (f64::from(second) < iso_level) {
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
                    f64::from(first) < iso_level,
                    inside_is_low,
                    adaptive,
                )?;
            }
        }
    }
    if faces.is_empty() {
        return Err(invalid_volume(
            "volume scalar field does not cross the requested iso-level",
        ));
    }
    if index_to_local.determinant() < 0.0 {
        for face in &mut faces {
            face.reverse();
        }
    }
    Mesh::from_positions_and_faces(positions, faces).map_err(|error| mesh_output_error(&error))
}

fn round_mesh_positions_to_f32(mesh: &mut Mesh) {
    for vertex in &mut mesh.vertices {
        vertex.co = vertex.co.map(|component| f64::from(component as f32));
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
            mask | ((if signs[*corner] { 1 } else { 0 }) << bit)
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
    for (cell, edges) in cell_edges {
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
        "float" => f64::from(iso_level as f32),
        "double" => iso_level,
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
    let mut mesh = dual_contour_grid_to_mesh(
        &dense_grid,
        GridSamples::F64(&values),
        iso_level,
        inside_is_low,
        adaptivity,
        outside_value,
    )?;
    for vertex in &mut mesh.vertices {
        vertex.co = grid.transform.index_to_world.transform_point3(vertex.co);
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

/// Sample a closed mesh's signed-distance approximation onto a regular grid.
///
/// The grid includes at least one exterior sample on every side; `padding` adds
/// further voxels beyond that minimum.
///
/// # Errors
///
/// Returns `InvalidArgument` for invalid/open meshes or a non-positive/non-finite
/// voxel size, and `LimitExceeded` when bounded grid or distance work is exceeded.
pub fn mesh_to_volume(mesh: &Mesh, voxel_size: f64, padding: u32) -> Result<VolumeData> {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(invalid_volume("voxel_size must be finite and positive"));
    }
    if mesh.vertices.len() > MAX_INPUT_VERTICES
        || mesh.faces.len() > MAX_INPUT_TRIANGLES
        || mesh.edges.len() > MAX_INPUT_TRIANGLES * 3
    {
        return Err(resource_error());
    }
    let input_corner_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(resource_error)?;
    if input_corner_count > MAX_INPUT_TRIANGLES * 3 {
        return Err(resource_error());
    }
    mesh.validate()
        .map_err(|error| invalid_mesh_input(&error))?;
    let bounds = mesh
        .bounds()
        .ok_or_else(|| invalid_volume("mesh_to_volume requires a non-empty closed mesh"))?;
    if mesh.faces.is_empty() {
        return Err(invalid_volume(
            "mesh_to_volume requires a non-empty closed mesh",
        ));
    }

    let mut edge_uses = HashMap::new();
    let mut surface_vertices = HashSet::with_capacity(mesh.vertices.len());
    let mut triangle_count = 0_usize;
    for face in &mesh.faces {
        triangle_count = triangle_count
            .checked_add(face.vertices.len() - 2)
            .ok_or_else(resource_error)?;
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            surface_vertices.insert(first);
            let key = edge_key(first, second);
            let uses = edge_uses.entry(key).or_insert(0_u8);
            *uses = uses.saturating_add(1);
        }
    }
    if triangle_count == 0 || triangle_count > MAX_INPUT_TRIANGLES {
        return Err(resource_error());
    }
    if surface_vertices.len() != mesh.vertices.len() || edge_uses.values().any(|count| *count != 2)
    {
        return Err(invalid_volume(
            "mesh_to_volume requires a closed two-manifold mesh",
        ));
    }

    let triangles = mesh
        .triangulate()
        .map_err(|error| invalid_mesh_input(&error))?;
    let vertex_positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut surface = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let points = [
            *vertex_positions
                .get(&triangle[0])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
            *vertex_positions
                .get(&triangle[1])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
            *vertex_positions
                .get(&triangle[2])
                .ok_or_else(|| invalid_volume("mesh triangle references a missing vertex"))?,
        ];
        surface.push(DistanceTriangle::new(points));
    }
    let triangles = TriangleBvh::new(surface)?;

    let effective_padding = padding.max(1) as usize;
    let size = bounds.size();
    if !size.is_finite() || size.cmplt(DVec3::ZERO).any() {
        return Err(invalid_volume("mesh bounds are not finite"));
    }
    let dims = [
        grid_axis(size.x, voxel_size, effective_padding)?,
        grid_axis(size.y, voxel_size, effective_padding)?,
        grid_axis(size.z, voxel_size, effective_padding)?,
    ];
    let sample_count = dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(resource_error)?;
    let cell_count = (dims[0] - 1)
        .checked_mul(dims[1] - 1)
        .and_then(|count| count.checked_mul(dims[2] - 1))
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_CELLS {
        return Err(resource_error());
    }

    let origin = bounds.min - DVec3::splat(voxel_size * effective_padding as f64);
    let last_position = origin
        + voxel_size
            * DVec3::new(
                (dims[0] - 1) as f64,
                (dims[1] - 1) as f64,
                (dims[2] - 1) as f64,
            );
    if !origin.is_finite() || !last_position.is_finite() {
        return Err(invalid_volume(
            "voxel grid transform produces non-finite positions",
        ));
    }

    let mut values = Vec::with_capacity(sample_count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let point = origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64);
                let distance = triangles.signed_distance(point);
                let value = distance as f32;
                if !distance.is_finite() || !value.is_finite() {
                    return Err(resource_error());
                }
                values.push(value);
            }
        }
    }
    Ok(VolumeData {
        grids: vec![VolumeGrid {
            dims: [
                u32::try_from(dims[0]).map_err(|_| resource_error())?,
                u32::try_from(dims[1]).map_err(|_| resource_error())?,
                u32::try_from(dims[2]).map_err(|_| resource_error())?,
            ],
            voxel_size,
            origin,
            index_to_local: None,
            values: Some(values),
            blob_f32: None,
            content_ref: None,
        }],
        source: VolumeSource::Generated(VolumeGeneratedSource {
            algorithm: "signed_distance".to_owned(),
        }),
        decoded_vdb: None,
    })
}

/// Rasterize a mesh as a Blender-compatible fog-density grid.
///
/// The dense inline representation stores integer-index samples over the source bounds plus one
/// zero-background sample outside each side. This makes the inline grid read identically through
/// [`sample_density`] and [`volume_to_mesh`] without exposing OpenVDB's sparse leaf topology.
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
    use_fill_volume: bool,
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
    if voxel_size < 1.0e-5 {
        return Ok(VolumeData {
            source: VolumeSource::Generated(VolumeGeneratedSource {
                algorithm: "mesh_to_volume_fog".to_owned(),
            }),
            ..VolumeData::default()
        });
    }
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Ok(VolumeData {
            source: VolumeSource::Generated(VolumeGeneratedSource {
                algorithm: "mesh_to_volume_fog".to_owned(),
            }),
            ..VolumeData::default()
        });
    }
    if !voxel_size.is_finite() || voxel_size <= 0.0 || f64::from(voxel_size).powi(3) < 3.0e-15 {
        return Ok(VolumeData {
            source: VolumeSource::Generated(VolumeGeneratedSource {
                algorithm: "mesh_to_volume_fog".to_owned(),
            }),
            ..VolumeData::default()
        });
    }
    let interior_band_width = interior_band_width as f32;
    let density = density as f32;
    let (bounds_min, bounds_max, triangles) = fog_mesh_geometry(mesh, voxel_size)?;
    let voxel_size_f64 = f64::from(voxel_size);
    let minimum_index = (bounds_min / voxel_size_f64).floor() - DVec3::ONE;
    let maximum_index = (bounds_max / voxel_size_f64).ceil() + DVec3::ONE;
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
    if sample_count > MAX_GRID_SAMPLES || cell_count > MAX_GRID_CELLS {
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
    let mut signed_distances = Vec::with_capacity(sample_count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let index_position = minimum_index + DVec3::new(x as f64, y as f64, z as f64);
                let nearest_squared = triangles.nearest_squared(index_position);
                let unsigned = (nearest_squared as f32).sqrt() * voxel_size;
                if !unsigned.is_finite() {
                    return Err(resource_error());
                }
                let inside = triangles.is_inside(index_position);
                let signed = if inside {
                    -unsigned.min(interior_distance)
                } else {
                    unsigned.min(voxel_size)
                };
                signed_distances.push(signed);
            }
        }
    }
    renormalize_signed_distances(&mut signed_distances, dims, voxel_size, interior_distance);
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
            }),
            decoded_vdb: None,
        });
    }
    let cutoff_distance = minimum_distance.max(-interior_distance);
    let cutoff_weight = 1.0_f32 / cutoff_distance;
    let mut values = Vec::with_capacity(sample_count);
    for signed in signed_distances {
        let mut fog = if use_fill_volume {
            if signed < 0.0 {
                signed * cutoff_weight
            } else {
                0.0
            }
        } else {
            ((interior_distance - signed.abs()) / interior_distance).clamp(0.0, 1.0)
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
        }),
        decoded_vdb: None,
    })
}

fn renormalize_signed_distances(
    distances: &mut [f32],
    dims: [usize; 3],
    voxel_size: f32,
    interior_distance: f32,
) {
    let [nx, ny, nz] = dims;
    let offset = (0.8_f64 * f64::from(voxel_size)) as f32;
    let epsilon = 1.0e-7_f32;
    let row_stride = nx;
    let plane_stride = nx * ny;
    let mut active = vec![false; distances.len()];
    let mut offset_values = distances.to_vec();
    for (index, value) in distances.iter().copied().enumerate() {
        let is_active = if value < 0.0 {
            -value < interior_distance
        } else {
            value < voxel_size
        };
        active[index] = is_active;
        if is_active {
            offset_values[index] -= offset;
        }
    }
    let mut renormalized = vec![0.0_f32; distances.len()];
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
                let norm = norm_squared.sqrt();
                let difference = norm / voxel_size - 1.0;
                let sign = phi / (phi * phi + norm_squared).sqrt();
                renormalized[index] = phi - voxel_size * sign * difference;
            }
        }
    }
    for (index, value) in distances.iter_mut().enumerate() {
        if !active[index] {
            continue;
        }
        let shifted = *value - offset;
        *value = shifted.min(renormalized[index]) + offset - epsilon;
        if *value < 0.0 {
            if !(*value > -interior_distance) {
                *value = -interior_distance;
            }
        } else if !(*value < voxel_size) {
            *value = voxel_size;
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

    fn nearest_squared(&self, point: DVec3) -> f64 {
        self.nearest_squared_in_node(0, point, f64::INFINITY)
    }

    fn nearest_squared_in_node(&self, node_index: usize, point: DVec3, best: f64) -> f64 {
        let node = &self.nodes[node_index];
        if distance_to_bounds_squared(point, node.minimum, node.maximum) >= best {
            return best;
        }
        if let Some((left, right)) = node.children {
            let left_node = &self.nodes[left];
            let right_node = &self.nodes[right];
            let (first, second) =
                if distance_to_bounds_squared(point, left_node.minimum, left_node.maximum)
                    <= distance_to_bounds_squared(point, right_node.minimum, right_node.maximum)
                {
                    (left, right)
                } else {
                    (right, left)
                };
            let best = self.nearest_squared_in_node(first, point, best);
            self.nearest_squared_in_node(second, point, best)
        } else {
            self.indices[node.start..node.end]
                .iter()
                .fold(best, |best, triangle| {
                    best.min(point_triangle_distance_squared(
                        point,
                        self.triangles[*triangle].points,
                    ))
                })
        }
    }

    fn signed_distance(&self, point: DVec3) -> f64 {
        let distance = self.nearest_squared(point).sqrt();
        if self.is_inside(point) {
            -distance
        } else {
            distance
        }
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

fn distance_to_bounds_squared(point: DVec3, minimum: DVec3, maximum: DVec3) -> f64 {
    let mut distance = 0.0;
    for axis in 0..3 {
        let delta = if point[axis] < minimum[axis] {
            minimum[axis] - point[axis]
        } else if point[axis] > maximum[axis] {
            point[axis] - maximum[axis]
        } else {
            0.0
        };
        distance += delta * delta;
    }
    distance
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

fn fog_mesh_geometry(mesh: &Mesh, voxel_size: f32) -> Result<(DVec3, DVec3, TriangleBvh)> {
    if mesh.vertices.len() > MAX_INPUT_VERTICES
        || mesh.faces.len() > MAX_INPUT_TRIANGLES
        || mesh.edges.len() > MAX_INPUT_TRIANGLES * 3
    {
        return Err(resource_error());
    }
    mesh.validate()
        .map_err(|error| invalid_mesh_input(&error))?;
    let bounds = mesh
        .bounds()
        .ok_or_else(|| invalid_volume("mesh-to-volume requires non-empty mesh geometry"))?;
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
        .triangulate()
        .map_err(|error| invalid_mesh_input(&error))?;
    if triangles.is_empty() || triangles.len() > MAX_INPUT_TRIANGLES {
        return Err(resource_error());
    }
    let reciprocal = 1.0_f32 / voxel_size;
    let vertex_positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let index = DVec3::new(
                f64::from(vertex.co.x as f32 * reciprocal),
                f64::from(vertex.co.y as f32 * reciprocal),
                f64::from(vertex.co.z as f32 * reciprocal),
            );
            (vertex.id, index)
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
    let bounds_min = bounds.min.map(|component| f64::from(component as f32));
    let bounds_max = bounds.max.map(|component| f64::from(component as f32));
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

fn grid_axis(extent: f64, voxel_size: f64, padding: usize) -> Result<usize> {
    let axis = (extent / voxel_size).ceil() + 1.0 + 2.0 * padding as f64;
    if !axis.is_finite() || axis < 3.0 || axis > MAX_GRID_AXIS as f64 {
        return Err(resource_error());
    }
    Ok(axis as usize)
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
        VolumeData, VolumeGrid, mesh_to_fog_volume, mesh_to_volume, sample_density,
        sample_density_checked, volume_to_mesh,
    };

    #[test]
    fn coarse_mesh_to_fog_returns_a_zero_density_grid() {
        let mut mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(0.1),
        })
        .unwrap();
        for vertex in &mut mesh.vertices {
            vertex.co += DVec3::splat(0.5);
        }

        let volume = mesh_to_fog_volume(&mesh, 1.0, 0.2, 1.0, true).unwrap();
        let grid = volume.grids.first().unwrap();
        assert!(grid.values.as_ref().is_some_and(|values| {
            !values.is_empty() && values.iter().all(|value| value.abs() <= f32::EPSILON)
        }));
    }

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

    #[test]
    fn mesh_volume_mesh_preserves_bounds_within_one_voxel() {
        let source = Mesh::box_mesh(BoxParams {
            size: DVec3::new(1.7, 2.1, 1.3),
        })
        .unwrap();
        let voxel_size = 0.3;
        let volume = mesh_to_volume(&source, voxel_size, 2).unwrap();
        let result = volume_to_mesh(&volume, 0.0).unwrap();
        let source_bounds = source.bounds().unwrap();
        let result_bounds = result.bounds().unwrap();
        for axis in 0..3 {
            assert!((source_bounds.min[axis] - result_bounds.min[axis]).abs() <= voxel_size);
            assert!((source_bounds.max[axis] - result_bounds.max[axis]).abs() <= voxel_size);
        }
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
}
