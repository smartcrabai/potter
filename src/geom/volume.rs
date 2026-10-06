//! Inline scalar volume grids and mesh conversion.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use glam::DVec3;
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
const MAX_DISTANCE_TESTS: usize = 100_000_000;
const MAX_GRID_COUNT: usize = 64;
const MAX_OUTPUT_FACES: usize = 500_000;

const TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

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
    /// Uniform spacing between neighboring samples.
    pub voxel_size: f64,
    /// Volume-local position of sample `[0, 0, 0]`.
    pub origin: DVec3,
    /// Optional inline f32 samples, laid out with x as the fastest-changing axis.
    pub values: Option<Vec<f32>>,
    /// Optional little-endian f32 byte blob with the same sample layout.
    pub blob_f32: Option<Vec<u8>>,
    /// Optional stable reference to a backing blob. Inline `values` or `blob_f32`,
    /// when present, are the decoded samples for this reference.
    pub content_ref: Option<String>,
}

impl Default for VolumeGrid {
    fn default() -> Self {
        Self {
            dims: [1, 1, 1],
            voxel_size: 1.0,
            origin: DVec3::ZERO,
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
}

impl GridSamples<'_> {
    fn get(self, index: usize) -> Option<f32> {
        match self {
            Self::Inline(values) => values.get(index).copied(),
            Self::Blob(bytes) => {
                let start = index.checked_mul(std::mem::size_of::<f32>())?;
                let end = start.checked_add(std::mem::size_of::<f32>())?;
                let sample: [u8; 4] = bytes.get(start..end)?.try_into().ok()?;
                Some(f32::from_le_bytes(sample))
            }
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
/// Positions are volume-local world coordinates. Inline grids use their stored origin and
/// voxel size; decoded VDB grids apply their own index-to-world transform. Since this
/// convenience API cannot return an error, malformed data or unresolved resources return `NaN`;
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
    let values = validate_grid(grid)?;

    let coordinate = (position - grid.origin) / grid.voxel_size;
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
                result += f64::from(sample) * wx * wy * wz;
            }
        }
    }
    Ok(result)
}

/// Extract the zero/iso surface of the first scalar grid using marching tetrahedra.
///
/// # Errors
///
/// Returns a typed error for unsupported file formats, unresolved VDB resources, invalid grid
/// data or iso-levels, resource-limit violations, and fields that do not cross the iso-level.
pub fn volume_to_mesh(volume: &VolumeData, iso_level: f64) -> Result<Mesh> {
    ensure_evaluable_source(volume)?;
    if !iso_level.is_finite() {
        return Err(invalid_volume("iso_level must be finite"));
    }
    if matches!(&volume.source, VolumeSource::File(_)) {
        let decoded = volume
            .decoded_vdb
            .as_ref()
            .ok_or_else(vdb_resource_unavailable)?;
        let grid = decoded
            .density_grid()
            .ok_or_else(|| invalid_volume("VDB archive contains no density grid"))?;
        return vdb_grid_to_mesh(grid, iso_level);
    }
    let grid = first_grid(volume)?;
    let values = validate_grid(grid)?;
    if grid.dims.iter().any(|dimension| *dimension < 2) {
        return Err(invalid_volume(
            "surface extraction requires at least two samples on each axis",
        ));
    }

    let nx = grid.dims[0] as usize;
    let ny = grid.dims[1] as usize;
    let nz = grid.dims[2] as usize;
    let cell_count = (nx - 1)
        .checked_mul(ny - 1)
        .and_then(|count| count.checked_mul(nz - 1))
        .ok_or_else(resource_error)?;
    if cell_count > MAX_GRID_CELLS {
        return Err(resource_error());
    }
    let last_position = grid.origin
        + grid.voxel_size * DVec3::new((nx - 1) as f64, (ny - 1) as f64, (nz - 1) as f64);
    if !last_position.is_finite() {
        return Err(invalid_volume(
            "grid transform produces non-finite positions",
        ));
    }

    let mut positions = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut crossings = HashMap::new();
    for z in 0..nz - 1 {
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let cube_nodes = cube_nodes(x, y, z, nx, ny);
                let cube_positions = cube_nodes
                    .map(|node| node_position(node, nx, ny, grid.origin, grid.voxel_size));
                let mut cube_values = [0.0_f32; 8];
                for (corner, node) in cube_nodes.iter().copied().enumerate() {
                    cube_values[corner] = values
                        .get(node)
                        .ok_or_else(|| invalid_volume("volume sample index is out of range"))?;
                }
                for tetra in TETRAHEDRA {
                    let mut inside = [0_usize; 4];
                    let mut outside = [0_usize; 4];
                    let mut inside_count = 0;
                    let mut outside_count = 0;
                    for corner in tetra {
                        if f64::from(cube_values[corner]) < iso_level {
                            inside[inside_count] = corner;
                            inside_count += 1;
                        } else {
                            outside[outside_count] = corner;
                            outside_count += 1;
                        }
                    }
                    if inside_count == 0 || inside_count == 4 {
                        continue;
                    }

                    let inside_center = average_positions(
                        inside[..inside_count]
                            .iter()
                            .map(|corner| cube_positions[*corner]),
                    );
                    let outside_center = average_positions(
                        outside[..outside_count]
                            .iter()
                            .map(|corner| cube_positions[*corner]),
                    );
                    if inside_count == 1 || inside_count == 3 {
                        let (source, targets) = if inside_count == 1 {
                            (inside[0], [outside[0], outside[1], outside[2]])
                        } else {
                            (outside[0], [inside[0], inside[1], inside[2]])
                        };
                        let triangle = [
                            crossing_vertex(
                                source,
                                targets[0],
                                cube_nodes,
                                cube_values,
                                iso_level,
                                cube_positions,
                                &mut positions,
                                &mut crossings,
                            )?,
                            crossing_vertex(
                                source,
                                targets[1],
                                cube_nodes,
                                cube_values,
                                iso_level,
                                cube_positions,
                                &mut positions,
                                &mut crossings,
                            )?,
                            crossing_vertex(
                                source,
                                targets[2],
                                cube_nodes,
                                cube_values,
                                iso_level,
                                cube_positions,
                                &mut positions,
                                &mut crossings,
                            )?,
                        ];
                        push_oriented_triangle(
                            triangle,
                            outside_center - inside_center,
                            &positions,
                            &mut faces,
                        )?;
                    } else {
                        let a = crossing_vertex(
                            inside[0],
                            outside[0],
                            cube_nodes,
                            cube_values,
                            iso_level,
                            cube_positions,
                            &mut positions,
                            &mut crossings,
                        )?;
                        let b = crossing_vertex(
                            inside[0],
                            outside[1],
                            cube_nodes,
                            cube_values,
                            iso_level,
                            cube_positions,
                            &mut positions,
                            &mut crossings,
                        )?;
                        let c = crossing_vertex(
                            inside[1],
                            outside[1],
                            cube_nodes,
                            cube_values,
                            iso_level,
                            cube_positions,
                            &mut positions,
                            &mut crossings,
                        )?;
                        let d = crossing_vertex(
                            inside[1],
                            outside[0],
                            cube_nodes,
                            cube_values,
                            iso_level,
                            cube_positions,
                            &mut positions,
                            &mut crossings,
                        )?;
                        push_oriented_triangle(
                            [a, b, c],
                            outside_center - inside_center,
                            &positions,
                            &mut faces,
                        )?;
                        push_oriented_triangle(
                            [a, c, d],
                            outside_center - inside_center,
                            &positions,
                            &mut faces,
                        )?;
                    }
                }
            }
        }
    }
    if faces.is_empty() {
        return Err(invalid_volume(
            "volume scalar field does not cross the requested iso-level",
        ));
    }

    let polygons = faces.into_iter().map(|[a, b, c]| vec![a, b, c]).collect();
    Mesh::from_positions_and_faces(positions, polygons).map_err(|error| mesh_output_error(&error))
}

fn vdb_grid_to_mesh(grid: &crate::geom::vdb::VdbGrid, iso_level: f64) -> Result<Mesh> {
    let (active_min, active_max) = grid
        .active_bbox
        .ok_or_else(|| invalid_volume("VDB grid has no active voxels"))?;
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
    let mut values = vec![0.0; sample_count];
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
                let value = grid.sample_index(coordinate);
                if !value.is_finite() {
                    return Err(PotError::with_details(
                        ErrorCode::EvaluationFailed,
                        "VDB grid contains a non-finite density value",
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
    let mut dense = VolumeData::default();
    dense.grids.push(VolumeGrid {
        dims,
        voxel_size: 1.0,
        origin: DVec3::new(
            f64::from(minimum[0]),
            f64::from(minimum[1]),
            f64::from(minimum[2]),
        ),
        values: Some(values),
        ..VolumeGrid::default()
    });
    let mut mesh = volume_to_mesh(&dense, iso_level)?;
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
        surface.push(points);
    }

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
    let distance_tests = sample_count
        .checked_mul(surface.len())
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES
        || cell_count > MAX_GRID_CELLS
        || distance_tests > MAX_DISTANCE_TESTS
    {
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
                let distance = signed_distance(point, &surface);
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

/// Rasterize a closed mesh as a Blender-compatible fog-density grid.
///
/// Grid dimensions and origin are aligned to `OpenVDB`'s eight-voxel leaf bounds. The input
/// distance grid is sampled at the resulting voxel centers before converting signed distance
/// to fog density.
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
    if !interior_band_width.is_finite()
        || interior_band_width < 0.0
        || !density.is_finite()
        || density < 0.0
    {
        return Err(invalid_volume(
            "mesh-to-volume band width and density must be finite and non-negative",
        ));
    }
    let source = mesh_to_volume(mesh, voxel_size, 8)?;
    if source.grids.is_empty() {
        return Err(invalid_volume("mesh conversion produced no distance grid"));
    }
    let bounds = mesh
        .bounds()
        .ok_or_else(|| invalid_volume("mesh_to_volume requires a non-empty closed mesh"))?;
    let leaf_size = voxel_size * 8.0;
    if !leaf_size.is_finite() || leaf_size <= 0.0 {
        return Err(invalid_volume("mesh-to-volume voxel size is invalid"));
    }
    let minimum_leaf = (bounds.min / leaf_size).floor();
    let maximum_leaf = (bounds.max / leaf_size).floor();
    let minimum_index = minimum_leaf * 8.0;
    let dims = [
        ((maximum_leaf.x - minimum_leaf.x + 1.0) * 8.0) as usize,
        ((maximum_leaf.y - minimum_leaf.y + 1.0) * 8.0) as usize,
        ((maximum_leaf.z - minimum_leaf.z + 1.0) * 8.0) as usize,
    ];
    if dims
        .iter()
        .any(|dimension| *dimension == 0 || *dimension > MAX_GRID_AXIS)
    {
        return Err(resource_error());
    }
    let sample_count = dims[0]
        .checked_mul(dims[1])
        .and_then(|count| count.checked_mul(dims[2]))
        .ok_or_else(resource_error)?;
    if sample_count > MAX_GRID_SAMPLES {
        return Err(resource_error());
    }
    let origin = minimum_index * voxel_size;
    if !origin.is_finite() {
        return Err(invalid_volume("mesh-to-volume grid origin is not finite"));
    }
    let mut values = Vec::with_capacity(sample_count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let point = origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64);
                let signed_distance = sample_density_checked(point, &source)?;
                let band = interior_band_width.max(voxel_size);
                let fog = if use_fill_volume {
                    (-signed_distance / band).clamp(0.0, 1.0)
                } else {
                    ((band - signed_distance.abs()) / band).clamp(0.0, 1.0)
                };
                let value = (fog * density) as f32;
                if !value.is_finite() {
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
            values: Some(values),
            blob_f32: None,
            content_ref: None,
        }],
        source: VolumeSource::Generated(VolumeGeneratedSource {
            algorithm: "mesh_to_volume_fog".to_owned(),
        }),
        decoded_vdb: None,
    })
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
    let maximum = grid.origin
        + grid.voxel_size
            * DVec3::new(
                f64::from(grid.dims[0] - 1),
                f64::from(grid.dims[1] - 1),
                f64::from(grid.dims[2] - 1),
            );
    if !maximum.is_finite() {
        return Err(invalid_volume(
            "volume grid transform produces non-finite positions",
        ));
    }
    Ok(samples)
}

fn cube_nodes(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> [usize; 8] {
    let base = (z * ny + y) * nx + x;
    let stride_y = nx;
    let stride_z = nx * ny;
    [
        base,
        base + 1,
        base + stride_y,
        base + stride_y + 1,
        base + stride_z,
        base + stride_z + 1,
        base + stride_z + stride_y,
        base + stride_z + stride_y + 1,
    ]
}

fn node_position(node: usize, nx: usize, ny: usize, origin: DVec3, voxel_size: f64) -> DVec3 {
    let x = node % nx;
    let y = (node / nx) % ny;
    let z = node / (nx * ny);
    origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64)
}

fn crossing_vertex(
    first_corner: usize,
    second_corner: usize,
    cube_nodes: [usize; 8],
    cube_values: [f32; 8],
    iso_level: f64,
    cube_positions: [DVec3; 8],
    positions: &mut Vec<DVec3>,
    crossings: &mut HashMap<(usize, usize), usize>,
) -> Result<usize> {
    let first_node = cube_nodes[first_corner];
    let second_node = cube_nodes[second_corner];
    let key = edge_key(first_node as u32, second_node as u32);
    // Grid nodes are bounded below u32::MAX by the sample limit.
    let key = (key.0 as usize, key.1 as usize);
    if let Some(index) = crossings.get(&key) {
        return Ok(*index);
    }
    if positions.len() >= u32::MAX as usize {
        return Err(resource_error());
    }
    let first_value = f64::from(cube_values[first_corner]);
    let second_value = f64::from(cube_values[second_corner]);
    let fraction = (iso_level - first_value) / (second_value - first_value);
    if !fraction.is_finite() {
        return Err(invalid_volume("iso-level interpolation is not finite"));
    }
    let position =
        cube_positions[first_corner].lerp(cube_positions[second_corner], fraction.clamp(0.0, 1.0));
    if !position.is_finite() {
        return Err(invalid_volume("iso-surface position is not finite"));
    }
    let index = positions.len();
    positions.push(position);
    crossings.insert(key, index);
    Ok(index)
}

fn push_oriented_triangle(
    mut triangle: [usize; 3],
    outward: DVec3,
    positions: &[DVec3],
    faces: &mut Vec<[usize; 3]>,
) -> Result<()> {
    if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
        return Ok(());
    }
    let normal = (positions[triangle[1]] - positions[triangle[0]])
        .cross(positions[triangle[2]] - positions[triangle[0]]);
    if !normal.is_finite() || normal.length_squared() <= f64::MIN_POSITIVE {
        return Ok(());
    }
    if normal.dot(outward) < 0.0 {
        triangle.swap(1, 2);
    }
    if faces.len() >= MAX_OUTPUT_FACES {
        return Err(resource_error());
    }
    faces.push(triangle);
    Ok(())
}

fn average_positions(points: impl Iterator<Item = DVec3>) -> DVec3 {
    let mut sum = DVec3::ZERO;
    let mut count = 0_u32;
    for point in points {
        sum += point;
        count += 1;
    }
    sum / f64::from(count)
}

fn grid_axis(extent: f64, voxel_size: f64, padding: usize) -> Result<usize> {
    let axis = (extent / voxel_size).ceil() + 1.0 + 2.0 * padding as f64;
    if !axis.is_finite() || axis < 3.0 || axis > MAX_GRID_AXIS as f64 {
        return Err(resource_error());
    }
    Ok(axis as usize)
}

fn signed_distance(point: DVec3, triangles: &[[DVec3; 3]]) -> f64 {
    let mut closest_squared = f64::INFINITY;
    let mut intersections = 0_usize;
    let ray = DVec3::new(1.0, 0.371_390_676_354_103_7, 0.529_173_248_214_697_2);
    for triangle in triangles {
        closest_squared = closest_squared.min(point_triangle_distance_squared(point, *triangle));
        if ray_intersects_triangle(point, ray, *triangle) {
            intersections += 1;
        }
    }
    let distance = closest_squared.sqrt();
    if intersections % 2 == 1 {
        -distance
    } else {
        distance
    }
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

    use glam::DVec3;
    use proptest::prelude::*;

    use crate::{
        error::ErrorCode,
        geom::{BoxParams, Mesh},
    };

    use super::{
        VolumeData, VolumeGrid, mesh_to_volume, sample_density, sample_density_checked,
        volume_to_mesh,
    };

    #[test]
    fn density_sampling_is_trilinear_and_zero_outside() {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [2, 2, 2],
                voxel_size: 1.0,
                origin: DVec3::ZERO,
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
