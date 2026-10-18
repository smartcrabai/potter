// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Typed access to Blender multires displacement stored on mesh loop corners.

use std::mem::size_of;

use glam::{DMat3, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

/// One top-level MDisps layer in Blender loop order.
#[derive(Clone, Debug)]
pub(super) struct DisplacementData {
    pub total_levels: u32,
    pub grid_levels: Option<Vec<u32>>,
    pub corner_grids: Option<Vec<Vec<[f32; 3]>>>,
    pub external_filepath: Option<String>,
    pub paint_mask_levels: Option<Vec<u32>>,
    pub paint_mask_grids: Option<Vec<Vec<f32>>>,
    pub hidden_grids: Option<Vec<Vec<bool>>>,
    pub(super) grid_side: usize,
    face_loop_starts: Vec<usize>,
    face_ptex_starts: Vec<usize>,
    ptex_face_corners: Vec<(usize, usize)>,
    pub(super) loop_face_corners: Vec<(usize, usize)>,
}

impl DisplacementData {
    pub(super) fn paint_mask_value(
        &self,
        loop_index: usize,
        x: usize,
        y: usize,
        sculpt_level: u32,
    ) -> Result<f32> {
        grid_paint_mask_value(
            self.paint_mask_levels.as_deref(),
            self.paint_mask_grids.as_deref(),
            loop_index,
            x,
            y,
            sculpt_level,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MultiresGridData {
    pub(crate) total_levels: u32,
    pub(crate) grid_levels: Option<Vec<u32>>,
    pub(crate) corner_grids: Option<Vec<Vec<[f32; 3]>>>,
    pub(crate) external_filepath: Option<String>,
    pub(crate) paint_mask_levels: Option<Vec<u32>>,
    pub(crate) paint_mask_grids: Option<Vec<Vec<f32>>>,
    pub(crate) hidden_grids: Option<Vec<Vec<bool>>>,
}

#[cfg(test)]
impl MultiresGridData {
    pub(crate) fn paint_mask_value(
        &self,
        loop_index: usize,
        x: usize,
        y: usize,
        sculpt_level: u32,
    ) -> Result<f32> {
        grid_paint_mask_value(
            self.paint_mask_levels.as_deref(),
            self.paint_mask_grids.as_deref(),
            loop_index,
            x,
            y,
            sculpt_level,
        )
    }
}
fn grid_paint_mask_value(
    levels: Option<&[u32]>,
    grids: Option<&[Vec<f32>]>,
    loop_index: usize,
    x: usize,
    y: usize,
    sculpt_level: u32,
) -> Result<f32> {
    let (Some(levels), Some(grids)) = (levels, grids) else {
        return Ok(0.0);
    };
    let Some(level) = levels.get(loop_index).copied() else {
        return Err(invalid("multires paint-mask loop index is out of range"));
    };
    let Some(grid) = grids.get(loop_index) else {
        return Err(invalid("multires paint-mask loop index is out of range"));
    };
    if level == 0 {
        return Ok(0.0);
    }
    if sculpt_level == 0 {
        return Err(invalid("sculpt level must be positive for paint-mask sampling"));
    }
    let side = grid_side(level)?;
    let sculpt_side = grid_side(sculpt_level)?;
    if x >= sculpt_side || y >= sculpt_side {
        return Err(invalid(
            "multires paint-mask sample is outside its sculpt grid",
        ));
    }
    // Blender's read_mask_grid rounds the grid coordinate into the stored mask grid at any level,
    // so a sculpt level finer or coarser than the stored one maps through the same rounding.
    let sculpt_maximum = (sculpt_side - 1) as f32;
    let stored_maximum = (side - 1) as f32;
    let stored_x = ((x as f32 / sculpt_maximum) * stored_maximum).round() as usize;
    let stored_y = ((y as f32 / sculpt_maximum) * stored_maximum).round() as usize;
    grid.get(stored_y * side + stored_x)
        .copied()
        .ok_or_else(|| invalid("multires paint-mask grid has an invalid size"))
}

/// Decode and validate the mesh's authoritative multires corner grids, if present.
pub(super) fn read(mesh: &Mesh) -> Result<Option<DisplacementData>> {
    let Some(record) = mesh.attributes.get("multires_displacement") else {
        return Ok(None);
    };
    let record = record
        .as_object()
        .ok_or_else(|| invalid("multires_displacement must be an object"))?;
    let total_levels = record
        .get("total_levels")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value <= 13)
        .ok_or_else(|| invalid("multires_displacement.total_levels must be between 0 and 13"))?;
    let external_filepath = record
        .get("external")
        .map(|external| {
            external
                .get("filepath")
                .and_then(Value::as_str)
                .filter(|path| !path.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid("multires_displacement.external.filepath must be non-empty"))
        })
        .transpose()?;
    if total_levels == 0 && external_filepath.is_some() {
        return Err(invalid(
            "multires_displacement.external requires a positive total_levels",
        ));
    }
    let corner_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(|| invalid("mesh corner count overflows"))?;
    let grid_levels = parse_grid_levels(record.get("grid_levels"), corner_count, total_levels)?;
    let grid_vectors = (0..corner_count)
        .map(|index| {
            let level = grid_levels
                .as_ref()
                .map_or(total_levels, |levels| levels[index]);
            corner_grid_samples(level)
        })
        .collect::<Result<Vec<_>>>()?;
    let (paint_mask_levels, paint_mask_grids) =
        parse_paint_masks(record.get("paint_mask"), corner_count, total_levels)?;
    let hidden_grids = parse_hidden_grids(record.get("hidden"), &grid_vectors)?;
    let paint_mask_samples = paint_mask_grids.as_deref().map(sample_count).transpose()?;
    let hidden_samples = hidden_grids.as_deref().map(sample_count).transpose()?;
    check_grid_capacity_for_grids(&grid_vectors, paint_mask_samples, hidden_samples, 0)?;
    let corner_grids = match record.get("values") {
        Some(_) if total_levels == 0 => {
            return Err(invalid(
                "multires_displacement.values requires a positive total_levels",
            ));
        }
        Some(raw_values) => Some(read_values(mesh, raw_values, &grid_vectors)?),
        None => None,
    };
    let (face_loop_starts, face_ptex_starts, ptex_face_corners, loop_face_corners) =
        ptex_mapping(mesh)?;
    let grid_side = grid_side(total_levels)?;
    Ok(Some(DisplacementData {
        total_levels,
        grid_levels,
        corner_grids,
        external_filepath,
        paint_mask_levels,
        paint_mask_grids,
        hidden_grids,
        grid_side,
        face_loop_starts,
        face_ptex_starts,
        ptex_face_corners,
        loop_face_corners,
    }))
}

fn parse_grid_levels(
    raw: Option<&Value>,
    corner_count: usize,
    total_levels: u32,
) -> Result<Option<Vec<u32>>> {
    raw.map(|raw| {
        let levels = raw
            .as_array()
            .filter(|levels| levels.len() == corner_count)
            .ok_or_else(|| invalid("multires_displacement.grid_levels must match mesh corners"))?;
        levels
            .iter()
            .map(|level| {
                level
                    .as_u64()
                    .and_then(|level| u32::try_from(level).ok())
                    .filter(|level| (1..=13).contains(level) && *level <= total_levels)
                    .ok_or_else(|| {
                        invalid("multires_displacement.grid_levels contains an invalid level")
                    })
            })
            .collect()
    })
    .transpose()
}

fn parse_paint_masks(
    raw: Option<&Value>,
    corner_count: usize,
    total_levels: u32,
) -> Result<(Option<Vec<u32>>, Option<Vec<Vec<f32>>>)> {
    let Some(raw) = raw else {
        return Ok((None, None));
    };
    let masks = raw
        .as_array()
        .filter(|masks| masks.len() == corner_count)
        .ok_or_else(|| invalid("multires_displacement.paint_mask must match mesh corners"))?;
    let mut levels = Vec::with_capacity(corner_count);
    let mut grids = Vec::with_capacity(corner_count);
    for (index, mask) in masks.iter().enumerate() {
        let level = mask
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| u32::try_from(level).ok())
            .filter(|level| *level <= 13 && *level <= total_levels)
            .ok_or_else(|| {
                invalid(format!(
                    "multires paint-mask grid {index} has an invalid level"
                ))
            })?;
        let values = mask
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid(format!(
                    "multires paint-mask grid {index} has invalid values"
                ))
            })?;
        let expected = corner_grid_samples(level)?;
        if values.len() != expected {
            return Err(invalid(format!(
                "multires paint-mask grid {index} has an invalid size"
            )));
        }
        let values = values
            .iter()
            .map(|value| {
                let value = value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid("multires paint-mask values must be finite"))?;
                let value = value as f32;
                if !value.is_finite() {
                    return Err(invalid("multires paint-mask value exceeds f32 range"));
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        levels.push(level);
        grids.push(values);
    }
    Ok((Some(levels), Some(grids)))
}

fn parse_hidden_grids(
    raw: Option<&Value>,
    grid_vectors: &[usize],
) -> Result<Option<Vec<Vec<bool>>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let hidden = raw
        .as_array()
        .filter(|hidden| hidden.len() == grid_vectors.len())
        .ok_or_else(|| invalid("multires_displacement.hidden must match mesh corners"))?;
    hidden
        .iter()
        .zip(grid_vectors)
        .enumerate()
        .map(|(index, (grid, expected))| {
            let grid = grid
                .as_array()
                .filter(|grid| grid.len() == *expected)
                .ok_or_else(|| {
                    invalid(format!("multires hidden grid {index} has an invalid size"))
                })?;
            grid.iter()
                .map(|value| {
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("multires hidden samples must be boolean"))
                })
                .collect()
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub(crate) fn read_grid_data(mesh: &Mesh) -> Result<Option<MultiresGridData>> {
    Ok(read(mesh)?.map(|data| MultiresGridData {
        total_levels: data.total_levels,
        grid_levels: data.grid_levels,
        corner_grids: data.corner_grids,
        external_filepath: data.external_filepath,
        paint_mask_levels: data.paint_mask_levels,
        paint_mask_grids: data.paint_mask_grids,
        hidden_grids: data.hidden_grids,
    }))
}

/// Side length of a corner grid at `level` (`2^(level - 1) + 1`); level 0 stores no grid.
pub(crate) fn grid_side(level: u32) -> Result<usize> {
    if level == 0 {
        return Ok(0);
    }
    1_usize
        .checked_shl(level - 1)
        .and_then(|side| side.checked_add(1))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "multires grid side overflows"))
}

/// Samples in one corner grid at `level`; level 0 stores no grid.
fn corner_grid_samples(level: u32) -> Result<usize> {
    if level == 0 {
        return Ok(0);
    }
    super::super::multires_external::grid_vector_count(level)
}

fn grid_side_for_loop(data: &DisplacementData, loop_index: usize) -> Result<usize> {
    match data
        .grid_levels
        .as_ref()
        .and_then(|levels| levels.get(loop_index))
    {
        Some(level) => grid_side(*level),
        None if loop_index < data.loop_face_corners.len() => Ok(data.grid_side),
        None => Err(invalid("multires loop-corner index is out of range")),
    }
}

fn sample_count<T>(grids: &[Vec<T>]) -> Result<u128> {
    grids
        .iter()
        .try_fold(0_u128, |count, grid| {
            count.checked_add(u128::try_from(grid.len()).ok()?)
        })
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "multires sample count overflows"))
}

pub(super) fn check_grid_capacity_for_corners(
    corner_count: usize,
    grid_vectors: usize,
    additional_bytes: u64,
) -> Result<()> {
    let samples = u128::try_from(corner_count)
        .ok()
        .and_then(|corners| corners.checked_mul(u128::try_from(grid_vectors).ok()?))
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "multires sample count overflows")
        })?;
    check_grid_capacity_for_samples(corner_count, samples, None, None, additional_bytes)
}

pub(crate) fn check_grid_capacity_for_grids(
    grid_vectors: &[usize],
    paint_mask_samples: Option<u128>,
    hidden_samples: Option<u128>,
    additional_bytes: u64,
) -> Result<()> {
    let displacement_samples = grid_vectors
        .iter()
        .try_fold(0_u128, |count, &size| {
            count.checked_add(u128::try_from(size).ok()?)
        })
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "multires sample count overflows")
        })?;
    check_grid_capacity_for_samples(
        grid_vectors.len(),
        displacement_samples,
        paint_mask_samples,
        hidden_samples,
        additional_bytes,
    )
}

fn check_grid_capacity_for_samples(
    corner_count: usize,
    displacement_samples: u128,
    paint_mask_samples: Option<u128>,
    hidden_samples: Option<u128>,
    additional_bytes: u64,
) -> Result<()> {
    let corners = u128::try_from(corner_count).ok();
    let value_size = size_of::<Value>() as u128;
    let peak_bytes = corners
        .and_then(|corners| {
            let displacement_typed =
                displacement_samples.checked_mul(size_of::<[f32; 3]>() as u128)?;
            let displacement_json = displacement_samples
                .checked_mul(5)?
                .checked_mul(value_size)?
                .checked_mul(3)?;
            let displacement_descriptors =
                corners.checked_mul(size_of::<Vec<[f32; 3]>>() as u128)?;
            let mask_samples = paint_mask_samples.unwrap_or(0);
            let mask_typed = mask_samples.checked_mul(size_of::<f32>() as u128)?;
            let mask_json = mask_samples
                .checked_mul(2)?
                .checked_mul(value_size)?
                .checked_mul(2)?;
            let mask_descriptors = corners.checked_mul(size_of::<Vec<f32>>() as u128)?;
            let hidden_samples = hidden_samples.unwrap_or(0);
            let hidden_typed = hidden_samples.checked_mul(size_of::<bool>() as u128)?;
            let hidden_json = hidden_samples
                .checked_mul(2)?
                .checked_mul(value_size)?
                .checked_mul(2)?;
            let hidden_descriptors = corners.checked_mul(size_of::<Vec<bool>>() as u128)?;
            let metadata_bytes =
                corners.checked_mul(size_of::<u32>() as u128 * 2 + value_size * 12)?;
            displacement_typed
                .checked_add(displacement_json)?
                .checked_add(displacement_descriptors)?
                .checked_add(mask_typed)?
                .checked_add(mask_json)?
                .checked_add(mask_descriptors)?
                .checked_add(hidden_typed)?
                .checked_add(hidden_json)?
                .checked_add(hidden_descriptors)?
                .checked_add(metadata_bytes)?
                .checked_add(u128::from(additional_bytes))
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "multires displacement size overflows",
            )
        })?;
    if peak_bytes > super::MAX_SUBDIVISION_MEMORY_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "multires displacement peak memory exceeds the 8-GiB memory budget",
        ));
    }
    Ok(())
}

fn read_values(
    mesh: &Mesh,
    raw_values: &Value,
    grid_vectors: &[usize],
) -> Result<Vec<Vec<[f32; 3]>>> {
    let values = raw_values
        .as_object()
        .ok_or_else(|| invalid("multires_displacement.values must be keyed by face ID"))?;
    if values.len() != mesh.faces.len() {
        return Err(invalid(
            "multires_displacement.values must contain every mesh face",
        ));
    }
    let corner_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let corner_count = corner_count.ok_or_else(|| invalid("mesh corner count overflows"))?;
    if grid_vectors.len() != corner_count {
        return Err(invalid("multires grid levels do not match mesh corners"));
    }
    let mut grids = Vec::with_capacity(corner_count);
    let mut grid_index = 0_usize;
    for face in &mesh.faces {
        let key = format!("f{}", face.id);
        let face_values = values
            .get(&key)
            .and_then(Value::as_array)
            .filter(|corners| corners.len() == face.vertices.len())
            .ok_or_else(|| {
                invalid(format!(
                    "multires_displacement.values.{key} has invalid corners"
                ))
            })?;
        for (corner, raw_grid) in face_values.iter().enumerate() {
            let grid_vectors = grid_vectors[grid_index];
            let grid = raw_grid
                .as_array()
                .filter(|grid| grid.len() == grid_vectors)
                .ok_or_else(|| {
                    invalid(format!("multires grid {key}[{corner}] has an invalid size"))
                })?;
            let mut vectors = Vec::with_capacity(grid_vectors);
            for (sample, vector) in grid.iter().enumerate() {
                let components = vector
                    .as_array()
                    .filter(|components| components.len() == 3)
                    .ok_or_else(|| {
                        invalid(format!(
                            "multires vector {key}[{corner}][{sample}] is invalid"
                        ))
                    })?;
                let mut value = [0.0_f32; 3];
                for (axis, component) in components.iter().enumerate() {
                    let number = component
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| invalid("multires displacement component must be finite"))?;
                    value[axis] = number as f32;
                    if !value[axis].is_finite() {
                        return Err(invalid("multires displacement component exceeds f32 range"));
                    }
                }
                vectors.push(value);
            }
            grids.push(vectors);
            grid_index += 1;
        }
    }
    Ok(grids)
}

#[derive(Clone, Copy)]
struct GridLocation {
    face_index: usize,
    face_corner: usize,
    grid_corner: usize,
    grid_index: usize,
    vector_index: usize,
    grid_u: f32,
    grid_v: f32,
    average: BoundaryAverage,
}

#[derive(Clone, Copy)]
enum BoundaryAverage {
    None,
    All,
    Previous,
    Next,
}

fn ptex_mapping(
    mesh: &Mesh,
) -> Result<(
    Vec<usize>,
    Vec<usize>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
)> {
    let loop_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let loop_count = loop_count.ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "multires mesh corner count overflows",
        )
    })?;
    let mut face_loop_starts = Vec::with_capacity(mesh.faces.len());
    let mut face_ptex_starts = Vec::with_capacity(mesh.faces.len());
    let mut ptex_face_corners = Vec::with_capacity(loop_count);
    let mut loop_face_corners = Vec::with_capacity(loop_count);
    let mut loop_start = 0_usize;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        face_loop_starts.push(loop_start);
        face_ptex_starts.push(ptex_face_corners.len());
        loop_face_corners.extend((0..face.vertices.len()).map(|corner| (face_index, corner)));
        if face.vertices.len() == 4 {
            ptex_face_corners.push((face_index, 0));
        } else {
            ptex_face_corners.extend((0..face.vertices.len()).map(|corner| (face_index, corner)));
        }
        loop_start = loop_start.checked_add(face.vertices.len()).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "multires mesh corner count overflows",
            )
        })?;
    }
    Ok((
        face_loop_starts,
        face_ptex_starts,
        ptex_face_corners,
        loop_face_corners,
    ))
}

fn grid_location(
    mesh: &Mesh,
    data: &DisplacementData,
    ptex_face_index: usize,
    u: f32,
    v: f32,
) -> Result<GridLocation> {
    if !u.is_finite() || !v.is_finite() || !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return Err(invalid("multires ptex sample is outside the face"));
    }
    let (face_index, source_corner) = data
        .ptex_face_corners
        .get(ptex_face_index)
        .copied()
        .ok_or_else(|| invalid("multires ptex face index is out of range"))?;
    let face = mesh
        .faces
        .get(face_index)
        .ok_or_else(|| invalid("multires ptex face mapping is invalid"))?;
    let (grid_corner, corner_u, corner_v) = if face.vertices.len() == 4 {
        rotate_quad_to_corner(u, v)
    } else {
        (source_corner, u, v)
    };
    let grid_u = 1.0 - corner_v;
    let grid_v = 1.0 - corner_u;
    let grid_index = data
        .face_loop_starts
        .get(face_index)
        .copied()
        .and_then(|start| start.checked_add(grid_corner))
        .ok_or_else(|| invalid("multires loop-corner mapping is invalid"))?;
    let grid_side = grid_side_for_loop(data, grid_index)?;
    if grid_side == 0 {
        return Err(invalid("multires displacement has no grid level"));
    }
    let maximum = (grid_side - 1) as f32;
    let x = (grid_u * maximum).round() as usize;
    let y = (grid_v * maximum).round() as usize;
    if x >= grid_side || y >= grid_side {
        return Err(invalid("multires ptex sample is outside its corner grid"));
    }
    Ok(GridLocation {
        face_index,
        face_corner: if face.vertices.len() == 4 {
            grid_corner
        } else {
            source_corner
        },
        grid_corner,
        grid_index,
        vector_index: y * grid_side + x,
        grid_u,
        grid_v,
        average: if x == 0 && y == 0 {
            BoundaryAverage::All
        } else if x == 0 {
            BoundaryAverage::Previous
        } else if y == 0 {
            BoundaryAverage::Next
        } else {
            BoundaryAverage::None
        },
    })
}

fn rotate_quad_to_corner(u: f32, v: f32) -> (usize, f32, f32) {
    if u <= 0.5 && v <= 0.5 {
        (0, 2.0 * u, 2.0 * v)
    } else if u > 0.5 && v <= 0.5 {
        (1, 2.0 * v, 2.0 * (1.0 - u))
    } else if u > 0.5 && v > 0.5 {
        (2, 2.0 * (1.0 - u), 2.0 * (1.0 - v))
    } else {
        (3, 2.0 * (1.0 - v), 2.0 * u)
    }
}

fn rotate_grid_to_quad(corner: usize, u: f32, v: f32) -> (f32, f32) {
    match corner {
        0 => (0.5 - v * 0.5, 0.5 - u * 0.5),
        1 => (0.5 + u * 0.5, 0.5 - v * 0.5),
        2 => (0.5 + v * 0.5, 0.5 + u * 0.5),
        _ => (0.5 - u * 0.5, 0.5 + v * 0.5),
    }
}

fn ptex_from_grid(num_corners: usize, corner: usize, u: f32, v: f32) -> (f32, f32) {
    if num_corners == 4 {
        rotate_grid_to_quad(corner, u, v)
    } else {
        (1.0 - v, 1.0 - u)
    }
}

#[derive(Clone, Copy)]
pub(super) struct GridSampleLocation {
    pub loop_index: usize,
    pub vector_index: usize,
    pub ptex_face_index: usize,
    pub u: f32,
    pub v: f32,
    pub tangent_corner: usize,
    pub grid_side: usize,
}

pub(super) fn grid_sample_location(
    mesh: &Mesh,
    data: &DisplacementData,
    loop_index: usize,
    vector_index: usize,
) -> Result<GridSampleLocation> {
    let (face_index, corner) = data
        .loop_face_corners
        .get(loop_index)
        .copied()
        .ok_or_else(|| invalid("multires loop-corner index is out of range"))?;
    let face = mesh
        .faces
        .get(face_index)
        .ok_or_else(|| invalid("multires loop-corner mapping is invalid"))?;
    let side = grid_side_for_loop(data, loop_index)?;
    let area = side
        .checked_mul(side)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "multires grid size overflows"))?;
    if side == 0 || vector_index >= area {
        return Err(invalid("multires grid sample index is out of range"));
    }
    let x = vector_index % side;
    let y = vector_index / side;
    let divisor = (side - 1) as f32;
    let grid_u = x as f32 / divisor;
    let grid_v = y as f32 / divisor;
    let ptex_face_index = if face.vertices.len() == 4 {
        data.face_ptex_starts[face_index]
    } else {
        data.face_ptex_starts[face_index] + corner
    };
    let (u, v) = ptex_from_grid(face.vertices.len(), corner, grid_u, grid_v);
    Ok(GridSampleLocation {
        grid_side: side,
        loop_index,
        vector_index,
        ptex_face_index,
        u,
        v,
        tangent_corner: if face.vertices.len() == 4 { corner } else { 0 },
    })
}

pub(super) fn grid_location_for_sample(
    mesh: &Mesh,
    data: &DisplacementData,
    ptex_face_index: usize,
    u: f32,
    v: f32,
) -> Result<GridSampleLocation> {
    let location = grid_location(mesh, data, ptex_face_index, u, v)?;
    Ok(GridSampleLocation {
        loop_index: location.grid_index,
        vector_index: location.vector_index,
        ptex_face_index,
        u,
        v,
        grid_side: grid_side_for_loop(data, location.grid_index)?,
        tangent_corner: location.grid_corner,
    })
}

pub(super) fn ptex_face_for_sample(
    mesh: &Mesh,
    data: &DisplacementData,
    face_index: usize,
    ptex: usize,
) -> Result<usize> {
    let face = mesh
        .faces
        .get(face_index)
        .ok_or_else(|| invalid("multires sample face index is out of range"))?;
    let ptex_count = if face.vertices.len() == 4 {
        1
    } else {
        face.vertices.len()
    };
    if ptex >= ptex_count {
        return Err(invalid("multires sample ptex index is out of range"));
    }
    data.face_ptex_starts[face_index]
        .checked_add(ptex)
        .ok_or_else(|| invalid("multires ptex face index overflows"))
}

pub(super) fn ptex_sample_for_face(
    mesh: &Mesh,
    data: &DisplacementData,
    ptex_face_index: usize,
) -> Result<(usize, usize)> {
    let (face_index, corner) = data
        .ptex_face_corners
        .get(ptex_face_index)
        .copied()
        .ok_or_else(|| invalid("multires ptex face index is out of range"))?;
    let face = mesh
        .faces
        .get(face_index)
        .ok_or_else(|| invalid("multires ptex face mapping is invalid"))?;
    Ok((
        face_index,
        if face.vertices.len() == 4 { 0 } else { corner },
    ))
}

pub(super) fn ptex_grid_denominator(
    mesh: &Mesh,
    data: &DisplacementData,
    ptex_face_index: usize,
    level: u32,
) -> Result<usize> {
    let (face_index, _) = ptex_sample_for_face(mesh, data, ptex_face_index)?;
    let face = &mesh.faces[face_index];
    let shift = if face.vertices.len() == 4 {
        level
    } else {
        level
            .checked_sub(1)
            .ok_or_else(|| invalid("multires ptex level must be positive"))?
    };
    1_usize.checked_shl(shift).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "multires ptex resolution overflows",
        )
    })
}

pub(super) fn grid_sample_value(
    data: &DisplacementData,
    location: GridSampleLocation,
) -> Result<[f32; 3]> {
    if data.corner_grids.is_none()
        && let Some(filepath) = &data.external_filepath
    {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            format!("multires external grid values are unavailable from `{filepath}`"),
            json!({"filepath": filepath}),
        ));
    }
    match &data.corner_grids {
        None => Ok([0.0; 3]),
        Some(grids) => grids
            .get(location.loop_index)
            .and_then(|grid| grid.get(location.vector_index))
            .copied()
            .ok_or_else(|| invalid("multires displacement grid mapping is invalid")),
    }
}

pub(super) fn set_grid_sample_value(
    mesh: &Mesh,
    data: &mut DisplacementData,
    location: GridSampleLocation,
    value: [f32; 3],
) -> Result<()> {
    if !value.iter().all(|component| component.is_finite()) {
        return Err(invalid("multires displacement component must be finite"));
    }
    if data.corner_grids.is_none()
        && let Some(filepath) = &data.external_filepath
    {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            format!("multires external grid values are unavailable from `{filepath}`"),
            json!({"filepath": filepath}),
        ));
    }
    let grid_len = location
        .grid_side
        .checked_mul(location.grid_side)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "multires grid size overflows"))?;
    if grid_len == 0 {
        return Err(invalid("multires displacement has no grid level"));
    }
    if data.corner_grids.is_none() {
        let mesh_corner_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        });
        if mesh_corner_count != Some(data.loop_face_corners.len()) {
            return Err(invalid(
                "multires loop mapping no longer matches mesh topology",
            ));
        }
        let grid_vectors = (0..data.loop_face_corners.len())
            .map(|loop_index| {
                let side = grid_side_for_loop(data, loop_index)?;
                side.checked_mul(side).ok_or_else(|| {
                    PotError::new(ErrorCode::LimitExceeded, "multires grid size overflows")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let paint_mask_samples = data
            .paint_mask_grids
            .as_deref()
            .map(sample_count)
            .transpose()?;
        let hidden_samples = data.hidden_grids.as_deref().map(sample_count).transpose()?;
        check_grid_capacity_for_grids(&grid_vectors, paint_mask_samples, hidden_samples, 0)?;
        data.corner_grids = Some(
            grid_vectors
                .into_iter()
                .map(|size| vec![[0.0; 3]; size])
                .collect(),
        );
    }
    let grid = data
        .corner_grids
        .as_mut()
        .and_then(|grids| grids.get_mut(location.loop_index))
        .and_then(|grid| grid.get_mut(location.vector_index))
        .ok_or_else(|| invalid("multires displacement grid mapping is invalid"))?;
    *grid = value;
    Ok(())
}

fn read_grid(
    data: &DisplacementData,
    location: GridLocation,
) -> Result<([f32; 3], BoundaryAverage)> {
    let Some(grids) = &data.corner_grids else {
        return Ok(([0.0; 3], BoundaryAverage::None));
    };
    let grid = grids
        .get(location.grid_index)
        .and_then(|grid| grid.get(location.vector_index))
        .copied()
        .ok_or_else(|| invalid("multires displacement grid mapping is invalid"))?;
    Ok((grid, location.average))
}

fn tangent_axes(du: DVec3, dv: DVec3, corner: usize) -> [DVec3; 3] {
    let (x, y) = match corner {
        0 => (-dv, -du),
        1 => (du, -dv),
        2 => (dv, du),
        _ => (-du, dv),
    };
    [
        x.normalize_or_zero(),
        y.normalize_or_zero(),
        du.cross(dv).normalize_or_zero(),
    ]
}

pub(super) fn tangent_to_object(du: DVec3, dv: DVec3, corner: usize, value: [f32; 3]) -> DVec3 {
    let [x, y, z] = tangent_axes(du, dv, corner);
    x * f64::from(value[0]) + y * f64::from(value[1]) + z * f64::from(value[2])
}

pub(super) fn tangent_from_object(
    du: DVec3,
    dv: DVec3,
    corner: usize,
    object_delta: DVec3,
) -> Result<[f32; 3]> {
    let [x_axis, y_axis, z_axis] = tangent_axes(du, dv, corner);
    let basis = DMat3::from_cols(x_axis, y_axis, z_axis);
    let determinant = basis.determinant();
    if !object_delta.is_finite() || !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return Err(invalid(
            "multires sculpt sample has an invalid tangent frame",
        ));
    }
    let tangent = basis.inverse() * object_delta;
    let value = [tangent.x as f32, tangent.y as f32, tangent.z as f32];
    if !value.iter().all(|component| component.is_finite()) {
        return Err(invalid("multires sculpt displacement exceeds f32 range"));
    }
    Ok(value)
}

/// Evaluate Blender's grid lookup and tangent-space conversion for a ptex sample.
pub(super) fn evaluate_sample(
    mesh: &Mesh,
    data: &DisplacementData,
    ptex_face_index: usize,
    u: f32,
    v: f32,
    mut derivatives: impl FnMut(usize, f32, f32) -> Result<(DVec3, DVec3)>,
) -> Result<DVec3> {
    if data.corner_grids.is_none()
        && let Some(filepath) = &data.external_filepath
    {
        return Err(PotError::with_details(
            ErrorCode::FileNotFound,
            format!("multires external grid values are unavailable from `{filepath}`"),
            json!({"filepath": filepath}),
        ));
    }
    if data.corner_grids.is_none() {
        return Ok(DVec3::ZERO);
    }
    let location = grid_location(mesh, data, ptex_face_index, u, v)?;
    let (tangent, average) = read_grid(data, location)?;
    let (du, dv) = derivatives(ptex_face_index, u, v)?;
    let corner_count = mesh.faces[location.face_index].vertices.len();
    let tangent_corner = if corner_count == 4 {
        location.grid_corner
    } else {
        0
    };
    let mut displacement = tangent_to_object(du, dv, tangent_corner, tangent);
    match average {
        BoundaryAverage::All => {
            // Blender's average_with_all folds each other corner in as a running half-average.
            for corner_delta in 1..corner_count {
                let other = average_with_other(
                    mesh,
                    data,
                    location,
                    corner_delta as isize,
                    0.0,
                    0.0,
                    &mut derivatives,
                )?;
                displacement = (displacement + other) * 0.5;
            }
        }
        BoundaryAverage::Previous => {
            let other = average_with_other(
                mesh,
                data,
                location,
                -1,
                location.grid_v,
                0.0,
                &mut derivatives,
            )?;
            displacement = (displacement + other) * 0.5;
        }
        BoundaryAverage::Next => {
            let other = average_with_other(
                mesh,
                data,
                location,
                1,
                0.0,
                location.grid_u,
                &mut derivatives,
            )?;
            displacement = (displacement + other) * 0.5;
        }
        BoundaryAverage::None => {}
    }
    Ok(displacement)
}

fn average_with_other(
    mesh: &Mesh,
    data: &DisplacementData,
    location: GridLocation,
    corner_delta: isize,
    grid_u: f32,
    grid_v: f32,
    derivatives: &mut impl FnMut(usize, f32, f32) -> Result<(DVec3, DVec3)>,
) -> Result<DVec3> {
    let num_corners = mesh.faces[location.face_index].vertices.len();
    let other_corner = (location.face_corner as isize + corner_delta + num_corners as isize)
        as usize
        % num_corners;
    let other_ptex_face = if num_corners == 4 {
        data.face_ptex_starts[location.face_index]
    } else {
        data.face_ptex_starts[location.face_index] + other_corner
    };
    let (u, v) = ptex_from_grid(num_corners, other_corner, grid_u, grid_v);
    let grid_index = data
        .face_loop_starts
        .get(location.face_index)
        .copied()
        .and_then(|start| start.checked_add(other_corner))
        .ok_or_else(|| invalid("multires loop-corner mapping is invalid"))?;
    let grid_side = grid_side_for_loop(data, grid_index)?;
    if grid_side == 0 {
        return Err(invalid("multires displacement has no grid level"));
    }
    let maximum = (grid_side - 1) as f32;
    let x = (grid_u * maximum).round() as usize;
    let y = (grid_v * maximum).round() as usize;
    if x >= grid_side || y >= grid_side {
        return Err(invalid("multires ptex sample is outside its corner grid"));
    }
    let other_location = GridLocation {
        face_index: location.face_index,
        face_corner: other_corner,
        grid_corner: other_corner,
        grid_index,
        vector_index: y * grid_side + x,
        grid_u,
        grid_v,
        average: BoundaryAverage::None,
    };
    let (tangent, _) = read_grid(data, other_location)?;
    let (du, dv) = derivatives(other_ptex_face, u, v)?;
    let tangent_corner = if num_corners == 4 { other_corner } else { 0 };
    Ok(tangent_to_object(du, dv, tangent_corner, tangent))
}

#[cfg(test)]
/// Add an object-space sculpt delta to the selected corner-grid sample.
pub(super) fn add_sample_delta(
    mesh: &Mesh,
    data: &mut DisplacementData,
    ptex_face_index: usize,
    u: f32,
    v: f32,
    object_delta: DVec3,
    derivatives: impl FnOnce(usize, f32, f32) -> Result<(DVec3, DVec3)>,
) -> Result<(usize, usize)> {
    let location = grid_location(mesh, data, ptex_face_index, u, v)?;
    let (du, dv) = derivatives(ptex_face_index, u, v)?;
    let tangent_delta = tangent_from_object(du, dv, location.grid_corner, object_delta)?;
    let grid_sample = GridSampleLocation {
        loop_index: location.grid_index,
        vector_index: location.vector_index,
        ptex_face_index,
        u,
        v,
        grid_side: grid_side_for_loop(data, location.grid_index)?,
        tangent_corner: location.grid_corner,
    };
    let mut value = grid_sample_value(data, grid_sample)?;
    for axis in 0..3 {
        value[axis] += tangent_delta[axis];
    }
    set_grid_sample_value(mesh, data, grid_sample, value)?;
    Ok((location.grid_index, location.vector_index))
}

pub(crate) fn write_grid_data(mesh: &mut Mesh, data: &MultiresGridData) -> Result<()> {
    let mut starts = Vec::with_capacity(mesh.faces.len());
    let mut corner_count = 0_usize;
    for face in &mesh.faces {
        starts.push(corner_count);
        corner_count = corner_count
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid("multires face corner range overflows"))?;
    }
    write_grid_record(
        mesh,
        data.total_levels,
        data.grid_levels.as_deref(),
        data.corner_grids.as_deref(),
        data.external_filepath.as_deref(),
        data.paint_mask_levels.as_deref(),
        data.paint_mask_grids.as_deref(),
        data.hidden_grids.as_deref(),
        &starts,
        corner_count,
    )
}

pub(super) fn write(mesh: &mut Mesh, data: &DisplacementData) -> Result<()> {
    let corner_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(|| invalid("mesh corner count overflows"))?;
    write_grid_record(
        mesh,
        data.total_levels,
        data.grid_levels.as_deref(),
        data.corner_grids.as_deref(),
        data.external_filepath.as_deref(),
        data.paint_mask_levels.as_deref(),
        data.paint_mask_grids.as_deref(),
        data.hidden_grids.as_deref(),
        &data.face_loop_starts,
        corner_count,
    )
}

fn write_grid_record(
    mesh: &mut Mesh,
    total_levels: u32,
    grid_levels: Option<&[u32]>,
    corner_grids: Option<&[Vec<[f32; 3]>]>,
    external_filepath: Option<&str>,
    paint_mask_levels: Option<&[u32]>,
    paint_mask_grids: Option<&[Vec<f32>]>,
    hidden_grids: Option<&[Vec<bool>]>,
    face_loop_starts: &[usize],
    corner_count: usize,
) -> Result<()> {
    if total_levels > 13 {
        return Err(invalid("multires total_levels must be between 0 and 13"));
    }
    let levels = match grid_levels {
        Some(levels) if levels.len() == corner_count => levels.to_vec(),
        Some(_) => return Err(invalid("multires grid levels do not match mesh corners")),
        None => vec![total_levels; corner_count],
    };
    if corner_grids.is_some() && total_levels == 0 {
        return Err(invalid("multires values require a positive total_levels"));
    }
    if grid_levels.is_some()
        && levels
            .iter()
            .any(|level| *level == 0 || *level > total_levels)
    {
        return Err(invalid(
            "multires grid levels must be positive and no greater than total_levels",
        ));
    }
    let vectors_per_grid = levels
        .iter()
        .map(|&level| corner_grid_samples(level))
        .collect::<Result<Vec<_>>>()?;
    let paint_mask_samples = paint_mask_grids.map(sample_count).transpose()?;
    let hidden_samples = hidden_grids.map(sample_count).transpose()?;
    check_grid_capacity_for_grids(&vectors_per_grid, paint_mask_samples, hidden_samples, 0)?;
    let mut record = Map::new();
    record.insert("total_levels".to_owned(), Value::from(total_levels));
    if let Some(grids) = corner_grids {
        if grids.len() != corner_count {
            return Err(invalid("multires grids do not match mesh corners"));
        }
        let mut values = Map::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let start = face_loop_starts
                .get(face_index)
                .copied()
                .ok_or_else(|| invalid("multires face mapping is invalid"))?;
            let end = start
                .checked_add(face.vertices.len())
                .ok_or_else(|| invalid("multires face corner range overflows"))?;
            let face_grids = grids
                .get(start..end)
                .ok_or_else(|| invalid("multires grids do not cover the mesh faces"))?;
            for (corner, grid) in face_grids.iter().enumerate() {
                if grid.len() != vectors_per_grid[start + corner]
                    || grid.iter().flatten().any(|value| !value.is_finite())
                {
                    return Err(invalid("multires corner grid has an invalid size or value"));
                }
            }
            values.insert(format!("f{}", face.id), json!(face_grids));
        }
        record.insert("values".to_owned(), Value::Object(values));
    } else if let Some(filepath) = external_filepath.filter(|filepath| !filepath.is_empty()) {
        record.insert("external".to_owned(), json!({"filepath":filepath}));
    }
    if grid_levels.is_some() {
        record.insert("grid_levels".to_owned(), json!(levels));
    }
    match (paint_mask_levels, paint_mask_grids) {
        (Some(levels), Some(grids))
            if levels.len() == corner_count && grids.len() == corner_count =>
        {
            let mut masks = Vec::with_capacity(corner_count);
            for (index, (level, grid)) in levels.iter().zip(grids).enumerate() {
                let expected = corner_grid_samples(*level)?;
                if *level > 13
                    || *level > total_levels
                    || grid.len() != expected
                    || grid.iter().any(|value| !value.is_finite())
                {
                    return Err(invalid(format!(
                        "multires paint-mask grid {index} is invalid"
                    )));
                }
                masks.push(json!({"level":level,"values":grid}));
            }
            record.insert("paint_mask".to_owned(), Value::Array(masks));
        }
        (None, None) => {}
        _ => return Err(invalid("multires paint-mask levels and grids do not match")),
    }
    if let Some(grids) = hidden_grids {
        if grids.len() != corner_count
            || grids
                .iter()
                .zip(&vectors_per_grid)
                .any(|(grid, expected)| grid.len() != *expected)
        {
            return Err(invalid(
                "multires hidden grids do not match their displacement grids",
            ));
        }
        record.insert("hidden".to_owned(), json!(grids));
    }
    mesh.attributes
        .insert("multires_displacement".to_owned(), Value::Object(record));
    Ok(())
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "small deterministic multires data checks"
    )]

    use glam::DVec3;
    use serde_json::{Map, json};

    use crate::geom::{BoxParams, Mesh};

    use super::{
        MultiresGridData, add_sample_delta, evaluate_sample, grid_sample_location,
        grid_sample_value, read, read_grid_data, write, write_grid_data,
    };

    #[test]
    fn missing_grids_are_zero_displacement_and_total_level_zero_is_valid() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1}),
        );
        assert!(read(&mesh).unwrap().unwrap().corner_grids.is_none());
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 0}),
        );
        assert_eq!(read(&mesh).unwrap().unwrap().total_levels, 0);
    }

    #[test]
    fn external_capacity_accepts_level_eleven_within_the_budget() {
        let corner_count = 4;
        let json_sidecar_bytes = |level| {
            let samples =
                crate::geom::multires_external::grid_vector_count(level).unwrap() * corner_count;
            u64::try_from(
                u128::try_from(samples).unwrap()
                    * 5
                    * std::mem::size_of::<serde_json::Value>() as u128,
            )
            .unwrap()
        };
        let additional_bytes = |level| {
            crate::geom::multires_external::expected_external_file_size(level, corner_count)
                .unwrap()
                + json_sidecar_bytes(level)
        };
        assert!(
            crate::geom::modifiers::check_multires_grid_capacity(
                corner_count,
                11,
                additional_bytes(11),
            )
            .is_ok()
        );
        assert_eq!(
            crate::geom::modifiers::check_multires_grid_capacity(
                corner_count,
                12,
                additional_bytes(12),
            )
            .unwrap_err()
            .code,
            crate::error::ErrorCode::LimitExceeded
        );
    }

    #[test]
    fn grid_data_roundtrips_masks_hidden_and_per_loop_levels() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let loop_count = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        let grid_levels = (0..loop_count)
            .map(|index| if index == 0 { 1 } else { 2 })
            .collect::<Vec<_>>();
        let corner_grids = grid_levels
            .iter()
            .enumerate()
            .map(|(corner, level)| {
                let samples = crate::geom::multires_external::grid_vector_count(*level).unwrap();
                (0..samples)
                    .map(|sample| [corner as f32, sample as f32, -(sample as f32)])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let paint_mask_levels = (0..loop_count)
            .map(|index| if index == 0 { 3 } else { 2 })
            .collect::<Vec<_>>();
        let paint_mask_grids = paint_mask_levels
            .iter()
            .map(|level| {
                (0..crate::geom::multires_external::grid_vector_count(*level).unwrap())
                    .map(|sample| sample as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let hidden_grids = grid_levels
            .iter()
            .map(|level| {
                (0..crate::geom::multires_external::grid_vector_count(*level).unwrap())
                    .map(|sample| sample % 3 == 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = MultiresGridData {
            total_levels: 3,
            grid_levels: Some(grid_levels),
            corner_grids: Some(corner_grids),
            external_filepath: None,
            paint_mask_levels: Some(paint_mask_levels),
            paint_mask_grids: Some(paint_mask_grids),
            hidden_grids: Some(hidden_grids),
        };
        write_grid_data(&mut mesh, &expected).unwrap();
        let actual = read_grid_data(&mesh).unwrap().unwrap();
        assert_eq!(actual.grid_levels, expected.grid_levels);
        assert_eq!(actual.corner_grids, expected.corner_grids);
        assert_eq!(actual.paint_mask_levels, expected.paint_mask_levels);
        assert_eq!(actual.paint_mask_grids, expected.paint_mask_grids);
        assert_eq!(actual.hidden_grids, expected.hidden_grids);
        assert_eq!(actual.paint_mask_value(0, 1, 2, 2).unwrap(), 22.0);
        // Sculpt level 3 samples the stored level-2 mask through Blender's rounded grid mapping.
        assert_eq!(actual.paint_mask_value(1, 4, 4, 3).unwrap(), 8.0);
        let displacement = read(&mesh).unwrap().unwrap();
        assert_eq!(displacement.paint_mask_value(0, 1, 2, 2).unwrap(), 22.0);
        assert_eq!(
            grid_sample_location(&mesh, &displacement, 0, 3)
                .unwrap()
                .grid_side,
            2
        );
        assert_eq!(
            grid_sample_location(&mesh, &displacement, 1, 8)
                .unwrap()
                .grid_side,
            3
        );
    }

    #[test]
    fn reads_corner_grids_in_mesh_loop_order() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut values = Map::new();
        for face in &mesh.faces {
            let corners = (0..face.vertices.len())
                .map(|_| json!(vec![[0.0, 0.0, 0.0]; 4]))
                .collect::<Vec<_>>();
            values.insert(format!("f{}", face.id), json!(corners));
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": values}),
        );
        let data = read(&mesh).unwrap().unwrap();
        assert_eq!(data.total_levels, 1);
        assert_eq!(data.corner_grids.as_ref().unwrap().len(), 24);
        assert!(
            data.corner_grids
                .unwrap()
                .iter()
                .all(|grid| grid.len() == 4 && grid.iter().all(|v| *v == [0.0; 3]))
        );
    }

    #[test]
    fn reads_nonzero_corner_grid_vectors_without_reordering() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut values = Map::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let corners = (0..face.vertices.len())
                .map(|corner| {
                    let mut grid = vec![[0.0_f32; 3]; 4];
                    for (sample, vector) in grid.iter_mut().enumerate() {
                        *vector = [
                            (face_index * 10 + corner) as f32,
                            sample as f32 + 0.25,
                            -(face_index as f32),
                        ];
                    }
                    json!(grid)
                })
                .collect::<Vec<_>>();
            values.insert(format!("f{}", face.id), json!(corners));
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": values}),
        );
        let grids = read(&mesh).unwrap().unwrap().corner_grids.unwrap();
        assert_eq!(grids[0][0], [0.0, 0.25, 0.0]);
        assert_eq!(grids[1][3], [1.0, 3.25, 0.0]);
        assert_eq!(grids[4][0], [10.0, 0.25, -1.0]);
    }

    #[test]
    fn evaluates_nonzero_quad_grid_and_sculpts_it_in_tangent_space() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut values = Map::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let corners = (0..face.vertices.len())
                .map(|corner| {
                    let mut grid = vec![[0.0_f32; 3]; 4];
                    if face_index == 0 && corner == 1 {
                        grid[3] = [1.0, 2.0, 3.0];
                    }
                    json!(grid)
                })
                .collect::<Vec<_>>();
            values.insert(format!("f{}", face.id), json!(corners));
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": values}),
        );
        let mut data = read(&mesh).unwrap().unwrap();
        let derivative = |_, _, _| Ok((DVec3::X, DVec3::Y));
        assert_eq!(
            evaluate_sample(&mesh, &data, 0, 0.75, 0.25, derivative).unwrap(),
            DVec3::new(1.0, -2.0, 3.0)
        );
        assert_eq!(
            add_sample_delta(
                &mesh,
                &mut data,
                0,
                0.75,
                0.25,
                DVec3::new(1.0, -2.0, 3.0),
                derivative,
            )
            .unwrap(),
            (1, 3)
        );
        assert_eq!(data.corner_grids.as_ref().unwrap()[1][3], [2.0, 4.0, 6.0]);
        write(&mut mesh, &data).unwrap();
        assert_eq!(
            read(&mesh).unwrap().unwrap().corner_grids.unwrap()[1][3],
            [2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn quad_center_averages_the_four_explicitly_selected_grids() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut values = Map::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let corners = (0..face.vertices.len())
                .map(|corner| {
                    let mut grid = vec![[0.0_f32; 3]; 4];
                    if face_index == 0 {
                        grid[0][2] = (corner + 1) as f32;
                    }
                    json!(grid)
                })
                .collect::<Vec<_>>();
            values.insert(format!("f{}", face.id), json!(corners));
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": values}),
        );
        let data = read(&mesh).unwrap().unwrap();
        let displacement = evaluate_sample(&mesh, &data, 0, 0.5, 0.5, |_, _, _| {
            Ok((DVec3::X, DVec3::Y))
        })
        .unwrap();
        assert_eq!(displacement, DVec3::new(0.0, 0.0, 3.125));
    }

    #[test]
    fn nonquad_samples_use_their_face_corner_grid_and_tangent_basis() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let mut corners = vec![vec![[0.0_f32; 3]; 4]; 3];
        corners[0][3] = [1.0, 2.0, 3.0];
        let mut values = Map::new();
        values.insert(format!("f{}", mesh.faces[0].id), json!(corners));
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": values}),
        );
        let data = read(&mesh).unwrap().unwrap();
        let location = grid_sample_location(&mesh, &data, 0, 3).unwrap();
        assert_eq!(
            (
                location.ptex_face_index,
                location.u,
                location.v,
                location.tangent_corner
            ),
            (0, 0.0, 0.0, 0)
        );
        assert_eq!(grid_sample_value(&data, location).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(
            evaluate_sample(&mesh, &data, 0, 0.25, 0.25, |_, _, _| {
                Ok((DVec3::X, DVec3::Y))
            })
            .unwrap(),
            DVec3::new(-2.0, -1.0, 3.0)
        );
    }
}
