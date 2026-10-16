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
    pub corner_grids: Option<Vec<Vec<[f32; 3]>>>,
    pub external_filepath: Option<String>,
    pub(super) grid_side: usize,
    face_loop_starts: Vec<usize>,
    face_ptex_starts: Vec<usize>,
    ptex_face_corners: Vec<(usize, usize)>,
    pub(super) loop_face_corners: Vec<(usize, usize)>,
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
    let corner_grids = match record.get("values") {
        Some(_) if total_levels == 0 => {
            return Err(invalid(
                "multires_displacement.values requires a positive total_levels",
            ));
        }
        Some(raw_values) => {
            let grid_vectors = super::super::multires_external::grid_vector_count(total_levels)?;
            check_grid_capacity(mesh, grid_vectors)?;
            Some(read_values(mesh, raw_values, grid_vectors)?)
        }
        None => None,
    };
    let (
        face_loop_starts,
        face_ptex_starts,
        ptex_face_corners,
        loop_face_corners,
    ) = ptex_mapping(mesh)?;
    let grid_side = if total_levels == 0 {
        0
    } else {
        let grid_vectors = super::super::multires_external::grid_vector_count(total_levels)?;
        let shift = total_levels
            .checked_sub(1)
            .ok_or_else(|| invalid("multires displacement grid level is invalid"))?;
        let side = 1_usize
            .checked_shl(shift)
            .and_then(|side| side.checked_add(1))
            .ok_or_else(|| {
                PotError::new(ErrorCode::LimitExceeded, "multires grid side overflows")
            })?;
        if side.checked_mul(side) != Some(grid_vectors) {
            return Err(invalid("multires grid side does not match its vector count"));
        }
        side
    };
    Ok(Some(DisplacementData {
        total_levels,
        corner_grids,
        external_filepath,
        grid_side,
        face_loop_starts,
        face_ptex_starts,
        ptex_face_corners,
        loop_face_corners,
    }))
}

fn check_grid_capacity(mesh: &Mesh, grid_vectors: usize) -> Result<()> {
    let corner_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    check_grid_capacity_for_corners(
        corner_count.ok_or_else(|| invalid("mesh corner count overflows"))?,
        grid_vectors,
        0,
    )
}

pub(super) fn check_grid_capacity_for_corners(
    corner_count: usize,
    grid_vectors: usize,
    additional_bytes: u64,
) -> Result<()> {
    let corner_count = u128::try_from(corner_count).ok();
    let grid_bytes = corner_count
        .and_then(|corners| {
            let grid_vectors = u128::try_from(grid_vectors).ok()?;
            corners.checked_mul(grid_vectors)
        })
        .and_then(|vectors| vectors.checked_mul(size_of::<[f32; 3]>() as u128))
        .and_then(|bytes| {
            let descriptors = corner_count?.checked_mul(size_of::<Vec<[f32; 3]>>() as u128)?;
            bytes.checked_add(descriptors)
        })
        .and_then(|bytes| bytes.checked_add(u128::from(additional_bytes)))
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "multires displacement size overflows")
        })?;
    if grid_bytes > super::MAX_SUBDIVISION_MEMORY_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "multires displacement exceeds the 8-GiB memory budget",
        ));
    }
    Ok(())
}

fn read_values(mesh: &Mesh, raw_values: &Value, grid_vectors: usize) -> Result<Vec<Vec<[f32; 3]>>> {
    let values = raw_values
        .as_object()
        .ok_or_else(|| invalid("multires_displacement.values must be keyed by face ID"))?;
    if values.len() != mesh.faces.len() {
        return Err(invalid("multires_displacement.values must contain every mesh face"));
    }
    let corner_count = mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let corner_count = corner_count.ok_or_else(|| invalid("mesh corner count overflows"))?;
    let mut grids = Vec::with_capacity(corner_count);
    for face in &mesh.faces {
        let key = format!("f{}", face.id);
        let face_values = values
            .get(&key)
            .and_then(Value::as_array)
            .filter(|corners| corners.len() == face.vertices.len())
            .ok_or_else(|| invalid(format!("multires_displacement.values.{key} has invalid corners")))?;
        for (corner, raw_grid) in face_values.iter().enumerate() {
            let grid = raw_grid
                .as_array()
                .filter(|grid| grid.len() == grid_vectors)
                .ok_or_else(|| invalid(format!("multires grid {key}[{corner}] has an invalid size")))?;
            let mut vectors = Vec::with_capacity(grid_vectors);
            for (sample, vector) in grid.iter().enumerate() {
                let components = vector
                    .as_array()
                    .filter(|components| components.len() == 3)
                    .ok_or_else(|| invalid(format!("multires vector {key}[{corner}][{sample}] is invalid")))?;
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
        PotError::new(ErrorCode::LimitExceeded, "multires mesh corner count overflows")
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
            PotError::new(ErrorCode::LimitExceeded, "multires mesh corner count overflows")
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
    if !u.is_finite() || !v.is_finite() || !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v)
    {
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
    let grid_side = data.grid_side;
    if grid_side == 0 {
        return Err(invalid("multires displacement has no grid level"));
    }
    let maximum = (grid_side - 1) as f32;
    let x = (grid_u * maximum).round() as usize;
    let y = (grid_v * maximum).round() as usize;
    if x >= grid_side || y >= grid_side {
        return Err(invalid("multires ptex sample is outside its corner grid"));
    }
    let grid_index = data
        .face_loop_starts
        .get(face_index)
        .copied()
        .and_then(|start| start.checked_add(grid_corner))
        .ok_or_else(|| invalid("multires loop-corner mapping is invalid"))?;
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
    let side = data.grid_side;
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
        loop_index,
        vector_index,
        ptex_face_index,
        u,
        v,
        tangent_corner: if face.vertices.len() == 4 {
            corner
        } else {
            0
        },
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
    Ok((face_index, if face.vertices.len() == 4 { 0 } else { corner }))
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
        PotError::new(ErrorCode::LimitExceeded, "multires ptex resolution overflows")
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
    let grid_len = data
        .grid_side
        .checked_mul(data.grid_side)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "multires grid size overflows"))?;
    if grid_len == 0 {
        return Err(invalid("multires displacement has no grid level"));
    }
    if data.corner_grids.is_none() {
        check_grid_capacity(mesh, grid_len)?;
        let mut grids = Vec::with_capacity(data.loop_face_corners.len());
        grids.resize_with(data.loop_face_corners.len(), || vec![[0.0; 3]; grid_len]);
        data.corner_grids = Some(grids);
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
    [x.normalize_or_zero(), y.normalize_or_zero(), du.cross(dv).normalize_or_zero()]
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
        return Err(invalid("multires sculpt sample has an invalid tangent frame"));
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
    let mut displacement = tangent_to_object(du, dv, location.grid_corner, tangent);
    let corner_count = mesh.faces[location.face_index].vertices.len();
    match average {
        BoundaryAverage::All => {
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
    let other_corner =
        (location.face_corner as isize + corner_delta + num_corners as isize) as usize
            % num_corners;
    let other_ptex_face = if num_corners == 4 {
        data.face_ptex_starts[location.face_index]
    } else {
        data.face_ptex_starts[location.face_index] + other_corner
    };
    let (u, v) = ptex_from_grid(num_corners, other_corner, grid_u, grid_v);
    let other_location = grid_location(mesh, data, other_ptex_face, u, v)?;
    let (tangent, _) = read_grid(data, other_location)?;
    let (du, dv) = derivatives(other_ptex_face, u, v)?;
    Ok(tangent_to_object(
        du,
        dv,
        other_location.grid_corner,
        tangent,
    ))
}

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
        tangent_corner: location.grid_corner,
    };
    let mut value = grid_sample_value(data, grid_sample)?;
    for axis in 0..3 {
        value[axis] += tangent_delta[axis];
    }
    set_grid_sample_value(mesh, data, grid_sample, value)?;
    Ok((location.grid_index, location.vector_index))
}

pub(super) fn write(mesh: &mut Mesh, data: &DisplacementData) -> Result<()> {
    let grids = data
        .corner_grids
        .as_ref()
        .ok_or_else(|| invalid("multires displacement grids are unavailable"))?;
    let mut values = Map::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let start = data
            .face_loop_starts
            .get(face_index)
            .copied()
            .ok_or_else(|| invalid("multires face mapping is invalid"))?;
        let end = start
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid("multires face corner range overflows"))?;
        let face_grids = grids
            .get(start..end)
            .ok_or_else(|| invalid("multires grids do not cover the mesh faces"))?;
        values.insert(format!("f{}", face.id), json!(face_grids));
    }
    let record = mesh
        .attributes
        .get_mut("multires_displacement")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid("multires_displacement must be an object"))?;
    record.insert("values".to_owned(), Value::Object(values));
    Ok(())
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "small deterministic multires data checks")]

    use glam::DVec3;
    use serde_json::{Map, json};

    use crate::geom::{BoxParams, Mesh, normals::corner_normal_domain};

    use super::{add_sample_delta, evaluate_sample, grid_sample_location, grid_sample_value, read, write};

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
    fn external_capacity_accounts_for_grid_storage_and_sidecar_bytes() {
        let accepted_size =
            crate::geom::multires_external::expected_external_file_size(13, 19).unwrap();
        assert!(
            crate::geom::modifiers::check_multires_grid_capacity(19, 13, accepted_size).is_ok()
        );
        let rejected_size =
            crate::geom::multires_external::expected_external_file_size(13, 40).unwrap();
        assert_eq!(
            crate::geom::modifiers::check_multires_grid_capacity(40, 13, rejected_size)
                .unwrap_err()
                .code,
            crate::error::ErrorCode::LimitExceeded
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
    fn mixed_sharp_input_uses_corner_normals_but_all_sharp_is_face_domain() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut edge_flags = (0..mesh.edges.len())
            .map(|_| json!({"sharp": false, "seam": false}))
            .collect::<Vec<_>>();
        edge_flags[0] = json!({"sharp": true, "seam": false});
        mesh.attributes
            .insert("blender_edge_flags".to_owned(), json!(edge_flags));
        assert!(corner_normal_domain(&mesh).unwrap());

        mesh.attributes.insert(
            "blender_edge_flags".to_owned(),
            json!(
                (0..mesh.edges.len())
                    .map(|_| json!({"sharp": true, "seam": false}))
                    .collect::<Vec<_>>()
            ),
        );
        assert!(!corner_normal_domain(&mesh).unwrap());
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
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({"total_levels": 1, "values": {"f1": corners}}),
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
