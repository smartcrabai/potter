// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Native reader and writer for Blender's external multires displacement (BCDF) files.
//!
//! The supported file has one `CD_MDISPS` mesh layer and uses little-endian fields.
//! - File magic: bytes 0..4 are `BCDF`.
//! - Endian flag: byte 4 is 0 for little-endian.
//! - Version: byte 5 is 0.
//! - Subversion: byte 6 is 0.
//! - File-header pad: byte 7 is 0.
//! - File-header struct size: bytes 8..12 contain 20.
//! - File type: bytes 12..16 contain 1 (`CDF_TYPE_MESH`).
//! - Layer count: bytes 16..20 contain 1.
//! - Mesh-header struct size: bytes 20..24 contain 4.
//! - Layer-header struct size: bytes 24..28 contain 88.
//! - Layer datatype: bytes 28..32 contain 0 (`float`).
//! - Layer data size: bytes 32..40 contain the little-endian u64 payload size.
//! - Layer type: bytes 40..44 contain 19 (`CD_MDISPS`).
//! - Layer name: bytes 44..108 are empty in Blender's probe file.
//! - Layer-header padding: bytes 108..112 are zero.
//! - Layer data: each corner has `s*s` consecutive `[f32; 3]` vectors, row-major `y*s+x`.
//! - Grid side: `s = (1 << (total_levels - 1)) + 1` for levels starting at 1.

use crate::error::{ErrorCode, PotError, Result};

const FILE_HEADER_SIZE: usize = 20;
const MESH_HEADER_SIZE: usize = 4;
const LAYER_HEADER_SIZE: usize = 88;
const DATA_OFFSET: usize = FILE_HEADER_SIZE + MESH_HEADER_SIZE + LAYER_HEADER_SIZE;
const FILE_MAGIC: &[u8; 4] = b"BCDF";
const FILE_TYPE_MESH: u32 = 1;
const LAYER_TYPE_MDISPS: u32 = 19;
const LAYER_DATATYPE_FLOAT: u32 = 0;

pub(crate) fn expected_external_file_size(total_levels: u32, corner_count: usize) -> Result<u64> {
    expected_external_sizes(total_levels, corner_count).map(|(_, file_size)| file_size)
}

fn expected_external_sizes(total_levels: u32, corner_count: usize) -> Result<(u64, u64)> {
    let grid_len = grid_vector_count(total_levels)?;
    let data_size = corner_count
        .checked_mul(grid_len)
        .and_then(|count| count.checked_mul(12))
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "BCDF data size overflows"))?;
    let data_size = u64::try_from(data_size).map_err(|_| {
        PotError::new(
            ErrorCode::InvalidArgument,
            "BCDF data size exceeds the layer size field",
        )
    })?;
    let file_size = usize::try_from(data_size)
        .ok()
        .and_then(|size| DATA_OFFSET.checked_add(size))
        .and_then(|size| u64::try_from(size).ok())
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "BCDF file size overflows"))?;
    Ok((data_size, file_size))
}

/// Decode a Blender external multires displacement file into one grid per corner.
pub(crate) fn read_external_grids(
    bytes: &[u8],
    total_levels: u32,
    corner_count: usize,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let grid_len = grid_vector_count(total_levels).map_err(|error| {
        PotError::new(
            ErrorCode::ImportFailed,
            format!("invalid Blender multires external grid level: {}", error.message),
        )
    })?;
    let (expected_data_size_u64, expected_file_size_u64) =
        expected_external_sizes(total_levels, corner_count).map_err(|error| {
            PotError::new(
                ErrorCode::ImportFailed,
                format!(
                    "invalid Blender multires external grid size: {}",
                    error.message
                ),
            )
        })?;
    let expected_file_size = usize::try_from(expected_file_size_u64)
        .map_err(|_| import_error("external file size exceeds this platform's address space"))?;

    if bytes.len() < DATA_OFFSET {
        return Err(import_error("file is truncated before the layer data"));
    }
    if bytes.get(0..4) != Some(FILE_MAGIC.as_slice()) {
        return Err(import_error("invalid BCDF magic"));
    }
    if bytes[4] != 0 {
        return Err(import_error("unsupported BCDF byte order"));
    }
    if bytes[5] != 0 || bytes[6] != 0 {
        return Err(import_error("unsupported BCDF version or subversion"));
    }
    if bytes[7] != 0 {
        return Err(import_error("invalid BCDF header padding"));
    }
    if read_u32(bytes, 8)? != FILE_HEADER_SIZE as u32 {
        return Err(import_error("invalid BCDF file-header struct size"));
    }
    if read_u32(bytes, 12)? != FILE_TYPE_MESH {
        return Err(import_error("BCDF file is not a mesh custom-data file"));
    }
    if read_u32(bytes, 16)? != 1 {
        return Err(import_error("BCDF file must contain exactly one layer"));
    }
    if read_u32(bytes, FILE_HEADER_SIZE)? != MESH_HEADER_SIZE as u32 {
        return Err(import_error("invalid BCDF mesh-header struct size"));
    }

    let layer = FILE_HEADER_SIZE + MESH_HEADER_SIZE;
    if read_u32(bytes, layer)? != LAYER_HEADER_SIZE as u32 {
        return Err(import_error("invalid BCDF layer-header struct size"));
    }
    if read_u32(bytes, layer + 4)? != LAYER_DATATYPE_FLOAT {
        return Err(import_error("BCDF displacement layer datatype is not float"));
    }
    if read_u64(bytes, layer + 8)? != expected_data_size_u64 {
        return Err(import_error(format!(
            "BCDF displacement data size does not match {corner_count} corners at level {total_levels}"
        )));
    }
    if read_u32(bytes, layer + 16)? != LAYER_TYPE_MDISPS {
        return Err(import_error("BCDF layer is not CD_MDISPS"));
    }
    if bytes.get(layer + 20..layer + 84) != Some(&[0; 64]) {
        return Err(import_error("BCDF CD_MDISPS layer name is not empty"));
    }
    if bytes.get(layer + 84..layer + LAYER_HEADER_SIZE) != Some(&[0; 4]) {
        return Err(import_error("invalid BCDF layer-header padding"));
    }
    if bytes.len() != expected_file_size {
        return Err(import_error(format!(
            "BCDF file length is {}, expected {expected_file_size}",
            bytes.len()
        )));
    }

    let mut grids = Vec::with_capacity(corner_count);
    let mut offset = DATA_OFFSET;
    for _ in 0..corner_count {
        let mut grid = Vec::with_capacity(grid_len);
        for _ in 0..grid_len {
            let x = read_f32(bytes, offset)?;
            let y = read_f32(bytes, offset + 4)?;
            let z = read_f32(bytes, offset + 8)?;
            grid.push([x, y, z]);
            offset += 12;
        }
        grids.push(grid);
    }
    Ok(grids)
}

/// Encode one row-major displacement grid for each corner into Blender's BCDF format.
pub(crate) fn write_external_grids(
    grids: &[Vec<[f32; 3]>],
    total_levels: u32,
) -> Result<Vec<u8>> {
    let grid_len = grid_vector_count(total_levels)?;
    for (corner, grid) in grids.iter().enumerate() {
        if grid.len() != grid_len {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                format!(
                    "multires corner {corner} has {} vectors; level {total_levels} requires {grid_len}",
                    grid.len()
                ),
            ));
        }
        if grid.iter().flatten().any(|component| !component.is_finite()) {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "multires displacement components must be finite",
            ));
        }
    }

    let data_size = grids
        .len()
        .checked_mul(grid_len)
        .and_then(|count| count.checked_mul(12))
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "BCDF data size overflows"))?;
    let data_size_u64 = u64::try_from(data_size).map_err(|_| {
        PotError::new(
            ErrorCode::InvalidArgument,
            "BCDF data size exceeds the layer size field",
        )
    })?;
    let file_size = DATA_OFFSET.checked_add(data_size).ok_or_else(|| {
        PotError::new(ErrorCode::InvalidArgument, "BCDF file size overflows")
    })?;

    let mut bytes = Vec::with_capacity(file_size);
    bytes.extend_from_slice(FILE_MAGIC);
    bytes.extend_from_slice(&[0; 4]);
    push_u32(&mut bytes, FILE_HEADER_SIZE as u32);
    push_u32(&mut bytes, FILE_TYPE_MESH);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, MESH_HEADER_SIZE as u32);
    push_u32(&mut bytes, LAYER_HEADER_SIZE as u32);
    push_u32(&mut bytes, LAYER_DATATYPE_FLOAT);
    bytes.extend_from_slice(&data_size_u64.to_le_bytes());
    push_u32(&mut bytes, LAYER_TYPE_MDISPS);
    bytes.extend_from_slice(&[0; 68]);
    for grid in grids {
        for vector in grid {
            for component in vector {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

pub(crate) fn grid_vector_count(total_levels: u32) -> Result<usize> {
    let invalid = || {
        PotError::new(
            ErrorCode::InvalidArgument,
            "multires total_levels must produce a representable grid side",
        )
    };
    let shift = total_levels.checked_sub(1).ok_or_else(invalid)?;
    let side = 1_usize
        .checked_shl(shift)
        .and_then(|side| side.checked_add(1))
        .ok_or_else(invalid)?;
    side.checked_mul(side).ok_or_else(invalid)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let field = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| import_error("file is truncated in a 32-bit header field"))?;
    let array: [u8; 4] = field
        .try_into()
        .map_err(|_| import_error("invalid 32-bit BCDF field"))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let field = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| import_error("file is truncated in a 64-bit header field"))?;
    let array: [u8; 8] = field
        .try_into()
        .map_err(|_| import_error("invalid 64-bit BCDF field"))?;
    Ok(u64::from_le_bytes(array))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32> {
    let field = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| import_error("file is truncated in the displacement data"))?;
    let array: [u8; 4] = field
        .try_into()
        .map_err(|_| import_error("invalid BCDF displacement component"))?;
    let value = f32::from_le_bytes(array);
    if !value.is_finite() {
        return Err(import_error("BCDF displacement component is not finite"));
    }
    Ok(value)
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn import_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "small deterministic BCDF fixture checks")]

    use super::{DATA_OFFSET, read_external_grids, write_external_grids};
    use crate::error::ErrorCode;

    const PROBE: &[u8] = include_bytes!("../../tests/fixtures/multires/ext_probe.btx");

    #[test]
    fn probe_roundtrips_exactly_and_decodes_24_nine_vector_grids() {
        let grids = read_external_grids(PROBE, 2, 24).unwrap();
        assert_eq!(grids.len(), 24);
        assert!(grids.iter().all(|grid| grid.len() == 9));
        assert_eq!(write_external_grids(&grids, 2).unwrap(), PROBE);
    }

    #[test]
    fn nonzero_grids_roundtrip_in_corner_and_sample_order() {
        let grids = (0..24)
            .map(|corner| {
                (0..9)
                    .map(|sample| {
                        [
                            corner as f32 + sample as f32 / 10.0,
                            -(sample as f32) / 3.0,
                            (corner * sample) as f32 / 7.0,
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let bytes = write_external_grids(&grids, 2).unwrap();
        assert_eq!(read_external_grids(&bytes, 2, 24).unwrap(), grids);
    }

    #[test]
    fn rejects_non_finite_displacement_components() {
        let mut bytes = PROBE.to_vec();
        bytes[DATA_OFFSET..DATA_OFFSET + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            read_external_grids(&bytes, 2, 24).unwrap_err().code,
            ErrorCode::ImportFailed
        );

        let mut grids = vec![vec![[0.0; 3]; 9]; 24];
        grids[0][0][1] = f32::INFINITY;
        assert_eq!(
            write_external_grids(&grids, 2).unwrap_err().code,
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn rejects_truncated_external_file() {
        let error = read_external_grids(&PROBE[..PROBE.len() - 1], 2, 24).unwrap_err();
        assert_eq!(error.code, ErrorCode::ImportFailed);
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bytes = PROBE.to_vec();
        bytes[0] = b'X';
        assert_eq!(
            read_external_grids(&bytes, 2, 24).unwrap_err().code,
            ErrorCode::ImportFailed
        );
    }

    #[test]
    fn rejects_wrong_corner_total() {
        assert_eq!(
            read_external_grids(PROBE, 2, 23).unwrap_err().code,
            ErrorCode::ImportFailed
        );
    }

    #[test]
    fn rejects_wrong_total_level() {
        assert_eq!(
            read_external_grids(PROBE, 3, 24).unwrap_err().code,
            ErrorCode::ImportFailed
        );
    }
}
