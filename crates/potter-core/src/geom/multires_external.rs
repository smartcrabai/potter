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
//! - Layer data: each corner's row-major `[f32; 3]` MDisps vectors are concatenated in corner order.
//!
//! Blender 5.2.2 reads every corner of this layer at the modifier's top level
//! (`multires_ensure_external_read`), so a file holds exactly `grid_vector_count(total_levels)`
//! vectors per corner. A coarser corner has no external form. The probe that pins this is
//! `tests/fixtures/multires/gen_external_level_probe.py`.

use crate::error::{ErrorCode, PotError, Result};

const FILE_HEADER_SIZE: usize = 20;
const MESH_HEADER_SIZE: usize = 4;
const LAYER_HEADER_SIZE: usize = 88;
const DATA_OFFSET: usize = FILE_HEADER_SIZE + MESH_HEADER_SIZE + LAYER_HEADER_SIZE;
const FILE_MAGIC: &[u8; 4] = b"BCDF";
const FILE_TYPE_MESH: u32 = 1;
const LAYER_TYPE_MDISPS: u32 = 19;
const LAYER_DATATYPE_FLOAT: u32 = 0;

/// Number of `[f32; 3]` vectors in one corner grid at `total_levels`: `(2^(L-1) + 1)^2`.
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

/// File size of the sidecar for `corner_count` corners at `total_levels`. Export and import both
/// size the file with this helper, and `read_external_grids` checks the same size.
pub(crate) fn expected_external_file_size(total_levels: u32, corner_count: usize) -> Result<u64> {
    let (_, file_size) = expected_external_sizes(total_levels, corner_count)?;
    Ok(file_size)
}

/// Rejects a corner whose level differs from `total_levels`. Blender cannot read such a corner
/// from an external sidecar, so the caller gets `code` instead of a file Blender would misread.
pub(crate) fn require_top_level_corners(
    levels: &[u32],
    total_levels: u32,
    code: ErrorCode,
) -> Result<()> {
    match levels
        .iter()
        .enumerate()
        .find(|(_, level)| **level != total_levels)
    {
        None => Ok(()),
        Some((corner, level)) => Err(PotError::new(
            code,
            format!(
                "multires corner {corner} is at level {level}; Blender's external sidecar stores every corner at total level {total_levels}"
            ),
        )),
    }
}

fn expected_external_sizes(total_levels: u32, corner_count: usize) -> Result<(u64, u64)> {
    let grid_len = grid_vector_count(total_levels)?;
    let data_size = corner_count
        .checked_mul(grid_len)
        .and_then(|samples| samples.checked_mul(12))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "BCDF data size overflows"))?;
    let file_size = u64::try_from(DATA_OFFSET)
        .ok()
        .and_then(|header| header.checked_add(data_size))
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "BCDF file size overflows"))?;
    Ok((data_size, file_size))
}

/// Decode a sidecar for `corner_count` corners at `total_levels` into one grid per corner.
pub(crate) fn read_external_grids(
    bytes: &[u8],
    total_levels: u32,
    corner_count: usize,
) -> Result<Vec<Vec<[f32; 3]>>> {
    let (expected_data_size, expected_file_size) = expected_external_sizes(total_levels, corner_count)
        .map_err(|error| {
            import_error(format!(
                "invalid Blender multires external grid size: {}",
                error.message
            ))
        })?;
    let grid_len = grid_vector_count(total_levels).map_err(|error| {
        import_error(format!(
            "invalid Blender multires external grid level: {}",
            error.message
        ))
    })?;

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
        return Err(import_error(
            "BCDF displacement layer datatype is not float",
        ));
    }
    if read_u64(bytes, layer + 8)? != expected_data_size {
        return Err(import_error(format!(
            "BCDF displacement data size does not match {corner_count} corners at total level {total_levels}"
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
    if u64::try_from(bytes.len()).ok() != Some(expected_file_size) {
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

/// Encode one grid per corner in corner order. Every grid must hold `grid_vector_count(total_levels)`
/// finite vectors.
pub(crate) fn write_external_grids(grids: &[Vec<[f32; 3]>], total_levels: u32) -> Result<Vec<u8>> {
    let grid_len = grid_vector_count(total_levels)?;
    let (data_size, file_size) = expected_external_sizes(total_levels, grids.len())?;
    for (corner, grid) in grids.iter().enumerate() {
        if grid.len() != grid_len {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                format!(
                    "multires corner {corner} has {} vectors; total level {total_levels} requires {grid_len}",
                    grid.len()
                ),
            ));
        }
        if grid
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "multires displacement components must be finite",
            ));
        }
    }

    let file_capacity = usize::try_from(file_size).map_err(|_| {
        PotError::new(
            ErrorCode::InvalidArgument,
            "BCDF file size exceeds this platform's address space",
        )
    })?;
    let mut bytes = Vec::with_capacity(file_capacity);
    bytes.extend_from_slice(FILE_MAGIC);
    bytes.extend_from_slice(&[0; 4]);
    push_u32(&mut bytes, FILE_HEADER_SIZE as u32);
    push_u32(&mut bytes, FILE_TYPE_MESH);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, MESH_HEADER_SIZE as u32);
    push_u32(&mut bytes, LAYER_HEADER_SIZE as u32);
    push_u32(&mut bytes, LAYER_DATATYPE_FLOAT);
    bytes.extend_from_slice(&data_size.to_le_bytes());
    push_u32(&mut bytes, LAYER_TYPE_MDISPS);
    bytes.extend_from_slice(&[0; 68]);
    for grid in grids {
        for vector in grid {
            for component in vector {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
    }
    debug_assert_eq!(u64::try_from(bytes.len()).ok(), Some(file_size));
    Ok(bytes)
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
    #![expect(
        clippy::unwrap_used,
        reason = "small deterministic BCDF fixture checks"
    )]

    use super::{
        DATA_OFFSET, expected_external_file_size, grid_vector_count, read_external_grids,
        require_top_level_corners, write_external_grids,
    };
    use crate::error::ErrorCode;

    const PROBE: &[u8] = include_bytes!("../../../../tests/fixtures/multires/ext_probe.btx");
    const BLENDER_UNIFORM_LEVEL2: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multires/probe_uniform_level2.btx"
    ));
    const PYTHON_UNIFORM_LEVEL1: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multires/probe_uniform_level1_total2.btx"
    ));
    const PYTHON_MIXED_LEVELS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/multires/probe_mixed_levels_total2.btx"
    ));

    #[test]
    fn probe_roundtrips_exactly_and_decodes_24_nine_vector_grids() {
        let grids = read_external_grids(PROBE, 2, 24).unwrap();
        assert_eq!(grids.len(), 24);
        assert!(grids.iter().all(|grid| grid.len() == 9));
        assert_eq!(write_external_grids(&grids, 2).unwrap(), PROBE);
    }

    #[test]
    fn blender_written_top_level_probe_has_the_pinned_layout() {
        assert_eq!(BLENDER_UNIFORM_LEVEL2.len(), 2704);
        assert_eq!(
            u64::try_from(BLENDER_UNIFORM_LEVEL2.len()).unwrap(),
            expected_external_file_size(2, 24).unwrap()
        );
        let grids = read_external_grids(BLENDER_UNIFORM_LEVEL2, 2, 24).unwrap();
        assert_eq!(write_external_grids(&grids, 2).unwrap(), BLENDER_UNIFORM_LEVEL2);
        // Blender reads corner i at DATA_OFFSET + 108 * i, i.e. 12 * 9 bytes per corner.
        assert_eq!(DATA_OFFSET + 108 * 23, 2596);
    }

    #[test]
    fn non_top_level_probes_are_rejected_at_the_decoder() {
        // Blender fails with "failed to read multires displacement" on both files; the decoder
        // rejects them on the size check before reading any corner.
        assert_eq!(PYTHON_UNIFORM_LEVEL1.len(), 1264);
        assert_eq!(PYTHON_MIXED_LEVELS.len(), 1984);
        for bytes in [PYTHON_UNIFORM_LEVEL1, PYTHON_MIXED_LEVELS] {
            assert_eq!(
                read_external_grids(bytes, 2, 24).unwrap_err().code,
                ErrorCode::ImportFailed
            );
        }
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
    fn writer_rejects_a_coarser_corner_grid() {
        let mut grids = vec![vec![[0.0; 3]; 9]; 24];
        grids[1] = vec![[0.0; 3]; grid_vector_count(1).unwrap()];
        assert_eq!(
            write_external_grids(&grids, 2).unwrap_err().code,
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn mixed_corner_levels_are_refused_before_any_bytes_are_written() {
        assert!(require_top_level_corners(&[2, 2, 2], 2, ErrorCode::UnrepresentableFeature).is_ok());
        let error = require_top_level_corners(&[2, 1, 2], 2, ErrorCode::UnrepresentableFeature)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::UnrepresentableFeature);
        assert!(error.message.contains("corner 1 is at level 1"));
        assert_eq!(
            require_top_level_corners(&[1, 1], 2, ErrorCode::ImportFailed)
                .unwrap_err()
                .code,
            ErrorCode::ImportFailed
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
