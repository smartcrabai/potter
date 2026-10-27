//! Blender 5.2.2 disk point-cache playback for cloth and soft-body surfaces.

use std::{
    collections::BTreeMap,
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use glam::{DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{Id, SceneDoc},
};

const MAGIC: &[u8; 8] = b"BPHYSICS";
const TYPE_SOFT_BODY: u32 = 0;
const TYPE_CLOTH: u32 = 2;
const FLAG_COMPRESS: u32 = 1 << 16;
const FLAG_EXTRA_DATA: u32 = 1 << 17;
const COMPRESSION_ZSTD_FILTERED: u8 = 3;
const COMPRESSION_ZSTD_FAST_DEPRECATED: u8 = 4;
const COMPRESSION_ZSTD_SLOW_DEPRECATED: u8 = 8;
const MAX_CACHE_FRAMES: usize = 1_000_000;
const MAX_CACHE_BYTES: usize = 512 * 1024 * 1024;
const CACHE_LOCATOR: &str = "_potter_cache_locator";

#[derive(Debug)]
struct CacheLocator {
    directory: PathBuf,
    prefix: String,
    stack_index: i32,
}

#[derive(Debug)]
struct CacheFrameResource {
    resource_id: Id,
    hash: String,
}

#[derive(Debug)]
struct CacheFrame {
    frame: i32,
    positions: Vec<[f32; 3]>,
    velocities: Vec<[f32; 3]>,
}

/// True only when every stored file is a complete, decodable Blender 5.2.2 cloth/soft-body cache.
pub(crate) fn supports_import(cache: &Map<String, Value>, system: &str) -> bool {
    if cache.get("is_baked").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    let Ok(locator) = cache_locator(cache, system) else {
        return false;
    };
    let Ok(frames) = cache_files(cache, &locator) else {
        return false;
    };
    if frames.is_empty() {
        return false;
    }
    let mut point_count = None;
    for (frame, path) in frames {
        let Ok(decoded) = read_cache_frame(&path, frame, system, None) else {
            return false;
        };
        if point_count.is_some_and(|expected| expected != decoded.positions.len()) {
            return false;
        }
        point_count = Some(decoded.positions.len());
    }
    true
}

/// Blender point-cache positions are stored in world space; modifiers emit local-space vertices.
pub(crate) fn evaluate_surface(
    settings: &Value,
    system: &str,
    frame: f64,
    world_to_local: DMat4,
    vertex_count: usize,
    subject: &str,
    doc: &SceneDoc,
    project_root: Option<&Path>,
) -> Result<Vec<DVec3>> {
    let cache = settings
        .get("point_cache")
        .and_then(Value::as_object)
        .ok_or_else(|| unsupported(subject, "baked point cache has no cache settings"))?;
    let frames =
        cache_resource_frames(cache, system).map_err(|reason| unsupported(subject, &reason))?;
    if frames.is_empty() {
        return Err(unsupported(
            subject,
            "baked point cache has no project frame resources",
        ));
    }

    if !frame.is_finite() || frame < i32::MIN as f64 || frame > i32::MAX as f64 {
        return Err(PotError::invalid_argument(
            "physics frame must fit Blender's integer frame range",
        ));
    }
    let frame = f64::from(frame as f32);
    if frame < i32::MIN as f64 || frame > i32::MAX as f64 {
        return Err(PotError::invalid_argument(
            "physics frame must fit Blender's integer frame range",
        ));
    }
    let lower_frame = frame.floor() as i32;
    let upper_frame = lower_frame.saturating_add(1);
    let lower = frames.range(..=lower_frame).next_back();
    let upper = frames.range(upper_frame..).next();
    let (first_frame, first_resource, second) = match (lower, upper) {
        (Some((frame, resource)), Some(second)) => (*frame, resource, Some(second)),
        (Some((frame, resource)), None) | (None, Some((frame, resource))) => {
            (*frame, resource, None)
        }
        (None, None) => {
            return Err(unsupported(
                subject,
                "baked point cache has no in-range frames",
            ));
        }
    };

    let first = read_cache_resource_frame(
        doc,
        project_root,
        first_resource,
        first_frame,
        system,
        vertex_count,
        subject,
    )?;
    let positions = if frame.to_bits() == f64::from(first.frame as f32).to_bits() {
        first.positions
    } else if let Some((second_frame, second_resource)) = second {
        let second = read_cache_resource_frame(
            doc,
            project_root,
            second_resource,
            *second_frame,
            system,
            vertex_count,
            subject,
        )?;
        interpolate_positions(&first, &second, frame)
    } else {
        first.positions
    };
    Ok(positions
        .into_iter()
        .map(|position| {
            world_to_local.transform_point3(DVec3::new(
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ))
        })
        .collect())
}

fn unsupported(subject: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({
            "feature_id":"physics.point_cache.is_baked",
            "reason":reason,
            "status":"not_supported",
            "node_id":subject,
        }),
    )
}

fn cache_locator(
    cache: &Map<String, Value>,
    system: &str,
) -> std::result::Result<CacheLocator, String> {
    if !matches!(system, "cloth" | "soft_body") {
        return Err(format!("baked {system} point caches are not supported"));
    }
    let locator = cache
        .get(CACHE_LOCATOR)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "baked cache is not a supported disk-backed cloth or soft-body cache".to_owned()
        })?;
    let directory = locator
        .get("directory")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "baked point cache directory is missing".to_owned())?;
    let prefix = locator
        .get("prefix")
        .and_then(Value::as_str)
        .ok_or_else(|| "baked point cache filename prefix is missing".to_owned())?;
    if prefix.contains('/') || prefix.contains('\\') {
        return Err(
            "baked point cache names containing path separators are not supported".to_owned(),
        );
    }
    let stack_index = locator
        .get("stack_index")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value >= 0)
        .ok_or_else(|| "baked point cache has no valid Blender stack index".to_owned())?;
    let stored_kind = locator
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "baked point cache type is missing".to_owned())?;
    if stored_kind != system {
        return Err("baked point cache type does not match its physics modifier".to_owned());
    }
    Ok(CacheLocator {
        directory: PathBuf::from(directory),
        prefix: prefix.to_owned(),
        stack_index,
    })
}

fn cache_files(
    cache: &Map<String, Value>,
    locator: &CacheLocator,
) -> std::result::Result<BTreeMap<i32, PathBuf>, String> {
    let start = cache_frame(cache, "frame_start", 1)?;
    let end = cache_frame(cache, "frame_end", 250)?;
    if start <= 0 || end < start {
        return Err("baked point cache frame range is invalid".to_owned());
    }
    let range_size = usize::try_from(i64::from(end) - i64::from(start) + 1)
        .map_err(|_| "baked point cache frame range exceeds platform limits".to_owned())?;
    if range_size > MAX_CACHE_FRAMES {
        return Err("baked point cache frame range exceeds the supported frame limit".to_owned());
    }

    let directory = fs::read_dir(&locator.directory)
        .map_err(|error| format!("cannot read baked point cache directory: {error}"))?;
    let suffix = format!("_{:02}.bphys", locator.stack_index);
    let mut frames = BTreeMap::new();
    for entry in directory {
        let entry = entry.map_err(|error| format!("cannot inspect baked point cache: {error}"))?;
        if !entry.path().is_file() {
            continue;
        }
        let Some(filename) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(frame_text) = filename
            .strip_prefix(&locator.prefix)
            .and_then(|name| name.strip_suffix(&suffix))
            .and_then(|name| name.strip_prefix('_'))
        else {
            continue;
        };
        let Ok(frame) = frame_text.parse::<i32>() else {
            continue;
        };
        if frame >= start && frame <= end {
            frames.insert(frame, entry.path());
            if frames.len() > MAX_CACHE_FRAMES {
                return Err("baked point cache contains too many frames".to_owned());
            }
        }
    }
    Ok(frames)
}

fn cache_resource_frames(
    cache: &Map<String, Value>,
    system: &str,
) -> std::result::Result<BTreeMap<i32, CacheFrameResource>, String> {
    let locator = cache
        .get(CACHE_LOCATOR)
        .and_then(Value::as_object)
        .ok_or_else(|| "baked cache has no project frame locator".to_owned())?;
    if locator.get("kind").and_then(Value::as_str) != Some(system) {
        return Err("baked point cache type does not match its physics modifier".to_owned());
    }
    let start = cache_frame(cache, "frame_start", 1)?;
    let end = cache_frame(cache, "frame_end", 250)?;
    if start <= 0 || end < start {
        return Err("baked point cache frame range is invalid".to_owned());
    }
    let range_size = usize::try_from(i64::from(end) - i64::from(start) + 1)
        .map_err(|_| "baked point cache frame range exceeds platform limits".to_owned())?;
    if range_size > MAX_CACHE_FRAMES {
        return Err("baked point cache frame range exceeds the supported frame limit".to_owned());
    }
    let source_frames = locator
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "baked point cache frame references are missing".to_owned())?;
    let mut frames = BTreeMap::new();
    for source in source_frames {
        let frame = source
            .get("frame")
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| "baked point cache frame number is invalid".to_owned())?;
        if frame < start || frame > end {
            return Err("baked point cache frame reference is outside its range".to_owned());
        }
        let resource_id = source
            .get("resource_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "baked point cache frame resource ID is missing".to_owned())?;
        let resource_id = Id::new(resource_id.to_owned())
            .map_err(|error| format!("baked point cache resource ID is invalid: {error}"))?;
        let hash = source
            .get("hash")
            .and_then(Value::as_str)
            .filter(|hash| hash.starts_with("sha256:"))
            .ok_or_else(|| "baked point cache frame hash is missing or invalid".to_owned())?;
        if frames
            .insert(
                frame,
                CacheFrameResource {
                    resource_id,
                    hash: hash.to_owned(),
                },
            )
            .is_some()
        {
            return Err("baked point cache contains duplicate frame references".to_owned());
        }
        if frames.len() > MAX_CACHE_FRAMES {
            return Err("baked point cache contains too many frames".to_owned());
        }
    }
    Ok(frames)
}

fn cache_frame(
    cache: &Map<String, Value>,
    key: &str,
    default: i32,
) -> std::result::Result<i32, String> {
    let Some(value) = cache.get(key) else {
        return Ok(default);
    };
    let Some(number) = value.as_f64() else {
        return Err(format!("baked point cache {key} must be an integer"));
    };
    if !number.is_finite()
        || number.fract() != 0.0
        || number < i32::MIN as f64
        || number > i32::MAX as f64
    {
        return Err(format!("baked point cache {key} must be an integer"));
    }
    Ok(number as i32)
}

fn read_cache_resource_frame(
    doc: &SceneDoc,
    project_root: Option<&Path>,
    resource: &CacheFrameResource,
    frame: i32,
    system: &str,
    expected_points: usize,
    subject: &str,
) -> Result<CacheFrame> {
    let (bytes, actual_hash) =
        crate::eval::read_cache_resource(doc, project_root, &resource.resource_id)?;
    if actual_hash != resource.hash {
        return Err(PotError::with_details(
            ErrorCode::AssetChanged,
            format!(
                "baked point cache frame `{}` changed after import",
                resource.resource_id
            ),
            json!({
                "node_id":subject,
                "resource_id":resource.resource_id,
                "expected_hash":resource.hash,
                "actual_hash":actual_hash,
            }),
        ));
    }
    read_cache_frame_bytes(&bytes, frame, system, Some(expected_points))
        .map_err(|reason| unsupported(subject, &reason))
}

fn read_cache_frame(
    path: &Path,
    frame: i32,
    system: &str,
    expected_points: Option<usize>,
) -> std::result::Result<CacheFrame, String> {
    let file = fs::File::open(path)
        .map_err(|error| format!("cannot read baked point cache frame: {error}"))?;
    let file_size = usize::try_from(
        file.metadata()
            .map_err(|error| format!("cannot inspect baked point cache frame: {error}"))?
            .len(),
    )
    .map_err(|_| "baked point cache frame exceeds platform limits".to_owned())?;
    if file_size > MAX_CACHE_BYTES {
        return Err("baked point cache frame exceeds the supported file size".to_owned());
    }
    let mut bytes = Vec::with_capacity(file_size);
    file.take(MAX_CACHE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read baked point cache frame: {error}"))?;
    read_cache_frame_bytes(&bytes, frame, system, expected_points)
}

fn read_cache_frame_bytes(
    bytes: &[u8],
    frame: i32,
    system: &str,
    expected_points: Option<usize>,
) -> std::result::Result<CacheFrame, String> {
    if bytes.len() > MAX_CACHE_BYTES {
        return Err("baked point cache frame exceeds the supported file size".to_owned());
    }
    let mut cursor = Cursor::new(bytes);
    if take(bytes, &mut cursor, MAGIC.len())? != MAGIC {
        return Err("baked point cache frame has an invalid header".to_owned());
    }
    let type_flags = read_u32(bytes, &mut cursor)?;
    let cache_type = type_flags & 0xffff;
    let flags = type_flags & 0xffff_0000;
    let expected_type = if system == "cloth" {
        TYPE_CLOTH
    } else {
        TYPE_SOFT_BODY
    };
    if cache_type != expected_type || flags & !(FLAG_COMPRESS | FLAG_EXTRA_DATA) != 0 {
        return Err("baked point cache type or flags are unsupported".to_owned());
    }
    let points = usize::try_from(read_u32(bytes, &mut cursor)?)
        .map_err(|_| "baked point cache vertex count exceeds platform limits".to_owned())?;
    let data_types = read_u32(bytes, &mut cursor)?;
    let expected_types = if system == "cloth" { 22 } else { 6 };
    let point_data_size = if system == "cloth" { 36 } else { 24 };
    if points == 0
        || points
            .checked_mul(point_data_size)
            .is_none_or(|size| size > MAX_CACHE_BYTES)
        || expected_points.is_some_and(|expected| points != expected)
        || data_types != expected_types
    {
        return Err("baked point cache topology or data types are unsupported".to_owned());
    }
    let mut positions = vec![[0.0; 3]; points];
    let mut velocities = vec![[0.0; 3]; points];
    if flags & FLAG_COMPRESS != 0 {
        for data_type in 0..8 {
            if data_types & (1 << data_type) == 0 {
                continue;
            }
            let item_size = cache_item_size(data_type)?;
            let payload = read_compressed_block(bytes, &mut cursor, points, item_size)?;
            match data_type {
                1 => decode_vec3s(&payload, &mut positions)?,
                2 => decode_vec3s(&payload, &mut velocities)?,
                _ => {}
            }
        }
    } else {
        for point in 0..points {
            for data_type in 0..8 {
                if data_types & (1 << data_type) == 0 {
                    continue;
                }
                let item_size = cache_item_size(data_type)?;
                let item = take(bytes, &mut cursor, item_size)?;
                match data_type {
                    1 => positions[point] = read_vec3(item)?,
                    2 => velocities[point] = read_vec3(item)?,
                    _ => {}
                }
            }
        }
    }
    if flags & FLAG_EXTRA_DATA != 0 {
        while cursor.position() < bytes.len() as u64 {
            let extra_type = read_u32(bytes, &mut cursor)? & 0xffff;
            let count = usize::try_from(read_u32(bytes, &mut cursor)?).map_err(|_| {
                "baked point cache extra data count exceeds platform limits".to_owned()
            })?;
            let item_size = match (system, extra_type, count) {
                ("cloth", 2, 1) => 12,
                _ => return Err("baked point cache extra data is unsupported".to_owned()),
            };
            if flags & FLAG_COMPRESS != 0 {
                let _ = read_compressed_block(bytes, &mut cursor, count, item_size)?;
            } else {
                let size = count
                    .checked_mul(item_size)
                    .ok_or_else(|| "baked point cache extra data size overflows".to_owned())?;
                let _ = take(bytes, &mut cursor, size)?;
            }
        }
    }
    if cursor.position() != bytes.len() as u64 {
        return Err("baked point cache frame contains trailing data".to_owned());
    }
    Ok(CacheFrame {
        frame,
        positions,
        velocities,
    })
}

fn cache_item_size(data_type: u32) -> std::result::Result<usize, String> {
    match data_type {
        1 | 2 | 4 | 6 => Ok(12),
        3 => Ok(16),
        _ => Err("baked point cache contains an unsupported data type".to_owned()),
    }
}

fn read_compressed_block(
    bytes: &[u8],
    cursor: &mut Cursor<&[u8]>,
    items: usize,
    item_size: usize,
) -> std::result::Result<Vec<u8>, String> {
    let expected_size = items
        .checked_mul(item_size)
        .filter(|size| *size <= MAX_CACHE_BYTES)
        .ok_or_else(|| "baked point cache data block exceeds the supported size".to_owned())?;
    let compression = take(bytes, cursor, 1)?[0];
    if compression == 0 {
        return Ok(take(bytes, cursor, expected_size)?.to_vec());
    }
    if !matches!(
        compression,
        COMPRESSION_ZSTD_FILTERED
            | COMPRESSION_ZSTD_FAST_DEPRECATED
            | COMPRESSION_ZSTD_SLOW_DEPRECATED
    ) {
        return Err("baked point cache compression is unsupported".to_owned());
    }
    let compressed_size = usize::try_from(read_u32(bytes, cursor)?)
        .map_err(|_| "baked point cache compressed block exceeds platform limits".to_owned())?;
    let compressed = take(bytes, cursor, compressed_size)?;
    if compressed_size == 0 {
        return Err("baked point cache compressed block is empty".to_owned());
    }
    let mut decoder = ruzstd::decoding::StreamingDecoder::new_with_max_window_size(
        compressed,
        u64::try_from(MAX_CACHE_BYTES)
            .map_err(|_| "baked point cache size exceeds platform limits".to_owned())?,
    )
    .map_err(|error| format!("cannot decode baked point cache zstd block: {error}"))?;
    let mut decoded = Vec::with_capacity(expected_size);
    decoder
        .by_ref()
        .take(expected_size as u64 + 1)
        .read_to_end(&mut decoded)
        .map_err(|error| format!("cannot decode baked point cache zstd block: {error}"))?;
    if decoded.len() != expected_size {
        return Err("baked point cache decompressed block has the wrong size".to_owned());
    }
    if compression == COMPRESSION_ZSTD_FILTERED {
        unfilter_transpose_delta(&decoded, items, item_size)
    } else {
        Ok(decoded)
    }
}

fn unfilter_transpose_delta(
    filtered: &[u8],
    items: usize,
    item_size: usize,
) -> std::result::Result<Vec<u8>, String> {
    let size = items
        .checked_mul(item_size)
        .filter(|size| *size == filtered.len())
        .ok_or_else(|| "baked point cache filtered block has the wrong size".to_owned())?;
    let mut decoded = vec![0; size];
    for byte_index in 0..item_size {
        let mut previous = 0_u8;
        for item_index in 0..items {
            let value = filtered[byte_index * items + item_index].wrapping_add(previous);
            decoded[item_index * item_size + byte_index] = value;
            previous = value;
        }
    }
    Ok(decoded)
}

fn decode_vec3s(bytes: &[u8], output: &mut [[f32; 3]]) -> std::result::Result<(), String> {
    if bytes.len() != output.len() * 12 {
        return Err("baked point cache vector block has the wrong size".to_owned());
    }
    let (items, _) = bytes.as_chunks::<12>();
    for (vector, item) in output.iter_mut().zip(items) {
        *vector = read_vec3(item)?;
    }
    Ok(())
}

fn read_vec3(bytes: &[u8]) -> std::result::Result<[f32; 3], String> {
    let x = bytes
        .get(0..4)
        .ok_or_else(|| "truncated point cache vector".to_owned())?;
    let y = bytes
        .get(4..8)
        .ok_or_else(|| "truncated point cache vector".to_owned())?;
    let z = bytes
        .get(8..12)
        .ok_or_else(|| "truncated point cache vector".to_owned())?;
    Ok([
        f32::from_le_bytes(x.try_into().map_err(|_| "invalid point cache vector")?),
        f32::from_le_bytes(y.try_into().map_err(|_| "invalid point cache vector")?),
        f32::from_le_bytes(z.try_into().map_err(|_| "invalid point cache vector")?),
    ])
}

fn interpolate_positions(first: &CacheFrame, second: &CacheFrame, frame: f64) -> Vec<[f32; 3]> {
    let delta_frame = (second.frame - first.frame) as f32;
    let t = (frame as f32 - first.frame as f32) / delta_frame;
    let t2 = t * t;
    let t3 = t2 * t;
    first
        .positions
        .iter()
        .zip(&first.velocities)
        .zip(&second.positions)
        .zip(&second.velocities)
        .map(|(((x1, v1), x2), v2)| {
            let mut result = [0.0; 3];
            for axis in 0..3 {
                let v1 = v1[axis] * delta_frame;
                let v2 = v2[axis] * delta_frame;
                let delta = x1[axis] - x2[axis];
                let a = v1 + v2 + 2.0 * delta;
                let b = -2.0 * v1 - v2 - 3.0 * delta;
                result[axis] = a * t3 + b * t2 + v1 * t + x1[axis];
            }
            result
        })
        .collect()
}

fn read_u32(bytes: &[u8], cursor: &mut Cursor<&[u8]>) -> std::result::Result<u32, String> {
    let value = take(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes(
        value
            .try_into()
            .map_err(|_| "truncated point cache integer".to_owned())?,
    ))
}

fn take<'a>(
    bytes: &'a [u8],
    cursor: &mut Cursor<&[u8]>,
    length: usize,
) -> std::result::Result<&'a [u8], String> {
    let start = usize::try_from(cursor.position())
        .map_err(|_| "point cache offset exceeds platform limits".to_owned())?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| "point cache offset overflow".to_owned())?;
    let value = bytes
        .get(start..end)
        .ok_or_else(|| "truncated baked point cache frame".to_owned())?;
    cursor.set_position(end as u64);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_interpolation_uses_cache_velocities_and_frame_gap() {
        let first = CacheFrame {
            frame: 1,
            positions: vec![[0.0, 0.0, 0.0]],
            velocities: vec![[1.0, 0.0, 0.0]],
        };
        let second = CacheFrame {
            frame: 3,
            positions: vec![[2.0, 0.0, 0.0]],
            velocities: vec![[0.0, 0.0, 0.0]],
        };

        assert_eq!(
            interpolate_positions(&first, &second, 2.0),
            [[1.25, 0.0, 0.0]]
        );
        assert_eq!(
            interpolate_positions(&first, &second, 1.0),
            [[0.0, 0.0, 0.0]]
        );
        assert_eq!(
            interpolate_positions(&first, &second, 3.0),
            [[2.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn filtered_delta_round_trip_matches_blender_byte_planes() -> Result<(), String> {
        let source = [1_u8, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12];
        let mut filtered = Vec::new();
        for byte_index in 0..4 {
            let mut previous = 0_u8;
            for item_index in 0..3 {
                let value = source[item_index * 4 + byte_index];
                filtered.push(value.wrapping_sub(previous));
                previous = value;
            }
        }
        let decoded = unfilter_transpose_delta(&filtered, 3, 4)?;
        assert_eq!(decoded.as_slice(), source.as_slice());
        Ok(())
    }
}
