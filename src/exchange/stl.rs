use std::{collections::BTreeMap, fs, path::Path};

use serde_json::json;

use super::ExchangeMesh;
use crate::error::{ErrorCode, PotError, Result};

pub(crate) fn export(meshes: &[ExchangeMesh]) -> Result<Vec<u8>> {
    let triangle_count = meshes
        .iter()
        .try_fold(0_usize, |sum, mesh| {
            mesh.faces.iter().try_fold(sum, |sum, face| {
                sum.checked_add(face.len().saturating_sub(2))
            })
        })
        .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "STL triangle count overflow"))?;
    let triangle_count_u32 = u32::try_from(triangle_count).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "STL supports at most u32 triangles",
        )
    })?;
    let body_len = triangle_count
        .checked_mul(50)
        .and_then(|length| length.checked_add(84))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "STL output is too large"))?;
    let mut output = Vec::with_capacity(body_len);
    let mut header = [0_u8; 80];
    let title = b"potter binary STL";
    header[..title.len()].copy_from_slice(title);
    output.extend_from_slice(&header);
    output.extend_from_slice(&triangle_count_u32.to_le_bytes());
    for mesh in meshes {
        for face in &mesh.faces {
            if face.len() < 3 {
                continue;
            }
            for index in 1..face.len() - 1 {
                let points = [
                    point(mesh, face[0])?,
                    point(mesh, face[index])?,
                    point(mesh, face[index + 1])?,
                ];
                let a = glam::Vec3::from_array(points[0]);
                let b = glam::Vec3::from_array(points[1]);
                let c = glam::Vec3::from_array(points[2]);
                let normal = (b - a).cross(c - a).normalize_or_zero();
                for value in normal
                    .to_array()
                    .into_iter()
                    .chain(points.into_iter().flatten())
                {
                    output.extend_from_slice(&value.to_le_bytes());
                }
                output.extend_from_slice(&0_u16.to_le_bytes());
            }
        }
    }
    Ok(output)
}

pub(crate) fn import(path: &Path) -> Result<Vec<ExchangeMesh>> {
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    if bytes.len() < 84 {
        return Err(stl_error("binary STL header is truncated"));
    }
    let count = usize::try_from(u32::from_le_bytes(
        bytes[80..84]
            .try_into()
            .map_err(|_| stl_error("binary STL header is truncated"))?,
    ))
    .map_err(|_| stl_error("binary STL triangle count exceeds platform range"))?;
    let expected = count
        .checked_mul(50)
        .and_then(|length| length.checked_add(84))
        .ok_or_else(|| stl_error("binary STL triangle count is invalid"))?;
    if bytes.len() != expected {
        return Err(PotError::with_details(
            ErrorCode::ImportFailed,
            "binary STL length does not match triangle count",
            json!({ "expected_bytes": expected, "actual_bytes": bytes.len() }),
        ));
    }
    let mut positions = Vec::new();
    let mut faces = Vec::with_capacity(count);
    let mut indices = BTreeMap::<[u32; 3], usize>::new();
    for triangle in bytes[84..].as_chunks::<50>().0 {
        let mut face = Vec::with_capacity(3);
        let mut key = [0_u32; 3];
        for corner in 0..3 {
            let start = 12 + corner * 12;
            let mut position = [0.0_f64; 3];
            for axis in 0..3 {
                let offset = start + axis * 4;
                let value = f32::from_le_bytes(
                    triangle[offset..offset + 4]
                        .try_into()
                        .map_err(|_| stl_error("binary STL vertex is truncated"))?,
                );
                if !value.is_finite() {
                    return Err(stl_error("binary STL contains a non-finite vertex"));
                }
                key[axis] = value.to_bits();
                position[axis] = f64::from(value);
            }
            let index = if let Some(index) = indices.get(&key) {
                *index
            } else {
                let index = positions.len();
                positions.push(position);
                indices.insert(key, index);
                index
            };
            face.push(index);
        }
        faces.push(face);
    }
    if faces.is_empty() {
        return Err(stl_error("binary STL contains no triangles"));
    }
    Ok(vec![ExchangeMesh {
        id: "stl_mesh".to_owned(),
        name: "STL Mesh".to_owned(),
        positions,
        faces,
    }])
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "STL is a float32 interchange format"
)]
fn point(mesh: &ExchangeMesh, index: usize) -> Result<[f32; 3]> {
    let position = mesh.positions.get(index).ok_or_else(|| {
        PotError::new(
            ErrorCode::ExportFailed,
            "STL face references missing vertex",
        )
    })?;
    let mut result = [0.0_f32; 3];
    for (target, value) in result.iter_mut().zip(position) {
        *target = *value as f32;
        if !target.is_finite() {
            return Err(PotError::new(
                ErrorCode::ExportFailed,
                "STL coordinate exceeds float32 range",
            ));
        }
    }
    Ok(result)
}

fn stl_error(message: &str) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

pub(crate) fn imported_graph(
    path: &Path,
    scene_id: String,
) -> Result<crate::exchange::ImportedGraph> {
    let meshes = import(path)?;
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "stl")
}
