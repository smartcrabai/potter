use std::{fs, path::Path};

use glam::DVec3;

use super::ExchangeMesh;
use crate::error::{ErrorCode, PotError, Result};

pub(crate) fn export(meshes: &[ExchangeMesh]) -> Result<Vec<u8>> {
    let vertex_count = meshes
        .iter()
        .try_fold(0_usize, |sum, mesh| sum.checked_add(mesh.positions.len()))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "PLY vertex count overflow"))?;
    let face_count = meshes
        .iter()
        .try_fold(0_usize, |sum, mesh| sum.checked_add(mesh.faces.len()))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "PLY face count overflow"))?;
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {vertex_count}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nproperty float u\nproperty float v\nelement face {face_count}\nproperty list uint uint vertex_indices\nend_header\n"
    );
    let mut output = header.into_bytes();
    for mesh in meshes {
        let normals = vertex_normals(mesh)?;
        for (index, position) in mesh.positions.iter().enumerate() {
            for value in position {
                push_f32(&mut output, *value)?;
            }
            for value in normals[index] {
                output.extend_from_slice(&value.to_le_bytes());
            }
            output.extend_from_slice(&[153, 153, 153, 255]);
            output.extend_from_slice(&0.0_f32.to_le_bytes());
            output.extend_from_slice(&0.0_f32.to_le_bytes());
        }
    }
    let mut base = 0_usize;
    for mesh in meshes {
        for face in &mesh.faces {
            let count = u32::try_from(face.len())
                .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "PLY face is too large"))?;
            output.extend_from_slice(&count.to_le_bytes());
            for index in face {
                let absolute = base.checked_add(*index).ok_or_else(|| {
                    PotError::new(ErrorCode::ExportFailed, "PLY vertex index overflow")
                })?;
                let index = u32::try_from(absolute).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "PLY supports at most u32 vertices",
                    )
                })?;
                output.extend_from_slice(&index.to_le_bytes());
            }
        }
        base = base
            .checked_add(mesh.positions.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "PLY vertex index overflow"))?;
    }
    Ok(output)
}

pub(crate) fn import(path: &Path) -> Result<Vec<ExchangeMesh>> {
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    let marker = b"end_header";
    let marker_end = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .and_then(|position| position.checked_add(marker.len()))
        .ok_or_else(|| ply_error("PLY header is incomplete"))?;
    let header_end = if bytes.get(marker_end..marker_end + 2) == Some(b"\r\n") {
        marker_end + 2
    } else if bytes.get(marker_end) == Some(&b'\n') {
        marker_end + 1
    } else {
        return Err(ply_error("PLY header terminator is invalid"));
    };
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| ply_error("PLY header is not UTF-8"))?;
    let parsed = parse_header(header)?;
    if parsed.format != "binary_little_endian" {
        return Err(ply_error("only binary little-endian PLY is supported"));
    }
    let mut cursor = header_end;
    let mut positions = Vec::with_capacity(parsed.vertex_count);
    for _ in 0..parsed.vertex_count {
        let mut position = [0.0_f64; 3];
        for property in &parsed.vertex_properties {
            let raw = take(&bytes, &mut cursor, scalar_size(&property.scalar_type)?)?;
            let axis = match property.name.as_str() {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                _ => None,
            };
            if let Some(axis) = axis {
                position[axis] = read_scalar(raw, &property.scalar_type)?;
            }
        }
        if !position.iter().all(|value| value.is_finite()) {
            return Err(ply_error("PLY contains non-finite vertex coordinates"));
        }
        positions.push(position);
    }
    let mut faces = Vec::with_capacity(parsed.face_count);
    for _ in 0..parsed.face_count {
        let mut face = None;
        for property in &parsed.face_properties {
            match property {
                FaceProperty::Scalar(scalar_type) => {
                    take(&bytes, &mut cursor, scalar_size(scalar_type)?)?;
                }
                FaceProperty::List(count_type, index_type, name) => {
                    let count = read_integer(
                        take(&bytes, &mut cursor, scalar_size(count_type)?)?,
                        count_type,
                    )?;
                    let count = usize::try_from(count)
                        .map_err(|_| ply_error("PLY list count is invalid"))?;
                    if count > positions.len() {
                        return Err(ply_error("PLY face list is larger than the vertex array"));
                    }
                    let mut indices = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = read_integer(
                            take(&bytes, &mut cursor, scalar_size(index_type)?)?,
                            index_type,
                        )?;
                        let index = usize::try_from(value)
                            .map_err(|_| ply_error("PLY face index is invalid"))?;
                        if index >= positions.len() {
                            return Err(ply_error("PLY face references a missing vertex"));
                        }
                        indices.push(index);
                    }
                    if name == "vertex_indices" || name == "vertex_index" {
                        face = Some(indices);
                    }
                }
            }
        }
        if let Some(face) = face.filter(|face| face.len() >= 3) {
            faces.push(face);
        }
    }
    if cursor != bytes.len() {
        return Err(ply_error("PLY has trailing or malformed binary data"));
    }
    if positions.is_empty() || faces.is_empty() {
        return Err(ply_error("PLY contains no mesh geometry"));
    }
    Ok(vec![ExchangeMesh {
        id: "ply_mesh".to_owned(),
        name: "PLY Mesh".to_owned(),
        positions,
        faces,
    }])
}

struct Header {
    format: String,
    vertex_count: usize,
    face_count: usize,
    vertex_properties: Vec<ScalarProperty>,
    face_properties: Vec<FaceProperty>,
}

struct ScalarProperty {
    scalar_type: String,
    name: String,
}

enum FaceProperty {
    Scalar(String),
    List(String, String, String),
}

fn parse_header(header: &str) -> Result<Header> {
    let mut format = None;
    let mut element = String::new();
    let mut vertex_count = 0;
    let mut face_count = 0;
    let mut vertex_properties = Vec::new();
    let mut face_properties = Vec::new();
    for line in header.lines().skip(1) {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("format") => format = fields.next().map(str::to_owned),
            Some("element") => {
                fields.next().unwrap_or_default().clone_into(&mut element);
                let count = fields
                    .next()
                    .ok_or_else(|| ply_error("PLY element count is missing"))?
                    .parse::<usize>()
                    .map_err(|_| ply_error("PLY element count is invalid"))?;
                match element.as_str() {
                    "vertex" => vertex_count = count,
                    "face" => face_count = count,
                    _ => {}
                }
            }
            Some("property") if element == "vertex" => {
                let scalar_type = fields
                    .next()
                    .ok_or_else(|| ply_error("PLY vertex property is invalid"))?;
                if scalar_size(scalar_type).is_err() {
                    return Err(ply_error(
                        "list-valued PLY vertex properties are unsupported",
                    ));
                }
                let name = fields
                    .next()
                    .ok_or_else(|| ply_error("PLY vertex property name is missing"))?;
                vertex_properties.push(ScalarProperty {
                    scalar_type: scalar_type.to_owned(),
                    name: name.to_owned(),
                });
            }
            Some("property") if element == "face" => {
                let first = fields
                    .next()
                    .ok_or_else(|| ply_error("PLY face property is invalid"))?;
                if first == "list" {
                    let count_type = fields
                        .next()
                        .ok_or_else(|| ply_error("PLY face count type is missing"))?;
                    let index_type = fields
                        .next()
                        .ok_or_else(|| ply_error("PLY face index type is missing"))?;
                    let name = fields
                        .next()
                        .ok_or_else(|| ply_error("PLY face property name is missing"))?;
                    scalar_size(count_type)?;
                    scalar_size(index_type)?;
                    face_properties.push(FaceProperty::List(
                        count_type.to_owned(),
                        index_type.to_owned(),
                        name.to_owned(),
                    ));
                } else {
                    scalar_size(first)?;
                    face_properties.push(FaceProperty::Scalar(first.to_owned()));
                }
            }
            _ => {}
        }
    }
    let has_coordinates = ["x", "y", "z"].iter().all(|name| {
        vertex_properties
            .iter()
            .any(|property| property.name == *name)
    });
    let has_indices = face_properties.iter().any(|property| matches!(property, FaceProperty::List(_, _, name) if name == "vertex_indices" || name == "vertex_index"));
    if !has_coordinates || !has_indices {
        return Err(ply_error(
            "PLY is missing required vertex or face properties",
        ));
    }
    Ok(Header {
        format: format.ok_or_else(|| ply_error("PLY format is missing"))?,
        vertex_count,
        face_count,
        vertex_properties,
        face_properties,
    })
}

fn vertex_normals(mesh: &ExchangeMesh) -> Result<Vec<[f32; 3]>> {
    let mut normals = vec![DVec3::ZERO; mesh.positions.len()];
    for face in &mesh.faces {
        if face.len() < 3 {
            continue;
        }
        let a = DVec3::from_array(
            *mesh
                .positions
                .get(face[0])
                .ok_or_else(|| ply_error("face index is invalid"))?,
        );
        let b = DVec3::from_array(
            *mesh
                .positions
                .get(face[1])
                .ok_or_else(|| ply_error("face index is invalid"))?,
        );
        let c = DVec3::from_array(
            *mesh
                .positions
                .get(face[2])
                .ok_or_else(|| ply_error("face index is invalid"))?,
        );
        let normal = (b - a).cross(c - a);
        for index in face {
            let target = normals
                .get_mut(*index)
                .ok_or_else(|| ply_error("face index is invalid"))?;
            *target += normal;
        }
    }
    normals
        .into_iter()
        .map(|normal| {
            let value = normal.normalize_or_zero();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "PLY normals use float32 by specification"
            )]
            let output = [value.x as f32, value.y as f32, value.z as f32];
            if output.iter().any(|component| !component.is_finite()) {
                return Err(PotError::new(
                    ErrorCode::ExportFailed,
                    "PLY normal exceeds float32 range",
                ));
            }
            Ok(output)
        })
        .collect()
}

fn push_f32(output: &mut Vec<u8>, value: f64) -> Result<()> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "PLY positions use float32 by specification"
    )]
    let value = value as f32;
    if !value.is_finite() {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "PLY coordinate exceeds float32 range",
        ));
    }
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn scalar_size(kind: &str) -> Result<usize> {
    match kind {
        "char" | "int8" | "uchar" | "uint8" => Ok(1),
        "short" | "int16" | "ushort" | "uint16" => Ok(2),
        "int" | "int32" | "uint" | "uint32" | "float" => Ok(4),
        "double" | "int64" | "uint64" => Ok(8),
        _ => Err(ply_error("unsupported PLY scalar type")),
    }
}

fn read_scalar(bytes: &[u8], kind: &str) -> Result<f64> {
    let value = match kind {
        "char" | "int8" => f64::from(i8::from_le_bytes([bytes[0]])),
        "uchar" | "uint8" => f64::from(bytes[0]),
        "short" | "int16" => f64::from(i16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        )),
        "ushort" | "uint16" => f64::from(u16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        )),
        "int" | "int32" => f64::from(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        )),
        "uint" | "uint32" => f64::from(u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        )),
        "float" => f64::from(f32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        )),
        "double" => f64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY scalar is truncated"))?,
        ),
        _ => return Err(ply_error("unsupported PLY coordinate type")),
    };
    Ok(value)
}

fn read_integer(bytes: &[u8], kind: &str) -> Result<i64> {
    let value = match kind {
        "char" | "int8" => i64::from(i8::from_le_bytes([bytes[0]])),
        "uchar" | "uint8" => i64::from(bytes[0]),
        "short" | "int16" => i64::from(i16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        )),
        "ushort" | "uint16" => i64::from(u16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        )),
        "int" | "int32" => i64::from(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        )),
        "uint" | "uint32" => i64::from(u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        )),
        "int64" => i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        ),
        "uint64" => i64::try_from(u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| ply_error("PLY integer is truncated"))?,
        ))
        .map_err(|_| ply_error("PLY integer exceeds supported range"))?,
        _ => return Err(ply_error("PLY list values must use integer types")),
    };
    Ok(value)
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, size: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(size)
        .ok_or_else(|| ply_error("PLY data length overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| ply_error("PLY data is truncated"))?;
    *cursor = end;
    Ok(value)
}

fn ply_error(message: &str) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

pub(crate) fn imported_graph(
    path: &Path,
    scene_id: String,
) -> Result<crate::exchange::ImportedGraph> {
    let meshes = import(path)?;
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "ply")
}
