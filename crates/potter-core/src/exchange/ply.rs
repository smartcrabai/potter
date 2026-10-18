use std::{collections::HashMap, fs, path::Path};

use glam::DVec3;

use super::{
    ExchangeMesh, ExchangeUvLayer, obj::active_vertex_colors, ply_byte_to_color, ply_color_to_byte,
    shading, single_uv_set,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::color_layers::{ColorDomain, ColorLayer, ColorStore},
};

pub(crate) fn export(meshes: &[ExchangeMesh]) -> Result<Vec<u8>> {
    let splits = meshes.iter().map(split_mesh).collect::<Result<Vec<_>>>()?;
    let vertex_count = splits
        .iter()
        .try_fold(0_usize, |sum, split| sum.checked_add(split.vertices.len()))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "PLY vertex count overflow"))?;
    let face_count = meshes
        .iter()
        .try_fold(0_usize, |sum, mesh| sum.checked_add(mesh.faces.len()))
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "PLY face count overflow"))?;
    // Blender writes shading normals only when a face is shaded (export_normals is off by default).
    let with_normals = meshes.iter().any(|mesh| !shading::is_flat(mesh));
    let normal_properties = if with_normals {
        "property float nx\nproperty float ny\nproperty float nz\n"
    } else {
        ""
    };
    // Blender writes `s t` only when the mesh has a UV layer. A file where some mesh has one gives
    // every mesh `s t`, and the meshes without a layer write zeros.
    let with_uv = meshes.iter().any(|mesh| active_layer(mesh).is_some());
    let uv_properties = if with_uv {
        "property float s\nproperty float t\n"
    } else {
        ""
    };
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {vertex_count}\nproperty float x\nproperty float y\nproperty float z\n{normal_properties}property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n{uv_properties}element face {face_count}\nproperty list uint uint vertex_indices\nend_header\n"
    );
    let mut output = header.into_bytes();
    for (mesh, split) in meshes.iter().zip(&splits) {
        let normals = if with_normals {
            vertex_normals(mesh)?
        } else {
            Vec::new()
        };
        let colors = active_vertex_colors(mesh)?
            .map(|colors| colors.into_iter().map(ply_rgba).collect::<Vec<_>>());
        for &(position, uv) in &split.vertices {
            for value in mesh.positions[position] {
                push_f32(&mut output, value)?;
            }
            if with_normals {
                for value in normals[position] {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
            let rgba = colors
                .as_ref()
                .and_then(|colors| colors.get(position))
                .copied()
                .unwrap_or([153, 153, 153, 255]);
            output.extend_from_slice(&rgba);
            if with_uv {
                for value in uv {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    let mut base = 0_usize;
    for split in &splits {
        for face in &split.faces {
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
            .checked_add(split.vertices.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "PLY vertex index overflow"))?;
    }
    Ok(output)
}

/// The active UV layer of `mesh`, the one PLY writes.
fn active_layer(mesh: &ExchangeMesh) -> Option<&ExchangeUvLayer> {
    mesh.uv_active.and_then(|index| mesh.uv_layers.get(index))
}

/// One mesh as PLY stores it. Without a UV layer the vertices are the positions. With the active
/// layer they are the distinct (position, uv) pairs in first-use order, which is how Blender splits
/// vertices: a position used with two UVs appears twice.
struct Split {
    /// Source position index and `s, t` of each output vertex. Without a layer, `s, t` is zero.
    vertices: Vec<(usize, [f32; 2])>,
    /// Faces as output vertex indices, in the world-winding order `baked_faces` gives.
    faces: Vec<Vec<usize>>,
}

fn split_mesh(mesh: &ExchangeMesh) -> Result<Split> {
    let (faces, _) = shading::baked_faces(mesh)?;
    let Some(layer) = active_layer(mesh) else {
        return Ok(Split {
            vertices: (0..mesh.positions.len())
                .map(|index| (index, [0.0; 2]))
                .collect(),
            faces,
        });
    };
    let uvs = shading::baked_corners(mesh, &layer.corners);
    let mut vertices = Vec::new();
    let mut lookup = HashMap::<(usize, [u32; 2]), usize>::new();
    let mut split_faces = Vec::with_capacity(faces.len());
    for (face, face_uvs) in faces.iter().zip(&uvs) {
        let mut indices = Vec::with_capacity(face.len());
        for (position, uv) in face.iter().zip(face_uvs) {
            let st = blender_st(*uv)?;
            let key = (*position, st.map(f32::to_bits));
            let index = *lookup.entry(key).or_insert_with(|| {
                vertices.push((*position, st));
                vertices.len() - 1
            });
            indices.push(index);
        }
        split_faces.push(indices);
    }
    Ok(Split {
        vertices,
        faces: split_faces,
    })
}

/// Blender's `s, t` for a storage UV. Storage V points down; Blender's PLY writes raw Blender V,
/// which points up, so `t` is flipped.
#[expect(
    clippy::cast_possible_truncation,
    reason = "PLY texture coordinates use float32 by specification"
)]
fn blender_st(uv: [f64; 2]) -> Result<[f32; 2]> {
    let st = [uv[0] as f32, (1.0 - uv[1]) as f32];
    if st.iter().any(|component| !component.is_finite()) {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "PLY texture coordinate exceeds float32 range",
        ));
    }
    Ok(st)
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
    let mut per_vertex_normals = Vec::with_capacity(parsed.vertex_count);
    let mut per_vertex_uvs = Vec::with_capacity(parsed.vertex_count);
    let mut per_vertex_colors = Vec::<[u8; 4]>::with_capacity(parsed.vertex_count);
    for _ in 0..parsed.vertex_count {
        let mut position = [0.0_f64; 3];
        let mut normal = [0.0_f64; 3];
        let mut uv = [0.0_f64; 2];
        let mut color = [0_u8, 0, 0, 255];
        for property in &parsed.vertex_properties {
            let raw = take(&bytes, &mut cursor, scalar_size(&property.scalar_type)?)?;
            if let Some(axis) = uv_axis(&property.name) {
                uv[axis] = read_scalar(raw, &property.scalar_type)?;
                continue;
            }
            if let Some(channel) = color_channel(&property.name).filter(|_| parsed.has_colors) {
                color[channel] = u8::try_from(read_integer(raw, &property.scalar_type)?)
                    .map_err(|_| ply_error("PLY color channel is out of range"))?;
                continue;
            }
            let (target, axis) = match property.name.as_str() {
                "x" => (&mut position, 0),
                "y" => (&mut position, 1),
                "z" => (&mut position, 2),
                "nx" => (&mut normal, 0),
                "ny" => (&mut normal, 1),
                "nz" => (&mut normal, 2),
                _ => continue,
            };
            target[axis] = read_scalar(raw, &property.scalar_type)?;
        }
        if !position.iter().all(|value| value.is_finite()) {
            return Err(ply_error("PLY contains non-finite vertex coordinates"));
        }
        if !uv.iter().all(|value| value.is_finite()) {
            return Err(ply_error("PLY contains non-finite texture coordinates"));
        }
        positions.push(position);
        per_vertex_normals.push(normal);
        // Blender's PLY writes raw V, which points up; Potter stores V down.
        per_vertex_uvs.push([uv[0], 1.0 - uv[1]]);
        if parsed.has_colors {
            per_vertex_colors.push(color);
        }
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
    // Zero vertices and zero faces is an empty mesh, which Potter writes for empty objects.
    // Vertices without faces are a point cloud, which is not a mesh.
    if faces.is_empty() && !positions.is_empty() {
        return Err(ply_error("PLY contains no mesh geometry"));
    }
    let normals = if parsed.has_normals {
        Some(import_corner_normals(&per_vertex_normals, &faces)?)
    } else {
        None
    };
    let (uv_layers, uv_active, uv_render) = if parsed.has_uv {
        single_uv_set(
            faces
                .iter()
                .map(|face| {
                    face.iter()
                        .map(|&index| per_vertex_uvs[index])
                        .collect::<Vec<_>>()
                })
                .collect(),
        )
    } else {
        (Vec::<ExchangeUvLayer>::new(), None, None)
    };
    let colors = if parsed.has_colors {
        ColorStore {
            layers: vec![ColorLayer {
                name: "Col".to_owned(),
                domain: ColorDomain::Point,
                values: per_vertex_colors
                    .iter()
                    .map(|rgba| {
                        [
                            ply_byte_to_color(rgba[0], false),
                            ply_byte_to_color(rgba[1], false),
                            ply_byte_to_color(rgba[2], false),
                            ply_byte_to_color(rgba[3], true),
                        ]
                    })
                    .collect(),
            }],
            active: Some("Col".to_owned()),
            render: Some("Col".to_owned()),
        }
    } else {
        ColorStore::default()
    };
    Ok(vec![ExchangeMesh {
        id: "ply_mesh".to_owned(),
        name: "PLY Mesh".to_owned(),
        positions,
        faces,
        normals,
        mirrored: false,
        singular: false,
        vertex_normals: Vec::new(),
        smooth: Vec::new(),
        uv_layers,
        uv_active,
        uv_render,
        colors,
        vertex_groups: Vec::new(),
        material_slots: Vec::new(),
        face_material_indices: Vec::new(),
    }])
}

struct Header {
    format: String,
    vertex_count: usize,
    face_count: usize,
    vertex_properties: Vec<ScalarProperty>,
    face_properties: Vec<FaceProperty>,
    has_normals: bool,
    has_uv: bool,
    has_colors: bool,
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
    let has_coordinates = has_vertex_properties(&vertex_properties, ["x", "y", "z"]);
    let has_normals = has_vertex_properties(&vertex_properties, ["nx", "ny", "nz"]);
    let has_uv = [0, 1].iter().all(|axis| {
        vertex_properties
            .iter()
            .any(|property| uv_axis(&property.name) == Some(*axis))
    });
    let has_colors = has_vertex_properties(&vertex_properties, ["red", "green", "blue"]);
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
        has_normals,
        has_uv,
        has_colors,
    })
}

fn has_vertex_properties(properties: &[ScalarProperty], names: [&str; 3]) -> bool {
    names
        .iter()
        .all(|name| properties.iter().any(|property| property.name == *name))
}

/// The UV axis a vertex property names: `s`, `u`, or `texture_u` for U; `t`, `v`, or `texture_v`
/// for V.
fn uv_axis(name: &str) -> Option<usize> {
    match name {
        "s" | "u" | "texture_u" => Some(0),
        "t" | "v" | "texture_v" => Some(1),
        _ => None,
    }
}

/// PLY's bytes for a linear RGBA color: RGB sRGB-encoded, alpha linear, each clamped to `[0, 1]`.
fn ply_rgba(color: [f64; 4]) -> [u8; 4] {
    let [red, green, blue, alpha] = color;
    [
        ply_color_to_byte(red, false),
        ply_color_to_byte(green, false),
        ply_color_to_byte(blue, false),
        ply_color_to_byte(alpha, true),
    ]
}

/// The RGBA channel a vertex color property names.
fn color_channel(name: &str) -> Option<usize> {
    match name {
        "red" => Some(0),
        "green" => Some(1),
        "blue" => Some(2),
        "alpha" => Some(3),
        _ => None,
    }
}

/// Blender's per-vertex normals (`mesh.vertex_normals`) in float32. A vertex whose faces cancel
/// out, or that only touches degenerate faces, writes zero.
fn vertex_normals(mesh: &ExchangeMesh) -> Result<Vec<[f32; 3]>> {
    mesh.vertex_normals
        .iter()
        .map(|value| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "PLY normals use float32 by specification"
            )]
            let output = [value[0] as f32, value[1] as f32, value[2] as f32];
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

/// Expands per-vertex PLY normals to face corners. Zero normals stay zero, as Blender writes
/// them; non-finite components are invalid.
fn import_corner_normals(per_vertex: &[[f64; 3]], faces: &[Vec<usize>]) -> Result<Vec<[f64; 3]>> {
    let unit = per_vertex
        .iter()
        .map(|normal| {
            if !normal.iter().all(|value| value.is_finite()) {
                return Err(ply_error("PLY vertex normal is not finite"));
            }
            Ok(DVec3::from_array(*normal).normalize_or_zero().to_array())
        })
        .collect::<Result<Vec<_>>>()?;
    faces
        .iter()
        .flatten()
        .map(|index| {
            unit.get(*index)
                .copied()
                .ok_or_else(|| ply_error("face index is invalid"))
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
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "ply", Vec::new())
}

#[cfg(test)]
mod tests {
    use super::{blender_st, export, import};
    use crate::{
        error::{ErrorCode, PotError, Result},
        exchange::{ExchangeMesh, ExchangeUvLayer},
    };
    use std::fs;

    /// Two triangles sharing positions 1 and 2, whose shared corners carry different UVs.
    fn quad_with_seam_uvs() -> ExchangeMesh {
        ExchangeMesh {
            id: "quad".to_owned(),
            name: "Quad".to_owned(),
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            faces: vec![vec![0, 1, 2], vec![2, 1, 3]],
            normals: Some(vec![[0.0, 0.0, 1.0]; 6]),
            mirrored: false,
            singular: false,
            vertex_normals: Vec::new(),
            smooth: vec![false, false],
            uv_layers: vec![ExchangeUvLayer {
                name: "UVMap".to_owned(),
                corners: vec![
                    vec![[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                    vec![[0.5, 0.0], [1.0, 0.5], [0.5, 0.5]],
                ],
            }],
            uv_active: Some(0),
            uv_render: Some(0),
            colors: Default::default(),
            vertex_groups: Vec::new(),
            material_slots: Vec::new(),
            face_material_indices: Vec::new(),
        }
    }

    fn read_f32(bytes: &[u8], offset: usize) -> Result<f32> {
        let raw = bytes
            .get(offset..offset + 4)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "PLY test read out of range"))?;
        Ok(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    #[test]
    fn splits_vertices_by_position_and_uv_and_writes_flipped_st() -> Result<()> {
        let bytes = export(&[quad_with_seam_uvs()])?;
        let marker = b"end_header\n";
        let header_end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "PLY test has no header"))?
            + marker.len();
        let header = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        // Positions 1 and 2 each appear twice, once per UV: 4 + 2 vertices.
        assert!(header.contains("element vertex 6\n"), "{header}");
        assert!(header.contains("property float s\nproperty float t\n"));
        assert!(!header.contains("property float u"));
        // Vertex 3 is position 2 with the second face's UV (0.5, 0.0). Its t is the flipped storage
        // V, 1 - 0.0. Records are 24 bytes: xyz, colour, then s and t.
        let vertex = header_end + 3 * 24;
        assert_eq!(read_f32(&bytes, vertex)?, 0.0);
        assert_eq!(read_f32(&bytes, vertex + 4)?, 1.0);
        assert_eq!(read_f32(&bytes, vertex + 8)?, 0.0);
        assert_eq!(read_f32(&bytes, vertex + 16)?, 0.5);
        assert_eq!(read_f32(&bytes, vertex + 20)?, 1.0);
        Ok(())
    }

    #[test]
    fn rejects_uv_overflow_after_float32_conversion() -> Result<()> {
        assert_eq!(blender_st([0.25, 0.75])?, [0.25, 0.25]);

        let mut mesh = quad_with_seam_uvs();
        mesh.uv_layers[0].corners[0][0][0] = f64::MAX;
        let error = export(&[mesh]).err().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "out-of-range UV should fail PLY export",
            )
        })?;
        assert_eq!(error.code, ErrorCode::ExportFailed);
        assert_eq!(
            error.message,
            "PLY texture coordinate exceeds float32 range"
        );
        Ok(())
    }

    #[test]
    fn imports_st_as_the_uv_layer_in_storage_convention() -> Result<()> {
        let original = quad_with_seam_uvs();
        let bytes = export(std::slice::from_ref(&original))?;
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), bytes).map_err(|error| PotError::io(&error))?;
        let imported = import(file.path())?;
        let layer = imported[0]
            .uv_layers
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "PLY import has no UV layer"))?;
        assert_eq!(layer.name, "UVMap");
        assert_eq!(imported[0].uv_active, Some(0));
        assert_eq!(imported[0].uv_render, Some(0));
        assert_eq!(layer.corners, original.uv_layers[0].corners);
        Ok(())
    }
}
