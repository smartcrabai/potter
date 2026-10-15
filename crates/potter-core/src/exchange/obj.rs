use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path};

use glam::DVec3;
use serde_json::json;

use super::{ExchangeMesh, Loss, shading};
use crate::error::{ErrorCode, PotError, Result};

/// Blender's default OBJ axes (forward -Z, up Y): Potter's Z-up `(x, y, z)` is written as
/// `(x, z, -y)`, so Blender's default importer reads the same placement.
fn to_obj_axes(value: [f64; 3]) -> [f64; 3] {
    [value[0], value[2], -value[1]]
}

/// Inverse of `to_obj_axes`.
fn from_obj_axes(value: [f64; 3]) -> [f64; 3] {
    [value[0], -value[2], value[1]]
}

pub(crate) fn export(meshes: &[ExchangeMesh]) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut object = String::from("mtllib scene.mtl\n");
    let mut base = 0_usize;
    let mut normal_base = 0_usize;
    for mesh in meshes {
        object.push_str("o ");
        object.push_str(&obj_name(&mesh.name));
        object.push('\n');
        for position in &mesh.positions {
            let [x, y, z] = to_obj_axes(*position);
            writeln!(object, "v {x:.9} {y:.9} {z:.9}")
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        }
        let (faces, normals) = shading::baked_faces(mesh);
        // Blender writes zero normals for singular transforms.
        let normals = if mesh.singular {
            vec![[0.0; 3]; normals.len()]
        } else {
            normals
        };
        let normals = normals.into_iter().map(to_obj_axes).collect::<Vec<_>>();
        let (normal_lines, corner_ids) = normal_table(&normals);
        for line in &normal_lines {
            object.push_str(line);
            object.push('\n');
        }
        object.push_str("usemtl potter_default\n");
        let smooth = shading::smooth_faces(mesh);
        let mut smooth_state = None::<bool>;
        let mut corner = 0_usize;
        for (face, is_smooth) in faces.iter().zip(smooth) {
            let first_corner = corner;
            corner = first_corner.checked_add(face.len()).ok_or_else(|| {
                PotError::new(ErrorCode::ExportFailed, "OBJ corner count overflow")
            })?;
            if face.len() < 3 {
                continue;
            }
            if smooth_state != Some(is_smooth) {
                writeln!(object, "s {}", u8::from(is_smooth))
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
                smooth_state = Some(is_smooth);
            }
            object.push('f');
            for (offset, index) in face.iter().enumerate() {
                let vertex = base
                    .checked_add(*index)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow")
                    })?;
                write!(object, " {vertex}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
                let normal = first_corner
                    .checked_add(offset)
                    .and_then(|corner| corner_ids.get(corner))
                    .and_then(|local| normal_base.checked_add(*local))
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::ExportFailed,
                            "OBJ normal count does not match faces",
                        )
                    })?;
                write!(object, "//{normal}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
            }
            object.push('\n');
        }
        base = base
            .checked_add(mesh.positions.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow"))?;
        normal_base = normal_base
            .checked_add(normal_lines.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ normal index overflow"))?;
    }
    let material = "newmtl potter_default\nKd 0.6 0.6 0.6\nKa 0 0 0\nKs 0 0 0\nNs 18\n";
    Ok((object.into_bytes(), material.as_bytes().to_vec()))
}

/// Blender's `vn` table for one object: unique normals in first-use order, as `vn` lines, plus
/// each corner's 0-based index into that table.
fn normal_table(normals: &[[f64; 3]]) -> (Vec<String>, Vec<usize>) {
    let mut lines = Vec::<String>::new();
    let mut lookup = BTreeMap::<String, usize>::new();
    let mut corner_ids = Vec::with_capacity(normals.len());
    for normal in normals {
        let line = format!("vn {:.6} {:.6} {:.6}", normal[0], normal[1], normal[2]);
        let index = *lookup.entry(line.clone()).or_insert_with(|| {
            lines.push(line);
            lines.len() - 1
        });
        corner_ids.push(index);
    }
    (lines, corner_ids)
}

pub(crate) fn import(path: &Path) -> Result<(Vec<ExchangeMesh>, Vec<Loss>)> {
    let text = fs::read_to_string(path).map_err(|error| PotError::io(&error))?;
    let mut positions = Vec::<[f64; 3]>::new();
    let mut normals = Vec::<DVec3>::new();
    let mut meshes = Vec::<ExchangeMesh>::new();
    let mut losses = Vec::<Loss>::new();
    let mut current = ExchangeMesh {
        id: "object_0".to_owned(),
        name: "Object".to_owned(),
        positions: Vec::new(),
        faces: Vec::new(),
        normals: None,
        mirrored: false,
        singular: false,
        vertex_normals: None,
        smooth: None,
    };
    // Normal named by each face corner of `current`, in face order.
    let mut corner_normals = Vec::<Option<DVec3>>::new();
    // Whether an `s` group is on at the current line, and whether any face of `current` was read
    // while one was on.
    let mut group_on = false;
    let mut smoothed = false;
    let mut remap = BTreeMap::<usize, usize>::new();
    for (line_number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else {
            continue;
        };
        match kind {
            "v" => {
                let values = parse_floats(words, 3, line_number)?;
                positions.push(from_obj_axes([values[0], values[1], values[2]]));
            }
            "vn" => {
                let values = parse_floats(words, 3, line_number)?;
                // Zero normals are valid: Blender writes them for degenerate faces and singular nodes.
                let normal = DVec3::from_array(from_obj_axes([values[0], values[1], values[2]]));
                if !normal.is_finite() {
                    return Err(parse_error(line_number));
                }
                normals.push(normal);
            }
            "o" | "g" => {
                let name = words.collect::<Vec<_>>().join(" ");
                if !current.faces.is_empty() || !current.positions.is_empty() {
                    finish_object(&mut current, &corner_normals, smoothed, &mut losses);
                    meshes.push(current);
                    current = ExchangeMesh {
                        id: format!("object_{}", meshes.len()),
                        name: if name.is_empty() {
                            format!("Object {}", meshes.len())
                        } else {
                            name.clone()
                        },
                        positions: Vec::new(),
                        faces: Vec::new(),
                        normals: None,
                        mirrored: false,
                        singular: false,
                        vertex_normals: None,
                        smooth: None,
                    };
                    corner_normals.clear();
                    smoothed = false;
                    remap.clear();
                } else if !name.is_empty() {
                    current.name = name;
                }
            }
            "s" => {
                group_on = words
                    .next()
                    .is_some_and(|group| group != "0" && group != "off");
            }
            "f" => {
                let mut face = Vec::new();
                let mut face_normals = Vec::new();
                for token in words {
                    let raw_index = token
                        .split('/')
                        .next()
                        .ok_or_else(|| parse_error(line_number))?;
                    let source_index = resolve_index(raw_index, positions.len(), line_number)?;
                    let position = *positions
                        .get(source_index)
                        .ok_or_else(|| parse_error(line_number))?;
                    let local_index = *remap.entry(source_index).or_insert_with(|| {
                        current.positions.push(position);
                        current.positions.len() - 1
                    });
                    face.push(local_index);
                    let normal = match token.split('/').nth(2).filter(|value| !value.is_empty()) {
                        Some(value) => {
                            let index = resolve_index(value, normals.len(), line_number)?;
                            Some(*normals.get(index).ok_or_else(|| parse_error(line_number))?)
                        }
                        None => None,
                    };
                    face_normals.push(normal);
                }
                if face.len() >= 3 {
                    current.faces.push(face);
                    corner_normals.extend(face_normals);
                    smoothed |= group_on;
                }
            }
            _ => {}
        }
    }
    if !current.faces.is_empty() || !current.positions.is_empty() {
        finish_object(&mut current, &corner_normals, smoothed, &mut losses);
        meshes.push(current);
    }
    if meshes.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::ImportFailed,
            "OBJ contains no faces",
            json!({ "path": path.display().to_string() }),
        ));
    }
    Ok((meshes, losses))
}

/// Resolves an OBJ index. Positive values count from 1; negative values count back from the end
/// of the `count` entries read so far.
fn resolve_index(token: &str, count: usize, line: usize) -> Result<usize> {
    let index = token.parse::<i64>().map_err(|_| parse_error(line))?;
    match index.cmp(&0) {
        std::cmp::Ordering::Greater => usize::try_from(index - 1).map_err(|_| parse_error(line)),
        std::cmp::Ordering::Less => count
            .checked_sub(usize::try_from(index.unsigned_abs()).map_err(|_| parse_error(line))?)
            .ok_or_else(|| parse_error(line)),
        std::cmp::Ordering::Equal => Err(parse_error(line)),
    }
}

/// Sets `mesh.normals` when every face corner names a normal. Partial normals are dropped and
/// smooth groups without any normals are kept as losses, so neither vanishes silently.
fn finish_object(
    mesh: &mut ExchangeMesh,
    corner_normals: &[Option<DVec3>],
    smoothed: bool,
    losses: &mut Vec<Loss>,
) {
    let supplied = corner_normals.iter().flatten().count();
    if supplied > 0 && supplied == corner_normals.len() {
        mesh.normals = Some(
            corner_normals
                .iter()
                .flatten()
                .map(DVec3::to_array)
                .collect(),
        );
    } else if supplied > 0 {
        losses.push(Loss {
            feature_id: "mesh.normals".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some face corners have no vn normal; partial normals are ignored and shading is recomputed".to_owned(),
            suggestion: Some("write a vn index on every face corner (f v//vn) to keep custom normals".to_owned()),
        });
    } else if smoothed {
        losses.push(Loss {
            feature_id: "mesh.smoothing".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "smooth groups without vn normals are not mapped to shading".to_owned(),
            suggestion: Some("write vn normals (f v//vn) to keep smooth shading".to_owned()),
        });
    }
}

fn parse_floats<'a>(
    mut words: impl Iterator<Item = &'a str>,
    count: usize,
    line: usize,
) -> Result<Vec<f64>> {
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let value = words
            .next()
            .ok_or_else(|| parse_error(line))?
            .parse::<f64>()
            .map_err(|_| parse_error(line))?;
        if !value.is_finite() {
            return Err(parse_error(line));
        }
        result.push(value);
    }
    Ok(result)
}

fn parse_error(line: usize) -> PotError {
    PotError::with_details(
        ErrorCode::ImportFailed,
        "invalid OBJ syntax",
        json!({ "line": line + 1 }),
    )
}

fn obj_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_whitespace() {
                '_'
            } else {
                character
            }
        })
        .collect()
}

pub(crate) fn imported_graph(
    path: &Path,
    scene_id: String,
) -> Result<crate::exchange::ImportedGraph> {
    let (meshes, losses) = import(path)?;
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "obj", losses)
}
