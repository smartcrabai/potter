use std::{fmt::Write as _, fs, path::Path};

use serde_json::json;

use super::ExchangeMesh;
use crate::error::{ErrorCode, PotError, Result};

pub(crate) fn export(meshes: &[ExchangeMesh]) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut object = String::from("mtllib scene.mtl\n");
    let mut base = 0_usize;
    for mesh in meshes {
        object.push_str("o ");
        object.push_str(&obj_name(&mesh.name));
        object.push('\n');
        for position in &mesh.positions {
            writeln!(
                object,
                "v {:.9} {:.9} {:.9}",
                position[0], position[1], position[2]
            )
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        }
        object.push_str("usemtl potter_default\n");
        for face in &mesh.faces {
            if face.len() < 3 {
                continue;
            }
            object.push('f');
            for index in face {
                let vertex = base
                    .checked_add(*index)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow")
                    })?;
                write!(object, " {vertex}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
            }
            object.push('\n');
        }
        base = base
            .checked_add(mesh.positions.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow"))?;
    }
    let material = "newmtl potter_default\nKd 0.6 0.6 0.6\nKa 0 0 0\nKs 0 0 0\nNs 18\n";
    Ok((object.into_bytes(), material.as_bytes().to_vec()))
}

pub(crate) fn import(path: &Path) -> Result<Vec<ExchangeMesh>> {
    let text = fs::read_to_string(path).map_err(|error| PotError::io(&error))?;
    let mut positions = Vec::<[f64; 3]>::new();
    let mut meshes = Vec::<ExchangeMesh>::new();
    let mut current = ExchangeMesh {
        id: "object_0".to_owned(),
        name: "Object".to_owned(),
        positions: Vec::new(),
        faces: Vec::new(),
    };
    let mut remap = std::collections::BTreeMap::<usize, usize>::new();
    for (line_number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else {
            continue;
        };
        match kind {
            "v" => {
                let values = parse_floats(words, 3, line_number)?;
                positions.push([values[0], values[1], values[2]]);
            }
            "o" | "g" => {
                let name = words.collect::<Vec<_>>().join(" ");
                if !current.faces.is_empty() || !current.positions.is_empty() {
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
                    };
                    remap.clear();
                } else if !name.is_empty() {
                    current.name = name;
                }
            }
            "f" => {
                let mut face = Vec::new();
                for token in words {
                    let raw_index = token
                        .split('/')
                        .next()
                        .ok_or_else(|| parse_error(line_number))?;
                    let index = raw_index
                        .parse::<i64>()
                        .map_err(|_| parse_error(line_number))?;
                    let source_index = match index.cmp(&0) {
                        std::cmp::Ordering::Greater => {
                            usize::try_from(index - 1).map_err(|_| parse_error(line_number))?
                        }
                        std::cmp::Ordering::Less => positions
                            .len()
                            .checked_sub(
                                usize::try_from(index.unsigned_abs())
                                    .map_err(|_| parse_error(line_number))?,
                            )
                            .ok_or_else(|| parse_error(line_number))?,
                        std::cmp::Ordering::Equal => return Err(parse_error(line_number)),
                    };
                    let position = *positions
                        .get(source_index)
                        .ok_or_else(|| parse_error(line_number))?;
                    let local_index = *remap.entry(source_index).or_insert_with(|| {
                        current.positions.push(position);
                        current.positions.len() - 1
                    });
                    face.push(local_index);
                }
                if face.len() >= 3 {
                    current.faces.push(face);
                }
            }
            _ => {}
        }
    }
    if !current.faces.is_empty() || !current.positions.is_empty() {
        meshes.push(current);
    }
    if meshes.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::ImportFailed,
            "OBJ contains no faces",
            json!({ "path": path.display().to_string() }),
        ));
    }
    Ok(meshes)
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
    let meshes = import(path)?;
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "obj")
}
