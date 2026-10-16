use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path};

use glam::DVec3;
use serde_json::json;

use super::{ExchangeMesh, Loss, shading, single_uv_set};
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
    let mut uv_base = 0_usize;
    for mesh in meshes {
        object.push_str("o ");
        object.push_str(&obj_name(&mesh.name));
        object.push('\n');
        for position in &mesh.positions {
            let [x, y, z] = to_obj_axes(*position);
            writeln!(object, "v {x:.9} {y:.9} {z:.9}")
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        }
        let (faces, normals) = shading::baked_faces(mesh)?;
        // Blender writes zero normals for singular transforms.
        let normals = if mesh.singular {
            vec![[0.0; 3]; normals.len()]
        } else {
            normals
        };
        let normals = normals.into_iter().map(to_obj_axes).collect::<Vec<_>>();
        let (normal_lines, corner_ids) = indexed_table(&normals, |normal| {
            format!("vn {:.6} {:.6} {:.6}", normal[0], normal[1], normal[2])
        });
        for line in &normal_lines {
            object.push_str(line);
            object.push('\n');
        }
        let uv_values = mesh
            .uv_active
            .and_then(|index| mesh.uv_layers.get(index))
            .map(|layer| {
                shading::baked_corners(mesh, &layer.corners)
                    .into_iter()
                    .flatten()
                    .map(|[u, v]| [u, 1.0 - v])
                    .collect::<Vec<_>>()
            });
        // Blender writes the active layer's `vt` lines in first-use order, one per distinct value.
        // Storage V points down; Blender's OBJ writes raw Blender V, which points up.
        let (uv_lines, uv_ids) = match uv_values {
            Some(values) => {
                let (lines, ids) = indexed_table(&values, |[u, v]| format!("vt {u:.6} {v:.6}"));
                (lines, Some(ids))
            }
            None => (Vec::new(), None),
        };
        for line in &uv_lines {
            object.push_str(line);
            object.push('\n');
        }
        object.push_str("usemtl potter_default\n");
        let smooth = &mesh.smooth;
        let mut smooth_state = None::<bool>;
        let mut corner = 0_usize;
        for (face, is_smooth) in faces.iter().zip(smooth.iter().copied()) {
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
                let uv = match &uv_ids {
                    Some(ids) => first_corner
                        .checked_add(offset)
                        .and_then(|corner| ids.get(corner))
                        .and_then(|local| uv_base.checked_add(*local))
                        .and_then(|index| index.checked_add(1))
                        .map(|index| index.to_string())
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::ExportFailed,
                                "OBJ UV count does not match faces",
                            )
                        })?,
                    None => String::new(),
                };
                write!(object, "/{uv}/{normal}")
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
        uv_base = uv_base
            .checked_add(uv_lines.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ UV index overflow"))?;
    }
    let material = "newmtl potter_default\nKd 0.6 0.6 0.6\nKa 0 0 0\nKs 0 0 0\nNs 18\n";
    Ok((object.into_bytes(), material.as_bytes().to_vec()))
}

/// Blender's indexed table for one object: unique entries in first-use order, as text lines, plus
/// each value's 0-based index into those lines.
fn indexed_table<T>(values: &[T], line: impl Fn(&T) -> String) -> (Vec<String>, Vec<usize>) {
    let mut lines = Vec::<String>::new();
    let mut lookup = BTreeMap::<String, usize>::new();
    let mut corner_ids = Vec::with_capacity(values.len());
    for value in values {
        let text = line(value);
        let index = *lookup.entry(text.clone()).or_insert_with(|| {
            lines.push(text);
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
        vertex_normals: Vec::new(),
        smooth: Vec::new(),
        uv_layers: Vec::new(),
        uv_active: None,
        uv_render: None,
    };
    // Normal named by each face corner of `current`, in face order.
    let mut corner_normals = Vec::<Option<DVec3>>::new();
    // Storage UV of each `vt` line, in file order (`vt` indices are global), and the UV named by
    // each face corner of `current`, in face order.
    let mut uvs = Vec::<[f64; 2]>::new();
    let mut corner_uvs = Vec::<Option<[f64; 2]>>::new();
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
            "vt" => {
                // Blender's OBJ writes raw V, which points up; Potter stores V down. The OBJ format
                // makes `v` optional with a default of 0, so `vt u` is a complete coordinate.
                let u = parse_floats(words.by_ref(), 1, line_number)?[0];
                let v = if words.clone().next().is_some() {
                    parse_floats(words, 1, line_number)?[0]
                } else {
                    0.0
                };
                uvs.push([u, 1.0 - v]);
            }
            "o" | "g" => {
                let name = words.collect::<Vec<_>>().join(" ");
                if !current.faces.is_empty() || !current.positions.is_empty() {
                    finish_object(
                        &mut current,
                        &corner_normals,
                        &corner_uvs,
                        smoothed,
                        &mut losses,
                    );
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
                        vertex_normals: Vec::new(),
                        smooth: Vec::new(),
                        uv_layers: Vec::new(),
                        uv_active: None,
                        uv_render: None,
                    };
                    corner_normals.clear();
                    corner_uvs.clear();
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
                let mut face_uvs = Vec::new();
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
                    let uv = match token.split('/').nth(1).filter(|value| !value.is_empty()) {
                        Some(value) => {
                            let index = resolve_index(value, uvs.len(), line_number)?;
                            Some(*uvs.get(index).ok_or_else(|| parse_error(line_number))?)
                        }
                        None => None,
                    };
                    face_uvs.push(uv);
                }
                if face.len() >= 3 {
                    current.faces.push(face);
                    corner_normals.extend(face_normals);
                    corner_uvs.extend(face_uvs);
                    smoothed |= group_on;
                }
            }
            _ => {}
        }
    }
    if !current.faces.is_empty() || !current.positions.is_empty() {
        finish_object(
            &mut current,
            &corner_normals,
            &corner_uvs,
            smoothed,
            &mut losses,
        );
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

/// Sets `mesh.normals` when every face corner names a normal, and the UV layer when every corner
/// names a `vt`. Partial values are dropped and smooth groups without any normals are kept as
/// losses, so nothing vanishes silently.
fn finish_object(
    mesh: &mut ExchangeMesh,
    corner_normals: &[Option<DVec3>],
    corner_uvs: &[Option<[f64; 2]>],
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
    let supplied_uvs = corner_uvs.iter().flatten().count();
    if supplied_uvs > 0 && supplied_uvs == corner_uvs.len() {
        let mut values = corner_uvs.iter().flatten().copied();
        let corners = mesh
            .faces
            .iter()
            .map(|face| values.by_ref().take(face.len()).collect::<Vec<_>>())
            .collect();
        (mesh.uv_layers, mesh.uv_active, mesh.uv_render) = single_uv_set(corners);
    } else if supplied_uvs > 0 {
        losses.push(Loss {
            feature_id: "mesh.uv_layer".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some face corners have no vt coordinate; partial UVs are ignored".to_owned(),
            suggestion: Some(
                "write a vt index on every face corner (f v/vt) to keep UV coordinates".to_owned(),
            ),
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

#[cfg(test)]
mod tests {
    use super::{export, import};
    use crate::{
        error::{ErrorCode, PotError, Result},
        exchange::test_support::triangle_with_layers,
    };
    use std::fs;

    #[test]
    fn writes_the_active_layer_as_blender_v_with_corner_indices() -> Result<()> {
        let (object, _) = export(&[triangle_with_layers()])?;
        let text = String::from_utf8_lossy(&object);
        let vt = text
            .lines()
            .filter(|line| line.starts_with("vt "))
            .collect::<Vec<_>>();
        // Lightmap is active; its storage V flips to Blender's V.
        assert_eq!(
            vt,
            [
                "vt 0.200000 0.900000",
                "vt 0.600000 0.700000",
                "vt 0.400000 1.000000"
            ]
        );
        let face = text
            .lines()
            .find(|line| line.starts_with("f "))
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "OBJ face is missing"))?;
        let corners = face.split_whitespace().skip(1).collect::<Vec<_>>();
        assert_eq!(corners.len(), 3);
        for (index, corner) in corners.iter().enumerate() {
            let fields = corner.split('/').collect::<Vec<_>>();
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[1], (index + 1).to_string());
            assert_ne!(fields[0], "");
            assert_ne!(fields[2], "");
        }
        Ok(())
    }

    #[test]
    fn imports_vt_as_the_uv_layer_in_storage_convention() -> Result<()> {
        let original = triangle_with_layers();
        let (object, _) = export(std::slice::from_ref(&original))?;
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), object).map_err(|error| PotError::io(&error))?;
        let (imported, _) = import(file.path())?;
        let layer = imported[0]
            .uv_layers
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ import has no UV layer"))?;
        assert_eq!(layer.name, "UVMap");
        // The export writes the active layer, Lightmap, so that is the layer that comes back.
        for (read, written) in layer.corners[0]
            .iter()
            .zip(&original.uv_layers[1].corners[0])
        {
            assert!((read[0] - written[0]).abs() < 1.0e-6);
            assert!((read[1] - written[1]).abs() < 1.0e-6);
        }
        Ok(())
    }
}
