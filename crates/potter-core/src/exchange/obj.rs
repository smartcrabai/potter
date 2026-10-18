use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use glam::DVec3;
use serde_json::json;

use super::{
    ExchangeMesh, ExchangeVertexGroup, Loss, obj_group_is_face_compatible, shading, single_uv_set,
    srgb_decode, srgb_encode,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::color_layers::{ColorDomain, ColorLayer, ColorStore},
    model::Id,
};

/// Material every file defines, and the `usemtl` of faces with no bound material slot.
const DEFAULT_MATERIAL: &str = "potter_default";
/// MTL body shared by every material entry.
const MATERIAL_BODY: &str = "Kd 0.6 0.6 0.6\nKa 0 0 0\nKs 0 0 0\nNs 18\n";

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
    // Whether the last `g` line written named groups. Groups stay in effect across `o` lines, so
    // an object that starts without groups must reset them.
    let mut groups_on = false;
    // Names of the bound materials written so far, each with an MTL entry.
    let mut bound_materials = BTreeSet::<String>::new();
    for mesh in meshes {
        object.push_str("o ");
        object.push_str(&obj_name(&mesh.name));
        object.push('\n');
        let colors = vertex_colors(mesh)?;
        for (index, position) in mesh.positions.iter().enumerate() {
            let [x, y, z] = to_obj_axes(*position);
            let written = match colors.as_ref().and_then(|colors| colors.get(index)) {
                Some([red, green, blue]) => {
                    writeln!(object, "v {x:.9} {y:.9} {z:.9} {red} {green} {blue}")
                }
                None => writeln!(object, "v {x:.9} {y:.9} {z:.9}"),
            };
            written.map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
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
        if !mesh.face_material_indices.is_empty()
            && mesh.face_material_indices.len() != mesh.faces.len()
        {
            return Err(PotError::new(
                ErrorCode::ExportFailed,
                "OBJ face material slots do not match the face count",
            ));
        }
        let face_materials = (0..mesh.faces.len())
            .map(|face| bound_material(mesh, face))
            .collect::<Vec<_>>();
        let compatible_groups = mesh
            .vertex_groups
            .iter()
            .filter(|group| obj_group_is_face_compatible(mesh, group))
            .map(|group| {
                (
                    group.name.as_str(),
                    group
                        .weights
                        .iter()
                        .map(|(vertex, _)| *vertex)
                        .collect::<BTreeSet<_>>(),
                )
            })
            .collect::<Vec<_>>();
        object.push_str("usemtl ");
        object.push_str(DEFAULT_MATERIAL);
        object.push('\n');
        if groups_on {
            object.push_str("g\n");
        }
        let mut material_state = Some(None::<&Id>);
        let mut group_state = Vec::<&str>::new();
        let mut face_groups = Vec::<&str>::new();
        let mut smooth_state = None::<bool>;
        let mut corner = 0_usize;
        for (face_index, (face, is_smooth)) in
            faces.iter().zip(mesh.smooth.iter().copied()).enumerate()
        {
            let first_corner = corner;
            corner = first_corner.checked_add(face.len()).ok_or_else(|| {
                PotError::new(ErrorCode::ExportFailed, "OBJ corner count overflow")
            })?;
            if face.len() < 3 {
                continue;
            }
            let material = face_materials.get(face_index).copied().flatten();
            if material_state != Some(material) {
                let name = material.map_or_else(|| DEFAULT_MATERIAL.to_owned(), mtl_name);
                writeln!(object, "usemtl {name}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
                if material.is_some() {
                    bound_materials.insert(name);
                }
                material_state = Some(material);
            }
            face_groups.clear();
            if let Some(&first_vertex) = face.first() {
                for (name, membership) in &compatible_groups {
                    if membership.contains(&first_vertex) {
                        face_groups.push(name);
                    }
                }
            }
            if face_groups != group_state {
                object.push('g');
                for group in &face_groups {
                    object.push(' ');
                    object.push_str(group);
                }
                object.push('\n');
                group_state.clear();
                group_state.extend_from_slice(&face_groups);
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
        groups_on = !group_state.is_empty();
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
    let mut material = format!("newmtl {DEFAULT_MATERIAL}\n{MATERIAL_BODY}");
    for name in &bound_materials {
        write!(material, "newmtl {name}\n{MATERIAL_BODY}")
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    }
    Ok((object.into_bytes(), material.into_bytes()))
}

/// The active color layer as linear RGBA per vertex: a point layer as stored, and a corner layer as
/// the value of each vertex's first corner. A vertex no face uses is opaque black. `None` without an
/// active layer. OBJ and PLY both store one color per vertex through this.
pub(super) fn active_vertex_colors(mesh: &ExchangeMesh) -> Result<Option<Vec<[f64; 4]>>> {
    let Some(layer) = mesh
        .colors
        .active
        .as_deref()
        .and_then(|name| mesh.colors.index_of(name))
        .and_then(|index| mesh.colors.layers.get(index))
    else {
        return Ok(None);
    };
    let corner_count = mesh.faces.iter().map(Vec::len).sum::<usize>();
    let expected = match layer.domain {
        ColorDomain::Point => mesh.positions.len(),
        ColorDomain::Corner => corner_count,
    };
    if layer.values.len() != expected {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "active color layer has an invalid value count",
        ));
    }
    let mut colors = vec![None::<[f64; 4]>; mesh.positions.len()];
    match layer.domain {
        ColorDomain::Point => {
            for (slot, value) in colors.iter_mut().zip(&layer.values) {
                *slot = Some(*value);
            }
        }
        ColorDomain::Corner => {
            for (vertex, value) in mesh.faces.iter().flatten().copied().zip(&layer.values) {
                if let Some(slot) = colors.get_mut(vertex)
                    && slot.is_none()
                {
                    *slot = Some(*value);
                }
            }
        }
    }
    let colors = colors
        .into_iter()
        .map(|color| color.unwrap_or([0.0, 0.0, 0.0, 1.0]))
        .collect::<Vec<_>>();
    if colors.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "color layer contains a non-finite value",
        ));
    }
    Ok(Some(colors))
}

/// The active color layer as the sRGB-encoded `r g b` of each vertex, the `v x y z r g b` form.
/// Alpha is not written: xyzrgb carries RGB only.
fn vertex_colors(mesh: &ExchangeMesh) -> Result<Option<Vec<[f64; 3]>>> {
    Ok(active_vertex_colors(mesh)?.map(|colors| {
        colors
            .into_iter()
            .map(|[red, green, blue, _]| [srgb_encode(red), srgb_encode(green), srgb_encode(blue)])
            .collect()
    }))
}

/// The material slot a face is bound to, or `None` when it is unassigned. An index past the last
/// slot clamps to it, as on import.
fn bound_material(mesh: &ExchangeMesh, face: usize) -> Option<&Id> {
    let last = mesh.material_slots.len().checked_sub(1)?;
    let index = usize::try_from(*mesh.face_material_indices.get(face)?)
        .unwrap_or(usize::MAX)
        .min(last);
    mesh.material_slots.get(index)?.as_ref()
}

/// `usemtl` and MTL name of a bound material. `potter_default` is the unassigned marker, so a
/// material with that id is written under a distinct name.
fn mtl_name(id: &Id) -> String {
    if id.as_str() == DEFAULT_MATERIAL {
        format!("{DEFAULT_MATERIAL}_material")
    } else {
        id.as_str().to_owned()
    }
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

/// An object with no geometry yet. Fields an OBJ file does not carry stay empty.
fn empty_mesh(id: String, name: String) -> ExchangeMesh {
    ExchangeMesh {
        id,
        name,
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
        colors: ColorStore::default(),
        vertex_groups: Vec::new(),
        material_slots: Vec::new(),
        face_material_indices: Vec::new(),
    }
}

pub(crate) fn import(path: &Path) -> Result<(Vec<ExchangeMesh>, Vec<Loss>)> {
    let text = fs::read_to_string(path).map_err(|error| PotError::io(&error))?;
    let mut positions = Vec::<[f64; 3]>::new();
    // Linear color of each `v` line, in file order (`v` indices are global).
    let mut position_colors = Vec::<Option<[f64; 4]>>::new();
    let mut normals = Vec::<DVec3>::new();
    let mut meshes = Vec::<ExchangeMesh>::new();
    let mut losses = Vec::<Loss>::new();
    let mut current = empty_mesh("object_0".to_owned(), "Object".to_owned());
    // Normal named by each face corner of `current`, in face order.
    let mut corner_normals = Vec::<Option<DVec3>>::new();
    // Storage UV of each `vt` line, in file order (`vt` indices are global), and the UV named by
    // each face corner of `current`, in face order.
    let mut uvs = Vec::<[f64; 2]>::new();
    let mut corner_uvs = Vec::<Option<[f64; 2]>>::new();
    // Linear color of each local vertex of `current`, in local order.
    let mut local_colors = Vec::<Option<[f64; 4]>>::new();
    // Groups named by the `g` line in effect, and the local vertices each group holds.
    let mut active_groups = Vec::<String>::new();
    let mut group_members = BTreeMap::<String, BTreeSet<usize>>::new();
    // Every `usemtl` name read, and the one in effect for each face of `current`, in face order.
    let mut material_names = Vec::<String>::new();
    let mut active_material = None::<usize>;
    let mut face_materials = Vec::<Option<usize>>::new();
    // Whether an `s` group is on at the current line, and whether any face of `current` was read
    // while one was on.
    let mut group_on = false;
    let mut smoothed = false;
    let mut remap = BTreeMap::<usize, usize>::new();
    // Names of every object declared by `o`. Objects without geometry are dropped from a file
    // that has geometry; a file with no geometry imports each declared object empty, so
    // Potter's own empty-object exports re-read.
    let mut declared = Vec::<String>::new();
    for (line_number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else {
            continue;
        };
        match kind {
            "v" => {
                let (position, color) = parse_vertex(words, line_number)?;
                positions.push(from_obj_axes(position));
                position_colors.push(color);
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
            "o" => {
                let name = words.collect::<Vec<_>>().join(" ");
                declared.push(name.clone());
                if !current.faces.is_empty() || !current.positions.is_empty() {
                    finish_object(
                        &mut current,
                        &corner_normals,
                        &corner_uvs,
                        &group_members,
                        smoothed,
                        &mut losses,
                    );
                    finish_colors(&mut current, &local_colors, &mut losses);
                    finish_materials(&mut current, &face_materials, &material_names, &mut losses);
                    meshes.push(current);
                    current = empty_mesh(
                        format!("object_{}", meshes.len()),
                        if name.is_empty() {
                            format!("Object {}", meshes.len())
                        } else {
                            name
                        },
                    );
                    corner_normals.clear();
                    corner_uvs.clear();
                    local_colors.clear();
                    group_members.clear();
                    face_materials.clear();
                    smoothed = false;
                    remap.clear();
                } else if !name.is_empty() {
                    current.name = name;
                }
            }
            "g" => {
                active_groups = words.map(str::to_owned).collect();
            }
            "usemtl" => {
                let name = words.collect::<Vec<_>>().join(" ");
                active_material = (!name.is_empty()).then(|| {
                    match material_names.iter().position(|known| *known == name) {
                        Some(index) => index,
                        None => {
                            material_names.push(name);
                            material_names.len() - 1
                        }
                    }
                });
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
                        local_colors.push(position_colors.get(source_index).copied().flatten());
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
                    for group in &active_groups {
                        group_members
                            .entry(group.clone())
                            .or_default()
                            .extend(face.iter().copied());
                    }
                    current.faces.push(face);
                    face_materials.push(active_material);
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
            &group_members,
            smoothed,
            &mut losses,
        );
        finish_colors(&mut current, &local_colors, &mut losses);
        finish_materials(&mut current, &face_materials, &material_names, &mut losses);
        meshes.push(current);
    }
    if meshes.is_empty() {
        if declared.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "OBJ contains no faces",
                json!({ "path": path.display().to_string() }),
            ));
        }
        let has_uvs = !uvs.is_empty();
        meshes = declared
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let mut mesh = empty_mesh(
                    format!("object_{index}"),
                    if name.is_empty() {
                        format!("Object {index}")
                    } else {
                        name
                    },
                );
                if has_uvs {
                    (mesh.uv_layers, mesh.uv_active, mesh.uv_render) = single_uv_set(Vec::new());
                }
                mesh
            })
            .collect();
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

/// Sets `mesh.normals` when every face corner names a normal, the UV layer when every corner
/// names a `vt`, and the vertex groups whose vertices fill whole faces. Partial values are dropped
/// and smooth groups without any normals are kept as losses, so nothing vanishes silently.
fn finish_object(
    mesh: &mut ExchangeMesh,
    corner_normals: &[Option<DVec3>],
    corner_uvs: &[Option<[f64; 2]>],
    group_members: &BTreeMap<String, BTreeSet<usize>>,
    smoothed: bool,
    losses: &mut Vec<Loss>,
) {
    for (name, vertices) in group_members {
        let group = ExchangeVertexGroup {
            name: name.clone(),
            weights: vertices.iter().map(|&vertex| (vertex, 1.0)).collect(),
        };
        if obj_group_is_face_compatible(mesh, &group) {
            mesh.vertex_groups.push(group);
        } else {
            losses.push(Loss {
                feature_id: "mesh.vertex_group".to_owned(),
                data_id: Some(mesh.name.clone()),
                reason: format!(
                    "OBJ group `{name}` crosses face boundaries and cannot be represented as a vertex group"
                ),
                suggestion: Some(
                    "use an OBJ group whose faces have disjoint or fully shared vertices".to_owned(),
                ),
            });
        }
    }
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

/// Sets the `Color` layer from the `v` colors when every vertex has one. A partial set is a loss.
fn finish_colors(
    mesh: &mut ExchangeMesh,
    local_colors: &[Option<[f64; 4]>],
    losses: &mut Vec<Loss>,
) {
    let supplied = local_colors.iter().flatten().count();
    if supplied > 0 && supplied == local_colors.len() {
        mesh.colors = ColorStore {
            layers: vec![ColorLayer {
                name: "Color".to_owned(),
                domain: ColorDomain::Point,
                values: local_colors.iter().flatten().copied().collect(),
            }],
            active: Some("Color".to_owned()),
            render: Some("Color".to_owned()),
        };
    } else if supplied > 0 {
        losses.push(Loss {
            feature_id: "mesh.color_attribute".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some OBJ vertices have no xyzrgb color; partial vertex colors are ignored"
                .to_owned(),
            suggestion: Some(
                "write three sRGB color values on every vertex to keep vertex colors".to_owned(),
            ),
        });
    }
}

/// Sets the material slots and per-face slot indices from the `usemtl` name in effect for each
/// face. Each distinct name is one slot, in first-use order. `potter_default` and faces read before
/// any `usemtl` are unassigned. A name that is not a Potter material id imports as unassigned and
/// is a loss. Nothing is set when no face binds a material.
fn finish_materials(
    mesh: &mut ExchangeMesh,
    face_materials: &[Option<usize>],
    material_names: &[String],
    losses: &mut Vec<Loss>,
) {
    // Name index of each slot, `None` for the unassigned slot.
    let mut keys = Vec::<Option<usize>>::new();
    let mut slots = Vec::<Option<Id>>::new();
    let mut indices = Vec::with_capacity(face_materials.len());
    for &material in face_materials {
        let key = material.filter(|&index| {
            material_names
                .get(index)
                .is_some_and(|name| name != DEFAULT_MATERIAL)
        });
        let slot = match keys.iter().position(|&known| known == key) {
            Some(slot) => slot,
            None => {
                let id = match key.and_then(|index| material_names.get(index)) {
                    None => None,
                    Some(name) => match Id::new(name.as_str()) {
                        Ok(id) => Some(id),
                        Err(_) => {
                            losses.push(Loss {
                                feature_id: "material.binding".to_owned(),
                                data_id: Some(mesh.name.clone()),
                                reason: format!(
                                    "OBJ material `{name}` is not a Potter material id; its faces are imported unassigned"
                                ),
                                suggestion: Some(
                                    "name OBJ materials with Potter material ids: lowercase letters, digits, `_`, and `-`"
                                        .to_owned(),
                                ),
                            });
                            None
                        }
                    },
                };
                keys.push(key);
                slots.push(id);
                slots.len() - 1
            }
        };
        indices.push(u32::try_from(slot).unwrap_or(u32::MAX));
    }
    if slots.iter().all(Option::is_none) {
        return;
    }
    mesh.material_slots = slots;
    mesh.face_material_indices = indices;
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

/// Parses a `v` line: `x y z`, optionally with a `w` before the color, and an optional sRGB
/// `r g b` after the position. The color is decoded to linear light with alpha 1.
fn parse_vertex<'a>(
    words: impl Iterator<Item = &'a str>,
    line: usize,
) -> Result<([f64; 3], Option<[f64; 4]>)> {
    let values = words
        .map(|word| {
            word.parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| parse_error(line))
        })
        .collect::<Result<Vec<_>>>()?;
    if !matches!(values.len(), 3 | 4 | 6 | 7) {
        return Err(parse_error(line));
    }
    let position = [values[0], values[1], values[2]];
    let color_start = match values.len() {
        6 => Some(3),
        7 => Some(4),
        _ => None,
    };
    let color = color_start.map(|start| {
        [
            srgb_decode(values[start]),
            srgb_decode(values[start + 1]),
            srgb_decode(values[start + 2]),
            1.0,
        ]
    });
    Ok((position, color))
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

    #[test]
    fn empty_declared_object_keeps_uv_metadata_when_texture_coordinates_exist() -> Result<()> {
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), "o empty\nvt 0.25\n").map_err(|error| PotError::io(&error))?;
        let (meshes, _) = import(file.path())?;
        let mesh = meshes
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ object is missing"))?;
        assert_eq!(mesh.faces, Vec::<Vec<usize>>::new());
        assert_eq!(mesh.uv_layers.len(), 1);
        assert_eq!(mesh.uv_layers[0].name, "UVMap");
        assert_eq!(mesh.uv_layers[0].corners, Vec::<Vec<[f64; 2]>>::new());
        assert_eq!(mesh.uv_active, Some(0));
        assert_eq!(mesh.uv_render, Some(0));
        Ok(())
    }
}
