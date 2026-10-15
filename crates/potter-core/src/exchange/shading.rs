//! Shading normals shared by every exchange format.
//!
//! Export: `evaluated_meshes` maps each mesh's local corner normals to world space (inverse
//! transpose of the world matrix, normalized) and records whether the node mirrors. World-space
//! writers (OBJ, PLY, STL) take `baked_faces`, which reverses mirrored faces so winding agrees
//! with the normals. Object-local writers (USD, FBX, Alembic) keep local winding and normals and
//! let the transform carry the placement, as Blender does. Import: each importer hands per-corner
//! normals to `import_shading`, which derives `shade_smooth` and `custom_normal` the way glTF does.

use std::collections::BTreeMap;

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Value, json};

use super::{ExchangeMesh, Loss, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

/// Corner normals and flatness compare within this distance, matching glTF import.
const TOLERANCE: f64 = 1.0e-4;

/// Whether the transform flips handedness, so exported faces must reverse their winding.
pub(super) fn is_mirrored(matrix: DMat4) -> bool {
    DMat3::from_mat4(matrix).determinant() < 0.0
}

/// World-space corner normals for each face of `mesh`, in `mesh.faces` order.
///
/// Normals come from local geometry (Blender computes shading before transforming it) and then
/// pass through the inverse transpose, which keeps them perpendicular under non-uniform scale.
pub(super) fn world_corner_normals(
    mesh: &Mesh,
    local: &[DVec3],
    matrix: DMat4,
) -> Result<Vec<Vec<[f64; 3]>>> {
    let local_normals = crate::geom::normals::corner_normals(mesh, local)?;
    let normal_matrix = DMat3::from_mat4(matrix).inverse().transpose();
    let mut start = 0;
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let end = start + face.vertices.len();
        let row = local_normals[start..end]
            .iter()
            .map(|normal| {
                (normal_matrix * *normal)
                    .try_normalize()
                    .map(|world| world.to_array())
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::ExportFailed,
                            "mesh normal cannot be transformed to world space",
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        faces.push(row);
        start = end;
    }
    Ok(faces)
}

/// Faces and corner normals as a world-baked writer stores them. A mirrored node reverses each
/// face so its winding agrees with the normals, as Blender's OBJ, PLY, and STL exporters do.
pub(super) fn baked_faces(mesh: &ExchangeMesh) -> (Vec<Vec<usize>>, Vec<[f64; 3]>) {
    let normals = corner_normals(mesh);
    if !mesh.mirrored {
        return (mesh.faces.clone(), normals);
    }
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut reversed = Vec::with_capacity(normals.len());
    let mut start = 0;
    for face in &mesh.faces {
        let end = start + face.len();
        faces.push(face.iter().rev().copied().collect());
        reversed.extend(normals[start..end].iter().rev().copied());
        start = end;
    }
    (faces, reversed)
}

/// Derives `shade_smooth` and `custom_normal` on `mesh` from per-corner normals.
///
/// `normals` holds one value per face corner in `mesh.faces` order. A face is smooth when any
/// corner differs from its flat normal. A corner whose normal differs from what the derived
/// shading computes gets a `custom_normal` entry.
pub(super) fn import_shading(mesh: &mut Mesh, normals: &[DVec3]) -> Result<()> {
    let index_of = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let corners =
        mesh.faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| {
                        index_of.get(id).copied().ok_or_else(|| {
                            import_error("imported face references a missing vertex")
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
    let corner_total = corners.iter().map(Vec::len).sum::<usize>();
    if normals.len() != corner_total {
        return Err(import_error(
            "imported normals do not match the mesh corner count",
        ));
    }
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let mut smooth = Vec::with_capacity(corners.len());
    let mut corner = 0;
    for face_corners in &corners {
        let flat = face_normal(
            &face_corners
                .iter()
                .map(|vertex| points[*vertex])
                .collect::<Vec<_>>(),
        );
        let end = corner + face_corners.len();
        smooth.push(
            normals[corner..end]
                .iter()
                .any(|normal| !close(*normal, flat)),
        );
        corner = end;
    }
    if smooth.iter().any(|value| *value) {
        let values = mesh
            .faces
            .iter()
            .zip(&smooth)
            .map(|(face, value)| (format!("f{}", face.id), json!(value)))
            .collect::<serde_json::Map<String, Value>>();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({ "domain": "face", "type": "bool", "values": values }),
        );
    }
    let computed = crate::geom::normals::corner_normals(mesh, &points).ok();
    let mut custom_values = serde_json::Map::new();
    let mut corner = 0;
    for (face, face_corners) in mesh.faces.iter().zip(&corners) {
        let end = corner + face_corners.len();
        let mismatch = match &computed {
            Some(expected) => (corner..end).any(|index| !close(expected[index], normals[index])),
            None => true,
        };
        if mismatch {
            let values = normals[corner..end]
                .iter()
                .map(|normal| normal.normalize_or_zero().to_array())
                .collect::<Vec<_>>();
            custom_values.insert(format!("f{}", face.id), json!(values));
        }
        corner = end;
    }
    if !custom_values.is_empty() {
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({ "domain": "corner", "type": "float3", "values": custom_values }),
        );
    }
    Ok(())
}

/// Loss entries for a format whose normal layout cannot carry every corner normal.
///
/// STL stores one normal per triangle, so any smooth or custom corner is lost. PLY stores one
/// normal per vertex, so a vertex whose corners disagree is lost. Meshes with no shading
/// (every corner equals its flat normal) write no normals in PLY and lose nothing.
pub(crate) fn shading_losses(format: &str, meshes: &[ExchangeMesh]) -> Vec<Loss> {
    let lossy = meshes.iter().any(|mesh| match format {
        "stl" => !is_flat(mesh),
        "ply" => !is_flat(mesh) && !vertex_normals_exact(mesh),
        _ => false,
    });
    if !lossy {
        return Vec::new();
    }
    let (reason, suggestion) = if format == "stl" {
        (
            "stl stores one normal per triangle and cannot represent smooth or custom shading",
            "export as obj, ply, usda, fbx, or alembic to preserve shading",
        )
    } else {
        (
            "ply stores one normal per vertex and cannot represent per-corner shading",
            "export as obj, usda, fbx, or alembic to preserve shading",
        )
    };
    vec![Loss {
        feature_id: "mesh.shading".to_owned(),
        data_id: None,
        reason: reason.to_owned(),
        suggestion: Some(suggestion.to_owned()),
    }]
}

/// Per-face smooth state: a face is smooth when any of its corners differs from its flat normal.
/// One value per face in `mesh.faces` order; all false when `normals` is None.
pub(super) fn smooth_faces(mesh: &ExchangeMesh) -> Vec<bool> {
    let Some(normals) = &mesh.normals else {
        return vec![false; mesh.faces.len()];
    };
    let mut corner = 0;
    mesh.faces
        .iter()
        .map(|face| {
            let end = corner + face.len();
            let mut points = face
                .iter()
                .filter_map(|index| mesh.positions.get(*index))
                .map(|point| DVec3::from_array(*point))
                .collect::<Vec<_>>();
            // Corner normals point along the world-space surface, so a mirrored face's winding must
            // be reversed before its geometric normal is compared with them.
            if mesh.mirrored {
                points.reverse();
            }
            let flat = face_normal(&points);
            let smooth = !normals[corner..end]
                .iter()
                .all(|normal| close(DVec3::from_array(*normal), flat));
            corner = end;
            smooth
        })
        .collect()
}

/// Whether every corner of `mesh` carries its face's flat normal.
pub(super) fn is_flat(mesh: &ExchangeMesh) -> bool {
    !smooth_faces(mesh).contains(&true)
}

/// World-space corner normals of `mesh`, in `faces` order. Meshes without normals are flat,
/// so each corner takes its face's normal.
pub(super) fn corner_normals(mesh: &ExchangeMesh) -> Vec<[f64; 3]> {
    if let Some(normals) = &mesh.normals {
        return normals.clone();
    }
    let mut normals = Vec::new();
    for face in &mesh.faces {
        let points = face
            .iter()
            .filter_map(|index| mesh.positions.get(*index))
            .map(|point| DVec3::from_array(*point))
            .collect::<Vec<_>>();
        let flat = face_normal(&points).to_array();
        normals.extend(std::iter::repeat_n(flat, face.len()));
    }
    normals
}

/// Whether all corners sharing a vertex agree, so one normal per vertex describes them all.
fn vertex_normals_exact(mesh: &ExchangeMesh) -> bool {
    let Some(normals) = &mesh.normals else {
        return true;
    };
    let mut first: Vec<Option<DVec3>> = vec![None; mesh.positions.len()];
    let mut corner = 0;
    for face in &mesh.faces {
        for index in face {
            let normal = DVec3::from_array(normals[corner]);
            corner += 1;
            let Some(slot) = first.get_mut(*index) else {
                return false;
            };
            match slot {
                Some(expected) if !close(*expected, normal) => return false,
                Some(_) => {}
                None => *slot = Some(normal),
            }
        }
    }
    true
}

/// Newell normal of a polygon, normalized; zero for degenerate polygons.
fn face_normal(points: &[DVec3]) -> DVec3 {
    let mut area = DVec3::ZERO;
    for (index, current) in points.iter().enumerate() {
        area += current.cross(points[(index + 1) % points.len()]);
    }
    area.normalize_or_zero()
}

fn close(left: DVec3, right: DVec3) -> bool {
    (left - right).abs().max_element() <= TOLERANCE
}
