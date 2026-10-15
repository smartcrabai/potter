//! Shading normals shared by every exchange format.
//!
//! Export: `evaluated_meshes` maps each mesh's local normals to world space through the normal
//! matrix (cofactor form of the world matrix, normalized) and records whether the node mirrors or
//! is singular. World-space writers (OBJ, PLY, STL) take `baked_faces`, which reverses mirrored
//! faces so winding agrees with the normals. Object-local writers (USD, FBX, Alembic) keep local winding and normals and
//! let the transform carry the placement, as Blender does. Import: each importer hands per-corner
//! normals to `import_shading`, which derives `shade_smooth` and `custom_normal` the way glTF does.

use std::collections::BTreeMap;

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Value, json};

use super::{ExchangeMesh, Loss, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        normals::{close, face_normal, normal_matrix},
    },
};

/// Whether the transform flips handedness, so exported faces must reverse their winding.
pub(super) fn is_mirrored(matrix: DMat4) -> bool {
    DMat3::from_mat4(matrix).determinant() < 0.0
}

/// Whether the transform is singular. Its normals have no inverse-transpose; OBJ writes zero
/// normals for such nodes, as Blender does.
pub(super) fn is_singular(matrix: DMat4) -> bool {
    let determinant = DMat3::from_mat4(matrix).determinant();
    determinant == 0.0 || !determinant.is_finite()
}

/// World-space shading of one mesh: corner normals in `mesh.faces` order, per-vertex normals
/// indexed like `mesh.vertices`, and per-face smooth flags.
pub(super) struct WorldNormals {
    pub corners: Vec<[f64; 3]>,
    pub vertices: Vec<[f64; 3]>,
    pub smooth: Vec<bool>,
}

/// World-space shading computed from local geometry. Blender computes shading before transforming
/// it, so normals come from local positions and then pass through the normal matrix.
pub(super) fn world_normals(mesh: &Mesh, local: &[DVec3], matrix: DMat4) -> Result<WorldNormals> {
    let shading = crate::geom::normals::export_normals(mesh, local)?;
    let normal_matrix = normal_matrix(matrix);
    let to_world = |normal: DVec3| (normal_matrix * normal).normalize_or_zero().to_array();
    Ok(WorldNormals {
        corners: shading.corners.into_iter().map(to_world).collect(),
        vertices: shading.vertices.into_iter().map(to_world).collect(),
        smooth: shading.smooth,
    })
}

/// Faces and corner normals of one mesh, as a world-baked writer stores them.
pub(super) type BakedFaces = (Vec<Vec<usize>>, Vec<[f64; 3]>);

/// Faces and corner normals as a world-baked writer stores them. A mirrored node reverses each
/// face so its winding agrees with the normals, as Blender's OBJ, PLY, and STL exporters do.
pub(super) fn baked_faces(mesh: &ExchangeMesh) -> Result<BakedFaces> {
    let normals = mesh
        .normals
        .as_deref()
        .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "mesh normals are missing"))?;
    if !mesh.mirrored {
        return Ok((mesh.faces.clone(), normals.to_vec()));
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
    Ok((faces, reversed))
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
/// STL stores one normal per triangle, so any smooth or custom corner is lost. PLY writes one
/// normal per vertex for every mesh in the file once any mesh is shaded, so a vertex whose corners
/// disagree is lost in any mesh. A flat-only file writes no normals and loses nothing.
pub(crate) fn shading_losses(format: &str, meshes: &[ExchangeMesh]) -> Vec<Loss> {
    let shaded = meshes.iter().any(|mesh| !is_flat(mesh));
    let lossy = match format {
        "stl" => shaded,
        "ply" => shaded && meshes.iter().any(|mesh| !vertex_normals_exact(mesh)),
        _ => false,
    };
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

/// Whether every face of `mesh` carries its flat normal at every corner.
pub(super) fn is_flat(mesh: &ExchangeMesh) -> bool {
    !mesh.smooth.contains(&true)
}

/// Whether all corners sharing a vertex agree, so one normal per vertex describes them all.
fn vertex_normals_exact(mesh: &ExchangeMesh) -> bool {
    let normals = mesh.normals.as_deref().unwrap_or_default();
    let mut first: Vec<Option<DVec3>> = vec![None; mesh.positions.len()];
    let mut corner = 0;
    for face in &mesh.faces {
        for index in face {
            let Some(normal) = normals.get(corner).map(|normal| DVec3::from_array(*normal)) else {
                return false;
            };
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
