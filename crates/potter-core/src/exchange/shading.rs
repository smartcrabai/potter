//! Shading normals shared by every exchange format.
//!
//! Export: `evaluated_meshes` maps each mesh's local normals to world space through the normal
//! matrix (cofactor form of the world matrix, normalized) and records whether the node mirrors or
//! is singular. World-space writers (OBJ, PLY, STL) take `baked_faces`, which reverses mirrored
//! faces so winding agrees with the normals. Object-local writers (USD, FBX, Alembic) keep local winding and normals and
//! let the transform carry the placement, as Blender does. Import: each importer hands per-corner
//! normals to `import_reader_shading`, which applies the format's Blender reader rule for `sharp_face`
//! and custom normals; glTF derives `shade_smooth` and `custom_normal` through `import_shading_flags`.

use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Value, json};

use super::{ExchangeMesh, Loss, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        clnor::{self, NormalCodeOptions, SetterInput},
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

/// Per-face corner values as a world-baked writer stores them: a mirrored node reverses each face,
/// as `baked_faces` does for normals.
pub(super) fn baked_corners<T: Clone>(mesh: &ExchangeMesh, corners: &[Vec<T>]) -> Vec<Vec<T>> {
    if !mesh.mirrored {
        return corners.to_vec();
    }
    corners
        .iter()
        .map(|face| face.iter().rev().cloned().collect())
        .collect()
}

/// Shading a Potter writer stores exactly. `smooth` is per face in `mesh.faces` order; `sharp`
/// holds the endpoint vertex indices (into `mesh.vertices`) of each sharp edge.
pub(super) struct ExactShading {
    pub(super) smooth: Vec<bool>,
    pub(super) sharp: Vec<[usize; 2]>,
}

/// Imports shading the way the format's Blender reader does. `sharp_faces` is the `sharp_face` attribute
/// the reader creates: `None` when it creates none, so every polygon stays smooth, and `Some(flags)` when
/// it writes `sharp_face`, which records its presence. `normals` become custom loop normals and never
/// decide flat faces (Blender's readers do not derive shading from them).
pub(super) fn import_reader_shading(
    mesh: &mut Mesh,
    normals: Option<&[DVec3]>,
    sharp_faces: Option<&[bool]>,
) -> Result<()> {
    match sharp_faces {
        Some(sharp) => {
            if sharp.len() != mesh.faces.len() {
                return Err(import_error(
                    "imported sharp_face flags do not match the face count",
                ));
            }
            import_exact_shading(mesh, normals, sharp.iter().map(|flag| !flag).collect())
        }
        None => {
            import_exact_shading(mesh, normals, vec![true; mesh.faces.len()])?;
            crate::geom::normals::settle_sharp_face(mesh)
        }
    }
}

/// Writes `smooth` as the `shade_smooth` flags and imports `normals` as custom normals when given.
/// Presence is recorded (`lookup_or_add` on the Blender side); callers that create no `sharp_face`
/// settle it afterwards.
fn import_exact_shading(
    mesh: &mut Mesh,
    normals: Option<&[DVec3]>,
    smooth: Vec<bool>,
) -> Result<()> {
    if let Some(normals) = normals {
        return import_shading_flags(
            mesh,
            normals,
            Some(&ExactShading {
                smooth,
                sharp: Vec::new(),
            }),
        );
    }
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
    crate::geom::normals::record_sharp_face(mesh);
    Ok(())
}

/// [`import_shading_flags`] with exact flags from a Potter writer: they replace the derived smooth flags
/// and add the sharp edges the normals were computed with.
pub(super) fn import_shading_flags(
    mesh: &mut Mesh,
    normals: &[DVec3],
    flags: Option<&ExactShading>,
) -> Result<()> {
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
    let smooth = if let Some(flags) = flags {
        flags.smooth.clone()
    } else {
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
        smooth
    };
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
    // Blender's glTF importer always ensures `sharp_face` (`io_scene_gltf2/blender/imp/mesh.py`,
    // `set_poly_smoothing`), so the imported mesh records it even when every face is smooth.
    crate::geom::normals::record_sharp_face(mesh);
    if let Some(flags) = flags.filter(|flags| !flags.sharp.is_empty()) {
        let edge_key = |first: u32, second: u32| (first.min(second), first.max(second));
        let mut sharp_keys = BTreeSet::new();
        for [first, second] in &flags.sharp {
            let vertex_id = |index: usize| {
                mesh.vertices
                    .get(index)
                    .map(|vertex| vertex.id)
                    .ok_or_else(|| {
                        import_error("glTF Potter sharp edge references a missing vertex")
                    })
            };
            sharp_keys.insert(edge_key(vertex_id(*first)?, vertex_id(*second)?));
        }
        let edge_keys = mesh
            .edges
            .iter()
            .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
            .collect::<BTreeSet<_>>();
        if !sharp_keys.is_subset(&edge_keys) {
            return Err(import_error("glTF Potter sharp edge is not a mesh edge"));
        }
        let values = mesh
            .edges
            .iter()
            .map(|edge| {
                let sharp = sharp_keys.contains(&edge_key(edge.vertices[0], edge.vertices[1]));
                (format!("e{}", edge.id), json!(sharp))
            })
            .collect::<serde_json::Map<String, Value>>();
        mesh.attributes.insert(
            "sharp".to_owned(),
            json!({ "domain": "edge", "type": "bool", "values": values }),
        );
    }
    let computed = crate::geom::normals::corner_normals(mesh, &points).ok();
    // A corner matching the computed normal is automatic; a differing one is authored.
    let vectors: Vec<Option<DVec3>> = (0..normals.len())
        .map(|index| {
            let differs = match &computed {
                Some(expected) => !close(expected[index], normals[index]),
                None => true,
            };
            differs.then_some(normals[index])
        })
        .collect();
    let any_authored = vectors.iter().any(Option::is_some);
    clnor::encode_vectors(
        mesh,
        any_authored.then_some(vectors.as_slice()),
        NormalCodeOptions {
            corner_order: None,
            setter_input: SetterInput::NormalizeInput,
        },
    )
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
