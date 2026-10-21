use std::{collections::BTreeMap, fs, path::Path};

use glam::DVec3;
use serde_json::json;

use super::{ExchangeMesh, Loss, shading};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::color_layers::ColorStore,
};

/// Per-component slack for stored facet normals, which exporters often round to two decimals.
const FACET_NORMAL_TOLERANCE: f64 = 1.0e-2;

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
        let (faces, _) = shading::baked_faces(mesh)?;
        for face in &faces {
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

pub(crate) fn import(path: &Path) -> Result<(Vec<ExchangeMesh>, Vec<Loss>)> {
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
    let mut facet_normals_disagree = false;
    for triangle in bytes[84..].as_chunks::<50>().0 {
        let stored = DVec3::from_array(read_vec3(&triangle[..12])?.map(f64::from));
        let mut face = Vec::with_capacity(3);
        let mut corners = [DVec3::ZERO; 3];
        for (corner, slot) in corners.iter_mut().enumerate() {
            let start = 12 + corner * 12;
            let value = read_vec3(&triangle[start..start + 12])?;
            if !value.iter().all(|component| component.is_finite()) {
                return Err(stl_error("binary STL contains a non-finite vertex"));
            }
            let position = value.map(f64::from);
            *slot = DVec3::from_array(position);
            let key = value.map(f32::to_bits);
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
        facet_normals_disagree |= facet_normal_disagrees(stored, corners);
        faces.push(face);
    }
    // Blender ignores stored facet normals and recomputes them from winding. Potter does the same,
    // but reports a contradicted normal instead of dropping it silently.
    let losses = if facet_normals_disagree {
        vec![Loss {
            feature_id: "mesh.normals".to_owned(),
            data_id: None,
            reason:
                "stored STL facet normals disagree with winding and are ignored, matching Blender"
                    .to_owned(),
            suggestion: None,
        }]
    } else {
        Vec::new()
    };
    // Blender's `stl_import_mesh.cc` calls `mesh_smooth_set(false)`: every face is flat (`sharp_face`
    // present), and facet normals are not imported unless `use_facet_normal` is set (off by default).
    let sharp_faces = vec![true; faces.len()];
    Ok((
        vec![ExchangeMesh {
            id: "stl_mesh".to_owned(),
            name: "STL Mesh".to_owned(),
            positions,
            faces,
            normals: None,
            mirrored: false,
            singular: false,
            vertex_normals: Vec::new(),
            smooth: Vec::new(),
            sharp_faces: Some(sharp_faces),
            uv_layers: Vec::new(),
            uv_active: None,
            uv_render: None,
            colors: ColorStore::default(),
            vertex_groups: Vec::new(),
            material_slots: Vec::new(),
            face_material_indices: Vec::new(),
        }],
        losses,
    ))
}

fn read_vec3(bytes: &[u8]) -> Result<[f32; 3]> {
    let mut value = [0.0_f32; 3];
    for (axis, target) in value.iter_mut().enumerate() {
        let raw = bytes
            .get(axis * 4..axis * 4 + 4)
            .ok_or_else(|| stl_error("binary STL vertex is truncated"))?;
        *target = f32::from_le_bytes(
            raw.try_into()
                .map_err(|_| stl_error("binary STL vertex is truncated"))?,
        );
    }
    Ok(value)
}

/// Whether a stored facet normal contradicts the winding normal of its triangle. Zero stored
/// normals are absent, and degenerate triangles have no winding direction to contradict.
fn facet_normal_disagrees(stored: DVec3, corners: [DVec3; 3]) -> bool {
    if stored == DVec3::ZERO {
        return false;
    }
    let Some(stored) = stored.try_normalize() else {
        return true;
    };
    let winding = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
    let Some(winding) = winding.try_normalize() else {
        return false;
    };
    (stored - winding).abs().max_element() > FACET_NORMAL_TOLERANCE
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
    let (meshes, losses) = import(path)?;
    crate::exchange::import_graph_from_meshes(meshes, path, scene_id, "stl", losses)
}
