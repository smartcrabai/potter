//! Parameter adapter for Blender's voxel and octree-style Remesh modifier modes.

use std::collections::{BTreeSet, HashMap};

use serde_json::{Map, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::Modifier,
};

use super::{bool_param, invalid_parameter, number_param, uint_param, validated};
mod dual_contour;
mod voxel;
mod voxel_edge_groups;

pub(super) fn remesh_modifier(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let mode = modifier
        .params
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("VOXEL");
    match mode {
        "VOXEL" => voxel_remesh(mesh, modifier),
        "BLOCKS" | "SMOOTH" | "SHARP" => octree_remesh(mesh, modifier, mode),
        _ => Err(invalid_parameter(
            modifier,
            "mode",
            "BLOCKS, SMOOTH, SHARP, or VOXEL",
        )),
    }
}

fn voxel_remesh(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let voxel_size = number_param(modifier, "voxel_size", 0.1)?;
    if voxel_size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "voxel_size",
            "a finite positive number",
        ));
    }
    let adaptivity = number_param(modifier, "adaptivity", 0.0)?;
    if !(0.0..=1.0).contains(&adaptivity) {
        return Err(invalid_parameter(
            modifier,
            "adaptivity",
            "a number in [0, 1]",
        ));
    }
    if adaptivity > 0.0 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Blender VOXEL adaptivity is not supported",
            json!({
                "feature_id":"modifier.remesh.voxel_adaptivity",
                "modifier_id":modifier.id,
                "mode":"VOXEL",
                "adaptivity":adaptivity
            }),
        ));
    }
    let voxel_size = voxel_size as f32;
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "voxel_size",
            "a positive float32 value",
        ));
    }
    let smooth_shade = bool_param(modifier, "use_smooth_shade", false)?;
    voxel::remesh(mesh, voxel_size, smooth_shade)
}

fn octree_remesh(mesh: &Mesh, modifier: &Modifier, mode: &str) -> Result<Mesh> {
    let depth = uint_param(modifier, "octree_depth", 4)?;
    if depth == 0 || depth > 24 {
        return Err(invalid_parameter(
            modifier,
            "octree_depth",
            "an integer from 1 to 24",
        ));
    }
    if depth > 6 {
        return Err(PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            format!(
                "dual-contouring grid at octree_depth {depth} exceeds the configured voxel budget (maximum depth 6)"
            ),
        ));
    }
    let scale = number_param(modifier, "scale", 0.9)?;
    if !scale.is_finite() || !(0.0..=0.99).contains(&scale) || scale <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "scale",
            "a finite number in (0, 0.99]",
        ));
    }
    let sharpness = number_param(modifier, "sharpness", 1.0)?;
    if !sharpness.is_finite() {
        return Err(invalid_parameter(modifier, "sharpness", "a finite number"));
    }
    let remove_disconnected = bool_param(modifier, "use_remove_disconnected", true)?;
    let smooth_shade = bool_param(modifier, "use_smooth_shade", false)?;
    let threshold = number_param(modifier, "threshold", 1.0)?;
    if !(0.0..=1.0).contains(&threshold) {
        return Err(invalid_parameter(
            modifier,
            "threshold",
            "a number in [0, 1]",
        ));
    }
    let depth = u32::try_from(depth).map_err(|_| {
        PotError::new(
            crate::error::ErrorCode::LimitExceeded,
            "dual-contour depth exceeds integer limits",
        )
    })?;
    let mut result = dual_contour::remesh(mesh, depth, scale, mode, sharpness, &modifier.id)?;
    finish_remesh(&mut result, smooth_shade, remove_disconnected, threshold)?;
    Ok(result)
}

fn finish_remesh(
    mesh: &mut Mesh,
    smooth_shade: bool,
    remove_disconnected: bool,
    threshold: f64,
) -> Result<()> {
    if remove_disconnected {
        remove_disconnected_components(mesh, threshold);
    }
    if smooth_shade {
        let values = mesh
            .faces
            .iter()
            .map(|face| (format!("f{}", face.id), json!(true)))
            .collect::<Map<_, _>>();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"faces","type":"bool","values":values}),
        );
    }
    validated(mesh)
}

fn remove_disconnected_components(mesh: &mut Mesh, threshold: f64) {
    if mesh.faces.len() < 2 || threshold <= 0.0 {
        return;
    }
    let mut vertex_faces = HashMap::<u32, Vec<usize>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            vertex_faces.entry(*vertex_id).or_default().push(face_index);
        }
    }
    let mut unseen = (0..mesh.faces.len()).collect::<BTreeSet<_>>();
    let mut components = Vec::new();
    while let Some(first) = unseen.iter().next().copied() {
        unseen.remove(&first);
        let mut stack = vec![first];
        let mut component = Vec::new();
        while let Some(face_index) = stack.pop() {
            component.push(face_index);
            for vertex_id in &mesh.faces[face_index].vertices {
                if let Some(adjacent) = vertex_faces.get(vertex_id) {
                    for adjacent_face in adjacent {
                        if unseen.remove(adjacent_face) {
                            stack.push(*adjacent_face);
                        }
                    }
                }
            }
        }
        components.push(component);
    }
    let largest = components.iter().map(Vec::len).max().unwrap_or(0);
    let minimum_size = (largest as f32 * threshold as f32) as usize;
    let retained = components
        .iter()
        .filter(|component| component.len() >= minimum_size)
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let retained_face_ids = mesh
        .faces
        .iter()
        .enumerate()
        .filter(|(index, _)| retained.contains(index))
        .map(|(_, face)| face.id)
        .collect::<BTreeSet<_>>();
    mesh.faces
        .retain(|face| retained_face_ids.contains(&face.id));
    let retained_vertices = mesh
        .faces
        .iter()
        .flat_map(|face| face.vertices.iter().copied())
        .collect::<BTreeSet<_>>();
    mesh.vertices
        .retain(|vertex| retained_vertices.contains(&vertex.id));
    mesh.edges.retain(|edge| {
        retained_vertices.contains(&edge.vertices[0])
            && retained_vertices.contains(&edge.vertices[1])
    });
}

#[cfg(test)]
mod tests {
    use crate::{
        geom::{BoxParams, Mesh},
        model::{Id, Modifier},
    };
    use serde_json::{Map, json};

    use super::remesh_modifier;

    #[test]
    fn octree_modes_map_depth_to_geometry_resolution() -> Result<(), Box<dyn std::error::Error>> {
        let source = Mesh::box_mesh(BoxParams::default())?;
        let modifier = Modifier {
            id: Id::new("remesh_test")?,
            modifier_type: "remesh".to_owned(),
            name: "Remesh".to_owned(),
            enabled: true,
            params: serde_json::from_value::<Map<String, serde_json::Value>>(json!({
                "mode":"BLOCKS", "octree_depth":3, "scale":0.9
            }))?,
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let result = remesh_modifier(&source, &modifier)?;
        assert!(result.validate().is_ok());
        assert!(!result.faces.is_empty());
        Ok(())
    }
}
