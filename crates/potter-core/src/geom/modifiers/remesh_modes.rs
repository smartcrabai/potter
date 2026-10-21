// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_remesh.cc, SPDX-FileCopyrightText: 2011 by Nicholas Bishop., GPL-2.0-or-later.

//! Parameter adapter for Blender's voxel and octree-style Remesh modifier modes.

use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::Modifier,
};

use super::{
    bool_param, invalid_parameter, number_param, string_param_with_expected, uint_param, validated,
};
mod dual_contour;
mod voxel;
mod voxel_edge_groups;

pub(super) fn remesh_modifier(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let mode = string_param_with_expected(modifier, "mode", "BLOCKS, SMOOTH, SHARP, or VOXEL")?;
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
    let voxel_size = number_param(modifier, "voxel_size")?;
    if voxel_size <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "voxel_size",
            "a finite positive number",
        ));
    }
    let adaptivity = number_param(modifier, "adaptivity")?;
    if !(0.0..=1.0).contains(&adaptivity) {
        return Err(invalid_parameter(
            modifier,
            "adaptivity",
            "a number in [0, 1]",
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
    let smooth_shade = bool_param(modifier, "use_smooth_shade")?;
    voxel::remesh(mesh, voxel_size, adaptivity as f32, smooth_shade)
}

fn octree_remesh(mesh: &Mesh, modifier: &Modifier, mode: &str) -> Result<Mesh> {
    let depth = uint_param(modifier, "octree_depth")?;
    if depth == 0 || depth > 24 {
        return Err(invalid_parameter(
            modifier,
            "octree_depth",
            "an integer from 1 to 24",
        ));
    }
    let scale = number_param(modifier, "scale")?;
    if !scale.is_finite() || !(0.0..=0.99).contains(&scale) || scale <= 0.0 {
        return Err(invalid_parameter(
            modifier,
            "scale",
            "a finite number in (0, 0.99]",
        ));
    }
    if depth > 20 {
        return Err(PotError::with_details(
            ErrorCode::LimitExceeded,
            format!("dual-contour depth {depth} has zero cell resolution on Blender's 2^20 grid"),
            json!({
                "modifier_id":modifier.id,
                "mode":mode,
                "octree_depth":depth
            }),
        ));
    }
    let sharpness = number_param(modifier, "sharpness")?;
    if !sharpness.is_finite() {
        return Err(invalid_parameter(modifier, "sharpness", "a finite number"));
    }
    let remove_disconnected = bool_param(modifier, "use_remove_disconnected")?;
    let smooth_shade = bool_param(modifier, "use_smooth_shade")?;
    let threshold = number_param(modifier, "threshold")?;
    if !(0.0..=1.0).contains(&threshold) {
        return Err(invalid_parameter(
            modifier,
            "threshold",
            "a number in [0, 1]",
        ));
    }
    let depth = u32::try_from(depth).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "dual-contour depth exceeds integer limits",
        )
    })?;
    let settings = dual_contour::DualConSettings {
        mode,
        depth,
        scale: scale as f32,
        sharpness: sharpness as f32,
        threshold: threshold as f32,
        flood_fill: remove_disconnected,
    };
    let mut result = dual_contour::remesh(mesh, &settings)?;
    finish_remesh(&mut result, smooth_shade)?;
    Ok(result)
}

fn finish_remesh(mesh: &mut Mesh, smooth_shade: bool) -> Result<()> {
    if smooth_shade {
        crate::geom::normals::shade_faces_smooth(mesh);
    }
    validated(mesh)
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
            baked_ocean_cache: None,
            id: Id::new("remesh_test")?,
            modifier_type: "remesh".to_owned(),
            name: "Remesh".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value::<Map<String, serde_json::Value>>(json!({
                "mode":"BLOCKS", "octree_depth":3, "scale":0.9
            }))?,
            binding_data: None,
            baked_ocean_cache: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let result = remesh_modifier(&source, &modifier)?;
        assert!(result.validate().is_ok());
        assert!(!result.faces.is_empty(), "remesh produced no faces");
        Ok(())
    }
}
