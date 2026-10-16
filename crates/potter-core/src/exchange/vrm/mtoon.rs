use std::path::Path;

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    float::equal_f64,
    model::{AvatarMtoon, AvatarOutlineWidthMode, SceneDoc, TextureRef},
};

use super::super::{
    gltf::{TextureExport, reads, unsupported_export},
    material_surface::{
        AlphaMode, ResolvedSurface, SampledTexture, TextureChannel, reference_sampled,
        remapped_reference_sampled,
    },
};

pub(crate) fn uv_textures(mtoon: &AvatarMtoon) -> impl Iterator<Item = &TextureRef> {
    [
        mtoon.shade_multiply_texture.as_ref(),
        mtoon
            .shading_shift_texture
            .as_ref()
            .map(|texture| &texture.texture),
        mtoon.rim_multiply_texture.as_ref(),
        mtoon.outline_width_multiply_texture.as_ref(),
        mtoon.uv_animation_mask_texture.as_ref(),
    ]
    .into_iter()
    .flatten()
}

/// Rejects resolved surface features that VRM `MToon` cannot express. Both VRM versions share it.
fn validate_mtoon_surface(surface: &ResolvedSurface) -> Result<()> {
    if surface.transmission.is_some() {
        return Err(unsupported_export(
            "vrm.mtoon_transmission",
            "VRM MToon does not support physical transmission; clear the material transmission factor or remove its MToon settings.",
        ));
    }
    let unsupported = [
        (
            surface.unlit,
            "vrm.mtoon_unlit",
            "unlit (Background) shading",
        ),
        (
            surface.clearcoat.is_some(),
            "vrm.mtoon_clearcoat",
            "clearcoat",
        ),
        (surface.sheen.is_some(), "vrm.mtoon_sheen", "sheen"),
        (surface.specular.is_some(), "vrm.mtoon_specular", "specular"),
        (
            surface.anisotropy.is_some(),
            "vrm.mtoon_anisotropy",
            "anisotropy",
        ),
    ];
    for (present, feature_id, name) in unsupported {
        if present {
            return Err(unsupported_export(
                feature_id,
                &format!(
                    "VRM MToon does not support {name}; remove it from the material or remove its MToon settings."
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_v1(surface: &ResolvedSurface) -> Result<()> {
    validate_mtoon_surface(surface)
}

pub(super) fn v1(
    surface: &ResolvedSurface,
    mtoon: &AvatarMtoon,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    validate_v1(surface)?;
    let mut value = json!({
        "specVersion": "1.0",
        "transparentWithZWrite": mtoon.transparent_with_z_write,
        "renderQueueOffsetNumber": mtoon.render_queue_offset_number,
        "shadeColorFactor": mtoon.shade_color_factor,
        "shadingShiftFactor": mtoon.shading_shift_factor,
        "shadingToonyFactor": mtoon.shading_toony_factor,
        "giEqualizationFactor": mtoon.gi_equalization_factor,
        "matcapFactor": mtoon.matcap_factor,
        "parametricRimColorFactor": mtoon.parametric_rim_color_factor,
        "rimLightingMixFactor": mtoon.rim_lighting_mix_factor,
        "parametricRimFresnelPowerFactor": mtoon.parametric_rim_fresnel_power_factor,
        "parametricRimLiftFactor": mtoon.parametric_rim_lift_factor,
        "outlineWidthMode": outline_mode_v1(mtoon.outline_width_mode),
        "outlineWidthFactor": mtoon.outline_width_factor,
        "outlineColorFactor": mtoon.outline_color_factor,
        "outlineLightingMixFactor": mtoon.outline_lighting_mix_factor,
        "uvAnimationScrollXSpeedFactor": mtoon.uv_animation_scroll_x_speed_factor,
        "uvAnimationScrollYSpeedFactor": mtoon.uv_animation_scroll_y_speed_factor,
        "uvAnimationRotationSpeedFactor": mtoon.uv_animation_rotation_speed_factor,
    });
    if let Some(texture) = &mtoon.shade_multiply_texture {
        value["shadeMultiplyTexture"] = texture_export.sampled_info(
            &reference_sampled(texture),
            reads::RGBA,
            doc,
            root_path,
            binary,
            views,
        )?;
    }
    if let Some(shift) = &mtoon.shading_shift_texture {
        let mut info = texture_export.sampled_info(
            &reference_sampled(&shift.texture),
            reads::RGBA,
            doc,
            root_path,
            binary,
            views,
        )?;
        info["scale"] = json!(shift.scale);
        value["shadingShiftTexture"] = info;
    }
    for (key, texture) in [
        ("matcapTexture", mtoon.matcap_texture.as_ref()),
        ("rimMultiplyTexture", mtoon.rim_multiply_texture.as_ref()),
        (
            "outlineWidthMultiplyTexture",
            mtoon.outline_width_multiply_texture.as_ref(),
        ),
        (
            "uvAnimationMaskTexture",
            mtoon.uv_animation_mask_texture.as_ref(),
        ),
    ] {
        if let Some(texture) = texture {
            value[key] = texture_export.sampled_info(
                &reference_sampled(texture),
                reads::RGBA,
                doc,
                root_path,
                binary,
                views,
            )?;
        }
    }
    for (key, transform) in &mtoon.texture_transforms {
        let info = value
            .get_mut(key.as_str())
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "MToon texture transform has no texture in its slot",
                )
            })?;
        info.insert(
            "extensions".to_owned(),
            json!({ "KHR_texture_transform": {
                "offset": transform.offset,
                "rotation": transform.rotation,
                "scale": transform.scale,
            } }),
        );
    }
    Ok(value)
}

pub(super) fn v0(
    name: &str,
    surface: &ResolvedSurface,
    mtoon: &AvatarMtoon,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    validate_v0(surface, mtoon)?;
    let main_tex_st = main_texture_st(surface)?;
    let (blend_mode, queue, alpha_tag, keyword) = alpha_properties(&surface.alpha_mode, mtoon);
    let (legacy_shade_shift, legacy_shade_toony) = legacy_shading_values(mtoon)?;
    let blend = matches!(surface.alpha_mode, AlphaMode::Blend);
    let cutoff = match &surface.alpha_mode {
        AlphaMode::Mask { cutoff } => *cutoff,
        AlphaMode::Opaque | AlphaMode::Blend => 0.5,
    };
    let base = surface.base_color.factor;
    let emissive = surface.emissive.factor;
    // VRM 0.x has no emission strength, so emitted colors clamp to [0, 1] before the legacy curve.
    let emission = |component: f64| legacy_color_component(component.clamp(0.0, 1.0));
    let float_properties = json!({
        "_MToonVersion": 39.0,
        "_Cutoff": cutoff,
        "_BumpScale": surface.normal.as_ref().map_or(1.0, |normal| normal.scale),
        "_ReceiveShadowRate": 1.0,
        "_ShadingGradeRate": 1.0,
        "_ShadeShift": legacy_shade_shift,
        "_ShadeToony": legacy_shade_toony,
        "_LightColorAttenuation": 0.0,
        "_IndirectLightIntensity": 1.0 - mtoon.gi_equalization_factor,
        "_RimLightingMix": mtoon.rim_lighting_mix_factor,
        "_RimFresnelPower": mtoon.parametric_rim_fresnel_power_factor,
        "_RimLift": mtoon.parametric_rim_lift_factor,
        "_OutlineWidth": mtoon.outline_width_factor * 100.0,
        "_OutlineScaledMaxDistance": 1.0,
        "_OutlineLightingMix": mtoon.outline_lighting_mix_factor,
        "_OutlineColorMode": if mtoon.outline_lighting_mix_factor == 0.0 { 0.0 } else { 1.0 },
        "_UvAnimScrollX": mtoon.uv_animation_scroll_x_speed_factor,
        "_UvAnimScrollY": -mtoon.uv_animation_scroll_y_speed_factor,
        "_UvAnimRotation": mtoon.uv_animation_rotation_speed_factor / std::f64::consts::TAU,
        "_BlendMode": blend_mode,
        "_OutlineWidthMode": match mtoon.outline_width_mode {
            AvatarOutlineWidthMode::None => 0.0,
            AvatarOutlineWidthMode::WorldCoordinates => 1.0,
            AvatarOutlineWidthMode::ScreenCoordinates => 2.0,
        },
        "_CullMode": if surface.double_sided { 0.0 } else { 2.0 },
        "_OutlineCullMode": 1.0,
        "_ZWrite": if !blend || mtoon.transparent_with_z_write { 1.0 } else { 0.0 },
        "_AlphaToMask": 0.0,
        "_SrcBlend": if blend { 5.0 } else { 1.0 },
        "_DstBlend": if blend { 10.0 } else { 0.0 },
    });
    let vector_properties = json!({
        "_Color": [
            legacy_color_component(base[0]),
            legacy_color_component(base[1]),
            legacy_color_component(base[2]),
            base[3]
        ],
        "_ShadeColor": [
            legacy_color_component(mtoon.shade_color_factor[0]),
            legacy_color_component(mtoon.shade_color_factor[1]),
            legacy_color_component(mtoon.shade_color_factor[2]),
            1.0
        ],
        "_EmissionColor": [emission(emissive[0]), emission(emissive[1]), emission(emissive[2]), 1.0],
        "_RimColor": [
            legacy_color_component(mtoon.parametric_rim_color_factor[0]),
            legacy_color_component(mtoon.parametric_rim_color_factor[1]),
            legacy_color_component(mtoon.parametric_rim_color_factor[2]),
            1.0
        ],
        "_OutlineColor": [
            legacy_color_component(mtoon.outline_color_factor[0]),
            legacy_color_component(mtoon.outline_color_factor[1]),
            legacy_color_component(mtoon.outline_color_factor[2]),
            1.0
        ],
        "_MainTex_ST": main_tex_st
    });
    let render_queue = queue + mtoon.render_queue_offset_number;
    let mut texture_properties = serde_json::Map::new();
    if let Some(texture) = &surface.base_color.texture {
        add_texture(
            &mut texture_properties,
            "_MainTex",
            texture,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(normal) = &surface.normal {
        add_texture(
            &mut texture_properties,
            "_BumpMap",
            &normal.texture,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &surface.emissive.texture {
        add_texture(
            &mut texture_properties,
            "_EmissionMap",
            texture,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    for (key, reference, remap) in [
        ("_ShadeTexture", mtoon.shade_multiply_texture.as_ref(), None),
        ("_SphereAdd", mtoon.matcap_texture.as_ref(), None),
        ("_RimTexture", mtoon.rim_multiply_texture.as_ref(), None),
        (
            "_OutlineWidthTexture",
            mtoon.outline_width_multiply_texture.as_ref(),
            Some(TextureChannel::Green),
        ),
        (
            "_UvAnimMaskTexture",
            mtoon.uv_animation_mask_texture.as_ref(),
            Some(TextureChannel::Blue),
        ),
    ] {
        if let Some(reference) = reference {
            let texture = match remap {
                Some(channel) => remapped_reference_sampled(reference, channel),
                None => reference_sampled(reference),
            };
            add_texture(
                &mut texture_properties,
                key,
                &texture,
                doc,
                root_path,
                texture_export,
                binary,
                views,
            )?;
        }
    }
    let mut keywords = serde_json::Map::new();
    if !keyword.is_empty() {
        keywords.insert(keyword.to_owned(), json!(true));
    }
    if surface.normal.is_some() {
        keywords.insert("_NORMALMAP".to_owned(), json!(true));
    }
    match mtoon.outline_width_mode {
        AvatarOutlineWidthMode::None => {}
        AvatarOutlineWidthMode::WorldCoordinates => {
            keywords.insert("MTOON_OUTLINE_WIDTH_WORLD".to_owned(), json!(true));
        }
        AvatarOutlineWidthMode::ScreenCoordinates => {
            keywords.insert("MTOON_OUTLINE_WIDTH_SCREEN".to_owned(), json!(true));
        }
    }
    if mtoon.outline_width_mode != AvatarOutlineWidthMode::None {
        let color_keyword = if mtoon.outline_lighting_mix_factor == 0.0 {
            "MTOON_OUTLINE_COLOR_FIXED"
        } else {
            "MTOON_OUTLINE_COLOR_MIXED"
        };
        keywords.insert(color_keyword.to_owned(), json!(true));
    }
    Ok(json!({
        "name": name,
        "shader": "VRM/MToon",
        "renderQueue": render_queue,
        "floatProperties": float_properties,
        "vectorProperties": vector_properties,
        "textureProperties": texture_properties,
        "keywordMap": Value::Object(keywords),
        "tagMap": { "RenderType": alpha_tag }
    }))
}

pub(super) fn validate_v0(surface: &ResolvedSurface, mtoon: &AvatarMtoon) -> Result<()> {
    validate_mtoon_surface(surface)?;
    main_texture_st(surface)?;
    if mtoon.shading_shift_texture.is_some() {
        return Err(unsupported_export(
            "vrm.v0.mtoon_shading_shift_texture",
            "VRM 0.x MToon shading-grade textures cannot preserve VRM 1.0 shadingShiftTexture; remove the map or export VRM 1.0.",
        ));
    }
    if mtoon
        .texture_transforms
        .values()
        .any(|transform| !transform.is_identity())
    {
        return Err(unsupported_export(
            "vrm.v0.mtoon_texture_transform",
            "VRM 0.x MToon has no per-texture transform for these slots; remove texture_transforms or export VRM 1.0.",
        ));
    }
    if mtoon.matcap_texture.is_some()
        && !mtoon
            .matcap_factor
            .iter()
            .all(|component| equal_f64(*component, 1.0))
    {
        return Err(unsupported_export(
            "vrm.v0.mtoon_matcap_factor",
            "VRM 0.x MToon has no matcap color factor; use [1,1,1] or export VRM 1.0.",
        ));
    }
    if !matches!(surface.alpha_mode, AlphaMode::Blend) && mtoon.render_queue_offset_number != 0 {
        return Err(unsupported_export(
            "vrm.v0.material_render_queue",
            "VRM 0.x renderQueue offsets apply only to blend materials; clear render_queue_offset_number or use VRM 1.0.",
        ));
    }
    if mtoon.uv_animation_rotation_speed_factor != 0.0
        && (mtoon.uv_animation_scroll_x_speed_factor != 0.0
            || mtoon.uv_animation_scroll_y_speed_factor != 0.0)
    {
        return Err(unsupported_export(
            "vrm.v0.mtoon_uv_animation_order",
            "VRM 0.x applies UV scrolling before rotation, while VRM 1.0 applies rotation before scrolling; nonzero scroll and rotation together cannot preserve the same motion. Clear either scroll or rotation, or export VRM 1.0.",
        ));
    }
    legacy_shading_values(mtoon)?;
    Ok(())
}

/// `_MainTex_ST` (tiling, offset) from the base texture's KHR transform. Unity-style VRM 0.x
/// shaders flip V relative to glTF, so the offset follows the expression texture-transform bind.
/// VRM 0.x has no rotation, so a rotated base texture is an error.
fn main_texture_st(surface: &ResolvedSurface) -> Result<[f64; 4]> {
    let Some(transform) = surface
        .base_color
        .texture
        .as_ref()
        .and_then(|texture| texture.transform.as_ref())
    else {
        return Ok([1.0, 1.0, 0.0, 0.0]);
    };
    if !equal_f64(transform.rotation, 0.0) {
        return Err(unsupported_export(
            "vrm.v0.mtoon_texture_transform",
            "VRM 0.x MToon `_MainTex_ST` cannot represent a base texture rotation; clear the rotation or export VRM 1.0.",
        ));
    }
    Ok(legacy_st(transform.scale, transform.offset))
}

/// VRM 0.x `_MainTex_ST` (scale x, scale y, offset x, offset y) from a glTF KHR transform. Unity-style
/// shaders flip V relative to glTF, so the offset y is `1 - offset_y - scale_y`. Shared by the
/// material property and the expression texture-transform bind.
pub(crate) fn legacy_st(scale: [f64; 2], offset: [f64; 2]) -> [f64; 4] {
    [scale[0], scale[1], offset[0], 1.0 - offset[1] - scale[1]]
}

fn add_texture(
    output: &mut serde_json::Map<String, Value>,
    name: &str,
    texture: &SampledTexture,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<()> {
    let index = texture_export.sampled_index(texture, doc, root_path, binary, views)?;
    output.insert(name.to_owned(), json!(index));
    Ok(())
}

/// Blend mode, render queue, `RenderType` tag, and keyword for a resolved alpha mode.
fn alpha_properties(
    alpha: &AlphaMode,
    mtoon: &AvatarMtoon,
) -> (f64, i32, &'static str, &'static str) {
    match alpha {
        AlphaMode::Opaque => (0.0, 2000, "Opaque", ""),
        AlphaMode::Mask { .. } => (1.0, 2450, "TransparentCutout", "_ALPHATEST_ON"),
        AlphaMode::Blend if mtoon.transparent_with_z_write => {
            (3.0, 2501, "Transparent", "_ALPHABLEND_ON")
        }
        AlphaMode::Blend => (2.0, 3000, "Transparent", "_ALPHABLEND_ON"),
    }
}

fn legacy_shading_values(mtoon: &AvatarMtoon) -> Result<(f64, f64)> {
    let shift = mtoon.shading_toony_factor - 1.0 - mtoon.shading_shift_factor;
    if !(-1.0..=1.0).contains(&shift) {
        return Err(unsupported_export(
            "vrm.v0.mtoon_shading_ramp",
            "VRM 1.0 shading shift/toony values cannot be represented by the VRM 0.x MToon ramp; adjust them or export VRM 1.0.",
        ));
    }
    let denominator = 1.0 - shift;
    let toony = if denominator.abs() <= f64::EPSILON {
        if equal_f64(mtoon.shading_toony_factor, 1.0) && equal_f64(mtoon.shading_shift_factor, -1.0)
        {
            1.0
        } else {
            return Err(unsupported_export(
                "vrm.v0.mtoon_shading_ramp",
                "VRM 1.0 shading shift/toony values cannot be represented by the VRM 0.x MToon ramp.",
            ));
        }
    } else {
        (2.0 * mtoon.shading_toony_factor - (1.0 + shift)) / denominator
    };
    if !toony.is_finite() || !(0.0..=1.0).contains(&toony) {
        return Err(unsupported_export(
            "vrm.v0.mtoon_shading_ramp",
            "VRM 1.0 shading shift/toony values cannot be represented by the VRM 0.x MToon ramp.",
        ));
    }
    Ok((shift, toony))
}

fn legacy_color_component(linear: f64) -> f64 {
    linear.powf(1.0 / 2.2)
}

fn outline_mode_v1(mode: AvatarOutlineWidthMode) -> &'static str {
    match mode {
        AvatarOutlineWidthMode::None => "none",
        AvatarOutlineWidthMode::WorldCoordinates => "worldCoordinates",
        AvatarOutlineWidthMode::ScreenCoordinates => "screenCoordinates",
    }
}

#[cfg(test)]
mod tests {
    use super::legacy_st;

    #[test]
    fn legacy_st_flips_v_and_keeps_scale_and_offset_x() {
        // glTF offset (0.25, 0) with scale (2, 2): Unity's offset y is 1 - 0 - 2 = -1.
        assert_eq!(legacy_st([2.0, 2.0], [0.25, 0.0]), [2.0, 2.0, 0.25, -1.0]);
        // Identity transform (scale 1, offset 0) maps to the identity _MainTex_ST.
        assert_eq!(legacy_st([1.0, 1.0], [0.0, 0.0]), [1.0, 1.0, 0.0, 0.0]);
    }
}
