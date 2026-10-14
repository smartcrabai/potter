use std::path::Path;

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    float::equal_f64,
    model::{AvatarMtoon, AvatarOutlineWidthMode, Id, Material, SceneDoc, TextureRef},
};

use super::super::gltf::{TextureExport, VrmWriterMappings, unsupported_export};

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

pub(super) fn validate_v1(material: &Material) -> Result<()> {
    validate_transmission(material)
}

pub(super) fn v1(
    material_id: &Id,
    mtoon: &AvatarMtoon,
    doc: &SceneDoc,
    _mappings: &VrmWriterMappings<'_>,
    _root: &Value,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    let material = doc
        .materials
        .get(material_id)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "MToon material is missing"))?;
    validate_v1(material)?;
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
        value["shadeMultiplyTexture"] =
            texture_export.texture_info(texture, doc, root_path, binary, views)?;
    }
    if let Some(shift) = &mtoon.shading_shift_texture {
        let mut info =
            texture_export.texture_info(&shift.texture, doc, root_path, binary, views)?;
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
            value[key] = texture_export.texture_info(texture, doc, root_path, binary, views)?;
        }
    }
    Ok(value)
}

pub(super) fn v0(
    _material_id: &Id,
    material: &Material,
    mtoon: &AvatarMtoon,
    doc: &SceneDoc,
    _mappings: &VrmWriterMappings<'_>,
    _root: &Value,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<Value> {
    validate_v0(material, mtoon)?;
    let (blend_mode, queue, alpha_tag, keyword) = alpha_properties(material)?;
    let (legacy_shade_shift, legacy_shade_toony) = legacy_shading_values(mtoon)?;
    let float_properties = json!({
        "_MToonVersion": 39.0,
        "_Cutoff": material.alpha_threshold,
        "_BumpScale": 1.0,
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
        "_CullMode": if material.double_sided { 0.0 } else { 2.0 },
        "_OutlineCullMode": 1.0,
        "_ZWrite": if material.alpha_mode != "blend" || mtoon.transparent_with_z_write { 1.0 } else { 0.0 },
        "_AlphaToMask": 0.0,
        "_SrcBlend": if material.alpha_mode == "blend" { 5.0 } else { 1.0 },
        "_DstBlend": if material.alpha_mode == "blend" { 10.0 } else { 0.0 },
    });
    let vector_properties = json!({
        "_Color": [
            legacy_color_component(material.base_color[0]),
            legacy_color_component(material.base_color[1]),
            legacy_color_component(material.base_color[2]),
            material.base_color[3]
        ],
        "_ShadeColor": [
            legacy_color_component(mtoon.shade_color_factor[0]),
            legacy_color_component(mtoon.shade_color_factor[1]),
            legacy_color_component(mtoon.shade_color_factor[2]),
            1.0
        ],
        "_EmissionColor": [
            legacy_color_component(material.emission_color[0] * material.emission_strength),
            legacy_color_component(material.emission_color[1] * material.emission_strength),
            legacy_color_component(material.emission_color[2] * material.emission_strength),
            1.0
        ],
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
        "_MainTex_ST": [1.0, 1.0, 0.0, 0.0]
    });
    let render_queue = queue + mtoon.render_queue_offset_number;
    let mut texture_properties = serde_json::Map::new();
    if let Some(texture) = &material.base_color_texture {
        add_texture(
            &mut texture_properties,
            "_MainTex",
            texture,
            None,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &material.normal_texture {
        add_texture(
            &mut texture_properties,
            "_BumpMap",
            texture,
            None,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &mtoon.shade_multiply_texture {
        add_texture(
            &mut texture_properties,
            "_ShadeTexture",
            texture,
            None,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &mtoon.matcap_texture {
        add_texture(
            &mut texture_properties,
            "_SphereAdd",
            texture,
            None,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &mtoon.rim_multiply_texture {
        add_texture(
            &mut texture_properties,
            "_RimTexture",
            texture,
            None,
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &mtoon.outline_width_multiply_texture {
        add_texture(
            &mut texture_properties,
            "_OutlineWidthTexture",
            texture,
            Some(1),
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    if let Some(texture) = &mtoon.uv_animation_mask_texture {
        add_texture(
            &mut texture_properties,
            "_UvAnimMaskTexture",
            texture,
            Some(2),
            doc,
            root_path,
            texture_export,
            binary,
            views,
        )?;
    }
    let mut keywords = serde_json::Map::new();
    if !keyword.is_empty() {
        keywords.insert(keyword.to_owned(), json!(true));
    }
    if material.normal_texture.is_some() {
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
        "name": material.name,
        "shader": "VRM/MToon",
        "renderQueue": render_queue,
        "floatProperties": float_properties,
        "vectorProperties": vector_properties,
        "textureProperties": texture_properties,
        "keywordMap": Value::Object(keywords),
        "tagMap": { "RenderType": alpha_tag }
    }))
}

pub(super) fn validate_v0(material: &Material, mtoon: &AvatarMtoon) -> Result<()> {
    alpha_properties(material)?;
    validate_transmission(material)?;
    if mtoon.shading_shift_texture.is_some() {
        return Err(unsupported_export(
            "vrm.v0.mtoon_shading_shift_texture",
            "VRM 0.x MToon shading-grade textures cannot preserve VRM 1.0 shadingShiftTexture; remove the map or export VRM 1.0.",
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
    if material.alpha_mode != "blend" && mtoon.render_queue_offset_number != 0 {
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

fn validate_transmission(material: &Material) -> Result<()> {
    if !equal_f64(material.transmission, 0.0) {
        return Err(unsupported_export(
            "vrm.mtoon_transmission",
            "VRM MToon does not support physical transmission; clear the material transmission factor or remove its MToon settings.",
        ));
    }
    Ok(())
}

fn add_texture(
    output: &mut serde_json::Map<String, Value>,
    name: &str,
    texture: &TextureRef,
    remap_channel: Option<usize>,
    doc: &SceneDoc,
    root_path: &Path,
    texture_export: &mut TextureExport,
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
) -> Result<()> {
    let index = if let Some(channel) = remap_channel {
        texture_export
            .texture_info_channel(texture, Some(channel), doc, root_path, binary, views)?
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "remapped glTF texture index is invalid",
                )
            })?
    } else {
        texture_export.texture_index(texture, doc, root_path, binary, views)?
    };
    output.insert(name.to_owned(), json!(index));
    Ok(())
}

fn alpha_properties(material: &Material) -> Result<(f64, i32, &'static str, &'static str)> {
    let (blend_mode, queue, alpha_tag, keyword) = match material.alpha_mode.as_str() {
        "opaque" => (0.0, 2000, "Opaque", ""),
        "clip" => (1.0, 2450, "TransparentCutout", "_ALPHATEST_ON"),
        "blend" => (2.0, 3000, "Transparent", "_ALPHABLEND_ON"),
        _ => {
            return Err(unsupported_export(
                "vrm.v0.material_alpha_mode",
                "VRM 0.x MToon does not recognize this material alpha_mode.",
            ));
        }
    };
    Ok((blend_mode, queue, alpha_tag, keyword))
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
