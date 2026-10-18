//! `MToon` import: inverts the `MToon` exporter in `exchange/vrm/mtoon.rs`. VRM 1.0 reads
//! `VRMC_materials_mtoon` (`mtoon::v1`); VRM 0.x reads the `VRM/MToon` entries of `materialProperties` (`mtoon::v0`).

use std::{collections::BTreeMap, f64::consts::TAU};

use serde_json::Value;

use crate::{
    error::Result,
    exchange::{
        gltf::{imported_uv_layer, texture_info_set},
        material_surface::TextureTransform,
    },
    float::equal_f64,
    image::ImageInterpolation,
    model::{
        AvatarMtoon, AvatarMtoonShadingShiftTexture, AvatarOutlineWidthMode,
        AvatarTextureTransform, AvatarVersion, Id, TextureRef,
    },
};

use super::{Decoder, import_error};

const SUGGESTION: &str = "use --allow-lossy to import the file without this MToon setting";

/// VRM 0.x colors are gamma-encoded with this exponent (`legacy_color_component`).
const LEGACY_GAMMA: f64 = 2.2;

/// VRM 1.0 `MToon` keys (`mtoon::v1`). `extensions` and `extras` are extension points, not settings.
const V1_FIELDS: &[&str] = &[
    "specVersion",
    "transparentWithZWrite",
    "renderQueueOffsetNumber",
    "shadeColorFactor",
    "shadeMultiplyTexture",
    "shadingShiftFactor",
    "shadingShiftTexture",
    "shadingToonyFactor",
    "giEqualizationFactor",
    "matcapFactor",
    "matcapTexture",
    "parametricRimColorFactor",
    "parametricRimFresnelPowerFactor",
    "parametricRimLiftFactor",
    "rimMultiplyTexture",
    "rimLightingMixFactor",
    "outlineWidthMode",
    "outlineWidthFactor",
    "outlineWidthMultiplyTexture",
    "outlineColorFactor",
    "outlineLightingMixFactor",
    "uvAnimationMaskTexture",
    "uvAnimationScrollXSpeedFactor",
    "uvAnimationScrollYSpeedFactor",
    "uvAnimationRotationSpeedFactor",
    "extensions",
    "extras",
];

/// VRM 1.0 `MToon` texture fields that can carry a `KHR_texture_transform` (`texture_transforms`).
const V1_TEXTURE_SLOTS: &[&str] = &[
    "shadeMultiplyTexture",
    "shadingShiftTexture",
    "matcapTexture",
    "rimMultiplyTexture",
    "outlineWidthMultiplyTexture",
    "uvAnimationMaskTexture",
];

/// VRM 0.x float properties Potter reads (`decode_v0_material`), or that repeat other material data:
/// `_Cutoff` (glTF `alphaCutoff`), `_CullMode`/`_ZWrite`/`_SrcBlend`/`_DstBlend` (glTF `doubleSided`
/// and `alphaMode`), and `_MToonVersion` (format version).
const V0_HANDLED_FLOATS: &[&str] = &[
    "_Cutoff",
    "_MToonVersion",
    "_CullMode",
    "_ZWrite",
    "_SrcBlend",
    "_DstBlend",
    "_BlendMode",
    "_ShadeShift",
    "_ShadeToony",
    "_IndirectLightIntensity",
    "_RimLightingMix",
    "_RimFresnelPower",
    "_RimLift",
    "_OutlineWidth",
    "_OutlineLightingMix",
    "_OutlineColorMode",
    "_OutlineWidthMode",
    "_UvAnimScrollX",
    "_UvAnimScrollY",
    "_UvAnimRotation",
];

/// VRM 0.x float properties Potter has no field for, with the value `mtoon::v0` writes.
/// Any other value is reported as a loss.
const V0_UNREPRESENTED_FLOATS: &[(&str, f64)] = &[
    ("_BumpScale", 1.0),
    ("_ReceiveShadowRate", 1.0),
    ("_ShadingGradeRate", 1.0),
    ("_LightColorAttenuation", 0.0),
    ("_OutlineScaledMaxDistance", 1.0),
    ("_OutlineCullMode", 1.0),
    ("_AlphaToMask", 0.0),
];

/// VRM 0.x color properties Potter reads, plus `_Color` and `_EmissionColor`, which repeat the glTF
/// base color and emissive factor.
const V0_HANDLED_VECTORS: &[&str] = &[
    "_ShadeColor",
    "_RimColor",
    "_OutlineColor",
    "_MainTex_ST",
    "_Color",
    "_EmissionColor",
];

/// VRM 0.x texture properties Potter reads, plus `_MainTex` and `_BumpMap`, which repeat the glTF
/// base color and normal maps.
const V0_HANDLED_TEXTURES: &[&str] = &[
    "_ShadeTexture",
    "_SphereAdd",
    "_RimTexture",
    "_OutlineWidthTexture",
    "_UvAnimMaskTexture",
    "_MainTex",
    "_BumpMap",
];

pub(super) fn decode(decoder: &mut Decoder<'_>) -> Result<BTreeMap<Id, AvatarMtoon>> {
    match decoder.version {
        AvatarVersion::V1 => decode_v1(decoder),
        AvatarVersion::V0 => decode_v0(decoder),
    }
}

fn decode_v1(decoder: &mut Decoder<'_>) -> Result<BTreeMap<Id, AvatarMtoon>> {
    let index = decoder.index;
    let materials = index.root["materials"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    let mut mtoon = BTreeMap::new();
    for (material_index, material) in materials.iter().enumerate() {
        let extension = &material["extensions"]["VRMC_materials_mtoon"];
        if !extension.is_object() {
            continue;
        }
        let material_id = decoder.material_id(material_index)?;
        let value = decode_v1_material(decoder, &material_id, extension)?;
        for material_id in decoder.material_ids(material_index)? {
            mtoon.insert(material_id, value.clone());
        }
    }
    Ok(mtoon)
}

/// Inverts `mtoon::v1` for one material. Every field is the VRM 1.0 key the exporter writes.
fn decode_v1_material(
    decoder: &mut Decoder<'_>,
    material: &Id,
    value: &Value,
) -> Result<AvatarMtoon> {
    if let Some(fields) = value.as_object() {
        for name in fields
            .keys()
            .filter(|name| !V1_FIELDS.contains(&name.as_str()))
        {
            decoder.loss(
                &format!("vrm.mtoon.{name}"),
                format!(
                    "MToon field `{name}` of material `{material}` is not represented by Potter"
                ),
                SUGGESTION,
            );
        }
    }
    let defaults = AvatarMtoon::default();
    Ok(AvatarMtoon {
        transparent_with_z_write: boolean(
            value,
            "transparentWithZWrite",
            defaults.transparent_with_z_write,
        )?, // mtoon::v1 transparentWithZWrite
        render_queue_offset_number: render_queue_offset(
            decoder,
            material,
            "vrm.mtoon.renderQueueOffsetNumber",
            integer(value, "renderQueueOffsetNumber", 0)?,
        ), // mtoon::v1 renderQueueOffsetNumber
        shade_color_factor: color_factor(decoder, material, value, "shadeColorFactor")
            .unwrap_or(defaults.shade_color_factor), // mtoon::v1 shadeColorFactor
        shade_multiply_texture: v1_texture_field(decoder, value, "shadeMultiplyTexture")?, // mtoon::v1 shadeMultiplyTexture
        shading_shift_factor: number(value, "shadingShiftFactor", defaults.shading_shift_factor)?, // mtoon::v1 shadingShiftFactor
        shading_shift_texture: v1_shading_shift(decoder, value)?, // mtoon::v1 shadingShiftTexture
        shading_toony_factor: number(value, "shadingToonyFactor", defaults.shading_toony_factor)?, // mtoon::v1 shadingToonyFactor
        gi_equalization_factor: number(
            value,
            "giEqualizationFactor",
            defaults.gi_equalization_factor,
        )?, // mtoon::v1 giEqualizationFactor
        matcap_factor: color_factor(decoder, material, value, "matcapFactor")
            .unwrap_or(defaults.matcap_factor), // mtoon::v1 matcapFactor
        matcap_texture: v1_texture_field(decoder, value, "matcapTexture")?, // mtoon::v1 matcapTexture
        parametric_rim_color_factor: color_factor(
            decoder,
            material,
            value,
            "parametricRimColorFactor",
        )
        .unwrap_or(defaults.parametric_rim_color_factor), // mtoon::v1 parametricRimColorFactor
        rim_multiply_texture: v1_texture_field(decoder, value, "rimMultiplyTexture")?, // mtoon::v1 rimMultiplyTexture
        rim_lighting_mix_factor: number(
            value,
            "rimLightingMixFactor",
            defaults.rim_lighting_mix_factor,
        )?, // mtoon::v1 rimLightingMixFactor
        parametric_rim_fresnel_power_factor: number(
            value,
            "parametricRimFresnelPowerFactor",
            defaults.parametric_rim_fresnel_power_factor,
        )?, // mtoon::v1 parametricRimFresnelPowerFactor
        parametric_rim_lift_factor: number(
            value,
            "parametricRimLiftFactor",
            defaults.parametric_rim_lift_factor,
        )?, // mtoon::v1 parametricRimLiftFactor
        outline_width_mode: v1_outline_mode(decoder, material, value), // mtoon::v1 outlineWidthMode
        outline_width_factor: number(value, "outlineWidthFactor", defaults.outline_width_factor)?, // mtoon::v1 outlineWidthFactor
        outline_width_multiply_texture: v1_texture_field(
            decoder,
            value,
            "outlineWidthMultiplyTexture",
        )?, // mtoon::v1 outlineWidthMultiplyTexture
        outline_color_factor: color_factor(decoder, material, value, "outlineColorFactor")
            .unwrap_or(defaults.outline_color_factor), // mtoon::v1 outlineColorFactor
        outline_lighting_mix_factor: number(
            value,
            "outlineLightingMixFactor",
            defaults.outline_lighting_mix_factor,
        )?, // mtoon::v1 outlineLightingMixFactor
        uv_animation_mask_texture: v1_texture_field(decoder, value, "uvAnimationMaskTexture")?, // mtoon::v1 uvAnimationMaskTexture
        uv_animation_scroll_x_speed_factor: number(
            value,
            "uvAnimationScrollXSpeedFactor",
            defaults.uv_animation_scroll_x_speed_factor,
        )?, // mtoon::v1 uvAnimationScrollXSpeedFactor
        uv_animation_scroll_y_speed_factor: number(
            value,
            "uvAnimationScrollYSpeedFactor",
            defaults.uv_animation_scroll_y_speed_factor,
        )?, // mtoon::v1 uvAnimationScrollYSpeedFactor
        uv_animation_rotation_speed_factor: number(
            value,
            "uvAnimationRotationSpeedFactor",
            defaults.uv_animation_rotation_speed_factor,
        )?, // mtoon::v1 uvAnimationRotationSpeedFactor
        // mtoon::v1 KHR_texture_transform of each texture info (`texture_transforms`)
        texture_transforms: v1_texture_transforms(value)?,
    })
}

/// A VRM 1.0 shading shift texture: a texture info object with `scale` (VRM default 1.0).
fn v1_shading_shift(
    decoder: &mut Decoder<'_>,
    value: &Value,
) -> Result<Option<AvatarMtoonShadingShiftTexture>> {
    let Some(info) = field(value, "shadingShiftTexture") else {
        return Ok(None);
    };
    let Some(texture) = v1_texture(decoder, "shadingShiftTexture", info)? else {
        return Ok(None);
    };
    Ok(Some(AvatarMtoonShadingShiftTexture {
        texture,
        scale: number(info, "scale", 1.0)?,
    }))
}

fn v1_texture_field(
    decoder: &mut Decoder<'_>,
    value: &Value,
    name: &str,
) -> Result<Option<TextureRef>> {
    match field(value, name) {
        Some(info) => v1_texture(decoder, name, info),
        None => Ok(None),
    }
}

/// A VRM 1.0 texture info object as a Potter texture on the layer its UV set selects (`None` is the render
/// layer). A set past the layer limit falls back to the render layer and is reported as a loss. A texture
/// transform is read by `v1_texture_transforms`.
fn v1_texture(decoder: &mut Decoder<'_>, name: &str, info: &Value) -> Result<Option<TextureRef>> {
    let set = texture_info_set(info);
    if usize::try_from(set).is_ok_and(|set| set >= crate::geom::uv_layers::MAX_LAYERS) {
        decoder.loss(
            crate::exchange::material_surface::feature_id::MESH_UV_LAYER,
            format!(
                "MToon texture `{name}` reads UV set {set}, past the layer limit, so it samples the render layer"
            ),
            SUGGESTION,
        );
    }
    let (image, interpolation) = texture_source(&decoder.index.root, index_value(&info["index"])?)?;
    Ok(Some(potter_texture(
        decoder.image_id(image)?,
        interpolation,
        imported_uv_layer(info),
    )))
}

/// The `KHR_texture_transform` of each VRM 1.0 texture slot that Potter stores, keyed by the slot field
/// name. The identity is not stored. The extension's `texCoord` selects the texture's layer, as in
/// `v1_texture`.
fn v1_texture_transforms(value: &Value) -> Result<BTreeMap<String, AvatarTextureTransform>> {
    let mut transforms = BTreeMap::new();
    for slot in V1_TEXTURE_SLOTS.iter().copied() {
        let Some(info) = field(value, slot) else {
            continue;
        };
        let Some(extension) = field(&info["extensions"], "KHR_texture_transform") else {
            continue;
        };
        let transform = AvatarTextureTransform {
            offset: vector2(extension, "offset", [0.0; 2])?,
            rotation: number(extension, "rotation", 0.0)?,
            scale: vector2(extension, "scale", [1.0; 2])?,
        };
        if !transform.is_identity() {
            transforms.insert(slot.to_owned(), transform);
        }
    }
    Ok(transforms)
}

/// Inverts `mtoon::v1` `outlineWidthMode` (`outline_mode_v1`).
fn v1_outline_mode(
    decoder: &mut Decoder<'_>,
    material: &Id,
    value: &Value,
) -> AvatarOutlineWidthMode {
    match field(value, "outlineWidthMode").and_then(Value::as_str) {
        None | Some("none") => AvatarOutlineWidthMode::None,
        Some("worldCoordinates") => AvatarOutlineWidthMode::WorldCoordinates,
        Some("screenCoordinates") => AvatarOutlineWidthMode::ScreenCoordinates,
        Some(other) => {
            decoder.loss(
                "vrm.mtoon.outlineWidthMode",
                format!("MToon outline mode `{other}` of material `{material}` is not supported"),
                SUGGESTION,
            );
            AvatarOutlineWidthMode::None
        }
    }
}

fn decode_v0(decoder: &mut Decoder<'_>) -> Result<BTreeMap<Id, AvatarMtoon>> {
    // `materialProperties` has one entry per glTF material, in the exporter's `doc.materials` order.
    let extension = decoder.extension;
    let properties = extension["materialProperties"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    let mut mtoon = BTreeMap::new();
    for (material_index, property) in properties.iter().enumerate() {
        let shader = property["shader"].as_str().unwrap_or_default();
        if shader == "VRM_USE_GLTFSHADER" {
            continue;
        }
        let material = decoder.material_id(material_index)?;
        if shader != "VRM/MToon" {
            decoder.loss(
                "vrm.mtoon.shader",
                format!(
                    "material `{material}` uses shader `{shader}`, which Potter cannot represent"
                ),
                SUGGESTION,
            );
            continue;
        }
        let value = decode_v0_material(decoder, &material, property)?;
        for material_id in decoder.material_ids(material_index)? {
            mtoon.insert(material_id, value.clone());
        }
    }
    Ok(mtoon)
}

/// Inverts `mtoon::v0` for one `VRM/MToon` material. Absent properties take the Potter defaults,
/// expressed in the legacy form where the exporter stores them that way.
fn decode_v0_material(
    decoder: &mut Decoder<'_>,
    material: &Id,
    property: &Value,
) -> Result<AvatarMtoon> {
    report_unrepresented_v0(decoder, material, property);
    let floats = &property["floatProperties"];
    let vectors = &property["vectorProperties"];
    let textures = &property["textureProperties"];
    let defaults = AvatarMtoon::default();
    let doc = decoder.doc;
    let alpha_mode = doc
        .materials
        .get(material)
        .map_or("opaque", |found| found.alpha_mode.as_str());

    // `_BlendMode` codes from `alpha_properties`: 3 is blend with z-write, 0-2 are not.
    let blend = enum_index(number(floats, "_BlendMode", 0.0)?, &[0.0, 1.0, 2.0, 3.0]);
    if blend.is_none() {
        decoder.loss(
            "vrm.mtoon._BlendMode",
            format!("MToon blend mode of material `{material}` is not a VRM 0.x blend code"),
            SUGGESTION,
        );
    }
    let transparent_with_z_write = blend == Some(3);

    // `renderQueue` = alpha queue base (`alpha_properties`) + `renderQueueOffsetNumber`.
    let base_queue = match (alpha_mode, transparent_with_z_write) {
        ("clip", _) => 2450,
        ("blend", true) => 2501,
        ("blend", false) => 3000,
        _ => 2000,
    };
    let render_queue_offset_number = match field(property, "renderQueue") {
        // VRM 0.x writes -1 for the shader's default queue, which is the alpha base with offset zero.
        Some(queue) if queue.as_i64() == Some(-1) => 0,
        Some(queue) => {
            let queue = queue
                .as_i64()
                .ok_or_else(|| import_error("VRM 0.x MToon renderQueue is not an integer"))?;
            render_queue_offset(
                decoder,
                material,
                "vrm.mtoon.renderQueue",
                queue.saturating_sub(base_queue),
            )
        }
        None => 0,
    };

    // `_ShadeShift`/`_ShadeToony` are the legacy form of `shading_shift_factor`/`shading_toony_factor`.
    let (default_shift, default_toony) =
        legacy_from_shading(defaults.shading_shift_factor, defaults.shading_toony_factor);
    let (shading_shift_factor, shading_toony_factor) = shading_from_legacy(
        number(floats, "_ShadeShift", default_shift)?,
        number(floats, "_ShadeToony", default_toony)?,
    );

    // `_OutlineColorMode` 0 is the fixed outline color, which is outline lighting mix 0.
    let outline_fixed = equal_f64(number(floats, "_OutlineColorMode", 1.0)?, 0.0);
    let outline_lighting_mix_factor = if outline_fixed {
        0.0
    } else {
        number(
            floats,
            "_OutlineLightingMix",
            defaults.outline_lighting_mix_factor,
        )?
    };

    Ok(AvatarMtoon {
        transparent_with_z_write,   // mtoon::v0 _BlendMode
        render_queue_offset_number, // mtoon::v0 renderQueue
        shade_color_factor: legacy_color(
            decoder,
            material,
            vectors,
            "_ShadeColor",
            defaults.shade_color_factor,
        ), // mtoon::v0 _ShadeColor
        shade_multiply_texture: v0_texture(decoder, textures, "_ShadeTexture")?, // mtoon::v0 _ShadeTexture
        shading_shift_factor,        // mtoon::v0 _ShadeShift
        shading_shift_texture: None, // mtoon::validate_v0 rejects it, so VRM 0.x never has one
        shading_toony_factor,        // mtoon::v0 _ShadeToony
        gi_equalization_factor: 1.0
            - number(
                floats,
                "_IndirectLightIntensity",
                1.0 - defaults.gi_equalization_factor,
            )?, // mtoon::v0 _IndirectLightIntensity
        matcap_factor: defaults.matcap_factor, // mtoon::validate_v0 requires [1,1,1] whenever matcap is set
        matcap_texture: v0_texture(decoder, textures, "_SphereAdd")?, // mtoon::v0 _SphereAdd
        parametric_rim_color_factor: legacy_color(
            decoder,
            material,
            vectors,
            "_RimColor",
            defaults.parametric_rim_color_factor,
        ), // mtoon::v0 _RimColor
        rim_multiply_texture: v0_texture(decoder, textures, "_RimTexture")?, // mtoon::v0 _RimTexture
        rim_lighting_mix_factor: number(
            floats,
            "_RimLightingMix",
            defaults.rim_lighting_mix_factor,
        )?, // mtoon::v0 _RimLightingMix
        parametric_rim_fresnel_power_factor: number(
            floats,
            "_RimFresnelPower",
            defaults.parametric_rim_fresnel_power_factor,
        )?, // mtoon::v0 _RimFresnelPower
        parametric_rim_lift_factor: number(
            floats,
            "_RimLift",
            defaults.parametric_rim_lift_factor,
        )?, // mtoon::v0 _RimLift
        outline_width_mode: v0_outline_mode(decoder, material, floats)?, // mtoon::v0 _OutlineWidthMode
        // `_OutlineWidth` is the factor times 100 (`vector_properties`).
        outline_width_factor: number(
            floats,
            "_OutlineWidth",
            defaults.outline_width_factor * 100.0,
        )? / 100.0, // mtoon::v0 _OutlineWidth
        outline_width_multiply_texture: v0_channel_texture(
            decoder,
            material,
            textures,
            "_OutlineWidthTexture",
            1,
        )?, // mtoon::v0 _OutlineWidthTexture (exporter remaps G)
        outline_color_factor: legacy_color(
            decoder,
            material,
            vectors,
            "_OutlineColor",
            defaults.outline_color_factor,
        ), // mtoon::v0 _OutlineColor
        outline_lighting_mix_factor, // mtoon::v0 _OutlineColorMode and _OutlineLightingMix
        uv_animation_mask_texture: v0_channel_texture(
            decoder,
            material,
            textures,
            "_UvAnimMaskTexture",
            2,
        )?, // mtoon::v0 _UvAnimMaskTexture (exporter remaps B)
        uv_animation_scroll_x_speed_factor: number(
            floats,
            "_UvAnimScrollX",
            defaults.uv_animation_scroll_x_speed_factor,
        )?, // mtoon::v0 _UvAnimScrollX
        uv_animation_scroll_y_speed_factor: -number(
            floats,
            "_UvAnimScrollY",
            -defaults.uv_animation_scroll_y_speed_factor,
        )?, // mtoon::v0 _UvAnimScrollY (negated)
        uv_animation_rotation_speed_factor: number(
            floats,
            "_UvAnimRotation",
            defaults.uv_animation_rotation_speed_factor / TAU,
        )? * TAU, // mtoon::v0 _UvAnimRotation (turns)
        texture_transforms: BTreeMap::new(), // VRM 0.x has no per-texture transform
    })
}

/// Reports VRM 0.x `MToon` values Potter cannot store: unknown properties, and known ones whose value
/// differs from what the exporter writes.
fn report_unrepresented_v0(decoder: &mut Decoder<'_>, material: &Id, property: &Value) {
    let mut names: Vec<&str> = Vec::new();
    if let Some(floats) = property["floatProperties"].as_object() {
        for (name, value) in floats {
            if V0_HANDLED_FLOATS.contains(&name.as_str()) {
                continue;
            }
            let default = unrepresented_default(name);
            if !value
                .as_f64()
                .is_some_and(|value| equal_f64(value, default))
            {
                names.push(name);
            }
        }
    }
    if let Some(vectors) = property["vectorProperties"].as_object() {
        for (name, value) in vectors {
            if V0_HANDLED_VECTORS.contains(&name.as_str()) || is_zero(value) {
                continue;
            }
            // The VRM Add-on names a texture's (offset, scale) vector after the texture and writes
            // `[0, 0, 1, 1]` when the texture is untransformed; that is the identity, not a loss.
            let texture_named =
                V0_HANDLED_TEXTURES.contains(&name.as_str()) || name.ends_with("Texture");
            if texture_named && is_vector(value, &[0.0, 0.0, 1.0, 1.0]) {
                continue;
            }
            // The main texture transform (`_MainTex` or `_MainTex_ST`) is reported below.
            if name == "_MainTex" {
                continue;
            }
            names.push(name);
        }
    }
    // The main texture transform is applied to the base color texture when the material has one. It is
    // reported when there is no base texture to carry it, or when it replaced the texture's own transform.
    let vectors = &property["vectorProperties"];
    let main_present =
        field(vectors, "_MainTex").is_some() || field(vectors, "_MainTex_ST").is_some();
    if main_present {
        let main_source = if field(vectors, "_MainTex").is_some() {
            "_MainTex"
        } else {
            "main_tex_transform"
        };
        let non_identity =
            main_texture_transform(vectors).is_some_and(|transform| !transform.is_identity());
        let applied = decoder.index.main_texture_transforms.contains(material);
        let replaced = decoder.index.discarded_base_transforms.contains(material);
        if (non_identity && !applied) || replaced {
            names.push(main_source);
        }
    }
    // Potter reads `_MainTex_ST` only when `_MainTex` is absent, so one beside `_MainTex` is not represented.
    if field(vectors, "_MainTex").is_some()
        && field(vectors, "_MainTex_ST")
            .is_some_and(|value| !is_vector(value, &[1.0, 1.0, 0.0, 0.0]))
    {
        names.push("main_tex_transform");
    }
    if let Some(textures) = property["textureProperties"].as_object() {
        for (name, value) in textures {
            if !V0_HANDLED_TEXTURES.contains(&name.as_str()) && !value.is_null() {
                names.push(name);
            }
        }
    }
    for name in names {
        decoder.loss(
            &format!("vrm.mtoon.{name}"),
            format!("VRM 0.x MToon setting `{name}` of material `{material}` is not represented by Potter"),
            SUGGESTION,
        );
    }
}

fn unrepresented_default(name: &str) -> f64 {
    V0_UNREPRESENTED_FLOATS
        .iter()
        .find(|(known, _)| *known == name)
        .map_or(0.0, |(_, default)| *default)
}

/// Inverts `mtoon::v0` `_OutlineWidthMode` (0 none, 1 world, 2 screen).
fn v0_outline_mode(
    decoder: &mut Decoder<'_>,
    material: &Id,
    floats: &Value,
) -> Result<AvatarOutlineWidthMode> {
    match enum_index(number(floats, "_OutlineWidthMode", 0.0)?, &[0.0, 1.0, 2.0]) {
        Some(0) => Ok(AvatarOutlineWidthMode::None),
        Some(1) => Ok(AvatarOutlineWidthMode::WorldCoordinates),
        Some(2) => Ok(AvatarOutlineWidthMode::ScreenCoordinates),
        _ => {
            decoder.loss(
                "vrm.mtoon._OutlineWidthMode",
                format!("MToon outline width mode of material `{material}` is not a VRM 0.x code"),
                SUGGESTION,
            );
            Ok(AvatarOutlineWidthMode::None)
        }
    }
}

/// A VRM 0.x texture property. The value is a glTF texture index, as `add_texture` writes it.
fn v0_texture(decoder: &Decoder<'_>, textures: &Value, name: &str) -> Result<Option<TextureRef>> {
    let Some(value) = field(textures, name) else {
        return Ok(None);
    };
    let (image, interpolation) = texture_source(&decoder.index.root, index_value(value)?)?;
    Ok(Some(potter_texture(
        decoder.image_id(image)?,
        interpolation,
        None,
    )))
}

/// A VRM 0.x texture the exporter channel-remapped to gray (`texture_info_channel`). The shader reads
/// `channel` (1 = G, 2 = B) in VRM 1.0, so only a Potter copy of that channel can be inverted.
fn v0_channel_texture(
    decoder: &mut Decoder<'_>,
    material: &Id,
    textures: &Value,
    name: &str,
    channel: u64,
) -> Result<Option<TextureRef>> {
    let Some(value) = field(textures, name) else {
        return Ok(None);
    };
    let (copy, interpolation) = texture_source(&decoder.index.root, index_value(value)?)?;
    let source = remapped_source(&decoder.index.root, copy, channel);
    if let Some(image) = source {
        Ok(Some(potter_texture(
            decoder.image_id(image)?,
            interpolation,
            None,
        )))
    } else {
        decoder.loss(
            &format!("vrm.mtoon.{name}"),
            format!("MToon texture `{name}` of material `{material}` is not a Potter copy of channel {channel}"),
            SUGGESTION,
        );
        Ok(None)
    }
}

/// The Potter image behind a gray channel copy (`extras.potter.remapped_channel` equal to `channel`),
/// or `None` when `image` is not such a copy. When the file also carries the source image, that image
/// is used: it has the same channel value, without the gray copy in its place.
fn remapped_source(root: &Value, image: usize, channel: u64) -> Option<usize> {
    let extras = &root["images"][image]["extras"]["potter"];
    if extras["remapped_channel"].as_u64() != Some(channel) {
        return None;
    }
    let id = extras["id"].as_str()?;
    let original = root["images"].as_array()?.iter().position(|candidate| {
        let extras = &candidate["extras"]["potter"];
        extras["id"].as_str() == Some(id) && extras["remapped_channel"].is_null()
    });
    Some(original.unwrap_or(image))
}

/// The glTF image a texture reads, and the interpolation its sampler implies (`gltf.rs` `import_texture_ref`).
fn texture_source(root: &Value, texture_index: usize) -> Result<(usize, ImageInterpolation)> {
    let texture = root["textures"]
        .get(texture_index)
        .ok_or_else(|| import_error("VRM MToon references a missing glTF texture"))?;
    let image = index_value(&texture["source"])?;
    let nearest = texture["sampler"]
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| root["samplers"].get(index))
        .is_some_and(|sampler| {
            sampler["magFilter"].as_u64() == Some(9728)
                || sampler["minFilter"].as_u64() == Some(9728)
        });
    let interpolation = if nearest {
        ImageInterpolation::Closest
    } else {
        ImageInterpolation::Linear
    };
    Ok((image, interpolation))
}

/// A `MToon` texture on `uv_layer` (`None` is the render layer). VRM 0.x textures have no `texCoord` and pass
/// `None`.
fn potter_texture(
    image: Id,
    interpolation: ImageInterpolation,
    uv_layer: Option<String>,
) -> TextureRef {
    TextureRef {
        image,
        uv_layer,
        interpolation,
    }
}

/// Inverts `mtoon::legacy_shading_values`. With legacy shift `d` and toony `t`, the VRM 1.0 toony
/// factor is `T = (t(1-d) + 1 + d) / 2`, and the shift factor is `T - 1 - d`.
fn shading_from_legacy(shift: f64, toony: f64) -> (f64, f64) {
    let toony_factor = (toony * (1.0 - shift) + 1.0 + shift) / 2.0;
    (toony_factor - 1.0 - shift, toony_factor)
}

/// `mtoon::legacy_shading_values`, used only to express the Potter defaults as legacy values.
fn legacy_from_shading(shift_factor: f64, toony_factor: f64) -> (f64, f64) {
    let shift = toony_factor - 1.0 - shift_factor;
    (shift, (2.0 * toony_factor - (1.0 + shift)) / (1.0 - shift))
}

/// A `MToon` color factor in Potter's 0..1 range, or `None` when the field is absent. A component
/// outside 0..1 is clamped and a malformed value is dropped, each reported as `vrm.mtoon.<key>` for
/// this material, so one bad color never discards the rest of the avatar.
fn color_factor(
    decoder: &mut Decoder<'_>,
    material: &Id,
    object: &Value,
    key: &str,
) -> Option<[f64; 3]> {
    let value = field(object, key)?;
    let components = value
        .as_array()
        .filter(|items| items.len() >= 3)
        .and_then(|items| {
            items
                .iter()
                .take(3)
                .map(Value::as_f64)
                .collect::<Option<Vec<f64>>>()
        });
    let Some(components) = components else {
        decoder.loss(
            &format!("vrm.mtoon.{key}"),
            format!("MToon color `{key}` of material `{material}` is not a color vector"),
            SUGGESTION,
        );
        return None;
    };
    let mut color = [0.0; 3];
    let mut out_of_range = false;
    for (slot, component) in color.iter_mut().zip(components) {
        out_of_range |= !(0.0..=1.0).contains(&component);
        *slot = component.clamp(0.0, 1.0);
    }
    if out_of_range {
        decoder.loss(
            &format!("vrm.mtoon.{key}"),
            format!("MToon color `{key}` of material `{material}` has a component outside 0 to 1; it is clamped"),
            SUGGESTION,
        );
    }
    Some(color)
}

/// Inverts `mtoon::legacy_color_component` for a VRM 0.x color: the stored value is clamped to 0..1,
/// then gamma-decoded. `default` is already linear.
fn legacy_color(
    decoder: &mut Decoder<'_>,
    material: &Id,
    vectors: &Value,
    key: &str,
    default: [f64; 3],
) -> [f64; 3] {
    color_factor(decoder, material, vectors, key).map_or(default, |color| {
        color.map(|component| component.powf(LEGACY_GAMMA))
    })
}

/// A render queue offset Potter can hold (-9..=9). Anything else is a loss, and the offset is 0.
fn render_queue_offset(
    decoder: &mut Decoder<'_>,
    material: &Id,
    feature_id: &str,
    offset: i64,
) -> i32 {
    match i32::try_from(offset) {
        Ok(offset) if (-9..=9).contains(&offset) => offset,
        _ => {
            decoder.loss(
                feature_id,
                format!("MToon render queue offset of material `{material}` is outside -9..=9"),
                SUGGESTION,
            );
            0
        }
    }
}

/// The position of `value` among VRM 0.x enum codes, which are stored as floats.
fn enum_index(value: f64, codes: &[f64]) -> Option<usize> {
    codes.iter().position(|code| equal_f64(value, *code))
}

fn index_value(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| import_error("VRM MToon has an invalid glTF index"))
}

fn is_zero(value: &Value) -> bool {
    value.as_array().is_some_and(|items| {
        items
            .iter()
            .all(|item| item.as_f64().is_some_and(|item| equal_f64(item, 0.0)))
    })
}

fn is_vector(value: &Value, expected: &[f64]) -> bool {
    value.as_array().is_some_and(|items| {
        items.len() == expected.len()
            && items.iter().zip(expected).all(|(item, expected)| {
                item.as_f64().is_some_and(|item| equal_f64(item, *expected))
            })
    })
}

/// A member of a JSON object, or `None` when it is absent or `null`.
fn field<'v>(object: &'v Value, key: &str) -> Option<&'v Value> {
    object.get(key).filter(|value| !value.is_null())
}

fn number(object: &Value, key: &str, default: f64) -> Result<f64> {
    match field(object, key) {
        Some(value) => value
            .as_f64()
            .ok_or_else(|| import_error(format!("VRM MToon field `{key}` is not a number"))),
        None => Ok(default),
    }
}

/// A two-component vector member, or `default` when it is absent.
fn vector2(object: &Value, key: &str, default: [f64; 2]) -> Result<[f64; 2]> {
    let Some(value) = field(object, key) else {
        return Ok(default);
    };
    let components = value.as_array().and_then(|items| match items.as_slice() {
        [x, y] => Some([x.as_f64()?, y.as_f64()?]),
        _ => None,
    });
    components.ok_or_else(|| {
        import_error(format!(
            "VRM MToon field `{key}` is not a 2-component vector"
        ))
    })
}

fn integer(object: &Value, key: &str, default: i64) -> Result<i64> {
    match field(object, key) {
        Some(value) => value
            .as_i64()
            .ok_or_else(|| import_error(format!("VRM MToon field `{key}` is not an integer"))),
        None => Ok(default),
    }
}

fn boolean(object: &Value, key: &str, default: bool) -> Result<bool> {
    match field(object, key) {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| import_error(format!("VRM MToon field `{key}` is not a boolean"))),
        None => Ok(default),
    }
}

/// The VRM 0.x `MToon` main texture transform of glTF material `material_index`, if the material has one.
/// The glTF import replaces the base color texture's own `KHR_texture_transform` with it, as the VRM add-on does.
pub(super) fn v0_main_texture_transform(
    root: &Value,
    material_index: usize,
) -> Option<TextureTransform> {
    let property = &root["extensions"]["VRM"]["materialProperties"][material_index];
    if property["shader"].as_str() != Some("VRM/MToon") {
        return None;
    }
    main_texture_transform(&property["vectorProperties"])
}

/// The main texture transform of a VRM 0.x material. The VRM Add-on reads `_MainTex` as (offset x, stored
/// offset y, scale x, scale y). Potter writes `_MainTex_ST` as (scale x, scale y, offset x, stored offset y)
/// and reads it when `_MainTex` is absent. As the add-on does, a scale whose magnitude is below
/// `f64::EPSILON` is 1, and the stored offset y is `1 - offset y - scale y`.
fn main_texture_transform(vectors: &Value) -> Option<TextureTransform> {
    let numbers = |value: &Value| -> Option<[f64; 4]> {
        let items = value.as_array().filter(|items| items.len() == 4)?;
        let mut output = [0.0; 4];
        for (slot, item) in output.iter_mut().zip(items) {
            *slot = item.as_f64()?;
        }
        Some(output)
    };
    let (offset_x, stored_y, scale_x, scale_y) =
        if let Some([offset_x, stored_y, scale_x, scale_y]) = numbers(&vectors["_MainTex"]) {
            (offset_x, stored_y, scale_x, scale_y)
        } else {
            let [scale_x, scale_y, offset_x, stored_y] = numbers(&vectors["_MainTex_ST"])?;
            (offset_x, stored_y, scale_x, scale_y)
        };
    let guard = |scale: f64| {
        if scale.abs() < f64::EPSILON {
            1.0
        } else {
            scale
        }
    };
    let scale = [guard(scale_x), guard(scale_y)];
    Some(TextureTransform {
        offset: [offset_x, 1.0 - stored_y - scale[1]],
        rotation: 0.0,
        scale,
    })
}
