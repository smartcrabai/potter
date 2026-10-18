//! VRM expressions. VRM 1.0 `expressions` inverts `vrm1_expressions` and `expression_v1`; VRM 0.x
//! `blendShapeMaster` inverts `vrm0_expressions`.

use std::collections::{BTreeMap, btree_map::Entry};

use serde_json::Value;

use crate::{
    error::Result,
    model::{
        AvatarExpression, AvatarExpressionOverride, AvatarMaterialColorBind,
        AvatarMaterialColorType, AvatarMorphTargetBind, AvatarMtoon, AvatarTextureTransformBind,
        AvatarVersion, Id,
    },
};

use super::Decoder;

const SUGGESTION: &str = "use --allow-lossy to import the file without this expression data";

/// Decodes the expressions. `mtoon` lists the materials with `MToon` settings; `MToon` color binds
/// need one.
pub(super) fn decode(
    decoder: &mut Decoder<'_>,
    mtoon: &BTreeMap<Id, AvatarMtoon>,
) -> Result<BTreeMap<String, AvatarExpression>> {
    let extension: &Value = decoder.extension;
    let decoded = match decoder.version {
        AvatarVersion::V1 => decode_v1(decoder, extension, mtoon)?,
        AvatarVersion::V0 => decode_v0(decoder, extension, mtoon),
    };
    let mut expressions = BTreeMap::new();
    for (name, expression) in decoded {
        if name.trim().is_empty() {
            decoder.loss("vrm.expression", "expression name is empty", SUGGESTION);
        } else {
            match expressions.entry(name) {
                Entry::Vacant(entry) => {
                    entry.insert(expression);
                }
                Entry::Occupied(entry) => {
                    decoder.loss(
                        &format!("vrm.expression.{}", entry.key()),
                        "expression name is duplicated",
                        SUGGESTION,
                    );
                }
            }
        }
    }
    Ok(expressions)
}

fn decode_v1(
    decoder: &mut Decoder<'_>,
    extension: &Value,
    mtoon: &BTreeMap<Id, AvatarMtoon>,
) -> Result<Vec<(String, AvatarExpression)>> {
    let mut decoded = Vec::new();
    for group in ["preset", "custom"] {
        let Some(entries) = extension["expressions"][group].as_object() else {
            continue;
        };
        for (name, value) in entries {
            let expression = v1_expression(decoder, name, value, mtoon)?;
            decoded.push((name.clone(), expression));
        }
    }
    Ok(decoded)
}

/// Inverts `expression_v1`.
fn v1_expression(
    decoder: &mut Decoder<'_>,
    name: &str,
    value: &Value,
    mtoon: &BTreeMap<Id, AvatarMtoon>,
) -> Result<AvatarExpression> {
    let feature = format!("vrm.expression.{name}");
    let mut expression = AvatarExpression {
        is_binary: value["isBinary"].as_bool().unwrap_or(false),
        override_blink: override_mode(decoder, &feature, &value["overrideBlink"]),
        override_mouth: override_mode(decoder, &feature, &value["overrideMouth"]),
        override_look_at: override_mode(decoder, &feature, &value["overrideLookAt"]),
        ..AvatarExpression::default()
    };
    for bind in items(&value["morphTargetBinds"]) {
        let (Some(node), Some(target)) = (index_of(&bind["node"]), index_of(&bind["index"])) else {
            decoder.loss(&feature, "morph target bind is malformed", SUGGESTION);
            continue;
        };
        let mesh = index_of(&decoder.gltf_node(node)?["mesh"]);
        let (Some(mesh), Some(id)) = (mesh, node_id(decoder, node)) else {
            decoder.loss(&feature, "morph target node has no mesh", SUGGESTION);
            continue;
        };
        let weight = bind["weight"].as_f64().unwrap_or(1.0);
        if let Some(bind) = morph_bind(decoder, &feature, id, mesh, target, weight) {
            expression.morph_target_binds.push(bind);
        }
    }
    for bind in items(&value["materialColorBinds"]) {
        let color_type = bind["type"].as_str().and_then(color_type_v1);
        let materials = materials_of(decoder, &bind["material"]);
        let (Some(color_type), Some(materials), Some(target)) = (
            color_type,
            materials,
            number_array::<4>(&bind["targetValue"]),
        ) else {
            decoder.loss(
                &feature,
                "material color bind is malformed or references a missing material",
                SUGGESTION,
            );
            continue;
        };
        let Some(material) = materials.first().cloned() else {
            continue;
        };
        if let Some(bind) = color_bind(decoder, &feature, mtoon, material, color_type, target) {
            expression.material_color_binds.push(bind.clone());
            for material in materials.into_iter().skip(1) {
                let mut variant_bind = bind.clone();
                variant_bind.material = material;
                expression.material_color_binds.push(variant_bind);
            }
        }
    }
    for bind in items(&value["textureTransformBinds"]) {
        let materials = materials_of(decoder, &bind["material"]);
        let (Some(materials), Some(scale), Some(offset)) = (
            materials,
            number_array::<2>(&bind["scale"]),
            number_array::<2>(&bind["offset"]),
        ) else {
            decoder.loss(
                &feature,
                "texture transform bind is malformed or references a missing material",
                SUGGESTION,
            );
            continue;
        };
        for material in materials {
            expression
                .texture_transform_binds
                .push(AvatarTextureTransformBind {
                    material,
                    scale,
                    offset,
                });
        }
    }
    Ok(expression)
}

/// Inverts `legacy_expression_preset` in `vrm.rs`: the VRM 1.0 preset a VRM 0.x preset name stands for.
fn v1_preset_for_legacy(preset: &str) -> Option<&'static str> {
    Some(match preset {
        "neutral" => "neutral",
        "a" => "aa",
        "i" => "ih",
        "u" => "ou",
        "e" => "ee",
        "o" => "oh",
        "blink" => "blink",
        "blink_l" => "blinkLeft",
        "blink_r" => "blinkRight",
        "joy" => "happy",
        "angry" => "angry",
        "sorrow" => "sad",
        "fun" => "relaxed",
        "lookup" => "lookUp",
        "lookdown" => "lookDown",
        "lookleft" => "lookLeft",
        "lookright" => "lookRight",
        _ => return None,
    })
}

fn decode_v0(
    decoder: &mut Decoder<'_>,
    extension: &Value,
    mtoon: &BTreeMap<Id, AvatarMtoon>,
) -> Vec<(String, AvatarExpression)> {
    let mut decoded = Vec::new();
    for group in items(&extension["blendShapeMaster"]["blendShapeGroups"]) {
        let Some(name) = group["name"].as_str() else {
            decoder.loss(
                "vrm.expression",
                "blend shape group has no name",
                SUGGESTION,
            );
            continue;
        };
        let name = v1_preset_for_legacy(group["presetName"].as_str().unwrap_or_default())
            .unwrap_or(name)
            .to_owned();
        let feature = format!("vrm.expression.{name}");
        let mut expression = AvatarExpression {
            is_binary: group["isBinary"].as_bool().unwrap_or(false),
            ..AvatarExpression::default()
        };
        for bind in items(&group["binds"]) {
            let (Some(mesh), Some(target)) = (index_of(&bind["mesh"]), index_of(&bind["index"]))
            else {
                decoder.loss(&feature, "blend shape bind is malformed", SUGGESTION);
                continue;
            };
            // Inverts the exporter's `weight * 100`.
            let weight = bind["weight"].as_f64().unwrap_or(100.0) / 100.0;
            let nodes = decoder.nodes_using_mesh(mesh);
            let [node] = nodes.as_slice() else {
                decoder.loss(
                    &feature,
                    format!("morph target mesh {mesh} is not used by exactly one node"),
                    SUGGESTION,
                );
                continue;
            };
            let Some(id) = node_id(decoder, *node) else {
                decoder.loss(&feature, "morph target node was not imported", SUGGESTION);
                continue;
            };
            if let Some(bind) = morph_bind(decoder, &feature, id, mesh, target, weight) {
                expression.morph_target_binds.push(bind);
            }
        }
        for value in items(&group["materialValues"]) {
            let property = value["propertyName"].as_str().unwrap_or_default();
            let Some(target) = number_array::<4>(&value["targetValue"]) else {
                decoder.loss(&feature, "material value target is malformed", SUGGESTION);
                continue;
            };
            let Some(materials) =
                legacy_material(decoder, &feature, value["materialName"].as_str())
            else {
                continue;
            };
            let Some(material) = materials.first().cloned() else {
                continue;
            };
            if property == "_MainTex_ST" {
                // Inverts `vrm0_expressions`: the stored offset y is `1 - offset_y - scale_y`.
                let [scale_x, scale_y, offset_x, stored_y] = target;
                for material in materials {
                    expression
                        .texture_transform_binds
                        .push(AvatarTextureTransformBind {
                            material,
                            scale: [scale_x, scale_y],
                            offset: [offset_x, 1.0 - stored_y - scale_y],
                        });
                }
            } else if let Some(color_type) = color_type_v0(property) {
                if let Some(bind) =
                    color_bind(decoder, &feature, mtoon, material, color_type, target)
                {
                    expression.material_color_binds.push(bind.clone());
                    for material in materials.into_iter().skip(1) {
                        let mut variant_bind = bind.clone();
                        variant_bind.material = material;
                        expression.material_color_binds.push(variant_bind);
                    }
                }
            } else {
                decoder.loss(
                    &feature,
                    format!("material property `{property}` is not represented by Potter"),
                    SUGGESTION,
                );
            }
        }
        decoded.push((name.clone(), expression));
    }
    decoded
}

/// Inverts the exporter's `overrideBlink` / `overrideMouth` / `overrideLookAt` names.
fn override_mode(
    decoder: &mut Decoder<'_>,
    feature: &str,
    value: &Value,
) -> AvatarExpressionOverride {
    match value.as_str() {
        None | Some("none") => AvatarExpressionOverride::None,
        Some("block") => AvatarExpressionOverride::Block,
        Some("blend") => AvatarExpressionOverride::Blend,
        Some(other) => {
            decoder.loss(
                feature,
                format!("expression override `{other}` is unknown"),
                SUGGESTION,
            );
            AvatarExpressionOverride::None
        }
    }
}

/// A morph target bind on `node`, whose glTF mesh is `mesh`. Records a loss and returns `None` when
/// the weight or shape key is unrepresentable.
fn morph_bind(
    decoder: &mut Decoder<'_>,
    feature: &str,
    node: Id,
    mesh: usize,
    target: usize,
    weight: f64,
) -> Option<AvatarMorphTargetBind> {
    if !(0.0..=1.0).contains(&weight) {
        decoder.loss(feature, "morph target weight is outside 0 to 1", SUGGESTION);
        return None;
    }
    let Ok(shape_key) = decoder.morph_shape_key(mesh, target) else {
        decoder.loss(
            feature,
            format!("morph target {target} of glTF mesh {mesh} was not imported"),
            SUGGESTION,
        );
        return None;
    };
    Some(AvatarMorphTargetBind {
        node,
        shape_key,
        weight,
    })
}

/// A material color bind. MToon-only colors need `MToon` settings on the material, as
/// `validate_avatar_settings` requires.
fn color_bind(
    decoder: &mut Decoder<'_>,
    feature: &str,
    mtoon: &BTreeMap<Id, AvatarMtoon>,
    material: Id,
    color_type: AvatarMaterialColorType,
    target_value: [f64; 4],
) -> Option<AvatarMaterialColorBind> {
    if needs_mtoon(color_type) && !mtoon.contains_key(&material) {
        decoder.loss(
            feature,
            "MToon color bind targets a material without MToon settings",
            SUGGESTION,
        );
        return None;
    }
    Some(AvatarMaterialColorBind {
        material,
        color_type,
        target_value,
    })
}

/// Resolves a VRM 0.x `materialName` against original glTF materials only.
fn legacy_material(
    decoder: &mut Decoder<'_>,
    feature: &str,
    name: Option<&str>,
) -> Option<Vec<Id>> {
    let Some(name) = name else {
        decoder.loss(feature, "material value has no materialName", SUGGESTION);
        return None;
    };
    let matches: Vec<usize> = decoder
        .index
        .material_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            decoder
                .doc
                .materials
                .get(id)
                .filter(|material| material.name == name)
                .map(|_| index)
        })
        .collect();
    match matches.as_slice() {
        [index] => decoder.material_ids(*index).ok(),
        [] => {
            decoder.loss(
                feature,
                format!("material `{name}` does not exist"),
                SUGGESTION,
            );
            None
        }
        _ => {
            decoder.loss(
                feature,
                format!("material name `{name}` is duplicated"),
                SUGGESTION,
            );
            None
        }
    }
}

fn needs_mtoon(color_type: AvatarMaterialColorType) -> bool {
    matches!(
        color_type,
        AvatarMaterialColorType::ShadeColor
            | AvatarMaterialColorType::MatcapColor
            | AvatarMaterialColorType::RimColor
            | AvatarMaterialColorType::OutlineColor
    )
}

fn color_type_v1(name: &str) -> Option<AvatarMaterialColorType> {
    match name {
        "color" => Some(AvatarMaterialColorType::Color),
        "emissionColor" => Some(AvatarMaterialColorType::EmissionColor),
        "shadeColor" => Some(AvatarMaterialColorType::ShadeColor),
        "matcapColor" => Some(AvatarMaterialColorType::MatcapColor),
        "rimColor" => Some(AvatarMaterialColorType::RimColor),
        "outlineColor" => Some(AvatarMaterialColorType::OutlineColor),
        _ => None,
    }
}

/// Inverts `legacy_material_color_property`.
fn color_type_v0(property: &str) -> Option<AvatarMaterialColorType> {
    match property {
        "_Color" => Some(AvatarMaterialColorType::Color),
        "_EmissionColor" => Some(AvatarMaterialColorType::EmissionColor),
        "_ShadeColor" => Some(AvatarMaterialColorType::ShadeColor),
        "_SphereAdd" => Some(AvatarMaterialColorType::MatcapColor),
        "_RimColor" => Some(AvatarMaterialColorType::RimColor),
        "_OutlineColor" => Some(AvatarMaterialColorType::OutlineColor),
        _ => None,
    }
}

/// The Potter node ID of glTF node `node`.
pub(super) fn node_id(decoder: &Decoder<'_>, node: usize) -> Option<Id> {
    decoder.index.node_ids.get(node).cloned()
}

/// The Potter materials of a glTF material index JSON value, original followed by color variants.
fn materials_of(decoder: &Decoder<'_>, value: &Value) -> Option<Vec<Id>> {
    index_of(value).and_then(|index| decoder.material_ids(index).ok())
}

/// The elements of a JSON array, or none when the value is absent or not an array.
pub(super) fn items(value: &Value) -> &[Value] {
    value.as_array().map_or(&[][..], Vec::as_slice)
}

/// A JSON non-negative integer as an index.
pub(super) fn index_of(value: &Value) -> Option<usize> {
    usize::try_from(value.as_u64()?).ok()
}

/// A JSON array of exactly `N` numbers.
pub(super) fn number_array<const N: usize>(value: &Value) -> Option<[f64; N]> {
    let values = value.as_array()?;
    if values.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(values) {
        *slot = item.as_f64()?;
    }
    Some(out)
}
