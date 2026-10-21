// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org): `psys_get_texture` (`particle.cc`),
// `RE_texture_evaluate` placement (`texture_procedural.cc`), and `texture_value_blend`.

//! Blender's particle texture influences (`psys_get_texture`).
//!
//! A texture slot mapped to a particle setting replaces that setting at birth: emission time
//! (`PAMAP_TIME`), density (`PAMAP_DENS`, which decides whether the particle exists), size
//! (`PAMAP_SIZE`), and life (`PAMAP_LIFE`). Blender samples the texture at the particle's birth
//! point through the slot's coordinates, placement, and blend mode. Channels and coordinate types
//! that are not modelled here are reported as import losses by the `.blend` importer.

use std::collections::BTreeMap;

use glam::{DMat4, DVec3};
use serde_json::Value;

use crate::{
    blender_rng::BlenderRng,
    error::{ErrorCode, PotError, Result},
    geom::legacy_texture::{self, ResolvedTexture},
    model::Id,
};

/// The particle setting a texture slot maps (Blender's `PAMAP_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channel {
    Time,
    Life,
    Density,
    Size,
}

/// Blender's `MTEX_*` blend modes. `Zero` stands for hue, saturation, value, and colour, which
/// `texture_value_blend` does not handle and returns zero for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blend {
    Mix,
    Multiply,
    Add,
    Subtract,
    Divide,
    Darken,
    Difference,
    Lighten,
    Screen,
    Overlay,
    SoftLight,
    LinearLight,
    Zero,
}

/// Blender's `texture_value_blend` (`texture_procedural.cc`). The caller passes the slot's
/// default value as `tex` and the sampled texture value as `fact`, as `SET_PARTICLE_TEXTURE` does.
fn texture_value_blend(tex: f64, out: f64, fact: f64, facg: f64, blend: Blend) -> f64 {
    let flip = facg < 0.0;
    let facg = facg.abs();
    let mut fact = fact * facg;
    let mut facm = 1.0 - fact;
    if flip {
        std::mem::swap(&mut fact, &mut facm);
    }
    match blend {
        Blend::Mix => fact * tex + facm * out,
        Blend::Multiply => (1.0 - facg + fact * tex) * out,
        Blend::Screen => 1.0 - (1.0 - facg + fact * (1.0 - tex)) * (1.0 - out),
        Blend::Overlay => {
            let facm = 1.0 - facg;
            if out < 0.5 {
                out * (facm + 2.0 * fact * tex)
            } else {
                1.0 - (facm + 2.0 * fact * (1.0 - tex)) * (1.0 - out)
            }
        }
        Blend::Subtract => -fact * tex + out,
        Blend::Add => fact * tex + out,
        Blend::Divide => {
            if tex != 0.0 {
                facm * out + fact * out / tex
            } else {
                0.0
            }
        }
        Blend::Difference => facm * out + fact * (tex - out).abs(),
        Blend::Darken => out.min(tex) * fact + out * facm,
        Blend::Lighten => out.max(tex) * fact + out * facm,
        Blend::SoftLight => {
            let scf = 1.0 - (1.0 - tex) * (1.0 - out);
            facm * out + fact * ((1.0 - out) * tex * out) + out * scf
        }
        Blend::LinearLight => {
            if tex > 0.5 {
                out + fact * (2.0 * (tex - 0.5))
            } else {
                out + fact * (2.0 * tex - 1.0)
            }
        }
        Blend::Zero => 0.0,
    }
}

/// Where a slot's texture coordinates come from.
#[derive(Clone, Copy)]
enum Coords {
    /// `TEXCO_GLOB`: the birth point in world space (`pa->state.co`).
    Global,
    /// `TEXCO_OBJECT`: the birth point in the slot object's space (`world_to_object`).
    Object(DMat4),
}

/// A texture slot that drives at least one supported particle setting.
pub struct TextureSlot<'a> {
    texture: &'a ResolvedTexture,
    coords: Coords,
    /// `mapping_x/y/z` as `projx/y/z`: 0 is none, 1 to 3 pick X, Y, or Z of the point.
    projection: [usize; 3],
    offset: [f64; 3],
    scale: [f64; 3],
    default_value: f64,
    blend: Blend,
    channels: Vec<(Channel, f64)>,
}

/// The values a particle's birth point gives its settings (`psys_get_texture` with each event).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BirthValues {
    pub(crate) time: f64,
    pub(crate) density: f64,
    pub(crate) life: f64,
    pub(crate) size: f64,
}

/// The texture resources the particle settings' supported slots refer to, by resource ID.
pub fn texture_ids(settings: &Value) -> Vec<String> {
    usable_slots(settings)
        .into_iter()
        .filter_map(|slot| {
            slot.get("texture")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

/// Whether a slot maps emission time, which moves births off the count schedule.
pub(crate) fn maps_time(slots: &[TextureSlot<'_>]) -> bool {
    slots.iter().any(|slot| {
        slot.channels
            .iter()
            .any(|(channel, _)| *channel == Channel::Time)
    })
}

/// Builds the supported texture slots from the particle settings' `texture_slots`.
///
/// Slots without a texture, without a supported mapped channel, or with a coordinate type other
/// than GLOBAL or OBJECT are left out; the importer reports them as losses.
pub fn texture_slots<'a>(
    settings: &Value,
    textures: &'a BTreeMap<String, ResolvedTexture>,
    object_matrices: &BTreeMap<Id, DMat4>,
) -> Result<Vec<TextureSlot<'a>>> {
    let mut slots = Vec::new();
    for slot in usable_slots(settings) {
        let Some(name) = slot.get("texture").and_then(Value::as_str) else {
            continue;
        };
        let texture = textures.get(name).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InternalError,
                "particle texture was not resolved",
                serde_json::json!({"texture": name}),
            )
        })?;
        let coords = match slot
            .get("texture_coords")
            .and_then(Value::as_str)
            .unwrap_or("GLOBAL")
        {
            "GLOBAL" => Coords::Global,
            "OBJECT" => match slot.get("object").and_then(Value::as_str) {
                None => Coords::Global,
                Some(object) => {
                    let id = Id::new(object.to_owned())?;
                    let world = object_matrices.get(&id).copied().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::DependencyMissing,
                            "particle texture object transform is missing",
                            serde_json::json!({"object": object}),
                        )
                    })?;
                    let inverse = world.inverse();
                    if !inverse.is_finite() {
                        return Err(PotError::invalid_argument(
                            "particle texture object transform must be invertible",
                        ));
                    }
                    Coords::Object(inverse)
                }
            },
            _ => continue,
        };
        let mut projection = [0_usize; 3];
        for (axis, key) in ["mapping_x", "mapping_y", "mapping_z"]
            .into_iter()
            .enumerate()
        {
            projection[axis] = match slot.get(key).and_then(Value::as_str).unwrap_or("NONE") {
                "NONE" => 0,
                "X" => 1,
                "Y" => 2,
                "Z" => 3,
                _ => {
                    return Err(PotError::invalid_argument(
                        "particle texture mapping is invalid",
                    ));
                }
            };
        }
        let blend = blend_type(
            slot.get("blend_type")
                .and_then(Value::as_str)
                .unwrap_or("MIX"),
        )?;
        slots.push(TextureSlot {
            texture,
            coords,
            projection,
            offset: vector(slot, "offset", [0.0; 3])?,
            scale: vector(slot, "scale", [1.0; 3])?,
            default_value: number(slot, "default_value", 1.0)?,
            blend,
            channels: supported_channels(slot)?,
        });
    }
    Ok(slots)
}

/// Samples the mapped settings at a birth point, as `psys_get_texture` does for each event.
///
/// `time` is the particle's default emission time fraction (its index over the count). Each
/// channel starts from its neutral value: time from `time`, and density, life, and size from 1.
pub(crate) fn birth_values(
    slots: &[TextureSlot<'_>],
    birth: DVec3,
    time: f64,
    rng: &mut BlenderRng,
) -> Result<BirthValues> {
    let mut values = BirthValues {
        time,
        density: 1.0,
        life: 1.0,
        size: 1.0,
    };
    // Blender's `setvars & PAMAP_TIME`: the first time slot sets the value, later ones blend.
    let mut time_set = false;
    for slot in slots {
        let mut sampled = None;
        for &(channel, factor) in &slot.channels {
            let value = match sampled {
                Some(value) => value,
                None => {
                    let value = slot.sample(birth, rng)?;
                    sampled = Some(value);
                    value
                }
            };
            let current = match channel {
                Channel::Time => values.time,
                Channel::Life => values.life,
                Channel::Density => values.density,
                Channel::Size => values.size,
            };
            let next = if channel == Channel::Time && !time_set {
                time_set = true;
                let flip = factor < 0.0;
                let timefac = factor.abs();
                current * (1.0 - timefac) + timefac * if flip { 1.0 - value } else { value }
            } else {
                texture_value_blend(slot.default_value, current, value, factor, slot.blend)
            };
            match channel {
                Channel::Time => values.time = next,
                Channel::Life => values.life = next,
                Channel::Density => values.density = next,
                Channel::Size => values.size = next,
            }
        }
    }
    Ok(values)
}

impl TextureSlot<'_> {
    /// Blender's `RE_texture_evaluate` intensity at the slot's placed coordinates.
    fn sample(&self, birth: DVec3, rng: &mut BlenderRng) -> Result<f64> {
        let point = match self.coords {
            Coords::Global => birth,
            Coords::Object(world_to_object) => world_to_object.transform_point3(birth),
        };
        let mut texvec = [0.0; 3];
        for axis in 0..3 {
            let base = match self.projection[axis] {
                0 => self.offset[axis],
                projection => point[projection - 1] + self.offset[axis],
            };
            texvec[axis] = self.scale[axis] * base;
        }
        let result = legacy_texture::sample(self.texture.source(), DVec3::from_array(texvec), rng)?;
        // `RE_texture_evaluate` sets `tin` to the luminance of an RGB result.
        Ok(if result.is_rgb {
            0.2126 * result.rgb[0] + 0.7152 * result.rgb[1] + 0.0722 * result.rgb[2]
        } else {
            result.intensity
        })
    }
}

/// The slot objects that the simulation can use: a texture, a supported coordinate type, and at
/// least one supported channel.
fn usable_slots(settings: &Value) -> Vec<&Value> {
    settings
        .get("texture_slots")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|slot| {
            slot.get("texture").and_then(Value::as_str).is_some()
                && matches!(
                    slot.get("texture_coords")
                        .and_then(Value::as_str)
                        .unwrap_or("GLOBAL"),
                    "GLOBAL" | "OBJECT"
                )
                && supported_channels(slot).is_ok_and(|channels| !channels.is_empty())
        })
        .collect()
}

/// The mapped supported channels of a slot with their factors, in Blender's order of use.
fn supported_channels(slot: &Value) -> Result<Vec<(Channel, f64)>> {
    let mut channels = Vec::new();
    for (key, channel, factor) in [
        ("use_map_time", Channel::Time, "time_factor"),
        ("use_map_life", Channel::Life, "life_factor"),
        ("use_map_density", Channel::Density, "density_factor"),
        ("use_map_size", Channel::Size, "size_factor"),
    ] {
        if slot.get(key).and_then(Value::as_bool).unwrap_or(false) {
            channels.push((channel, number(slot, factor, 1.0)?));
        }
    }
    Ok(channels)
}

fn blend_type(name: &str) -> Result<Blend> {
    Ok(match name {
        "MIX" => Blend::Mix,
        "MULTIPLY" => Blend::Multiply,
        "ADD" => Blend::Add,
        "SUBTRACT" => Blend::Subtract,
        "DIVIDE" => Blend::Divide,
        "DARKEN" => Blend::Darken,
        "DIFFERENCE" => Blend::Difference,
        "LIGHTEN" => Blend::Lighten,
        "SCREEN" => Blend::Screen,
        "OVERLAY" => Blend::Overlay,
        "SOFT_LIGHT" => Blend::SoftLight,
        "LINEAR_LIGHT" => Blend::LinearLight,
        "HUE" | "SATURATION" | "VALUE" | "COLOR" => Blend::Zero,
        _ => {
            return Err(PotError::invalid_argument(
                "particle texture blend type is invalid",
            ));
        }
    })
}

fn number(slot: &Value, key: &str, default: f64) -> Result<f64> {
    match slot.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                PotError::invalid_argument(format!("particle texture {key} must be finite"))
            }),
    }
}

fn vector(slot: &Value, key: &str, default: [f64; 3]) -> Result<[f64; 3]> {
    match slot.get(key) {
        None => Ok(default),
        Some(value) => {
            let values = value
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| {
                    PotError::invalid_argument(format!(
                        "particle texture {key} must have three components"
                    ))
                })?;
            let mut result = [0.0; 3];
            for (index, component) in values.iter().enumerate() {
                result[index] = component
                    .as_f64()
                    .filter(|number| number.is_finite())
                    .ok_or_else(|| {
                        PotError::invalid_argument(format!("particle texture {key} must be finite"))
                    })?;
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Blend, texture_value_blend};

    #[test]
    fn blend_modes_match_blender_texture_value_blend_formulas() {
        // Mix with full factor returns the texture-weighted default: fact*tex + (1-fact)*out.
        assert!((texture_value_blend(0.25, 0.75, 1.0, 1.0, Blend::Mix) - 0.25).abs() < 1.0e-12);
        // Add and Subtract use the same factor, negated for Subtract.
        assert!((texture_value_blend(0.5, 0.25, 1.0, 1.0, Blend::Add) - 0.75).abs() < 1.0e-12);
        assert!((texture_value_blend(0.5, 0.25, 1.0, 1.0, Blend::Subtract) + 0.25).abs() < 1.0e-12);
        // Divide by a zero default returns zero, as Blender leaves its result unset.
        assert_eq!(texture_value_blend(0.0, 0.5, 1.0, 1.0, Blend::Divide), 0.0);
        // A negative factor flips which side takes the texture weight.
        assert!((texture_value_blend(1.0, 0.0, 0.5, -1.0, Blend::Mix) - 0.5).abs() < 1.0e-12);
    }
}
