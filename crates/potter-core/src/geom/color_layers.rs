// SPDX-License-Identifier: GPL-3.0-or-later

//! The mesh's color layers: one canonical store of linear RGBA values, shared by authoring, render,
//! glTF/VRM, and the `.blend` bridge.
//!
//! Each layer is a named mesh attribute holding its values once:
//!
//! `mesh.attributes[storage_key(name)] = {"domain": "point" | "corner", "type": "color", "values": …}`
//!
//! - `point`: `values["v<vertex id>"] = [r, g, b, a]`.
//! - `corner`: `values["f<face id>"] = [[r, g, b, a] per face corner]`, in face corner order.
//!
//! Values are linear (Blender's `Color Attribute.color`, not `color_srgb`). Nothing clamps them: HDR
//! floats are stored as they are.
//!
//! The layer list keeps the color layers in Blender's order and names the active (edit target) and
//! render (glTF `COLOR_0`, shading) layers:
//!
//! `mesh.attributes["color_layers"] = {"layers": [name…], "active": name|null, "render": name|null}`
//!
//! Source encoding (`FLOAT_COLOR` or `BYTE_COLOR`) is Blender-only and lives in
//! `blender_metadata.color_data_types`, never next to the values.
//!
//! [`read`] returns the values in storage order (one entry per vertex for `point`, one per corner
//! for `corner`), and [`write`] writes them back keyed by persistent ID, so the store survives
//! element reordering.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use super::{Mesh, attribute_keys};
use crate::color::{linear_to_srgb, srgb_to_linear};
use crate::error::{ErrorCode, PotError, Result};

/// Attribute name of the color layer store.
pub(crate) const ATTRIBUTE: &str = "color_layers";
/// Attribute `type` shared by every color layer.
pub(crate) const COLOR_TYPE: &str = "color";

/// Domain of a color layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorDomain {
    /// One value per vertex.
    Point,
    /// One value per face corner.
    Corner,
}

impl ColorDomain {
    /// Attribute `domain` string.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Corner => "corner",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "point" => Some(Self::Point),
            "corner" => Some(Self::Corner),
            _ => None,
        }
    }
}

/// One color layer with values in storage order: one entry per vertex for [`ColorDomain::Point`],
/// one per corner (mesh face order, then corner order) for [`ColorDomain::Corner`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColorLayer {
    pub(crate) name: String,
    pub(crate) domain: ColorDomain,
    pub(crate) values: Vec<[f64; 4]>,
}

/// The whole store: layers in order plus the active and render layer names.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ColorStore {
    pub(crate) layers: Vec<ColorLayer>,
    pub(crate) active: Option<String>,
    pub(crate) render: Option<String>,
}

impl ColorStore {
    /// Position of the layer called `name`.
    pub(crate) fn index_of(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|layer| layer.name == name)
    }

    /// Position of the render layer.
    pub(crate) fn render_index(&self) -> Option<usize> {
        self.render.as_deref().and_then(|name| self.index_of(name))
    }

    /// Resolves a layer reference. `None` names the render layer. A name that no layer has
    /// resolves to `None` and is returned as the second element, so the caller can report it.
    pub(crate) fn resolve(&self, reference: Option<&str>) -> (Option<usize>, Option<String>) {
        match reference {
            None => (self.render_index(), None),
            Some(name) => match self.index_of(name) {
                Some(index) => (Some(index), None),
                None => (None, Some(name.to_owned())),
            },
        }
    }
}

/// Parses one `[r, g, b, a]` value.
fn color_value(value: &Value) -> Option<[f64; 4]> {
    match value.as_array()?.as_slice() {
        [r, g, b, a] => Some([r.as_f64()?, g.as_f64()?, b.as_f64()?, a.as_f64()?]),
        _ => None,
    }
}

/// Encoded `BYTE_COLOR` channels scaled to `[0, 255]`, before integer quantization.
///
/// RGB is sRGB encoded and alpha stays linear, matching Blender's `BLI_color::encode`.
pub(crate) fn byte_color_encoded_components(linear: [f64; 4]) -> [f32; 4] {
    std::array::from_fn(|index| {
        let channel = linear[index].clamp(0.0, 1.0) as f32;
        let encoded = if index < 3 {
            blender_linear_to_srgb_f32(channel)
        } else {
            channel
        };
        encoded * 255.0
    })
}

// Blender's fastpow approximation affects which byte an encoded channel rounds to.
fn blender_linear_to_srgb_f32(value: f32) -> f32 {
    if value < 0.003_130_8 {
        if value < 0.0 { 0.0 } else { value * 12.92 }
    } else {
        1.055 * blender_fastpow512(value) - 0.055
    }
}

fn blender_fastpow512(arg: f32) -> f32 {
    fn fastpow(exp: i32, e2coeff: i32, arg: f32) -> f32 {
        let mut value = arg * f32::from_bits(e2coeff as u32);
        value = (value.to_bits() as i32) as f32;
        value *= f32::from_bits(exp as u32);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        let value = value.round_ties_even() as i32;
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let value = value as i32;
        f32::from_bits(value as u32)
    }

    #[cfg(target_arch = "aarch64")]
    fn rsqrt(value: f32) -> f32 {
        use std::arch::aarch64::{vdup_n_f32, vget_lane_f32, vmul_f32, vrsqrte_f32, vrsqrts_f32};

        // SAFETY: this fast-power path calls rsqrt only with positive finite scalars.
        unsafe {
            let input = vdup_n_f32(value);
            let mut estimate = vrsqrte_f32(input);
            estimate = vmul_f32(estimate, vrsqrts_f32(vmul_f32(input, estimate), estimate));
            estimate = vmul_f32(estimate, vrsqrts_f32(vmul_f32(input, estimate), estimate));
            vget_lane_f32::<0>(estimate)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn rsqrt(value: f32) -> f32 {
        use std::arch::x86_64::{_mm_add_ps, _mm_cvtss_f32, _mm_mul_ps, _mm_rsqrt_ps, _mm_set1_ps};

        // SAFETY: this fast-power path calls rsqrt only with positive finite scalars.
        unsafe {
            let input = _mm_set1_ps(value);
            let mut estimate = _mm_rsqrt_ps(input);
            estimate = _mm_add_ps(
                _mm_mul_ps(_mm_set1_ps(1.5), estimate),
                _mm_mul_ps(
                    _mm_mul_ps(_mm_mul_ps(input, _mm_set1_ps(-0.5)), estimate),
                    _mm_mul_ps(estimate, estimate),
                ),
            );
            _mm_cvtss_f32(estimate)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn rsqrt(value: f32) -> f32 {
        1.0 / value.sqrt()
    }

    let xf = fastpow(0x3f2aaaab, 0x5eb504f3, arg);
    let xover = arg * xf;
    let xfm1 = rsqrt(xf);
    let x2 = arg * arg;
    let xunder = x2 * xfm1;
    let mut xavg = (1.0 / (3.0 * 0.629_960_524_947_437) * 0.999_852) * (xover + xunder);
    xavg *= rsqrt(xavg);
    xavg *= rsqrt(xavg);
    xavg
}

/// Decodes stored `BYTE_COLOR` channels scaled to `[0, 255]` into canonical linear RGBA.
pub(crate) fn byte_color_decoded_components(bytes: [f32; 4]) -> [f64; 4] {
    const BYTE_TO_UNIT: f32 = 1.0 / 255.0;
    std::array::from_fn(|index| {
        let encoded = bytes[index] * BYTE_TO_UNIT;
        let decoded = if index < 3 {
            if encoded <= 0.04045 {
                encoded * (1.0 / 12.92)
            } else {
                ((encoded + 0.055) * (1.0 / 1.055)).powf(2.4)
            }
        } else {
            encoded
        };
        f64::from(decoded)
    })
}

/// Mix canonical linear colors with Blender's `ColorGeometry4bMixer`: encode and quantize source
/// channels to stored bytes, accumulate the bytes by weight in `f32`, truncate the normalized result
/// to bytes, then decode to canonical linear RGBA.
pub(crate) fn mix_byte_color(values: &[[f64; 4]], sources: &[(usize, f64)]) -> [f64; 4] {
    let mut sums = [0.0_f32; 4];
    let mut total = 0.0_f32;
    let mut found = false;
    for (index, weight) in sources {
        let Some(color) = values.get(*index) else {
            continue;
        };
        found = true;
        let channels =
            byte_color_encoded_components(*color).map(|channel| channel.round().clamp(0.0, 255.0));
        let weight = *weight as f32;
        total += weight;
        for (sum, channel) in sums.iter_mut().zip(channels) {
            *sum += channel * weight;
        }
    }
    if !found || !(total > 0.0) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let weight_inv = 1.0_f32 / total;
    let bytes = sums.map(|sum| (sum * weight_inv).trunc().clamp(0.0, 255.0));
    byte_color_decoded_components(bytes)
}
fn corner_key(face_id: u32) -> String {
    format!("f{face_id}")
}

fn vertex_key(vertex_id: u32) -> String {
    format!("v{vertex_id}")
}

/// Reads the store leniently. Missing or malformed values fall back to opaque black, and
/// [`validate`] reports them at load time.
pub(crate) fn read(mesh: &Mesh) -> ColorStore {
    let Some(object) = mesh.attributes.get(ATTRIBUTE).and_then(Value::as_object) else {
        return ColorStore::default();
    };
    let layers = object
        .get("layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|name| read_layer(mesh, name))
        .collect();
    let name = |key: &str| object.get(key).and_then(Value::as_str).map(str::to_owned);
    ColorStore {
        layers,
        active: name("active"),
        render: name("render"),
    }
}

fn read_layer(mesh: &Mesh, name: &str) -> Option<ColorLayer> {
    let attribute = mesh
        .attributes
        .get(attribute_keys::storage_key(name).as_ref())?
        .as_object()?;
    if attribute.get("type").and_then(Value::as_str) != Some(COLOR_TYPE) {
        return None;
    }
    let domain = ColorDomain::parse(attribute.get("domain")?.as_str()?)?;
    let values = attribute.get("values")?.as_object()?;
    let fallback = [0.0, 0.0, 0.0, 1.0];
    let flat = match domain {
        ColorDomain::Point => mesh
            .vertices
            .iter()
            .map(|vertex| {
                values
                    .get(&vertex_key(vertex.id))
                    .and_then(color_value)
                    .unwrap_or(fallback)
            })
            .collect(),
        ColorDomain::Corner => mesh
            .faces
            .iter()
            .flat_map(|face| {
                let row = values.get(&corner_key(face.id)).and_then(Value::as_array);
                (0..face.vertices.len()).map(move |corner| {
                    row.and_then(|row| row.get(corner))
                        .and_then(color_value)
                        .unwrap_or(fallback)
                })
            })
            .collect(),
    };
    Some(ColorLayer {
        name: name.to_owned(),
        domain,
        values: flat,
    })
}

/// Writes `store` as the mesh's color layer attributes and layer list. Layers that are no longer
/// in the store are removed from the mesh. A stale `active` or `render` name falls back to the
/// first layer, or to none when the store is empty.
///
/// Values are keyed by the mesh's element IDs. A layer's values shorter than the element count
/// are padded with opaque black.
pub(crate) fn write(mesh: &mut Mesh, store: &ColorStore) {
    let previous: Vec<String> = mesh
        .attributes
        .get(ATTRIBUTE)
        .and_then(|value| value.get("layers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    for name in previous {
        if store.index_of(&name).is_none() {
            mesh.attributes
                .remove(attribute_keys::storage_key(&name).as_ref());
        }
    }
    let fallback = [0.0, 0.0, 0.0, 1.0];
    for layer in &store.layers {
        let mut values = Map::new();
        match layer.domain {
            ColorDomain::Point => {
                for (vertex, color) in mesh.vertices.iter().zip(
                    layer
                        .values
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(fallback)),
                ) {
                    values.insert(vertex_key(vertex.id), json!(color));
                }
            }
            ColorDomain::Corner => {
                let mut corner = 0_usize;
                for face in &mesh.faces {
                    let count = face.vertices.len();
                    let row: Vec<Value> = (corner..corner + count)
                        .map(|index| json!(layer.values.get(index).copied().unwrap_or(fallback)))
                        .collect();
                    values.insert(corner_key(face.id), Value::Array(row));
                    corner += count;
                }
            }
        }
        mesh.attributes.insert(
            attribute_keys::storage_key(&layer.name).into_owned(),
            json!({"domain": layer.domain.name(), "type": COLOR_TYPE, "values": values}),
        );
    }
    if store.layers.is_empty() {
        mesh.attributes.remove(ATTRIBUTE);
        return;
    }
    let named = |name: &Option<String>| {
        name.as_ref()
            .filter(|name| store.index_of(name).is_some())
            .or(store.layers.first().map(|layer| &layer.name))
            .cloned()
    };
    let layers: Vec<Value> = store.layers.iter().map(|layer| json!(layer.name)).collect();
    mesh.attributes.insert(
        ATTRIBUTE.to_owned(),
        json!({
            "layers": layers,
            "active": named(&store.active),
            "render": named(&store.render),
        }),
    );
}

/// Checks the store and every layer it names. Run at load time for every data-block mesh.
pub(crate) fn validate(mesh: &Mesh) -> Result<()> {
    let Some(value) = mesh.attributes.get(ATTRIBUTE) else {
        return Ok(());
    };
    let invalid = |message: String| PotError::new(ErrorCode::SceneInvalid, message);
    let object = value
        .as_object()
        .ok_or_else(|| invalid("color_layers must be an object".to_owned()))?;
    let names = object
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("color_layers.layers must be an array".to_owned()))?;
    let mut seen = HashSet::new();
    let mut layer_names = Vec::with_capacity(names.len());
    for name in names {
        let name = name
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("color layer name must be a non-empty string".to_owned()))?;
        if !seen.insert(name) {
            return Err(invalid(format!("color layer name `{name}` is not unique")));
        }
        layer_names.push(name);
        let attribute = mesh
            .attributes
            .get(attribute_keys::storage_key(name).as_ref())
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("color layer `{name}` has no attribute")))?;
        let domain = attribute
            .get("domain")
            .and_then(Value::as_str)
            .and_then(ColorDomain::parse)
            .ok_or_else(|| {
                invalid(format!(
                    "color layer `{name}` domain must be point or corner"
                ))
            })?;
        if attribute.get("type").and_then(Value::as_str) != Some(COLOR_TYPE) {
            return Err(invalid(format!("color layer `{name}` type must be color")));
        }
        let values = attribute
            .get("values")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("color layer `{name}` values must be an object")))?;
        let finite = |value: &Value| {
            color_value(value).is_some_and(|color| color.iter().all(|channel| channel.is_finite()))
        };
        match domain {
            ColorDomain::Point => {
                for vertex in &mesh.vertices {
                    let value = values
                        .get(&vertex_key(vertex.id))
                        .filter(|value| finite(value));
                    if value.is_none() {
                        return Err(invalid(format!(
                            "color layer `{name}` has no finite value for vertex {}",
                            vertex.id
                        )));
                    }
                }
            }
            ColorDomain::Corner => {
                let counts: HashMap<u32, usize> = mesh
                    .faces
                    .iter()
                    .map(|face| (face.id, face.vertices.len()))
                    .collect();
                for face in &mesh.faces {
                    let row = values
                        .get(&corner_key(face.id))
                        .and_then(Value::as_array)
                        .filter(|row| row.len() == counts[&face.id] && row.iter().all(finite));
                    if row.is_none() {
                        return Err(invalid(format!(
                            "color layer `{name}` has no finite corner values for face {}",
                            face.id
                        )));
                    }
                }
            }
        }
    }
    for key in ["active", "render"] {
        let value = object.get(key).unwrap_or(&Value::Null);
        let valid = match value {
            Value::Null => layer_names.is_empty(),
            Value::String(name) => layer_names.contains(&name.as_str()),
            _ => false,
        };
        if !valid {
            return Err(invalid(format!(
                "color_layers.{key} must name a color layer"
            )));
        }
    }
    Ok(())
}
