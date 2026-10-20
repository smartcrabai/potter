use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use glam::{DQuat, DVec3, EulerRot};
use serde_json::{Value as JsonValue, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    image::{ImageData, ImageExtension, ImageInterpolation, sample_with_extension_alpha},
    model::{Id, Material, Registry, TextureRef},
};

/// Blender's Principled "Specular IOR Level" default: `SOCKET_IN_FLOAT(specular_ior_level,
/// "Specular IOR Level", 0.5f)` in `intern/cycles/scene/shader_nodes.cpp`, `PrincipledBsdfNode`.
pub(crate) const DEFAULT_SPECULAR_IOR_LEVEL: f64 = 0.5;

/// Dielectric reflectance at normal incidence for Blender's Principled specular layer: the Fresnel
/// `F0` of the material IOR, scaled by `2 × IOR level` (Cycles `svm_node_closure_bsdf`, Principled
/// branch, `intern/cycles/kernel/svm/closure.h`). The default level 0.5 leaves the IOR's own
/// reflectance, so IOR 1.5 gives F0 = ((1.5 − 1)/(1.5 + 1))² = 0.04. The Blender manual defines
/// the level as "0.5 means no adjustment … 1 doubles them at normal incidence"
/// (`docs.blender.org/manual/en/5.2/render/shader_nodes/shader/principled.html`).
/// Every Principled evaluation (path tracer, shaded hits, the world term) uses this function.
#[must_use]
pub(crate) fn dielectric_f0(ior: f64, level: f64) -> f64 {
    let fresnel = ((ior - 1.0) / (ior + 1.0)).powi(2);
    (fresnel * 2.0 * level.max(0.0)).min(1.0)
}

/// The Principled inputs that change Cycles' result but that Potter's lobe model does not evaluate,
/// as `(feature_id, Blender input)` pairs in a fixed array, so checking a hit allocates nothing.
pub(crate) fn unevaluated_principled_inputs(
    params: &BsdfParams,
) -> [Option<(&'static str, &'static str)>; 6] {
    [
        (params.subsurface_weight > 0.0)
            .then_some(("shader.principled.subsurface", "Subsurface Weight")),
        (params.anisotropic_ior > 0.0).then_some(("shader.principled.anisotropic", "Anisotropic")),
        (params.thin_film_thickness > 0.0)
            .then_some(("shader.principled.thin_film", "Thin Film Thickness")),
        (params.transmission > 0.0 && params.roughness > 0.0)
            .then_some(("shader.principled.transmission_ggx", "Transmission Weight")),
        (params.diffuse_roughness > 0.0)
            .then_some(("shader.principled.diffuse_roughness", "Diffuse Roughness")),
        params
            .thin_wall
            .then_some(("shader.principled.thin_wall", "Thin Wall")),
    ]
}

/// Records `UNSUPPORTED_FEATURE` for each Principled input in `params` that Potter's lobe model
/// does not evaluate. `node` names the Principled node of a graph material; a native material
/// passes `None`. Deduped per (material, input) in `warnings`, the sink the render output reports.
pub(crate) fn record_unevaluated_inputs(
    warnings: Option<&RefCell<Vec<JsonValue>>>,
    material: &str,
    node: Option<&Id>,
    params: &BsdfParams,
) {
    let inputs = unevaluated_principled_inputs(params);
    if inputs.iter().all(Option::is_none) {
        return;
    }
    let Some(warnings) = warnings else {
        return;
    };
    for (feature_id, input) in inputs.into_iter().flatten() {
        let mut recorded = warnings.borrow_mut();
        let duplicate = recorded.iter().any(|warning| {
            warning["material"].as_str() == Some(material)
                && warning["input"].as_str() == Some(input)
        });
        if duplicate {
            continue;
        }
        recorded.push(match node {
            Some(node) => json!({
                "code": "UNSUPPORTED_FEATURE",
                "feature_id": feature_id,
                "material": material,
                "node_id": node.to_string(),
                "input": input,
                "message": format!(
                    "Principled input '{input}' of material '{material}' (node '{node}') is not evaluated by Potter's lobe model; its Cycles contribution is omitted or approximated"
                ),
            }),
            None => json!({
                "code": "UNSUPPORTED_FEATURE",
                "feature_id": feature_id,
                "material": material,
                "input": input,
                "message": format!(
                    "Principled input '{input}' of material '{material}' is not evaluated by Potter's lobe model; its Cycles contribution is omitted or approximated"
                ),
            }),
        });
    }
}

/// Per-channel volume closure weights per unit density (Cycles' closure weight times density).
/// The scatter weight feeds σs, the absorption weight σa; extinction is σt = σs + σa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VolumeWeights {
    pub(crate) scatter: [f64; 3],
    pub(crate) absorption: [f64; 3],
}

/// Principled Volume weights (`intern/cycles/kernel/svm/closure.h`, `svm_node_principled_volume`):
/// scatter `color`, absorption `(1 - color) · (1 - sqrt(absorption_color))`, both clamped at zero.
pub(crate) fn principled_volume_weights(
    color: [f64; 3],
    absorption_color: [f64; 3],
) -> VolumeWeights {
    VolumeWeights {
        scatter: color.map(|channel| channel.max(0.0)),
        absorption: std::array::from_fn(|index| {
            (1.0 - color[index]).max(0.0) * (1.0 - absorption_color[index].max(0.0).sqrt()).max(0.0)
        }),
    }
}

/// The Cycles volume closure a volume node compiles to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VolumeClosure {
    Principled,
    Scatter,
    Absorption,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsdfParams {
    pub base_color: [f64; 4],
    pub metallic: f64,
    pub roughness: f64,
    pub emission_color: [f64; 3],
    pub emission_strength: f64,
    pub transmission: f64,
    pub ior: f64,
    pub alpha: f64,
    /// Tint of the transparent share, Cycles' Transparent BSDF colour. The share is `1 - alpha`, so a camera
    /// pass-through multiplies by this tint and a shadow ray's transmittance is `(1 - alpha) * tint`
    /// (`render/path.rs`). Non-transparent closures keep white.
    pub transparent_tint: [f64; 3],
    /// Weight of the reflective closures (diffuse, specular, transmission, environment). Emission
    /// contributes no reflection, so an Emission shader sets this to 0. Mix lerps it and Add sums
    /// it, clamped to 1.
    pub reflectance: f64,
    pub subsurface_weight: f64,
    pub subsurface_radius: [f64; 3],
    pub subsurface_scale: f64,
    pub coat_weight: f64,
    pub coat_roughness: f64,
    pub coat_ior: f64,
    /// Coat Tint: the colour the coat absorbs on the way in and out (Cycles' coat tint).
    pub coat_tint: [f64; 3],
    /// Coat Normal, which the coat lobe and its tint use. `None` (unlinked) follows the final
    /// shading normal, so a bump or a flip applies to the coat too, as Cycles' `sd->N` does.
    pub coat_normal: Option<DVec3>,
    pub sheen_weight: f64,
    pub sheen_roughness: f64,
    pub sheen_tint: [f64; 3],
    pub specular_ior_level: f64,
    pub specular_tint: [f64; 3],
    pub anisotropic_ior: f64,
    pub anisotropic_rotation: f64,
    pub thin_film_thickness: f64,
    pub thin_film_ior: f64,
    /// Oren-Nayar roughness of the diffuse lobe. Potter's diffuse lobe is Lambert, so a value above
    /// zero is reported as unevaluated (Cycles `bsdf_oren_nayar_setup`).
    pub diffuse_roughness: f64,
    /// Thin Wall (Cycles' `thin_wall`): transmission through a thin sheet. Potter does not evaluate
    /// it; it is reported as unevaluated.
    pub thin_wall: bool,
    /// Cycles' scattering coefficient σs per channel, absolute: each closure's scatter weight times
    /// its Density. Add Shader sums it and Mix Shader mixes it per closure, as Cycles weights closures.
    pub volume_scatter: [f64; 3],
    /// Cycles' absorption coefficient σa per channel, with the rule of `volume_scatter`. Extinction
    /// is σt = σs + σa, and the albedo is σs / σt.
    pub volume_absorption: [f64; 3],
    pub volume_anisotropy: f64,
    /// Emission radiance per unit density per channel: colour × strength × Density.
    pub volume_emission: [f64; 3],
    pub displacement_height: f64,
    pub normal: DVec3,
}

/// One layered closure's parameters, as Mix and Add combine them.
#[derive(Clone, Copy)]
struct LayerSide {
    weight: f64,
    tint: [f64; 3],
    roughness: f64,
    normal: DVec3,
}

/// The one rule for layered-lobe parameters under Mix and Add. Each side's layer enters with its
/// share (`1 - f` and `f` for Mix, 1 and 1 for Add). The combined weight is the sum of the scaled
/// weights, so `weight · tint` adds as the closures do. The tint, roughness and normal are averages
/// weighted by the same scaled weights. With no weight on either side, the first side's values stand.
fn combine_layers(
    first: LayerSide,
    first_share: f64,
    second: LayerSide,
    second_share: f64,
) -> LayerSide {
    let first_term = first_share * first.weight.max(0.0);
    let second_term = second_share * second.weight.max(0.0);
    let weight = first_term + second_term;
    if weight <= 0.0 {
        return LayerSide { weight, ..first };
    }
    let average = |a: f64, b: f64| (first_term * a + second_term * b) / weight;
    LayerSide {
        weight,
        tint: [
            average(first.tint[0], second.tint[0]),
            average(first.tint[1], second.tint[1]),
            average(first.tint[2], second.tint[2]),
        ],
        roughness: average(first.roughness, second.roughness),
        normal: normalize_or(
            first.normal * first_term + second.normal * second_term,
            first.normal,
        ),
    }
}

impl BsdfParams {
    pub(crate) fn from_material(material: &Material, normal: DVec3) -> Self {
        let volume = principled_volume_weights(material.volume_color, [0.0; 3]);
        let density = material.volume_density.max(0.0);
        Self {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            emission_color: material.emission_color,
            emission_strength: material.emission_strength,
            transmission: material.transmission,
            ior: material.ior,
            alpha: material.base_color[3],
            transparent_tint: [1.0; 3],
            reflectance: 1.0,
            subsurface_weight: 0.0,
            subsurface_radius: [1.0; 3],
            subsurface_scale: 0.05,
            coat_weight: 0.0,
            coat_roughness: 0.03,
            coat_ior: 1.5,
            sheen_weight: 0.0,
            sheen_roughness: 0.5,
            sheen_tint: [1.0; 3],
            specular_ior_level: DEFAULT_SPECULAR_IOR_LEVEL,
            specular_tint: [1.0; 3],
            anisotropic_ior: 0.0,
            anisotropic_rotation: 0.0,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.33,
            diffuse_roughness: 0.0,
            thin_wall: false,
            volume_scatter: volume.scatter.map(|weight| weight * density),
            volume_absorption: volume.absorption.map(|weight| weight * density),
            volume_anisotropy: material.volume_anisotropy.clamp(-0.999, 0.999),
            volume_emission: material
                .emission_color
                .map(|channel| channel * material.emission_strength.max(0.0) * density),
            displacement_height: 0.0,
            normal: normalize_or(normal, DVec3::Z),
            coat_tint: [1.0; 3],
            coat_normal: None,
        }
    }

    /// The coat's normal: the linked Coat Normal, or the final shading normal when unlinked.
    pub(crate) fn coat_normal_or_shading(&self) -> DVec3 {
        self.coat_normal.unwrap_or(self.normal)
    }

    /// The coat as one layered closure, for `combine_layers`.
    fn coat_layer(&self) -> LayerSide {
        LayerSide {
            weight: self.coat_weight,
            tint: self.coat_tint,
            roughness: self.coat_roughness,
            normal: self.coat_normal_or_shading(),
        }
    }

    /// The sheen as one layered closure, for `combine_layers`. Sheen has no normal of its own: the
    /// combined normal is the coat's, so this normal is never read.
    fn sheen_layer(&self) -> LayerSide {
        LayerSide {
            weight: self.sheen_weight,
            tint: self.sheen_tint,
            roughness: self.sheen_roughness,
            normal: self.normal,
        }
    }

    /// Whether the volume coefficients put a medium on the surface's path.
    pub(crate) fn has_medium(&self) -> bool {
        self.volume_scatter
            .iter()
            .chain(&self.volume_absorption)
            .any(|weight| *weight > 0.0)
    }

    fn mix(first: Self, second: Self, factor: f64) -> Self {
        let factor = bounded_factor(factor);
        // Alpha is the share of rays that reach the surface's own shading (Cycles' Transparent closure carries only
        // that share). The shading mixes the arms by their share of the hits, so a Transparent arm adds no colour,
        // reflectance, or transmission.
        let alpha = (1.0 - factor) * first.alpha + factor * second.alpha;
        let shade = if alpha > 0.0 {
            factor * second.alpha / alpha
        } else {
            factor
        };
        // Cycles weights each closure by its factor and its transparent weight is `(1 - alpha) * tint`, so the mixed
        // transparency is the factor-weighted sum of the arms' transparencies (`tint_for_transparency`).
        let transparency = std::array::from_fn(|channel| {
            (1.0 - factor) * (1.0 - first.alpha) * first.transparent_tint[channel]
                + factor * (1.0 - second.alpha) * second.transparent_tint[channel]
        });
        let transparent_tint = tint_for_transparency(alpha, transparency);
        let mix_scalar = |a: f64, b: f64| a + (b - a) * shade;
        let mix3 = |a: [f64; 3], b: [f64; 3]| {
            [
                mix_scalar(a[0], b[0]),
                mix_scalar(a[1], b[1]),
                mix_scalar(a[2], b[2]),
            ]
        };
        let base_color = [
            mix_scalar(first.base_color[0], second.base_color[0]),
            mix_scalar(first.base_color[1], second.base_color[1]),
            mix_scalar(first.base_color[2], second.base_color[2]),
            mix_scalar(first.base_color[3], second.base_color[3]),
        ];
        // The coat and sheen are shading like every other field, so they take the same share of the hits
        // (`shade`), not the raw factor: a Transparent arm then adds no coat or sheen, as it adds no colour.
        let coat = combine_layers(first.coat_layer(), 1.0 - shade, second.coat_layer(), shade);
        let sheen = combine_layers(
            first.sheen_layer(),
            1.0 - shade,
            second.sheen_layer(),
            shade,
        );
        Self {
            base_color,
            metallic: mix_scalar(first.metallic, second.metallic),
            roughness: mix_scalar(first.roughness, second.roughness),
            emission_color: mix3(first.emission_color, second.emission_color),
            emission_strength: mix_scalar(first.emission_strength, second.emission_strength),
            transmission: mix_scalar(first.transmission, second.transmission),
            ior: mix_scalar(first.ior, second.ior),
            alpha,
            transparent_tint,
            reflectance: mix_scalar(first.reflectance, second.reflectance),
            subsurface_weight: mix_scalar(first.subsurface_weight, second.subsurface_weight),
            subsurface_radius: mix3(first.subsurface_radius, second.subsurface_radius),
            subsurface_scale: mix_scalar(first.subsurface_scale, second.subsurface_scale),
            coat_weight: coat.weight,
            coat_roughness: coat.roughness,
            coat_ior: mix_scalar(first.coat_ior, second.coat_ior),
            sheen_weight: sheen.weight,
            sheen_roughness: sheen.roughness,
            sheen_tint: sheen.tint,
            specular_ior_level: mix_scalar(first.specular_ior_level, second.specular_ior_level),
            specular_tint: mix3(first.specular_tint, second.specular_tint),
            anisotropic_ior: mix_scalar(first.anisotropic_ior, second.anisotropic_ior),
            anisotropic_rotation: mix_scalar(
                first.anisotropic_rotation,
                second.anisotropic_rotation,
            ),
            thin_film_thickness: mix_scalar(first.thin_film_thickness, second.thin_film_thickness),
            thin_film_ior: mix_scalar(first.thin_film_ior, second.thin_film_ior),
            diffuse_roughness: mix_scalar(first.diffuse_roughness, second.diffuse_roughness),
            thin_wall: first.thin_wall || second.thin_wall,
            volume_scatter: mix3(first.volume_scatter, second.volume_scatter),
            volume_absorption: mix3(first.volume_absorption, second.volume_absorption),
            volume_anisotropy: mix_scalar(first.volume_anisotropy, second.volume_anisotropy),
            volume_emission: mix3(first.volume_emission, second.volume_emission),
            displacement_height: mix_scalar(first.displacement_height, second.displacement_height),
            normal: normalize_or(first.normal.lerp(second.normal, shade), first.normal),
            coat_tint: coat.tint,
            coat_normal: (first.coat_normal.is_some() || second.coat_normal.is_some())
                .then_some(coat.normal),
        }
    }

    fn add(first: Self, second: Self) -> Self {
        let add3 = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let coat = combine_layers(first.coat_layer(), 1.0, second.coat_layer(), 1.0);
        let sheen = combine_layers(first.sheen_layer(), 1.0, second.sheen_layer(), 1.0);
        Self {
            base_color: [
                first.base_color[0] + second.base_color[0],
                first.base_color[1] + second.base_color[1],
                first.base_color[2] + second.base_color[2],
                first.base_color[3].max(second.base_color[3]),
            ],
            metallic: first.metallic.max(second.metallic),
            roughness: first.roughness.min(second.roughness),
            emission_color: add3(first.emission_color, second.emission_color),
            emission_strength: first.emission_strength.max(second.emission_strength),
            transmission: first.transmission.max(second.transmission),
            ior: first.ior,
            // Add keeps alpha as the larger opacity (its shading is Potter's one-scalar blend), and sums the arms'
            // transparent weights as Cycles sums closures. A fully opaque arm leaves no pass chance, so a transparent
            // share beside it is dropped: the deviation the CHANGELOG records.
            alpha: first.alpha.max(second.alpha),
            transparent_tint: tint_for_transparency(
                first.alpha.max(second.alpha),
                std::array::from_fn(|channel| {
                    (1.0 - first.alpha) * first.transparent_tint[channel]
                        + (1.0 - second.alpha) * second.transparent_tint[channel]
                }),
            ),
            reflectance: (first.reflectance + second.reflectance).min(1.0),
            subsurface_weight: first.subsurface_weight.max(second.subsurface_weight),
            subsurface_radius: add3(first.subsurface_radius, second.subsurface_radius),
            subsurface_scale: first.subsurface_scale.max(second.subsurface_scale),
            coat_weight: coat.weight,
            coat_roughness: coat.roughness,
            coat_ior: first.coat_ior,
            sheen_weight: sheen.weight,
            sheen_roughness: sheen.roughness,
            sheen_tint: sheen.tint,
            specular_ior_level: first.specular_ior_level.max(second.specular_ior_level),
            specular_tint: add3(first.specular_tint, second.specular_tint),
            anisotropic_ior: first.anisotropic_ior.max(second.anisotropic_ior),
            anisotropic_rotation: first.anisotropic_rotation,
            thin_film_thickness: first.thin_film_thickness.max(second.thin_film_thickness),
            thin_film_ior: first.thin_film_ior,
            diffuse_roughness: first.diffuse_roughness.max(second.diffuse_roughness),
            thin_wall: first.thin_wall || second.thin_wall,
            volume_scatter: add3(first.volume_scatter, second.volume_scatter),
            volume_absorption: add3(first.volume_absorption, second.volume_absorption),
            volume_anisotropy: first.volume_anisotropy,
            volume_emission: add3(first.volume_emission, second.volume_emission),
            displacement_height: first.displacement_height + second.displacement_height,
            normal: normalize_or(first.normal + second.normal, first.normal),
            coat_tint: coat.tint,
            coat_normal: (first.coat_normal.is_some() || second.coat_normal.is_some())
                .then_some(coat.normal),
        }
    }
}

/// The transparent tint that gives a pass chance `1 - alpha` the expected throughput `transparency` (Cycles' summed
/// transparent weight): `(1 - alpha) * tint = transparency`. A surface with no pass chance (`alpha` 1) cannot pass,
/// so the tint is white and the transparency is dropped.
fn tint_for_transparency(alpha: f64, transparency: [f64; 3]) -> [f64; 3] {
    let chance = 1.0 - alpha;
    if chance <= 0.0 {
        return [1.0; 3];
    }
    transparency.map(|weight| weight / chance)
}

/// The UV layers of the triangle a hit lies on, borrowed from the render geometry.
///
/// Layer `index` is `names[index]`, with corner values `corners[index]`. A layer's UV at the hit
/// blends those corners by `barycentric`, so every layer shares the hit's position on the triangle.
/// Moving the hit therefore moves each layer by its own UV derivative: the per-layer change is the
/// affine map between the layer's corner values and the triangle.
#[derive(Clone, Copy, Debug)]
pub struct HitUvLayers<'a> {
    /// Layer names in mesh order. A name resolves to its index by a search of this slice.
    pub names: &'a [String],
    /// Corner `[u, v]` values of the triangle for each layer, in `names` order.
    pub corners: &'a [[[f64; 2]; 3]],
    /// Corner positions of the triangle.
    pub positions: [DVec3; 3],
    /// Barycentric weights of the hit over `positions`.
    pub barycentric: [f64; 3],
    /// Index of the render layer; `None` when the object has no UV layer.
    pub render: Option<usize>,
    /// Source-member layers actually available; `None` preserves ordinary-mesh name resolution.
    pub available_mask: Option<u8>,
}

impl HitUvLayers<'_> {
    /// The UV of layer `index` at the hit; `None` when the layer has no corner values.
    #[must_use]
    pub fn uv(&self, index: usize) -> Option<[f64; 2]> {
        let corners = self.corners.get(index)?;
        let weights = self.barycentric;
        Some([
            corners[0][0] * weights[0] + corners[1][0] * weights[1] + corners[2][0] * weights[2],
            corners[0][1] * weights[0] + corners[1][1] * weights[1] + corners[2][1] * weights[2],
        ])
    }

    /// `dP/du` and `dP/dv` of layer `index` on the hit triangle, or `None` when its corner UVs are
    /// degenerate.
    fn derivatives(&self, index: usize) -> Option<(DVec3, DVec3)> {
        let corners = self.corners.get(index)?;
        let [origin, second, third] = self.positions;
        let (edge1, edge2) = (second - origin, third - origin);
        let uv1 = [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]];
        let uv2 = [corners[2][0] - corners[0][0], corners[2][1] - corners[0][1]];
        let determinant = uv1[0] * uv2[1] - uv1[1] * uv2[0];
        if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
            return None;
        }
        let du = (edge1 * uv2[1] - edge2 * uv1[1]) / determinant;
        let dv = (edge2 * uv1[0] - edge1 * uv2[0]) / determinant;
        (du.is_finite() && dv.is_finite()).then_some((du, dv))
    }

    /// The barycentric weights of the point `offset` away from the hit, moved within the triangle's
    /// plane. A degenerate triangle keeps the hit's weights, so its UVs do not move.
    fn shifted_barycentric(&self, offset: DVec3) -> [f64; 3] {
        let [origin, second, third] = self.positions;
        let (edge1, edge2) = (second - origin, third - origin);
        let (a11, a12, a22) = (edge1.dot(edge1), edge1.dot(edge2), edge2.dot(edge2));
        let determinant = a11 * a22 - a12 * a12;
        if !determinant.is_finite() || determinant <= f64::EPSILON * a11 * a22 {
            return self.barycentric;
        }
        let (r1, r2) = (offset.dot(edge1), offset.dot(edge2));
        let s1 = (a22 * r1 - a12 * r2) / determinant;
        let s2 = (a11 * r2 - a12 * r1) / determinant;
        let [b0, b1, b2] = self.barycentric;
        [b0 - s1 - s2, b1 + s1, b2 + s2]
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.names
            .iter()
            .position(|candidate| candidate.as_str() == name)
    }
}

/// Per-hit coordinates and graph context used by procedural shader nodes.
#[derive(Clone, Copy, Debug)]
pub struct HitContext<'a> {
    pub node_groups: &'a Registry<NodeGroup>,
    /// Resolved linear image buffers keyed by their persistent image ID.
    pub image_data: Option<&'a BTreeMap<String, ImageData>>,
    /// The render-layer UV at the hit. Without `uv_layers` it is the only UV.
    pub uv: [f64; 2],
    /// The UV layers of the hit triangle; `None` when only `uv` is known.
    pub uv_layers: Option<HitUvLayers<'a>>,
    /// Sink for `uv.layer_missing` warnings; see [`HitContext::with_warnings`].
    pub warnings: Option<&'a RefCell<Vec<JsonValue>>>,
    pub generated: DVec3,
    pub position: DVec3,
    pub object: DVec3,
    pub normal: DVec3,
    pub view_direction: DVec3,
    /// The render color layer at this hit: the value a Vertex Color node with an empty layer name
    /// reads.
    pub color_render: Option<[f64; 4]>,
    /// Color layers at this hit, by layer name in layer order; see
    /// [`HitContext::with_color_layers`].
    pub color_layers: Option<&'a [(&'a str, [f64; 4])]>,
}

impl<'a> HitContext<'a> {
    #[must_use]
    pub const fn new(node_groups: &'a Registry<NodeGroup>) -> Self {
        Self {
            image_data: None,
            node_groups,
            uv: [0.0; 2],
            uv_layers: None,
            warnings: None,
            generated: DVec3::ZERO,
            position: DVec3::ZERO,
            object: DVec3::ZERO,
            normal: DVec3::Z,
            view_direction: DVec3::Z,
            color_render: None,
            color_layers: None,
        }
    }

    /// Add pre-resolved pixels for Image Texture nodes and material texture references.
    #[must_use]
    pub const fn with_images(mut self, image_data: &'a BTreeMap<String, ImageData>) -> Self {
        self.image_data = Some(image_data);
        self
    }

    /// Attach the UV layers of the hit triangle. Named layers resolve against them, and normal maps
    /// take their frames from them.
    #[must_use]
    pub const fn with_uv_layers(mut self, layers: HitUvLayers<'a>) -> Self {
        self.uv_layers = Some(layers);
        self
    }

    /// Provide the per-hit color of each color layer by name, plus the render layer's color. Missing
    /// layers use the node-specific Blender defaults: transparent black for Vertex Color and opaque
    /// black for Attribute.
    #[must_use]
    pub const fn with_color_layers(
        mut self,
        color_layers: &'a [(&'a str, [f64; 4])],
        color_render: Option<[f64; 4]>,
    ) -> Self {
        self.color_layers = Some(color_layers);
        self.color_render = color_render;
        self
    }
    /// Collect `uv.layer_missing` warnings raised while evaluating this hit. Without a sink the
    /// render UV is still used, but the warning is not recorded.
    #[must_use]
    pub const fn with_warnings(mut self, warnings: &'a RefCell<Vec<JsonValue>>) -> Self {
        self.warnings = Some(warnings);
        self
    }

    /// This hit moved by one finite-difference step. With UV layers every layer is re-blended at
    /// the moved position; without them only `uv` moves, by the step's UV change.
    fn shifted(&self, step: SurfaceStep) -> Self {
        let mut shifted = *self;
        shifted.generated += step.world;
        shifted.position += step.world;
        shifted.object += step.world;
        if let Some(layers) = self.uv_layers {
            let moved = HitUvLayers {
                barycentric: layers.shifted_barycentric(step.world),
                ..layers
            };
            shifted.uv_layers = Some(moved);
            if let Some(uv) = moved.render.and_then(|index| moved.uv(index)) {
                shifted.uv = uv;
            }
        } else {
            shifted.uv[0] += step.uv[0];
            shifted.uv[1] += step.uv[1];
        }
        shifted
    }
}

/// Writes `(layer name, color)` pairs into `table` in layer order, ready for
/// [`HitContext::with_color_layers`]. Nothing is allocated.
///
/// # Errors
///
/// Returns `LIMIT_EXCEEDED` when there are more layers than `table` holds.
pub fn fill_color_layers<'a, 'n>(
    table: &'a mut [(&'n str, [f64; 4])],
    layers: impl IntoIterator<Item = (&'n str, [f64; 4])>,
) -> Result<&'a [(&'n str, [f64; 4])]> {
    let mut count = 0;
    for layer in layers {
        let slot = table.get_mut(count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "mesh has more layers than one hit can carry",
            )
        })?;
        *slot = layer;
        count += 1;
    }
    Ok(&table[..count])
}

/// Evaluate a material's shader graph at a surface hit. Materials without a node tree use their PBR fields.
///
/// # Errors
///
/// Returns `TARGET_NOT_FOUND` for a missing shader graph or image, `INVALID_OPERATION` for an invalid graph,
/// `EVALUATION_FAILED` for a cycle or missing material output, and `UNSUPPORTED_FEATURE` for unsupported
/// node behavior or unresolved image pixels in the hit context.
pub fn evaluate_surface(material: &Material, hit_ctx: &HitContext<'_>) -> Result<BsdfParams> {
    let prepared = PreparedMaterial::new(material, hit_ctx.node_groups);
    let mut workspace = EvaluatorWorkspace::default();
    evaluate_surface_with_workspace(material, &prepared, hit_ctx, &mut workspace)
}

/// Evaluates a material surface using caller-owned reusable graph scratch.
pub fn evaluate_surface_with_workspace(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
) -> Result<BsdfParams> {
    workspace.active.clear();
    let result = evaluate_surface_inner(material, prepared, hit_ctx, workspace);
    workspace.active.clear();
    result
}

fn evaluate_surface_inner(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
) -> Result<BsdfParams> {
    let Some(graph_id) = &material.node_tree else {
        let mut params = BsdfParams::from_material(material, hit_ctx.normal);
        record_unevaluated_inputs(hit_ctx.warnings, &material.name, None, &params);
        apply_material_textures(material, hit_ctx, &mut params)?;
        return Ok(params);
    };
    let group = hit_ctx.node_groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("shader graph `{graph_id}` was not found"),
            json!({"graph_id":graph_id}),
        )
    })?;
    if group.kind != GraphKind::Shader {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("material graph `{graph_id}` is not a shader graph"),
        ));
    }
    workspace.active.reserve(group.links.len());
    let mut fallback_output = None;
    let mut active_output = None;
    for (id, node) in &group.nodes {
        if matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
            if node
                .properties
                .get("is_active_output")
                .and_then(JsonValue::as_bool)
                == Some(true)
            {
                active_output = Some(id);
                break;
            }
            fallback_output.get_or_insert(id);
        }
    }
    let output = active_output.or(fallback_output).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "shader graph has no OutputMaterial node",
        )
    })?;
    let has_volume = group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Volume");
    let has_displacement = group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Displacement");
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx: *hit_ctx,
        workspace,
        alpha_linked_images: &prepared.alpha_linked_images,
    };
    let Value::Shader(mut params) = evaluator.input(output, "Surface", ValueKind::Shader)? else {
        return Err(eval_error("material surface output is not a shader"));
    };
    // The volume coefficients come from the Volume output alone. The Surface's nodes carry the
    // material's base coefficients, which Add and Mix Shader must not sum.
    let volume = if has_volume {
        let Value::Shader(volume) = evaluator.input(output, "Volume", ValueKind::Shader)? else {
            return Err(eval_error("material volume output is not a volume shader"));
        };
        volume
    } else {
        BsdfParams::from_material(material, hit_ctx.normal)
    };
    params.volume_scatter = volume.volume_scatter;
    params.volume_absorption = volume.volume_absorption;
    params.volume_anisotropy = volume.volume_anisotropy;
    params.volume_emission = volume.volume_emission;
    if has_displacement {
        params.displacement_height =
            match evaluator.input(output, "Displacement", ValueKind::Vector)? {
                Value::Vector(displacement) => displacement.dot(params.normal),
                Value::Scalar(displacement) => displacement,
                Value::Color(displacement) => {
                    DVec3::new(displacement[0], displacement[1], displacement[2]).dot(params.normal)
                }
                Value::Shader(_) => return Err(eval_error("material displacement is a shader")),
            };
    }
    apply_material_textures(material, hit_ctx, &mut params)?;
    Ok(params)
}

/// Evaluates the connected `OutputMaterial` displacement input without evaluating surface textures.
pub fn evaluate_displacement(material: &Material, hit_ctx: &HitContext<'_>) -> Result<f64> {
    let prepared = PreparedMaterial::new(material, hit_ctx.node_groups);
    let mut workspace = EvaluatorWorkspace::default();
    evaluate_displacement_with_workspace(material, &prepared, hit_ctx, &mut workspace)
}

/// Evaluates material displacement using caller-owned reusable graph scratch.
pub fn evaluate_displacement_with_workspace(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
) -> Result<f64> {
    workspace.active.clear();
    let result = evaluate_displacement_inner(material, prepared, hit_ctx, workspace);
    workspace.active.clear();
    result
}

fn evaluate_displacement_inner(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
) -> Result<f64> {
    let Some(graph_id) = &material.node_tree else {
        return Ok(0.0);
    };
    let group = hit_ctx.node_groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("shader graph `{graph_id}` was not found"),
            json!({"graph_id": graph_id}),
        )
    })?;
    if group.kind != GraphKind::Shader {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("material graph `{graph_id}` is not a shader graph"),
        ));
    }
    let mut fallback_output = None;
    let mut active_output = None;
    for (id, node) in &group.nodes {
        if matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
            if node
                .properties
                .get("is_active_output")
                .and_then(JsonValue::as_bool)
                == Some(true)
            {
                active_output = Some(id);
                break;
            }
            fallback_output.get_or_insert(id);
        }
    }
    let Some(output) = active_output.or(fallback_output) else {
        return Ok(0.0);
    };
    if !group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Displacement")
    {
        return Ok(0.0);
    }
    workspace.active.reserve(group.links.len());
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx: *hit_ctx,
        workspace,
        alpha_linked_images: &prepared.alpha_linked_images,
    };
    match evaluator.input(output, "Displacement", ValueKind::Vector)? {
        Value::Vector(displacement) => Ok(displacement.dot(hit_ctx.normal)),
        Value::Scalar(displacement) => Ok(displacement),
        Value::Color(displacement) => {
            Ok(DVec3::new(displacement[0], displacement[1], displacement[2]).dot(hit_ctx.normal))
        }
        Value::Shader(_) => Err(eval_error("material displacement is a shader")),
    }
}

fn apply_material_textures(
    material: &Material,
    hit_ctx: &HitContext<'_>,
    params: &mut BsdfParams,
) -> Result<()> {
    let has_textures = material.base_color_texture.is_some()
        || material.roughness_texture.is_some()
        || material.metallic_texture.is_some()
        || material.normal_texture.is_some();
    if !has_textures {
        return Ok(());
    }
    // Blender's exporter links a base texture's Alpha only for a non-opaque material, and wires roughness, metallic,
    // and normal from the texture's Color with Alpha unlinked. Cycles' colour read depends on that link.
    if let Some(texture) = &material.base_color_texture {
        let alpha_linked = material.alpha_mode != "opaque";
        let pixel = sample_material_texture(
            texture,
            hit_ctx,
            texture_layer(hit_ctx, texture),
            alpha_linked,
        )?;
        for (channel, texel) in params.base_color[..3].iter_mut().zip(pixel) {
            *channel *= texel;
        }
        if alpha_linked {
            params.alpha *= pixel[3];
        }
    }
    if let Some(texture) = &material.roughness_texture {
        let layer = texture_layer(hit_ctx, texture);
        params.roughness *= sample_material_texture(texture, hit_ctx, layer, false)?[0];
    }
    if let Some(texture) = &material.metallic_texture {
        let layer = texture_layer(hit_ctx, texture);
        params.metallic *= sample_material_texture(texture, hit_ctx, layer, false)?[0];
    }
    if let Some(texture) = &material.normal_texture {
        let layer = texture_layer(hit_ctx, texture);
        let pixel = sample_material_texture(texture, hit_ctx, layer, false)?;
        let tangent_normal = normalize_or(
            DVec3::new(
                pixel[0] * 2.0 - 1.0,
                pixel[1] * 2.0 - 1.0,
                pixel[2] * 2.0 - 1.0,
            ),
            DVec3::Z,
        );
        let normal = normalize_or(params.normal, hit_ctx.normal);
        let (tangent, bitangent) = layer_frame(hit_ctx, layer, normal);
        params.normal = normalize_or(
            tangent * tangent_normal.x + bitangent * tangent_normal.y + normal * tangent_normal.z,
            normal,
        );
    }
    params.base_color = params.base_color.map(|value| value.clamp(0.0, 1.0));
    params.alpha = params.alpha.clamp(0.0, 1.0);
    params.metallic = params.metallic.clamp(0.0, 1.0);
    params.roughness = params.roughness.clamp(0.0, 1.0);
    Ok(())
}

fn sample_material_texture(
    texture: &TextureRef,
    hit_ctx: &HitContext<'_>,
    layer: Option<usize>,
    alpha_linked: bool,
) -> Result<[f64; 4]> {
    let image_id = texture.image.as_str();
    let images = hit_ctx.image_data.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Material texture evaluation requires resolved image pixels in the hit context",
            json!({"feature_id":"shader.material_texture.image_data","image_id":image_id}),
        )
    })?;
    let image = images.get(image_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("image data `{image_id}` was not found in the hit context"),
            json!({"image_id":image_id}),
        )
    })?;
    // The field samples through the same path as an Image Texture node with the field's extension.
    let blender_uv = storage_to_blender_uv(layer_uv(hit_ctx, layer));
    let udim = image.udim;
    let pixel = sample_with_extension_alpha(
        image,
        blender_uv,
        udim,
        texture.interpolation,
        texture.extension,
        alpha_linked,
    );
    if pixel.iter().any(|component| !component.is_finite()) {
        return Err(eval_error(format!(
            "Material texture `{image_id}` sampled non-finite pixels"
        )));
    }
    Ok(pixel)
}

#[derive(Clone, Copy, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Shader variant stays inline so Value remains Copy; boxing it would add a heap allocation per shader-node evaluation"
)]
enum Value {
    Scalar(f64),
    Vector(DVec3),
    Color([f64; 4]),
    Shader(BsdfParams),
}

#[derive(Clone, Copy)]
enum ValueKind {
    Scalar,
    Vector,
    Color,
    Shader,
}
#[derive(Clone, Copy)]
enum NodeKind {
    OutputMaterial,
    Principled,
    VolumePrincipled,
    VolumeAbsorption,
    VolumeScatter,
    Displacement,
    ImageTexture,
    Emission,
    Diffuse,
    Glossy,
    Glass,
    Transparent,
    MixShader,
    AddShader,
    Bump,
    Checker,
    Noise,
    Voronoi,
    Gradient,
    ColorRamp,
    MixRgb,
    Math,
    VectorMath,
    Mapping,
    SeparateColor,
    SeparateXyz,
    CombineXyz,
    TexCoord,
    UvMap,
    NormalMap,
    Fresnel,
    LayerWeight,
    Attribute,
    VertexColor,
}

// Tokens identify a canonical node key and a source socket link without owning either string.
#[derive(Clone, Copy)]
struct ActiveSocket {
    node: usize,
    socket: usize,
}

/// Reusable scratch storage for evaluating shader graphs on render hits.
#[derive(Default)]
pub struct EvaluatorWorkspace {
    active: Vec<ActiveSocket>,
}

/// The per-material state a shader evaluation reads. A render builds it once for each material and passes it to every
/// evaluation of that material, so no hit walks the graph again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PreparedMaterial {
    /// The Image Texture nodes whose Alpha output reaches the graph's output (see [`linked_alpha_image_nodes`]).
    alpha_linked_images: BTreeSet<Id>,
    /// Whether EEVEE would compile the material as transparent (`GPU_MATFLAG_TRANSPARENT`), which decides whether its
    /// surfaces cast transparent shadows (see [`graph_is_transparent`]).
    transparent: bool,
}

impl PreparedMaterial {
    /// Prepares `material` against the node groups its graph is looked up in. A material without a shader graph prepares
    /// to its own PBR fields: its export is a Principled whose Alpha is the base colour alpha for a non-opaque material
    /// (`blend_bridge.py` `_set_principled_values`), so it is transparent when that alpha is below one.
    #[must_use]
    pub fn new(material: &Material, node_groups: &Registry<NodeGroup>) -> Self {
        let Some(group) = material
            .node_tree
            .as_ref()
            .and_then(|graph_id| node_groups.get(graph_id))
        else {
            return Self {
                transparent: material.alpha_mode != "opaque" && material.base_color[3] < 1.0,
                ..Self::default()
            };
        };
        let Some(output) = Self::active_output(group) else {
            return Self::default();
        };
        Self {
            alpha_linked_images: linked_alpha_image_nodes(group, output)
                .into_iter()
                .cloned()
                .collect(),
            transparent: graph_is_transparent(group, output),
        }
    }

    /// Whether a surface of this material casts a transparent shadow, as EEVEE's static transparency decides it.
    pub(crate) fn is_transparent(&self) -> bool {
        self.transparent
    }

    /// The output node a surface evaluates from: the active one, else the first `OutputMaterial`.
    fn active_output(group: &NodeGroup) -> Option<&Id> {
        let mut fallback = None;
        for (id, node) in &group.nodes {
            if matches!(
                node.node_type.as_str(),
                "OutputMaterial" | "ShaderNodeOutputMaterial"
            ) {
                if node
                    .properties
                    .get("is_active_output")
                    .and_then(JsonValue::as_bool)
                    == Some(true)
                {
                    return Some(id);
                }
                fallback.get_or_insert(id);
            }
        }
        fallback
    }
}

struct Evaluator<'graph, 'material, 'registry, 'workspace, 'prepared> {
    group: &'graph NodeGroup,
    material: &'material Material,
    hit_ctx: HitContext<'registry>,
    workspace: &'workspace mut EvaluatorWorkspace,
    /// The Alpha-linked Image Texture nodes of the material, from [`PreparedMaterial`].
    alpha_linked_images: &'prepared BTreeSet<Id>,
}

impl<'registry> Evaluator<'_, '_, 'registry, '_, '_> {
    fn input(&mut self, node_id: &Id, socket: &str, kind: ValueKind) -> Result<Value> {
        let node = self.node(node_id)?;
        let incoming =
            self.group.links.iter().enumerate().find(|(_, link)| {
                link.to_node == *node_id && socket_matches(&link.to_socket, socket)
            });
        if let Some((socket_index, link)) = incoming {
            return self.output(&link.from_node, &link.from_socket, socket_index);
        }
        let raw = node
            .inputs
            .iter()
            .find(|(name, _)| socket_matches(name, socket))
            .map(|(_, value)| value)
            .filter(|value| !value.is_null());
        let default = raw.or_else(|| socket_default(node, socket));
        match kind {
            ValueKind::Scalar => Ok(Value::Scalar(
                default.map_or(f64::NAN, |value| number(value, f64::NAN)),
            )),
            ValueKind::Vector => Ok(Value::Vector(
                default.map_or(DVec3::splat(f64::NAN), |value| {
                    vector(value, DVec3::splat(f64::NAN))
                }),
            )),
            ValueKind::Color => {
                Ok(Value::Color(default.map_or([f64::NAN; 4], |value| {
                    color(value, [f64::NAN; 4])
                })))
            }
            ValueKind::Shader => match raw {
                Some(_) => Err(invalid("a shader socket cannot use a literal value")),
                None => Ok(Value::Shader(self.closure_params())),
            },
        }
    }

    fn output(&mut self, node_id: &Id, socket: &str, socket_index: usize) -> Result<Value> {
        let Some((node_key, node)) = self.group.nodes.get_key_value(node_id) else {
            return Err(invalid(format!(
                "shader graph references absent node `{node_id}`"
            )));
        };
        let node_key = node_key.as_str().as_ptr() as usize;
        if self.workspace.active.iter().any(|active| {
            active.node == node_key
                && self
                    .group
                    .links
                    .get(active.socket)
                    .is_some_and(|link| link.from_socket.as_str() == socket)
        }) {
            return Err(eval_error(format!(
                "shader graph cycle at node `{node_id}`"
            )));
        }
        self.workspace.active.push(ActiveSocket {
            node: node_key,
            socket: socket_index,
        });
        // Map the borrowed node type to a value before dispatch mutably through the evaluator.
        let kind = match node.node_type.as_str() {
            "OutputMaterial" | "ShaderNodeOutputMaterial" => Ok(NodeKind::OutputMaterial),
            "ShaderNodeBsdfPrincipled" => Ok(NodeKind::Principled),
            "ShaderNodeVolumePrincipled" => Ok(NodeKind::VolumePrincipled),
            "ShaderNodeVolumeAbsorption" => Ok(NodeKind::VolumeAbsorption),
            "ShaderNodeVolumeScatter" => Ok(NodeKind::VolumeScatter),
            "ShaderNodeDisplacement" => Ok(NodeKind::Displacement),
            "ShaderNodeTexImage" => Ok(NodeKind::ImageTexture),
            "ShaderNodeEmission" => Ok(NodeKind::Emission),
            "ShaderNodeBsdfDiffuse" => Ok(NodeKind::Diffuse),
            "ShaderNodeBsdfGlossy" | "ShaderNodeBsdfAnisotropic" => Ok(NodeKind::Glossy),
            "ShaderNodeBsdfGlass" => Ok(NodeKind::Glass),
            "ShaderNodeBsdfTransparent" => Ok(NodeKind::Transparent),
            "ShaderNodeMixShader" => Ok(NodeKind::MixShader),
            "ShaderNodeAddShader" => Ok(NodeKind::AddShader),
            "ShaderNodeBump" => Ok(NodeKind::Bump),
            "ShaderNodeTexChecker" => Ok(NodeKind::Checker),
            "ShaderNodeTexNoise" => Ok(NodeKind::Noise),
            "ShaderNodeTexVoronoi" => Ok(NodeKind::Voronoi),
            "ShaderNodeTexGradient" => Ok(NodeKind::Gradient),
            "ShaderNodeValToRGB" => Ok(NodeKind::ColorRamp),
            "ShaderNodeMixRGB" | "ShaderNodeMix" => Ok(NodeKind::MixRgb),
            "ShaderNodeMath" => Ok(NodeKind::Math),
            "ShaderNodeVectorMath" => Ok(NodeKind::VectorMath),
            "ShaderNodeMapping" => Ok(NodeKind::Mapping),
            "ShaderNodeSeparateColor" => Ok(NodeKind::SeparateColor),
            "ShaderNodeSeparateXYZ" => Ok(NodeKind::SeparateXyz),
            "ShaderNodeCombineXYZ" => Ok(NodeKind::CombineXyz),
            "ShaderNodeTexCoord" => Ok(NodeKind::TexCoord),
            "ShaderNodeUVMap" => Ok(NodeKind::UvMap),
            "ShaderNodeNormalMap" => Ok(NodeKind::NormalMap),
            "ShaderNodeFresnel" => Ok(NodeKind::Fresnel),
            "ShaderNodeLayerWeight" => Ok(NodeKind::LayerWeight),
            "ShaderNodeAttribute" => Ok(NodeKind::Attribute),
            "ShaderNodeVertexColor" => Ok(NodeKind::VertexColor),
            unknown => Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("shader node `{unknown}` is not supported by the native evaluator"),
                json!({"feature_id":format!("shader.node.{unknown}"),"node_id":node_id}),
            )),
        };
        let result = kind.and_then(|kind| self.evaluate_node(node_id, kind, socket));
        self.workspace.active.pop();
        result
    }

    fn node(&self, node_id: &Id) -> Result<&GraphNode> {
        self.group
            .nodes
            .get(node_id)
            .ok_or_else(|| invalid(format!("shader graph references absent node `{node_id}`")))
    }

    /// Whether the Alpha output of the Image Texture `id` reaches the group's output (see [`PreparedMaterial`]).
    fn alpha_linked_image(&self, id: &Id) -> bool {
        self.alpha_linked_images.contains(id)
    }

    fn evaluate_node(&mut self, node_id: &Id, kind: NodeKind, socket: &str) -> Result<Value> {
        match kind {
            NodeKind::OutputMaterial => Err(invalid("material output has no output sockets")),
            NodeKind::Principled => Ok(Value::Shader(self.principled(node_id)?)),
            NodeKind::VolumePrincipled => Ok(Value::Shader(
                self.volume(node_id, VolumeClosure::Principled)?,
            )),
            NodeKind::VolumeScatter => {
                Ok(Value::Shader(self.volume(node_id, VolumeClosure::Scatter)?))
            }
            NodeKind::VolumeAbsorption => Ok(Value::Shader(
                self.volume(node_id, VolumeClosure::Absorption)?,
            )),
            NodeKind::Displacement => self.displacement(node_id, socket),
            NodeKind::ImageTexture => self.image_texture(node_id, socket),
            NodeKind::Emission => Ok(Value::Shader(self.emission(node_id)?)),
            NodeKind::Diffuse => {
                // Blender's Diffuse BSDF is a Lambert lobe with no specular. Its Roughness input is Oren-Nayar, which
                // Potter does not evaluate, so the default dielectric layer must not come from it: keeping it made a
                // roughness-0 floor under a sun a 1600x GGX spike (`source/blender/nodes/shader/nodes/node_shader_bsdf_diffuse.cc`).
                let mut params = self.diffuse(node_id)?;
                params.specular_ior_level = 0.0;
                Ok(Value::Shader(params))
            }
            NodeKind::Glossy => Ok(Value::Shader(self.glossy(node_id)?)),
            NodeKind::Glass => Ok(Value::Shader(self.glass(node_id)?)),
            NodeKind::Transparent => Ok(Value::Shader(self.transparent(node_id)?)),
            NodeKind::MixShader => Ok(Value::Shader(self.mix_shader(node_id)?)),
            NodeKind::AddShader => Ok(Value::Shader(self.add_shader(node_id)?)),
            NodeKind::Bump => self.bump(node_id, socket),
            NodeKind::Checker => self.checker(node_id, socket),
            NodeKind::Noise => self.noise(node_id, socket),
            NodeKind::Voronoi => self.voronoi(node_id, socket),
            NodeKind::Gradient => self.gradient(node_id, socket),
            NodeKind::ColorRamp => self.color_ramp(node_id, socket),
            NodeKind::MixRgb => self.mix_rgb(node_id, socket),
            NodeKind::Math => self.math(node_id, socket),
            NodeKind::VectorMath => self.vector_math(node_id, socket),
            NodeKind::Mapping => self.mapping(node_id, socket),
            NodeKind::SeparateColor => self.separate_color(node_id, socket),
            NodeKind::SeparateXyz => self.separate_xyz(node_id, socket),
            NodeKind::CombineXyz => self.combine_xyz(node_id, socket),
            NodeKind::TexCoord => self.tex_coord(socket),
            NodeKind::UvMap => self.uv_map(node_id, socket),
            NodeKind::NormalMap => self.normal_map(node_id, socket),
            NodeKind::Fresnel => self.fresnel(node_id, socket),
            NodeKind::LayerWeight => self.layer_weight(node_id, socket),
            NodeKind::Attribute => self.attribute(node_id, socket),
            NodeKind::VertexColor => self.vertex_color(node_id, socket),
        }
    }

    /// The parameters of a closure that adds no transparency of its own: alpha 1. A graph's material base colour
    /// alpha is not the surface's opacity (Cycles reads only Transparent BSDF, Principled Alpha and Mix through
    /// them), so only those closures set alpha. The non-graph path keeps `from_material`'s base colour alpha.
    fn closure_params(&self) -> BsdfParams {
        BsdfParams {
            alpha: 1.0,
            ..BsdfParams::from_material(self.material, self.hit_ctx.normal)
        }
    }

    fn principled(&mut self, id: &Id) -> Result<BsdfParams> {
        let node = self.node(id)?;
        if node
            .properties
            .get("potter_simple_material")
            .and_then(JsonValue::as_bool)
            == Some(true)
        {
            let mut params = self.closure_params();
            // A simple-material graph holds no Alpha input unless the overlay wrote one (`write_principled_inputs`, for a
            // non-opaque material). Without it the material's own alpha applies, as before the closure change: its base
            // colour alpha, or 1 for an OPAQUE material, whose graph carries no transparency.
            let default_alpha = if self.material.alpha_mode == "opaque" {
                1.0
            } else {
                self.material.base_color[3]
            };
            params.alpha = self.scalar_input(id, &["Alpha"], default_alpha)?;
            record_unevaluated_inputs(
                self.hit_ctx.warnings,
                &self.material.name,
                Some(id),
                &params,
            );
            return Ok(params);
        }
        let mut params = self.closure_params();
        params.base_color = self.color_input(id, &["Base Color"], [0.8, 0.8, 0.8, 1.0])?;
        params.metallic = self.scalar_input(id, &["Metallic"], 0.0)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.5)?;
        params.ior = self.scalar_input(id, &["IOR"], 1.5)?;
        params.alpha = self.scalar_input(id, &["Alpha"], 1.0)?;
        params.transmission =
            self.scalar_input(id, &["Transmission Weight", "Transmission"], 0.0)?;
        params.subsurface_weight =
            self.scalar_input(id, &["Subsurface Weight", "Subsurface"], 0.0)?;
        params.subsurface_radius = self
            .vector_input(id, &["Subsurface Radius"], DVec3::ONE)?
            .to_array();
        params.subsurface_scale = self.scalar_input(id, &["Subsurface Scale"], 0.05)?;
        params.coat_weight = self.scalar_input(id, &["Coat Weight", "Clearcoat"], 0.0)?;
        params.coat_roughness =
            self.scalar_input(id, &["Coat Roughness", "Clearcoat Roughness"], 0.03)?;
        params.coat_ior = self.scalar_input(id, &["Coat IOR"], 1.5)?;
        params.coat_tint = self.color_input(id, &["Coat Tint"], [1.0, 1.0, 1.0, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid coat tint"))?;
        params.sheen_weight = self.scalar_input(id, &["Sheen Weight", "Sheen"], 0.0)?;
        params.sheen_roughness = self.scalar_input(id, &["Sheen Roughness"], 0.5)?;
        params.sheen_tint = self.color_input(id, &["Sheen Tint"], [1.0, 1.0, 1.0, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid sheen tint"))?;
        params.specular_ior_level = self.scalar_input(
            id,
            &["Specular IOR Level", "Specular"],
            DEFAULT_SPECULAR_IOR_LEVEL,
        )?;
        params.specular_tint = self.color_input(id, &["Specular Tint"], [1.0, 1.0, 1.0, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid specular tint"))?;
        params.anisotropic_ior =
            self.scalar_input(id, &["Anisotropic IOR Level", "Anisotropic"], 0.0)?;
        params.anisotropic_rotation = self.scalar_input(id, &["Anisotropic Rotation"], 0.0)?;
        params.thin_film_thickness = self.scalar_input(id, &["Thin Film Thickness"], 0.0)?;
        params.thin_film_ior = self.scalar_input(id, &["Thin Film IOR"], 1.33)?;
        params.diffuse_roughness = self.scalar_input(id, &["Diffuse Roughness"], 0.0)?;
        params.thin_wall = self.scalar_input(id, &["Thin Wall"], 0.0)? > 0.5;
        params.emission_color =
            self.color_input(id, &["Emission Color", "Emission"], [0.0, 0.0, 0.0, 1.0])?[..3]
                .try_into()
                .map_err(|_| invalid("invalid emission color"))?;
        params.emission_strength = self.scalar_input(id, &["Emission Strength"], 0.0)?;
        params.normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        // An unlinked socket has no vector (zero or non-finite), so the coat follows the final normal.
        let coat_normal = self.vector_input(id, &["Coat Normal"], DVec3::ZERO)?;
        params.coat_normal =
            (coat_normal.length_squared() > f64::EPSILON).then(|| coat_normal.normalize());
        record_unevaluated_inputs(
            self.hit_ctx.warnings,
            &self.material.name,
            Some(id),
            &params,
        );
        Ok(params)
    }

    fn volume(&mut self, id: &Id, closure: VolumeClosure) -> Result<BsdfParams> {
        let mut params = self.closure_params();
        let density = self.scalar_input(id, &["Density"], 1.0)?.max(0.0);
        // Blender's node defaults: Principled Volume 0.5; Volume Absorption and Scatter 0.8.
        let default_color = match closure {
            VolumeClosure::Principled => [0.5, 0.5, 0.5, 1.0],
            VolumeClosure::Scatter | VolumeClosure::Absorption => [0.8, 0.8, 0.8, 1.0],
        };
        let color: [f64; 3] = self.color_input(id, &["Color"], default_color)?[..3]
            .try_into()
            .map_err(|_| invalid("invalid volume color"))?;
        let weights = match closure {
            VolumeClosure::Principled => {
                let absorption_color: [f64; 3] =
                    self.color_input(id, &["Absorption Color"], [0.0, 0.0, 0.0, 1.0])?[..3]
                        .try_into()
                        .map_err(|_| invalid("invalid volume absorption color"))?;
                principled_volume_weights(color, absorption_color)
            }
            // Cycles' scatter closure: the colour weights scattering and extinction alike.
            VolumeClosure::Scatter => VolumeWeights {
                scatter: color.map(|channel| channel.max(0.0)),
                absorption: [0.0; 3],
            },
            // Cycles' absorption closure: extinction weight `1 - color`, no scattering.
            VolumeClosure::Absorption => VolumeWeights {
                scatter: [0.0; 3],
                absorption: color.map(|channel| (1.0 - channel).max(0.0)),
            },
        };
        // Cycles' closure weight times the Density socket (`svm_node_closure_volume`).
        params.volume_scatter = weights.scatter.map(|weight| weight * density);
        params.volume_absorption = weights.absorption.map(|weight| weight * density);
        params.volume_anisotropy = self
            .scalar_input(id, &["Anisotropy"], 0.0)?
            .clamp(-0.999, 0.999);
        params.volume_emission = if closure == VolumeClosure::Absorption {
            [0.0; 3]
        } else {
            let emission: [f64; 3] =
                self.color_input(id, &["Emission Color", "Emission"], [0.0, 0.0, 0.0, 1.0])?[..3]
                    .try_into()
                    .map_err(|_| invalid("invalid volume emission color"))?;
            let strength = self.scalar_input(id, &["Emission Strength"], 0.0)?.max(0.0);
            emission.map(|channel| channel * strength * density)
        };
        Ok(params)
    }

    fn displacement(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if !matches!(socket, "Displacement" | "Vector") {
            return Err(invalid(format!("unknown Displacement output `{socket}`")));
        }
        let height = self.scalar_input(id, &["Height"], 0.0)?;
        let midlevel = self.scalar_input(id, &["Midlevel"], self.material.displacement_midlevel)?;
        let scale = self.scalar_input(id, &["Scale"], self.material.displacement_scale)?;
        let normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        Ok(Value::Vector(normal * ((height - midlevel) * scale)))
    }

    fn emission(&mut self, id: &Id) -> Result<BsdfParams> {
        let color = self.color_input(id, &["Color", "Emission"], [1.0, 1.0, 1.0, 1.0])?;
        let mut params = self.closure_params();
        params.base_color = color;
        params.emission_color = [color[0], color[1], color[2]];
        params.emission_strength = self.scalar_input(id, &["Strength"], 1.0)?;
        // An emission closure alone adds radiance and no reflection: Cycles'
        // `EmissionNode::compile` (intern/cycles/scene/shader_nodes.cpp:3071-3098) emits only
        // `NODE_CLOSURE_EMISSION`.
        params.reflectance = 0.0;
        Ok(params)
    }

    fn diffuse(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.closure_params();
        params.base_color = self.color_input(id, &["Color"], [0.8, 0.8, 0.8, 1.0])?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.0)?;
        params.normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        Ok(params)
    }

    fn glossy(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.metallic = 1.0;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.5)?;
        Ok(params)
    }

    fn glass(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.transmission = 1.0;
        params.ior = self.scalar_input(id, &["IOR"], 1.5)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.0)?;
        Ok(params)
    }

    fn transparent(&mut self, id: &Id) -> Result<BsdfParams> {
        // Blender's Transparent BSDF defaults to white. Its Color is the closure weight that a pass-through carries
        // (`intern/cycles/kernel/closure/bsdf_transparent.h`), so it is the transparent tint, not a surface colour.
        let tint = self.color_input(id, &["Color"], [1.0; 4])?;
        let mut params = self.diffuse(id)?;
        params.base_color = [tint[0], tint[1], tint[2], 0.0];
        params.alpha = 0.0;
        params.transparent_tint = [tint[0], tint[1], tint[2]];
        params.transmission = 1.0;
        Ok(params)
    }

    fn mix_shader(&mut self, id: &Id) -> Result<BsdfParams> {
        let factor = self.scalar_input(id, &["Fac"], 0.5)?;
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(BsdfParams::mix(first, second, factor))
    }

    fn add_shader(&mut self, id: &Id) -> Result<BsdfParams> {
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(BsdfParams::add(first, second))
    }

    fn shader_input(&mut self, id: &Id, names: &[&str]) -> Result<BsdfParams> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Shader"),
            ValueKind::Shader,
        )? {
            Value::Shader(value) => Ok(value),
            _ => Err(invalid("shader socket type mismatch")),
        }
    }

    fn image_texture(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let uv = self.vector_input(
            id,
            &["Vector"],
            DVec3::new(self.hit_ctx.uv[0], 1.0 - self.hit_ctx.uv[1], 0.0),
        )?;
        let node = self.node(id)?;
        let image_id = node
            .properties
            .get("image")
            .or_else(|| node.properties.get("image_id"))
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid(format!("Image Texture node `{id}` requires an image ID")))?;
        if !Id::is_valid(image_id) {
            return Err(invalid(format!(
                "Image Texture node `{id}` has an invalid image ID"
            )));
        }
        let images = self.hit_ctx.image_data.ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Image Texture evaluation requires resolved image pixels in the hit context",
                json!({"feature_id":"shader.node.ShaderNodeTexImage.image_data","node_id":id,"image_id":image_id}),
            )
        })?;
        let image = images.get(image_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("image data `{image_id}` was not found in the hit context"),
                json!({"image_id":image_id,"node_id":id}),
            )
        })?;
        // A UDIM image routes its lookup from the UV; the node has no tile property (Cycles' kernel_image_udim_map).
        let udim = image.udim;
        let interpolation = match node
            .properties
            .get("interpolation")
            .and_then(JsonValue::as_str)
        {
            Some("linear" | "Linear" | "LINEAR") => ImageInterpolation::Linear,
            Some("closest" | "Closest" | "CLOSEST") => ImageInterpolation::Closest,
            Some(interpolation) => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("Image Texture interpolation `{interpolation}` is not supported"),
                    json!({"feature_id":"shader.node.ShaderNodeTexImage.interpolation","node_id":id,"interpolation":interpolation}),
                ));
            }
            None => image.interpolation,
        };
        let extension = match node.properties.get("extension").and_then(JsonValue::as_str) {
            None | Some("REPEAT") => ImageExtension::Repeat,
            Some("EXTEND") => ImageExtension::Extend,
            Some("CLIP") => ImageExtension::Clip,
            Some("MIRROR") => ImageExtension::Mirror,
            Some(extension) => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("Image Texture extension `{extension}` is not supported"),
                    json!({"feature_id":"shader.node.ShaderNodeTexImage.extension","node_id":id,"extension":extension}),
                ));
            }
        };
        // Cycles unassociates an Image Texture's colour only when its Alpha output still has a link after
        // dead nodes are removed, so the colour read depends on that link (see `linked_alpha_image_nodes`).
        let alpha_linked = self.alpha_linked_image(id);
        let pixel = sample_with_extension_alpha(
            image,
            [uv.x, uv.y],
            udim,
            interpolation,
            extension,
            alpha_linked,
        );
        if pixel.iter().any(|component| !component.is_finite()) {
            return Err(eval_error(format!(
                "Image Texture node `{id}` sampled non-finite pixels"
            )));
        }
        match socket {
            "Color" => Ok(Value::Color(pixel)),
            "Alpha" => Ok(Value::Scalar(pixel[3])),
            _ => Err(invalid(format!("unknown Image Texture output `{socket}`"))),
        }
    }

    fn bump(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Normal" {
            return Err(invalid(format!("unknown Bump output `{socket}`")));
        }
        let normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        let strength = self.scalar_input(id, &["Strength"], 1.0)?;
        let distance = self.scalar_input(id, &["Distance"], 1.0)?;
        if strength.abs() <= f64::EPSILON || distance.abs() <= f64::EPSILON {
            return Ok(Value::Vector(normal));
        }
        let context = self.hit_ctx;
        let gradient = height_gradient(&context, normal, |context| self.height_at(id, context))?;
        let invert = self
            .node(id)?
            .properties
            .get("invert")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let sign = if invert { -1.0 } else { 1.0 };
        let perturbed = normal - gradient * (strength * distance * sign);
        Ok(Value::Vector(normalize_or(perturbed, normal)))
    }

    fn height_at(&mut self, id: &Id, context: &HitContext<'registry>) -> Result<f64> {
        let previous = self.hit_ctx;
        self.hit_ctx = *context;
        let result = self.scalar_input(id, &["Height"], 0.5);
        self.hit_ctx = previous;
        result
    }
    fn checker(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(
            id,
            &["Vector"],
            DVec3::new(self.hit_ctx.uv[0], self.hit_ctx.uv[1], 0.0),
        )?;
        let scale = self.scalar_input(id, &["Scale"], 5.0)?;
        let scaled = vector * scale;
        let parity = scaled.x.floor() + scaled.y.floor() + scaled.z.floor();
        let first = self.color_input(id, &["Color1"], [0.2, 0.2, 0.2, 1.0])?;
        let second = self.color_input(id, &["Color2"], [0.8, 0.8, 0.8, 1.0])?;
        let first_chosen = parity.rem_euclid(2.0) < 1.0;
        let chosen = if first_chosen { first } else { second };
        match socket {
            "Color" => Ok(Value::Color(chosen)),
            "Fac" | "Factor" => Ok(Value::Scalar(if first_chosen { 0.0 } else { 1.0 })),
            _ => Err(invalid(format!(
                "unknown Checker Texture output `{socket}`"
            ))),
        }
    }

    fn noise(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let scale = self.scalar_input(id, &["Scale"], 5.0)?;
        let detail = self.scalar_input(id, &["Detail"], 2.0)?.clamp(0.0, 8.0);
        let roughness = self.scalar_input(id, &["Roughness"], 0.5)?.clamp(0.0, 1.0);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut total_amplitude = 0.0;
        let mut frequency = scale;
        for _ in 0..=detail.floor() as usize {
            value += perlin(vector * frequency) * amplitude;
            total_amplitude += amplitude;
            frequency *= 2.0;
            amplitude *= roughness;
        }
        let factor = (0.5 + 0.5 * value / total_amplitude.max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(factor)),
            "Color" => Ok(Value::Color([factor, factor, factor, 1.0])),
            _ => Err(invalid(format!("unknown Noise Texture output `{socket}`"))),
        }
    }

    fn voronoi(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?
            * self.scalar_input(id, &["Scale"], 5.0)?;
        let cell = vector.floor();
        let mut nearest = f64::INFINITY;
        let mut nearest_point = DVec3::ZERO;
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let coordinate = cell + DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    let offset = DVec3::new(
                        lattice_random(coordinate, 0),
                        lattice_random(coordinate, 1),
                        lattice_random(coordinate, 2),
                    );
                    let point = coordinate + offset;
                    let distance = (vector - point).length();
                    if distance < nearest {
                        nearest = distance;
                        nearest_point = point;
                    }
                }
            }
        }
        match socket {
            "Distance" | "Fac" => Ok(Value::Scalar(nearest)),
            "Color" => Ok(Value::Color([
                lattice_random(nearest_point, 3),
                lattice_random(nearest_point, 4),
                lattice_random(nearest_point, 5),
                1.0,
            ])),
            "Position" => Ok(Value::Vector(nearest_point)),
            _ => Err(invalid(format!(
                "unknown Voronoi Texture output `{socket}`"
            ))),
        }
    }

    fn gradient(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let gradient_type = self
            .node(id)?
            .properties
            .get("gradient_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("LINEAR");
        let factor = match gradient_type {
            "LINEAR" => vector.x,
            "QUADRATIC" => vector.x.max(0.0).powi(2),
            "EASING" => {
                let x = vector.x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            }
            "SPHERICAL" => vector.length(),
            "QUADRATIC_SPHERE" => vector.length_squared(),
            "RADIAL" => vector.y.atan2(vector.x) / std::f64::consts::TAU + 0.5,
            other => return Err(unsupported_node(id, &format!("Gradient:{other}"))),
        }
        .clamp(0.0, 1.0);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(factor)),
            "Color" => Ok(Value::Color([factor, factor, factor, 1.0])),
            _ => Err(invalid(format!(
                "unknown Gradient Texture output `{socket}`"
            ))),
        }
    }

    fn color_ramp(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let factor = self.scalar_input(id, &["Fac"], 0.0)?.clamp(0.0, 1.0);
        let node = self.node(id)?;
        let result = if let Some(elements) = node
            .properties
            .get("elements")
            .and_then(JsonValue::as_array)
        {
            if elements.len() < 2 {
                return Err(invalid("ColorRamp requires at least two color elements"));
            }
            let interpolation = node
                .properties
                .get("interpolation")
                .and_then(JsonValue::as_str)
                .unwrap_or("LINEAR");
            let mut previous = ramp_element(&elements[0])?;
            if factor <= previous.0 {
                previous.1
            } else {
                let mut result = previous.1;
                for raw in elements.iter().skip(1) {
                    let next = ramp_element(raw)?;
                    if next.0 < previous.0 {
                        return Err(invalid("ColorRamp element positions must be ordered"));
                    }
                    if factor <= next.0 {
                        let local_factor = if next.0 - previous.0 <= f64::EPSILON {
                            1.0
                        } else {
                            ((factor - previous.0) / (next.0 - previous.0)).clamp(0.0, 1.0)
                        };
                        let local_factor = match interpolation {
                            "LINEAR" => local_factor,
                            "EASE" => local_factor * local_factor * (3.0 - 2.0 * local_factor),
                            "CONSTANT" => 0.0,
                            other => {
                                return Err(unsupported_node(id, &format!("ColorRamp:{other}")));
                            }
                        };
                        result = color_lerp(previous.1, next.1, local_factor);
                        break;
                    }
                    previous = next;
                    result = previous.1;
                }
                result
            }
        } else {
            let first = node
                .properties
                .get("color1")
                .map_or([0.0, 0.0, 0.0, 1.0], |value| {
                    color(value, [0.0, 0.0, 0.0, 1.0])
                });
            let second = node
                .properties
                .get("color2")
                .map_or([1.0; 4], |value| color(value, [1.0; 4]));
            color_lerp(first, second, factor)
        };
        match socket {
            "Color" => Ok(Value::Color(result)),
            "Alpha" => Ok(Value::Scalar(result[3])),
            _ => Err(invalid(format!("unknown ColorRamp output `{socket}`"))),
        }
    }

    /// `ShaderNodeMix` (RGBA, Blender 4.x and later) and the legacy `ShaderNodeMixRGB`.
    fn mix_rgb(&mut self, id: &Id, socket: &str) -> Result<Value> {
        #[derive(Clone, Copy)]
        enum Blend {
            Mix,
            Add,
            Multiply,
            Subtract,
            Divide,
        }

        enum BlendSelection {
            Supported(Blend),
            Unsupported(String),
        }

        let (generic, blend, label) = {
            let node = self.node(id)?;
            let generic = node.node_type == "ShaderNodeMix";
            let label = if generic { "Mix" } else { "MixRGB" };
            let data_type = node
                .properties
                .get("data_type")
                .and_then(JsonValue::as_str)
                .unwrap_or(if generic { "FLOAT" } else { "RGBA" });
            if data_type != "RGBA" {
                return Err(unsupported_node(id, &format!("Mix:{data_type}")));
            }
            let blend = match node
                .properties
                .get("blend_type")
                .and_then(JsonValue::as_str)
                .unwrap_or("MIX")
            {
                "MIX" => BlendSelection::Supported(Blend::Mix),
                "ADD" => BlendSelection::Supported(Blend::Add),
                "MULTIPLY" => BlendSelection::Supported(Blend::Multiply),
                "SUBTRACT" => BlendSelection::Supported(Blend::Subtract),
                "DIVIDE" => BlendSelection::Supported(Blend::Divide),
                other => BlendSelection::Unsupported(other.to_owned()),
            };
            (generic, blend, label)
        };
        // The generic node names its sockets by type; the legacy node uses plain names.
        let (factor_socket, first_socket, second_socket, result_socket) = if generic {
            ("Factor_Float", "A_Color", "B_Color", "Result_Color")
        } else {
            ("Fac", "Color1", "Color2", "Color")
        };
        let factor = self
            .scalar_input(id, &[factor_socket], 0.5)?
            .clamp(0.0, 1.0);
        let first = self.color_input(id, &[first_socket], [0.5, 0.5, 0.5, 1.0])?;
        let second = self.color_input(id, &[second_socket], [0.5, 0.5, 0.5, 1.0])?;
        let blend = match blend {
            BlendSelection::Supported(blend) => blend,
            BlendSelection::Unsupported(other) => {
                return Err(unsupported_node(id, &format!("{label}:{other}")));
            }
        };
        let blended = match blend {
            Blend::Mix => second,
            Blend::Add => color_binary(first, second, |a, b| a + b),
            Blend::Multiply => color_binary(first, second, |a, b| a * b),
            Blend::Subtract => color_binary(first, second, |a, b| a - b),
            Blend::Divide => color_binary(first, second, |a, b| {
                if b.abs() <= f64::EPSILON { a } else { a / b }
            }),
        };
        let result = color_lerp(first, blended, factor);
        if socket == result_socket {
            return Ok(Value::Color(result));
        }
        match socket {
            "Alpha" if !generic => Ok(Value::Scalar(result[3])),
            _ => Err(invalid(format!("unknown {label} output `{socket}`"))),
        }
    }
    /// `ShaderNodeSeparateXYZ`: one component of its vector input as a scalar.
    fn separate_xyz(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], DVec3::ZERO)?;
        let component = match socket {
            "X" => vector.x,
            "Y" => vector.y,
            "Z" => vector.z,
            _ => return Err(invalid(format!("unknown Separate XYZ output `{socket}`"))),
        };
        Ok(Value::Scalar(component))
    }

    /// `ShaderNodeCombineXYZ`: a vector from its three scalar inputs.
    fn combine_xyz(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Vector" {
            return Err(invalid(format!("unknown Combine XYZ output `{socket}`")));
        }
        let x = self.scalar_input(id, &["X"], 0.0)?;
        let y = self.scalar_input(id, &["Y"], 0.0)?;
        let z = self.scalar_input(id, &["Z"], 0.0)?;
        Ok(Value::Vector(DVec3::new(x, y, z)))
    }

    fn math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let first = self.scalar_input(id, &["Value", "Value_001"], 0.0)?;
        let second = self.scalar_input(id, &["Value_001", "Value_002"], 0.0)?;
        let third = self.scalar_input(id, &["Value_002", "Value_003"], 0.0)?;
        let operation = self
            .node(id)?
            .properties
            .get("operation")
            .and_then(JsonValue::as_str)
            .unwrap_or("ADD");
        let result = match operation {
            "ADD" => first + second,
            "SUBTRACT" => first - second,
            "MULTIPLY" => first * second,
            "DIVIDE" => {
                if crate::float::equal_f64(second, 0.0) {
                    0.0
                } else {
                    first / second
                }
            }
            "MULTIPLY_ADD" => first * second + third,
            "POWER" if first >= 0.0 || second.fract() == 0.0 => first.powf(second),
            "POWER" => 0.0,
            "SINE" => first.sin(),
            "COSINE" => first.cos(),
            "TANGENT" => first.tan(),
            "SQRT" => first.max(0.0).sqrt(),
            "ABSOLUTE" => first.abs(),
            "MINIMUM" => first.min(second),
            "MAXIMUM" => first.max(second),
            "LESS_THAN" => f64::from(first < second),
            "GREATER_THAN" => f64::from(first > second),
            "COMPARE" => {
                let epsilon = self.scalar_input(id, &["Value_002", "Value_003"], 0.001)?;
                f64::from((first - second).abs() <= epsilon.max(0.0))
            }
            "WRAP" => {
                // Blender's `wrapf(value, max, min)`: the second input is the maximum and the third the minimum.
                let range = second - third;
                if crate::float::equal_f64(range, 0.0) {
                    third
                } else {
                    first - range * ((first - third) / range).floor()
                }
            }
            "PINGPONG" => {
                // Blender's `pingpongf(value, scale)` (`math_base_inline.cc`): a zero scale gives 0, and otherwise the
                // triangle wave of the signed scale, `|fract((value - scale) / (2 scale)) * 2 scale - scale|`.
                let wave = if crate::float::equal_f64(second, 0.0) {
                    0.0
                } else {
                    let period = second * 2.0;
                    let ratio = (first - second) / period;
                    (ratio - ratio.floor()) * period - second
                };
                wave.abs()
            }
            other => return Err(unsupported_node(id, &format!("Math:{other}"))),
        };
        if !result.is_finite() {
            return Err(eval_error(format!(
                "Math node `{id}` produced a non-finite result"
            )));
        }
        match socket {
            "Value" => Ok(Value::Scalar(result)),
            _ => Err(invalid(format!("unknown Math output `{socket}`"))),
        }
    }

    fn vector_math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let first = self.vector_input(id, &["Vector", "Vector_001"], DVec3::ZERO)?;
        let second = self.vector_input(id, &["Vector_001", "Vector_002"], DVec3::ZERO)?;
        let operation = self
            .node(id)?
            .properties
            .get("operation")
            .and_then(JsonValue::as_str)
            .unwrap_or("ADD");
        let (vector, scalar) = match operation {
            "ADD" => (first + second, None),
            "SUBTRACT" => (first - second, None),
            "MULTIPLY" => (first * second, None),
            "CROSS_PRODUCT" => (first.cross(second), None),
            "DOT_PRODUCT" => (DVec3::ZERO, Some(first.dot(second))),
            "LENGTH" => (DVec3::ZERO, Some(first.length())),
            "DISTANCE" => (DVec3::ZERO, Some(first.distance(second))),
            "NORMALIZE" => (normalize_or(first, DVec3::ZERO), None),
            "SCALE" => (first * self.scalar_input(id, &["Scale"], 1.0)?, None),
            other => return Err(unsupported_node(id, &format!("VectorMath:{other}"))),
        };
        if !vector.is_finite() || scalar.is_some_and(|value| !value.is_finite()) {
            return Err(eval_error(format!(
                "Vector Math node `{id}` produced a non-finite result"
            )));
        }
        if socket == "Value" {
            scalar
                .map(Value::Scalar)
                .ok_or_else(|| invalid("Vector Math node has no scalar result"))
        } else if socket == "Vector" {
            Ok(Value::Vector(vector))
        } else {
            Err(invalid(format!("unknown Vector Math output `{socket}`")))
        }
    }

    fn mapping(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let location = self.vector_input(id, &["Location"], DVec3::ZERO)?;
        let rotation = self.vector_input(id, &["Rotation"], DVec3::ZERO)?;
        let scale = self.vector_input(id, &["Scale"], DVec3::ONE)?;
        let rotation = DQuat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z);
        let vector_type = self
            .node(id)?
            .properties
            .get("vector_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("POINT");
        let mapped = match vector_type {
            "POINT" => location + rotation * (vector * scale),
            "VECTOR" => rotation * (vector * scale),
            "TEXTURE" | "NORMAL" => {
                if scale.x == 0.0 || scale.y == 0.0 || scale.z == 0.0 {
                    return Err(unsupported_node(
                        id,
                        &format!("Mapping:{vector_type}:zero_scale"),
                    ));
                }
                if vector_type == "TEXTURE" {
                    rotation.conjugate() * (vector - location) / scale
                } else {
                    (rotation * (vector / scale)).normalize_or_zero()
                }
            }
            other => return Err(unsupported_node(id, &format!("Mapping:{other}"))),
        };
        match socket {
            "Vector" => Ok(Value::Vector(mapped)),
            _ => Err(invalid(format!("unknown Mapping output `{socket}`"))),
        }
    }
    /// `ShaderNodeSeparateColor`: one channel of its color input as a scalar.
    fn separate_color(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if let Some(mode) = self
            .node(id)?
            .properties
            .get("mode")
            .and_then(JsonValue::as_str)
            .filter(|mode| *mode != "RGB")
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("Separate Color mode `{mode}` is not implemented"),
                json!({"feature_id":format!("shader.node.SeparateColor.{mode}"),"node_id":id}),
            ));
        }
        let color = self.color_input(id, &["Color"], [0.0; 4])?;
        let channel = match socket {
            "Red" => color[0],
            "Green" => color[1],
            "Blue" => color[2],
            _ => return Err(invalid(format!("unknown Separate Color output `{socket}`"))),
        };
        Ok(Value::Scalar(channel))
    }

    fn tex_coord(&self, socket: &str) -> Result<Value> {
        let value = match socket {
            "Generated" => self.hit_ctx.generated,
            "UV" => {
                let uv = storage_to_blender_uv(self.hit_ctx.uv);
                DVec3::new(uv[0], uv[1], 0.0)
            }
            "Object" => self.hit_ctx.object,
            "Normal" => self.hit_ctx.normal,
            "Position" => self.hit_ctx.position,
            "Camera" | "Window" | "Reflection" => {
                return Err(unsupported_node_id("TexCoord", socket));
            }
            _ => {
                return Err(invalid(format!(
                    "unknown Texture Coordinate output `{socket}`"
                )));
            }
        };
        Ok(Value::Vector(value))
    }
    /// `ShaderNodeUVMap`: the UV of the layer named by the node, or the render UV when it names none.
    fn uv_map(&self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "UV" {
            return Err(invalid(format!("unknown UV Map output `{socket}`")));
        }
        let node = self.node(id)?;
        let layer = layer_index(
            &self.hit_ctx,
            node.properties.get("uv_map").and_then(JsonValue::as_str),
        );
        let uv = storage_to_blender_uv(layer_uv(&self.hit_ctx, layer));
        Ok(Value::Vector(DVec3::new(uv[0], uv[1], 0.0)))
    }
    fn normal_map(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Normal" {
            return Err(invalid(format!("unknown Normal Map output `{socket}`")));
        }
        let space = self
            .node(id)?
            .properties
            .get("space")
            .and_then(JsonValue::as_str)
            .unwrap_or("TANGENT_SPACE");
        if !matches!(space, "TANGENT_SPACE" | "TANGENT") {
            return Err(unsupported_node(id, &format!("NormalMap:{space}")));
        }
        let imported_normal_map = self
            .node(id)?
            .properties
            .get("potter_gltf_imported_normal_map")
            .and_then(JsonValue::as_bool)
            == Some(true);
        let color = self.color_input(id, &["Color"], [0.5, 0.5, 1.0, 1.0])?;
        let tangent_normal = DVec3::new(
            color[0] * 2.0 - 1.0,
            color[1] * 2.0 - 1.0,
            color[2] * 2.0 - 1.0,
        );
        let base_normal = normalize_or(self.hit_ctx.normal, DVec3::Z);
        let uv_map = self
            .node(id)?
            .properties
            .get("uv_map")
            .and_then(JsonValue::as_str);
        let layer = layer_index(&self.hit_ctx, uv_map);
        let (tangent_u, tangent_v) = if imported_normal_map {
            layer_frame(&self.hit_ctx, layer, base_normal)
        } else {
            blender_layer_frame(&self.hit_ctx, layer, base_normal)
        };
        let mapped_normal = normalize_or(
            tangent_u * tangent_normal.x
                + tangent_v * tangent_normal.y
                + base_normal * tangent_normal.z,
            base_normal,
        );
        let strength = self.scalar_input(id, &["Strength"], 1.0)?.clamp(0.0, 1.0);
        Ok(Value::Vector(normalize_or(
            base_normal.lerp(mapped_normal, strength),
            base_normal,
        )))
    }

    fn fresnel(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let ior = self.scalar_input(id, &["IOR"], 1.45)?.max(1.0);
        let cos_theta = normalize_or(self.hit_ctx.normal, DVec3::Z)
            .dot(normalize_or(self.hit_ctx.view_direction, DVec3::Z))
            .abs()
            .clamp(0.0, 1.0);
        let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
        let facing = r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(facing)),
            _ => Err(invalid(format!("unknown Fresnel output `{socket}`"))),
        }
    }

    fn layer_weight(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let blend = self.scalar_input(id, &["Blend"], 0.5)?.clamp(0.0, 1.0);
        let cos_theta = normalize_or(self.hit_ctx.normal, DVec3::Z)
            .dot(normalize_or(self.hit_ctx.view_direction, DVec3::Z))
            .abs()
            .clamp(0.0, 1.0);
        let facing = cos_theta.powf(blend.max(0.001));
        match socket {
            "Facing" => Ok(Value::Scalar(facing)),
            "Fresnel" => Ok(Value::Scalar(1.0 - facing)),
            _ => Err(invalid(format!("unknown Layer Weight output `{socket}`"))),
        }
    }

    /// `ShaderNodeAttribute`. The `Color` output is RGB with alpha 1.0: EEVEE's `node_attribute`
    /// sets `outcol = float4(attr.xyz, 1.0f)` (`gpu/shaders/material/gpu_shader_material_attribute.glsl`).
    /// Cycles' Color is a float3 output (`intern/cycles/scene/shader_nodes.cpp`,
    /// `AttributeNode::compile`, `NODE_ATTR_OUTPUT_FLOAT3`). The `Alpha` output carries the layer
    /// alpha (EEVEE `outalpha = attr.w`; Cycles `NODE_ATTR_OUTPUT_FLOAT_ALPHA`). A missing layer
    /// loads as opaque black.
    fn attribute(&self, id: &Id, socket: &str) -> Result<Value> {
        let node = self.node(id)?;
        let name = node
            .properties
            .get("attribute_name")
            .or_else(|| node.properties.get("name"))
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid("Attribute node requires attribute_name"))?;
        let [red, green, blue, alpha] =
            layer_color(&self.hit_ctx, name, false, [0.0, 0.0, 0.0, 1.0]);
        match socket {
            "Color" => Ok(Value::Color([red, green, blue, 1.0])),
            _ => color_output([red, green, blue, alpha], socket, "Attribute"),
        }
    }

    /// `ShaderNodeVertexColor`: the named color layer, or the render layer when the name is empty.
    fn vertex_color(&self, id: &Id, socket: &str) -> Result<Value> {
        let node = self.node(id)?;
        let name = node
            .properties
            .get("layer_name")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        color_output(
            layer_color(&self.hit_ctx, name, name.is_empty(), [0.0, 0.0, 0.0, 0.0]),
            socket,
            "Vertex Color",
        )
    }

    fn scalar_input(&mut self, id: &Id, names: &[&str], default: f64) -> Result<f64> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Value"),
            ValueKind::Scalar,
        )? {
            Value::Scalar(value) => Ok(value),
            Value::Vector(value) => Ok(value.x),
            Value::Color(value) => Ok(value[0]),
            Value::Shader(_) => Err(invalid("expected a scalar shader socket")),
        }
        .map(|value| if value.is_finite() { value } else { default })
    }

    fn vector_input(&mut self, id: &Id, names: &[&str], default: DVec3) -> Result<DVec3> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Vector"),
            ValueKind::Vector,
        )? {
            Value::Vector(value) => Ok(value),
            Value::Scalar(value) => Ok(DVec3::splat(value)),
            Value::Color(value) => Ok(DVec3::new(value[0], value[1], value[2])),
            Value::Shader(_) => Err(invalid("expected a vector shader socket")),
        }
        .map(|value| if value.is_finite() { value } else { default })
    }

    fn color_input(&mut self, id: &Id, names: &[&str], default: [f64; 4]) -> Result<[f64; 4]> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Color"),
            ValueKind::Color,
        )? {
            Value::Color(value) => Ok(value),
            Value::Scalar(value) => Ok([value, value, value, 1.0]),
            Value::Vector(value) => Ok([value.x, value.y, value.z, 1.0]),
            Value::Shader(_) => Err(invalid("expected a color shader socket")),
        }
        .map(|value| {
            if value.iter().all(|part| part.is_finite()) {
                value
            } else {
                default
            }
        })
    }
}

fn socket_default<'node>(node: &'node GraphNode, socket: &str) -> Option<&'node JsonValue> {
    let alias = match socket {
        "Base Color" => "Base Color",
        "Transmission Weight" => "Transmission",
        "Sheen Weight" => "Sheen",
        "Coat Weight" => "Clearcoat",
        "Emission Color" => "Emission",
        "Shader" => "Shader",
        _ => socket,
    };
    node.inputs.get(alias)
}

fn socket_matches(candidate: &str, requested: &str) -> bool {
    if candidate == requested {
        return true;
    }
    matches!(
        (candidate, requested),
        ("Shader", "Shader 1")
            | ("Shader 1", "Shader")
            | ("Shader_001", "Shader 2")
            | ("Shader 2", "Shader_001")
            | ("Transmission", "Transmission Weight")
            | ("Transmission Weight", "Transmission")
            | ("Subsurface", "Subsurface Weight")
            | ("Subsurface Weight", "Subsurface")
            | ("Clearcoat", "Coat Weight")
            | ("Coat Weight", "Clearcoat")
            | ("Sheen", "Sheen Weight")
            | ("Sheen Weight", "Sheen")
            | ("Emission", "Emission Color")
            | ("Emission Color", "Emission")
    )
}

/// The Image Texture nodes whose Alpha output reaches the group's output. Cycles decides this in two steps.
/// `ShaderGraph::clean` (`intern/cycles/scene/shader_graph.cpp`, ~lines 801-860) runs `break_cycles(output())`,
/// which visits every node reachable backwards from the output through all of its inputs, then deletes the input
/// links of unvisited nodes and the unvisited nodes. An Image Texture unassociates its colour only when its Alpha
/// output still has a link after that pass (`intern/cycles/scene/shader_nodes.cpp`, ~line 449). So a link from Alpha
/// into a dead node does not count. This function applies the same visit: reachability starts at `output` and
/// follows every incoming link, and the result keeps Alpha links whose target was visited.
fn linked_alpha_image_nodes<'graph>(
    group: &'graph NodeGroup,
    output: &'graph Id,
) -> BTreeSet<&'graph Id> {
    let reachable = reachable_nodes(group, output);
    group
        .links
        .iter()
        .filter(|link| link.from_socket == "Alpha" && reachable.contains(&link.to_node))
        .map(|link| &link.from_node)
        .filter(|from| {
            group
                .nodes
                .get(*from)
                .is_some_and(|node| node.node_type == "ShaderNodeTexImage")
        })
        .collect()
}

/// The nodes that feed `output`, `output` included. Blender's shader compile keeps only these (see the Alpha visit above).
fn reachable_nodes<'graph>(group: &'graph NodeGroup, output: &'graph Id) -> BTreeSet<&'graph Id> {
    let mut reachable: BTreeSet<&'graph Id> = BTreeSet::new();
    let mut pending = vec![output];
    while let Some(node) = pending.pop() {
        if reachable.insert(node) {
            pending.extend(
                group
                    .links
                    .iter()
                    .filter(|link| link.to_node == *node)
                    .map(|link| &link.from_node),
            );
        }
    }
    reachable
}

/// Whether EEVEE compiles the graph as transparent, `GPU_MATFLAG_TRANSPARENT`, which is set statically per node:
/// a Transparent BSDF whose Color is linked or not black (`node_shader_bsdf_transparent.cc:30`), and a Principled
/// whose Alpha is linked or not one (`node_shader_bsdf_principled.cc:400`). Only the nodes that feed the output count.
/// Specular and Ray Portal set the same flag in Blender, but Potter's evaluator rejects both node types, so a material
/// with one never evaluates and needs no rule.
fn graph_is_transparent(group: &NodeGroup, output: &Id) -> bool {
    let reachable = reachable_nodes(group, output);
    let linked = |node: &Id, socket: &str| {
        group
            .links
            .iter()
            .any(|link| link.to_node == *node && link.to_socket == socket)
    };
    group.nodes.iter().any(|(id, node)| {
        if !reachable.contains(id) {
            return false;
        }
        match node.node_type.as_str() {
            "ShaderNodeBsdfTransparent" => {
                linked(id, "Color")
                    || node
                        .inputs
                        .get("Color")
                        .is_none_or(|value| color(value, [1.0; 4])[..3].iter().any(|c| *c != 0.0))
            }
            "ShaderNodeBsdfPrincipled" => {
                linked(id, "Alpha")
                    || node.inputs.get("Alpha").is_some_and(|value| {
                        // Blender's `alpha != 1.0`. `number` never yields a NaN, and `total_cmp` is exact for 1.0.
                        number(value, 1.0).total_cmp(&1.0).is_ne()
                    })
            }
            _ => false,
        }
    })
}

fn number(value: &JsonValue, default: f64) -> f64 {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn vector(value: &JsonValue, default: DVec3) -> DVec3 {
    let Some(values) = value.as_array() else {
        return default;
    };
    if values.len() < 3 {
        return default;
    }
    let Some(x) = values[0].as_f64() else {
        return default;
    };
    let Some(y) = values[1].as_f64() else {
        return default;
    };
    let Some(z) = values[2].as_f64() else {
        return default;
    };
    let result = DVec3::new(x, y, z);
    if result.is_finite() { result } else { default }
}

fn color(value: &JsonValue, default: [f64; 4]) -> [f64; 4] {
    let Some(values) = value.as_array() else {
        return default;
    };
    if values.len() < 3 {
        return default;
    }
    let Some(red) = values[0].as_f64() else {
        return default;
    };
    let Some(green) = values[1].as_f64() else {
        return default;
    };
    let Some(blue) = values[2].as_f64() else {
        return default;
    };
    let alpha = values.get(3).and_then(JsonValue::as_f64).unwrap_or(1.0);
    let result = [red, green, blue, alpha];
    if result.iter().all(|part| part.is_finite()) {
        result
    } else {
        default
    }
}

fn ramp_element(value: &JsonValue) -> Result<(f64, [f64; 4])> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("ColorRamp elements must be objects"))?;
    let position = object
        .get("position")
        .or_else(|| object.get("pos"))
        .and_then(JsonValue::as_f64)
        .filter(|position| position.is_finite())
        .ok_or_else(|| invalid("ColorRamp element position must be finite"))?;
    let color_value = object
        .get("color")
        .ok_or_else(|| invalid("ColorRamp element requires a color"))?;
    let color = color(color_value, [f64::NAN; 4]);
    if color.iter().any(|component| !component.is_finite()) {
        return Err(invalid("ColorRamp element color must be finite RGBA"));
    }
    Ok((position, color))
}

fn color_lerp(first: [f64; 4], second: [f64; 4], factor: f64) -> [f64; 4] {
    let factor = bounded_factor(factor);
    std::array::from_fn(|index| first[index] + (second[index] - first[index]) * factor)
}

fn color_binary(first: [f64; 4], second: [f64; 4], f: impl Fn(f64, f64) -> f64) -> [f64; 4] {
    std::array::from_fn(|index| f(first[index], second[index]))
}

fn bounded_factor(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalize_or(value: DVec3, fallback: DVec3) -> DVec3 {
    if value.is_finite() && value.length_squared() > f64::EPSILON {
        value.normalize()
    } else {
        fallback
    }
}

fn tangent_basis(normal: DVec3) -> (DVec3, DVec3) {
    let tangent_u = if normal.z.abs() > 0.999 {
        DVec3::X
    } else {
        normalize_or(DVec3::Z.cross(normal), DVec3::X)
    };
    let tangent_v = normalize_or(normal.cross(tangent_u), DVec3::Y);
    (tangent_u, tangent_v)
}

/// Index of the layer that a reference names. `None` and the empty name select the render layer.
/// A name absent from the source member (including a layer contributed by another joined member)
/// falls back to the render layer and records `uv.layer_missing`.
fn layer_index(hit_ctx: &HitContext<'_>, layer: Option<&str>) -> Option<usize> {
    let render = hit_ctx.uv_layers.and_then(|layers| layers.render);
    let Some(name) = layer.filter(|name| !name.is_empty()) else {
        return render;
    };
    if let Some(index) = hit_ctx.uv_layers.and_then(|layers| layers.index_of(name))
        && hit_ctx
            .uv_layers
            .and_then(|layers| layers.available_mask)
            .is_none_or(|mask| index < 8 && mask & (1_u8 << index) != 0)
    {
        return Some(index);
    }
    if let Some(warnings) = hit_ctx.warnings {
        let warning = json!({
            "code": "uv.layer_missing",
            "message": format!("UV layer `{name}` is not on this mesh; the render layer was used"),
            "layer": name,
        });
        let mut recorded = warnings.borrow_mut();
        if !recorded.contains(&warning) {
            recorded.push(warning);
        }
    }
    render
}

/// The UV at the hit of layer `index`, or the render `uv` when there is no such layer.
fn layer_uv(hit_ctx: &HitContext<'_>, index: Option<usize>) -> [f64; 2] {
    index
        .and_then(|index| hit_ctx.uv_layers.and_then(|layers| layers.uv(index)))
        .unwrap_or(hit_ctx.uv)
}
/// Converts the canonical V-down coordinates to the V-up coordinates used by Blender shader
/// graph UV sources.
fn storage_to_blender_uv(uv: [f64; 2]) -> [f64; 2] {
    [uv[0], 1.0 - uv[1]]
}

/// The layer of a material texture reference, resolved by [`layer_index`].
fn texture_layer(hit_ctx: &HitContext<'_>, texture: &TextureRef) -> Option<usize> {
    layer_index(hit_ctx, texture.uv_layer.as_deref())
}

/// Tangent and bitangent of layer `index` at the hit, orthogonal to `normal`. The tangent follows
/// the layer's `dP/du` and the bitangent its `dP/dv`, so a mirrored layer gets the bitangent its
/// UVs define. Without a usable layer, the frame of `normal` is used.
fn layer_frame(hit_ctx: &HitContext<'_>, index: Option<usize>, normal: DVec3) -> (DVec3, DVec3) {
    layer_frame_with_v_direction(hit_ctx, index, normal, 1.0)
}

/// Blender shader graphs use V-up coordinates. Blender's Normal Map node follows the selected
/// layer's V derivative, so its bitangent is the inverse of the canonical V-down derivative.
fn blender_layer_frame(
    hit_ctx: &HitContext<'_>,
    index: Option<usize>,
    normal: DVec3,
) -> (DVec3, DVec3) {
    layer_frame_with_v_direction(hit_ctx, index, normal, -1.0)
}

fn layer_frame_with_v_direction(
    hit_ctx: &HitContext<'_>,
    index: Option<usize>,
    normal: DVec3,
    v_direction: f64,
) -> (DVec3, DVec3) {
    let fallback = tangent_basis(normal);
    let derivatives = index.and_then(|index| {
        hit_ctx
            .uv_layers
            .and_then(|layers| layers.derivatives(index))
    });
    let Some((du, dv)) = derivatives else {
        return fallback;
    };
    let tangent = normalize_or(du - normal * du.dot(normal), fallback.0);
    let dv = dv * v_direction;
    let bitangent = normalize_or(
        dv - normal * dv.dot(normal) - tangent * dv.dot(tangent),
        fallback.1,
    );
    (tangent, bitangent)
}

/// Finite-difference step across the surface, in render-UV units.
const HEIGHT_STEP: f64 = 1.0e-3;

/// One finite-difference step: the surface displacement, and the UV change it implies when the hit
/// has no UV layers.
#[derive(Clone, Copy)]
struct SurfaceStep {
    world: DVec3,
    uv: [f64; 2],
}

impl SurfaceStep {
    fn reversed(self) -> Self {
        Self {
            world: -self.world,
            uv: [-self.uv[0], -self.uv[1]],
        }
    }
}

/// The step that moves the render UV by `HEIGHT_STEP` along `axis` (0 for U, 1 for V). It follows
/// the render layer's surface derivative, projected into the tangent plane of `normal`. Without one
/// it follows the tangent frame of `normal`.
fn surface_step(hit_ctx: &HitContext<'_>, normal: DVec3, axis: usize) -> SurfaceStep {
    let (tangent, bitangent) = tangent_basis(normal);
    let derivatives = hit_ctx
        .uv_layers
        .and_then(|layers| layers.render.and_then(|index| layers.derivatives(index)));
    let (direction, uv) = match (axis, derivatives) {
        (0, Some((du, _))) => (du, [HEIGHT_STEP, 0.0]),
        (_, Some((_, dv))) => (dv, [0.0, HEIGHT_STEP]),
        (0, None) => (tangent, [HEIGHT_STEP, 0.0]),
        (_, None) => (bitangent, [0.0, HEIGHT_STEP]),
    };
    let in_plane = direction - normal * direction.dot(normal);
    SurfaceStep {
        world: in_plane * HEIGHT_STEP,
        uv,
    }
}

/// Gradient across the surface at the hit of the height field that `height` samples. A surface
/// displaced by the field has the normal `normal - gradient`, before it is normalised.
///
/// Bump nodes and displacement both use it. The steps follow the render layer's surface
/// derivatives, so the render UV moves by `HEIGHT_STEP` per step and every other layer moves by its
/// own derivative. A degenerate render UV leaves UV-driven heights flat, while procedural heights
/// still vary along the tangent frame.
///
/// # Errors
///
/// Returns the error of `height` at any sample.
pub fn height_gradient<'a>(
    context: &HitContext<'a>,
    normal: DVec3,
    mut height: impl FnMut(&HitContext<'a>) -> Result<f64>,
) -> Result<DVec3> {
    let step_u = surface_step(context, normal, 0);
    let step_v = surface_step(context, normal, 1);
    let slope_u =
        (height(&context.shifted(step_u))? - height(&context.shifted(step_u.reversed()))?) / 2.0;
    let slope_v =
        (height(&context.shifted(step_v))? - height(&context.shifted(step_v.reversed()))?) / 2.0;
    // Solve for the gradient in the span of the two steps: `gradient · step = slope` for both.
    let (world_u, world_v) = (step_u.world, step_v.world);
    let (a, b, c) = (
        world_u.dot(world_u),
        world_u.dot(world_v),
        world_v.dot(world_v),
    );
    let determinant = a * c - b * b;
    if !determinant.is_finite() || determinant <= f64::EPSILON * a * c {
        return Ok(DVec3::ZERO);
    }
    let along_u = (c * slope_u - b * slope_v) / determinant;
    let along_v = (a * slope_v - b * slope_u) / determinant;
    Ok(world_u * along_u + world_v * along_v)
}

fn perlin(point: DVec3) -> f64 {
    let base = point.floor();
    let local = point - base;
    let fade = |value: f64| value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
    let fade = DVec3::new(fade(local.x), fade(local.y), fade(local.z));
    let mut total = 0.0;
    for z in 0..=1 {
        for y in 0..=1 {
            for x in 0..=1 {
                let lattice = base + DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                let delta = local - DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                let angle = lattice_random(lattice, 0) * std::f64::consts::TAU;
                let vertical = lattice_random(lattice, 1) * 2.0 - 1.0;
                let horizontal = (1.0 - vertical * vertical).max(0.0).sqrt();
                let gradient =
                    DVec3::new(horizontal * angle.cos(), horizontal * angle.sin(), vertical);
                let wx = if x == 0 { 1.0 - fade.x } else { fade.x };
                let wy = if y == 0 { 1.0 - fade.y } else { fade.y };
                let wz = if z == 0 { 1.0 - fade.z } else { fade.z };
                total += gradient.dot(delta) * wx * wy * wz;
            }
        }
    }
    total * 1.6
}

fn lattice_random(point: DVec3, channel: i32) -> f64 {
    let mut hash = 0x811c_9dc5_u32 ^ channel as u32;
    for value in [
        point.x.floor() as i64,
        point.y.floor() as i64,
        point.z.floor() as i64,
    ] {
        hash = (hash ^ value as u32).wrapping_mul(0x0100_0193);
        hash ^= (value >> 32) as u32;
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb_352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846c_a68b);
    hash ^= hash >> 16;
    f64::from(hash) / f64::from(u32::MAX)
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::InvalidOperation, message)
}

fn eval_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, message)
}

/// Color of one named layer, or the render layer when `use_render` is true. Blender's two color
/// nodes use different alpha defaults when a requested layer is unavailable.
fn layer_color(
    hit_ctx: &HitContext<'_>,
    name: &str,
    use_render: bool,
    missing: [f64; 4],
) -> [f64; 4] {
    let found = if use_render {
        hit_ctx.color_render
    } else {
        hit_ctx
            .color_layers
            .and_then(|layers| layers.iter().find(|entry| entry.0 == name))
            .map(|entry| entry.1)
    };
    if let Some(color) = found {
        return color;
    }
    if let Some(warnings) = hit_ctx.warnings {
        let label = if use_render { "render" } else { name };
        let fallback = if missing[3] == 0.0 {
            "transparent black"
        } else {
            "opaque black"
        };
        let warning = json!({
            "code": "mesh.color_layer_missing",
            "message": format!("color layer `{label}` is not on this mesh; {fallback} was used"),
            "layer": name,
        });
        let mut recorded = warnings.borrow_mut();
        if !recorded.contains(&warning) {
            recorded.push(warning);
        }
    }
    missing
}

/// The `Color` or `Alpha` output of a node that reads one color.
fn color_output(color: [f64; 4], socket: &str, node: &str) -> Result<Value> {
    match socket {
        "Color" => Ok(Value::Color(color)),
        "Alpha" => Ok(Value::Scalar(color[3])),
        _ => Err(invalid(format!("unknown {node} output `{socket}`"))),
    }
}

fn unsupported_node(id: &Id, kind: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("shader node behavior `{kind}` is not implemented"),
        json!({"feature_id":format!("shader.node.{kind}"),"node_id":id}),
    )
}

fn unsupported_node_id(node: &str, output: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("shader output `{output}` of `{node}` is not implemented"),
        json!({"feature_id":format!("shader.node.{node}.{output}")}),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use std::{cell::RefCell, collections::BTreeMap};

    use glam::DVec3;
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use crate::{
        graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
        image::{ImageData, ImageExtension, ImageInterpolation},
        model::{Id, Material, Registry, TextureRef},
    };

    use super::{
        BsdfParams, HitContext, HitUvLayers, PreparedMaterial, color_lerp, evaluate_displacement,
        evaluate_surface, layer_index, layer_uv,
    };

    fn set_properties(group: &mut NodeGroup, id: &str, properties: Value) {
        let Value::Object(properties) = properties else {
            unreachable!("shader test properties must be an object")
        };
        group
            .nodes
            .get_mut(&Id::new(id.to_owned()).unwrap())
            .unwrap()
            .properties = properties;
    }

    fn mapped_image_color(
        vector_type: &str,
        location: [f64; 3],
        rotation: [f64; 3],
        scale: [f64; 3],
        extension: Option<&str>,
    ) -> [f64; 4] {
        let mut graph = NodeGroup::new("Mapping", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "principled",
            "ShaderNodeBsdfPrincipled",
            json!({"Base Color":[1.0,1.0,1.0,1.0]}),
        );
        insert_node(&mut graph, "uv", "ShaderNodeUVMap", json!({}));
        insert_node(
            &mut graph,
            "mapping",
            "ShaderNodeMapping",
            json!({"Location":location,"Rotation":rotation,"Scale":scale}),
        );
        insert_node(&mut graph, "image", "ShaderNodeTexImage", json!({}));
        set_properties(&mut graph, "uv", json!({"uv_map":"UVMap"}));
        set_properties(&mut graph, "mapping", json!({"vector_type":vector_type}));
        let mut image_properties = json!({"image":"image","interpolation":"CLOSEST"});
        if let Some(extension) = extension {
            image_properties["extension"] = json!(extension);
        }
        set_properties(&mut graph, "image", image_properties);
        link(&mut graph, "uv", "UV", "mapping", "Vector");
        link(&mut graph, "mapping", "Vector", "image", "Vector");
        link(&mut graph, "image", "Color", "principled", "Base Color");
        link(&mut graph, "principled", "BSDF", "output", "Surface");

        let graph_id = Id::new("mapping_shader".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            base_color: [1.0; 4],
            ..Material::default()
        };
        let names = vec!["UVMap".to_owned()];
        let corners = vec![[[0.2, 0.3]; 3]];
        let mut context = HitContext::new(&groups).with_uv_layers(HitUvLayers {
            names: &names,
            corners: &corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric: [1.0 / 3.0; 3],
            render: Some(0),
            available_mask: None,
        });
        let mut pixels = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                let blender_y = 7 - y;
                pixels.push([
                    f64::from(x) / 7.0,
                    f64::from(blender_y) / 7.0,
                    f64::from(x + 2 * blender_y) / 21.0,
                    1.0,
                ]);
            }
        }
        let images = BTreeMap::from([(
            "image".to_owned(),
            ImageData {
                colour: crate::image::ColourCache::default(),
                storage: crate::image::ImageStorage::default(),
                width: 8,
                height: 8,
                pixels,
                tiles: BTreeMap::new(),
                interpolation: ImageInterpolation::Closest,
                srgb: false,
                udim: false,
            },
        )]);
        context = context.with_images(&images);
        evaluate_surface(&material, &context).unwrap().base_color
    }

    /// Red at storage U `u` of a plain field (no graph) whose texture uses `extension`. Pixels are 8 columns of
    /// increasing red, so column 7 is the edge texel with red 1.0.
    fn field_red_at(extension: ImageExtension, u: f64) -> f64 {
        let material = Material {
            base_color: [1.0; 4],
            base_color_texture: Some(TextureRef {
                image: Id::new("image".to_owned()).unwrap(),
                uv_layer: None,
                interpolation: ImageInterpolation::Closest,
                extension,
            }),
            ..Material::default()
        };
        let names = vec!["UVMap".to_owned()];
        let corners = vec![[[u, 0.5]; 3]];
        let groups = Registry::new();
        let context = HitContext::new(&groups).with_uv_layers(HitUvLayers {
            names: &names,
            corners: &corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric: [1.0 / 3.0; 3],
            render: Some(0),
            available_mask: None,
        });
        let mut pixels = Vec::with_capacity(64);
        for _ in 0..8 {
            for x in 0..8 {
                pixels.push([f64::from(x) / 7.0, 0.0, 0.0, 1.0]);
            }
        }
        let images = BTreeMap::from([(
            "image".to_owned(),
            ImageData {
                colour: crate::image::ColourCache::default(),
                storage: crate::image::ImageStorage::default(),
                width: 8,
                height: 8,
                pixels,
                tiles: BTreeMap::new(),
                interpolation: ImageInterpolation::Closest,
                srgb: false,
                udim: false,
            },
        )]);
        let context = context.with_images(&images);
        evaluate_surface(&material, &context).unwrap().base_color[0]
    }

    #[test]
    fn material_field_extension_sets_the_sample_past_its_edge() {
        // Blender's Image Texture: extend holds the edge texel past the UV edge, repeat wraps around.
        let edge = field_red_at(ImageExtension::Extend, 0.99);
        assert!((edge - 1.0).abs() < 1e-12, "the edge texel is column 7");
        assert!(
            (field_red_at(ImageExtension::Extend, 1.25) - edge).abs() < 1e-12,
            "extend past the edge equals the edge texel"
        );
        let wrapped = field_red_at(ImageExtension::Repeat, 1.25);
        assert!(
            (wrapped - 2.0 / 7.0).abs() < 1e-12,
            "repeat wraps 1.25 to column 2, got {wrapped}"
        );
    }

    fn insert_node(group: &mut NodeGroup, id: &str, node_type: &str, inputs: Value) {
        let mut node = GraphNode::new(node_type);
        let Value::Object(inputs) = inputs else {
            unreachable!("shader test inputs must be an object")
        };
        node.inputs = inputs.into_iter().collect::<BTreeMap<_, _>>();
        group.nodes.insert(Id::new(id.to_owned()).unwrap(), node);
    }

    fn link(group: &mut NodeGroup, from: &str, output: &str, to: &str, input: &str) {
        group.links.push(GraphLink {
            from_node: Id::new(from.to_owned()).unwrap(),
            from_socket: output.to_owned(),
            to_node: Id::new(to.to_owned()).unwrap(),
            to_socket: input.to_owned(),
        });
    }

    /// A graph whose Principled BSDF feeds the output with the given inputs.
    fn principled_output(principled_inputs: Value) -> NodeGroup {
        let mut graph = NodeGroup::new("Principled", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "principled",
            "ShaderNodeBsdfPrincipled",
            principled_inputs,
        );
        link(&mut graph, "principled", "BSDF", "output", "Surface");
        graph
    }

    fn is_transparent(graph: NodeGroup) -> bool {
        let id = Id::new("transparency_graph").unwrap();
        let mut groups = Registry::new();
        groups.insert(id.clone(), graph);
        let material = Material {
            node_tree: Some(id),
            ..Material::default()
        };
        PreparedMaterial::new(&material, &groups).is_transparent()
    }

    #[test]
    fn a_principled_is_transparent_only_when_its_alpha_is_linked_or_below_one() {
        assert!(!is_transparent(principled_output(json!({}))));
        assert!(!is_transparent(principled_output(json!({"Alpha": 1.0}))));
        assert!(is_transparent(principled_output(json!({"Alpha": 0.5}))));

        let mut linked = principled_output(json!({}));
        insert_node(&mut linked, "texture", "ShaderNodeTexImage", json!({}));
        link(&mut linked, "texture", "Alpha", "principled", "Alpha");
        assert!(is_transparent(linked));
    }

    #[test]
    fn a_transparent_bsdf_is_transparent_unless_its_color_is_black() {
        let transparent = |color: Value| {
            let mut graph = NodeGroup::new("Transparent", GraphKind::Shader);
            insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
            insert_node(
                &mut graph,
                "transparent",
                "ShaderNodeBsdfTransparent",
                color,
            );
            link(&mut graph, "transparent", "BSDF", "output", "Surface");
            graph
        };
        assert!(is_transparent(transparent(json!({}))));
        assert!(is_transparent(transparent(
            json!({"Color": [0.9, 0.4, 0.2, 1.0]})
        )));
        assert!(!is_transparent(transparent(
            json!({"Color": [0.0, 0.0, 0.0, 1.0]})
        )));
    }

    #[test]
    fn only_a_node_that_feeds_the_output_makes_a_graph_transparent() {
        // A Transparent BSDF behind a Mix that reaches the output is transparent.
        let mut mixed = NodeGroup::new("Mix", GraphKind::Shader);
        insert_node(&mut mixed, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut mixed,
            "principled",
            "ShaderNodeBsdfPrincipled",
            json!({}),
        );
        insert_node(
            &mut mixed,
            "transparent",
            "ShaderNodeBsdfTransparent",
            json!({}),
        );
        insert_node(
            &mut mixed,
            "mix",
            "ShaderNodeMixShader",
            json!({"Fac": 0.5}),
        );
        link(&mut mixed, "principled", "BSDF", "mix", "Shader");
        link(&mut mixed, "transparent", "BSDF", "mix", "Shader_001");
        link(&mut mixed, "mix", "Shader", "output", "Surface");
        assert!(is_transparent(mixed));

        // The same Transparent BSDF, left unconnected, feeds nothing and does not make the graph transparent.
        let mut dangling = principled_output(json!({}));
        insert_node(
            &mut dangling,
            "transparent",
            "ShaderNodeBsdfTransparent",
            json!({}),
        );
        assert!(!is_transparent(dangling));
    }

    #[test]
    fn unavailable_joined_uv_layer_warns_and_falls_back_to_member_render() {
        use std::cell::RefCell;

        let groups = Registry::new();
        let names = vec!["JoinedOnly".to_owned(), "MemberRender".to_owned()];
        let corners = vec![[[0.8, 0.2]; 3], [[0.25, 0.75]; 3]];
        let warnings = RefCell::new(Vec::new());
        let context = HitContext::new(&groups)
            .with_uv_layers(HitUvLayers {
                names: &names,
                corners: &corners,
                positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
                barycentric: [1.0 / 3.0; 3],
                render: Some(1),
                available_mask: Some(0b10),
            })
            .with_warnings(&warnings);

        let selected = layer_index(&context, Some("JoinedOnly"));
        assert_eq!(selected, Some(1));
        assert_eq!(layer_uv(&context, selected), [0.25, 0.75]);
        assert_eq!(layer_index(&context, None), Some(1));
        assert_eq!(warnings.borrow().len(), 1);
        assert_eq!(warnings.borrow()[0]["code"], "uv.layer_missing");

        let ordinary = HitContext::new(&groups).with_uv_layers(HitUvLayers {
            names: &names,
            corners: &corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric: [1.0 / 3.0; 3],
            render: Some(1),
            available_mask: None,
        });
        assert_eq!(layer_index(&ordinary, Some("JoinedOnly")), Some(0));
    }

    #[test]
    fn material_volume_output_evaluates_principled_density_color_and_anisotropy() {
        let mut graph = NodeGroup::new("Volume", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "volume",
            "ShaderNodeVolumePrincipled",
            json!({"Density":0.7,"Color":[0.2,0.4,0.6,1.0],"Anisotropy":0.35}),
        );
        link(&mut graph, "volume", "Volume", "output", "Volume");
        let graph_id = Id::new("volume_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let params = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
        // Principled Volume at Density 0.7: σs = Color · 0.7 and σa = (1 - Color) · 0.7.
        assert!(channels_close(params.volume_scatter, [0.14, 0.28, 0.42]));
        assert!(channels_close(params.volume_absorption, [0.56, 0.42, 0.28]));
        assert_eq!(params.volume_anisotropy, 0.35);
    }

    fn channels_close(actual: [f64; 3], expected: [f64; 3]) -> bool {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12)
    }

    /// Two Volume Absorption closures (σa = (1 - Color) · Density: (0.8, 0.5, 0.2) and (0.25, 0.25, 0.25))
    /// joined by `combiner`; returns the evaluated volume coefficients.
    fn two_absorption_closures(combiner: &str, inputs: Value) -> BsdfParams {
        let mut graph = NodeGroup::new("Volume", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "first",
            "ShaderNodeVolumeAbsorption",
            json!({"Color":[0.2,0.5,0.8,1.0],"Density":1.0}),
        );
        insert_node(
            &mut graph,
            "second",
            "ShaderNodeVolumeAbsorption",
            json!({"Color":[0.5,0.5,0.5,1.0],"Density":0.5}),
        );
        insert_node(&mut graph, "combine", combiner, inputs);
        link(&mut graph, "first", "Volume", "combine", "Shader");
        link(&mut graph, "second", "Volume", "combine", "Shader_001");
        link(&mut graph, "combine", "Shader", "output", "Volume");
        let graph_id = Id::new("volume_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        evaluate_surface(&material, &HitContext::new(&groups)).unwrap()
    }

    #[test]
    fn add_and_mix_shader_combine_each_closures_own_absorption() {
        // Cycles adds closures' extinctions (Add Shader) and weights each by its mix factor (Mix
        // Shader), so the coefficients sum or mix per closure, not as (Σ weight)(Σ density).
        let added = two_absorption_closures("ShaderNodeAddShader", json!({}));
        assert!(channels_close(added.volume_absorption, [1.05, 0.75, 0.45]));
        let mixed = two_absorption_closures("ShaderNodeMixShader", json!({"Fac":0.25}));
        assert!(channels_close(
            mixed.volume_absorption,
            [0.6625, 0.4375, 0.2125]
        ));
    }

    #[test]
    fn unevaluated_principled_inputs_warn_once_per_material_and_input() {
        use std::cell::RefCell;

        let sink = RefCell::new(Vec::new());
        let mut groups = Registry::new();
        // A native material sets rough transmission on the material itself.
        let native = Material {
            name: "Native".to_owned(),
            transmission: 0.5,
            roughness: 0.3,
            ..Material::default()
        };
        let context = HitContext::new(&groups).with_warnings(&sink);
        evaluate_surface(&native, &context).unwrap();
        evaluate_surface(&native, &context).unwrap();

        // A graph material sets subsurface and rough transmission on its Principled node. Coat is
        // evaluated, so it must not warn.
        let mut graph = NodeGroup::new("Graph", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "principled",
            "ShaderNodeBsdfPrincipled",
            json!({"Subsurface Weight":0.5,"Transmission Weight":0.5,"Roughness":0.3}),
        );
        link(&mut graph, "principled", "BSDF", "output", "Surface");
        let graph_id = Id::new("graph_shader".to_owned()).unwrap();
        groups.insert(graph_id.clone(), graph);
        let graph_material = Material {
            name: "Graph".to_owned(),
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let graph_context = HitContext::new(&groups).with_warnings(&sink);
        evaluate_surface(&graph_material, &graph_context).unwrap();
        evaluate_surface(&graph_material, &graph_context).unwrap();

        let warnings = sink.borrow();
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        let count = |material: &str, input: &str| {
            warnings
                .iter()
                .filter(|warning| {
                    warning["material"].as_str() == Some(material)
                        && warning["input"].as_str() == Some(input)
                })
                .count()
        };
        assert_eq!(count("Native", "Transmission Weight"), 1);
        assert_eq!(count("Graph", "Subsurface Weight"), 1);
        assert_eq!(count("Graph", "Coat Weight"), 0);
        assert_eq!(count("Graph", "Transmission Weight"), 1);
        let native_warning = warnings.iter().find(|w| w["material"] == "Native").unwrap();
        assert!(native_warning.get("node_id").is_none());
        let subsurface_warning = warnings
            .iter()
            .find(|w| w["input"] == "Subsurface Weight")
            .unwrap();
        assert_eq!(subsurface_warning["node_id"], "principled");
    }

    #[test]
    fn add_of_two_coats_adds_weight_times_tint_and_averages_the_rest() {
        let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let mut first = BsdfParams::from_material(&Material::default(), DVec3::Z);
        first.coat_weight = 0.5;
        first.coat_tint = [1.0, 0.5, 0.25];
        first.coat_roughness = 0.1;
        let mut second = BsdfParams::from_material(&Material::default(), DVec3::Z);
        second.coat_weight = 0.2;
        second.coat_tint = [0.2, 0.2, 0.2];
        second.coat_roughness = 0.3;
        // weight · tint adds: 0.5·[1, 0.5, 0.25] + 0.2·[0.2, 0.2, 0.2], over the summed weight 0.7.
        let sum = BsdfParams::add(first, second);
        assert!((sum.coat_weight - 0.7).abs() < 1e-12);
        assert!(close(sum.coat_tint, [0.54 / 0.7, 0.29 / 0.7, 0.165 / 0.7]));
        assert!((sum.coat_roughness - (0.05 + 0.06) / 0.7).abs() < 1e-12);
        // Unlinked on both sides, the coat keeps following the shading normal.
        assert_eq!(sum.coat_normal, None);
        // A linked normal on one side enters the average with that side's weight.
        second.coat_normal = Some(DVec3::X);
        let sum = BsdfParams::add(first, second);
        let expected = (DVec3::Z * 0.5 + DVec3::X * 0.2).normalize();
        assert!((sum.coat_normal.unwrap_or(DVec3::ZERO) - expected).length() < 1e-12);
    }

    #[test]
    fn displacement_output_evaluates_scale_midlevel_and_surface_normal() {
        let mut graph = NodeGroup::new("Displacement", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "displacement",
            "ShaderNodeDisplacement",
            json!({"Height":0.75,"Midlevel":0.5,"Scale":2.0}),
        );
        link(
            &mut graph,
            "displacement",
            "Displacement",
            "output",
            "Displacement",
        );
        let graph_id = Id::new("displacement_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let mut context = HitContext::new(&groups);
        context.normal = DVec3::Z;
        let params = evaluate_surface(&material, &context).unwrap();
        assert!((params.displacement_height - 0.5).abs() < f64::EPSILON);
        assert!((evaluate_displacement(&material, &context).unwrap() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn vertex_color_reads_the_interpolated_corner_color_by_layer_name() {
        // Corner colors of one triangle, blended with barycentric weights 0.2, 0.3, 0.5.
        let weights = [0.2, 0.3, 0.5];
        let corners = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.5],
        ];
        let blended: [f64; 4] = std::array::from_fn(|channel| {
            corners
                .iter()
                .zip(weights)
                .map(|(corner, weight)| corner[channel] * weight)
                .sum()
        });
        let mut graph = NodeGroup::new("VertexColor", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(&mut graph, "bsdf", "ShaderNodeBsdfPrincipled", json!({}));
        insert_node(&mut graph, "vertex", "ShaderNodeVertexColor", json!({}));
        let vertex = Id::new("vertex".to_owned()).unwrap();
        graph
            .nodes
            .get_mut(&vertex)
            .unwrap()
            .properties
            .insert("layer_name".to_owned(), json!("Col"));
        link(&mut graph, "vertex", "Color", "bsdf", "Base Color");
        link(&mut graph, "bsdf", "BSDF", "output", "Surface");
        let graph_id = Id::new("vertex_color_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };

        let layers = [("Col", blended)];
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&layers, None),
        )
        .unwrap();
        for (actual, expected) in params.base_color.iter().zip([0.2, 0.3, 0.5, 0.75]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "base color {:?}",
                params.base_color
            );
        }

        // An explicitly missing Vertex Color layer is transparent black and records a warning.
        let sink = RefCell::new(Vec::new());
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups)
                .with_color_layers(&[], Some([0.91, 0.37, 0.64, 0.22]))
                .with_warnings(&sink),
        )
        .unwrap();
        assert_eq!(params.base_color, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(sink.borrow()[0]["code"], json!("mesh.color_layer_missing"));
        assert_eq!(sink.borrow()[0]["layer"], json!("Col"));
    }

    #[test]
    fn default_vertex_color_reads_render_and_missing_render_is_transparent() {
        let mut graph = NodeGroup::new("DefaultVertexColor", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(&mut graph, "bsdf", "ShaderNodeBsdfPrincipled", json!({}));
        insert_node(&mut graph, "vertex", "ShaderNodeVertexColor", json!({}));
        link(&mut graph, "vertex", "Color", "bsdf", "Base Color");
        link(&mut graph, "bsdf", "BSDF", "output", "Surface");
        let graph_id = Id::new("default_vertex_color_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let render = [0.2, 0.4, 0.8, 0.35];
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&[], Some(render)),
        )
        .unwrap();
        assert_eq!(params.base_color, render);

        let sink = RefCell::new(Vec::new());
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups)
                .with_color_layers(&[], None)
                .with_warnings(&sink),
        )
        .unwrap();
        assert_eq!(params.base_color, [0.0; 4]);
        assert_eq!(sink.borrow()[0]["layer"], json!(""));
    }

    #[test]
    fn missing_attribute_color_is_opaque_black() {
        let mut graph = NodeGroup::new("MissingAttributeColor", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(&mut graph, "bsdf", "ShaderNodeBsdfPrincipled", json!({}));
        insert_node(&mut graph, "attribute", "ShaderNodeAttribute", json!({}));
        set_properties(
            &mut graph,
            "attribute",
            json!({"attribute_type":"GEOMETRY","attribute_name":"Ghost"}),
        );
        link(&mut graph, "attribute", "Color", "bsdf", "Base Color");
        link(&mut graph, "bsdf", "BSDF", "output", "Surface");
        let graph_id = Id::new("missing_attribute_color_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let source = [0.23, 0.71, 0.42, 0.36];
        let found = [("Ghost", source)];
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&found, None),
        )
        .unwrap();
        assert_eq!(params.base_color, [source[0], source[1], source[2], 1.0]);

        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&[], Some([0.87, 0.54, 0.19, 0.27])),
        )
        .unwrap();
        assert_eq!(params.base_color, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn vertex_color_multiplies_the_base_color_through_a_mix_node() {
        let mut graph = NodeGroup::new("VertexMix", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(&mut graph, "bsdf", "ShaderNodeBsdfPrincipled", json!({}));
        insert_node(
            &mut graph,
            "mix",
            "ShaderNodeMix",
            json!({"A_Color":[0.5,0.25,1.0,1.0],"Factor_Float":1.0}),
        );
        insert_node(&mut graph, "vertex", "ShaderNodeVertexColor", json!({}));
        let mix = Id::new("mix".to_owned()).unwrap();
        let mix_node = graph.nodes.get_mut(&mix).unwrap();
        mix_node
            .properties
            .insert("blend_type".to_owned(), json!("MULTIPLY"));
        mix_node
            .properties
            .insert("data_type".to_owned(), json!("RGBA"));
        let vertex = Id::new("vertex".to_owned()).unwrap();
        graph
            .nodes
            .get_mut(&vertex)
            .unwrap()
            .properties
            .insert("layer_name".to_owned(), json!("Col"));
        link(&mut graph, "vertex", "Color", "mix", "B_Color");
        link(&mut graph, "mix", "Result_Color", "bsdf", "Base Color");
        link(&mut graph, "bsdf", "BSDF", "output", "Surface");
        let graph_id = Id::new("vertex_mix_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };

        let layers = [("Col", [0.2, 0.6, 0.4, 1.0])];
        let params = evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&layers, None),
        )
        .unwrap();
        // Multiply: each channel of A times the interpolated vertex color B, at factor 1.
        for (actual, expected) in params.base_color.iter().zip([0.1, 0.15, 0.4, 1.0]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "base color {:?}",
                params.base_color
            );
        }
    }

    #[test]
    fn separate_color_refuses_a_non_rgb_mode() {
        let mut graph = NodeGroup::new("Separate", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "displacement",
            "ShaderNodeDisplacement",
            json!({"Height":0.75,"Midlevel":0.5,"Scale":2.0}),
        );
        insert_node(
            &mut graph,
            "split",
            "ShaderNodeSeparateColor",
            json!({"Color":[0.1,0.2,0.3,1.0]}),
        );
        let split = Id::new("split".to_owned()).unwrap();
        graph
            .nodes
            .get_mut(&split)
            .unwrap()
            .properties
            .insert("mode".to_owned(), json!("HSV"));
        link(&mut graph, "split", "Red", "displacement", "Height");
        link(
            &mut graph,
            "displacement",
            "Displacement",
            "output",
            "Displacement",
        );
        let graph_id = Id::new("separate_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let mut context = HitContext::new(&groups);
        context.normal = DVec3::Z;
        let error = evaluate_displacement(&material, &context).unwrap_err();
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
    }

    #[test]
    fn mapping_vector_types_match_blender_sampled_texels_in_v_up_space() {
        let cases = [
            (
                "POINT",
                [0.1, 0.2, 0.0],
                [0.0, 0.0, std::f64::consts::FRAC_PI_2],
                [1.0, 1.0, 1.0],
                [3.0 / 7.0, 3.0 / 7.0, 9.0 / 21.0, 1.0],
            ),
            (
                "TEXTURE",
                [0.1, 0.2, 0.0],
                [0.0, 0.0, std::f64::consts::PI / 6.0],
                [2.0, 1.0, 1.0],
                [1.0 / 7.0, 3.0 / 7.0, 7.0 / 21.0, 1.0],
            ),
            (
                "VECTOR",
                [0.1, 0.2, 0.0],
                [0.0, 0.0, std::f64::consts::PI / 6.0],
                [2.0, 1.0, 1.0],
                [1.0, 6.0 / 7.0, 19.0 / 21.0, 1.0],
            ),
            (
                "NORMAL",
                [0.1, 0.2, 0.0],
                [0.0, 0.0, std::f64::consts::PI / 6.0],
                [2.0, 1.0, 1.0],
                [5.0 / 7.0, 1.0, 19.0 / 21.0, 1.0],
            ),
        ];
        for (vector_type, location, rotation, scale, expected) in cases {
            let actual = mapped_image_color(vector_type, location, rotation, scale, Some("REPEAT"));
            for (component, reference) in actual.into_iter().zip(expected) {
                assert!(
                    (component - reference).abs() < 1.0e-12,
                    "{vector_type}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    #[test]
    fn image_texture_extensions_match_blender_edges_and_default_repeat() {
        let cases = [
            (None, [1.0 / 7.0, 5.0 / 7.0, 11.0 / 21.0, 1.0]),
            (Some("REPEAT"), [1.0 / 7.0, 5.0 / 7.0, 11.0 / 21.0, 1.0]),
            (Some("EXTEND"), [1.0 / 7.0, 1.0, 15.0 / 21.0, 1.0]),
            (Some("CLIP"), [0.0; 4]),
            (Some("MIRROR"), [1.0 / 7.0, 2.0 / 7.0, 5.0 / 21.0, 1.0]),
        ];
        for (extension, expected) in cases {
            let actual =
                mapped_image_color("POINT", [0.0, 1.0, 0.0], [0.0; 3], [1.0; 3], extension);
            for (component, reference) in actual.into_iter().zip(expected) {
                assert!(
                    (component - reference).abs() < 1.0e-12,
                    "extension {extension:?}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    /// The Roughness a Math node drives on a material, for `Value`, `Value_001` and `Value_002`.
    fn math_roughness(operation: &str, value: f64, second: f64, third: f64) -> f64 {
        let mut graph = NodeGroup::new("Math", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "principled",
            "ShaderNodeBsdfPrincipled",
            json!({}),
        );
        insert_node(
            &mut graph,
            "math",
            "ShaderNodeMath",
            json!({"Value": value, "Value_001": second, "Value_002": third}),
        );
        set_properties(&mut graph, "math", json!({"operation": operation}));
        link(&mut graph, "math", "Value", "principled", "Roughness");
        link(&mut graph, "principled", "BSDF", "output", "Surface");
        let graph_id = Id::new("math_shader".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        evaluate_surface(&material, &HitContext::new(&groups))
            .unwrap()
            .roughness
    }

    #[test]
    fn math_wrap_and_pingpong_match_blenders_wrapf_and_pingpongf() {
        // Blender's `wrapf(value, max, min)` takes the second input as the maximum and the third as the minimum, and
        // `pingpongf(value, scale)` is the triangle wave of the signed scale (`math_base_inline.cc`).
        let cases = [
            // The importer's REPEAT wrap (max 0, min 1) maps the negative and the overflowing input into [0, 1].
            ("WRAP", -0.3, 0.0, 1.0, 0.7),
            ("WRAP", 1.3, 0.0, 1.0, 0.3),
            ("WRAP", -0.3, 1.0, 0.0, 0.7),
            ("WRAP", 0.4, 1.0, 0.0, 0.4),
            // A zero range returns the minimum.
            ("WRAP", 0.4, 0.0, 0.0, 0.0),
            ("PINGPONG", -0.4, 1.0, 0.0, 0.4),
            ("PINGPONG", 2.5, 1.0, 0.0, 0.5),
            // The signed scale: a negative scale gives the same wave.
            ("PINGPONG", -0.4, -1.0, 0.0, 0.4),
            // A zero scale returns zero.
            ("PINGPONG", 0.5, 0.0, 0.0, 0.0),
        ];
        for (operation, value, second, third, expected) in cases {
            let actual = math_roughness(operation, value, second, third);
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{operation}({value}, {second}, {third}): expected {expected}, got {actual}"
            );
        }
    }

    proptest! {
        #[test]
        fn color_mix_matches_linear_reference(factor in 0.0_f64..=1.0) {
            let first = [0.2, 0.4, 0.6, 0.8];
            let second = [0.8, 0.6, 0.4, 0.2];
            let actual = color_lerp(first, second, factor);
            for (index, value) in actual.into_iter().enumerate() {
                let reference = first[index] + (second[index] - first[index]) * factor;
                prop_assert!((value - reference).abs() <= f64::EPSILON);
            }
        }

        #[test]
        fn color_mix_clamps_factor(factor in -1000.0_f64..=1000.0) {
            let value = color_lerp([0.0; 4], [1.0; 4], factor);
            prop_assert!(value.iter().all(|component| (0.0..=1.0).contains(component)));
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn kani_bounded_mix_factor_stays_unit_interval() {
    let factor: f64 = kani::any();
    kani::assume(factor.is_finite());
    let bounded = bounded_factor(factor);
    assert!(bounded >= 0.0 && bounded <= 1.0);
}
