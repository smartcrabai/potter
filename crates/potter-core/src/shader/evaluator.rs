use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use glam::{DMat4, DQuat, DVec3, EulerRot, Mat4, Vec3};
use serde_json::{Value as JsonValue, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{GeneratedCorners, TextureSpace},
    graph::{GraphKind, GraphNode, NodeGroup},
    image::{ImageData, ImageExtension, ImageInterpolation, sample_with_extension_alpha},
    model::{Id, Material, Registry, TextureRef, ThicknessMode},
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

/// Cycles' Principled glass tint, `sqrt(clamped base colour)` (`kernel/svm/closure.h`, `clamped_base_color`).
pub(crate) fn principled_glass_tint(base: [f64; 4]) -> [f64; 3] {
    std::array::from_fn(|channel| base[channel].clamp(0.0, 1.0).sqrt())
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

/// Cycles' subsurface method, the Principled node's `subsurface_method` property (`node_shader_bsdf_principled.cc`,
/// `CLOSURE_BSSRDF_*_ID`). The default is random walk. Each method is its own closure: a Mix or Add of two arms with
/// different methods keeps both closures, so there is no mixed method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SubsurfaceMethod {
    Burley,
    #[default]
    RandomWalk,
    RandomWalkSkin,
    RandomWalkLegacy,
}

impl SubsurfaceMethod {
    /// The Blender enum identifier of a method, as the node property stores it.
    pub(crate) fn parse(token: &str) -> Option<Self> {
        match token {
            "BURLEY" => Some(Self::Burley),
            "RANDOM_WALK" => Some(Self::RandomWalk),
            "RANDOM_WALK_SKIN" => Some(Self::RandomWalkSkin),
            "RANDOM_WALK_LEGACY" => Some(Self::RandomWalkLegacy),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AnisotropyModel {
    #[default]
    Principled,
    Glossy,
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
    /// The Material Output's Thickness socket when it is linked (EEVEE's `nodetree_thickness`); `None` takes the
    /// object's bounds. A linked zero is no thickness, as EEVEE's `Thickness::from` gives it.
    pub thickness: Option<f64>,
    /// Blender's `Material.thickness_mode` is Slab rather than Sphere (`eevee_thickness_lib.bsl.hh`).
    pub thickness_slab: bool,
    /// Tints of the rough glass lobe (Cycles' `generalized_schlick_setup`): the reflection tint and the
    /// transmission tint. The Principled BSDF takes white reflection and the square root of its base colour
    /// for transmission (`kernel/svm/closure.h`); the Glass BSDF node takes its Color for both.
    pub glass_reflection_tint: [f64; 3],
    pub glass_transmission_tint: [f64; 3],
    /// The view is on the back of the geometric normal (Cycles' `SD_BACKFACING`), where a glass closure uses `1 / ior`.
    pub back_facing: bool,
    pub alpha: f64,
    /// Tint of the transparent share, Cycles' Transparent BSDF colour. The share is `1 - alpha`, so a camera
    /// pass-through multiplies by this tint and a shadow ray's transmittance is `(1 - alpha) * tint`
    /// (`render/path.rs`). Non-transparent closures keep white.
    pub transparent_tint: [f64; 3],
    /// Weight of the reflective share of this closure (diffuse, specular, transmission, environment). Emission
    /// contributes no reflection, so an Emission closure sets this to 0. Mix and Add scale the closure's weight
    /// instead of this value.
    pub reflectance: f64,
    pub subsurface_weight: f64,
    pub subsurface_radius: [f64; 3],
    pub subsurface_scale: f64,
    /// Cycles' subsurface method (`SubsurfaceMethod`), read from the Principled node's `subsurface_method` property.
    pub subsurface_method: SubsurfaceMethod,
    /// The Principled Subsurface IOR, used by the random walk skin method's refractive entry.
    pub subsurface_ior: f64,
    /// The Principled Subsurface Anisotropy: the thin branch's reflection and transmission split (`bsdf_thin_subsurface_setup`), and the random walk's phase function g.
    pub subsurface_anisotropy: f64,
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
    pub(crate) anisotropy_model: AnisotropyModel,
    pub anisotropy: f64,
    pub anisotropic_rotation: f64,
    /// Effective world-space Cycles tangent: a linked node or, for unlinked anisotropy, Geometry Tangent.
    pub anisotropic_tangent: Option<DVec3>,
    pub thin_film_thickness: f64,
    pub thin_film_ior: f64,
    /// Oren-Nayar roughness of the diffuse lobe (`bsdf_oren_nayar_setup`); Cycles uses the
    /// improved energy-preserving model when this is at least 1e-5.
    pub diffuse_roughness: f64,
    /// Thin Wall (Cycles' `thin_wall`, `svm/closure.h`): selects thin-glass and thin-subsurface
    /// lobes rather than a volume or BSSRDF closure.
    pub thin_wall: bool,
    /// Path state: Cycles replaces singular thin-glass transmission with a transparent closure for non-camera rays.
    pub(crate) thin_glass_transparent: bool,
    /// Cycles' scattering coefficient σs per channel, absolute: the volume closure's scatter weight times its
    /// Density. The medium sums the volume closures by their weights (`SurfaceShading::volume`).
    pub volume_scatter: [f64; 3],
    /// Cycles' absorption coefficient σa per channel, with the rule of `volume_scatter`. Extinction
    /// is σt = σs + σa, and the albedo is σs / σt.
    pub volume_absorption: [f64; 3],
    pub volume_anisotropy: f64,
    /// Emission radiance per unit density per channel: colour × strength × Density.
    pub volume_emission: [f64; 3],
    /// Blender's `cycles.volume_sampling` of the material this medium belongs to.
    pub volume_sampling: crate::model::VolumeSampling,
    /// Blender's `cycles.volume_interpolation` of the material this medium belongs to.
    pub volume_interpolation: crate::model::VolumeInterpolation,
    pub displacement_height: f64,
    pub normal: DVec3,
}

/// One closure of a surface, weighted as Cycles weights closures (`closure_alloc`): `weight` is the share of
/// the hit the closure takes, and `params` is the closure's own parameter set, with its own normal, roughness
/// and layers. Mix Shader and Add Shader never blend parameters: they scale and concatenate closures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShaderClosure {
    pub weight: f64,
    pub params: BsdfParams,
}

/// The shading of one hit: the surface's closures and the volume closures of its Volume output. Cycles
/// sums volume closures, so the medium is built from their weighted coefficients.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceShading {
    pub closures: Vec<ShaderClosure>,
    pub volume: Vec<ShaderClosure>,
    /// The Displacement output's height along the hit normal.
    pub displacement_height: f64,
}

impl SurfaceShading {
    /// The heaviest closure's parameters. A single-closure surface (any graph without Mix or Add, and every material
    /// without a graph) has exactly its one closure here.
    #[must_use]
    pub fn dominant(&self) -> Option<&BsdfParams> {
        heaviest(&self.closures, |closure| closure.weight).map(|closure| &closure.params)
    }
}

/// The heaviest item by its closure weight, the one Cycles' closure set takes its dominant normal from. `None` for no item.
pub(crate) fn heaviest<T>(items: &[T], weight: impl Fn(&T) -> f64) -> Option<&T> {
    items
        .iter()
        .max_by(|first, second| weight(first).total_cmp(&weight(second)))
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
            thickness: None,
            thickness_slab: material.thickness_mode == ThicknessMode::Slab,
            glass_reflection_tint: [1.0; 3],
            glass_transmission_tint: principled_glass_tint(material.base_color),
            back_facing: false,
            alpha: material.base_color[3],
            transparent_tint: [1.0; 3],
            reflectance: 1.0,
            subsurface_weight: 0.0,
            subsurface_radius: [1.0, 0.2, 0.1],
            subsurface_scale: 0.005,
            subsurface_method: SubsurfaceMethod::default(),
            subsurface_anisotropy: 0.0,
            subsurface_ior: 1.4,
            coat_weight: 0.0,
            coat_roughness: 0.03,
            coat_ior: 1.5,
            sheen_weight: 0.0,
            sheen_roughness: 0.5,
            sheen_tint: [1.0; 3],
            specular_ior_level: DEFAULT_SPECULAR_IOR_LEVEL,
            specular_tint: [1.0; 3],
            anisotropy_model: AnisotropyModel::Principled,
            anisotropy: 0.0,
            anisotropic_rotation: 0.0,
            anisotropic_tangent: None,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.33,
            diffuse_roughness: 0.0,
            thin_wall: false,
            thin_glass_transparent: false,
            volume_scatter: volume.scatter.map(|weight| weight * density),
            volume_absorption: volume.absorption.map(|weight| weight * density),
            volume_anisotropy: material.volume_anisotropy.clamp(-0.999, 0.999),
            volume_emission: material
                .emission_color
                .map(|channel| channel * material.emission_strength.max(0.0) * density),
            volume_sampling: material.volume_sampling,
            volume_interpolation: material.volume_interpolation,
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

    /// Whether the volume coefficients put a medium on the surface's path.
    pub(crate) fn has_medium(&self) -> bool {
        self.volume_scatter
            .iter()
            .chain(&self.volume_absorption)
            .any(|weight| *weight > 0.0)
    }
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

/// The realtime screen-space differentials used by EEVEE's shader bump evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitDifferentials {
    pub world_dx: DVec3,
    pub world_dy: DVec3,
    pub uv_dx: [f64; 2],
    pub uv_dy: [f64; 2],
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
    /// Per-corner world-space Mikk tangents in UV layer order; absent when mesh tangent data is unavailable.
    uv_tangents: Option<&'a [[DVec3; 3]]>,
    /// Sink for `uv.layer_missing` warnings; see [`HitContext::with_warnings`].
    pub warnings: Option<&'a RefCell<Vec<JsonValue>>>,
    /// The Generated texture coordinate: the object's texture space applied to its local point (see
    /// [`HitContext::with_generated`]).
    pub generated: DVec3,
    /// How far `generated` moves per unit of object-local movement (`1 / (2 size)` per axis).
    pub generated_per_local: DVec3,
    /// The Generated/orco values of this hit's triangle, used to transform radial tangent axes.
    generated_corners: Option<GeneratedCorners>,
    pub position: DVec3,
    /// The EEVEE world position interpolated from triangle vertices using f32 arithmetic.
    realtime_position_f32: Option<Vec3>,
    pub object: DVec3,
    /// The object's evaluated transform into world space for Vector Transform and Object Info.
    pub object_to_world: DMat4,
    /// The inverse of `object_to_world`; identity when it cannot be inverted.
    pub world_to_object: DMat4,
    /// Blender Object color/alpha and Cycles Object Info data resolved for this hit.
    pub object_color: [f64; 4],
    pub object_pass_index: f64,
    pub object_random_number: Option<f64>,
    pub material_pass_index: Option<f64>,
    /// The render camera's transform into world space; absent when shading has no camera context.
    pub camera_to_world: Option<DMat4>,
    /// The inverse of `camera_to_world`; absent when shading has no camera context.
    pub world_to_camera: Option<DMat4>,
    /// EEVEE's f32 world-to-camera matrix, cached when the camera matrices are assigned.
    world_to_camera_f32: Option<Mat4>,
    pub normal: DVec3,
    pub geometric_normal: DVec3,
    pub back_facing: bool,
    /// EEVEE screen-space surface and render-UV differentials for Bump.
    pub realtime_differentials: Option<HitDifferentials>,

    pub view_direction: DVec3,
    /// Cycles' `ShaderData.ray_length` for the current path hit; absent in realtime shading.
    pub light_falloff_distance: Option<f64>,
    /// Whether this hit is evaluated by Cycles semantics rather than the realtime shader path.
    pub path_traced: bool,
    /// The render color layer at this hit: the value a Vertex Color node with an empty layer name
    /// reads.
    pub color_render: Option<[f64; 4]>,
    /// Color layers at this hit, by layer name in layer order; see
    /// [`HitContext::with_color_layers`].
    pub color_layers: Option<&'a [(&'a str, [f64; 4])]>,
    /// The screen differential of a path-traced hit (Cycles' `sd->dP`), required for exact Bump evaluation.
    /// Realtime and preview contexts without this differential refuse Bump rather than approximate it.
    pub footprint: Option<HitFootprint>,
    /// Centre position reused by Camera Data while this hit is sampled for Bump Height.
    bump_center_position: DVec3,
    bump_center_realtime_position_f32: Option<Vec3>,
    /// Shadow opacity evaluation must not evaluate Bump Height when only the shading normal uses it.
    opacity_only: bool,
    bump_height_shifted: bool,
}

/// The camera footprint of a path-traced hit: Cycles' compact `sd->dP` (`kernel/geom/shader_data.h`) and the
/// geometric normal that orients its screen differentials (`differential_from_compact`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitFootprint {
    /// The compact differential: the average length of the hit's screen differentials, in world units.
    pub dp: f64,
    /// The geometric normal of the hit's triangle.
    pub geometric_normal: DVec3,
}

impl<'a> HitContext<'a> {
    #[must_use]
    pub const fn new(node_groups: &'a Registry<NodeGroup>) -> Self {
        Self {
            image_data: None,
            node_groups,
            uv: [0.0; 2],
            uv_layers: None,
            uv_tangents: None,
            warnings: None,
            generated: DVec3::ZERO,
            generated_per_local: DVec3::ONE,
            generated_corners: None,
            position: DVec3::ZERO,
            object_to_world: DMat4::IDENTITY,
            world_to_object: DMat4::IDENTITY,
            object_color: [1.0; 4],
            object_pass_index: 0.0,
            object_random_number: None,
            material_pass_index: None,
            camera_to_world: None,
            world_to_camera: None,
            realtime_position_f32: None,
            world_to_camera_f32: None,
            object: DVec3::ZERO,
            normal: DVec3::Z,
            geometric_normal: DVec3::Z,
            back_facing: false,
            view_direction: DVec3::Z,
            light_falloff_distance: None,
            path_traced: false,
            color_render: None,
            color_layers: None,
            footprint: None,
            realtime_differentials: None,
            bump_center_position: DVec3::ZERO,
            bump_center_realtime_position_f32: None,
            opacity_only: false,
            bump_height_shifted: false,
        }
    }

    /// Set the camera transforms and cache EEVEE's f32 world-to-camera matrix.
    pub fn set_camera_matrices(&mut self, camera_to_world: DMat4, world_to_camera: DMat4) {
        self.camera_to_world = Some(camera_to_world);
        self.world_to_camera = Some(world_to_camera);
        self.world_to_camera_f32 = Some(Mat4::from_cols_array(
            &world_to_camera.to_cols_array().map(|value| value as f32),
        ));
    }

    /// Set EEVEE's interpolated world position for a realtime triangle hit.
    #[must_use]
    pub(crate) fn with_realtime_position_f32(
        mut self,
        positions: [DVec3; 3],
        barycentric: [f64; 3],
    ) -> Self {
        let [first, second, third] = positions
            .map(|position| Vec3::new(position.x as f32, position.y as f32, position.z as f32));
        let [first_weight, second_weight, third_weight] = barycentric.map(|weight| weight as f32);
        self.realtime_position_f32 =
            Some(first * first_weight + second * second_weight + third * third_weight);
        self
    }

    /// Attach the camera footprint of a path-traced hit, which bump nodes read (Cycles' `sd->dP`).
    #[must_use]
    pub const fn with_footprint(mut self, footprint: HitFootprint) -> Self {
        self.footprint = Some(footprint);
        self
    }

    /// Mark evaluation as reading only opacity/shadow coverage; shading-normal-only Bump nodes are skipped.
    #[must_use]
    pub(crate) const fn with_opacity_only(mut self) -> Self {
        self.opacity_only = true;
        self
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

    /// Attach Mikk tangents for the hit triangle, in the same layer order as its UVs.
    #[must_use]
    pub(crate) const fn with_uv_tangents(mut self, tangents: &'a [[DVec3; 3]]) -> Self {
        self.uv_tangents = Some(tangents);
        self
    }

    /// Set the Generated coordinate of the object-local point `local` through the object's texture space
    /// (Cycles' `ATTR_STD_GENERATED`), and the Jacobian that moves it with `local` in shifted samples.
    #[must_use]
    pub fn with_generated(mut self, texture_space: TextureSpace, local: DVec3) -> Self {
        self.generated = texture_space.generated(local);
        self.generated_per_local = texture_space.generated_per_local();
        self.generated_corners = None;
        self
    }

    /// Set the Generated coordinate of a hit on a triangle whose corners carry theirs: the barycentric blend of the
    /// corner values, as Cycles interpolates the per-vertex `ATTR_STD_GENERATED` (`intern/cycles/blender/mesh.cpp`).
    /// A corner at zero contributes exactly zero, so a face at the texture space's edge stays at zero, as it does in
    /// Cycles. The Jacobian is the triangle's, since a triangle belongs to one object or collection member.
    #[must_use]
    pub fn with_generated_corners(
        mut self,
        generated: GeneratedCorners,
        barycentric: [f64; 3],
    ) -> Self {
        let [first, second, third] = generated.corners;
        self.generated = first * barycentric[0] + second * barycentric[1] + third * barycentric[2];
        self.generated_per_local = generated.per_local;
        self.generated_corners = Some(generated);
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
        if !self.bump_height_shifted {
            shifted.bump_center_position = self.position;
            shifted.bump_center_realtime_position_f32 = self.realtime_position_f32;
        }
        shifted.generated += step.world * self.generated_per_local;
        shifted.realtime_position_f32 = shifted.realtime_position_f32.map(|position| {
            position
                + Vec3::new(
                    step.world.x as f32,
                    step.world.y as f32,
                    step.world.z as f32,
                )
        });
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

    fn shifted_realtime(&self, step: SurfaceStep) -> Option<Self> {
        if !self.world_to_object.is_finite() {
            return None;
        }
        let local_step = self.world_to_object.transform_vector3(step.world);
        if !local_step.is_finite() {
            return None;
        }
        let mut shifted = self.shifted(step);
        shifted.object = self.object + local_step;
        shifted.generated = self.generated + local_step * self.generated_per_local;
        shifted.uv = [self.uv[0] + step.uv[0], self.uv[1] + step.uv[1]];
        Some(shifted)
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
pub fn evaluate_surface(material: &Material, hit_ctx: &HitContext<'_>) -> Result<SurfaceShading> {
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
) -> Result<SurfaceShading> {
    evaluate_surface_with_workspace_semantics(
        material,
        prepared,
        hit_ctx,
        workspace,
        crate::node_kernels::MathSemantics::Cycles,
    )
}

pub(crate) fn evaluate_surface_with_workspace_semantics(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<SurfaceShading> {
    let mut shading = SurfaceShading::default();
    evaluate_surface_with_workspace_into_semantics(
        material,
        prepared,
        hit_ctx,
        workspace,
        &mut shading,
        math_semantics,
    )?;
    Ok(shading)
}

pub(crate) fn evaluate_surface_with_workspace_into_semantics(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
    shading: &mut SurfaceShading,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<()> {
    workspace.reset();
    let result = evaluate_surface_inner_into(
        material,
        prepared,
        hit_ctx,
        workspace,
        shading,
        math_semantics,
    );
    workspace.reset();
    result
}

fn evaluate_surface_inner_into(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
    shading: &mut SurfaceShading,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<()> {
    shading.closures.clear();
    shading.volume.clear();
    shading.displacement_height = 0.0;
    let Some(graph_id) = &material.node_tree else {
        let mut params = BsdfParams::from_material(material, hit_ctx.normal);
        // The material is its own one arm, so its glass tint follows its textured base colour here.
        params.base_color = textured_base(material, hit_ctx, params.base_color)?;
        params.glass_transmission_tint = principled_glass_tint(params.base_color);
        // The material's volume coefficients are its Volume output, not a surface closure.
        shading.volume.push(ShaderClosure {
            weight: 1.0,
            params,
        });
        let mut surface_params = without_volume(params);
        apply_material_textures(material, hit_ctx, &mut surface_params)?;
        shading.closures.push(ShaderClosure {
            weight: 1.0,
            params: surface_params,
        });
        return Ok(());
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
    let (has_surface, has_volume, has_displacement) =
        prepared.output_feeds(graph_id, group, output);
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx: *hit_ctx,
        workspace,
        alpha_linked_images: &prepared.alpha_linked_images,
        links: prepared.fold_index(graph_id, group),
        group_input_binding: None,
        group_depth: 0,
        math_semantics,
    };
    // Only an unlinked Surface alongside a linked Volume is empty; other Shader sockets keep their defaults.
    if !has_surface && has_volume {
        shading.closures.clear();
    } else {
        let Value::Shader(surface) = evaluator.input(output, "Surface", ValueKind::Shader)? else {
            return Err(eval_error("material surface output is not a shader"));
        };
        shading
            .closures
            .extend_from_slice(evaluator.workspace.closures_of(surface));
    }
    // The volume closures come from the Volume output alone. Without one, the material's own
    // coefficients stand, as they do for a material with no graph.
    if has_volume {
        let Value::Shader(volume) = evaluator.input(output, "Volume", ValueKind::Shader)? else {
            return Err(eval_error("material volume output is not a volume shader"));
        };
        shading
            .volume
            .extend_from_slice(evaluator.workspace.closures_of(volume));
    } else {
        shading.volume.push(ShaderClosure {
            weight: 1.0,
            params: BsdfParams::from_material(material, hit_ctx.normal),
        });
    }
    // The Thickness socket, when linked, is the refraction thickness of the whole material (EEVEE's `nodetree_thickness`);
    // unlinked, the object's bounds decide it. The mode is the material's.
    let thickness = if group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Thickness")
    {
        match evaluator.input(output, "Thickness", ValueKind::Scalar)? {
            Value::Scalar(thickness) => Some(thickness.max(0.0)),
            _ => return Err(eval_error("material thickness is not a scalar")),
        }
    } else {
        None
    };
    let displacement_height = if has_displacement {
        let normal = hit_ctx.normal;
        match evaluator.input(output, "Displacement", ValueKind::Vector)? {
            Value::Vector(displacement) => displacement.dot(normal),
            Value::Scalar(displacement) => displacement,
            Value::Color(displacement) => {
                DVec3::new(displacement[0], displacement[1], displacement[2]).dot(normal)
            }
            Value::Shader(_) => return Err(eval_error("material displacement is a shader")),
        }
    } else {
        0.0
    };
    for closure in &mut shading.closures {
        apply_material_textures(material, hit_ctx, &mut closure.params)?;
        closure.params.thickness = thickness;
        closure.params.thickness_slab = material.thickness_mode == ThicknessMode::Slab;
    }
    shading.displacement_height = displacement_height;
    Ok(())
}

/// The surface closure's parameters without volume coefficients: those belong to the Volume output.
fn without_volume(mut params: BsdfParams) -> BsdfParams {
    params.volume_scatter = [0.0; 3];
    params.volume_absorption = [0.0; 3];
    params.volume_anisotropy = 0.0;
    params.volume_emission = [0.0; 3];
    params
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
    evaluate_displacement_with_workspace_semantics(
        material,
        prepared,
        hit_ctx,
        workspace,
        crate::node_kernels::MathSemantics::Cycles,
    )
}

pub(crate) fn evaluate_displacement_with_workspace_semantics(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<f64> {
    workspace.reset();
    let result =
        evaluate_displacement_inner(material, prepared, hit_ctx, workspace, math_semantics);
    workspace.reset();
    result
}

fn evaluate_displacement_inner(
    material: &Material,
    prepared: &PreparedMaterial,
    hit_ctx: &HitContext<'_>,
    workspace: &mut EvaluatorWorkspace,
    math_semantics: crate::node_kernels::MathSemantics,
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
    let (_, _, has_displacement) = prepared.output_feeds(graph_id, group, output);
    if !has_displacement {
        return Ok(0.0);
    }
    workspace.active.reserve(group.links.len());
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx: *hit_ctx,
        workspace,
        math_semantics,
        alpha_linked_images: &prepared.alpha_linked_images,
        links: prepared.fold_index(graph_id, group),
        group_input_binding: None,
        group_depth: 0,
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

/// The material-level base colour texture multiplied into a Principled arm's base colour. The arm derives its
/// glass tint from the result (Cycles' `clamped_base_color`). Blender's exporter links the texture's Alpha only for
/// a non-opaque material, and Cycles' colour read depends on that link (`apply_material_textures`).
fn textured_base(
    material: &Material,
    hit_ctx: &HitContext<'_>,
    base: [f64; 4],
) -> Result<[f64; 4]> {
    let Some(texture) = &material.base_color_texture else {
        return Ok(base);
    };
    let alpha_linked = material.alpha_mode != "opaque";
    let pixel = sample_material_texture(
        texture,
        hit_ctx,
        texture_layer(hit_ctx, texture),
        alpha_linked,
    )?;
    let mut textured = base;
    for (channel, texel) in textured[..3].iter_mut().zip(pixel) {
        *channel *= texel;
    }
    Ok(textured)
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
    // The base colour is multiplied into each Principled arm (`textured_base`), where its glass tint follows it. The
    // texture's Alpha is multiplied here, and only for a non-opaque material (Blender's exporter links it only then).
    if let Some(texture) = &material.base_color_texture
        && material.alpha_mode != "opaque"
    {
        let pixel =
            sample_material_texture(texture, hit_ctx, texture_layer(hit_ctx, texture), true)?;
        params.alpha *= pixel[3];
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
enum Value {
    Scalar(f64),
    Vector(DVec3),
    Color([f64; 4]),
    /// The closures of a shader socket, as a span of the workspace arena.
    Shader(ClosureSpan),
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
    VectorRotate,
    VectorTransform,
    MapRange,
    Clamp,
    FloatCurve,
    RgbCurve,
    VectorCurve,
    Mapping,
    Value,
    Rgb,
    CombineColor,
    RgbToBw,
    Invert,
    Gamma,
    BrightContrast,
    HueSaturation,
    Normal,
    Wavelength,
    Blackbody,
    LightFalloff,
    ObjectInfo,
    Geometry,
    CameraData,
    Group,
    GroupInput,
    SeparateXyz,
    CombineXyz,
    SeparateColor,
    TexCoord,
    UvMap,
    Tangent,
    NormalMap,
    Fresnel,
    LayerWeight,
    Attribute,
    VertexColor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VectorRotationType {
    AxisAngle,
    AxisX,
    AxisY,
    AxisZ,
    EulerXyz,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VectorTransformType {
    Point,
    Vector,
    Normal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VectorTransformSpace {
    Object,
    World,
    Camera,
}

impl VectorTransformSpace {
    fn name(self) -> &'static str {
        match self {
            Self::Object => "OBJECT",
            Self::World => "WORLD",
            Self::Camera => "CAMERA",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NormalMapSpace {
    Tangent,
    Object,
    World,
    BlenderObject,
    BlenderWorld,
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
    /// The closures of the evaluation in progress. A `Value::Shader` names a span of them, so a node's value stays
    /// `Copy` and Mix and Add append to the arena instead of allocating per node.
    closures: Vec<ShaderClosure>,
    /// Active nested graph identities; prevents recursive node groups from exhausting the render stack.
    active_groups: Vec<usize>,
    /// Prepared static node mappings survive hit resets; source JSON equality invalidates edits.
    curve_mappings: Vec<PreparedNodeCurve>,
    /// Prepared ramp elements and their lookup tables survive hit resets; source-property equality invalidates edits.
    ramps: Vec<PreparedNodeRamp>,
}
struct PreparedNodeRamp {
    graph: usize,
    node: usize,
    source: serde_json::Map<String, JsonValue>,
    elements: Vec<(f32, [f32; 4])>,
    table: Vec<[f32; 4]>,
    eevee_table: Vec<[f32; 4]>,
    interpolation: &'static str,
    color_mode: &'static str,
}
struct PreparedNodeCurve {
    graph: usize,
    node: usize,
    source: Option<JsonValue>,
    prepared: crate::color::curve::PreparedCurve,
}

/// A run of closures in the workspace arena (`EvaluatorWorkspace::closures`).
#[derive(Clone, Copy, Debug)]
struct ClosureSpan {
    start: usize,
    len: usize,
}

impl EvaluatorWorkspace {
    /// Clears the scratch for a new evaluation. Spans from an earlier evaluation are invalid after this.
    fn reset(&mut self) {
        self.active.clear();
        self.closures.clear();
        self.active_groups.clear();
    }

    /// A node's single closure at weight 1.
    fn push_single(&mut self, params: &BsdfParams) -> ClosureSpan {
        let start = self.closures.len();
        self.closures.push(ShaderClosure {
            weight: 1.0,
            params: *params,
        });
        ClosureSpan { start, len: 1 }
    }

    fn closures_of(&self, span: ClosureSpan) -> &[ShaderClosure] {
        &self.closures[span.start..span.start + span.len]
    }

    /// Appends a copy of `span` with every weight scaled by `factor`.
    fn push_scaled(&mut self, span: ClosureSpan, factor: f64) {
        for index in span.start..span.start + span.len {
            let closure = self.closures[index];
            self.closures.push(ShaderClosure {
                weight: closure.weight * factor,
                ..closure
            });
        }
    }

    /// Mix Shader: the first arm's closures at `1 - factor` and the second's at `factor`. Each closure keeps its
    /// own normal, roughness and layers, so the arms are never averaged.
    fn mix(&mut self, first: ClosureSpan, second: ClosureSpan, factor: f64) -> ClosureSpan {
        let factor = bounded_factor(factor);
        let start = self.closures.len();
        self.push_scaled(first, 1.0 - factor);
        self.push_scaled(second, factor);
        ClosureSpan {
            start,
            len: self.closures.len() - start,
        }
    }

    /// Add Shader: the closures of both arms. Closure weights add, as Cycles sums closures.
    fn add(&mut self, first: ClosureSpan, second: ClosureSpan) -> ClosureSpan {
        let start = self.closures.len();
        self.push_scaled(first, 1.0);
        self.push_scaled(second, 1.0);
        ClosureSpan {
            start,
            len: self.closures.len() - start,
        }
    }
}

/// The per-material state a shader evaluation reads. A render builds it once for each material and passes it to every
/// evaluation of that material, so no hit walks the graph again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PreparedMaterial {
    /// The Image Texture nodes whose Alpha output reaches the graph's output (see [`linked_alpha_image_nodes`]).
    alpha_linked_images: BTreeSet<Id>,
    /// The first Bump whose output feeds realtime transparency or its coverage factor.
    bump_transparency_node: Option<Id>,
    /// Whether EEVEE would compile the material as transparent (`GPU_MATFLAG_TRANSPARENT`), which decides whether its
    /// surfaces cast transparent shadows (see [`graph_is_transparent`]).
    transparent: bool,
    /// The graph's incoming links by target node, and the output's incoming sockets (see [`LinkIndex`]).
    links: LinkIndex,
}

/// The links of a shader graph, indexed once per material so that a socket read does not scan every link. It is only
/// used for the graph it was built from (see [`PreparedMaterial::link_index`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct LinkIndex {
    /// The graph this index was built from; `None` when the material has no graph.
    graph: Option<Id>,
    /// The address and length of the graph's `links` buffer when the index was built. This is a guard, not a content
    /// check: a buffer replaced by one of the same address and length, or a link rewritten in place, is not detected.
    /// The crate does not keep a `PreparedMaterial` across document edits: each render prepares its materials from the
    /// document it reads, so an index is never used against links it was not built from. The lookup also re-checks the
    /// target node and socket of each candidate link.
    links_address: usize,
    link_count: usize,
    /// Prebuilt incoming-link index and primitive folds for every nested group invocation.
    folds: super::inline_fold::FoldIndex,
    /// The output node that the socket flags below describe.
    output: Option<Id>,
    /// Whether a link reaches the output's `Surface` socket.
    has_surface: bool,
    /// Whether a link reaches the output's `Volume` socket.
    has_volume: bool,
    /// Whether a link reaches the output's `Displacement` socket.
    has_displacement: bool,
}

impl LinkIndex {
    fn new(
        graph_id: &Id,
        group: &NodeGroup,
        output: Option<&Id>,
        node_groups: &Registry<NodeGroup>,
    ) -> Self {
        let folds = super::inline_fold::FoldIndex::new(graph_id, group, node_groups);
        let (has_surface, has_volume, has_displacement) = output
            .map_or((false, false, false), |output| {
                scan_output_feeds(group, output)
            });
        Self {
            graph: Some(graph_id.clone()),
            links_address: group.links.as_ptr().addr(),
            link_count: group.links.len(),
            folds,
            output: output.cloned(),
            has_volume,
            has_surface,
            has_displacement,
        }
    }
}

/// Whether links reach the `Surface`, `Volume`, and `Displacement` sockets of `output`.
fn scan_output_feeds(group: &NodeGroup, output: &Id) -> (bool, bool, bool) {
    (
        group
            .links
            .iter()
            .any(|link| link.to_node == *output && link.to_socket == "Surface"),
        group
            .links
            .iter()
            .any(|link| link.to_node == *output && link.to_socket == "Volume"),
        group
            .links
            .iter()
            .any(|link| link.to_node == *output && link.to_socket == "Displacement"),
    )
}

impl PreparedMaterial {
    /// Prepares `material` against the node groups its graph is looked up in. A material without a shader graph prepares
    /// to its own PBR fields: its export is a Principled whose Alpha is the base colour alpha for a non-opaque material
    /// (`blend_bridge.py` `_set_principled_values`), so it is transparent when that alpha is below one.
    #[must_use]
    pub fn new(material: &Material, node_groups: &Registry<NodeGroup>) -> Self {
        let Some((graph_id, group)) = material
            .node_tree
            .as_ref()
            .and_then(|graph_id| node_groups.get(graph_id).map(|group| (graph_id, group)))
        else {
            return Self {
                transparent: material.alpha_mode != "opaque" && material.base_color[3] < 1.0,
                ..Self::default()
            };
        };
        let output = Self::active_output(group);
        let links = LinkIndex::new(graph_id, group, output, node_groups);
        let Some(output) = output else {
            return Self {
                links,
                ..Self::default()
            };
        };
        let mut alpha_linked_images = BTreeSet::new();
        let mut visited_groups = BTreeSet::new();
        visited_groups.insert(graph_id.clone());
        collect_linked_alpha_images(
            node_groups,
            group,
            output,
            &mut visited_groups,
            &mut alpha_linked_images,
        );
        let transparent = graph_is_transparent(graph_id, group, node_groups, output);
        Self {
            alpha_linked_images,
            bump_transparency_node: transparent
                .then(|| bump_affecting_transparency(graph_id, group, node_groups, output))
                .flatten(),
            transparent,
            links,
        }
    }

    /// The index built from `group` when this material was prepared from that graph and its links buffer is the same
    /// one, with the same length. `None`, as for a `PreparedMaterial::default()` or another graph, makes the evaluator
    /// scan the links.
    fn link_index(&self, graph_id: &Id, group: &NodeGroup) -> Option<&LinkIndex> {
        let index = &self.links;
        (index.graph.as_ref() == Some(graph_id)
            && index.links_address == group.links.as_ptr().addr()
            && index.link_count == group.links.len())
        .then_some(index)
    }

    fn fold_index(
        &self,
        graph_id: &Id,
        group: &NodeGroup,
    ) -> Option<&super::inline_fold::FoldIndex> {
        self.link_index(graph_id, group).map(|index| &index.folds)
    }
    /// Whether a link reaches the `Surface`, `Volume`, and `Displacement` sockets of `output` in `group`.
    fn output_feeds(&self, graph_id: &Id, group: &NodeGroup, output: &Id) -> (bool, bool, bool) {
        match self
            .link_index(graph_id, group)
            .filter(|index| index.output.as_ref() == Some(output))
        {
            Some(index) => (index.has_surface, index.has_volume, index.has_displacement),
            None => scan_output_feeds(group, output),
        }
    }

    /// Whether a surface of this material casts a transparent shadow, as EEVEE's static transparency decides it.
    pub(crate) fn is_transparent(&self) -> bool {
        self.transparent
    }
    /// A Bump that changes realtime transparent-shadow coverage must be refused, not evaluated with an unbumped normal.
    pub(crate) fn realtime_transparency_bump(&self) -> Option<&Id> {
        self.bump_transparency_node.as_ref()
    }
    /// Whether this material graph has a linked Volume output.
    pub(crate) fn has_volume_output(&self) -> bool {
        self.links.graph.is_some() && self.links.has_volume
    }

    /// Whether the graph has a linked Volume output and no linked Surface output.
    pub(crate) fn is_volume_only(&self) -> bool {
        self.has_volume_output() && !self.links.has_surface
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

#[derive(Clone, Copy)]
struct GroupInputBinding<'a> {
    group: &'a NodeGroup,
    node_id: &'a Id,
    binding: Option<&'a GroupInputBinding<'a>>,
    links: Option<&'a super::inline_fold::FoldIndex>,
    depth: usize,
}

struct Evaluator<'graph, 'material, 'registry, 'workspace, 'prepared, 'scope> {
    group: &'graph NodeGroup,
    material: &'material Material,
    hit_ctx: HitContext<'registry>,
    workspace: &'workspace mut EvaluatorWorkspace,
    math_semantics: crate::node_kernels::MathSemantics,
    /// The Alpha-linked Image Texture nodes of the material, from [`PreparedMaterial`].
    alpha_linked_images: &'prepared BTreeSet<Id>,
    /// The graph's prebuilt incoming-link and fold index; `None` scans the links and skips prepared folding.
    links: Option<&'scope super::inline_fold::FoldIndex>,
    /// The call site whose socket values feed this group's Group Input nodes.
    group_input_binding: Option<&'scope GroupInputBinding<'scope>>,
    group_depth: usize,
}

impl<'registry> Evaluator<'_, '_, 'registry, '_, '_, '_> {
    fn group_interface_input_default<'a>(
        &'a self,
        node: &GraphNode,
        socket: &str,
    ) -> Option<&'a JsonValue> {
        let group_id = node.properties.get("node_tree")?.as_str()?;
        self.hit_ctx
            .node_groups
            .get(group_id)?
            .interface
            .inputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
            .map(|candidate| &candidate.default)
            .filter(|default| !default.is_null())
    }

    fn input(&mut self, node_id: &Id, socket: &str, kind: ValueKind) -> Result<Value> {
        let node = self.node(node_id)?;
        let group = self.group;
        let incoming = match self.links {
            Some(index) => {
                index
                    .incoming
                    .get(node_id)
                    .into_iter()
                    .flatten()
                    .find_map(|&socket_index| {
                        let link = group.links.get(socket_index)?;
                        (link.to_node == *node_id
                            && input_socket_matches(
                                self.hit_ctx.node_groups,
                                node,
                                &link.to_socket,
                                socket,
                            ))
                        .then_some((socket_index, link))
                    })
            }
            None => group.links.iter().enumerate().find(|(_, link)| {
                link.to_node == *node_id
                    && input_socket_matches(self.hit_ctx.node_groups, node, &link.to_socket, socket)
            }),
        };
        if let Some((socket_index, link)) = incoming {
            return self.output(&link.from_node, &link.from_socket, socket_index);
        }
        let raw = node
            .inputs
            .iter()
            .find(|(name, _)| input_socket_matches(self.hit_ctx.node_groups, node, name, socket))
            .map(|(_, value)| value)
            .filter(|value| !value.is_null());
        let default = raw.or_else(|| socket_default(node, socket)).or_else(|| {
            (node.node_type == "ShaderNodeGroup")
                .then(|| self.group_interface_input_default(node, socket))
                .flatten()
        });
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
            ValueKind::Shader => {
                if raw.is_some() {
                    Err(invalid("a shader socket cannot use a literal value"))
                } else {
                    let params = self.closure_params();
                    Ok(self.closure_value(&params))
                }
            }
        }
    }

    fn group_input(&mut self, node_id: &Id, socket: &str) -> Result<Value> {
        let Some(binding) = self.group_input_binding else {
            return Err(unsupported_node(node_id, "NodeGroupInput.call_site"));
        };
        let Some(input_socket) = self
            .group
            .interface
            .inputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return Err(invalid(format!("unknown Group Input output `{socket}`")));
        };
        let kind = group_socket_value_kind(&input_socket.socket_type)
            .ok_or_else(|| unsupported_node_id("NodeGroupInput", socket))?;
        let (parent_group, parent_node_id, parent_binding, parent_links, parent_depth) = (
            binding.group,
            binding.node_id,
            binding.binding,
            binding.links,
            binding.depth,
        );
        let mut parent = Evaluator {
            group: parent_group,
            material: self.material,
            hit_ctx: self.hit_ctx,
            workspace: self.workspace,
            math_semantics: self.math_semantics,
            alpha_linked_images: self.alpha_linked_images,
            links: parent_links,
            group_input_binding: parent_binding,
            group_depth: parent_depth,
        };
        parent.input(parent_node_id, &input_socket.id, kind)
    }

    fn group_node(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if self.group_depth >= 64 {
            return Err(unsupported_node(id, "ShaderNodeGroup.recursion"));
        }
        let node = self.node(id)?;
        let Some(group_id) = node.properties.get("node_tree").and_then(JsonValue::as_str) else {
            return Err(unsupported_node(id, "ShaderNodeGroup.node_tree"));
        };
        let node_groups = self.hit_ctx.node_groups;
        let Some((_, child_group)) = node_groups.get_key_value(group_id) else {
            return Err(unsupported_node(id, "ShaderNodeGroup.node_tree"));
        };
        if child_group.kind != GraphKind::Shader {
            return Err(unsupported_node(id, "ShaderNodeGroup.graph_kind"));
        }
        let Some(output_socket) = child_group
            .interface
            .outputs
            .iter()
            .find(|candidate| candidate.id == socket || candidate.name == socket)
        else {
            return Err(invalid(format!("unknown Shader Group output `{socket}`")));
        };
        let output_kind = group_socket_value_kind(&output_socket.socket_type)
            .ok_or_else(|| unsupported_node(id, "ShaderNodeGroup.output_socket_type"))?;
        let Some(output_node) = active_group_output(child_group) else {
            return Err(unsupported_node(id, "ShaderNodeGroup.output"));
        };

        let child_key = std::ptr::from_ref(child_group).addr();
        if self.workspace.active_groups.contains(&child_key) {
            return Err(unsupported_node(id, "ShaderNodeGroup.recursion"));
        }
        let active_start = self.workspace.active_groups.len();
        let group_depth = self.group_depth + 1;
        let material = self.material;
        let hit_ctx = self.hit_ctx;
        let alpha_linked_images = self.alpha_linked_images;
        let binding = GroupInputBinding {
            group: self.group,
            node_id: id,
            binding: self.group_input_binding,
            links: self.links,
            depth: self.group_depth,
        };
        self.workspace.active_groups.push(child_key);
        let mut nested = Evaluator {
            group: child_group,
            material,
            hit_ctx,
            workspace: self.workspace,
            math_semantics: self.math_semantics,
            alpha_linked_images,
            links: self.links.and_then(|index| index.children.get(id)),
            group_input_binding: Some(&binding),
            group_depth,
        };
        let result = nested.input(output_node, &output_socket.name, output_kind);
        self.workspace.active_groups.truncate(active_start);
        result
    }

    /// Whether `socket` of `node_id` takes a link, so its value comes from another node rather than the socket's
    /// stored default.
    fn is_linked(&self, node_id: &Id, socket: &str) -> bool {
        match self.links {
            Some(index) => index
                .incoming
                .get(node_id)
                .into_iter()
                .flatten()
                .filter_map(|&link_index| self.group.links.get(link_index))
                .any(|link| link.to_node == *node_id && socket_matches(&link.to_socket, socket)),
            None => self
                .group
                .links
                .iter()
                .any(|link| link.to_node == *node_id && socket_matches(&link.to_socket, socket)),
        }
    }
    fn inputs_fold_to_primitives(&self, node_id: &Id) -> bool {
        self.links.map_or_else(
            || super::inline_fold::node_inputs_all_primitive(self.group, node_id),
            |links| links.primitive_nodes.contains(node_id),
        )
    }
    /// The effective lobe tangent: explicit Tangent link, or Cycles' implicit Geometry Tangent.
    fn anisotropic_tangent(&mut self, id: &Id, active: bool) -> Result<Option<DVec3>> {
        if !active {
            return Ok(None);
        }
        if !self.is_linked(id, "Tangent") {
            return self.radial_tangent(id, "Z").map(Some);
        }
        let tangent = match self.input(id, "Tangent", ValueKind::Vector)? {
            Value::Vector(value) => value,
            Value::Scalar(value) => DVec3::splat(value),
            Value::Color(value) => DVec3::new(value[0], value[1], value[2]),
            Value::Shader(_) => return Err(invalid("expected an anisotropy tangent vector")),
        };
        if !tangent.is_finite() || tangent.length_squared() <= f64::EPSILON {
            return Err(unsupported_node(id, "Anisotropic:Tangent"));
        }
        Ok(Some(tangent.normalize()))
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
            "ShaderNodeGroup" => Ok(NodeKind::Group),
            "NodeGroupInput" => Ok(NodeKind::GroupInput),
            "ShaderNodeValue" => Ok(NodeKind::Value),
            "ShaderNodeRGB" => Ok(NodeKind::Rgb),
            "ShaderNodeCombineColor" => Ok(NodeKind::CombineColor),
            "ShaderNodeMapRange" => Ok(NodeKind::MapRange),
            "ShaderNodeClamp" => Ok(NodeKind::Clamp),
            "ShaderNodeFloatCurve" => Ok(NodeKind::FloatCurve),
            "ShaderNodeRGBCurve" => Ok(NodeKind::RgbCurve),
            "ShaderNodeVectorCurve" => Ok(NodeKind::VectorCurve),
            "ShaderNodeSeparateColor" => Ok(NodeKind::SeparateColor),
            "ShaderNodeRGBToBW" => Ok(NodeKind::RgbToBw),
            "ShaderNodeInvert" => Ok(NodeKind::Invert),
            "ShaderNodeGamma" => Ok(NodeKind::Gamma),
            "ShaderNodeBrightContrast" => Ok(NodeKind::BrightContrast),
            "ShaderNodeHueSaturation" => Ok(NodeKind::HueSaturation),
            "ShaderNodeNormal" => Ok(NodeKind::Normal),
            "ShaderNodeWavelength" => Ok(NodeKind::Wavelength),
            "ShaderNodeBlackbody" => Ok(NodeKind::Blackbody),
            "ShaderNodeLightFalloff" => Ok(NodeKind::LightFalloff),
            "ShaderNodeObjectInfo" => Ok(NodeKind::ObjectInfo),
            "ShaderNodeNewGeometry" => Ok(NodeKind::Geometry),
            "ShaderNodeCameraData" => Ok(NodeKind::CameraData),
            "ShaderNodeVectorRotate" => Ok(NodeKind::VectorRotate),
            "ShaderNodeVectorTransform" => Ok(NodeKind::VectorTransform),
            "ShaderNodeMapping" => Ok(NodeKind::Mapping),
            "ShaderNodeSeparateXYZ" => Ok(NodeKind::SeparateXyz),
            "ShaderNodeCombineXYZ" => Ok(NodeKind::CombineXyz),
            "ShaderNodeTexCoord" => Ok(NodeKind::TexCoord),
            "ShaderNodeUVMap" => Ok(NodeKind::UvMap),
            "ShaderNodeTangent" => Ok(NodeKind::Tangent),
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
    fn output_default(&self, node_id: &Id, socket: &str) -> Option<&JsonValue> {
        self.node(node_id)
            .ok()?
            .properties
            .get("potter_output_defaults")?
            .as_array()?
            .iter()
            .find(|entry| entry.get("identifier").and_then(JsonValue::as_str) == Some(socket))
            .and_then(|entry| entry.get("default"))
            .filter(|value| !value.is_null())
    }
    fn required_output_default(
        &self,
        node_id: &Id,
        node_type: &str,
        socket: &str,
    ) -> Result<Option<&JsonValue>> {
        let default = self.output_default(node_id, socket);
        let node = self.node(node_id)?;
        if default.is_none()
            && (node.properties.contains_key("bl_idname")
                || node.properties.contains_key("potter_output_defaults"))
        {
            return Err(unsupported_node(
                node_id,
                &format!("{node_type}.output_defaults"),
            ));
        }
        Ok(default)
    }
    fn prepared_ramp(&mut self, id: &Id) -> Result<usize> {
        let node = self
            .group
            .nodes
            .get(id)
            .ok_or_else(|| invalid(format!("shader graph references absent node `{id}`")))?;
        let graph = std::ptr::from_ref(self.group) as usize;
        let node_key = std::ptr::from_ref(node) as usize;
        if let Some(index) = self.workspace.ramps.iter().position(|ramp| {
            ramp.graph == graph && ramp.node == node_key && ramp.source == node.properties
        }) {
            return Ok(index);
        }
        let settings = match node.properties.get("color_ramp") {
            Some(JsonValue::Object(settings)) => {
                if let Some(field) = settings.keys().find(|field| {
                    !matches!(
                        field.as_str(),
                        "color_mode" | "hue_interpolation" | "interpolation" | "elements"
                    )
                }) {
                    return Err(unsupported_node(
                        id,
                        &format!("ShaderNodeValToRGB.color_ramp.{field}"),
                    ));
                }
                settings
            }
            Some(_) => {
                return Err(unsupported_node(id, "ShaderNodeValToRGB.color_ramp.state"));
            }
            None => &node.properties,
        };
        let raw_elements = settings
            .get("elements")
            .or_else(|| node.properties.get("elements"));
        let elements = if let Some(raw_elements) = raw_elements {
            let raw_elements = raw_elements
                .as_array()
                .ok_or_else(|| unsupported_node(id, "ShaderNodeValToRGB.color_ramp.elements"))?;
            if raw_elements.is_empty() {
                return Err(invalid("ColorRamp requires at least one color element"));
            }
            let mut elements = Vec::with_capacity(raw_elements.len());
            for raw in raw_elements {
                let (position, color) = ramp_element(raw)?;
                elements.push((position as f32, color.map(|component| component as f32)));
            }
            elements
        } else if node.properties.contains_key("color_ramp") {
            return Err(unsupported_node(id, "ShaderNodeValToRGB.color_ramp.state"));
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
            vec![
                (0.0, first.map(|component| component as f32)),
                (1.0, second.map(|component| component as f32)),
            ]
        };
        if elements.iter().any(|(position, color)| {
            !position.is_finite() || color.iter().any(|component| !component.is_finite())
        }) {
            return Err(invalid("ColorRamp elements must contain finite f32 values"));
        }
        if elements.windows(2).any(|pair| pair[1].0 < pair[0].0) {
            return Err(invalid("ColorRamp element positions must be ordered"));
        }
        let interpolation = node_setting_string(
            id,
            settings,
            "interpolation",
            "LINEAR",
            "ShaderNodeValToRGB.ColorRamp",
            RAMP_INTERPOLATIONS,
        )?;
        let color_mode = node_setting_string(
            id,
            settings,
            "color_mode",
            "RGB",
            "ShaderNodeValToRGB.ColorRamp",
            RAMP_COLOR_MODES,
        )?;
        let hue_interpolation = node_setting_string(
            id,
            settings,
            "hue_interpolation",
            "NEAR",
            "ShaderNodeValToRGB.ColorRamp",
            RAMP_HUE_INTERPOLATIONS,
        )?;
        let table = (0..crate::node_kernels::COLOR_RAMP_TABLE_SIZE)
            .map(|index| {
                crate::node_kernels::color_ramp(
                    &elements,
                    index as f32 / (crate::node_kernels::COLOR_RAMP_TABLE_SIZE - 1) as f32,
                    interpolation,
                    color_mode,
                    hue_interpolation,
                )
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| unsupported_node(id, "ShaderNodeValToRGB.color_ramp"))?;
        let eevee_table = table
            .iter()
            .map(|color| {
                std::array::from_fn(|channel| exr::prelude::f16::from_f32(color[channel]).to_f32())
            })
            .collect();
        let ramp = PreparedNodeRamp {
            graph,
            node: node_key,
            source: node.properties.clone(),
            elements,
            table,
            eevee_table,
            interpolation,
            color_mode,
        };
        if let Some(index) = self
            .workspace
            .ramps
            .iter()
            .position(|cached| cached.graph == graph && cached.node == node_key)
        {
            self.workspace.ramps[index] = ramp;
            Ok(index)
        } else {
            let index = self.workspace.ramps.len();
            self.workspace.ramps.push(ramp);
            Ok(index)
        }
    }
    fn prepared_curve(&mut self, id: &Id) -> Result<usize> {
        let node = self
            .group
            .nodes
            .get(id)
            .ok_or_else(|| invalid(format!("shader graph references absent node `{id}`")))?;
        let graph = std::ptr::from_ref(self.group) as usize;
        let node_key = std::ptr::from_ref(node) as usize;
        let source = node.properties.get("mapping");
        if let Some(index) = self.workspace.curve_mappings.iter().position(|curve| {
            curve.graph == graph && curve.node == node_key && curve.source.as_ref() == source
        }) {
            return Ok(index);
        }
        let mapping = match source {
            Some(value) => {
                crate::node_kernels::curve_mapping_from_value(value).ok_or_else(|| {
                    unsupported_node(id, &format!("{}.mapping.CurveMapping", node.node_type))
                })?
            }
            None => crate::color::curve::CurveMapping::default(),
        };
        let prepared = crate::node_kernels::curve_mapping_prepare(&mapping);
        let curve = PreparedNodeCurve {
            graph,
            node: node_key,
            source: source.cloned(),
            prepared,
        };
        if let Some(index) = self
            .workspace
            .curve_mappings
            .iter()
            .position(|cached| cached.graph == graph && cached.node == node_key)
        {
            self.workspace.curve_mappings[index] = curve;
            Ok(index)
        } else {
            let index = self.workspace.curve_mappings.len();
            self.workspace.curve_mappings.push(curve);
            Ok(index)
        }
    }

    /// Whether the Alpha output of the Image Texture `id` reaches the group's output (see [`PreparedMaterial`]).
    fn alpha_linked_image(&self, id: &Id) -> bool {
        self.alpha_linked_images.contains(id)
    }

    fn evaluate_node(&mut self, node_id: &Id, kind: NodeKind, socket: &str) -> Result<Value> {
        match kind {
            NodeKind::Group => self.group_node(node_id, socket),
            NodeKind::GroupInput => self.group_input(node_id, socket),
            NodeKind::Value => self.value(node_id, socket),
            NodeKind::Rgb => self.rgb(node_id, socket),
            NodeKind::CombineColor => self.combine_color(node_id, socket),
            NodeKind::SeparateColor => self.separate_color(node_id, socket),
            NodeKind::RgbToBw => self.rgb_to_bw(node_id, socket),
            NodeKind::Invert => self.invert(node_id, socket),
            NodeKind::Gamma => self.gamma(node_id, socket),
            NodeKind::BrightContrast => self.bright_contrast(node_id, socket),
            NodeKind::HueSaturation => self.hue_saturation(node_id, socket),
            NodeKind::Normal => self.normal(node_id, socket),
            NodeKind::Wavelength => self.wavelength(node_id, socket),
            NodeKind::Blackbody => self.blackbody(node_id, socket),
            NodeKind::LightFalloff => self.light_falloff(node_id, socket),
            NodeKind::ObjectInfo => self.object_info(node_id, socket),
            NodeKind::Geometry => self.geometry(node_id, socket),
            NodeKind::CameraData => self.camera_data(node_id, socket),
            NodeKind::OutputMaterial => Err(invalid("material output has no output sockets")),
            NodeKind::Principled => {
                let params = self.principled(node_id)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::VolumePrincipled => {
                let params = self.volume(node_id, VolumeClosure::Principled)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::VolumeScatter => {
                let params = self.volume(node_id, VolumeClosure::Scatter)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::VolumeAbsorption => {
                let params = self.volume(node_id, VolumeClosure::Absorption)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::Displacement => self.displacement(node_id, socket),
            NodeKind::ImageTexture => self.image_texture(node_id, socket),
            NodeKind::Emission => {
                let params = self.emission(node_id)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::Diffuse => {
                // Blender's Diffuse BSDF has no specular layer; its Roughness selects the same
                // Oren-Nayar closure as Principled's Diffuse Roughness.
                let mut params = self.diffuse(node_id)?;
                params.specular_ior_level = 0.0;
                Ok(self.closure_value(&params))
            }
            NodeKind::Glossy => {
                let params = self.glossy(node_id)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::Glass => {
                let params = self.glass(node_id)?;
                Ok(self.closure_value(&params))
            }
            NodeKind::Transparent => {
                let params = self.transparent(node_id)?;
                Ok(self.closure_value(&params))
            }
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
            NodeKind::VectorRotate => self.vector_rotate(node_id, socket),
            NodeKind::VectorTransform => self.vector_transform(node_id, socket),
            NodeKind::MapRange => self.map_range(node_id, socket),
            NodeKind::Clamp => self.clamp(node_id, socket),
            NodeKind::FloatCurve => self.float_curve(node_id, socket),
            NodeKind::RgbCurve => self.rgb_curve(node_id, socket),
            NodeKind::VectorCurve => self.vector_curve(node_id, socket),
            NodeKind::Mapping => self.mapping(node_id, socket),
            NodeKind::SeparateXyz => self.separate_xyz(node_id, socket),
            NodeKind::CombineXyz => self.combine_xyz(node_id, socket),
            NodeKind::TexCoord => self.tex_coord(node_id, socket),
            NodeKind::UvMap => self.uv_map(node_id, socket),
            NodeKind::Tangent => self.tangent(node_id, socket),
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
        without_volume(BsdfParams {
            alpha: 1.0,
            // A closure is not glass: only Glass and Principled set transmission. The material's own value
            // belongs to the simple material's Principled (set there).
            transmission: 0.0,
            ..BsdfParams::from_material(self.material, self.hit_ctx.normal)
        })
    }

    /// Appends a node's single closure to the workspace arena and returns it as a socket value.
    fn closure_value(&mut self, params: &BsdfParams) -> Value {
        Value::Shader(self.workspace.push_single(params))
    }

    fn principled(&mut self, id: &Id) -> Result<BsdfParams> {
        let node = self.node(id)?;
        // Blender's default is random walk; a node without the property (an authored graph) is that default.
        let method_token = node
            .properties
            .get("subsurface_method")
            .and_then(JsonValue::as_str)
            .unwrap_or("RANDOM_WALK")
            .to_owned();
        if node
            .properties
            .get("potter_simple_material")
            .and_then(JsonValue::as_bool)
            == Some(true)
        {
            let mut params = self.closure_params();
            params.transmission = self.material.transmission;
            params.base_color = textured_base(self.material, &self.hit_ctx, params.base_color)?;
            params.glass_transmission_tint = principled_glass_tint(params.base_color);
            // A simple-material graph holds no Alpha input unless the overlay wrote one (`write_principled_inputs`, for a
            // non-opaque material). Without it the material's own alpha applies, as before the closure change: its base
            // colour alpha, or 1 for an OPAQUE material, whose graph carries no transparency.
            let default_alpha = if self.material.alpha_mode == "opaque" {
                1.0
            } else {
                self.material.base_color[3]
            };
            params.alpha = self.scalar_input(id, &["Alpha"], default_alpha)?;
            return Ok(params);
        }
        let mut params = self.closure_params();
        let base = self.color_input(id, &["Base Color"], [0.8, 0.8, 0.8, 1.0])?;
        params.base_color = textured_base(self.material, &self.hit_ctx, base)?;
        params.glass_transmission_tint = principled_glass_tint(params.base_color);
        params.metallic = self.scalar_input(id, &["Metallic"], 0.0)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.5)?;
        params.ior = self.scalar_input(id, &["IOR"], 1.5)?;
        params.alpha = self.scalar_input(id, &["Alpha"], 1.0)?;
        params.transmission =
            self.scalar_input(id, &["Transmission Weight", "Transmission"], 0.0)?;
        params.subsurface_weight =
            self.scalar_input(id, &["Subsurface Weight", "Subsurface"], 0.0)?;
        params.thin_wall = self.scalar_input(id, &["Thin Wall"], 0.0)? > 0.5;
        if !params.thin_wall {
            params.subsurface_radius = self
                .vector_input(id, &["Subsurface Radius"], DVec3::new(1.0, 0.2, 0.1))?
                .to_array();
            params.subsurface_scale = self.scalar_input(id, &["Subsurface Scale"], 0.005)?;
            params.subsurface_ior = self.scalar_input(id, &["Subsurface IOR"], 1.4)?;
        }
        let method = method_token.as_str();
        params.subsurface_method = SubsurfaceMethod::parse(method)
            .ok_or_else(|| invalid(format!("unknown Principled subsurface method `{method}`")))?;
        params.subsurface_anisotropy = self.scalar_input(id, &["Subsurface Anisotropy"], 0.0)?;
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
        params.anisotropy = self.scalar_input(id, &["Anisotropic"], 0.0)?;
        params.anisotropic_rotation = self.scalar_input(id, &["Anisotropic Rotation"], 0.0)?;
        params.anisotropic_tangent = self.anisotropic_tangent(id, params.anisotropy >= 1.0e-5)?;
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
            emission.map(|channel| channel * strength)
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
        params.diffuse_roughness = params.roughness;
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
        let anisotropy = self.scalar_input(id, &["Anisotropy"], 0.0)?;
        if anisotropy.abs() > 1.0e-4 {
            let distribution = self
                .node(id)?
                .properties
                .get("distribution")
                .and_then(JsonValue::as_str)
                .unwrap_or("MULTI_GGX")
                .to_owned();
            if !matches!(distribution.as_str(), "GGX" | "MULTI_GGX") {
                return Err(unsupported_node(
                    id,
                    &format!("Glossy:distribution:{distribution}"),
                ));
            }
        }
        params.anisotropy = anisotropy;
        params.anisotropy_model = AnisotropyModel::Glossy;
        params.anisotropic_rotation = self.scalar_input(id, &["Rotation"], 0.0)?;
        params.anisotropic_tangent = self.anisotropic_tangent(id, anisotropy.abs() > 1.0e-4)?;
        Ok(params)
    }

    fn glass(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.transmission = 1.0;
        params.ior = self.scalar_input(id, &["IOR"], 1.5)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.0)?;
        // Cycles' Glass closure takes its Color as both the reflection and the transmission tint.
        let color = [
            params.base_color[0],
            params.base_color[1],
            params.base_color[2],
        ];
        params.glass_reflection_tint = color;
        params.glass_transmission_tint = color;
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
        Ok(params)
    }
    fn mix_shader(&mut self, id: &Id) -> Result<ClosureSpan> {
        let factor = self.scalar_input(id, &["Fac"], 0.5)?;
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(self.workspace.mix(first, second, factor))
    }

    fn add_shader(&mut self, id: &Id) -> Result<ClosureSpan> {
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(self.workspace.add(first, second))
    }

    fn shader_input(&mut self, id: &Id, names: &[&str]) -> Result<ClosureSpan> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Shader"),
            ValueKind::Shader,
        )? {
            Value::Shader(span) => Ok(span),
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
        if self.hit_ctx.opacity_only {
            return Ok(Value::Vector(self.hit_ctx.normal));
        }
        let normal = if self.is_linked(id, "Normal") {
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?
        } else {
            self.hit_ctx.normal
        };
        // An unlinked Height is a true no-op in both render engines, including a linked Normal.
        if !self.is_linked(id, "Height") {
            return Ok(Value::Vector(normal));
        }
        let strength = self.scalar_input(id, &["Strength"], 1.0)?.max(0.0);
        let distance = self.scalar_input(id, &["Distance"], 0.001)?;
        let invert = self
            .node(id)?
            .properties
            .get("invert")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let scale = if invert { -distance } else { distance };
        let filter_width = self.scalar_input(id, &["Filter Width"], 0.1)?;
        if self.hit_ctx.path_traced {
            if let Some(footprint) = self.hit_ctx.footprint {
                self.footprint_bump(id, normal, strength, scale, filter_width, footprint)
            } else {
                Err(unsupported_node(id, "Bump.path_footprint"))
            }
        } else if let Some(differentials) = self.hit_ctx.realtime_differentials {
            self.realtime_bump(id, normal, strength, scale, filter_width, differentials)
        } else {
            Err(unsupported_node(id, "Bump.realtime_derivatives"))
        }
    }

    /// EEVEE's `node_bump` (`gpu_shader_material_bump.glsl`): finite-difference the height along the two
    /// screen derivatives, then use the world-space surface gradient and signed determinant.
    fn realtime_bump(
        &mut self,
        id: &Id,
        normal: DVec3,
        strength: f64,
        scale: f64,
        filter_width: f64,
        differentials: HitDifferentials,
    ) -> Result<Value> {
        let normal = normalize_or(normal, DVec3::Z);
        let dp_x = differentials.world_dx;
        let dp_y = differentials.world_dy;
        let rx = dp_y.cross(normal);
        let ry = normal.cross(dp_x);
        let det = dp_x.dot(rx);
        let context = self.hit_ctx;
        let center = self.height_at(id, &context, false)?;
        let shifted_x = context
            .shifted_realtime(SurfaceStep {
                world: dp_x * filter_width,
                uv: [
                    differentials.uv_dx[0] * filter_width,
                    differentials.uv_dx[1] * filter_width,
                ],
            })
            .ok_or_else(|| unsupported_node(id, "Bump.realtime_derivatives.object_transform"))?;
        let shifted_y = context
            .shifted_realtime(SurfaceStep {
                world: dp_y * filter_width,
                uv: [
                    differentials.uv_dy[0] * filter_width,
                    differentials.uv_dy[1] * filter_width,
                ],
            })
            .ok_or_else(|| unsupported_node(id, "Bump.realtime_derivatives.object_transform"))?;
        let height_x = self.height_at(id, &shifted_x, true)?;
        let height_y = self.height_at(id, &shifted_y, true)?;
        let surfgrad = (height_x - center) * rx + (height_y - center) * ry;
        let facing_scale = if self.hit_ctx.back_facing {
            -scale
        } else {
            scale
        };
        let raw = normal * (filter_width * det.abs()) - surfgrad * (facing_scale * det.signum());
        let bumped = normalize_or(raw, normal);
        Ok(Value::Vector(normalize_or(
            normal.lerp(bumped, strength),
            normal,
        )))
    }

    /// Cycles' `svm_node_set_bump` (`kernel/svm/displace.h`). The screen differentials `dP/dx` and `dP/dy` are
    /// `differential_from_compact(Ng, dP)` (`kernel/util/differential.h`): the compact footprint along the two
    /// axes of Ng's orthonormal basis. The height is taken at the centre and one filter width along each
    /// differential, and the normal is `filter_width |det| N - scale sign(det) surfgrad`, where
    /// `det = dPdx · (dPdy × N)` and `surfgrad = (h_x - h_c)(dPdy × N) + (h_y - h_c)(N × dPdx)`.
    fn footprint_bump(
        &mut self,
        id: &Id,
        normal: DVec3,
        strength: f64,
        scale: f64,
        filter_width: f64,
        footprint: HitFootprint,
    ) -> Result<Value> {
        let (axis_x, axis_y) = make_orthonormals(footprint.geometric_normal);
        let dp_x = axis_x * footprint.dp;
        let dp_y = axis_y * footprint.dp;
        let rx = dp_y.cross(normal);
        let ry = normal.cross(dp_x);
        let det = dp_x.dot(rx);
        let context = self.hit_ctx;
        let center = self.height_at(id, &context, false)?;
        let shifted_x = context.shifted(SurfaceStep {
            world: dp_x * filter_width,
            uv: [0.0; 2],
        });
        let shifted_y = context.shifted(SurfaceStep {
            world: dp_y * filter_width,
            uv: [0.0; 2],
        });
        let height_x = self.height_at(id, &shifted_x, true)?;
        let height_y = self.height_at(id, &shifted_y, true)?;
        let surfgrad = (height_x - center) * rx + (height_y - center) * ry;
        // Cycles' `signf`, which maps zero to +1.
        let sign = if det < 0.0 { -1.0 } else { 1.0 };
        let raw = normal * (filter_width * det.abs()) - surfgrad * (scale * sign);
        let bent = raw.normalize_or_zero();
        if bent == DVec3::ZERO {
            return Ok(Value::Vector(normal));
        }
        Ok(Value::Vector(normalize_or(
            bent * strength + normal * (1.0 - strength),
            normal,
        )))
    }
    /// Evaluates the Height input at a shifted surface context, restoring the current hit even when evaluation fails.
    fn height_at(
        &mut self,
        id: &Id,
        context: &HitContext<'registry>,
        shifted: bool,
    ) -> Result<f64> {
        let previous = self.hit_ctx;
        self.hit_ctx = *context;
        self.hit_ctx.bump_height_shifted = shifted;
        let result = self.scalar_input(id, &["Height"], 0.0);
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
        if !matches!(socket, "Color" | "Alpha") {
            return Err(invalid(format!("unknown ColorRamp output `{socket}`")));
        }
        let factor = self.scalar_input(id, &["Fac"], 0.5)? as f32;
        let index = self.prepared_ramp(id)?;
        let ramp = &self.workspace.ramps[index];
        let table = if self.math_semantics == crate::node_kernels::MathSemantics::Eevee {
            &ramp.eevee_table
        } else {
            &ramp.table
        };
        let result = crate::node_kernels::shader_color_ramp(
            &ramp.elements,
            table,
            factor,
            ramp.interpolation,
            ramp.color_mode,
            self.math_semantics,
        )
        .ok_or_else(|| unsupported_node(id, "ShaderNodeValToRGB.color_ramp"))?;
        if result.iter().any(|component| !component.is_finite()) {
            return Err(eval_error(format!(
                "Color Ramp node `{id}` produced a non-finite result"
            )));
        }
        let result = result.map(f64::from);
        match socket {
            "Color" => Ok(Value::Color(result)),
            "Alpha" => Ok(Value::Scalar(result[3])),
            _ => unreachable!(),
        }
    }

    /// `ShaderNodeMix` and the legacy `ShaderNodeMixRGB`.
    fn mix_rgb(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let (generic, data_type, blend, factor_mode, clamp_factor, clamp_result) = {
            let node = self.node(id)?;
            let generic = node.node_type == "ShaderNodeMix";
            let data_type = if generic {
                node_setting_string(
                    id,
                    &node.properties,
                    "data_type",
                    "FLOAT",
                    "ShaderNodeMix",
                    MIX_DATA_TYPES,
                )?
            } else {
                "RGBA"
            };
            let blend = if !generic || data_type == "RGBA" {
                node_setting_string(
                    id,
                    &node.properties,
                    "blend_type",
                    "MIX",
                    if generic {
                        "ShaderNodeMix"
                    } else {
                        "ShaderNodeMixRGB"
                    },
                    MIX_BLEND_TYPES,
                )?
            } else {
                "MIX"
            };
            let factor_mode = if generic && data_type == "VECTOR" {
                node_setting_string(
                    id,
                    &node.properties,
                    "factor_mode",
                    "UNIFORM",
                    "ShaderNodeMix",
                    MIX_FACTOR_MODES,
                )?
            } else {
                "UNIFORM"
            };
            let clamp_factor = if generic {
                node_setting_bool(id, &node.properties, "clamp_factor", true, "ShaderNodeMix")?
            } else {
                true
            };
            let clamp_result = if generic {
                node_setting_bool(id, &node.properties, "clamp_result", false, "ShaderNodeMix")?
            } else {
                node_setting_bool(id, &node.properties, "use_clamp", false, "ShaderNodeMixRGB")?
            };
            (
                generic,
                data_type,
                blend,
                factor_mode,
                clamp_factor,
                clamp_result,
            )
        };
        let (factor_socket, first_socket, second_socket, result_socket) = match (generic, data_type)
        {
            (false, _) => ("Fac", "Color1", "Color2", "Color"),
            (true, "FLOAT") => ("Factor_Float", "A_Float", "B_Float", "Result_Float"),
            (true, "VECTOR") => ("Factor_Float", "A_Vector", "B_Vector", "Result_Vector"),
            (true, "RGBA") => ("Factor_Float", "A_Color", "B_Color", "Result_Color"),
            _ => unreachable!(),
        };
        if socket != result_socket {
            return Err(invalid(format!("unknown Mix output `{socket}`")));
        }
        let factor_default = if generic { 1.0 } else { 0.5 };
        let result = match data_type {
            "FLOAT" => {
                let factor = self.scalar_input(id, &[factor_socket], factor_default)? as f32;
                let first = self.scalar_input(id, &[first_socket], 0.0)? as f32;
                let second = self.scalar_input(id, &[second_socket], 0.0)? as f32;
                let result = crate::node_kernels::mix_float(
                    "MIX",
                    factor,
                    first,
                    second,
                    clamp_factor,
                    clamp_result,
                )
                .ok_or_else(|| unsupported_node(id, "ShaderNodeMix.data_type.FLOAT"))?;
                if !result.is_finite() {
                    return Err(eval_error(format!(
                        "Mix node `{id}` produced a non-finite result"
                    )));
                }
                Value::Scalar(f64::from(result))
            }
            "VECTOR" => {
                let factor = match factor_mode {
                    "UNIFORM" => {
                        let factor =
                            self.scalar_input(id, &["Factor_Float"], factor_default)? as f32;
                        [factor; 3]
                    }
                    "NON_UNIFORM" => self
                        .vector_input(id, &["Factor_Vector"], DVec3::splat(0.5))?
                        .to_array()
                        .map(|component| component as f32),
                    other => {
                        return Err(unsupported_node(
                            id,
                            &format!("ShaderNodeMix.factor_mode.{other}"),
                        ));
                    }
                };
                let first = self
                    .vector_input(id, &[first_socket], DVec3::ZERO)?
                    .to_array()
                    .map(|component| component as f32);
                let second = self
                    .vector_input(id, &[second_socket], DVec3::ZERO)?
                    .to_array()
                    .map(|component| component as f32);
                let result = crate::node_kernels::mix_vector(
                    "MIX",
                    factor,
                    first,
                    second,
                    clamp_factor,
                    clamp_result,
                )
                .ok_or_else(|| unsupported_node(id, "ShaderNodeMix.data_type.VECTOR"))?;
                if result.iter().any(|component| !component.is_finite()) {
                    return Err(eval_error(format!(
                        "Mix node `{id}` produced a non-finite result"
                    )));
                }
                Value::Vector(DVec3::from_array(result.map(f64::from)))
            }
            "RGBA" => {
                let factor = self.scalar_input(id, &[factor_socket], factor_default)? as f32;
                let first = self
                    .color_input(id, &[first_socket], [0.5, 0.5, 0.5, 1.0])?
                    .map(|component| component as f32);
                let second = self
                    .color_input(id, &[second_socket], [0.5, 0.5, 0.5, 1.0])?
                    .map(|component| component as f32);
                let result = crate::node_kernels::mix_color(
                    blend,
                    factor,
                    first,
                    second,
                    clamp_factor || !generic,
                    clamp_result,
                )
                .ok_or_else(|| {
                    unsupported_node(
                        id,
                        &format!(
                            "{}.blend_type.{blend}",
                            if generic {
                                "ShaderNodeMix"
                            } else {
                                "ShaderNodeMixRGB"
                            }
                        ),
                    )
                })?;
                if result.iter().any(|component| !component.is_finite()) {
                    return Err(eval_error(format!(
                        "Mix node `{id}` produced a non-finite result"
                    )));
                }
                Value::Color(result.map(f64::from))
            }
            _ => unreachable!(),
        };
        Ok(result)
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
    fn value(&self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Value" {
            return Err(invalid(format!("unknown Value output `{socket}`")));
        }
        let value = match self.required_output_default(id, "ShaderNodeValue", socket)? {
            Some(value) => JsonValue::as_f64(value)
                .ok_or_else(|| unsupported_node(id, "ShaderNodeValue.output_defaults"))?,
            None => 0.0,
        };
        if !value.is_finite() {
            return Err(invalid(format!(
                "Value node `{id}` has a non-finite output"
            )));
        }
        Ok(Value::Scalar(value))
    }

    fn rgb(&self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown RGB output `{socket}`")));
        }
        let color = match self.required_output_default(id, "ShaderNodeRGB", socket)? {
            Some(value) => {
                let color = color(value, [f64::NAN; 4]);
                if color.iter().any(|channel| !channel.is_finite()) {
                    return Err(unsupported_node(id, "ShaderNodeRGB.output_defaults"));
                }
                color
            }
            None => [0.5, 0.5, 0.5, 1.0],
        };
        Ok(Value::Color(color))
    }

    fn combine_color(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown Combine Color output `{socket}`")));
        }
        let mode = match self
            .node(id)?
            .properties
            .get("mode")
            .and_then(JsonValue::as_str)
        {
            None | Some("RGB") => "RGB",
            Some("HSV") => "HSV",
            Some("HSL") => "HSL",
            Some(mode) => return Err(unsupported_node(id, &format!("CombineColor:{mode}"))),
        };
        let red = self.scalar_input(id, &["Red"], 0.0)?;
        let green = self.scalar_input(id, &["Green"], 0.0)?;
        let blue = self.scalar_input(id, &["Blue"], 0.0)?;
        super::converters::combine_color(mode, red, green, blue)
            .map(Value::Color)
            .ok_or_else(|| unsupported_node(id, &format!("CombineColor:{mode}")))
    }

    fn rgb_to_bw(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Val" {
            return Err(invalid(format!("unknown RGB to BW output `{socket}`")));
        }
        let color = self.color_input(id, &["Color"], [0.5, 0.5, 0.5, 1.0])?;
        Ok(Value::Scalar(super::converters::rgb_to_bw(color)))
    }

    fn invert(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown Invert Color output `{socket}`")));
        }
        let factor = self.scalar_input(id, &["Fac", "Factor"], 1.0)?;
        let color = self.color_input(id, &["Color"], [0.0, 0.0, 0.0, 1.0])?;
        Ok(Value::Color(super::converters::invert_color(color, factor)))
    }

    fn gamma(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown Gamma output `{socket}`")));
        }
        let color = self.color_input(id, &["Color"], [1.0; 4])?;
        let gamma = self.scalar_input(id, &["Gamma"], 1.0)?;
        Ok(Value::Color(super::converters::gamma_color(color, gamma)))
    }

    fn bright_contrast(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!(
                "unknown Brightness/Contrast output `{socket}`"
            )));
        }
        let color = self.color_input(id, &["Color"], [1.0; 4])?;
        let brightness = self.scalar_input(id, &["Bright", "Brightness"], 0.0)?;
        let contrast = self.scalar_input(id, &["Contrast"], 0.0)?;
        Ok(Value::Color(super::converters::brightness_contrast(
            color, brightness, contrast,
        )))
    }

    fn hue_saturation(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!(
                "unknown Hue/Saturation/Value output `{socket}`"
            )));
        }
        let hue = self.scalar_input(id, &["Hue"], 0.5)?;
        let saturation = self.scalar_input(id, &["Saturation"], 1.0)?;
        let value = self.scalar_input(id, &["Value"], 1.0)?;
        let factor = self.scalar_input(id, &["Fac", "Factor"], 1.0)?;
        let color = self.color_input(id, &["Color"], [0.8, 0.8, 0.8, 1.0])?;
        let mut result = super::converters::hue_saturation(color, hue, saturation, value, factor);
        if self.hit_ctx.path_traced {
            result[..3]
                .iter_mut()
                .for_each(|channel| *channel = channel.max(0.0));
        }
        Ok(Value::Color(result))
    }

    fn normal(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let direction = match self.required_output_default(id, "ShaderNodeNormal", "Normal")? {
            Some(value) => {
                let direction = vector(value, DVec3::splat(f64::NAN));
                if !direction.is_finite() {
                    return Err(unsupported_node(id, "ShaderNodeNormal.output_defaults"));
                }
                direction
            }
            None => DVec3::Z,
        };
        match socket {
            "Normal" => Ok(Value::Vector(direction)),
            "Dot" => {
                let normal = self.vector_input(id, &["Normal"], DVec3::Z)?;
                Ok(Value::Scalar(normalize_or(normal, DVec3::Z).dot(direction)))
            }
            _ => Err(invalid(format!("unknown Normal output `{socket}`"))),
        }
    }

    fn wavelength(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown Wavelength output `{socket}`")));
        }
        let wavelength = self.scalar_input(id, &["Wavelength"], 500.0)? as f32;
        let [red, green, blue] = if self.hit_ctx.path_traced {
            super::converters::wavelength_rgb(wavelength)
        } else {
            super::converters::eevee_wavelength_rgb(wavelength)
        };
        Ok(Value::Color([
            f64::from(red),
            f64::from(green),
            f64::from(blue),
            1.0,
        ]))
    }

    fn blackbody(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown Blackbody output `{socket}`")));
        }
        let temperature = self.scalar_input(id, &["Temperature"], 6500.0)? as f32;
        let [red, green, blue] = if self.hit_ctx.path_traced {
            super::blackbody::blackbody_rgb(temperature).map(|channel| channel.max(0.0))
        } else {
            super::converters::eevee_blackbody_rgb(temperature)
        };
        Ok(Value::Color([
            f64::from(red.max(0.0)),
            f64::from(green.max(0.0)),
            f64::from(blue.max(0.0)),
            1.0,
        ]))
    }

    fn light_falloff(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if !matches!(socket, "Quadratic" | "Linear" | "Constant") {
            return Err(invalid(format!("unknown Light Falloff output `{socket}`")));
        }
        let strength = self.scalar_input(id, &["Strength"], 100.0)? as f32;
        let Some(distance) = self.hit_ctx.light_falloff_distance else {
            if self.hit_ctx.path_traced {
                return Err(unsupported_node(
                    id,
                    "ShaderNodeLightFalloff.path_ray_length",
                ));
            }
            // EEVEE's `node_light_falloff` outputs Strength for all falloff sockets.
            return Ok(Value::Scalar(f64::from(strength)));
        };
        let distance = distance as f32;
        let mut result = match socket {
            "Quadratic" => strength,
            "Linear" => strength * distance,
            "Constant" => strength * distance * distance,
            _ => return Err(invalid(format!("unknown Light Falloff output `{socket}`"))),
        };
        let smooth = self.scalar_input(id, &["Smooth"], 0.0)? as f32;
        if smooth > 0.0 {
            let squared = distance * distance;
            result *= squared / (smooth + squared);
        }
        if !result.is_finite() {
            return Err(eval_error(format!(
                "Light Falloff node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Scalar(f64::from(result)))
    }

    fn object_info(&self, id: &Id, socket: &str) -> Result<Value> {
        match socket {
            "Location" => {
                if !self.hit_ctx.object_to_world.is_finite() {
                    return Err(unsupported_node(
                        id,
                        "ShaderNodeObjectInfo.Location.object_matrix",
                    ));
                }
                Ok(Value::Vector(
                    self.hit_ctx.object_to_world.w_axis.truncate(),
                ))
            }
            "Color" => Ok(Value::Color(self.hit_ctx.object_color)),
            "Alpha" => Ok(Value::Scalar(self.hit_ctx.object_color[3])),
            "Object Index" => Ok(Value::Scalar(self.hit_ctx.object_pass_index)),
            "Material Index" => self
                .hit_ctx
                .material_pass_index
                .map(Value::Scalar)
                .ok_or_else(|| {
                    unsupported_node(
                        id,
                        "ShaderNodeObjectInfo.Material Index.material_pass_index",
                    )
                }),
            "Random" => self
                .hit_ctx
                .object_random_number
                .map(Value::Scalar)
                .ok_or_else(|| {
                    unsupported_node(id, "ShaderNodeObjectInfo.Random.object_random_number")
                }),
            _ => Err(invalid(format!("unknown Object Info output `{socket}`"))),
        }
    }

    fn geometry(&self, id: &Id, socket: &str) -> Result<Value> {
        if self.hit_ctx.bump_height_shifted
            && (matches!(socket, "Incoming" | "Tangent")
                || (socket == "Normal" && !self.hit_ctx.path_traced))
        {
            return Err(unsupported_shifted_bump_context(
                id,
                "ShaderNodeNewGeometry",
                socket,
            ));
        }
        match socket {
            "Position" => Ok(Value::Vector(self.hit_ctx.position)),
            "Normal" => Ok(Value::Vector(self.hit_ctx.normal)),
            // Both SVM `shading_incoming` and EEVEE `coordinate_incoming` point toward the viewer.
            "Incoming" => Ok(Value::Vector(self.hit_ctx.view_direction)),
            "True Normal" => Ok(Value::Vector(self.hit_ctx.geometric_normal)),
            "Tangent" if !self.hit_ctx.path_traced => {
                let point = self.hit_ctx.generated;
                let tangent = DVec3::new(-(point.y - 0.5), point.x - 0.5, 0.0);
                let tangent = self.hit_ctx.object_to_world.transform_vector3(tangent);
                surface_tangent(tangent, self.hit_ctx.normal)
                    .map(Value::Vector)
                    .ok_or_else(|| {
                        unsupported_node(id, "ShaderNodeNewGeometry.Tangent.orco_attribute")
                    })
            }
            "Tangent" => {
                let Some(layers) = self.hit_ctx.uv_layers else {
                    return Err(unsupported_node(
                        id,
                        "ShaderNodeNewGeometry.Tangent.triangle_positions",
                    ));
                };
                if self.hit_ctx.generated_corners.is_some() {
                    return self.radial_tangent(id, "Z").map(Value::Vector);
                }

                let [first, second, third] = layers.positions;
                let mut tangent = second - first;
                if (second - first)
                    .cross(third - first)
                    .dot(self.hit_ctx.view_direction)
                    < 0.0
                {
                    tangent = -tangent;
                }
                Ok(Value::Vector(tangent.normalize_or_zero()))
            }
            "Parametric" => self
                .hit_ctx
                .uv_layers
                .map(|layers| {
                    Value::Vector(DVec3::new(
                        1.0 - layers.barycentric[1] - layers.barycentric[2],
                        layers.barycentric[1],
                        0.0,
                    ))
                })
                .ok_or_else(|| {
                    unsupported_node(id, "ShaderNodeNewGeometry.Parametric.barycentric")
                }),
            "Backfacing" => Ok(Value::Scalar(if self.hit_ctx.back_facing {
                1.0
            } else {
                0.0
            })),
            "Pointiness" if !self.hit_ctx.path_traced => Ok(Value::Scalar(0.5)),
            "Random Per Island" if !self.hit_ctx.path_traced => Ok(Value::Scalar(0.0)),
            "Pointiness" => Err(unsupported_node(
                id,
                "ShaderNodeNewGeometry.Pointiness.pointiness_attribute",
            )),
            "Random Per Island" => Err(unsupported_node(
                id,
                "ShaderNodeNewGeometry.Random Per Island.random_per_island_attribute",
            )),
            _ => Err(invalid(format!("unknown Geometry output `{socket}`"))),
        }
    }

    fn camera_data(&self, id: &Id, socket: &str) -> Result<Value> {
        if !matches!(socket, "View Vector" | "View Z Depth" | "View Distance") {
            return Err(invalid(format!("unknown Camera Data output `{socket}`")));
        }
        if self.hit_ctx.path_traced {
            let position = if self.hit_ctx.bump_height_shifted {
                self.hit_ctx.bump_center_position
            } else {
                self.hit_ctx.position
            };
            let camera_point = self
                .hit_ctx
                .world_to_camera
                .filter(DMat4::is_finite)
                .map(|matrix| matrix.transform_point3(position))
                .filter(|point| point.is_finite())
                .ok_or_else(|| {
                    unsupported_node(id, &format!("ShaderNodeCameraData.{socket}.camera_context"))
                })?;
            return match socket {
                "View Vector" => Ok(Value::Vector(camera_point.normalize_or_zero())),
                "View Z Depth" => Ok(Value::Scalar(-camera_point.z)),
                "View Distance" => Ok(Value::Scalar(camera_point.length())),
                _ => Err(invalid(format!("unknown Camera Data output `{socket}`"))),
            };
        }
        let camera_position = if self.hit_ctx.bump_height_shifted {
            self.hit_ctx
                .bump_center_realtime_position_f32
                .unwrap_or_else(|| {
                    let position = self.hit_ctx.bump_center_position;
                    Vec3::new(position.x as f32, position.y as f32, position.z as f32)
                })
        } else {
            self.hit_ctx.realtime_position_f32.unwrap_or_else(|| {
                let position = self.hit_ctx.position;
                Vec3::new(position.x as f32, position.y as f32, position.z as f32)
            })
        };
        let camera_point = self
            .hit_ctx
            .world_to_camera_f32
            .filter(Mat4::is_finite)
            .map(|matrix| matrix.transform_point3(camera_position))
            .filter(|point| point.is_finite())
            .ok_or_else(|| {
                unsupported_node(id, &format!("ShaderNodeCameraData.{socket}.camera_context"))
            })?;
        match socket {
            "View Vector" => {
                let view = Vec3::new(camera_point.x, camera_point.y, -camera_point.z).normalize();
                Ok(Value::Vector(DVec3::new(
                    f64::from(view.x),
                    f64::from(view.y),
                    f64::from(view.z),
                )))
            }
            "View Z Depth" => Ok(Value::Scalar(f64::from(camera_point.z.abs()))),
            "View Distance" => Ok(Value::Scalar(f64::from(camera_point.length()))),
            _ => Err(invalid(format!("unknown Camera Data output `{socket}`"))),
        }
    }

    fn math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Value" {
            return Err(invalid(format!("unknown Math output `{socket}`")));
        }
        let operation = {
            let node = self.node(id)?;
            node_setting_string(
                id,
                &node.properties,
                "operation",
                "ADD",
                "ShaderNodeMath",
                SHADER_MATH_OPERATIONS,
            )?
        };
        let first = self.scalar_input(id, &["Value"], 0.5)? as f32;
        let second = self.scalar_input(id, &["Value_001"], 0.5)? as f32;
        let third = self.scalar_input(id, &["Value_002"], 0.5)? as f32;
        let clamp = {
            let node = self.node(id)?;
            node_setting_bool(id, &node.properties, "use_clamp", false, "ShaderNodeMath")?
        };
        let math_semantics = if self.inputs_fold_to_primitives(id) {
            crate::node_kernels::MathSemantics::Geometry
        } else {
            self.math_semantics
        };
        if math_semantics == crate::node_kernels::MathSemantics::Geometry
            && operation == "POWER"
            && first == 0.0
            && second < 0.0
            && !clamp
        {
            return Err(unsupported_node(
                id,
                "ShaderNodeMath.operation.POWER.non_finite_constant_fold",
            ));
        }
        if math_semantics == crate::node_kernels::MathSemantics::Eevee {
            let undefined = match operation {
                "INV_SQRT" if first <= 0.0 => {
                    Some("ShaderNodeMath.operation.INV_SQRT.glsl_undefined_domain")
                }
                "INVERSE_SQRT" if first <= 0.0 => {
                    Some("ShaderNodeMath.operation.INVERSE_SQRT.glsl_undefined_domain")
                }
                "LOGARITHM" if first > 0.0 && second.total_cmp(&1.0).is_eq() => {
                    Some("ShaderNodeMath.operation.LOGARITHM.glsl_undefined_domain")
                }
                _ => None,
            };
            if let Some(kind) = undefined {
                return Err(unsupported_node(id, kind));
            }
        }
        let result = if math_semantics == crate::node_kernels::MathSemantics::Eevee
            && operation == "MODULO"
        {
            eevee_modulo_result(first, second)
                .map(|value| if clamp { value.clamp(0.0, 1.0) } else { value })
                .ok_or_else(|| {
                    unsupported_node(id, "ShaderNodeMath.operation.MODULO.quotient_out_of_range")
                })?
        } else {
            crate::node_kernels::math(operation, first, second, third, clamp, math_semantics)
                .ok_or_else(|| {
                    unsupported_node(id, &format!("ShaderNodeMath.operation.{operation}"))
                })?
        };
        if !result.is_finite() {
            return Err(eval_error(format!(
                "Math node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Scalar(f64::from(result)))
    }
    fn vector_math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let operation = {
            let node = self.node(id)?;
            node_setting_string(
                id,
                &node.properties,
                "operation",
                "ADD",
                "ShaderNodeVectorMath",
                VECTOR_MATH_OPERATIONS,
            )?
        };
        let scalar_result = matches!(operation, "DOT_PRODUCT" | "DISTANCE" | "LENGTH");
        let output = match socket {
            "Value" if scalar_result => "Value",
            "Vector" if !scalar_result => "Vector",
            "Value" | "Vector" => {
                return Err(invalid(format!(
                    "Vector Math operation `{operation}` has no `{socket}` output"
                )));
            }
            _ => return Err(invalid(format!("unknown Vector Math output `{socket}`"))),
        };
        let first = self
            .vector_input(id, &["Vector"], DVec3::ZERO)?
            .to_array()
            .map(|component| component as f32);
        let second = self
            .vector_input(id, &["Vector_001"], DVec3::ZERO)?
            .to_array()
            .map(|component| component as f32);
        let third = self
            .vector_input(id, &["Vector_002"], DVec3::ZERO)?
            .to_array()
            .map(|component| component as f32);
        let scale = self.scalar_input(id, &["Scale"], 1.0)? as f32;
        let math_semantics = if self.inputs_fold_to_primitives(id) {
            crate::node_kernels::MathSemantics::Geometry
        } else {
            self.math_semantics
        };
        let result = match (math_semantics, operation) {
            (crate::node_kernels::MathSemantics::Eevee, "MODULO") => {
                let mut value = [0.0; 3];
                for index in 0..3 {
                    let Some(result) = eevee_modulo_result(first[index], second[index]) else {
                        return Err(unsupported_node(
                            id,
                            "ShaderNodeVectorMath.operation.MODULO.quotient_out_of_range",
                        ));
                    };
                    value[index] = result;
                }
                crate::node_kernels::KernelValue::Vector(value)
            }
            (crate::node_kernels::MathSemantics::Eevee, "NORMALIZE") => {
                crate::node_kernels::KernelValue::Vector(
                    crate::node_kernels::eevee_normalize(first).ok_or_else(|| {
                        unsupported_node(
                            id,
                            "ShaderNodeVectorMath.operation.NORMALIZE.subnormal_input",
                        )
                    })?,
                )
            }
            _ => crate::node_kernels::vector_math(
                operation,
                output,
                first,
                second,
                third,
                scale,
                math_semantics,
            )
            .ok_or_else(|| {
                unsupported_node(id, &format!("ShaderNodeVectorMath.operation.{operation}"))
            })?,
        };
        match result {
            crate::node_kernels::KernelValue::Scalar(value) if value.is_finite() => {
                Ok(Value::Scalar(f64::from(value)))
            }
            crate::node_kernels::KernelValue::Vector(value)
                if value.iter().all(|component| component.is_finite()) =>
            {
                Ok(Value::Vector(DVec3::from_array(value.map(f64::from))))
            }
            _ => Err(eval_error(format!(
                "Vector Math node `{id}` produced a non-finite result"
            ))),
        }
    }
    fn map_range(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let (data_type, interpolation, clamp) = {
            let node = self.node(id)?;
            (
                node_setting_string(
                    id,
                    &node.properties,
                    "data_type",
                    "FLOAT",
                    "ShaderNodeMapRange",
                    MAP_RANGE_DATA_TYPES,
                )?,
                node_setting_string(
                    id,
                    &node.properties,
                    "interpolation_type",
                    "LINEAR",
                    "ShaderNodeMapRange",
                    MAP_RANGE_INTERPOLATIONS,
                )?,
                node_setting_bool(id, &node.properties, "clamp", true, "ShaderNodeMapRange")?,
            )
        };
        let vector = match data_type {
            "FLOAT" => false,
            "FLOAT_VECTOR" => true,
            _ => unreachable!(),
        };
        let result_socket = if vector { "Vector" } else { "Result" };
        if socket != result_socket {
            return Err(invalid(format!(
                "unknown Map Range output `{socket}` for {data_type}"
            )));
        }
        let (value, from_min, from_max, to_min, to_max, steps) = if vector {
            let value = self
                .vector_input(id, &["Vector"], DVec3::ZERO)?
                .to_array()
                .map(|component| component as f32);
            let from_min = self
                .vector_input(id, &["From_Min_FLOAT3"], DVec3::ZERO)?
                .to_array()
                .map(|component| component as f32);
            let from_max = self
                .vector_input(id, &["From_Max_FLOAT3"], DVec3::ONE)?
                .to_array()
                .map(|component| component as f32);
            let to_min = self
                .vector_input(id, &["To_Min_FLOAT3"], DVec3::ZERO)?
                .to_array()
                .map(|component| component as f32);
            let to_max = self
                .vector_input(id, &["To_Max_FLOAT3"], DVec3::ONE)?
                .to_array()
                .map(|component| component as f32);
            let steps = self
                .vector_input(id, &["Steps_FLOAT3"], DVec3::splat(4.0))?
                .to_array()
                .map(|component| component as f32);
            (
                crate::node_kernels::KernelValue::Vector(value),
                crate::node_kernels::KernelValue::Vector(from_min),
                crate::node_kernels::KernelValue::Vector(from_max),
                crate::node_kernels::KernelValue::Vector(to_min),
                crate::node_kernels::KernelValue::Vector(to_max),
                crate::node_kernels::KernelValue::Vector(steps),
            )
        } else {
            (
                crate::node_kernels::KernelValue::Scalar(
                    self.scalar_input(id, &["Value"], 1.0)? as f32
                ),
                crate::node_kernels::KernelValue::Scalar(self.scalar_input(
                    id,
                    &["From Min"],
                    0.0,
                )? as f32),
                crate::node_kernels::KernelValue::Scalar(self.scalar_input(
                    id,
                    &["From Max"],
                    1.0,
                )? as f32),
                crate::node_kernels::KernelValue::Scalar(
                    self.scalar_input(id, &["To Min"], 0.0)? as f32
                ),
                crate::node_kernels::KernelValue::Scalar(
                    self.scalar_input(id, &["To Max"], 1.0)? as f32
                ),
                crate::node_kernels::KernelValue::Scalar(
                    self.scalar_input(id, &["Steps"], 4.0)? as f32
                ),
            )
        };
        let math_semantics = if self.inputs_fold_to_primitives(id) {
            crate::node_kernels::MathSemantics::Geometry
        } else {
            self.math_semantics
        };
        let result = crate::node_kernels::shader_map_range(
            value,
            from_min,
            from_max,
            to_min,
            to_max,
            steps,
            interpolation,
            clamp,
            math_semantics,
        )
        .ok_or_else(|| unsupported_node(id, "ShaderNodeMapRange"))?;
        match result {
            crate::node_kernels::KernelValue::Scalar(value) if !vector && value.is_finite() => {
                Ok(Value::Scalar(f64::from(value)))
            }
            crate::node_kernels::KernelValue::Vector(value)
                if vector && value.iter().all(|component| component.is_finite()) =>
            {
                Ok(Value::Vector(DVec3::from_array(value.map(f64::from))))
            }
            _ => Err(eval_error(format!(
                "Map Range node `{id}` produced an invalid result"
            ))),
        }
    }

    fn clamp(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Result" {
            return Err(invalid(format!("unknown Clamp output `{socket}`")));
        }
        let mode = {
            let node = self.node(id)?;
            node_setting_string(
                id,
                &node.properties,
                "clamp_type",
                "MINMAX",
                "ShaderNodeClamp",
                CLAMP_TYPES,
            )?
        };
        let value = self.scalar_input(id, &["Value"], 1.0)? as f32;
        let minimum = self.scalar_input(id, &["Min"], 0.0)? as f32;
        let maximum = self.scalar_input(id, &["Max"], 1.0)? as f32;
        let result = crate::node_kernels::clamp(
            crate::node_kernels::KernelValue::Scalar(value),
            crate::node_kernels::KernelValue::Scalar(minimum),
            crate::node_kernels::KernelValue::Scalar(maximum),
            mode,
        )
        .ok_or_else(|| unsupported_node(id, &format!("ShaderNodeClamp.clamp_type.{mode}")))?;
        match result {
            crate::node_kernels::KernelValue::Scalar(value) if value.is_finite() => {
                Ok(Value::Scalar(f64::from(value)))
            }
            _ => Err(eval_error(format!(
                "Clamp node `{id}` produced a non-finite result"
            ))),
        }
    }

    fn float_curve(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Value" {
            return Err(invalid(format!("unknown Float Curve output `{socket}`")));
        }
        let factor = self.scalar_input(id, &["Factor"], 1.0)? as f32;
        let input = self.scalar_input(id, &["Value"], 1.0)? as f32;
        let index = self.prepared_curve(id)?;
        let curve = &self.workspace.curve_mappings[index];
        let mapped = crate::node_kernels::shader_curve_mapping_value(&curve.prepared, 0, input)
            .ok_or_else(|| unsupported_node(id, "ShaderNodeFloatCurve.mapping.curves"))?;
        let result = crate::node_kernels::mix_float("MIX", factor, input, mapped, false, false)
            .ok_or_else(|| unsupported_node(id, "ShaderNodeFloatCurve.mapping"))?;
        if !result.is_finite() {
            return Err(eval_error(format!(
                "Float Curve node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Scalar(f64::from(result)))
    }

    fn rgb_curve(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Color" {
            return Err(invalid(format!("unknown RGB Curve output `{socket}`")));
        }
        let factor = self.scalar_input(id, &["Fac"], 1.0)? as f32;
        let input = self
            .color_input(id, &["Color"], [1.0; 4])?
            .map(|component| component as f32);
        let index = self.prepared_curve(id)?;
        let curve = &self.workspace.curve_mappings[index];
        let mapped =
            crate::node_kernels::curve_mapping_rgb(&curve.prepared, [input[0], input[1], input[2]]);
        let result = crate::node_kernels::mix_color(
            "MIX",
            factor,
            [input[0], input[1], input[2], 1.0],
            [mapped[0], mapped[1], mapped[2], 1.0],
            false,
            false,
        )
        .ok_or_else(|| unsupported_node(id, "ShaderNodeRGBCurve.mapping"))?;
        if result.iter().any(|component| !component.is_finite()) {
            return Err(eval_error(format!(
                "RGB Curve node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Color(result.map(f64::from)))
    }
    fn vector_curve(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Vector" {
            return Err(invalid(format!("unknown Vector Curve output `{socket}`")));
        }
        let factor = self.scalar_input(id, &["Fac"], 1.0)? as f32;
        let input = self
            .vector_input(id, &["Vector"], DVec3::ZERO)?
            .to_array()
            .map(|component| component as f32);
        let index = self.prepared_curve(id)?;
        let curve = &self.workspace.curve_mappings[index];
        let mapped = crate::node_kernels::curve_mapping_vector(&curve.prepared, input);
        let result =
            crate::node_kernels::mix_vector("MIX", [factor; 3], input, mapped, false, false)
                .ok_or_else(|| unsupported_node(id, "ShaderNodeVectorCurve.mapping"))?;
        if result.iter().any(|component| !component.is_finite()) {
            return Err(eval_error(format!(
                "Vector Curve node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Vector(DVec3::from_array(result.map(f64::from))))
    }
    /// Blender 5.2.2 `node_shader_vector_rotate.cc`: rotate about the center; invert reverses
    /// the axis-angle direction or inverts the Euler rotation matrix.
    fn vector_rotate(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Vector" {
            return Err(invalid(format!("unknown Vector Rotate output `{socket}`")));
        }
        let vector = self.vector_input(id, &["Vector"], DVec3::ZERO)?;
        let center = self.vector_input(id, &["Center"], DVec3::ZERO)?;
        let (rotation_type, invert) = {
            let node = self.node(id)?;
            let rotation_type = match node
                .properties
                .get("rotation_type")
                .and_then(JsonValue::as_str)
                .unwrap_or("AXIS_ANGLE")
            {
                "AXIS_ANGLE" => VectorRotationType::AxisAngle,
                "X_AXIS" => VectorRotationType::AxisX,
                "Y_AXIS" => VectorRotationType::AxisY,
                "Z_AXIS" => VectorRotationType::AxisZ,
                "EULER_XYZ" => VectorRotationType::EulerXyz,
                other => return Err(unsupported_node(id, &format!("VectorRotate:{other}"))),
            };
            let invert = node
                .properties
                .get("invert")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            (rotation_type, invert)
        };
        let rotation = match rotation_type {
            VectorRotationType::AxisAngle => {
                let axis = self
                    .vector_input(id, &["Axis"], DVec3::Z)?
                    .normalize_or_zero();
                let angle = self.scalar_input(id, &["Angle"], 0.0)?;
                let angle = if invert { -angle } else { angle };
                if axis == DVec3::ZERO {
                    DQuat::IDENTITY
                } else {
                    DQuat::from_axis_angle(axis, angle)
                }
            }
            VectorRotationType::AxisX | VectorRotationType::AxisY | VectorRotationType::AxisZ => {
                let axis = match rotation_type {
                    VectorRotationType::AxisX => DVec3::X,
                    VectorRotationType::AxisY => DVec3::Y,
                    _ => DVec3::Z,
                };
                let angle = self.scalar_input(id, &["Angle"], 0.0)?;
                DQuat::from_axis_angle(axis, if invert { -angle } else { angle })
            }
            VectorRotationType::EulerXyz => {
                let euler = self.vector_input(id, &["Rotation"], DVec3::ZERO)?;
                let rotation = DQuat::from_euler(EulerRot::XYZ, euler.x, euler.y, euler.z);
                if invert {
                    rotation.conjugate()
                } else {
                    rotation
                }
            }
        };
        let rotated = rotation * (vector - center) + center;
        if !rotated.is_finite() {
            return Err(eval_error(format!(
                "Vector Rotate node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Vector(rotated))
    }

    /// Blender uses affine camera and object transforms for points/directions, and inverse-transpose
    /// transforms for normals.
    fn vector_transform(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Vector" {
            return Err(invalid(format!(
                "unknown Vector Transform output `{socket}`"
            )));
        }
        let vector = self.vector_input(id, &["Vector"], DVec3::splat(0.5))?;
        let (vector_type, from, to) = {
            let node = self.node(id)?;
            let vector_type = match node
                .properties
                .get("vector_type")
                .and_then(JsonValue::as_str)
                .unwrap_or("VECTOR")
            {
                "POINT" => VectorTransformType::Point,
                "VECTOR" => VectorTransformType::Vector,
                "NORMAL" => VectorTransformType::Normal,
                other => {
                    return Err(unsupported_node(
                        id,
                        &format!("VectorTransform:type:{other}"),
                    ));
                }
            };
            let from = match node
                .properties
                .get("convert_from")
                .and_then(JsonValue::as_str)
                .unwrap_or("WORLD")
            {
                "OBJECT" => VectorTransformSpace::Object,
                "WORLD" => VectorTransformSpace::World,
                "CAMERA" => VectorTransformSpace::Camera,
                other => {
                    return Err(unsupported_node(
                        id,
                        &format!("VectorTransform:convert_from:{other}"),
                    ));
                }
            };
            let to = match node
                .properties
                .get("convert_to")
                .and_then(JsonValue::as_str)
                .unwrap_or("OBJECT")
            {
                "OBJECT" => VectorTransformSpace::Object,
                "WORLD" => VectorTransformSpace::World,
                "CAMERA" => VectorTransformSpace::Camera,
                other => {
                    return Err(unsupported_node(
                        id,
                        &format!("VectorTransform:convert_to:{other}"),
                    ));
                }
            };
            (vector_type, from, to)
        };

        let transformed = if from == to {
            if vector_type == VectorTransformType::Normal {
                vector.normalize_or_zero()
            } else {
                vector
            }
        } else {
            let unsupported_camera = || {
                unsupported_node(
                    id,
                    &format!(
                        "VectorTransform:{}_to_{}:camera_matrices",
                        from.name(),
                        to.name()
                    ),
                )
            };
            let from_to_world = match from {
                VectorTransformSpace::Object => self.hit_ctx.object_to_world,
                VectorTransformSpace::World => DMat4::IDENTITY,
                VectorTransformSpace::Camera => self
                    .hit_ctx
                    .camera_to_world
                    .ok_or_else(unsupported_camera)?,
            };
            let world_to_to = match to {
                VectorTransformSpace::Object => self.hit_ctx.world_to_object,
                VectorTransformSpace::World => DMat4::IDENTITY,
                VectorTransformSpace::Camera => self
                    .hit_ctx
                    .world_to_camera
                    .ok_or_else(unsupported_camera)?,
            };
            let vector = if from == VectorTransformSpace::Camera {
                DVec3::new(vector.x, vector.y, -vector.z)
            } else {
                vector
            };
            let matrix = world_to_to * from_to_world;
            let normal_matrix = matrix.inverse().transpose();
            let matrix_context =
                if from == VectorTransformSpace::Object || to == VectorTransformSpace::Object {
                    "object_matrices"
                } else {
                    "camera_matrices"
                };
            let transformed = match vector_type {
                VectorTransformType::Point => {
                    if !matrix.is_finite() {
                        return Err(unsupported_node(
                            id,
                            &format!(
                                "VectorTransform:{}_to_{}:{matrix_context}",
                                from.name(),
                                to.name()
                            ),
                        ));
                    }
                    matrix.transform_point3(vector)
                }
                VectorTransformType::Vector => {
                    if !matrix.is_finite() {
                        return Err(unsupported_node(
                            id,
                            &format!(
                                "VectorTransform:{}_to_{}:{matrix_context}",
                                from.name(),
                                to.name()
                            ),
                        ));
                    }
                    matrix.transform_vector3(vector)
                }
                VectorTransformType::Normal => {
                    if !normal_matrix.is_finite() {
                        return Err(unsupported_node(
                            id,
                            &format!(
                                "VectorTransform:{}_to_{}:{matrix_context}",
                                from.name(),
                                to.name()
                            ),
                        ));
                    }
                    normal_matrix.transform_vector3(vector).normalize_or_zero()
                }
            };
            if to == VectorTransformSpace::Camera {
                DVec3::new(transformed.x, transformed.y, -transformed.z)
            } else {
                transformed
            }
        };
        if !transformed.is_finite() {
            return Err(eval_error(format!(
                "Vector Transform node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Vector(transformed))
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
    /// `ShaderNodeSeparateColor`: one color component in the node's selected color model.
    fn separate_color(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let mode = match self
            .node(id)?
            .properties
            .get("mode")
            .and_then(JsonValue::as_str)
        {
            None | Some("RGB") => "RGB",
            Some("HSV") => "HSV",
            Some("HSL") => "HSL",
            Some(mode) => return Err(unsupported_node(id, &format!("SeparateColor.{mode}"))),
        };
        let color = self.color_input(id, &["Color"], [0.8, 0.8, 0.8, 1.0])?;
        let channels = super::converters::separate_color(mode, color)
            .ok_or_else(|| unsupported_node(id, &format!("SeparateColor.{mode}")))?;
        let channel = match socket {
            "Red" => channels[0],
            "Green" => channels[1],
            "Blue" => channels[2],
            _ => return Err(invalid(format!("unknown Separate Color output `{socket}`"))),
        };
        Ok(Value::Scalar(channel))
    }

    fn tex_coord(&self, id: &Id, socket: &str) -> Result<Value> {
        if self.hit_ctx.bump_height_shifted && matches!(socket, "Normal" | "Reflection") {
            return Err(unsupported_shifted_bump_context(
                id,
                "ShaderNodeTexCoord",
                socket,
            ));
        }
        let value = match socket {
            "Generated" => self.hit_ctx.generated,
            "UV" => {
                let uv = storage_to_blender_uv(self.hit_ctx.uv);
                DVec3::new(uv[0], uv[1], 0.0)
            }
            "Object" => self.hit_ctx.object,
            "Normal" => self.hit_ctx.normal,
            "Position" => self.hit_ctx.position,
            "Reflection" => normalize_or(
                -self
                    .hit_ctx
                    .view_direction
                    .reflect(normalize_or(self.hit_ctx.normal, DVec3::Z)),
                DVec3::ZERO,
            ),
            "Camera" => {
                return Err(unsupported_node_id("TexCoord", "Camera.camera_projection"));
            }
            "Window" => {
                return Err(unsupported_node_id(
                    "TexCoord",
                    "Window.window_position_and_resolution",
                ));
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
        if node
            .properties
            .get("from_instancer")
            .and_then(JsonValue::as_bool)
            == Some(true)
        {
            return Err(unsupported_node(id, "ShaderNodeUVMap.from_instancer"));
        }
        let layer = layer_index(
            &self.hit_ctx,
            node.properties.get("uv_map").and_then(JsonValue::as_str),
        );
        let uv = storage_to_blender_uv(layer_uv(&self.hit_ctx, layer));
        Ok(Value::Vector(DVec3::new(uv[0], uv[1], 0.0)))
    }
    /// `ShaderNodeTangent`: Cycles' `svm_node_tangent` (`kernel/svm/tex_coord.h`).
    fn tangent(&self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Tangent" {
            return Err(invalid(format!("unknown Tangent output `{socket}`")));
        }
        let node = self.node(id)?;
        // DNA_node_types.h sets direction_type's default storage value to 0: RADIAL.
        let direction = match node.properties.get("direction_type") {
            None => "RADIAL",
            Some(value) => value
                .as_str()
                .ok_or_else(|| unsupported_node(id, "Tangent:direction_type"))?,
        };
        let tangent = match direction {
            "RADIAL" => {
                let axis = match node.properties.get("axis") {
                    None => "Z",
                    Some(value) => value
                        .as_str()
                        .ok_or_else(|| unsupported_node(id, "Tangent:RADIAL:axis"))?,
                };
                self.radial_tangent(id, axis)?
            }
            "UV_MAP" => {
                let Some(layers) = self.hit_ctx.uv_layers else {
                    return Err(unsupported_node(id, "Tangent:UV_MAP:uv_layers"));
                };
                let layer_name = match node.properties.get("uv_map") {
                    Some(value) => value
                        .as_str()
                        .ok_or_else(|| unsupported_node(id, "Tangent:UV_MAP:uv_map"))?,
                    None => "",
                };
                let layer = if layer_name.is_empty() {
                    layers.render
                } else {
                    layers.index_of(layer_name).filter(|&index| {
                        layers
                            .available_mask
                            .is_none_or(|mask| index < 8 && mask & (1_u8 << index) != 0)
                    })
                }
                .ok_or_else(|| unsupported_node(id, "Tangent:UV_MAP:uv_map"))?;
                let tangent_corners = self
                    .hit_ctx
                    .uv_tangents
                    .and_then(|tangents| tangents.get(layer))
                    .ok_or_else(|| unsupported_node(id, "Tangent:UV_MAP:uv_tangent"))?;
                let [first, second, third] = *tangent_corners;
                let [b0, b1, b2] = layers.barycentric;
                let tangent = first * b0 + second * b1 + third * b2;
                surface_tangent(tangent, self.hit_ctx.normal)
                    .ok_or_else(|| unsupported_node(id, "Tangent:UV_MAP:direction"))?
            }
            other => return Err(unsupported_node(id, &format!("Tangent:{other}"))),
        };
        Ok(Value::Vector(tangent))
    }
    /// The implicit `LINK_TANGENT` fallback: `GeometryNode::Tangent` uses RADIAL around Z
    /// (`shader_graph.cpp:934-940`, `kernel/geom/primitive.h:162-185`).
    fn radial_tangent(&self, id: &Id, axis: &str) -> Result<DVec3> {
        let generated = self
            .hit_ctx
            .generated_corners
            .ok_or_else(|| unsupported_node(id, "Tangent:RADIAL:Generated"))?;
        let point = self.hit_ctx.generated;
        if !point.is_finite() {
            return Err(unsupported_node(id, "Tangent:RADIAL:Generated"));
        }
        let local_tangent = match axis {
            "X" => DVec3::new(0.0, -(point.z - 0.5), point.y - 0.5),
            "Y" => DVec3::new(-(point.z - 0.5), 0.0, point.x - 0.5),
            "Z" => DVec3::new(-(point.y - 0.5), point.x - 0.5, 0.0),
            _ => return Err(unsupported_node(id, &format!("Tangent:RADIAL:{axis}"))),
        };
        let world_tangent = object_normal_tangent_to_world(local_tangent, generated, &self.hit_ctx)
            .ok_or_else(|| unsupported_node(id, "Tangent:RADIAL:Generated"))?;
        surface_tangent(world_tangent, self.hit_ctx.normal)
            .ok_or_else(|| unsupported_node(id, "Tangent:RADIAL:direction"))
    }
    fn normal_map(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Normal" {
            return Err(invalid(format!("unknown Normal Map output `{socket}`")));
        }
        let (space, imported_normal_map) = {
            let node = self.node(id)?;
            let space = match node
                .properties
                .get("space")
                .and_then(JsonValue::as_str)
                .unwrap_or("TANGENT_SPACE")
            {
                "TANGENT_SPACE" | "TANGENT" => NormalMapSpace::Tangent,
                "OBJECT" => NormalMapSpace::Object,
                "WORLD" => NormalMapSpace::World,
                "BLENDER_OBJECT" => NormalMapSpace::BlenderObject,
                "BLENDER_WORLD" => NormalMapSpace::BlenderWorld,
                other => return Err(unsupported_node(id, &format!("NormalMap:{other}"))),
            };
            let imported_normal_map = node
                .properties
                .get("potter_gltf_imported_normal_map")
                .and_then(JsonValue::as_bool)
                == Some(true);
            (space, imported_normal_map)
        };
        let color = self.color_input(id, &["Color"], [0.5, 0.5, 1.0, 1.0])?;
        let base_normal = normalize_or(self.hit_ctx.normal, DVec3::Z);
        if self.hit_ctx.path_traced {
            return self.cycles_normal_map(id, space, imported_normal_map, color, base_normal);
        }
        if space == NormalMapSpace::Tangent {
            let tangent_normal = DVec3::new(
                color[0] * 2.0 - 1.0,
                color[1] * 2.0 - 1.0,
                color[2] * 2.0 - 1.0,
            );
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
            return Ok(Value::Vector(normalize_or(
                base_normal.lerp(mapped_normal, strength),
                base_normal,
            )));
        }

        let direct_x = match self
            .node(id)?
            .properties
            .get("convention")
            .and_then(JsonValue::as_str)
            .unwrap_or("OPENGL")
        {
            "OPENGL" => false,
            "DIRECTX" => true,
            other => {
                return Err(unsupported_node(
                    id,
                    &format!("NormalMap:convention:{other}"),
                ));
            }
        };
        // Blender 5.2.2's color_to_blender_normal_new_shading maps channels to
        // (2r - 1, -2g - 1, -2b - 1).
        let mut mapped_normal = match space {
            NormalMapSpace::BlenderObject | NormalMapSpace::BlenderWorld => DVec3::new(
                color[0] * 2.0 - 1.0,
                color[1] * -2.0 - 1.0,
                color[2] * -2.0 - 1.0,
            ),
            _ => DVec3::new(
                color[0] * 2.0 - 1.0,
                color[1] * 2.0 - 1.0,
                color[2] * 2.0 - 1.0,
            ),
        };
        if direct_x {
            mapped_normal.y = -mapped_normal.y;
        }
        if matches!(
            space,
            NormalMapSpace::Object | NormalMapSpace::BlenderObject
        ) {
            if !self.hit_ctx.world_to_object.is_finite() {
                return Err(unsupported_node(id, "NormalMap:OBJECT:object_matrices"));
            }
            mapped_normal = self
                .hit_ctx
                .world_to_object
                .transpose()
                .transform_vector3(mapped_normal);
        }
        let strength = self.scalar_input(id, &["Strength"], 1.0)?.max(0.0);
        let mixed = base_normal.lerp(mapped_normal, strength);
        if !mixed.is_finite() {
            return Err(eval_error(format!(
                "Normal Map node `{id}` produced a non-finite result"
            )));
        }
        Ok(Value::Vector(normalize_or(mixed, base_normal)))
    }

    fn cycles_normal_map(
        &mut self,
        id: &Id,
        space: NormalMapSpace,
        imported_normal_map: bool,
        color: [f64; 4],
        base_normal: DVec3,
    ) -> Result<Value> {
        let direct_x = match self
            .node(id)?
            .properties
            .get("convention")
            .and_then(JsonValue::as_str)
            .unwrap_or("OPENGL")
        {
            "OPENGL" => false,
            "DIRECTX" => true,
            other => {
                return Err(unsupported_node(
                    id,
                    &format!("NormalMap:convention:{other}"),
                ));
            }
        };
        let mut mapped = DVec3::new(
            color[0] * 2.0 - 1.0,
            color[1] * 2.0 - 1.0,
            color[2] * 2.0 - 1.0,
        );
        if direct_x {
            mapped.y = -mapped.y;
        }
        if matches!(
            space,
            NormalMapSpace::BlenderObject | NormalMapSpace::BlenderWorld
        ) {
            mapped.y = -mapped.y;
            mapped.z = -mapped.z;
        }

        if space == NormalMapSpace::Tangent {
            let strength = self.scalar_input(id, &["Strength"], 1.0)?;
            mapped.x *= strength;
            mapped.y *= strength;
            mapped.z = 1.0 + (mapped.z - 1.0) * strength.clamp(0.0, 1.0);
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
            let mut normal = normalize_or(
                tangent_u * mapped.x + tangent_v * mapped.y + base_normal * mapped.z,
                base_normal,
            );
            if self.hit_ctx.back_facing {
                normal = -normal;
            }
            return Ok(Value::Vector(normalize_or(normal, base_normal)));
        }

        if matches!(
            space,
            NormalMapSpace::Object | NormalMapSpace::BlenderObject
        ) {
            if !self.hit_ctx.world_to_object.is_finite() {
                return Err(unsupported_node(id, "NormalMap:OBJECT:object_matrices"));
            }
            mapped = self
                .hit_ctx
                .world_to_object
                .transpose()
                .transform_vector3(mapped);
        }
        mapped = normalize_or(mapped, base_normal);
        if self.hit_ctx.back_facing {
            mapped = -mapped;
        }
        let strength = self.scalar_input(id, &["Strength"], 1.0)?;
        let mixed = if strength.to_bits() == 1.0_f64.to_bits() {
            mapped
        } else {
            base_normal.lerp(mapped, strength.max(0.0))
        };
        Ok(Value::Vector(normalize_or(mixed, base_normal)))
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
        if self.hit_ctx.bump_height_shifted
            && self
                .hit_ctx
                .color_layers
                .is_some_and(|layers| layers.iter().any(|(layer, _)| *layer == name))
        {
            return Err(unsupported_shifted_bump_color_layer(
                id,
                "ShaderNodeAttribute",
                socket,
            ));
        }
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
        let use_render = name.is_empty();
        let layer_available = if use_render {
            self.hit_ctx.color_render.is_some()
        } else {
            self.hit_ctx
                .color_layers
                .is_some_and(|layers| layers.iter().any(|(layer, _)| *layer == name))
        };
        if self.hit_ctx.bump_height_shifted && layer_available {
            return Err(unsupported_shifted_bump_color_layer(
                id,
                "ShaderNodeVertexColor",
                socket,
            ));
        }
        color_output(
            layer_color(&self.hit_ctx, name, use_render, [0.0, 0.0, 0.0, 0.0]),
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
fn input_socket_matches(
    groups: &Registry<NodeGroup>,
    node: &GraphNode,
    candidate: &str,
    requested: &str,
) -> bool {
    if socket_matches(candidate, requested) {
        return true;
    }
    if node.node_type != "ShaderNodeGroup" {
        return false;
    }
    node.properties
        .get("node_tree")
        .and_then(JsonValue::as_str)
        .and_then(|group_id| groups.get(group_id))
        .is_some_and(|group| {
            group.interface.inputs.iter().any(|socket| {
                (socket.id == candidate || socket.name == candidate)
                    && (socket.id == requested || socket.name == requested)
            })
        })
}
fn group_socket_value_kind(socket_type: &str) -> Option<ValueKind> {
    match socket_type {
        "float"
        | "integer"
        | "boolean"
        | "NodeSocketFloat"
        | "NodeSocketFloatFactor"
        | "NodeSocketFloatAngle"
        | "NodeSocketFloatWavelength"
        | "NodeSocketFloatColorTemperature"
        | "NodeSocketInt"
        | "NodeSocketBool" => Some(ValueKind::Scalar),
        "vector" | "NodeSocketVector" | "NodeSocketVectorDirection" | "NodeSocketVectorEuler" => {
            Some(ValueKind::Vector)
        }
        "color" | "NodeSocketColor" => Some(ValueKind::Color),
        "shader" | "NodeSocketShader" => Some(ValueKind::Shader),
        _ => None,
    }
}

fn active_group_output(group: &NodeGroup) -> Option<&Id> {
    let mut fallback = None;
    for (id, node) in &group.nodes {
        if node.node_type != "NodeGroupOutput" {
            continue;
        }
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
    fallback
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
fn collect_linked_alpha_images(
    node_groups: &Registry<NodeGroup>,
    group: &NodeGroup,
    output: &Id,
    visited_groups: &mut BTreeSet<Id>,
    alpha_linked_images: &mut BTreeSet<Id>,
) {
    for image in linked_alpha_image_nodes(group, output) {
        alpha_linked_images.insert(image.clone());
    }
    for node_id in reachable_nodes(group, output) {
        let Some(node) = group.nodes.get(node_id) else {
            continue;
        };
        if node.node_type != "ShaderNodeGroup" {
            continue;
        }
        let Some(child_id) = node.properties.get("node_tree").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some((child_key, child_group)) = node_groups.get_key_value(child_id) else {
            continue;
        };
        if child_group.kind != GraphKind::Shader || !visited_groups.insert(child_key.clone()) {
            continue;
        }
        if let Some(child_output) = active_group_output(child_group) {
            collect_linked_alpha_images(
                node_groups,
                child_group,
                child_output,
                visited_groups,
                alpha_linked_images,
            );
        }
    }
}

/// Whether EEVEE compiles the graph as transparent, `GPU_MATFLAG_TRANSPARENT`, which is set statically per node:
/// a Transparent BSDF whose Color is linked or not black (`node_shader_bsdf_transparent.cc:30`), and a Principled
/// whose Alpha is linked or not one (`node_shader_bsdf_principled.cc:400`). Only the nodes that feed the output count.
/// Specular and Ray Portal set the same flag in Blender, but Potter's evaluator rejects both node types, so a material
/// with one never evaluates and needs no rule.
fn graph_is_transparent(
    graph_id: &Id,
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    output: &Id,
) -> bool {
    let mut group_stack = BTreeSet::new();
    group_stack.insert(graph_id.clone());
    graph_is_transparent_output(group, node_groups, output, "Surface", &mut group_stack)
}

fn graph_is_transparent_output(
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    output: &Id,
    output_socket: &str,
    group_stack: &mut BTreeSet<Id>,
) -> bool {
    let reachable = nodes_reaching_socket(group, output, output_socket);
    let linked = |node: &Id, socket: &str| {
        group
            .links
            .iter()
            .any(|link| link.to_node == *node && link.to_socket == socket)
    };
    if group.nodes.iter().any(|(id, node)| {
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
                    || node
                        .inputs
                        .get("Alpha")
                        .is_some_and(|value| number(value, 1.0).total_cmp(&1.0).is_ne())
            }
            _ => false,
        }
    }) {
        return true;
    }
    for node_id in &reachable {
        let Some(node) = group.nodes.get(node_id) else {
            continue;
        };
        if node.node_type != "ShaderNodeGroup" {
            continue;
        }
        let Some(child_name) = node.properties.get("node_tree").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some((child_id, child)) = node_groups.get_key_value(child_name) else {
            continue;
        };
        if child.kind != GraphKind::Shader || !group_stack.insert(child_id.clone()) {
            continue;
        }
        let Some(child_output) = active_group_output(child) else {
            group_stack.remove(child_id);
            continue;
        };
        for link in group.links.iter().filter(|link| {
            link.from_node == *node_id
                && (reachable.contains(&link.to_node)
                    || (link.to_node == *output && link.to_socket == output_socket))
        }) {
            let Some(interface_output) =
                child.interface.outputs.iter().find(|socket| {
                    socket.id == link.from_socket || socket.name == link.from_socket
                })
            else {
                continue;
            };
            if graph_is_transparent_output(
                child,
                node_groups,
                child_output,
                &interface_output.name,
                group_stack,
            ) {
                group_stack.remove(child_id);
                return true;
            }
        }
        group_stack.remove(child_id);
    }
    false
}

fn nodes_reaching_socket(group: &NodeGroup, sink: &Id, socket: &str) -> BTreeSet<Id> {
    let mut pending: Vec<Id> = group
        .links
        .iter()
        .filter(|link| link.to_node == *sink && link.to_socket == socket)
        .map(|link| link.from_node.clone())
        .collect();
    let mut ancestors = BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if ancestors.insert(node_id.clone()) {
            pending.extend(
                group
                    .links
                    .iter()
                    .filter(|link| link.to_node == node_id)
                    .map(|link| link.from_node.clone()),
            );
        }
    }
    ancestors
}

/// Finds a Bump whose output contributes to a transparency input in the active surface graph.
fn bump_affecting_transparency(
    graph_id: &Id,
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    output: &Id,
) -> Option<Id> {
    let mut group_stack = BTreeSet::new();
    group_stack.insert(graph_id.clone());
    bump_in_surface_output(group, node_groups, output, "Surface", &mut group_stack)
}

fn bump_in_surface_output(
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    output: &Id,
    output_socket: &str,
    group_stack: &mut BTreeSet<Id>,
) -> Option<Id> {
    let surface_nodes = nodes_reaching_socket(group, output, output_socket);
    for node_id in &surface_nodes {
        let Some(node) = group.nodes.get(node_id) else {
            continue;
        };
        let socket = match node.node_type.as_str() {
            "ShaderNodeBsdfPrincipled" => "Alpha",
            "ShaderNodeBsdfTransparent" => "Color",
            "ShaderNodeMixShader" => "Fac",
            _ => continue,
        };
        let ancestors = nodes_reaching_socket(group, node_id, socket);
        if let Some(bump) = bump_in_transparency_ancestors(
            node_groups,
            group,
            &ancestors,
            node_id,
            socket,
            group_stack,
        ) {
            return Some(bump);
        }
    }
    for node_id in &surface_nodes {
        let Some(node) = group.nodes.get(node_id) else {
            continue;
        };
        if node.node_type != "ShaderNodeGroup" {
            continue;
        }
        let Some(child_name) = node.properties.get("node_tree").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some((child_id, child)) = node_groups.get_key_value(child_name) else {
            continue;
        };
        if child.kind != GraphKind::Shader || !group_stack.insert(child_id.clone()) {
            continue;
        }
        let Some(child_output) = active_group_output(child) else {
            group_stack.remove(child_id);
            continue;
        };
        for link in group.links.iter().filter(|link| {
            link.from_node == *node_id
                && (surface_nodes.contains(&link.to_node)
                    || (link.to_node == *output && link.to_socket == output_socket))
        }) {
            let Some(interface_output) =
                child.interface.outputs.iter().find(|socket| {
                    socket.id == link.from_socket || socket.name == link.from_socket
                })
            else {
                continue;
            };
            if let Some(bump) = bump_in_surface_output(
                child,
                node_groups,
                child_output,
                &interface_output.name,
                group_stack,
            ) {
                group_stack.remove(child_id);
                return Some(bump);
            }
        }
        group_stack.remove(child_id);
    }
    None
}

fn bump_in_transparency_ancestors(
    node_groups: &Registry<NodeGroup>,
    group: &NodeGroup,
    ancestors: &BTreeSet<Id>,
    sink: &Id,
    socket: &str,
    group_stack: &mut BTreeSet<Id>,
) -> Option<Id> {
    for node_id in ancestors {
        let Some(node) = group.nodes.get(node_id) else {
            continue;
        };
        if node.node_type == "ShaderNodeBump" {
            return Some(node_id.clone());
        }
        if node.node_type != "ShaderNodeGroup" {
            continue;
        }
        let Some(child_name) = node.properties.get("node_tree").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some((child_id, child)) = node_groups.get_key_value(child_name) else {
            continue;
        };
        if child.kind != GraphKind::Shader || !group_stack.insert(child_id.clone()) {
            continue;
        }
        let Some(child_output) = active_group_output(child) else {
            group_stack.remove(child_id);
            continue;
        };
        for link in group.links.iter().filter(|link| {
            link.from_node == *node_id
                && (ancestors.contains(&link.to_node)
                    || (link.to_node == *sink && link.to_socket == socket))
        }) {
            let Some(interface_output) =
                child.interface.outputs.iter().find(|output| {
                    output.id == link.from_socket || output.name == link.from_socket
                })
            else {
                continue;
            };
            let child_ancestors =
                nodes_reaching_socket(child, child_output, &interface_output.name);
            if let Some(bump) = bump_in_transparency_ancestors(
                node_groups,
                child,
                &child_ancestors,
                child_output,
                &interface_output.name,
                group_stack,
            ) {
                group_stack.remove(child_id);
                return Some(bump);
            }
        }
        group_stack.remove(child_id);
    }
    None
}

fn number(value: &JsonValue, default: f64) -> f64 {
    value
        .as_bool()
        .map(|value| if value { 1.0 } else { 0.0 })
        .or_else(|| value.as_f64().filter(|value| value.is_finite()))
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

const SHADER_MATH_OPERATIONS: &[&str] = &[
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "SINE",
    "COSINE",
    "TANGENT",
    "ARCSINE",
    "ARCCOSINE",
    "ARCTANGENT",
    "POWER",
    "LOGARITHM",
    "MINIMUM",
    "MAXIMUM",
    "ROUND",
    "LESS_THAN",
    "GREATER_THAN",
    "MODULO",
    "FLOORED_MODULO",
    "ABSOLUTE",
    "ARCTAN2",
    "FLOOR",
    "CEIL",
    "FRACTION",
    "FRACT",
    "SQRT",
    "INV_SQRT",
    "INVERSE_SQRT",
    "SIGN",
    "EXPONENT",
    "RADIANS",
    "DEGREES",
    "SINH",
    "COSH",
    "TANH",
    "TRUNC",
    "SNAP",
    "WRAP",
    "COMPARE",
    "MULTIPLY_ADD",
    "PINGPONG",
    "SMOOTH_MIN",
    "SMOOTH_MAX",
];
const VECTOR_MATH_OPERATIONS: &[&str] = &[
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "CROSS_PRODUCT",
    "PROJECT",
    "REFLECT",
    "DOT_PRODUCT",
    "DISTANCE",
    "LENGTH",
    "SCALE",
    "NORMALIZE",
    "SNAP",
    "ROUND",
    "FLOOR",
    "CEIL",
    "MODULO",
    "FRACTION",
    "ABSOLUTE",
    "MINIMUM",
    "MAXIMUM",
    "WRAP",
    "SINE",
    "COSINE",
    "TANGENT",
    "REFRACT",
    "FACEFORWARD",
    "MULTIPLY_ADD",
    "POWER",
    "SIGN",
];
const MIX_DATA_TYPES: &[&str] = &["FLOAT", "VECTOR", "RGBA"];
const MIX_BLEND_TYPES: &[&str] = &[
    "MIX",
    "ADD",
    "MULTIPLY",
    "SUBTRACT",
    "SCREEN",
    "DIVIDE",
    "DIFFERENCE",
    "DARKEN",
    "LIGHTEN",
    "OVERLAY",
    "DODGE",
    "BURN",
    "HUE",
    "SATURATION",
    "VALUE",
    "COLOR",
    "SOFT_LIGHT",
    "LINEAR_LIGHT",
    "EXCLUSION",
];
const MIX_FACTOR_MODES: &[&str] = &["UNIFORM", "NON_UNIFORM"];
const RAMP_INTERPOLATIONS: &[&str] = &["LINEAR", "EASE", "CONSTANT", "CARDINAL", "B_SPLINE"];
const RAMP_COLOR_MODES: &[&str] = &["RGB", "HSV", "HSL"];
const RAMP_HUE_INTERPOLATIONS: &[&str] = &["NEAR", "FAR", "CW", "CCW"];
const MAP_RANGE_DATA_TYPES: &[&str] = &["FLOAT", "FLOAT_VECTOR"];
const MAP_RANGE_INTERPOLATIONS: &[&str] = &["LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"];
const CLAMP_TYPES: &[&str] = &["MINMAX", "RANGE"];

fn node_setting_string(
    id: &Id,
    settings: &serde_json::Map<String, JsonValue>,
    field: &str,
    default: &'static str,
    feature: &str,
    supported: &[&'static str],
) -> Result<&'static str> {
    let value = match settings.get(field) {
        None => return Ok(default),
        Some(JsonValue::String(value)) => value.as_str(),
        Some(_) => {
            return Err(unsupported_node(id, &format!("{feature}.{field}.invalid")));
        }
    };
    supported
        .iter()
        .copied()
        .find(|option| *option == value)
        .ok_or_else(|| unsupported_node(id, &format!("{feature}.{field}.{value}")))
}

fn node_setting_bool(
    id: &Id,
    settings: &serde_json::Map<String, JsonValue>,
    field: &str,
    default: bool,
    feature: &str,
) -> Result<bool> {
    match settings.get(field) {
        None => Ok(default),
        Some(JsonValue::Bool(value)) => Ok(*value),
        Some(_) => Err(unsupported_node(id, &format!("{feature}.{field}.invalid"))),
    }
}

fn eevee_modulo_result(a: f32, b: f32) -> Option<f32> {
    match crate::node_kernels::eevee_compatible_mod(a, b) {
        Ok(value) => Some(value),
        Err(crate::node_kernels::EeveeModuloError::PrecisionBoundary) => {
            Some(a - (a / b) as i32 as f32 * b)
        }
        Err(crate::node_kernels::EeveeModuloError::QuotientOutOfRange) => None,
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

#[cfg(test)]
fn color_lerp(first: [f64; 4], second: [f64; 4], factor: f64) -> [f64; 4] {
    let factor = bounded_factor(factor);
    std::array::from_fn(|index| first[index] + (second[index] - first[index]) * factor)
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
/// Cycles' `make_orthonormals` (`util/math_float3.h`): a tangent from the cross product with
/// `(1, 1, 1)`, chosen by exact component equality, and the bitangent `N × tangent`.
#[expect(
    clippy::float_cmp,
    reason = "Cycles selects the tangent basis by exact component equality of the normal"
)]
pub(crate) fn make_orthonormals(normal: DVec3) -> (DVec3, DVec3) {
    let tangent = if normal.x != normal.y || normal.x != normal.z {
        DVec3::new(
            normal.z - normal.y,
            normal.x - normal.z,
            normal.y - normal.x,
        )
    } else {
        DVec3::new(
            normal.z - normal.y,
            normal.x + normal.z,
            -normal.y - normal.x,
        )
    }
    .normalize_or_zero();
    (tangent, normal.cross(tangent))
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
/// The Cycles tangent node transforms object-space tangents by the inverse transpose. Generated
/// corner coordinates and the hit triangle recover that transform's projection onto the world
/// geometric tangent plane without falling back to an arbitrary frame.
fn object_normal_tangent_to_world(
    tangent: DVec3,
    generated: GeneratedCorners,
    hit_ctx: &HitContext<'_>,
) -> Option<DVec3> {
    let layers = hit_ctx.uv_layers?;
    let [g0, g1, g2] = generated.corners;
    let scale = generated.per_local;
    if !tangent.is_finite()
        || !g0.is_finite()
        || !g1.is_finite()
        || !g2.is_finite()
        || !scale.is_finite()
        || scale.abs().min_element() <= f64::EPSILON
    {
        return None;
    }
    let local_edge1 = DVec3::new(
        (g1.x - g0.x) / scale.x,
        (g1.y - g0.y) / scale.y,
        (g1.z - g0.z) / scale.z,
    );
    let local_edge2 = DVec3::new(
        (g2.x - g0.x) / scale.x,
        (g2.y - g0.y) / scale.y,
        (g2.z - g0.z) / scale.z,
    );
    let local_normal = local_edge1.cross(local_edge2);
    let local_normal_length_squared = local_normal.length_squared();
    let [p0, p1, p2] = layers.positions;
    let world_edge1 = p1 - p0;
    let world_edge2 = p2 - p0;
    let (a11, a12, a22) = (
        world_edge1.dot(world_edge1),
        world_edge1.dot(world_edge2),
        world_edge2.dot(world_edge2),
    );
    let determinant = a11 * a22 - a12 * a12;
    if !local_normal_length_squared.is_finite()
        || local_normal_length_squared <= f64::EPSILON
        || !determinant.is_finite()
        || determinant <= f64::EPSILON * a11 * a22
    {
        return None;
    }
    let tangent = tangent - local_normal * tangent.dot(local_normal) / local_normal_length_squared;
    let (r1, r2) = (tangent.dot(local_edge1), tangent.dot(local_edge2));
    let world_tangent = world_edge1 * ((a22 * r1 - a12 * r2) / determinant)
        + world_edge2 * ((a11 * r2 - a12 * r1) / determinant);
    world_tangent.is_finite().then_some(world_tangent)
}

/// Cycles' `cross(N, normalize(cross(T, N)))` tangent projection.
fn surface_tangent(tangent: DVec3, normal: DVec3) -> Option<DVec3> {
    if !tangent.is_finite() || !normal.is_finite() || normal.length_squared() <= f64::EPSILON {
        return None;
    }
    let normal = normal.normalize();
    let crossed = tangent.cross(normal);
    if !crossed.is_finite() || crossed.length_squared() <= f64::EPSILON {
        return None;
    }
    let tangent = normal.cross(crossed.normalize());
    (tangent.is_finite() && tangent.length_squared() > f64::EPSILON).then(|| tangent.normalize())
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

fn unsupported_shifted_bump_color_layer(id: &Id, node_type: &str, socket: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "{node_type} output `{socket}` cannot be differentiated for shifted Bump Height samples because per-corner color data is unavailable"
        ),
        json!({
            "feature_id": format!("shader.node.{node_type}.bump_height_interpolation"),
            "node_id": id,
            "output": socket,
        }),
    )
}

fn unsupported_shifted_bump_context(id: &Id, node_type: &str, socket: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "{node_type} output `{socket}` cannot be differentiated for shifted Bump Height samples because shifted normal/view derivatives are unavailable"
        ),
        json!({
            "feature_id": format!("shader.node.{node_type}.bump_height_interpolation"),
            "node_id": id,
            "output": socket,
        }),
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

    use std::{
        cell::RefCell,
        collections::{BTreeMap, BTreeSet},
    };

    use glam::{DMat4, DQuat, DVec3};
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use crate::{
        error::ErrorCode,
        geom::GeneratedCorners,
        graph::{GraphInterface, GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
        image::{ImageData, ImageExtension, ImageInterpolation},
        model::{Id, Material, Registry, TextureRef},
    };

    use super::{
        BsdfParams, Evaluator, EvaluatorWorkspace, HitContext, HitDifferentials, HitUvLayers,
        PreparedMaterial, SurfaceShading, color_lerp, evaluate_displacement, evaluate_surface,
        layer_index, layer_uv,
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
    fn evaluate_vector_output(
        node_type: &str,
        output: &str,
        inputs: Value,
        properties: Value,
        matrices: Option<&(DMat4, DMat4)>,
    ) -> DVec3 {
        let mut group = NodeGroup::new("Vector Node", GraphKind::Shader);
        insert_node(&mut group, "node", node_type, inputs);
        set_properties(&mut group, "node", properties);
        let graph_id = Id::new("vector_node_graph".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), group);
        let graph = groups.get(&graph_id).unwrap();
        let mut context = HitContext::new(&groups);
        if let Some((object_to_world, world_to_object)) = matrices {
            context.object_to_world = *object_to_world;
            context.world_to_object = *world_to_object;
        }
        let material = Material::default();
        let alpha_linked_images = BTreeSet::new();
        let mut workspace = EvaluatorWorkspace::default();
        let mut evaluator = Evaluator {
            group: graph,
            material: &material,
            hit_ctx: context,
            workspace: &mut workspace,
            math_semantics: crate::node_kernels::MathSemantics::Cycles,
            alpha_linked_images: &alpha_linked_images,
            links: None,
            group_input_binding: None,
            group_depth: 0,
        };
        let node_id = Id::new("node".to_owned()).unwrap();
        let super::Value::Vector(value) = evaluator.output(&node_id, output, 0).unwrap() else {
            panic!("the vector shader output must evaluate to a vector")
        };
        value
    }

    fn evaluate_camera_data_view_vector(position: DVec3, world_to_camera: DMat4) -> DVec3 {
        let mut group = NodeGroup::new("Camera Node", GraphKind::Shader);
        insert_node(&mut group, "node", "ShaderNodeCameraData", json!({}));
        let graph_id = Id::new("camera_data_graph".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), group);
        let graph = groups.get(&graph_id).unwrap();
        let mut context = HitContext::new(&groups);
        context.position = position;
        context.set_camera_matrices(world_to_camera.inverse(), world_to_camera);
        let material = Material::default();
        let alpha_linked_images = BTreeSet::new();
        let mut workspace = EvaluatorWorkspace::default();
        let mut evaluator = Evaluator {
            group: graph,
            material: &material,
            hit_ctx: context,
            workspace: &mut workspace,
            alpha_linked_images: &alpha_linked_images,
            math_semantics: crate::node_kernels::MathSemantics::Cycles,
            links: None,
            group_input_binding: None,
            group_depth: 0,
        };
        let node_id = Id::new("node".to_owned()).unwrap();
        let super::Value::Vector(value) = evaluator.output(&node_id, "View Vector", 0).unwrap()
        else {
            panic!("Camera Data View Vector must evaluate to a vector")
        };
        value
    }

    #[test]
    fn vector_rotate_modes_invert_and_vector_transform_spaces_match_blender() {
        let quarter_turn = std::f64::consts::FRAC_PI_2;
        let cases = [
            ("AXIS_ANGLE", DVec3::X, DVec3::Y, false),
            ("AXIS_ANGLE", DVec3::X, -DVec3::Y, true),
            ("X_AXIS", DVec3::Y, DVec3::Z, false),
            ("Y_AXIS", DVec3::Z, DVec3::X, false),
            ("Z_AXIS", DVec3::X, DVec3::Y, false),
            ("EULER_XYZ", DVec3::X, DVec3::Y, false),
            ("EULER_XYZ", DVec3::X, -DVec3::Y, true),
        ];
        for (rotation_type, input, expected, invert) in cases {
            let actual = evaluate_vector_output(
                "ShaderNodeVectorRotate",
                "Vector",
                json!({
                    "Vector": input.to_array(),
                    "Center": [0.0, 0.0, 0.0],
                    "Axis": [0.0, 0.0, 1.0],
                    "Angle": quarter_turn,
                    "Rotation": [0.0, 0.0, quarter_turn],
                }),
                json!({"rotation_type":rotation_type,"invert":invert}),
                None,
            );
            assert!(
                (actual - expected).length() < 1e-12,
                "{rotation_type}: {actual:?}"
            );
        }
        let centered = evaluate_vector_output(
            "ShaderNodeVectorRotate",
            "Vector",
            json!({
                "Vector": [2.0, 0.0, 0.0],
                "Center": [1.0, 0.0, 0.0],
                "Axis": [0.0, 0.0, 1.0],
                "Angle": quarter_turn,
            }),
            json!({"rotation_type":"AXIS_ANGLE","invert":false}),
            None,
        );
        assert!((centered - DVec3::new(1.0, 1.0, 0.0)).length() < 1e-12);

        let object_to_world = DMat4::from_scale_rotation_translation(
            DVec3::new(2.0, 3.0, 4.0),
            DQuat::from_rotation_z(quarter_turn),
            DVec3::new(4.0, 5.0, 6.0),
        );
        let world_to_object = object_to_world.inverse();
        let matrices = (object_to_world, world_to_object);
        let point = evaluate_vector_output(
            "ShaderNodeVectorTransform",
            "Vector",
            json!({"Vector":[1.0,0.0,0.0]}),
            json!({"vector_type":"POINT","convert_from":"OBJECT","convert_to":"WORLD"}),
            Some(&matrices),
        );
        assert!((point - DVec3::new(4.0, 7.0, 6.0)).length() < 1e-12);
        let direction = evaluate_vector_output(
            "ShaderNodeVectorTransform",
            "Vector",
            json!({"Vector":[1.0,0.0,0.0]}),
            json!({"vector_type":"VECTOR","convert_from":"OBJECT","convert_to":"WORLD"}),
            Some(&matrices),
        );
        assert!((direction - DVec3::new(0.0, 2.0, 0.0)).length() < 1e-12);
        let normal = evaluate_vector_output(
            "ShaderNodeVectorTransform",
            "Vector",
            json!({"Vector":[1.0,0.0,0.0]}),
            json!({"vector_type":"NORMAL","convert_from":"OBJECT","convert_to":"WORLD"}),
            Some(&matrices),
        );
        assert!((normal - DVec3::Y).length() < 1e-12);
        let default_direction = evaluate_vector_output(
            "ShaderNodeVectorTransform",
            "Vector",
            json!({"Vector":[1.0,0.0,0.0]}),
            json!({"convert_from":"OBJECT","convert_to":"WORLD"}),
            Some(&matrices),
        );
        assert!((default_direction - DVec3::new(0.0, 2.0, 0.0)).length() < 1e-12);
        let camera_normal = evaluate_vector_output(
            "ShaderNodeVectorTransform",
            "Vector",
            json!({"Vector":[0.0,3.0,4.0]}),
            json!({"vector_type":"NORMAL","convert_from":"CAMERA","convert_to":"CAMERA"}),
            None,
        );
        assert!((camera_normal - DVec3::new(0.0, 0.6, 0.8)).length() < 1e-12);
    }

    #[test]
    fn camera_data_view_vector_flips_view_space_z() {
        let camera_to_world = DMat4::from_translation(DVec3::new(0.0, 0.0, 3.0));
        let view = evaluate_camera_data_view_vector(DVec3::ZERO, camera_to_world.inverse());
        assert_eq!(view, DVec3::Z);
    }
    #[test]
    fn normal_map_non_tangent_spaces_transform_world_normals() {
        let quarter_turn = std::f64::consts::FRAC_PI_2;
        let object_to_world = DMat4::from_scale_rotation_translation(
            DVec3::new(2.0, 3.0, 4.0),
            DQuat::from_rotation_z(quarter_turn),
            DVec3::new(4.0, 5.0, 6.0),
        );
        let matrices = (object_to_world, object_to_world.inverse());
        let mapped = evaluate_vector_output(
            "ShaderNodeNormalMap",
            "Normal",
            json!({"Color":[1.0,0.5,0.5,1.0],"Strength":1.0}),
            json!({"space":"OBJECT","convention":"OPENGL"}),
            Some(&matrices),
        );
        assert!((mapped - DVec3::Y).length() < 1e-12);
        let world_normal = evaluate_vector_output(
            "ShaderNodeNormalMap",
            "Normal",
            json!({"Color":[1.0,0.5,0.5,1.0],"Strength":1.0}),
            json!({"space":"WORLD","convention":"OPENGL"}),
            None,
        );
        assert!((world_normal - DVec3::X).length() < 1e-12);
        let directx = evaluate_vector_output(
            "ShaderNodeNormalMap",
            "Normal",
            json!({"Color":[0.5,0.75,1.0,1.0],"Strength":1.0}),
            json!({"space":"WORLD","convention":"DIRECTX"}),
            None,
        );
        assert!((directx - DVec3::new(0.0, -1.0, 2.0).normalize()).length() < 1e-12);

        let blender_world = evaluate_vector_output(
            "ShaderNodeNormalMap",
            "Normal",
            json!({"Color":[0.5,0.5,1.0,1.0],"Strength":1.0}),
            json!({"space":"BLENDER_WORLD","convention":"OPENGL"}),
            None,
        );
        assert!((blender_world - DVec3::new(0.0, -2.0, -3.0).normalize()).length() < 1e-12);
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
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .base_color
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
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .base_color[0]
    }

    #[test]
    fn textured_base_sets_the_principled_glass_tint() {
        // Cycles' Principled glass tint is the square root of the clamped base colour. A base texture of 0.25 gives
        // 0.5; an untextured base colour of 1.0 would give 1.0, so the tint must follow the texture.
        let material = Material {
            base_color: [1.0; 4],
            roughness: 0.2,
            transmission: 1.0,
            base_color_texture: Some(TextureRef {
                image: Id::new("image".to_owned()).unwrap(),
                uv_layer: None,
                interpolation: ImageInterpolation::Closest,
                extension: ImageExtension::Repeat,
            }),
            ..Material::default()
        };
        let names = vec!["UVMap".to_owned()];
        let corners = vec![[[0.5, 0.5]; 3]];
        let groups = Registry::new();
        let context = HitContext::new(&groups).with_uv_layers(HitUvLayers {
            names: &names,
            corners: &corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric: [1.0 / 3.0; 3],
            render: Some(0),
            available_mask: None,
        });
        let images = BTreeMap::from([(
            "image".to_owned(),
            ImageData {
                colour: crate::image::ColourCache::default(),
                storage: crate::image::ImageStorage::default(),
                width: 8,
                height: 8,
                pixels: vec![[0.25, 0.25, 0.25, 1.0]; 64],
                tiles: BTreeMap::new(),
                interpolation: ImageInterpolation::Closest,
                srgb: false,
                udim: false,
            },
        )]);
        let context = context.with_images(&images);
        let params = *evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap();
        for channel in params.glass_transmission_tint {
            assert!(
                (channel - 0.5).abs() < 1.0e-9,
                "glass tint {:?} should be sqrt(0.25) per channel",
                params.glass_transmission_tint
            );
        }
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

    #[test]
    fn thin_wall_does_not_evaluate_disconnected_subsurface_inputs() {
        let mut graph = principled_output(json!({"Thin Wall": 1.0, "Subsurface Weight": 0.5}));
        insert_node(
            &mut graph,
            "ignored",
            "ShaderNodeUnsupportedForThinWallTest",
            json!({}),
        );
        link(
            &mut graph,
            "ignored",
            "Vector",
            "principled",
            "Subsurface Radius",
        );
        link(
            &mut graph,
            "ignored",
            "Value",
            "principled",
            "Subsurface Scale",
        );
        link(
            &mut graph,
            "ignored",
            "Value",
            "principled",
            "Subsurface IOR",
        );
        let graph_id = Id::new("thin_wall_graph".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let surface = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
        assert!(surface.closures[0].params.thin_wall);
    }

    #[test]
    fn a_mix_of_two_subsurface_methods_keeps_each_closure_its_own_method() {
        use super::SubsurfaceMethod;
        // Each Principled arm carries its own method. Mix Shader concatenates the arms' closures, so both methods reach
        // the surface and neither is replaced by a mixed or first-arm method (Cycles keeps the closures separate).
        let mut graph = NodeGroup::new("MixMethods", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "mix",
            "ShaderNodeMixShader",
            json!({"Fac": 0.5}),
        );
        insert_node(
            &mut graph,
            "burley",
            "ShaderNodeBsdfPrincipled",
            json!({"Subsurface Weight": 1.0}),
        );
        insert_node(
            &mut graph,
            "walk",
            "ShaderNodeBsdfPrincipled",
            json!({"Subsurface Weight": 1.0}),
        );
        for (id, method) in [("burley", "BURLEY"), ("walk", "RANDOM_WALK")] {
            let node = graph
                .nodes
                .get_mut(&Id::new(id.to_owned()).unwrap())
                .unwrap();
            node.properties
                .insert("subsurface_method".to_owned(), json!(method));
        }
        link(&mut graph, "burley", "BSDF", "mix", "Shader");
        link(&mut graph, "walk", "BSDF", "mix", "Shader_001");
        link(&mut graph, "mix", "Shader", "output", "Surface");
        let id = Id::new("mix_methods".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(id.clone(), graph);
        let material = Material {
            node_tree: Some(id),
            ..Material::default()
        };
        let shading = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
        let methods: Vec<SubsurfaceMethod> = shading
            .closures
            .iter()
            .map(|closure| closure.params.subsurface_method)
            .collect();
        assert_eq!(
            methods,
            [SubsurfaceMethod::Burley, SubsurfaceMethod::RandomWalk]
        );
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
    fn nested_group_transparency_tracks_bump_alpha_dependency() {
        let child_id = Id::new("transparent_child".to_owned()).unwrap();
        let root_id = Id::new("transparent_root".to_owned()).unwrap();
        let mut child = NodeGroup::new("Transparent Child", GraphKind::Shader);
        child.interface.outputs.push(GraphSocket {
            id: "Shader".to_owned(),
            name: "Shader".to_owned(),
            socket_type: "shader".to_owned(),
            default: Value::Null,
        });
        insert_node(&mut child, "output", "NodeGroupOutput", json!({}));
        insert_node(
            &mut child,
            "principled",
            "ShaderNodeBsdfPrincipled",
            json!({}),
        );
        insert_node(&mut child, "bump", "ShaderNodeBump", json!({}));
        insert_node(&mut child, "separate", "ShaderNodeSeparateXYZ", json!({}));
        insert_node(&mut child, "height", "ShaderNodeValue", json!({}));
        link(&mut child, "height", "Value", "bump", "Height");
        link(&mut child, "bump", "Normal", "separate", "Vector");
        link(&mut child, "separate", "Z", "principled", "Alpha");
        link(&mut child, "principled", "BSDF", "output", "Shader");

        let mut root = NodeGroup::new("Transparent Root", GraphKind::Shader);
        insert_node(&mut root, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(&mut root, "child", "ShaderNodeGroup", json!({}));
        set_properties(&mut root, "child", json!({"node_tree": child_id.as_str()}));
        link(&mut root, "child", "Shader", "output", "Surface");

        let mut groups = Registry::new();
        groups.insert(child_id, child);
        groups.insert(root_id.clone(), root);
        let material = Material {
            node_tree: Some(root_id),
            ..Material::default()
        };
        let prepared = PreparedMaterial::new(&material, &groups);
        assert!(prepared.is_transparent());
        assert_eq!(
            prepared.realtime_transparency_bump().map(Id::as_str),
            Some("bump")
        );
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
        let shading = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
        // Principled Volume at Density 0.7: σs = Color · 0.7 and σa = (1 - Color) · 0.7.
        assert!(channels_close(
            weighted_volume(&shading, |params| params.volume_scatter),
            [0.14, 0.28, 0.42]
        ));
        assert!(channels_close(
            weighted_volume(&shading, |params| params.volume_absorption),
            [0.56, 0.42, 0.28]
        ));
        assert_eq!(shading.volume[0].params.volume_anisotropy, 0.35);
    }

    fn channels_close(actual: [f64; 3], expected: [f64; 3]) -> bool {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12)
    }

    /// The volume coefficient summed over the Volume output's closures, each scaled by its closure weight.
    fn weighted_volume(
        shading: &SurfaceShading,
        coefficient: fn(&BsdfParams) -> [f64; 3],
    ) -> [f64; 3] {
        let mut total = [0.0; 3];
        for closure in &shading.volume {
            for (sum, channel) in total.iter_mut().zip(coefficient(&closure.params)) {
                *sum += closure.weight * channel;
            }
        }
        total
    }

    /// Two Volume Absorption closures (σa = (1 - Color) · Density: (0.8, 0.5, 0.2) and (0.25, 0.25, 0.25))
    /// joined by `combiner`; returns the evaluated shading, whose volume closures carry the coefficients.
    fn two_absorption_closures(combiner: &str, inputs: Value) -> SurfaceShading {
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
        assert_eq!(added.volume.len(), 2);
        assert!(channels_close(
            weighted_volume(&added, |params| params.volume_absorption),
            [1.05, 0.75, 0.45]
        ));
        let mixed = two_absorption_closures("ShaderNodeMixShader", json!({"Fac":0.25}));
        let weights: Vec<f64> = mixed.volume.iter().map(|closure| closure.weight).collect();
        assert_eq!(weights, [0.75, 0.25]);
        assert!(channels_close(
            weighted_volume(&mixed, |params| params.volume_absorption),
            [0.6625, 0.4375, 0.2125]
        ));
    }

    /// Two Principled BSDFs joined by `combiner` (Mix or Add Shader), evaluated as the surface's closures. The first
    /// is red with roughness 0.1, normal +X and coat 0.5; the second is blue with roughness 0.9, normal +Z and coat
    /// 0.2. `inputs` are the combiner's own inputs.
    fn two_principled_surface(combiner: &str, inputs: Value) -> SurfaceShading {
        let mut graph = NodeGroup::new("Surface", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "first",
            "ShaderNodeBsdfPrincipled",
            json!({
                "Base Color":[1.0,0.0,0.0,1.0],"Roughness":0.1,
                "Normal":[1.0,0.0,0.0],"Coat Weight":0.5
            }),
        );
        insert_node(
            &mut graph,
            "second",
            "ShaderNodeBsdfPrincipled",
            json!({
                "Base Color":[0.0,0.0,1.0,1.0],"Roughness":0.9,
                "Normal":[0.0,0.0,1.0],"Coat Weight":0.2
            }),
        );
        insert_node(&mut graph, "combine", combiner, inputs);
        link(&mut graph, "first", "BSDF", "combine", "Shader");
        link(&mut graph, "second", "BSDF", "combine", "Shader_001");
        link(&mut graph, "combine", "Shader", "output", "Surface");
        let graph_id = Id::new("surface_shader".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        evaluate_surface(&material, &HitContext::new(&groups)).unwrap()
    }

    #[test]
    fn mix_shader_weights_closures_and_keeps_each_closures_parameters() {
        // Mix Shader gives the first arm its 1 - Fac share and the second its Fac share. Nothing is averaged: each
        // closure keeps its own normal, roughness and coat.
        let shading = two_principled_surface("ShaderNodeMixShader", json!({"Fac":0.25}));
        assert_eq!(shading.closures.len(), 2);
        let (first, second) = (shading.closures[0], shading.closures[1]);
        assert!((first.weight - 0.75).abs() < 1e-12);
        assert!((second.weight - 0.25).abs() < 1e-12);
        assert_eq!(first.params.normal, DVec3::X);
        assert_eq!(second.params.normal, DVec3::Z);
        assert_eq!(first.params.roughness, 0.1);
        assert_eq!(second.params.roughness, 0.9);
        assert_eq!(first.params.coat_weight, 0.5);
        assert_eq!(second.params.coat_weight, 0.2);
        // The heavier first arm is the dominant closure, so its normal is the one displacement is measured along.
        assert_eq!(shading.dominant().unwrap().normal, DVec3::X);
    }

    #[test]
    fn add_shader_concatenates_closures_with_unit_weight() {
        // Add Shader keeps both arms' closures whole at weight 1: no coat or normal is merged across them.
        let shading = two_principled_surface("ShaderNodeAddShader", json!({}));
        assert_eq!(shading.closures.len(), 2);
        assert!(shading.closures.iter().all(|closure| closure.weight == 1.0));
        assert_eq!(shading.closures[0].params.coat_weight, 0.5);
        assert_eq!(shading.closures[1].params.coat_weight, 0.2);
        assert_eq!(shading.closures[0].params.coat_normal, None);
        assert_eq!(shading.closures[1].params.normal, DVec3::Z);
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
        let shading = evaluate_surface(&material, &context).unwrap();
        assert!((shading.displacement_height - 0.5).abs() < f64::EPSILON);
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
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&layers, None),
        )
        .unwrap()
        .dominant()
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
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups)
                .with_color_layers(&[], Some([0.91, 0.37, 0.64, 0.22]))
                .with_warnings(&sink),
        )
        .unwrap()
        .dominant()
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
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&[], Some(render)),
        )
        .unwrap()
        .dominant()
        .unwrap();
        assert_eq!(params.base_color, render);

        let sink = RefCell::new(Vec::new());
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups)
                .with_color_layers(&[], None)
                .with_warnings(&sink),
        )
        .unwrap()
        .dominant()
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
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&found, None),
        )
        .unwrap()
        .dominant()
        .unwrap();
        assert_eq!(params.base_color, [source[0], source[1], source[2], 1.0]);

        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&[], Some([0.87, 0.54, 0.19, 0.27])),
        )
        .unwrap()
        .dominant()
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
        let params = *evaluate_surface(
            &material,
            &HitContext::new(&groups).with_color_layers(&layers, None),
        )
        .unwrap()
        .dominant()
        .unwrap();
        // Multiply: each channel of A times the interpolated vertex color B, at factor 1.
        for (actual, expected) in params.base_color.iter().zip([
            f64::from(0.1_f32),
            f64::from(0.15_f32),
            f64::from(0.4_f32),
            1.0,
        ]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "base color {:?}",
                params.base_color
            );
        }
    }

    #[test]
    fn separate_color_hsv_mode_matches_the_blender_color_kernel() {
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
        let displacement = evaluate_displacement(&material, &context).unwrap();
        assert!((displacement - 1.0 / 6.0).abs() < 1.0e-6);
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
            .dominant()
            .unwrap()
            .roughness
    }

    #[test]
    fn math_wrap_and_pingpong_match_blenders_wrapf_and_pingpongf() {
        // Blender's `wrapf(value, max, min)` takes the second input as the maximum and the third as the minimum, and
        // `pingpongf(value, scale)` is the triangle wave of the signed scale (`math_base_inline.cc`).
        let cases = [
            // The importer's REPEAT wrap (max 0, min 1) maps the negative and the overflowing input into [0, 1].
            ("WRAP", -0.3, 0.0, 1.0, f64::from(0.7_f32)),
            ("WRAP", 1.3, 0.0, 1.0, f64::from(0.299_999_95_f32)),
            ("WRAP", -0.3, 1.0, 0.0, f64::from(0.7_f32)),
            ("WRAP", 0.4, 1.0, 0.0, f64::from(0.4_f32)),
            // A zero range returns the minimum.
            ("WRAP", 0.4, 0.0, 0.0, 0.0),
            ("PINGPONG", -0.4, 1.0, 0.0, f64::from(0.399_999_98_f32)),
            ("PINGPONG", 2.5, 1.0, 0.0, 0.5),
            // The signed scale: a negative scale gives the same wave.
            ("PINGPONG", -0.4, -1.0, 0.0, f64::from(0.399_999_98_f32)),
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

    fn input_node_graph() -> (Registry<NodeGroup>, Id, Id, Id) {
        let mut group = NodeGroup::new("Shader Inputs", GraphKind::Shader);
        insert_node(&mut group, "object", "ShaderNodeObjectInfo", json!({}));
        insert_node(&mut group, "geometry", "ShaderNodeNewGeometry", json!({}));
        let graph_id = Id::new("shader_input_graph".to_owned()).unwrap();
        let object_id = Id::new("object".to_owned()).unwrap();
        let geometry_id = Id::new("geometry".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), group);
        (groups, graph_id, object_id, geometry_id)
    }

    #[test]
    fn group_input_is_lazy_and_uses_shifted_realtime_bump_context() {
        let child_id = Id::new("bump_group".to_owned()).unwrap();
        let mut child = NodeGroup::new("Bump Group", GraphKind::Shader);
        child.interface = GraphInterface {
            inputs: vec![
                GraphSocket {
                    id: "height".to_owned(),
                    name: "Height".to_owned(),
                    socket_type: "NodeSocketFloat".to_owned(),
                    default: json!(0.0),
                },
                GraphSocket {
                    id: "unused".to_owned(),
                    name: "Unused".to_owned(),
                    socket_type: "NodeSocketFloat".to_owned(),
                    default: json!(0.0),
                },
            ],
            outputs: vec![GraphSocket {
                id: "normal".to_owned(),
                name: "Normal".to_owned(),
                socket_type: "NodeSocketVector".to_owned(),
                default: json!([0.0, 0.0, 1.0]),
            }],
        };
        insert_node(&mut child, "input", "NodeGroupInput", json!({}));
        insert_node(
            &mut child,
            "bump",
            "ShaderNodeBump",
            json!({"Strength": 1.0, "Distance": 1.0, "Filter Width": 1.0}),
        );
        insert_node(
            &mut child,
            "unused_sink",
            "ShaderNodeUnsupported",
            json!({}),
        );
        insert_node(&mut child, "output", "NodeGroupOutput", json!({}));
        link(&mut child, "input", "height", "bump", "Height");
        link(&mut child, "input", "unused", "unused_sink", "Value");
        link(&mut child, "bump", "Normal", "output", "Normal");

        let graph_id = Id::new("parent_graph".to_owned()).unwrap();
        let mut parent = NodeGroup::new("Parent", GraphKind::Shader);
        insert_node(
            &mut parent,
            "group",
            "ShaderNodeGroup",
            json!({"Height": 0.0, "Unused": 0.0}),
        );
        set_properties(
            &mut parent,
            "group",
            json!({"node_tree": child_id.as_str()}),
        );
        insert_node(&mut parent, "geometry", "ShaderNodeNewGeometry", json!({}));
        insert_node(
            &mut parent,
            "unsupported",
            "ShaderNodeUnsupported",
            json!({}),
        );
        link(&mut parent, "geometry", "Position", "group", "Height");
        link(&mut parent, "unsupported", "Value", "group", "Unused");

        let mut groups = Registry::new();
        groups.insert(child_id, child);
        groups.insert(graph_id.clone(), parent);
        let mut context = HitContext::new(&groups);
        context.realtime_differentials = Some(HitDifferentials {
            world_dx: DVec3::new(0.1, 0.0, 0.0),
            world_dy: DVec3::new(0.0, 0.1, 0.0),
            uv_dx: [0.0; 2],
            uv_dy: [0.0; 2],
        });
        let group_id = Id::new("group".to_owned()).unwrap();
        let super::Value::Vector(normal) =
            evaluate_input_node_output(&groups, &graph_id, &group_id, "Normal", &context).unwrap()
        else {
            panic!("group output must be a vector")
        };
        assert!(
            normal.x < -0.5,
            "shifted Group Input height was not reevaluated: {normal:?}"
        );
    }

    #[test]
    fn engine_specific_normal_map_hsv_and_wavelength_outputs_are_distinct() {
        let mut graph = NodeGroup::new("Engine Semantics", GraphKind::Shader);
        insert_node(
            &mut graph,
            "hsv",
            "ShaderNodeHueSaturation",
            json!({
                "Hue": 0.5,
                "Saturation": 1.0,
                "Value": 1.0,
                "Fac": 0.0,
                "Color": [-0.25, 0.2, 0.2, 1.0]
            }),
        );
        insert_node(
            &mut graph,
            "normal_map",
            "ShaderNodeNormalMap",
            json!({"Color": [0.5, 0.75, 0.25, 1.0], "Strength": 1.0}),
        );
        set_properties(
            &mut graph,
            "normal_map",
            json!({"space": "WORLD", "convention": "OPENGL"}),
        );
        insert_node(
            &mut graph,
            "wavelength",
            "ShaderNodeWavelength",
            json!({"Wavelength": 300.0}),
        );
        insert_node(
            &mut graph,
            "blackbody",
            "ShaderNodeBlackbody",
            json!({"Temperature": 12_000.0}),
        );
        insert_node(&mut graph, "geometry", "ShaderNodeNewGeometry", json!({}));
        insert_node(&mut graph, "tex_coord", "ShaderNodeTexCoord", json!({}));
        insert_node(&mut graph, "camera", "ShaderNodeCameraData", json!({}));
        let graph_id = Id::new("engine_semantics".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let mut context = HitContext::new(&groups);
        context.set_camera_matrices(DMat4::IDENTITY, DMat4::IDENTITY);
        context.position = DVec3::new(0.0, 0.0, 3.0);
        context.view_direction = DVec3::Z;
        context.back_facing = true;
        for path_traced in [false, true] {
            context.path_traced = path_traced;
            let super::Value::Vector(incoming) = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("geometry".to_owned()).unwrap(),
                "Incoming",
                &context,
            )
            .unwrap() else {
                panic!("Geometry Incoming output must be a vector")
            };
            let super::Value::Vector(reflection) = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("tex_coord".to_owned()).unwrap(),
                "Reflection",
                &context,
            )
            .unwrap() else {
                panic!("Texture Coordinate Reflection output must be a vector")
            };
            let super::Value::Scalar(depth) = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("camera".to_owned()).unwrap(),
                "View Z Depth",
                &context,
            )
            .unwrap() else {
                panic!("Camera Data View Z Depth output must be scalar")
            };
            assert_eq!(depth, if path_traced { -3.0 } else { 3.0 });
            assert_eq!(incoming, DVec3::Z);
            assert_eq!(reflection, DVec3::Z);
        }
        context.path_traced = false;

        let super::Value::Color(eevee_hsv) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("hsv".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("HSV output must be a color")
        };
        let super::Value::Vector(eevee_normal) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("normal_map".to_owned()).unwrap(),
            "Normal",
            &context,
        )
        .unwrap() else {
            panic!("Normal Map output must be a vector")
        };
        let super::Value::Color(eevee_wavelength) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("wavelength".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("Wavelength output must be a color")
        };
        let super::Value::Color(eevee_blackbody) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("blackbody".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("Blackbody output must be a color")
        };

        context.path_traced = true;
        let super::Value::Color(cycles_hsv) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("hsv".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("HSV output must be a color")
        };
        let super::Value::Vector(cycles_normal) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("normal_map".to_owned()).unwrap(),
            "Normal",
            &context,
        )
        .unwrap() else {
            panic!("Normal Map output must be a vector")
        };
        let super::Value::Color(cycles_wavelength) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("wavelength".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("Wavelength output must be a color")
        };
        let super::Value::Color(cycles_blackbody) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &Id::new("blackbody".to_owned()).unwrap(),
            "Color",
            &context,
        )
        .unwrap() else {
            panic!("Blackbody output must be a color")
        };

        assert_eq!(eevee_hsv[0], -0.25);
        assert_eq!(cycles_hsv[0], 0.0);
        assert!(eevee_normal.y > 0.6);
        assert!(eevee_normal.z < -0.6);
        assert!(cycles_normal.y < -0.6);
        assert!(cycles_normal.z > 0.6);
        assert!(eevee_wavelength[2] > 0.0);
        assert!(cycles_wavelength[..3].iter().all(|channel| *channel == 0.0));
        assert_ne!(eevee_blackbody[0], cycles_blackbody[0]);
    }

    #[test]
    fn imported_node_output_defaults_fail_closed_when_missing() {
        for (node_type, output) in [
            ("ShaderNodeValue", "Value"),
            ("ShaderNodeRGB", "Color"),
            ("ShaderNodeNormal", "Normal"),
        ] {
            let mut graph = NodeGroup::new("Imported Defaults", GraphKind::Shader);
            insert_node(&mut graph, "node", node_type, json!({}));
            set_properties(&mut graph, "node", json!({"bl_idname": node_type}));
            let graph_id = Id::new(format!("defaults_{}", node_type.to_lowercase())).unwrap();
            let mut groups = Registry::new();
            groups.insert(graph_id.clone(), graph);
            let context = HitContext::new(&groups);
            let error = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("node".to_owned()).unwrap(),
                output,
                &context,
            )
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::UnsupportedFeature);
            let expected_feature = format!("shader.node.{node_type}.output_defaults");
            assert_eq!(
                error.details["feature_id"].as_str(),
                Some(expected_feature.as_str())
            );
        }
    }

    #[test]
    fn shifted_bump_height_refuses_available_interpolated_color_layers() {
        for node_type in ["ShaderNodeAttribute", "ShaderNodeVertexColor"] {
            let mut graph = NodeGroup::new("Interpolated Bump Height", GraphKind::Shader);
            insert_node(&mut graph, "color", node_type, json!({}));
            let properties = if node_type == "ShaderNodeAttribute" {
                json!({"attribute_name": "Col"})
            } else {
                json!({"layer_name": "Col"})
            };
            set_properties(&mut graph, "color", properties);
            insert_node(
                &mut graph,
                "bump",
                "ShaderNodeBump",
                json!({"Strength": 1.0, "Distance": 1.0}),
            );
            link(&mut graph, "color", "Color", "bump", "Height");
            let graph_id = Id::new(format!("bump_color_{}", node_type.to_lowercase())).unwrap();
            let mut groups = Registry::new();
            groups.insert(graph_id.clone(), graph);
            let color_layer = [("Col", [0.2, 0.3, 0.4, 1.0])];
            let mut available_context =
                HitContext::new(&groups).with_color_layers(&color_layer, None);
            available_context.realtime_differentials = Some(HitDifferentials {
                world_dx: DVec3::new(0.1, 0.0, 0.0),
                world_dy: DVec3::new(0.0, 0.1, 0.0),
                uv_dx: [0.0; 2],
                uv_dy: [0.0; 2],
            });
            let error = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("bump".to_owned()).unwrap(),
                "Normal",
                &available_context,
            )
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::UnsupportedFeature);
            let expected_feature = format!("shader.node.{node_type}.bump_height_interpolation");
            assert_eq!(
                error.details["feature_id"].as_str(),
                Some(expected_feature.as_str())
            );
            assert!(
                error
                    .message
                    .contains("per-corner color data is unavailable")
            );

            let mut missing_context = HitContext::new(&groups).with_color_layers(&[], None);
            missing_context.realtime_differentials = available_context.realtime_differentials;
            let super::Value::Vector(normal) = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new("bump".to_owned()).unwrap(),
                "Normal",
                &missing_context,
            )
            .unwrap() else {
                panic!("Bump output must be a vector")
            };
            assert_eq!(normal, DVec3::Z);
        }
    }

    #[test]
    fn shifted_bump_height_refuses_unshifted_geometry_and_texture_directions() {
        let mut graph = NodeGroup::new("Shifted Directions", GraphKind::Shader);
        insert_node(&mut graph, "geometry", "ShaderNodeNewGeometry", json!({}));
        insert_node(&mut graph, "tex_coord", "ShaderNodeTexCoord", json!({}));
        let graph_id = Id::new("shifted_directions".to_owned()).unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let mut context = HitContext::new(&groups);
        context.bump_height_shifted = true;
        for (node_id, node_type, socket) in [
            ("geometry", "ShaderNodeNewGeometry", "Normal"),
            ("geometry", "ShaderNodeNewGeometry", "Incoming"),
            ("geometry", "ShaderNodeNewGeometry", "Tangent"),
            ("tex_coord", "ShaderNodeTexCoord", "Normal"),
            ("tex_coord", "ShaderNodeTexCoord", "Reflection"),
        ] {
            let error = evaluate_input_node_output(
                &groups,
                &graph_id,
                &Id::new(node_id.to_owned()).unwrap(),
                socket,
                &context,
            )
            .unwrap_err();
            assert_eq!(error.code, ErrorCode::UnsupportedFeature);
            let expected_feature = format!("shader.node.{node_type}.bump_height_interpolation");
            assert_eq!(
                error.details["feature_id"].as_str(),
                Some(expected_feature.as_str())
            );
            assert!(
                error
                    .message
                    .contains("shifted normal/view derivatives are unavailable")
            );
        }
    }

    fn evaluate_input_node_output(
        groups: &Registry<NodeGroup>,
        graph_id: &Id,
        node_id: &Id,
        output: &str,
        hit_ctx: &HitContext<'_>,
    ) -> crate::error::Result<super::Value> {
        let group = groups.get(graph_id).unwrap();
        let material = Material::default();
        let alpha_linked_images = BTreeSet::new();
        let mut workspace = EvaluatorWorkspace::default();
        let mut evaluator = Evaluator {
            group,
            material: &material,
            hit_ctx: *hit_ctx,
            workspace: &mut workspace,
            math_semantics: crate::node_kernels::MathSemantics::Cycles,
            alpha_linked_images: &alpha_linked_images,
            links: None,
            group_input_binding: None,
            group_depth: 0,
        };
        evaluator.output(node_id, output, 0)
    }

    #[test]
    fn object_info_location_uses_world_transform_and_other_outputs_name_required_data() {
        let (groups, graph_id, object_id, _) = input_node_graph();
        let location = DVec3::new(2.0, -3.0, 5.0);
        let mut context = HitContext::new(&groups);
        context.object_to_world = DMat4::from_translation(location);

        let super::Value::Vector(actual) =
            evaluate_input_node_output(&groups, &graph_id, &object_id, "Location", &context)
                .unwrap()
        else {
            panic!("Object Info Location must be a vector")
        };
        assert_eq!(actual, location);

        let error = evaluate_input_node_output(&groups, &graph_id, &object_id, "Random", &context)
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            json!("shader.node.ShaderNodeObjectInfo.Random.object_random_number")
        );
        let error =
            evaluate_input_node_output(&groups, &graph_id, &object_id, "Material Index", &context)
                .unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            json!("shader.node.ShaderNodeObjectInfo.Material Index.material_pass_index")
        );

        context.material_pass_index = Some(23.0);
        let super::Value::Scalar(actual) =
            evaluate_input_node_output(&groups, &graph_id, &object_id, "Material Index", &context)
                .unwrap()
        else {
            panic!("Object Info Material Index must be a scalar")
        };
        assert_eq!(actual, 23.0);
    }

    #[test]
    fn geometry_tangent_and_parametric_use_cycles_hit_data() {
        let (groups, graph_id, _, geometry_id) = input_node_graph();
        let barycentric = [0.2, 0.3, 0.5];
        let names: [String; 0] = [];
        let uv_corners: [[[f64; 2]; 3]; 0] = [];
        let layers = HitUvLayers {
            names: &names,
            corners: &uv_corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric,
            render: None,
            available_mask: None,
        };
        let generated = GeneratedCorners {
            corners: [DVec3::ZERO, DVec3::X, DVec3::Y],
            per_local: DVec3::ONE,
        };
        let mut context = HitContext::new(&groups)
            .with_uv_layers(layers)
            .with_generated_corners(generated, barycentric);
        context.path_traced = true;
        context.normal = DVec3::Z;

        let super::Value::Vector(tangent) =
            evaluate_input_node_output(&groups, &graph_id, &geometry_id, "Tangent", &context)
                .unwrap()
        else {
            panic!("Geometry Tangent must be a vector")
        };
        assert!(tangent.abs_diff_eq(-DVec3::Y, 1.0e-12));
        let mut derivative_context = HitContext::new(&groups).with_uv_layers(layers);
        derivative_context.path_traced = true;
        derivative_context.normal = DVec3::Z;
        let super::Value::Vector(front_tangent) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &geometry_id,
            "Tangent",
            &derivative_context,
        )
        .unwrap() else {
            panic!("Geometry Tangent must use the surface derivative without Generated data")
        };
        assert_eq!(front_tangent, DVec3::X);

        derivative_context.view_direction = -DVec3::Z;
        let super::Value::Vector(back_tangent) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &geometry_id,
            "Tangent",
            &derivative_context,
        )
        .unwrap() else {
            panic!("Geometry Tangent must orient the surface derivative on a backface")
        };
        assert_eq!(back_tangent, -DVec3::X);

        let super::Value::Vector(parametric) =
            evaluate_input_node_output(&groups, &graph_id, &geometry_id, "Parametric", &context)
                .unwrap()
        else {
            panic!("Geometry Parametric must be a vector")
        };
        assert!((parametric - DVec3::new(0.2, 0.3, 0.0)).length() <= 1.0e-15);

        let super::Value::Scalar(front_facing) =
            evaluate_input_node_output(&groups, &graph_id, &geometry_id, "Backfacing", &context)
                .unwrap()
        else {
            panic!("Geometry Backfacing must be a scalar")
        };
        assert_eq!(front_facing, 0.0);

        let mut back_facing_context = context;
        back_facing_context.view_direction = -DVec3::Z;
        back_facing_context.back_facing = true;
        let super::Value::Scalar(back_facing) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &geometry_id,
            "Backfacing",
            &back_facing_context,
        )
        .unwrap() else {
            panic!("Geometry Backfacing must be a scalar")
        };
        assert_eq!(back_facing, 1.0);

        let mut realtime_context = HitContext::new(&groups);
        realtime_context.back_facing = true;
        let super::Value::Scalar(realtime_back_facing) = evaluate_input_node_output(
            &groups,
            &graph_id,
            &geometry_id,
            "Backfacing",
            &realtime_context,
        )
        .unwrap() else {
            panic!("realtime Geometry Backfacing must be a scalar")
        };
        assert_eq!(realtime_back_facing, 1.0);
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

#[cfg(test)]
mod subsurface_method_tests {
    use super::SubsurfaceMethod;

    #[test]
    fn subsurface_method_parses_the_blender_identifiers() {
        assert_eq!(
            SubsurfaceMethod::parse("BURLEY"),
            Some(SubsurfaceMethod::Burley)
        );
        assert_eq!(
            SubsurfaceMethod::parse("RANDOM_WALK"),
            Some(SubsurfaceMethod::RandomWalk)
        );
        assert_eq!(
            SubsurfaceMethod::parse("RANDOM_WALK_SKIN"),
            Some(SubsurfaceMethod::RandomWalkSkin)
        );
        assert_eq!(
            SubsurfaceMethod::parse("RANDOM_WALK_LEGACY"),
            Some(SubsurfaceMethod::RandomWalkLegacy)
        );
        assert_eq!(SubsurfaceMethod::parse("random_walk"), None);
    }
}

#[cfg(test)]
mod generated_corner_tests {
    use glam::DVec3;

    use crate::{geom::GeneratedCorners, geom::TextureSpace, graph::NodeGroup, model::Registry};

    use super::HitContext;

    /// A texture space of half-extent 2 at the origin: Generated is `local / 4 + 0.5` per axis.
    fn texture_space() -> TextureSpace {
        TextureSpace::manual(DVec3::ZERO, DVec3::splat(2.0))
    }

    #[test]
    fn zero_corner_generated_stays_exactly_zero_under_any_weights() {
        let groups: Registry<NodeGroup> = Registry::new();
        let generated = GeneratedCorners {
            corners: [DVec3::ZERO; 3],
            per_local: texture_space().generated_per_local(),
        };
        let context = HitContext::new(&groups).with_generated_corners(generated, [0.2, 0.3, 0.5]);
        assert_eq!(context.generated, DVec3::ZERO);
    }

    #[test]
    fn corner_generated_blends_to_the_generated_of_the_interpolated_point() {
        let groups: Registry<NodeGroup> = Registry::new();
        let space = texture_space();
        let local = [
            DVec3::new(-2.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ];
        let barycentric = [0.5, 0.25, 0.25];
        let point =
            local[0] * barycentric[0] + local[1] * barycentric[1] + local[2] * barycentric[2];
        let generated = GeneratedCorners {
            corners: local.map(|corner| space.generated(corner)),
            per_local: space.generated_per_local(),
        };
        let context = HitContext::new(&groups).with_generated_corners(generated, barycentric);
        assert!((context.generated - space.generated(point)).length() < 1e-12);
    }

    #[test]
    fn corner_hit_takes_the_triangles_jacobian() {
        let groups: Registry<NodeGroup> = Registry::new();
        let per_local = DVec3::new(0.1, 0.2, 0.3);
        let generated = GeneratedCorners {
            corners: [DVec3::ZERO; 3],
            per_local,
        };
        let context = HitContext::new(&groups).with_generated_corners(generated, [1.0 / 3.0; 3]);
        assert_eq!(context.generated_per_local, per_local);
    }
}
