use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, Range},
    path::Path,
};

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::volume::{
        VolumeData, local_to_index_matrix, sample_density_checked_with,
        sample_density_realtime_checked_with, sample_index_density_checked_with,
        sample_index_density_realtime_checked_with, volume_attribute_is_present,
    },
    image::{ImageData, ImageInterpolation},
    model::{Id, LightType, Material, RayKind, SceneDoc, VolumeSampling},
    shader::{
        BsdfParams, EvaluatorWorkspace, HitContext, HitFootprint, HitUvLayers, PreparedMaterial,
        ShaderClosure, SubsurfaceMethod, SurfaceShading, equirectangular_uv,
        evaluate_displacement_with_workspace_semantics, evaluate_surface_with_workspace,
        evaluate_surface_with_workspace_into_semantics, evaluate_surface_with_workspace_semantics,
        heaviest, height_gradient, make_orthonormals,
    },
};

use super::principled_bsdf::{
    GlassSample, Layers, base_color, continuous_density, continuous_lobes, diffuse_coefficient,
    evaluate_masked, layers_at, lobe_probabilities, sample_glass, sample_specular_ggx_half,
    sheen_coefficient, specular_eta, subsurface_closure_weight, thin_glass_shadow_transparency,
};
use super::subsurface::{
    MAX_HITS, ProbeHits, SubsurfaceExit, burley_radius, channel_mask, eval as burley_eval,
    exit_params, pdf_mean, pick_channel, raw_radius, sample_radius,
};
use super::subsurface_walk::{
    WalkEntry, WalkMethod, entry_direction, skin_entry_direction, walk, walk_anisotropy,
    walk_radius, walk_uses_van_de_hulst,
};
use super::world_probe::volume_ambient_l0;
use super::{
    bvh::{Bvh, BvhHit, WalkRay},
    eevee_volume_resolve::{
        FroxelTextureLayout, LinearSampleWeights, VolumeDepthMapping, VolumeProjection,
        linear_sample_weights, occupancy_boundary_index, resolve_z_with_end_of_slice_bias,
        screen_uv_to_resolve_uv,
    },
    light_model::{
        AreaShape, Spread, area_radiance, area_shape, emitter_radiance, influence_fade,
        inside_emitter, local_light_influence_radius_squared, point_intensity, spot_attenuation,
        spot_cone, spread_attenuation, spread_region,
    },
    pixel_filter::PixelFilter,
    raster::{
        Camera, CameraFrame, CameraFrustum, CameraLightProjectedQuad, CameraLightRasterQuad, Mode,
        RasterOutput, StereoMode, Triangle, blend_corner_normals, camera_eye, camera_frame,
        camera_frustum, camera_light_display_quad, project_camera_light_quad, rasterize_with_lines,
        ray_for_sample_in_frame, smooth_shadow_origin,
    },
    scene::{
        ElementRecord, Geometry, HitPose, LayerTables, ObjectRecord, layered_context,
        object_info_color, object_info_pass_index, object_space,
    },
};

const RAY_EPSILON: f64 = 1.0e-5;
const MAX_VOLUME_STEPS: usize = 2048;
const CYCLES_VOLUME_MAX_STEPS: usize = 1024;
/// Cycles' `VOLUME_BOUNDS_MAX` (`intern/cycles/kernel/types.h`): volume-only boundary crossings
/// have their own counter, and the path ends past this many.
const VOLUME_BOUNDS_MAX: u32 = 1024;

#[derive(Clone, Copy)]
struct PointLight {
    position: DVec3,
    direction: DVec3,
    /// The evaluated object transform, including scale used by EEVEE's camera-visible spot cone.
    world_transform: DMat4,
    kind: LightType,
    color: DVec3,
    energy: f64,
    radius: f64,
    soft_falloff: bool,
    spot_size: f64,
    spot_blend: f64,
    /// The emitting shape of an area light; `None` for every other light type.
    area: Option<AreaShape>,
    /// Blender's area `spread` below π, as a cone about the emitting normal; `None` at the default.
    spread: Option<Spread>,
    /// Blender's `use_multiple_importance_sampling`: a BSDF-sampled ray may hit this light.
    use_mis: bool,
    /// Blender's `volume_factor` (`Light.volume_fac`), which EEVEE multiplies the light's volume in-scatter by
    /// (`eevee_light.cc`, `power[LIGHT_VOLUME]`). The path tracer does not apply it, as Cycles does not.
    volume_factor: f64,
    /// The squared radius of the light's volume influence (`light_attenuation_volume`), infinite for a sun. The realtime
    /// in-scatter fades with it; the path tracer does not.
    volume_radius_squared: f64,
    /// Blender's object ray mask of the light; a ray kind not in it passes the light by.
    ray_mask: u8,
    /// Whether EEVEE's local-light culler includes this camera-visible shape. The factors affect culling, not shape radiance.
    camera_shape_eligible: bool,
}

impl PointLight {
    /// The fade of this light's volume in-scatter at `point` (`light_attenuation_volume`): one for a sun, and for a local
    /// light inside its volume influence radius.
    fn volume_fade(&self, point: DVec3) -> f64 {
        if self.volume_radius_squared.is_infinite() {
            return 1.0;
        }
        let distance_squared = (self.position - point).length_squared().max(1.0e-6);
        influence_fade(distance_squared, self.volume_radius_squared)
    }

    /// Whether this light has an emitter shape a ray can meet: an area light with area and a spread
    /// above zero (Cycles' `area_light_spread_attenuation` is zero for a zero spread), or a point or
    /// spot light with a radius. A zero-radius point or spot light is a delta light, and the sun has
    /// no shape.
    fn has_shape(&self) -> bool {
        match self.kind {
            LightType::Area => {
                self.area.is_some_and(|shape| shape.area > 0.0)
                    && self.spread.is_none_or(|spread| !spread.is_zero())
            }
            LightType::Point | LightType::Spot => self.radius > 0.0,
            LightType::Sun => false,
        }
    }

    /// Whether a BSDF-sampled ray can hit this light (Cycles' `SHADER_USE_MIS`): a shaped light that
    /// samples with multiple importance sampling.
    fn hittable(&self) -> bool {
        self.use_mis && self.has_shape()
    }

    /// Whether a ray of kind `ray` can intersect this light, as Cycles' `lights_intersect` decides. A
    /// camera ray meets any shaped light whose camera visibility is on. A BSDF ray meets a shaped light
    /// only if it samples lights with MIS.
    fn intersects(&self, ray: RayKind) -> bool {
        if ray == RayKind::Camera {
            self.has_shape() && self.ray_mask & RayKind::Camera.bit() != 0
        } else {
            self.hittable()
        }
    }
}

/// One Volume closure of a voxel volume: its share of the volume's scatter, and its Henyey-Greenstein anisotropy. Cycles
/// sums the closures' phases weighted by their scatter (`volume_shader_phase_eval`), so the volume's phase is `Σ share·HG`.
#[derive(Clone, Copy)]
struct ScatterLobe {
    share: f64,
    anisotropy: f64,
}

impl VolumeInstance<'_> {
    /// The volume's phase at `cosine`: each closure's lobe weighted by its share of the scatter.
    fn phase_density(&self, cosine: f64) -> f64 {
        self.lobes
            .iter()
            .map(|lobe| lobe.share * henyey_greenstein(cosine, lobe.anisotropy))
            .sum()
    }
}

#[derive(Clone)]
struct VolumeInstance<'a> {
    volume: &'a VolumeData,
    world_to_local: DMat4,
    bounds_min: DVec3,
    bounds_max: DVec3,
    voxel_size: f64,
    meters_per_unit: f64,
    density_range: Option<(f64, f64)>,
    albedo: DVec3,
    /// The phase lobes of the volume's Volume closures, one per closure (`scatter_lobes`).
    lobes: Vec<ScatterLobe>,
    emission: DVec3,
    scattering: bool,
    /// Per-channel σt per unit grid density for path, EEVEE, and froxel transport.
    extinction_rgb_scale: DVec3,
    /// Blender's `cycles.volume_sampling` of the volume's material.
    sampling: crate::model::VolumeSampling,
    /// Blender's `cycles.volume_interpolation` of the volume's material.
    interpolation: crate::model::VolumeInterpolation,
    /// The inverse of an inline grid's index-to-local matrix, built once per volume (`local_to_index_matrix`). `None` for a
    /// decoded VDB, which samples at its local position.
    index_from_local: Option<DMat4>,
    /// The owning object's ray mask: a ray kind it excludes passes the volume by, as Cycles' volume bounds are not
    /// intersected by such rays.
    ray_mask: u8,
    /// Whether grid reads use Cycles' full-precision taps rather than EEVEE's half-float texture taps.
    path_traced: bool,
}
struct VolumeTemperature {
    attribute: String,
    temperature: f32,
    blackbody: f32,
    tint: [f32; 3],
    closure_weight: f64,
    center_emission: DVec3,
    interpolation: crate::model::VolumeInterpolation,
    path_traced: bool,
}

#[derive(Clone, Copy)]
struct VolumeEvent {
    point: DVec3,
    scattering_weight: DVec3,
}

struct VolumeSegment {
    transmittance: DVec3,
    event: Option<VolumeEvent>,
    stochastic: bool,
    channel_pdf: DVec3,
}

/// One closure of a hit, weighted as Cycles weights closures (`closure_alloc`): `weight` is the share of the hit the
/// closure takes, and `params` is the closure's own shading, with its alpha after the material's alpha mode, its own
/// normal, and its layers set up for the hit's view.
#[derive(Clone, Copy)]
struct HitClosure {
    weight: f64,
    params: BsdfParams,
    layers: Layers,
    pass_tint: DVec3,
}

impl HitClosure {
    fn new(
        weight: f64,
        params: &BsdfParams,
        view: DVec3,
        geometric: DVec3,
        bump_correction: bool,
    ) -> Self {
        let closure_layers = layers_at(params, params.normal, view, geometric, bump_correction);
        Self {
            weight,
            params: *params,
            layers: closure_layers,
            pass_tint: transparent_tint(params) * (1.0 - params.alpha),
        }
    }

    /// Selection weight of the reflective share, `w · α` (Cycles' sample weight of a BSDF closure). A closure without
    /// reflectance, such as Emission, takes no share.
    fn reflect_weight(&self) -> f64 {
        if self.params.reflectance > 0.0 {
            self.weight * self.params.alpha
        } else {
            0.0
        }
    }

    /// Selection weight of the transparent closure(s) on this surface, the weight times mean tint.
    fn pass_weight(&self) -> f64 {
        self.weight * self.pass_tint.element_sum() / 3.0
    }
}

/// A closure's transparent tint, clamped at zero: Cycles' Transparent colour.
fn transparent_tint(params: &BsdfParams) -> DVec3 {
    DVec3::from_array(params.transparent_tint).max(DVec3::ZERO)
}

/// Cycles' shadow transparency `Σ w · (1 − α) · tint` over each closure's `(weight, alpha, tint)`, one when no closure
/// has weight (`surface_shader_transparency`). `Closures::new` and the shadow probe both take it, so they cannot drift.
fn shadow_transparency(closures: impl Iterator<Item = (f64, f64, DVec3)>) -> DVec3 {
    let mut transparency = DVec3::ZERO;
    let mut weight_total = 0.0;
    for (weight, alpha, tint) in closures {
        transparency += tint * (weight * (1.0 - alpha));
        weight_total += weight;
    }
    if weight_total > 0.0 {
        transparency
    } else {
        DVec3::ONE
    }
}

/// The share a surface event takes: a reflective draw from closure `Reflect`, or a camera pass-through whose
/// `multiplier` is the throughput Cycles' transparent closure gives it.
enum Pick {
    Reflect(usize),
    Pass { multiplier: DVec3 },
}

/// Cycles' shading-side test of `bump_shadowing_term` (`intern/cycles/kernel/closure/bsdf.h:94-137`). A closure whose
/// normal `normal` is bent away from the smooth normal `smooth`, by bump, a normal map, or the specular correction,
/// evaluates to zero when `light` lies on the other side of the smooth surface than the closure normal does: the sign
/// of `cosNsI * cosNsN * cosNI`, with the smooth normal taken for the light and the closure normal for `cosNI`. A closure
/// whose normal equals the smooth normal always passes.
fn shading_side_passes(smooth: DVec3, normal: DVec3, light: DVec3) -> bool {
    if normal == smooth {
        return true;
    }
    smooth.dot(light) * smooth.dot(normal) * normal.dot(light) >= 0.0
}

/// Cycles' diffuse smoothing of `bump_shadowing_term` (`bsdf.h:94-137`): at grazing light, a diffuse lobe whose normal is
/// bent from the smooth normal takes the GGX shadowing term of Conty et al. (`A Microfacet-Based Shadowing Function to
/// Solve the Bump Terminator Problem`), with `alpha² = saturate(tan²θ_bend / 8)`. The factor is `G = 1 / (1 + Λ)` at the
/// light's cosine, with Cycles' GGX `Λ` (`bsdf_microfacet.h:602-623`).
fn diffuse_smoothing(smooth: DVec3, normal: DVec3, light: DVec3) -> f64 {
    if normal == smooth {
        return 1.0;
    }
    let cos_light = smooth.dot(light).abs();
    let cos_bend = smooth.dot(normal).abs();
    if cos_bend >= 1.0 || cos_light >= 1.0 {
        return 1.0;
    }
    if cos_light < 1.0e-6 {
        return 0.0;
    }
    let alpha2 = ((1.0 / (cos_bend * cos_bend) - 1.0) / 8.0).clamp(0.0, 1.0);
    let tan2_light = (1.0 / (cos_light * cos_light) - 1.0).max(0.0);
    let lambda = 0.5 * ((1.0 + alpha2 * tan2_light).sqrt() - 1.0);
    1.0 / (1.0 + lambda)
}

/// The closures of one hit with the totals Cycles' closure loops compute: selection, emission, and coverage.
struct Closures {
    items: Vec<HitClosure>,
    /// The smooth shading normal, Cycles' `sd->N` before bump or normal mapping: the reference of the shading-side test.
    smooth: DVec3,
    /// The geometric normal, Cycles' `sd->Ng`: the reference that bends the specular lobes (`layers_at`).
    geometric: DVec3,
    /// Cycles' `SD_USE_BUMP_MAP_CORRECTION` (`scene/shader.cpp`), the material's `use_bump_map_correction`: it enables
    /// the grazing diffuse smoothing and the lifted specular normal (`maybe_ensure_valid_specular_reflection`).
    bump_correction: bool,
    /// Sum of every share's selection weight, reflective and pass-through: the selection normaliser.
    selection_total: f64,
    /// Sum of the reflective selection weights.
    reflect_total: f64,
    /// Emitted radiance with each closure's coverage, `Σ w · α · E · λ` (`λ` is the layers' factor): the beauty's
    /// emission term.
    emission: DVec3,
    /// Emitted radiance without coverage, `Σ w · E · λ`: the emission pass.
    emission_pass: DVec3,
    /// Weight-averaged base colour: the albedo pass.
    albedo: [f64; 4],
    /// Camera coverage, one minus the mean shadow transparency of the closures; zero for a hit without closures.
    coverage: f64,
}

impl Closures {
    fn new(items: Vec<HitClosure>, smooth: DVec3, geometric: DVec3, bump_correction: bool) -> Self {
        let mut selection_total = 0.0;
        let mut reflect_total = 0.0;
        let mut emission = DVec3::ZERO;
        let mut emission_pass = DVec3::ZERO;
        let mut albedo = [0.0; 4];
        let mut weight_total = 0.0;
        for closure in &items {
            let (weight, params) = (closure.weight, &closure.params);
            let reflect = closure.reflect_weight();
            selection_total += reflect + closure.pass_weight();
            reflect_total += reflect;
            let emitted = DVec3::from_array(params.emission_color)
                * params.emission_strength
                * closure.layers.lower();
            emission += emitted * (weight * params.alpha);
            emission_pass += emitted * weight;
            for (channel, value) in albedo.iter_mut().zip(params.base_color) {
                *channel += weight * value;
            }
            weight_total += weight;
        }
        if weight_total > 0.0 {
            albedo = albedo.map(|channel| channel / weight_total);
        }
        let transparency = shadow_transparency(items.iter().map(|closure| {
            (
                closure.weight,
                closure.params.alpha,
                transparent_tint(&closure.params),
            )
        }));
        // Cycles' film alpha is one minus the mean transparent weight (`surface_shader_transparency`), tinted. An Add with a
        // Transparent arm therefore covers the pixel by its other arms only, and a hit without closures covers nothing.
        let coverage = (1.0 - transparency.element_sum() / 3.0).clamp(0.0, 1.0);
        Self {
            items,
            smooth,
            geometric,
            bump_correction,
            selection_total,
            reflect_total,
            emission,
            emission_pass,
            albedo,
            coverage,
        }
    }

    /// Chooses one share with probability proportional to its selection weight, as Cycles' `surface_shader_bsdf_sample`
    /// picks a closure by its sample weight. `None` when nothing is selectable.
    fn pick(&self, unit: f64) -> Option<Pick> {
        if self.items.is_empty() {
            return Some(Pick::Pass {
                multiplier: DVec3::ONE,
            });
        }
        let target = unit * self.selection_total;
        let mut cumulative = 0.0;
        let mut picked = None;
        for (index, closure) in self.items.iter().enumerate() {
            let reflect = closure.reflect_weight();
            if reflect > 0.0 {
                cumulative += reflect;
                picked = Some(Pick::Reflect(index));
                if target < cumulative {
                    break;
                }
            }
            let pass = closure.pass_weight();
            if pass > 0.0 {
                cumulative += pass;
                picked = Some(Pick::Pass {
                    multiplier: self.pass_multiplier(closure),
                });
                if target < cumulative {
                    break;
                }
            }
        }
        picked
    }

    /// The throughput of this closure's pass share: its tint times `S / mean(tint)`, so the pick leaves `w · tint`.
    fn pass_multiplier(&self, closure: &HitClosure) -> DVec3 {
        let tint = closure.pass_tint;
        tint * (self.selection_total / (tint.element_sum() / 3.0))
    }

    /// The reflective share at `light`: `Σ w·α·f·|cos|` over the closures, each with its own lobes, normal and cosine
    /// (Cycles' `surface_shader_bsdf_eval`). `sampled` is the closure a draw picked and the ray kind of that draw. Cycles'
    /// `bsdf_sample` evaluates the picked closure with `is_eval = false`, so only its diffuse draw keeps the shading-side
    /// test; every other closure takes it (`bump_shadowing_term`, `kernel/closure/bsdf.h`). `ray_mask` keeps the lobes a
    /// light admits (`evaluate_masked`); a path's own continuation takes every kind.
    fn bsdf_eval(&self, light: DVec3, sampled: Option<(usize, RayKind)>, ray_mask: u8) -> DVec3 {
        self.items
            .iter()
            .enumerate()
            .fold(DVec3::ZERO, |sum, (index, closure)| {
                let is_eval = match sampled {
                    Some((picked, ray)) if picked == index => ray == RayKind::Diffuse,
                    _ => true,
                };
                sum + self.closure_eval(closure, light, is_eval, ray_mask)
            })
    }

    /// One closure's reflective term at `light`, `w·α·f·|cos|`, with its diffuse and sheen lobes smoothed at grazing light.
    fn closure_eval(
        &self,
        closure: &HitClosure,
        light: DVec3,
        is_eval: bool,
        ray_mask: u8,
    ) -> DVec3 {
        let reflect = closure.reflect_weight();
        if reflect <= 0.0 {
            return DVec3::ZERO;
        }
        let (params, layers) = (&closure.params, &closure.layers);
        if is_eval && !shading_side_passes(self.smooth, layers.normal, light) {
            return DVec3::ZERO;
        }
        let cosine = layers.normal.dot(light).abs();
        // The diffuse and sheen lobes sit in the above-shading share of `evaluate_masked`, so their smoothing applies to that
        // share only; the specular and glass lobes take none. A light that excludes the diffuse kind drops both.
        let above = layers.normal.dot(light) > 0.0;
        let diffuse_admitted = ray_mask & RayKind::Diffuse.bit() != 0;
        let diffuse = if above && diffuse_admitted {
            diffuse_coefficient(params, layers, light)
        } else {
            DVec3::ZERO
        };
        let sheen = if above && diffuse_admitted {
            sheen_coefficient(params, layers, light)
        } else {
            DVec3::ZERO
        };
        // Cycles' `bump_shadowing_term` leaves the diffuse lobe unsmoothed when `SD_USE_BUMP_MAP_CORRECTION` is off
        // (`kernel/closure/bsdf.h`); the sheen lobe takes the same test, as it is a diffuse closure.
        let smoothing = |normal: DVec3| {
            if self.bump_correction {
                diffuse_smoothing(self.smooth, normal, light)
            } else {
                1.0
            }
        };
        let diffuse_correction = diffuse * (smoothing(layers.normal) - 1.0);
        let sheen_correction = sheen * (smoothing(layers.sheen_normal) - 1.0);
        let shaded = evaluate_masked(params, layers, light, ray_mask)
            + diffuse_correction
            + sheen_correction;
        shaded * (reflect * cosine)
    }

    /// Density of the continuous draws at `light`: each closure's lobe mixture weighted by its selection weight, over
    /// the whole selection, pass-through included (Cycles' `_surface_shader_bsdf_eval_mis`).
    fn continuous_pdf(&self, light: DVec3) -> f64 {
        if self.selection_total <= 0.0 {
            return 0.0;
        }
        let mixed: f64 = self
            .items
            .iter()
            .map(|closure| {
                let reflect = closure.reflect_weight();
                if reflect > 0.0 {
                    reflect * continuous_density(&closure.params, &closure.layers, light)
                } else {
                    0.0
                }
            })
            .sum();
        mixed / self.selection_total
    }

    /// Draws the reflective share of closure `index`, which `pick` chose, and returns the throughput weight of the
    /// closure set. A continuous draw carries `Σ w·α·f·|cos| / p_mix` with `p_mix = Σ w·α·p / S`. A delta or subsurface
    /// draw carries the lobe weight times `S`, since the picked share's `w·α` over its pick probability `w·α / S` is `S`.
    fn reflect(&self, index: usize, geometric: DVec3, rng: &mut Pcg32) -> Reflection {
        let closure = &self.items[index];
        match sample_lobes(&closure.params, &closure.layers, geometric, rng) {
            LobeDraw::Absorbed => Reflection::Absorbed,
            LobeDraw::Delta {
                direction,
                weight,
                ray,
                pdf,
                mis_weight,
            } => Reflection::Delta {
                direction,
                weight: weight * self.selection_total,
                ray,
                pdf,
                mis_weight,
            },
            LobeDraw::Subsurface { weight, method } => Reflection::Subsurface {
                weight: weight * self.selection_total,
                method,
            },
            LobeDraw::Continuous { direction, ray } => {
                let pdf = self.continuous_pdf(direction);
                Reflection::Continuous {
                    direction,
                    weight: self.bsdf_eval(direction, Some((index, ray)), RayKind::ALL_MASK)
                        / pdf.max(1.0e-8),
                    pdf,
                    ray,
                }
            }
        }
    }

    /// The closures after Cycles' glossy blur (`bsdf_microfacet_blur`, `kernel/closure/bsdf_microfacet.h`). It raises each
    /// cached microfacet alpha independently, while preserving setup-time energy compensation.
    fn blurred(&self, blur_alpha: f64) -> Self {
        let items = self
            .items
            .iter()
            .map(|closure| {
                let mut params = closure.params;
                params.coat_roughness = blurred_roughness(params.coat_roughness, blur_alpha);
                HitClosure {
                    weight: closure.weight,
                    params,
                    layers: closure.layers.blurred(blur_alpha),
                    pass_tint: closure.pass_tint,
                }
            })
            .collect();
        Self::new(items, self.smooth, self.geometric, self.bump_correction)
    }

    /// Cycles' average of the closures' squared roughness that `bsdf_widen_dD` takes (`_surface_shader_bsdf_eval_mis`,
    /// `kernel/integrator/surface_shader.h`): each closure's continuous lobes at `wo`, the direction the path leaves along,
    /// weighted by its selection weight and pdf, with a singular draw's `singular_weight` added to the total with zero
    /// roughness. A singular lobe has no density and no roughness, so it is left out, as Cycles' singular eval is zero.
    fn average_roughness_squared(&self, wo: DVec3, singular_weight: f64) -> f64 {
        let mut total = singular_weight;
        let mut weighted = 0.0;
        for closure in &self.items {
            let reflect = closure.reflect_weight();
            if reflect <= 0.0 {
                continue;
            }
            for lobe in continuous_lobes(&closure.params, &closure.layers, wo) {
                if lobe.roughness_squared > 0.0 {
                    total += reflect * lobe.weight;
                    weighted += reflect * lobe.weight * lobe.roughness_squared;
                }
            }
        }
        if total > 0.0 { weighted / total } else { 0.0 }
    }

    /// Cycles' falling back to diffuse after a diffuse bounce (`bssrdf_setup` with `PATH_RAY_DIFFUSE_ANCESTOR`): a
    /// Burley closure loses its subsurface weight. The random walk keeps its closure.
    fn disable_burley_subsurface(&mut self) {
        let burley = |closure: &HitClosure| {
            closure.params.subsurface_method == SubsurfaceMethod::Burley
                && !closure.params.thin_wall
                && closure.params.subsurface_weight > 0.0
        };
        if !self.items.iter().any(burley) {
            return;
        }
        let items: Vec<HitClosure> = self
            .items
            .iter()
            .map(|closure| {
                let mut params = closure.params;
                if params.subsurface_method == SubsurfaceMethod::Burley && !params.thin_wall {
                    params.subsurface_weight = 0.0;
                }
                let mut item = HitClosure::new(
                    closure.weight,
                    &params,
                    closure.layers.view,
                    self.geometric,
                    self.bump_correction,
                );
                item.pass_tint = closure.pass_tint;
                item
            })
            .collect();
        *self = Closures::new(items, self.smooth, self.geometric, self.bump_correction);
    }
}

/// A hit's shading: its closures, the volume media it bounds, and its geometry.
struct Surface {
    point: DVec3,
    /// The shading normal of the dominant closure. It is the reference normal of light and environment sampling, of the
    /// shadow offsets, and of the normal and occlusion passes.
    normal: DVec3,
    /// Unit normal on the viewer's side of the triangle; offsets ray origins to avoid self-hits.
    geometric_normal: DVec3,
    /// Origin of shadow rays: `point` on flat triangles, lifted onto the smooth surface otherwise.
    shadow_point: DVec3,
    closures: Closures,
    /// One medium per volume closure that scatters or absorbs light, keyed by (object, closure index), as Cycles' object
    /// volume stack keys its entries (`kernel/types.h`, `VolumeStack`).
    media: Vec<MeshMedium>,
    /// A linked Volume output with no linked Surface remains a boundary even at zero density.
    volume_boundary: bool,
    object_index: u32,
    element_index: u32,
    /// The triangle the hit lies on, which a random walk entering through it must not hit first (`ray.self.prim`).
    triangle_index: usize,
    /// The hit's camera footprint, Cycles' `sd->dP`: the bump's screen differentials, and the footprint of the
    /// shadow and light rays from this vertex (which carry `dP` unchanged, with no `dD`).
    footprint: f64,
}

/// Cycles' compact camera ray differentials (`kernel/camera/camera.h`): `dP`, the change of the ray origin, and `dD`,
/// the change of its direction, per raster pixel. A path's footprint at a hit is `ray.dP + t · ray.dD` (Cycles'
/// `differential_transfer_compact`, `kernel/util/differential.h`), with `t` the distance from the ray origin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CameraFootprint {
    dp: f64,
    dd: f64,
}

impl CameraFootprint {
    /// The footprint at distance `t` along a ray that carries this differential (`sd->dP`).
    fn at_distance(self, t: f64) -> f64 {
        self.dp + t * self.dd
    }
}

/// The camera footprint of the sample (`sample_x`, `sample_y`) of pixel (`x`, `y`), whose centre ray is `centre` (from
/// `ray_for_sample`, so the caller's ray is not computed twice). It is the compact lengths of the origin and direction
/// changes to the rays one pixel to the right and one pixel down, as Cycles' pinhole branch takes them
/// (`dD = normalize(D + dx) - normalize(D)`, compacted by `differential_make_compact` at `differential_scale` 1).
/// A perspective camera has `dP = 0` and an orthographic camera `dD = 0`. Panoramic, fisheye and depth-of-field
/// cameras use the same finite difference rather than Cycles' per-projection formulas, and a one-pixel-wide or
/// one-pixel-high image has no neighbour along that axis, which contributes nothing.
/// A side-by-side stereo footprint keeps its horizontal neighbour in the same packed eye.
fn camera_footprint(
    frame: &CameraFrame,
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
    (origin, direction): (DVec3, DVec3),
) -> Result<CameraFootprint> {
    let (view_start, view_end) = if frame.is_side_by_side() {
        let split = width / 2;
        if x < split {
            (0, split)
        } else {
            (split, width)
        }
    } else {
        (0, width)
    };
    let right = if x + 1 < view_end {
        x + 1
    } else if x > view_start {
        x - 1
    } else {
        x
    };
    let down = if y + 1 < height {
        y + 1
    } else {
        y.saturating_sub(1)
    };
    let mut dp = 0.0;
    let mut dd = 0.0;
    for (neighbour_x, neighbour_y) in [(right, y), (x, down)] {
        if let Some((neighbour_origin, neighbour_direction)) = ray_for_sample_in_frame(
            frame,
            camera,
            width,
            height,
            neighbour_x,
            neighbour_y,
            sample_x,
            sample_y,
        )? {
            dp += (neighbour_origin - origin).length();
            dd += (neighbour_direction.normalize_or_zero() - direction).length();
        }
    }
    Ok(CameraFootprint {
        dp: 0.5 * dp,
        dd: 0.5 * dd,
    })
}

/// The scene's Filter Glossy (`cycles.blur_glossy`, Blender 1.0 by default), unclamped: the glossy blur reads it as
/// `1 / filter_glossy`, and the widening clamps it to one.
fn filter_glossy(doc: &SceneDoc) -> f64 {
    doc.scenes
        .get(&doc.active_scene)
        .map_or(0.0, |scene| scene.render.filter_glossy)
}

/// The glossy blur's `alpha` (Cycles' `blur_roughness`, `kernel/integrator/surface_shader.h`,
/// `surface_shader_prepare_closures`): after a bounce whose sampled pdf is `min_ray_pdf`, `blur_pdf` is
/// `filter_glossy · min_ray_pdf` and the microfacet closures are blurred to `sqrt(1 - blur_pdf) / 2` when that is
/// below one. A Filter Glossy of zero disables the blur (Cycles stores `FLT_MAX` for it).
fn glossy_blur_alpha(filter_glossy: f64, min_ray_pdf: f64) -> Option<f64> {
    if filter_glossy <= 0.0 {
        return None;
    }
    let blur_pdf = min_ray_pdf / filter_glossy;
    (blur_pdf < 1.0).then(|| (1.0 - blur_pdf).sqrt() * 0.5)
}

/// A microfacet lobe's roughness after the glossy blur: Cycles' `bsdf_microfacet_blur` sets `alpha = max(blur, alpha)`
/// with alpha the squared roughness (`kernel/closure/bsdf_microfacet.h`), so the roughness is the root of that.
fn blurred_roughness(roughness: f64, blur_alpha: f64) -> f64 {
    roughness.powi(2).max(blur_alpha).sqrt()
}

/// The filter glossy widening scale of the scene, Cycles' `differential_widen_scale = min(1, filter_glossy)`
/// (`intern/cycles/scene/integrator.cpp`). Blender's Filter Glossy (`cycles.blur_glossy`) defaults to 1.0.
fn filter_glossy_widen_scale(doc: &SceneDoc) -> f64 {
    filter_glossy(doc).clamp(0.0, 1.0)
}

/// Cycles' `bsdf_widen_dD` (`kernel/closure/bsdf.h`): `dD` widens to `scale · sqrt(avg roughness²)` when that is
/// larger, and keeps its value for a zero average, as a singular bounce has.
fn widen_ray_dd(ray_dd: f64, scale: f64, average_roughness_squared: f64) -> f64 {
    if average_roughness_squared > 0.0 {
        ray_dd.max(scale * average_roughness_squared.sqrt())
    } else {
        ray_dd
    }
}

/// The result of a reflective draw of the closure set. `Continuous` carries its throughput weight, the mixture density
/// that MIS weighs light sampling against, and the ray kind of the lobe it came from (`PATH_RAY_*`: a Burley subsurface
/// closure falls back to diffuse after a diffuse one). `Delta` is a sharp glass direction with its ray kind. `Subsurface`
/// enters the surface through the subsurface lobe of its method; the path leaves at that method's exit. `Absorbed` ends
/// the path.
enum Reflection {
    Continuous {
        direction: DVec3,
        weight: DVec3,
        pdf: f64,
        ray: RayKind,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
        ray: RayKind,
        /// The lobe mixture's pdf of the draw, Cycles' `bsdf_pdf` of a singular closure: its `1e6` weight plus the
        /// other lobes' densities at `direction`. It lowers `min_ray_pdf`.
        pdf: f64,
        /// Cycles' singular pdf for the widening average, `1e6 · branch · share`.
        mis_weight: f64,
    },
    Subsurface {
        weight: DVec3,
        method: SubsurfaceMethod,
    },
    Absorbed,
}

#[derive(Clone, Copy)]
struct LightSample {
    direction: DVec3,
    distance: f64,
    radiance: DVec3,
    /// Solid-angle density of `direction` under light sampling. Only set for hittable lights.
    pdf: f64,
}

/// A BSDF-sampled ray's meeting with an emitter, as Cycles' `lights_intersect` with
/// `light_eval_from_intersection`. `radiance` is before the MIS weight. `visible` is whether the
/// light's ray mask admits this ray's kind: a hidden light still ends the segment and spends the
/// transparent budget, but emits nothing (`integrate_light_forward`, `is_light_shader_visible_to_path`).
#[derive(Clone, Copy)]
struct LightHit {
    distance: f64,
    radiance: DVec3,
    pdf: f64,
    visible: bool,
}

#[derive(Clone)]
struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    fn for_sample(seed: u32, x: u32, y: u32, sample: u32) -> Self {
        let mut rng = Self {
            state: 0,
            increment: (mix64(u64::from(seed) ^ (u64::from(x) << 32) ^ u64::from(y)) << 1) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(mix64(
            u64::from(seed) << 32
                ^ u64::from(x).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ u64::from(y).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                ^ u64::from(sample).wrapping_mul(0x94d0_49bb_1331_11eb),
        ));
        let _ = rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.increment);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rotation = (old_state >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
}

struct WorldImportanceSampler {
    width: u32,
    height: u32,
    width_usize: usize,
    cumulative_weights: Vec<f64>,
    total_weight: f64,
}

impl WorldImportanceSampler {
    fn build(
        doc: &SceneDoc,
        scene_id: &Id,
        images: &BTreeMap<String, ImageData>,
    ) -> Result<Option<Self>> {
        let Some(image_id) = doc
            .scenes
            .get(scene_id)
            .and_then(|scene| scene.world.as_ref())
            .and_then(|id| doc.worlds.get(id))
            .and_then(|world| world.node_tree.as_ref())
            .and_then(|graph_id| doc.node_groups.get(graph_id))
            .and_then(|group| {
                group.nodes.values().find_map(|node| {
                    matches!(node.node_type.as_str(), "ShaderNodeTexEnvironment")
                        .then(|| {
                            node.properties
                                .get("image")
                                .or_else(|| node.properties.get("image_id"))
                                .and_then(serde_json::Value::as_str)
                        })
                        .flatten()
                })
            })
        else {
            return Ok(None);
        };
        let Some(image) = images.get(image_id) else {
            return Ok(None);
        };
        let width_usize = usize::try_from(image.width).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "environment image width is too large",
            )
        })?;
        let height_usize = usize::try_from(image.height).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "environment image height is too large",
            )
        })?;
        let pixel_count = width_usize.checked_mul(height_usize).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "environment image dimensions overflow",
            )
        })?;
        let mut cumulative_weights = Vec::new();
        cumulative_weights
            .try_reserve_exact(pixel_count)
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "environment sampling distribution exceeds available memory",
                )
            })?;
        let mut total_weight = 0.0;
        let environment_pixels = image.straight_pixels();
        for row in 0..height_usize {
            let latitude = std::f64::consts::PI
                * (f64::from(u32::try_from(row).unwrap_or(u32::MAX)) + 0.5)
                / f64::from(image.height);
            let solid_angle_weight = latitude.sin().max(f64::MIN_POSITIVE);
            for column in 0..width_usize {
                let pixel_index = row
                    .checked_mul(width_usize)
                    .and_then(|offset| offset.checked_add(column))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "environment pixel index overflows",
                        )
                    })?;
                let pixel = environment_pixels.get(pixel_index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "environment image pixel data is incomplete",
                    )
                })?;
                let luminance =
                    (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]).max(0.0);
                total_weight += (luminance + 1.0e-8) * solid_angle_weight;
                cumulative_weights.push(total_weight);
            }
        }
        Ok(
            (total_weight.is_finite() && total_weight > 0.0).then_some(Self {
                width: image.width,
                height: image.height,
                width_usize,
                cumulative_weights,
                total_weight,
            }),
        )
    }

    fn sample(&self, rng: &mut Pcg32) -> (DVec3, f64) {
        let target = rng.next_f64() * self.total_weight;
        let pixel_index = self
            .cumulative_weights
            .partition_point(|weight| *weight <= target)
            .min(self.cumulative_weights.len().saturating_sub(1));
        let row = pixel_index / self.width_usize;
        let column = pixel_index % self.width_usize;
        let u = (f64::from(u32::try_from(column).unwrap_or(u32::MAX)) + rng.next_f64())
            / f64::from(self.width);
        let v = (f64::from(u32::try_from(row).unwrap_or(u32::MAX)) + rng.next_f64())
            / f64::from(self.height);
        // Cycles' longitude: u = 0.5 - atan2(y, x) / 2π, so the column u gives longitude (0.5 - u)·2π.
        let longitude = (0.5 - u) * std::f64::consts::TAU;
        let latitude = v * std::f64::consts::PI;
        let direction = DVec3::new(
            latitude.sin() * longitude.cos(),
            latitude.sin() * longitude.sin(),
            latitude.cos(),
        );
        (direction, self.pdf(direction))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "normalized environment UVs are clamped to pixel bounds before conversion"
    )]
    fn pdf(&self, direction: DVec3) -> f64 {
        let direction = direction.normalize_or_zero();
        if direction == DVec3::ZERO {
            return 0.0;
        }
        // The importance map's rows run top-down, as the image's storage does: row 0 is the zenith, as `sample` lays them out.
        let [u, v_up] = equirectangular_uv(direction);
        let v = 1.0 - v_up;
        let column = (u * f64::from(self.width)).floor() as usize;
        let row = (v * f64::from(self.height)).floor() as usize;
        let pixel_index = row
            .min(usize::try_from(self.height.saturating_sub(1)).unwrap_or(usize::MAX))
            .saturating_mul(self.width_usize)
            .saturating_add(
                column.min(usize::try_from(self.width.saturating_sub(1)).unwrap_or(usize::MAX)),
            );
        let Some(weight) = self.cumulative_weights.get(pixel_index).copied() else {
            return 0.0;
        };
        let previous = pixel_index
            .checked_sub(1)
            .and_then(|index| self.cumulative_weights.get(index).copied())
            .unwrap_or(0.0);
        let pixel_probability = (weight - previous) / self.total_weight;
        let sin_latitude = (std::f64::consts::PI * v).sin().max(1.0e-8);
        let pixel_solid_angle = 2.0 * std::f64::consts::PI.powi(2) * sin_latitude
            / (f64::from(self.width) * f64::from(self.height));
        pixel_probability / pixel_solid_angle
    }
}

/// Stack size of a render worker, so recursive shader evaluation has the headroom of the main thread.
pub(super) const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

/// The read-only geometry a path trace reads. It has no interior mutability, so every worker shares it.
#[derive(Clone, Copy)]
pub(super) struct SharedGeometry<'a> {
    pub(super) triangles: &'a [Triangle],
    pub(super) materials: &'a [Material],
    pub(super) prepared: &'a [PreparedMaterial],
    pub(super) objects: &'a [ObjectRecord],
    pub(super) elements: &'a [ElementRecord],
    pub(super) shadows: Option<&'a super::scene::RealtimeShadows>,
}

impl<'a> SharedGeometry<'a> {
    pub(super) fn of(geometry: &'a Geometry) -> Self {
        Self {
            triangles: geometry.triangles.as_slice(),
            materials: geometry.materials.as_slice(),
            prepared: geometry.prepared.as_slice(),
            objects: geometry.objects.as_slice(),
            elements: geometry.elements.as_slice(),
            shadows: geometry.shadows.as_ref(),
        }
    }
}

/// A trace's view of the geometry: the shared part, and the sink its shading warnings are recorded in. Each worker
/// records into its own sink, and the sinks are merged in pixel order once the workers finish.
#[derive(Clone, Copy)]
pub(super) struct TraceGeometry<'a> {
    pub(super) shared: SharedGeometry<'a>,
    warnings: &'a RefCell<Vec<serde_json::Value>>,
}

impl<'a> TraceGeometry<'a> {
    fn of(geometry: &'a Geometry) -> Self {
        Self::with_warnings(SharedGeometry::of(geometry), &geometry.shader_warnings)
    }

    pub(super) fn with_warnings(
        shared: SharedGeometry<'a>,
        warnings: &'a RefCell<Vec<serde_json::Value>>,
    ) -> Self {
        Self { shared, warnings }
    }
}

impl<'a> Deref for TraceGeometry<'a> {
    type Target = SharedGeometry<'a>;

    fn deref(&self) -> &SharedGeometry<'a> {
        &self.shared
    }
}

/// What the workers of one render share. Every field is read-only, so the job is shared by reference.
struct RenderJob<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    geometry: SharedGeometry<'a>,
    bvh: &'a Bvh,
    lights: &'a [PointLight],
    volumes: &'a [VolumeInstance<'a>],
    volume_temperatures: &'a [Option<VolumeTemperature>],
    images: &'a BTreeMap<String, ImageData>,
    importance: Option<&'a WorldImportanceSampler>,
    pixel_filter: &'a PixelFilter,
    /// The camera and its frame for each eye: one for a mono camera, the left then the right eye for anaglyph.
    eyes: &'a [(Camera, CameraFrame)],
    anaglyph: bool,
    width: u32,
    height: u32,
    width_usize: usize,
    samples: u32,
    seed: u32,
    max_bounces: u32,
    volume_bounces: u32,
    transparent_max_bounces: u32,
    film_transparent: bool,
    environment: DVec3,
}

/// Renders the rows `rows` of the frame into `linear` and `rgba`, the bands of the output this worker owns. Its
/// shading warnings are recorded in `warnings`.
fn render_rows(
    job: &RenderJob<'_>,
    rows: Range<usize>,
    linear: &mut [[f32; 4]],
    rgba: &mut [u8],
    warnings: &RefCell<Vec<serde_json::Value>>,
) -> Result<()> {
    let geometry = TraceGeometry {
        shared: job.geometry,
        warnings,
    };
    let scene = SceneRefs {
        doc: job.doc,
        snapshot: job.snapshot,
        geometry: &geometry,
        bvh: job.bvh,
        lights: job.lights,
        volumes: job.volumes,
        images: job.images,
        importance: job.importance,
        camera_matrices: None,
    };
    let camera = job.eyes[0].0;
    let side_by_side = camera.stereo_mode == StereoMode::SideBySide;
    let left_camera = if side_by_side {
        camera_eye(&camera, -1.0)
    } else {
        camera
    };
    let first_camera_matrices = camera_matrices_for_render(&left_camera)?;
    let second_camera_matrices = if let Some((camera, _)) = job.eyes.get(1) {
        camera_matrices_for_render(camera)?
    } else if side_by_side {
        let right_camera = camera_eye(&camera, 1.0);
        camera_matrices_for_render(&right_camera)?
    } else {
        None
    };
    let bounces = Bounces {
        surface: job.max_bounces,
        volume: job.volume_bounces,
        transparent: job.transparent_max_bounces,
    };
    let mut evaluator_workspace = EvaluatorWorkspace::default();
    for (row_index, row) in rows.enumerate() {
        let y = u32::try_from(row).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "image height exceeds platform limits",
            )
        })?;
        for x in 0..job.width {
            let column = usize::try_from(x).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "image width exceeds platform limits",
                )
            })?;
            let index = row_index * job.width_usize + column;
            let mut sum = DVec3::ZERO;
            let mut alpha_sum = 0.0;
            for sample_index in 0..job.samples {
                let mut rng = Pcg32::for_sample(job.seed, x, y, sample_index);
                let filter_u = rng.next_f64();
                let filter_v = rng.next_f64();
                let (sample_x, sample_y) =
                    job.pixel_filter.sample(sample_index, filter_u, filter_v);
                let (radiance, alpha) = if job.anaglyph {
                    let (left_camera, left_frame) = &job.eyes[0];
                    let (right_camera, right_frame) = &job.eyes[1];
                    let left_ray = ray_for_sample_in_frame(
                        left_frame,
                        left_camera,
                        job.width,
                        job.height,
                        x,
                        y,
                        sample_x,
                        sample_y,
                    )?;
                    let right_ray = ray_for_sample_in_frame(
                        right_frame,
                        right_camera,
                        job.width,
                        job.height,
                        x,
                        y,
                        sample_x,
                        sample_y,
                    )?;
                    let mut left_rng = rng.clone();
                    let mut right_rng = rng.clone();
                    let right_camera_matrices = second_camera_matrices.as_ref();
                    // A sample with no ray (outside a fisheye's circle, or past its field of view) is opaque black: Blender's
                    // film writes a pixel with no ray as black with alpha 1 under either film (`probe_transparent_film_fisheye.py`).
                    let (left, left_alpha) = match left_ray {
                        Some((left_origin, left_direction)) => trace_path_with_temperatures(
                            scene.with_camera_matrices(first_camera_matrices.as_ref()),
                            job.volume_temperatures,
                            bounces,
                            left_origin,
                            left_direction,
                            camera_footprint(
                                left_frame,
                                left_camera,
                                job.width,
                                job.height,
                                x,
                                y,
                                sample_x,
                                sample_y,
                                (left_origin, left_direction),
                            )?,
                            job.environment,
                            job.film_transparent,
                            &mut left_rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    };
                    let (right, right_alpha) = match right_ray {
                        Some((right_origin, right_direction)) => trace_path_with_temperatures(
                            scene.with_camera_matrices(right_camera_matrices),
                            job.volume_temperatures,
                            bounces,
                            right_origin,
                            right_direction,
                            camera_footprint(
                                right_frame,
                                right_camera,
                                job.width,
                                job.height,
                                x,
                                y,
                                sample_x,
                                sample_y,
                                (right_origin, right_direction),
                            )?,
                            job.environment,
                            job.film_transparent,
                            &mut right_rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    };
                    (
                        DVec3::new(left.x, right.y, right.z),
                        f64::midpoint(left_alpha, right_alpha),
                    )
                } else {
                    let (camera, frame) = &job.eyes[0];
                    let camera_matrices =
                        if camera.stereo_mode == StereoMode::SideBySide && x >= job.width / 2 {
                            second_camera_matrices.as_ref()
                        } else {
                            first_camera_matrices.as_ref()
                        };
                    match ray_for_sample_in_frame(
                        frame, camera, job.width, job.height, x, y, sample_x, sample_y,
                    )? {
                        Some((origin, direction)) => trace_path_with_temperatures(
                            scene.with_camera_matrices(camera_matrices),
                            job.volume_temperatures,
                            bounces,
                            origin,
                            direction,
                            camera_footprint(
                                frame,
                                camera,
                                job.width,
                                job.height,
                                x,
                                y,
                                sample_x,
                                sample_y,
                                (origin, direction),
                            )?,
                            job.environment,
                            job.film_transparent,
                            &mut rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    }
                };
                sum += radiance;
                alpha_sum += alpha;
            }
            let sample_scale = 1.0 / f64::from(job.samples);
            let color = sum * sample_scale;
            let alpha = if job.film_transparent {
                alpha_sum * sample_scale
            } else {
                1.0
            };
            linear[index] = [
                finite_f32(color.x),
                finite_f32(color.y),
                finite_f32(color.z),
                finite_f32(alpha),
            ];
            let display_rgb = display_rgb_from_associated([color.x, color.y, color.z], alpha);
            rgba[index * 4..index * 4 + 4].copy_from_slice(&[
                to_srgb_byte(display_rgb[0]),
                to_srgb_byte(display_rgb[1]),
                to_srgb_byte(display_rgb[2]),
                (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
    }
    Ok(())
}

/// The contiguous row bands that split `height` rows across at most `band_count` workers, in row order. Each band has
/// `ceil(height / band_count)` rows except the last, and no band is empty.
pub(super) fn band_rows(height: usize, band_count: usize) -> Vec<Range<usize>> {
    let rows_per_band = height.div_ceil(band_count.max(1)).max(1);
    (0..height)
        .step_by(rows_per_band)
        .map(|start| start..(start + rows_per_band).min(height))
        .collect()
}

/// Appends one worker's shading warnings to `sink`, dropping those the serial render would not have recorded. A
/// Principled input is recorded once per (material, input) (`record_unevaluated_inputs`); the layer warnings once per
/// equal warning (`layer_index`, `color_layer_missing`). Workers record into their own sinks, so a duplicate across
/// workers is caught here, against the workers that came before in pixel order.
pub(super) fn merge_shader_warnings(
    sink: &RefCell<Vec<serde_json::Value>>,
    warnings: Vec<serde_json::Value>,
) {
    let mut recorded = sink.borrow_mut();
    for warning in warnings {
        let duplicate = match warning["input"].as_str() {
            Some(input) => recorded.iter().any(|existing| {
                existing["material"] == warning["material"]
                    && existing["input"].as_str() == Some(input)
            }),
            None => recorded.contains(&warning),
        };
        if !duplicate {
            recorded.push(warning);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "camera and sampling configuration are explicit at the render boundary"
)]
pub(super) fn render(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    camera: &Camera,
    width: u32,
    height: u32,
    samples: u32,
    seed: u32,
    max_bounces: u32,
    volume_bounces: u32,
    transparent_max_bounces: u32,
    film_transparent: bool,
    pixel_filter: &PixelFilter,
    images: &BTreeMap<String, ImageData>,
) -> Result<RasterOutput> {
    if samples == 0 {
        return Err(PotError::invalid_argument(
            "sample count must be greater than zero",
        ));
    }
    if max_bounces > 1024 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "path tracer bounce count must not exceed 1024",
        ));
    }
    if transparent_max_bounces > 1024 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "path tracer transparent bounce count must not exceed 1024",
        ));
    }
    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "render dimensions exceed platform limits",
            )
        })?;
    let rgba_len = pixel_count.checked_mul(4).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "render dimensions exceed platform limits",
        )
    })?;
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let world = scene
        .world
        .as_ref()
        .and_then(|world_id| doc.worlds.get(world_id))
        .map_or(DVec3::ZERO, |world| {
            DVec3::from_array(world.color) * world.strength
        });
    let bvh = Bvh::build(&geometry.triangles)?;
    let mut evaluator_workspace = EvaluatorWorkspace::default();
    let lights = collect_lights(doc, snapshot);
    let (volumes, volume_temperatures) = collect_volumes(
        doc,
        snapshot,
        true,
        scene.unit.scale_length,
        images,
        &geometry.shader_warnings,
        crate::node_kernels::MathSemantics::Cycles,
    )?;
    let world_importance = WorldImportanceSampler::build(doc, &snapshot.scene_id, images)?;
    let mut output = RasterOutput {
        rgba: allocate(rgba_len, 0_u8, "RGBA image")?,
        linear_rgba: allocate(pixel_count, [0.0_f32; 4], "linear RGBA image")?,
        // Path RGB is associated surface radiance plus alpha-independent visible-light radiance.
        linear_rgba_associated: true,
        ids: allocate(pixel_count, 0_u32, "object ID buffer")?,
        depths: allocate(pixel_count, 0.0_f32, "depth buffer")?,
        elements: allocate(pixel_count, 0_u32, "element ID buffer")?,
        normals: allocate(pixel_count, [0.0_f32; 3], "normal pass")?,
        albedo: allocate(pixel_count, [0.0_f32; 4], "albedo pass")?,
        emission: allocate(pixel_count, [0.0_f32; 4], "emission pass")?,
        ambient_occlusion: allocate(pixel_count, 1.0_f32, "ambient occlusion pass")?,
    };
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let environment = world.max(DVec3::ZERO);

    if pixel_count > 0 {
        let anaglyph = camera.stereo_mode == StereoMode::Anaglyph;
        let eyes = if anaglyph {
            let left_camera = camera_eye(camera, -1.0);
            let right_camera = camera_eye(camera, 1.0);
            vec![
                (left_camera, camera_frame(&left_camera, width, height)?),
                (right_camera, camera_frame(&right_camera, width, height)?),
            ]
        } else {
            vec![(*camera, camera_frame(camera, width, height)?)]
        };
        let height_usize = usize::try_from(height).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "image height exceeds platform limits",
            )
        })?;
        let job = RenderJob {
            doc,
            snapshot,
            geometry: SharedGeometry::of(geometry),
            bvh: &bvh,
            lights: &lights,
            volumes: &volumes,
            volume_temperatures: &volume_temperatures,
            images,
            importance: world_importance.as_ref(),
            pixel_filter,
            eyes: &eyes,
            anaglyph,
            width,
            height,
            width_usize,
            samples,
            seed,
            max_bounces,
            volume_bounces,
            transparent_max_bounces,
            film_transparent,
            environment,
        };
        let bands = band_rows(
            height_usize,
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        );
        let outcomes = std::thread::scope(|scope| {
            let job = &job;
            let mut workers = Vec::with_capacity(bands.len());
            let mut linear_rest = output.linear_rgba.as_mut_slice();
            let mut rgba_rest = output.rgba.as_mut_slice();
            for rows in bands {
                let band_len = rows.len() * width_usize;
                let (linear, linear_tail) = std::mem::take(&mut linear_rest).split_at_mut(band_len);
                let (rgba, rgba_tail) = std::mem::take(&mut rgba_rest).split_at_mut(band_len * 4);
                linear_rest = linear_tail;
                rgba_rest = rgba_tail;
                workers.push(
                    std::thread::Builder::new()
                        .stack_size(WORKER_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            let warnings = RefCell::new(Vec::new());
                            let result = render_rows(job, rows, linear, rgba, &warnings);
                            (result, warnings.into_inner())
                        }),
                );
            }
            workers
                .into_iter()
                .map(|worker| match worker {
                    Ok(handle) => handle.join().unwrap_or_else(|_| {
                        (
                            Err(PotError::new(
                                ErrorCode::InternalError,
                                "render worker panicked",
                            )),
                            Vec::new(),
                        )
                    }),
                    Err(error) => (
                        Err(PotError::new(
                            ErrorCode::InternalError,
                            format!("cannot start a render worker: {error}"),
                        )),
                        Vec::new(),
                    ),
                })
                .collect::<Vec<_>>()
        });
        for (result, warnings) in outcomes {
            merge_shader_warnings(&geometry.shader_warnings, warnings);
            result?;
        }
    }

    write_center_passes(
        doc,
        &TraceGeometry::of(geometry),
        &bvh,
        &volumes,
        images,
        camera,
        width,
        height,
        film_transparent,
        &mut output,
        &mut evaluator_workspace,
    )?;
    overlay_rasterized_lines(
        geometry,
        camera,
        width,
        height,
        seed,
        film_transparent,
        &mut output,
    )?;
    Ok(output)
}

/// Per-render camera light data. Realtime volume transport already owns the collected point lights; otherwise the
/// camera pass collects them once here.
enum RealtimeCameraLights<'a> {
    Borrowed(&'a [PointLight]),
    Owned(Vec<PointLight>),
}

enum RealtimeCameraLightInput<'a> {
    Borrowed(&'a [PointLight], &'a [CameraLightPrepared]),
    Owned(Vec<PointLight>, Vec<CameraLightPrepared>),
}

#[derive(Clone, Copy)]
struct CameraLightPrepared {
    influence_radius_max: f64,
    area_display: Option<AreaShape>,
}

pub(super) struct RealtimeCameraLightContext<'a> {
    lights: RealtimeCameraLights<'a>,
    cameras: [Camera; 2],
    frames: [CameraFrame; 2],
    view_widths: [u32; 2],
    height: u32,
    culling: Vec<CameraLightCull>,
    suns: Vec<SunDisc>,
}

struct CameraLightCull {
    area_display: Option<AreaShape>,
    visible: [bool; 2],
    point_display: [Option<NativeCameraLightDisplay>; 2],
}

struct NativeCameraLightDisplay {
    center: DVec3,
    plane_normal: DVec3,
    positions: [DVec3; 4],
    quad: Option<CameraLightProjectedQuad>,
}

pub(super) struct CameraLightRasterSamples {
    rasters: Vec<Option<CameraLightRasterQuad>>,
    projected: Option<Vec<Option<CameraLightProjectedQuad>>>,
    sample_count: usize,
    first_eye: usize,
    eye_count: usize,
    light_count: usize,
}

impl CameraLightRasterSamples {
    fn shape_for(
        &self,
        sample_index: u32,
        eye: usize,
        light_index: usize,
    ) -> Option<(&CameraLightRasterQuad, Option<&CameraLightProjectedQuad>)> {
        let sample_index = usize::try_from(sample_index).ok()?;
        if sample_index >= self.sample_count
            || eye < self.first_eye
            || eye >= self.first_eye + self.eye_count
            || light_index >= self.light_count
        {
            return None;
        }
        let eye_index = eye - self.first_eye;
        let index = (sample_index * self.eye_count + eye_index) * self.light_count + light_index;
        let raster = self.rasters.get(index)?.as_ref()?;
        let projected = self
            .projected
            .as_ref()
            .and_then(|projected| projected.get(index))
            .and_then(Option::as_ref);
        Some((raster, projected))
    }
}

pub(super) struct RealtimeCameraLightSample<'context, 'lights, 'samples> {
    view: RealtimeCameraLightView<'context, 'lights>,
    raster_samples: Option<&'samples CameraLightRasterSamples>,
    sample_index: u32,
    jitter_x: f64,
    jitter_y: f64,
    pixel_x: u32,
    pixel_y: u32,
}
impl RealtimeCameraLightContext<'_> {
    fn local_events<'a>(
        &'a self,
        eye: usize,
        origin: DVec3,
        direction: DVec3,
        far_distance: f64,
        raster_samples: Option<&'a CameraLightRasterSamples>,
        sample_index: u32,
        jitter_x: f64,
        jitter_y: f64,
        pixel_x: u32,
        pixel_y: u32,
    ) -> impl Iterator<Item = CameraLightEvent> + 'a {
        self.lights()
            .iter()
            .zip(&self.culling)
            .enumerate()
            .filter_map(move |(light_index, (light, culling))| {
                if !culling.visible[eye] {
                    return None;
                }
                if let Some(area_display) = culling.area_display {
                    return camera_light_shape_hit_projected(
                        light,
                        Some(area_display),
                        None,
                        None,
                        None,
                        origin,
                        direction,
                        far_distance,
                        (pixel_x, pixel_y),
                    );
                }
                let display = culling.point_display[eye].as_ref()?;
                if let Some(samples) = raster_samples {
                    let (raster, projected) = samples.shape_for(sample_index, eye, light_index)?;
                    let quad = projected.or(display.quad.as_ref())?;
                    return camera_light_shape_hit_projected(
                        light,
                        None,
                        Some(display),
                        Some(quad),
                        Some(raster),
                        origin,
                        direction,
                        far_distance,
                        (pixel_x, pixel_y),
                    );
                }

                let projected = match self.frames[eye].lens_sample(jitter_x, jitter_y) {
                    Some(lens_sample) if !lens_sample.is_zero() => Some(project_camera_light_quad(
                        self.cameras[eye].position,
                        &self.frames[eye],
                        self.view_widths[eye],
                        self.height,
                        &display.positions,
                        Some(lens_sample),
                    )?),
                    _ => None,
                };
                let quad = projected.as_ref().or(display.quad.as_ref())?;
                let raster = quad.rasterize(jitter_x, jitter_y);
                camera_light_shape_hit_projected(
                    light,
                    None,
                    Some(display),
                    Some(quad),
                    Some(&raster),
                    origin,
                    direction,
                    far_distance,
                    (pixel_x, pixel_y),
                )
            })
    }
}

#[derive(Clone, Copy)]
pub(super) struct RealtimeCameraLightView<'context, 'lights> {
    context: &'context RealtimeCameraLightContext<'lights>,
    eye: usize,
}

#[derive(Clone, Copy)]
pub(super) struct CameraLightEvent {
    pub(super) distance: f64,
    pub(super) radiance: DVec3,
}

impl<'lights> RealtimeCameraLightContext<'lights> {
    fn lights(&self) -> &[PointLight] {
        match &self.lights {
            RealtimeCameraLights::Borrowed(lights) => lights,
            RealtimeCameraLights::Owned(lights) => lights,
        }
    }

    pub(super) fn view(&self, eye: usize) -> RealtimeCameraLightView<'_, 'lights> {
        RealtimeCameraLightView {
            context: self,
            eye: eye.min(1),
        }
    }

    pub(super) fn sun_radiance(&self, direction: DVec3) -> DVec3 {
        self.suns
            .iter()
            .filter(|sun| direction.dot(sun.to_sun) >= sun.cos_half_angle)
            .fold(DVec3::ZERO, |radiance, sun| radiance + sun.radiance)
    }
}

impl<'context, 'lights> RealtimeCameraLightView<'context, 'lights> {
    pub(super) fn for_eye(self, eye: usize) -> Self {
        Self {
            eye: eye.min(1),
            ..self
        }
    }

    pub(super) fn prepare_raster_samples(
        &self,
        sample_offsets: impl ExactSizeIterator<Item = (f64, f64)>,
        both_eyes: bool,
    ) -> Result<CameraLightRasterSamples> {
        let first_eye = if both_eyes { 0 } else { self.eye };
        let eye_count = if both_eyes { 2 } else { 1 };
        let sample_count = sample_offsets.len();
        let light_count = self.context.culling.len();
        let shape_count = sample_count
            .checked_mul(eye_count)
            .and_then(|count| count.checked_mul(light_count))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "camera light sample cache exceeds platform limits",
                )
            })?;
        let mut rasters = Vec::new();
        rasters.try_reserve_exact(shape_count).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "cannot allocate camera light sample cache",
            )
        })?;
        rasters.resize_with(shape_count, || None);
        let has_aperture = (first_eye..first_eye + eye_count)
            .any(|eye| self.context.frames[eye].lens_sample(0.5, 0.5).is_some());
        let mut projected = if has_aperture {
            let mut projected = Vec::new();
            projected.try_reserve_exact(shape_count).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "cannot allocate camera light sample cache",
                )
            })?;
            projected.resize_with(shape_count, || None);
            Some(projected)
        } else {
            None
        };
        for (sample_index, (jitter_x, jitter_y)) in sample_offsets.enumerate() {
            for eye_offset in 0..eye_count {
                let eye = first_eye + eye_offset;
                let frame = &self.context.frames[eye];
                let lens_sample = frame.lens_sample(jitter_x, jitter_y);
                for (light_index, culling) in self.context.culling.iter().enumerate() {
                    let Some(display) = culling.point_display[eye].as_ref() else {
                        continue;
                    };
                    let index = (sample_index * eye_count + eye_offset) * light_count + light_index;
                    if let Some(lens_sample) = lens_sample.filter(|sample| !sample.is_zero()) {
                        let Some(quad) = project_camera_light_quad(
                            self.context.cameras[eye].position,
                            frame,
                            self.context.view_widths[eye],
                            self.context.height,
                            &display.positions,
                            Some(lens_sample),
                        ) else {
                            continue;
                        };
                        rasters[index] = Some(quad.rasterize(jitter_x, jitter_y));
                        if let Some(projected) = &mut projected {
                            projected[index] = Some(quad);
                        }
                    } else {
                        let Some(quad) = display.quad.as_ref() else {
                            continue;
                        };
                        rasters[index] = Some(quad.rasterize(jitter_x, jitter_y));
                    }
                }
            }
        }
        Ok(CameraLightRasterSamples {
            rasters,
            projected,
            sample_count,
            first_eye,
            eye_count,
            light_count,
        })
    }

    pub(super) fn at_pixel<'samples>(
        self,
        raster_samples: Option<&'samples CameraLightRasterSamples>,
        sample_index: u32,
        jitter_x: f64,
        jitter_y: f64,
        pixel_x: u32,
        pixel_y: u32,
    ) -> RealtimeCameraLightSample<'context, 'lights, 'samples> {
        RealtimeCameraLightSample {
            view: self,
            raster_samples,
            sample_index,
            jitter_x,
            jitter_y,
            pixel_x,
            pixel_y,
        }
    }
}

impl RealtimeCameraLightSample<'_, '_, '_> {
    pub(super) fn local_events(
        &self,
        origin: DVec3,
        direction: DVec3,
        far_distance: f64,
    ) -> impl Iterator<Item = CameraLightEvent> + '_ {
        self.view.context.local_events(
            self.view.eye,
            origin,
            direction,
            far_distance,
            self.raster_samples,
            self.sample_index,
            self.jitter_x,
            self.jitter_y,
            self.pixel_x,
            self.pixel_y,
        )
    }

    pub(super) fn sun_radiance(&self, direction: DVec3) -> DVec3 {
        self.view.context.sun_radiance(direction)
    }
}

pub(super) fn prepare_realtime_camera_light_context<'a>(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    transport: Option<&'a RealtimeVolumeTransport<'_>>,
    camera: &Camera,
    width: u32,
    height: u32,
) -> Result<RealtimeCameraLightContext<'a>> {
    let light_input = transport.map_or_else(
        || {
            let (lights, prepared) = collect_lights_with_camera_data(doc, snapshot);
            RealtimeCameraLightInput::Owned(lights, prepared)
        },
        |transport| {
            RealtimeCameraLightInput::Borrowed(&transport.lights, &transport.camera_light_prepared)
        },
    );
    let light_slice = match &light_input {
        RealtimeCameraLightInput::Borrowed(lights, _) => *lights,
        RealtimeCameraLightInput::Owned(lights, _) => lights,
    };
    let prepared = match &light_input {
        RealtimeCameraLightInput::Borrowed(_, prepared) => *prepared,
        RealtimeCameraLightInput::Owned(_, prepared) => prepared,
    };
    let cameras = camera_light_cameras(camera);
    let view_widths = if camera.stereo_mode == StereoMode::SideBySide {
        [width / 2, width - width / 2]
    } else {
        [width, width]
    };
    let frame_width = if camera.stereo_mode == StereoMode::SideBySide {
        width / 2
    } else {
        width
    };
    let frames = [
        camera_frame(&cameras[0], frame_width, height)?,
        camera_frame(&cameras[1], frame_width, height)?,
    ];
    let frusta = camera_light_frusta(camera, &cameras, width, height)?;
    let culling = light_slice
        .iter()
        .zip(prepared.iter().copied())
        .map(|(light, prepared)| {
            let visible = [
                camera_light_visible_in_view(
                    light,
                    prepared.influence_radius_max,
                    frusta[0].as_ref(),
                ),
                camera_light_visible_in_view(
                    light,
                    prepared.influence_radius_max,
                    frusta[1].as_ref(),
                ),
            ];
            let point_display = std::array::from_fn(|eye| {
                (visible[eye]
                    && prepared.area_display.is_none()
                    && matches!(light.kind, LightType::Point | LightType::Spot))
                .then(|| {
                    native_camera_light_display(
                        light,
                        &cameras[eye],
                        &frames[eye],
                        view_widths[eye],
                        height,
                    )
                })
                .flatten()
            });
            CameraLightCull {
                area_display: prepared.area_display,
                visible,
                point_display,
            }
        })
        .collect();
    let lights = match light_input {
        RealtimeCameraLightInput::Borrowed(lights, _) => RealtimeCameraLights::Borrowed(lights),
        RealtimeCameraLightInput::Owned(lights, _) => RealtimeCameraLights::Owned(lights),
    };
    Ok(RealtimeCameraLightContext {
        lights,
        cameras,
        frames,
        view_widths,
        height,
        culling,
        suns: camera_sun_discs(doc, snapshot),
    })
}

fn camera_light_cameras(camera: &Camera) -> [Camera; 2] {
    match camera.stereo_mode {
        StereoMode::None => [*camera, *camera],
        StereoMode::SideBySide | StereoMode::Anaglyph => {
            [camera_eye(camera, -1.0), camera_eye(camera, 1.0)]
        }
    }
}

fn camera_light_frusta(
    camera: &Camera,
    cameras: &[Camera; 2],
    width: u32,
    height: u32,
) -> Result<[Option<CameraFrustum>; 2]> {
    let mut frusta = [None, None];
    match camera.stereo_mode {
        StereoMode::None => frusta[0] = camera_frustum(&cameras[0], width, height)?,
        StereoMode::SideBySide => {
            let split = width / 2;
            frusta[0] = camera_frustum(&cameras[0], split, height)?;
            frusta[1] = camera_frustum(&cameras[1], split, height)?;
        }
        StereoMode::Anaglyph => {
            frusta[0] = camera_frustum(&cameras[0], width, height)?;
            frusta[1] = camera_frustum(&cameras[1], width, height)?;
        }
    }
    Ok(frusta)
}

fn camera_light_visible_in_view(
    light: &PointLight,
    influence_radius_max: f64,
    frustum: Option<&CameraFrustum>,
) -> bool {
    if !light.camera_shape_eligible
        || light.ray_mask & RayKind::Camera.bit() == 0
        || influence_radius_max < 1.0e-8
    {
        return false;
    }
    let Some(frustum) = frustum else {
        return true;
    };
    if light.kind == LightType::Spot
        && let Some(pyramid) = camera_light_spot_pyramid(light, influence_radius_max)
        && !frustum.intersects_pyramid(pyramid)
    {
        return false;
    }
    frustum.intersects_sphere(light.position, influence_radius_max)
}

fn camera_light_spot_pyramid(light: &PointLight, influence_radius_max: f64) -> Option<[DVec3; 5]> {
    let spot_tan = (light.spot_size * 0.5)
        .min(std::f64::consts::FRAC_PI_2 - 1.0e-4)
        .tan();
    if !influence_radius_max.is_finite()
        || !spot_tan.is_finite()
        || spot_tan <= 0.0
        || spot_tan >= 10.0
    {
        return None;
    }
    let transform = light.world_transform;
    let mut x_axis = transform.x_axis.truncate();
    let mut y_axis = transform.y_axis.truncate();
    let mut z_axis = transform.z_axis.truncate();
    let scale_x = x_axis.length();
    let scale_y = y_axis.length();
    let scale_z = z_axis.length();
    if !scale_x.is_finite()
        || !scale_y.is_finite()
        || !scale_z.is_finite()
        || scale_x <= 1.0e-8
        || scale_y <= 1.0e-8
        || scale_z <= 1.0e-8
    {
        return None;
    }
    x_axis = x_axis.normalize_or_zero();
    y_axis = y_axis.normalize_or_zero();
    z_axis = z_axis.normalize_or_zero();
    if x_axis.cross(y_axis).dot(z_axis) < 0.0 {
        y_axis = -y_axis;
    }
    let spot_size_inv_x = scale_z / scale_x.max(1.0e-8);
    let spot_size_inv_y = scale_z / scale_y.max(1.0e-8);
    let half_x = x_axis * (influence_radius_max * spot_tan / spot_size_inv_x);
    let half_y = y_axis * (influence_radius_max * spot_tan / spot_size_inv_y);
    let base = light.position - z_axis * influence_radius_max;
    Some([
        light.position,
        base - half_x - half_y,
        base + half_x - half_y,
        base + half_x + half_y,
        base - half_x + half_y,
    ])
}

/// A sun light's disc, as EEVEE draws it (`eevee_light_shape_display.bsl.hh`, `shape_display_light_radiance_get`).
struct SunDisc {
    /// The unit direction towards the sun.
    to_sun: DVec3,
    /// The cosine of the disc's half angle: `1 / sqrt(1 + r²)` for `r = tan(angle / 2)` (`cos_from_tan`).
    cos_half_angle: f64,
    /// The disc's radiance, `color · (1 + 1 / r²) / π` (`M_1_PI * inv_sin_sq`), with the colour scaled by energy.
    radiance: DVec3,
}

/// The visible sun lights, with the disc EEVEE draws for each: its direction is the light's local +Z
/// (`l_sun.direction = z_axis`), which points away from the shine, and its radius is clamped to `[0.001, 20]`
/// (`eevee_light.cc`, `shape_radius`), with the angle capped at 179.9°.
fn camera_sun_discs(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<SunDisc> {
    doc.nodes
        .iter()
        .filter(|(node_id, node)| {
            node.kind == "light"
                && snapshot.node_is_visible(node_id, node)
                && crate::exchange::evaluated_visibility(snapshot, node_id, node).1
                && node.resolved_ray_visibility().allows(RayKind::Camera)
        })
        .filter_map(|(node_id, node)| {
            let evaluated = snapshot.nodes.get(node_id)?;
            let data = evaluated.evaluated_light.as_ref().or_else(|| {
                node.data
                    .as_ref()
                    .and_then(|data_id| doc.data_blocks.get(data_id))
                    .and_then(|data| data.light.as_ref())
            })?;
            if data.light_type != LightType::Sun {
                return None;
            }
            let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
            let to_sun = matrix.transform_vector3(DVec3::Z).normalize_or_zero();
            let half_angle = data.sun_angle.min(179.9_f64.to_radians()) / 2.0;
            let radius = half_angle.tan().clamp(0.001, 20.0);
            let cos_half_angle = 1.0 / (1.0 + radius * radius).sqrt();
            let shape = (1.0 + 1.0 / (radius * radius)) / std::f64::consts::PI;
            let colour = DVec3::from_array(data.color).max(DVec3::ZERO);
            Some(SunDisc {
                to_sun,
                cos_half_angle,
                radiance: colour * (data.energy.max(0.0) * shape),
            })
        })
        .collect()
}
/// The camera-visible local-light shape from EEVEE's `shape_display_light_position_get` and
/// `shape_display_light_radiance_get`. This intentionally does not apply Cycles' physical spread
/// eligibility or EEVEE's diffuse/specular/transmission/volume power factors.
#[cfg(test)]
fn camera_light_shape_hit(
    light: &PointLight,
    camera: &Camera,
    origin: DVec3,
    direction: DVec3,
    limit: f64,
) -> Option<CameraLightEvent> {
    let area_display = if light.camera_shape_eligible {
        light.area.map(camera_light_display_area)
    } else {
        None
    };
    camera_light_shape_hit_with_area(light, area_display, camera, origin, direction, limit)
}

fn camera_light_display_area(shape: AreaShape) -> AreaShape {
    AreaShape::new(
        shape.axis_u,
        shape.axis_v,
        shape.half_u.max(0.003),
        shape.half_v.max(0.003),
        shape.ellipse,
    )
}

fn native_camera_light_display(
    light: &PointLight,
    camera: &Camera,
    frame: &CameraFrame,
    width: u32,
    height: u32,
) -> Option<NativeCameraLightDisplay> {
    let (center, plane_normal, positions, quad) = camera_light_display_quad(
        camera.position,
        frame,
        light.position,
        light.radius,
        light.soft_falloff,
        width,
        height,
    )?;
    Some(NativeCameraLightDisplay {
        center,
        plane_normal,
        positions,
        quad,
    })
}

fn camera_light_shape_hit_projected(
    light: &PointLight,
    area_display: Option<AreaShape>,
    point_display: Option<&NativeCameraLightDisplay>,
    projected_quad: Option<&CameraLightProjectedQuad>,
    raster_quad: Option<&CameraLightRasterQuad>,
    origin: DVec3,
    direction: DVec3,
    limit: f64,
    pixel: (u32, u32),
) -> Option<CameraLightEvent> {
    if !light.camera_shape_eligible || light.ray_mask & RayKind::Camera.bit() == 0 {
        return None;
    }

    if let Some(display_shape) = area_display {
        let cos_light = -light.direction.dot(direction);
        if display_shape.area <= 0.0 || cos_light <= 0.0 {
            return None;
        }
        let (distance, point) = ray_plane_intersection(
            origin,
            direction,
            light.position,
            light.direction,
            cos_light,
            limit,
        )?;
        if !inside_emitter(&display_shape, light.position, point) {
            return None;
        }
        return Some(CameraLightEvent {
            distance,
            radiance: light.color * area_radiance(light.energy, display_shape.area),
        });
    }

    if !matches!(light.kind, LightType::Point | LightType::Spot) {
        return None;
    }
    let display = point_display?;
    let cos_light = -display.plane_normal.dot(direction);
    let (distance, _) = ray_plane_intersection(
        origin,
        direction,
        display.center,
        display.plane_normal,
        cos_light,
        limit,
    )?;
    let projected_quad = projected_quad?;
    let raster_quad = raster_quad?;
    if !projected_quad.contains_pixel(raster_quad, pixel.0, pixel.1) {
        return None;
    }
    let spot = if light.kind == LightType::Spot {
        spot_cone(
            &light.world_transform,
            direction,
            light.spot_size,
            light.spot_blend,
        )
    } else {
        1.0
    };
    Some(CameraLightEvent {
        distance,
        radiance: light.color * (emitter_radiance(light.energy, light.radius.max(0.001)) * spot),
    })
}

#[cfg(test)]
fn camera_light_shape_hit_with_area(
    light: &PointLight,
    area_display: Option<AreaShape>,
    camera: &Camera,
    origin: DVec3,
    direction: DVec3,
    limit: f64,
) -> Option<CameraLightEvent> {
    if !light.camera_shape_eligible || light.ray_mask & RayKind::Camera.bit() == 0 {
        return None;
    }

    if let Some(display_shape) = area_display {
        let cos_light = -light.direction.dot(direction);
        if display_shape.area <= 0.0 || cos_light <= 0.0 {
            return None;
        }
        let (distance, point) = ray_plane_intersection(
            origin,
            direction,
            light.position,
            light.direction,
            cos_light,
            limit,
        )?;
        if !inside_emitter(&display_shape, light.position, point) {
            return None;
        }
        return Some(CameraLightEvent {
            distance,
            radiance: light.color * area_radiance(light.energy, display_shape.area),
        });
    }

    if !matches!(light.kind, LightType::Point | LightType::Spot) {
        return None;
    }
    let emitter_radius = light.radius.max(0.001);
    let camera_to_light = light.position - camera.position;
    let camera_distance = camera_to_light.length();
    let plane_normal = if camera_distance > 1.0e-8 {
        camera_to_light / camera_distance
    } else {
        (camera.target - camera.position).normalize_or_zero()
    };
    let display_radius = if light.soft_falloff {
        emitter_radius
    } else if camera_distance > emitter_radius {
        light_sphere_disk_radius(emitter_radius, camera_distance)
    } else {
        camera.far
    };
    let cos_light = -plane_normal.dot(direction);
    let (distance, point) = ray_plane_intersection(
        origin,
        direction,
        light.position,
        plane_normal,
        cos_light,
        limit,
    )?;
    if (point - light.position).length_squared() > display_radius * display_radius {
        return None;
    }
    let spot = if light.kind == LightType::Spot {
        spot_cone(
            &light.world_transform,
            direction,
            light.spot_size,
            light.spot_blend,
        )
    } else {
        1.0
    };
    Some(CameraLightEvent {
        distance,
        radiance: light.color * (emitter_radiance(light.energy, emitter_radius) * spot),
    })
}

/// Apparent disk radius of a sphere at `distance`, from `eevee_light_lib.bsl.hh`.
#[cfg(test)]
fn light_sphere_disk_radius(radius: f64, distance: f64) -> f64 {
    let ratio = radius / distance;
    radius / (1.0 - ratio * ratio).max(1.0e-8).sqrt()
}

#[cfg(test)]
mod camera_light_shape_tests {
    use std::error::Error;

    use glam::{DMat4, DQuat, DVec3};

    use super::{
        PointLight, camera_light_shape_hit, camera_sun_discs, eevee_volume_light_sample, light_hit,
    };
    use crate::{
        eval::{EvaluationContext, Snapshot},
        model::{DataBlock, Id, LightData, LightType, Node, RayKind, SceneDoc},
        render::{
            light_model::{AreaShape, Spread, area_radiance},
            raster::{Camera, Projection, StereoMode},
        },
    };

    fn make_camera(position: DVec3, target: DVec3) -> Camera {
        Camera {
            position,
            target,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 8.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.0,
        }
    }

    fn local_light(kind: LightType, position: DVec3, direction: DVec3, radius: f64) -> PointLight {
        PointLight {
            position,
            direction,
            world_transform: DMat4::from_rotation_translation(
                DQuat::from_rotation_arc(-DVec3::Z, direction.normalize_or_zero()),
                position,
            ),
            kind,
            color: DVec3::ONE,
            energy: 1.0,
            radius,
            soft_falloff: false,
            spot_size: 1.0,
            spot_blend: 0.0,
            area: None,
            spread: None,
            use_mis: true,
            volume_factor: 0.0,
            volume_radius_squared: 0.0,
            ray_mask: RayKind::Camera.bit(),
            camera_shape_eligible: true,
        }
    }

    #[test]
    fn eevee_area_volume_lights_sample_the_closest_emitter_point() -> Result<(), Box<dyn Error>> {
        let mut light = local_light(LightType::Area, DVec3::new(0.0, 0.0, 3.0), -DVec3::Z, 0.0);
        light.area = Some(AreaShape::new(DVec3::X, DVec3::Y, 0.75, 0.75, true));
        light.volume_factor = 1.0;
        light.volume_radius_squared = f64::INFINITY;

        let sample = eevee_volume_light_sample(&light, DVec3::new(0.2, 0.1, 0.0))
            .ok_or("area volume-light sample is missing")?;
        assert!((sample.distance - 3.0).abs() <= f64::EPSILON);
        assert!(sample.direction.abs_diff_eq(DVec3::Z, 1.0e-12));
        Ok(())
    }

    #[test]
    fn point_light_is_not_a_camera_sun_disc() -> Result<(), Box<dyn Error>> {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node_id = Id::new("point_light")?;
        let data_id = Id::new("point_light_data")?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "light".to_owned(),
                light: Some(LightData {
                    light_type: LightType::Point,
                    ..LightData::default()
                }),
                ..DataBlock::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("root collection is missing")?
            .objects
            .push(node_id.clone());
        doc.nodes.insert(
            node_id,
            Node {
                name: "Point Light".to_owned(),
                kind: "light".to_owned(),
                data: Some(data_id),
                ..Node::default()
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        assert!(camera_sun_discs(&doc, &snapshot).is_empty());
        Ok(())
    }

    #[test]
    fn hard_point_display_uses_center_plane_and_inside_far_disk() -> Result<(), Box<dyn Error>> {
        let camera = make_camera(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO);
        let point = local_light(LightType::Point, DVec3::ZERO, DVec3::Z, 1.0);
        let origin = DVec3::new(0.0, 0.0, 10.0);
        let direction = -DVec3::Z;
        let shape = camera_light_shape_hit(&point, &camera, origin, direction, camera.far)
            .ok_or("point camera shape was not hit")?;
        let physical = light_hit(&point, origin, direction, origin, camera.far)
            .ok_or("physical point sphere was not hit")?;
        assert!((shape.distance - 10.0).abs() < 1.0e-12);
        assert!((physical.0 - 9.0).abs() < 1.0e-12);

        let hard_edge = camera_light_shape_hit(
            &point,
            &camera,
            DVec3::new(1.003, 0.0, 10.0),
            direction,
            camera.far,
        );
        assert!(
            hard_edge.is_some(),
            "the hard sphere's apparent radius expands in projection"
        );
        let spot = local_light(LightType::Spot, DVec3::ZERO, DVec3::Z, 1.0);
        let spot_shape = camera_light_shape_hit(&spot, &camera, origin, direction, camera.far)
            .ok_or("spot camera disk was not hit")?;
        assert!((spot_shape.distance - 10.0).abs() < 1.0e-12);

        let inside_camera = make_camera(DVec3::new(0.0, 0.0, 0.5), DVec3::ZERO);
        let off_axis_origin = DVec3::new(5.0, 0.0, 0.5);
        let inside_shape = camera_light_shape_hit(
            &point,
            &inside_camera,
            off_axis_origin,
            -DVec3::Z,
            inside_camera.far,
        )
        .ok_or("inside-sphere camera disk was not hit")?;
        assert!((inside_shape.distance - 0.5).abs() < 1.0e-12);
        assert!(inside_shape.radiance.x > 0.0);
        Ok(())
    }

    #[test]
    fn camera_spot_shape_uses_the_light_object_scale() -> Result<(), Box<dyn Error>> {
        let angle: f64 = 0.4;
        let position = DVec3::new(10.0 * angle.sin(), 0.0, 10.0 * angle.cos());
        let camera = make_camera(position, DVec3::ZERO);
        let origin = position;
        let direction = -position.normalize();
        let spot = local_light(LightType::Spot, DVec3::ZERO, DVec3::Z, 1.0);
        let unscaled = camera_light_shape_hit(&spot, &camera, origin, direction, camera.far)
            .ok_or("unscaled spot camera disk was not hit")?;
        let scaled = PointLight {
            world_transform: spot.world_transform * DMat4::from_scale(DVec3::new(1.0, 1.0, 2.0)),
            ..spot
        };
        let scaled_event = camera_light_shape_hit(&scaled, &camera, origin, direction, camera.far)
            .ok_or("scaled spot camera disk was not hit")?;
        assert!(unscaled.radiance.x > 0.0);
        assert_eq!(scaled_event.radiance, DVec3::ZERO);
        Ok(())
    }

    #[test]
    fn area_camera_shape_is_invariant_to_physical_spread() -> Result<(), Box<dyn Error>> {
        let shape = AreaShape::new(DVec3::X, DVec3::Y, 1.0, 1.0, false);
        let spread = Spread::new(0.0).ok_or("zero spread should be representable")?;
        let area = PointLight {
            position: DVec3::ZERO,
            direction: DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Area,
            color: DVec3::ONE,
            energy: 4.0,
            radius: 0.0,
            soft_falloff: false,
            spot_size: 1.0,
            spot_blend: 0.0,
            area: Some(shape),
            spread: Some(spread),
            use_mis: true,
            volume_factor: 0.0,
            volume_radius_squared: 0.0,
            ray_mask: RayKind::Camera.bit(),
            camera_shape_eligible: true,
        };
        let camera = make_camera(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO);
        let origin = camera.position;
        let direction = -DVec3::Z;
        let zero_spread_event =
            camera_light_shape_hit(&area, &camera, origin, direction, camera.far)
                .ok_or("area camera shape was suppressed by zero physical spread")?;

        let nonzero_spread = Spread::new(1.0).ok_or("nonzero spread should be representable")?;
        let nonzero_area = PointLight {
            spread: Some(nonzero_spread),
            ..area
        };
        let nonzero_spread_event =
            camera_light_shape_hit(&nonzero_area, &camera, origin, direction, camera.far)
                .ok_or("area camera shape was suppressed by nonzero physical spread")?;
        let expected = area_radiance(area.energy, shape.area);
        for channel in 0..3 {
            assert!((zero_spread_event.radiance[channel] - expected).abs() < 1.0e-12);
            assert!(
                (zero_spread_event.radiance[channel] - nonzero_spread_event.radiance[channel])
                    .abs()
                    < 1.0e-12
            );
        }
        assert!((zero_spread_event.distance - nonzero_spread_event.distance).abs() < 1.0e-12);
        Ok(())
    }
}
fn geometry_has_volume_material(geometry: &Geometry) -> bool {
    geometry.triangles.iter().any(|triangle| {
        if triangle.ray_mask & RayKind::Camera.bit() == 0 {
            return false;
        }
        let Some(material) = geometry.materials.get(triangle.material_index) else {
            return false;
        };
        geometry
            .prepared
            .get(triangle.material_index)
            .is_some_and(PreparedMaterial::has_volume_output)
            || (material.node_tree.is_none()
                && material.volume_density.is_finite()
                && material.volume_density > 0.0)
    })
}
/// The view-depth extent of all camera-visible volume objects, matching EEVEE's object integration bounds.
fn realtime_volume_depth_bounds(
    geometry: &SharedGeometry<'_>,
    volumes: &[VolumeInstance<'_>],
    camera: Camera,
) -> Option<(f64, f64)> {
    let mut depth_min = f64::INFINITY;
    let mut depth_max = f64::NEG_INFINITY;
    let mut include_point = |point: DVec3| {
        let depth = camera.depth_of(point);
        depth_min = depth_min.min(depth);
        depth_max = depth_max.max(depth);
    };

    for triangle in geometry.triangles {
        if triangle.ray_mask & RayKind::Camera.bit() == 0 {
            continue;
        }
        let Some(material) = geometry.materials.get(triangle.material_index) else {
            continue;
        };
        let is_volume = geometry
            .prepared
            .get(triangle.material_index)
            .is_some_and(PreparedMaterial::has_volume_output)
            || (material.node_tree.is_none()
                && material.volume_density.is_finite()
                && material.volume_density > 0.0);
        if is_volume {
            for point in triangle.positions {
                include_point(point);
            }
        }
    }

    for volume in volumes {
        if volume.ray_mask & RayKind::Camera.bit() == 0 {
            continue;
        }
        let local_to_world = volume.world_to_local.inverse();
        for x in [volume.bounds_min.x, volume.bounds_max.x] {
            for y in [volume.bounds_min.y, volume.bounds_max.y] {
                for z in [volume.bounds_min.z, volume.bounds_max.z] {
                    include_point(local_to_world.transform_point3(DVec3::new(x, y, z)));
                }
            }
        }
    }

    (depth_min.is_finite() && depth_max.is_finite()).then_some((depth_min, depth_max))
}
fn scene_has_world_volume(doc: &SceneDoc, scene_id: &Id) -> bool {
    let Some(scene) = doc.scenes.get(scene_id) else {
        return false;
    };
    let Some(world) = scene
        .world
        .as_ref()
        .and_then(|world_id| doc.worlds.get(world_id))
    else {
        return false;
    };
    let Some(tree) = world
        .node_tree
        .as_ref()
        .and_then(|tree_id| doc.node_groups.get(tree_id))
    else {
        return false;
    };
    tree.links.iter().any(|link| {
        link.to_socket == "Volume"
            && tree
                .nodes
                .get(&link.to_node)
                .is_some_and(|node| node.node_type == "ShaderNodeOutputWorld")
    })
}

pub(super) struct RealtimeVolumeTransport<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    bvh: Option<&'a Bvh>,
    volumes: Vec<VolumeInstance<'a>>,
    volume_temperatures: Vec<Option<VolumeTemperature>>,
    has_temperature_emission: bool,
    lights: Vec<PointLight>,
    camera_light_prepared: Vec<CameraLightPrepared>,
    /// EEVEE's isotropic world-probe SH L0 used for volume in-scattering.
    world_ambient: DVec3,
    /// Whether the L0-only world probe loses directional radiance needed by anisotropic scattering.
    world_nonuniform: bool,
    meters_per_unit: f64,
    has_mesh_volumes: bool,
    volumetric_shadows: bool,
}

const FROXEL_COLUMN_CACHE_SLOTS: usize = 4;

#[derive(Clone, Copy, Default)]
struct FroxelColumnCacheSlot {
    column: Option<usize>,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct RealtimeVolumeScratch {
    media: Vec<MeshMedium>,
    initial_media: Vec<MeshMedium>,
    hit_media: Vec<MeshMedium>,
    boundaries: Vec<(f64, MeshMedium)>,
    evaluator_workspace: EvaluatorWorkspace,
    froxel_shading: SurfaceShading,
    froxel_media: Vec<FroxelMeshMedium>,
    froxel_initial_media: Vec<FroxelMeshMedium>,
    froxel_hit_media: Vec<FroxelMeshMedium>,
    froxel_mesh_boundary_hits: Vec<(f64, FroxelMeshMedium)>,
    froxel_extinction: Vec<DVec3>,
    froxel_source: Vec<DVec3>,
    froxel_prefix: Vec<DVec3>,
    froxel_scattering: Vec<DVec3>,
    froxel_boundaries: Vec<(usize, FroxelMeshMedium)>,
    froxel_resolve: [Option<RealtimeVolumeResolveCache>; 2],
}

struct RealtimeVolumeResolveCache {
    layout: FroxelTextureLayout,
    depth_mapping: VolumeDepthMapping,
    width: u32,
    height: u32,
    tile_size: u32,
    slice_count: usize,
    sample_count: usize,
    depth_start: f64,
    has_temperature_emission: bool,
    column_slots: Vec<FroxelColumnCacheSlot>,
    column_prefix: Vec<DVec3>,
    column_radiance: Vec<DVec3>,
    column_clock: u64,
}

impl RealtimeVolumeResolveCache {
    fn cached_column_slot(&mut self, sample: usize, column: usize) -> Option<usize> {
        if sample >= self.sample_count {
            return None;
        }
        let first = sample.checked_mul(FROXEL_COLUMN_CACHE_SLOTS)?;
        let last = first.checked_add(FROXEL_COLUMN_CACHE_SLOTS)?;
        for slot in first..last {
            if self.column_slots.get(slot)?.column == Some(column) {
                self.column_clock = self.column_clock.wrapping_add(1);
                self.column_slots[slot].last_used = self.column_clock;
                return Some(slot);
            }
        }
        None
    }

    fn replacement_column_slot(&self, sample: usize) -> Option<usize> {
        if sample >= self.sample_count {
            return None;
        }
        let first = sample.checked_mul(FROXEL_COLUMN_CACHE_SLOTS)?;
        let last = first.checked_add(FROXEL_COLUMN_CACHE_SLOTS)?;
        let slots = self.column_slots.get(first..last)?;
        for (offset, slot) in slots.iter().enumerate() {
            if slot.column.is_none() {
                return Some(first + offset);
            }
        }
        slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.last_used)
            .map(|(offset, _)| first + offset)
    }

    fn store_column(
        &mut self,
        slot: usize,
        column: usize,
        prefix: &[DVec3],
        radiance: &[DVec3],
    ) -> Result<()> {
        if prefix.len() != self.slice_count
            || (self.has_temperature_emission && radiance.len() != self.slice_count)
            || (!self.has_temperature_emission && !radiance.is_empty())
        {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "realtime froxel column has an invalid slice count",
            ));
        }
        let start = slot.checked_mul(self.slice_count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "realtime froxel cache index overflowed",
            )
        })?;
        let end = start.checked_add(self.slice_count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "realtime froxel cache index overflowed",
            )
        })?;
        self.column_prefix
            .get_mut(start..end)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "realtime froxel cache slot is out of range",
                )
            })?
            .copy_from_slice(prefix);
        if self.has_temperature_emission {
            self.column_radiance
                .get_mut(start..end)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "realtime froxel cache slot is out of range",
                    )
                })?
                .copy_from_slice(radiance);
        }
        let entry = self.column_slots.get_mut(slot).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "realtime froxel cache slot is out of range",
            )
        })?;
        self.column_clock = self.column_clock.wrapping_add(1);
        *entry = FroxelColumnCacheSlot {
            column: Some(column),
            last_used: self.column_clock,
        };
        Ok(())
    }

    fn column_prefix(&self, slot: usize, column: usize) -> Option<&[DVec3]> {
        if self.column_slots.get(slot)?.column != Some(column) {
            return None;
        }
        let start = slot.checked_mul(self.slice_count)?;
        let end = start.checked_add(self.slice_count)?;
        self.column_prefix.get(start..end)
    }

    fn column_radiance(&self, slot: usize, column: usize) -> Option<&[DVec3]> {
        if !self.has_temperature_emission || self.column_slots.get(slot)?.column != Some(column) {
            return None;
        }
        let start = slot.checked_mul(self.slice_count)?;
        let end = start.checked_add(self.slice_count)?;
        self.column_radiance.get(start..end)
    }

    // Preserve the allocated prefix buffers while invalidating cached column keys.
    #[cfg(test)]
    fn clear_columns(&mut self) {
        self.column_slots.fill(FroxelColumnCacheSlot::default());
        self.column_clock = 0;
    }
}

pub(super) struct RealtimeVolumePass<'pass, 'scene> {
    transport: &'pass RealtimeVolumeTransport<'scene>,
    geometry: TraceGeometry<'pass>,
    scratch: &'pass mut RealtimeVolumeScratch,
    seed: u32,
}

#[derive(Clone, Copy)]
pub(super) struct RealtimeVolumePassTemplate<'pass, 'scene> {
    transport: &'pass RealtimeVolumeTransport<'scene>,
    geometry: SharedGeometry<'pass>,
    seed: u32,
}

pub(super) struct RealtimeVolumeSampleInfo<'a> {
    pub(super) camera: &'a Camera,
    pub(super) frame: &'a CameraFrame,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) sample_x: f64,
    pub(super) sample_y: f64,
    pub(super) ray: (DVec3, DVec3),
    pub(super) sample_index: u32,
    pub(super) sample_count: u32,
    pub(super) view_index: usize,
    pub(super) camera_matrices: Option<(DMat4, DMat4)>,
}

pub(super) struct RealtimeVolumeSample<'sample, 'scene> {
    transport: &'sample RealtimeVolumeTransport<'scene>,
    geometry: TraceGeometry<'sample>,
    scratch: &'sample mut RealtimeVolumeScratch,
    camera: Camera,
    frame: CameraFrame,
    camera_matrices: Option<(DMat4, DMat4)>,
    width: u32,
    height: u32,
    y: u32,
    layout_x: u32,
    layout_width: u32,
    packed_x_offset: u32,
    resolve_view: usize,
    /// Blender's image-render AA sample index, also the per-sample volume projection jitter.
    sample_index: u32,
    sample_count: u32,
    origin: DVec3,
    direction: DVec3,
    pub(super) far_distance: f64,
    pub(super) near_distance: f64,
    footprint: CameraFootprint,
    rng: Pcg32,
    ambient: Option<DVec3>,
    mesh_prepared: bool,
}

impl<'pass, 'scene> RealtimeVolumePass<'pass, 'scene> {
    pub(super) fn new(
        transport: &'pass RealtimeVolumeTransport<'scene>,
        geometry: &'pass Geometry,
        scratch: &'pass mut RealtimeVolumeScratch,
        seed: u32,
    ) -> Self {
        scratch.froxel_resolve = [None, None];
        Self {
            transport,
            geometry: TraceGeometry::of(geometry),
            scratch,
            seed,
        }
    }

    pub(super) fn worker_template(&self) -> RealtimeVolumePassTemplate<'pass, 'scene> {
        RealtimeVolumePassTemplate {
            transport: self.transport,
            geometry: self.geometry.shared,
            seed: self.seed,
        }
    }

    pub(super) fn merge_worker_warnings(&self, warnings: Vec<serde_json::Value>) {
        merge_shader_warnings(self.geometry.warnings, warnings);
    }

    pub(super) fn begin_sample<'sample>(
        &'sample mut self,
        sample: &RealtimeVolumeSampleInfo<'_>,
    ) -> Result<RealtimeVolumeSample<'sample, 'scene>>
    where
        'pass: 'sample,
    {
        let (resolve_view, layout_x, layout_width, packed_x_offset) =
            if sample.camera.stereo_mode == StereoMode::SideBySide {
                let split = sample.width / 2;
                if sample.x < split {
                    (0, sample.x, split, 0)
                } else {
                    (1, sample.x - split, sample.width - split, split)
                }
            } else {
                (sample.view_index, sample.x, sample.width, 0)
            };
        let footprint = camera_footprint(
            sample.frame,
            sample.camera,
            sample.width,
            sample.height,
            sample.x,
            sample.y,
            sample.sample_x,
            sample.sample_y,
            sample.ray,
        )?;
        Ok(RealtimeVolumeSample {
            transport: self.transport,
            geometry: self.geometry,
            scratch: &mut *self.scratch,
            camera: *sample.camera,
            frame: *sample.frame,
            camera_matrices: sample.camera_matrices,
            width: sample.width,
            height: sample.height,
            layout_x,
            layout_width,
            packed_x_offset,
            resolve_view,
            y: sample.y,
            sample_index: sample.sample_index,
            sample_count: sample.sample_count,
            origin: sample.ray.0,
            direction: sample.ray.1,
            far_distance: sample
                .camera
                .ray_distance_for_depth(sample.ray.1, sample.camera.far),
            near_distance: sample
                .camera
                .ray_distance_for_depth(sample.ray.1, sample.camera.near),
            footprint,
            rng: Pcg32::for_sample(self.seed, sample.x, sample.y, sample.sample_index),
            ambient: None,
            mesh_prepared: false,
        })
    }
}

impl<'pass, 'scene> RealtimeVolumePassTemplate<'pass, 'scene> {
    pub(super) fn new_pass<'scratch>(
        &self,
        scratch: &'scratch mut RealtimeVolumeScratch,
        warnings: &'scratch RefCell<Vec<serde_json::Value>>,
    ) -> RealtimeVolumePass<'scratch, 'scene>
    where
        'pass: 'scratch,
    {
        RealtimeVolumePass {
            transport: self.transport,
            geometry: TraceGeometry::with_warnings(self.geometry, warnings),
            scratch,
            seed: self.seed,
        }
    }
}

impl RealtimeVolumeSample<'_, '_> {
    pub(super) fn integrate_segment(
        &mut self,
        start_distance: f64,
        end_distance: f64,
    ) -> Result<(DVec3, DVec3)> {
        self.integrate_segment_with_sources(start_distance, end_distance, true)
    }

    /// Returns EEVEE's filtered cumulative froxel attenuation to a camera-visible light event.
    pub(super) fn transmittance_to(&mut self, end_distance: f64) -> Result<DVec3> {
        self.resolve_froxel_transmittance_to(end_distance)
    }

    fn resolve_cache(&self) -> Result<&RealtimeVolumeResolveCache> {
        self.scratch.froxel_resolve[self.resolve_view]
            .as_ref()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "realtime volume froxel cache is unavailable",
                )
            })
    }

    fn resolve_cache_mut(&mut self) -> Result<&mut RealtimeVolumeResolveCache> {
        self.scratch.froxel_resolve[self.resolve_view]
            .as_mut()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "realtime volume froxel cache is unavailable",
                )
            })
    }

    /// Resolves cumulative froxel radiance and transmittance with EEVEE's XY/Z texture filtering.
    fn resolve_froxel_prefix_to(&mut self, end_distance: f64) -> Result<(DVec3, DVec3)> {
        if !self.initialize_froxel_resolve()? {
            return Ok((DVec3::ZERO, DVec3::ONE));
        }
        let sample_index = usize::try_from(self.sample_index).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "realtime sample index exceeds platform limits",
            )
        })?;
        let event_depth = self
            .camera
            .depth_of(self.origin + self.direction * end_distance);
        if event_depth <= self.resolve_cache()?.depth_start {
            return Ok((DVec3::ZERO, DVec3::ONE));
        }
        let (x_weights, y_weights, z_weights, grid_width) = {
            let cache = self.resolve_cache()?;
            let coord_z = resolve_z_with_end_of_slice_bias(
                cache.depth_mapping.view_z_to_volume_z(-event_depth as f32),
                cache.slice_count,
            );
            let screen_uv = [
                (f64::from(self.layout_x) + 0.5) / f64::from(cache.width),
                1.0 - (f64::from(self.y) + 0.5) / f64::from(cache.height),
            ];
            let resolve_uv = screen_uv_to_resolve_uv(
                [screen_uv[0] as f32, screen_uv[1] as f32],
                cache.layout.coord_scale,
            );
            (
                linear_sample_weights(resolve_uv[0], cache.layout.dimensions[0] as usize),
                linear_sample_weights(resolve_uv[1], cache.layout.dimensions[1] as usize),
                linear_sample_weights(coord_z, cache.slice_count),
                cache.layout.dimensions[0] as usize,
            )
        };

        for x in [x_weights.lower_index, x_weights.upper_index] {
            for y in [y_weights.lower_index, y_weights.upper_index] {
                let column = y * grid_width + x;
                if self
                    .resolve_cache_mut()?
                    .cached_column_slot(sample_index, column)
                    .is_none()
                {
                    let slot = self
                        .resolve_cache()?
                        .replacement_column_slot(sample_index)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "realtime volume sample cache has no column slot",
                            )
                        })?;
                    self.build_froxel_column(column)?;
                    let scratch = &mut *self.scratch;
                    let cache = scratch.froxel_resolve[self.resolve_view]
                        .as_mut()
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "realtime volume froxel cache is unavailable",
                            )
                        })?;
                    cache.store_column(
                        slot,
                        column,
                        &scratch.froxel_prefix,
                        &scratch.froxel_scattering,
                    )?;
                }
            }
        }

        let mut radiance = DVec3::ZERO;
        let mut transmittance = DVec3::ZERO;
        for (x, x_weight) in axis_weights(x_weights) {
            for (y, y_weight) in axis_weights(y_weights) {
                let index = y * grid_width + x;
                let slot = self
                    .resolve_cache_mut()?
                    .cached_column_slot(sample_index, index)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "realtime volume froxel column was not prepared",
                        )
                    })?;
                let cache = self.resolve_cache()?;
                let weight = x_weight * y_weight;
                if cache.has_temperature_emission {
                    let column_radiance = cache.column_radiance(slot, index).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "realtime volume radiance column was not prepared",
                        )
                    })?;
                    radiance += axis_sample(column_radiance, z_weights) * weight;
                }
                let column_prefix = cache.column_prefix(slot, index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "realtime volume transmittance column was not prepared",
                    )
                })?;
                transmittance += axis_sample(column_prefix, z_weights) * weight;
            }
        }
        Ok((radiance, transmittance))
    }

    fn resolve_froxel_transmittance_to(&mut self, end_distance: f64) -> Result<DVec3> {
        self.resolve_froxel_prefix_to(end_distance)
            .map(|(_, transmittance)| transmittance)
    }

    fn initialize_froxel_resolve(&mut self) -> Result<bool> {
        if self.scratch.froxel_resolve[self.resolve_view].is_some() {
            return Ok(true);
        }
        let scene = self
            .transport
            .doc
            .scenes
            .get(&self.transport.snapshot.scene_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist")
            })?;
        let settings = &scene.render;
        let mut depth_start = if settings.eevee_use_volume_custom_range {
            settings.eevee_volumetric_start
        } else {
            self.camera.near
        };
        let mut depth_end = if settings.eevee_use_volume_custom_range {
            settings.eevee_volumetric_end
        } else {
            self.camera.far
        };
        if !scene_has_world_volume(self.transport.doc, &self.transport.snapshot.scene_id)
            && let Some((object_depth_min, object_depth_max)) = realtime_volume_depth_bounds(
                &self.geometry.shared,
                &self.transport.volumes,
                self.camera,
            )
        {
            depth_start = depth_start.max(object_depth_min);
            depth_end = depth_end.min(object_depth_max);
        }
        depth_start = depth_start.max(self.camera.near - 1.0e-4);
        depth_end = depth_end.min(self.camera.far + 1.0e-4);
        if !depth_start.is_finite() || !depth_end.is_finite() || depth_end <= depth_start {
            return Ok(false);
        }

        let tile_size = settings.eevee_volumetric_tile_size;
        let layout = FroxelTextureLayout::new(self.layout_width, self.height, tile_size);
        let slice_count = settings.eevee_volumetric_samples as usize;
        let projection = if matches!(
            self.camera.projection,
            super::raster::Projection::Perspective { .. }
        ) {
            VolumeProjection::Perspective
        } else {
            VolumeProjection::Orthographic
        };
        // EEVEE's view-space Z is negative in front of the camera. Its range is clipped to the
        // camera and, without a world volume, to `VolumePipeline::object_integration_range()`.
        let depth_mapping = VolumeDepthMapping::new(
            projection,
            -depth_start as f32,
            -depth_end as f32,
            settings.eevee_volumetric_sample_distribution as f32,
        );
        let sample_count = usize::try_from(self.sample_count).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "realtime volume sample count exceeds platform limits",
            )
        })?;
        let cache_slot_count = sample_count
            .checked_mul(FROXEL_COLUMN_CACHE_SLOTS)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "realtime volume sample cache exceeds platform limits",
                )
            })?;
        let prefix_count = cache_slot_count.checked_mul(slice_count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "realtime volume sample cache exceeds platform limits",
            )
        })?;
        let mut column_slots = Vec::new();
        column_slots
            .try_reserve_exact(cache_slot_count)
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "cannot allocate realtime volume sample cache",
                )
            })?;
        column_slots.resize(cache_slot_count, FroxelColumnCacheSlot::default());
        let mut column_prefix = Vec::new();
        column_prefix.try_reserve_exact(prefix_count).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "cannot allocate realtime volume sample cache",
            )
        })?;
        column_prefix.resize(prefix_count, DVec3::ZERO);
        let has_temperature_emission = self.transport.has_temperature_emission;
        let mut column_radiance = Vec::new();
        if has_temperature_emission {
            column_radiance
                .try_reserve_exact(prefix_count)
                .map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "cannot allocate realtime volume source cache",
                    )
                })?;
            column_radiance.resize(prefix_count, DVec3::ZERO);
        }
        self.scratch.froxel_resolve[self.resolve_view] = Some(RealtimeVolumeResolveCache {
            layout,
            depth_mapping,
            width: self.layout_width,
            column_radiance,
            height: self.height,
            tile_size,
            slice_count,
            sample_count,
            depth_start,
            has_temperature_emission,
            column_slots,
            column_prefix,
            column_clock: 0,
        });
        Ok(true)
    }

    /// Prepares the shared mesh-boundary transport once for this jittered froxel-column ray.
    fn prepare_froxel_mesh_boundaries(&mut self) -> Result<()> {
        if !self.transport.has_mesh_volumes || self.mesh_prepared {
            return Ok(());
        }
        let trace_geometry = self.geometry;
        let bvh = self.transport.bvh.ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "realtime mesh volume tracing requires the geometry BVH",
            )
        })?;
        let scene = SceneRefs {
            doc: self.transport.doc,
            snapshot: self.transport.snapshot,
            geometry: &trace_geometry,
            bvh,
            lights: &self.transport.lights,
            volumes: &self.transport.volumes,
            images: self.transport.images,
            importance: None,
            camera_matrices: self.camera_matrices.as_ref(),
        };
        let query = RealtimeMeshQuery {
            transport: self.transport,
            geometry: &trace_geometry,
            bvh,
            scene,
        };
        prepare_realtime_mesh_sample(
            query,
            RealtimeVolumeRay {
                origin: self.origin,
                direction: self.direction,
                start_distance: 0.0,
                max_distance: self.far_distance,
                footprint: self.footprint,
            },
            self.scratch,
        )?;
        self.mesh_prepared = true;
        Ok(())
    }

    fn mesh_froxel_extinction_at_point(
        &mut self,
        geometry: &TraceGeometry<'_>,
        point: DVec3,
    ) -> Result<DVec3> {
        let RealtimeVolumeScratch {
            froxel_media,
            evaluator_workspace,
            froxel_shading,
            ..
        } = self.scratch;
        let mut extinction = DVec3::ZERO;
        for (index, medium) in froxel_media.iter().enumerate() {
            if froxel_media[..index].iter().any(|previous| {
                previous.object_index == medium.object_index
                    && previous.material_index == medium.material_index
            }) {
                continue;
            }
            let object =
                object_for_index(geometry.objects, medium.object_index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "froxel volume references an unmapped object",
                    )
                })?;
            let material = geometry
                .materials
                .get(medium.material_index)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "froxel volume references an unmapped material",
                    )
                })?;
            let prepared = geometry
                .prepared
                .get(medium.material_index)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "froxel volume references an unprepared material",
                    )
                })?;
            let local_point = object_space(object, point);
            let mut context =
                HitContext::new(&self.transport.doc.node_groups).with_images(self.transport.images);
            context.path_traced = false;
            context.light_falloff_distance = None;
            if let Some((camera_to_world, world_to_camera)) = self.camera_matrices {
                context.set_camera_matrices(camera_to_world, world_to_camera);
            }
            context.position = point;
            context.object_to_world = object.object_to_world;
            context.world_to_object = object.world_to_local.unwrap_or(DMat4::IDENTITY);
            context.object = local_point;
            context.object_to_world = object.object_to_world;
            context.world_to_object = object
                .world_to_local
                .unwrap_or_else(|| DMat4::from_cols_array(&[f64::NAN; 16]));
            context.object_color = object.object_color;
            context.object_pass_index = object.object_pass_index;
            context.object_random_number = object.object_random_number;
            context.material_pass_index = material.pass_index.map(f64::from);
            context.generated = medium.generated_origin
                + DVec3::new(
                    local_point.x * medium.generated_per_local.x,
                    local_point.y * medium.generated_per_local.y,
                    local_point.z * medium.generated_per_local.z,
                );
            context.generated_per_local = medium.generated_per_local;
            context.normal = -self.direction;
            context.view_direction = -self.direction;
            context.back_facing = false;
            evaluate_surface_with_workspace_into_semantics(
                material,
                prepared,
                &context,
                evaluator_workspace,
                froxel_shading,
                crate::node_kernels::MathSemantics::Eevee,
            )?;
            for active in froxel_media.iter().filter(|active| {
                active.object_index == medium.object_index
                    && active.material_index == medium.material_index
            }) {
                let Some(closure) = froxel_shading.volume.get(active.closure) else {
                    continue;
                };
                let scatter = DVec3::from_array(closure.params.volume_scatter);
                let absorption = DVec3::from_array(closure.params.volume_absorption);
                extinction += (scatter + absorption) * closure.weight;
            }
        }
        Ok(extinction)
    }

    /// Builds one sample's cumulative source radiance and transmittance from its jittered froxel column.
    fn build_froxel_column(&mut self, column_index: usize) -> Result<()> {
        let (slice_count, depth_start, depth_mapping, has_temperature_emission) = {
            let cache = self.resolve_cache()?;
            (
                cache.slice_count,
                cache.depth_start,
                cache.depth_mapping,
                cache.has_temperature_emission,
            )
        };
        let Some(integration_direction) = self.accumulate_froxel_extinction(column_index)? else {
            self.scratch.froxel_prefix.clear();
            self.scratch.froxel_prefix.resize(slice_count, DVec3::ONE);
            self.scratch.froxel_scattering.clear();
            if has_temperature_emission {
                self.scratch
                    .froxel_scattering
                    .resize(slice_count, DVec3::ZERO);
            }
            return Ok(());
        };
        let mut cumulative_radiance = [0.0_f32; 3];
        let mut cumulative_transmittance = [1.0_f32; 3];
        let mut previous_distance = self
            .camera
            .ray_distance_for_depth(integration_direction, depth_start);
        self.scratch.froxel_prefix.clear();
        self.scratch.froxel_prefix.resize(slice_count, DVec3::ZERO);
        self.scratch.froxel_scattering.clear();
        if has_temperature_emission {
            self.scratch
                .froxel_scattering
                .resize(slice_count, DVec3::ZERO);
        }
        for slice in 0..slice_count {
            let normalized_z = (slice + 1) as f32 / slice_count as f32;
            let slice_depth = -f64::from(depth_mapping.volume_z_to_view_z(normalized_z));
            let end_distance = self
                .camera
                .ray_distance_for_depth(integration_direction, slice_depth);
            let step_length = (end_distance - previous_distance).abs() as f32;
            let extinction = eevee_volume_radiance_storage(self.scratch.froxel_extinction[slice])
                .to_array()
                .map(|channel| channel as f32);
            let source = if has_temperature_emission {
                self.scratch.froxel_source[slice]
                    .to_array()
                    .map(|channel| channel as f32)
            } else {
                [0.0_f32; 3]
            };
            for channel in 0..3 {
                let attenuation = (-extinction[channel] * step_length).exp();
                let opacity = 1.0 - attenuation;
                let source_distance = if extinction[channel] == 0.0 {
                    step_length
                } else {
                    opacity / extinction[channel]
                };
                cumulative_radiance[channel] +=
                    cumulative_transmittance[channel] * source[channel] * source_distance;
                cumulative_transmittance[channel] *= attenuation;
            }
            self.scratch.froxel_prefix[slice] = eevee_volume_radiance_storage(DVec3::from_array(
                cumulative_transmittance.map(f64::from),
            ));
            if has_temperature_emission {
                self.scratch.froxel_scattering[slice] = eevee_volume_radiance_storage(
                    DVec3::from_array(cumulative_radiance.map(f64::from)),
                );
            }
            previous_distance = end_distance;
        }
        Ok(())
    }

    /// Samples occupied mesh and voxel froxel centers for this EEVEE image-render AA sample.
    fn accumulate_froxel_extinction(&mut self, column_index: usize) -> Result<Option<DVec3>> {
        let (layout, tile_size, slice_count, depth_mapping, width, height) = {
            let cache = self.resolve_cache()?;
            (
                cache.layout,
                cache.tile_size,
                cache.slice_count,
                cache.depth_mapping,
                cache.width,
                cache.height,
            )
        };
        let dimensions = layout.dimensions;
        let cell_x = column_index as u32 % dimensions[0];
        let cell_y = column_index as u32 / dimensions[0];
        let cell_center = layout.cell_center_in_pixels(cell_x, cell_y);
        self.scratch.froxel_extinction.clear();
        self.scratch
            .froxel_extinction
            .resize(slice_count, DVec3::ZERO);
        self.scratch.froxel_source.clear();
        if self.transport.has_temperature_emission {
            self.scratch.froxel_source.resize(slice_count, DVec3::ZERO);
        }
        let [jitter_x, jitter_y] = eevee_volume_jitter(self.sample_index);
        let jitter_z =
            eevee_volume_froxel_jitter(cell_x, cell_y, eevee_volume_w(self.sample_index));
        let jitter_offset = layout.jitter_in_pixels([jitter_x, jitter_y]);
        let pixel_x = cell_center[0] - jitter_offset[0];
        let pixel_y = cell_center[1] + jitter_offset[1];
        let local_raster_x = pixel_x.floor().clamp(0.0, f64::from(width - 1)) as u32;
        let raster_x = local_raster_x + self.packed_x_offset;
        let raster_y = pixel_y.floor().clamp(0.0, f64::from(height - 1)) as u32;
        let sample_x = pixel_x - f64::from(local_raster_x);
        let sample_y = pixel_y - f64::from(raster_y);
        let Some((origin, direction)) = ray_for_sample_in_frame(
            &self.frame,
            &self.camera,
            self.width,
            self.height,
            raster_x,
            raster_y,
            sample_x,
            sample_y,
        )?
        else {
            return Ok(None);
        };
        let mut footprint = camera_footprint(
            &self.frame,
            &self.camera,
            self.width,
            self.height,
            raster_x,
            raster_y,
            sample_x,
            sample_y,
            (origin, direction),
        )?;
        footprint.dp *= f64::from(tile_size);
        footprint.dd *= f64::from(tile_size);
        let saved = (
            self.origin,
            self.direction,
            self.near_distance,
            self.far_distance,
            self.footprint,
        );
        self.origin = origin;
        self.direction = direction;
        self.near_distance = self
            .camera
            .ray_distance_for_depth(direction, self.camera.near);
        self.far_distance = self
            .camera
            .ray_distance_for_depth(direction, self.camera.far);
        self.footprint = footprint;
        self.mesh_prepared = false;
        self.prepare_froxel_mesh_boundaries()?;

        let trace_geometry = self.geometry;
        self.scratch.froxel_boundaries.clear();
        if self.transport.has_mesh_volumes {
            for (distance, medium) in &self.scratch.froxel_mesh_boundary_hits {
                let point = origin + direction * *distance;
                let view_z = -self.camera.depth_of(point) as f32;
                let volume_z = depth_mapping.view_z_to_volume_z(view_z);
                let boundary = occupancy_boundary_index(volume_z, jitter_z, slice_count);
                self.scratch.froxel_boundaries.push((boundary, *medium));
            }
            self.scratch
                .froxel_media
                .clone_from(&self.scratch.froxel_initial_media);
        } else {
            self.scratch.froxel_media.clear();
        }

        let mut boundary_index = 0;
        for slice in 0..slice_count {
            while boundary_index < self.scratch.froxel_boundaries.len()
                && self.scratch.froxel_boundaries[boundary_index].0 <= slice
            {
                let medium = self.scratch.froxel_boundaries[boundary_index].1;
                toggle_froxel_mesh_medium(&mut self.scratch.froxel_media, &medium);
                boundary_index += 1;
            }
            let normalized_z = (slice as f32 + 0.5 - jitter_z) / slice_count as f32;
            let view_z = depth_mapping.volume_z_to_view_z(normalized_z);
            let depth = -f64::from(view_z);
            let distance = self.camera.ray_distance_for_depth(direction, depth);
            let point = origin + direction * distance;
            let mesh_extinction = if self.transport.has_mesh_volumes {
                self.mesh_froxel_extinction_at_point(&trace_geometry, point)?
            } else {
                DVec3::ZERO
            };
            let mut extinction = mesh_extinction;
            let mut temperature_emission = DVec3::ZERO;
            for (volume_index, volume) in self
                .transport
                .volumes
                .iter()
                .enumerate()
                .filter(|(_, volume)| volume.ray_mask & RayKind::Camera.bit() != 0)
            {
                if !volume_contains_point(volume, point) {
                    continue;
                }
                let density = volume_density_at(volume, point)?;
                extinction +=
                    volume.extinction_rgb_scale * (density * self.transport.meters_per_unit);
                if self.transport.has_temperature_emission
                    && let Some(temperature) = self
                        .transport
                        .volume_temperatures
                        .get(volume_index)
                        .and_then(Option::as_ref)
                {
                    temperature_emission += volume_emission_at(volume, Some(temperature), point)?
                        * self.transport.meters_per_unit;
                }
            }
            self.scratch.froxel_extinction[slice] = extinction;
            if self.transport.has_temperature_emission {
                self.scratch.froxel_source[slice] =
                    eevee_volume_radiance_storage(temperature_emission);
            }
        }

        self.origin = saved.0;
        self.direction = saved.1;
        self.near_distance = saved.2;
        self.far_distance = saved.3;
        self.footprint = saved.4;
        self.mesh_prepared = false;
        Ok(Some(direction))
    }

    fn temperature_emission_segment(
        &mut self,
        start_distance: f64,
        end_distance: f64,
    ) -> Result<(DVec3, DVec3)> {
        let start_prefix = self.resolve_froxel_prefix_to(start_distance)?;
        let end_prefix = self.resolve_froxel_prefix_to(end_distance)?;
        Ok(froxel_segment_from_prefix(start_prefix, end_prefix))
    }

    fn integrate_segment_with_sources(
        &mut self,
        start_distance: f64,
        end_distance: f64,
        include_sources: bool,
    ) -> Result<(DVec3, DVec3)> {
        if end_distance <= start_distance {
            return Ok((DVec3::ZERO, DVec3::ONE));
        }
        let ray = RealtimeVolumeRay {
            origin: self.origin,
            direction: self.direction,
            start_distance,
            max_distance: end_distance,
            footprint: self.footprint,
        };
        let use_froxel_temperature = include_sources && self.transport.has_temperature_emission;
        let (mut radiance, mut transmittance) = if self.transport.has_mesh_volumes {
            let trace_geometry = self.geometry;
            let bvh = self.transport.bvh.ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "realtime mesh volume tracing requires the geometry BVH",
                )
            })?;
            let scene = SceneRefs {
                doc: self.transport.doc,
                snapshot: self.transport.snapshot,
                geometry: &trace_geometry,
                bvh,
                lights: &self.transport.lights,
                volumes: &self.transport.volumes,
                images: self.transport.images,
                importance: None,
                camera_matrices: self.camera_matrices.as_ref(),
            };
            let query = RealtimeMeshQuery {
                transport: self.transport,
                geometry: &trace_geometry,
                bvh,
                scene,
            };
            if !self.mesh_prepared {
                prepare_realtime_mesh_sample(
                    query,
                    RealtimeVolumeRay {
                        origin: self.origin,
                        direction: self.direction,
                        start_distance: 0.0,
                        max_distance: self.far_distance,
                        footprint: self.footprint,
                    },
                    self.scratch,
                )?;
                self.mesh_prepared = true;
            }
            realtime_mesh_volume_integral(
                query,
                ray,
                self.scratch,
                &mut self.rng,
                &mut self.ambient,
                include_sources,
                use_froxel_temperature,
            )?
        } else {
            realtime_voxel_integral(
                self.transport,
                ray,
                &mut self.rng,
                &mut self.ambient,
                include_sources,
                use_froxel_temperature,
            )?
        };
        if use_froxel_temperature {
            let (temperature_radiance, froxel_transmittance) =
                self.temperature_emission_segment(start_distance, end_distance)?;
            radiance += temperature_radiance;
            transmittance = froxel_transmittance;
        }
        Ok((radiance, transmittance))
    }
}

// A camera compositor splits the ray at surfaces; derive local source and attenuation from the
// same cumulative froxel prefix so sequential segments share one transport basis.
fn froxel_segment_from_prefix(start: (DVec3, DVec3), end: (DVec3, DVec3)) -> (DVec3, DVec3) {
    let mut radiance = [0.0; 3];
    let mut transmittance = [0.0; 3];
    for channel in 0..3 {
        let start_transmittance = start.1[channel];
        if start_transmittance > 0.0 {
            radiance[channel] = (end.0[channel] - start.0[channel]) / start_transmittance;
            transmittance[channel] = end.1[channel] / start_transmittance;
        }
    }
    (
        DVec3::from_array(radiance),
        DVec3::from_array(transmittance),
    )
}

fn axis_weights(weights: LinearSampleWeights) -> [(usize, f64); 2] {
    let upper = f64::from(weights.upper_weight);
    [
        (weights.lower_index, 1.0 - upper),
        (weights.upper_index, upper),
    ]
}

fn axis_sample(values: &[DVec3], weights: LinearSampleWeights) -> DVec3 {
    let [(lower_index, lower_weight), (upper_index, upper_weight)] = axis_weights(weights);
    values[lower_index] * lower_weight + values[upper_index] * upper_weight
}

/// EEVEE's render-history projection jitter, from `eevee_sampling.cc:231-241` and
/// `eevee_volume.cc:413-421`. The shader inputs are floats, so wrap after conversion to f32.
fn eevee_volume_jitter(sample_index: u32) -> [f64; 2] {
    let index = sample_index + 1;
    let volume_u = fract_f32(halton(index, 2) as f32 + 0.5);
    let volume_v = fract_f32(halton(index, 3) as f32 + 2.0 / 3.0);
    [
        f64::from(fract_f32(volume_u + 0.5) * 2.0 - 1.0),
        f64::from(fract_f32(volume_v + 0.5) * 2.0 - 1.0),
    ]
}

/// EEVEE's volume-W Halton dimension (`eevee_sampling.cc:231-241`).
fn eevee_volume_w(sample_index: u32) -> f32 {
    fract_f32(halton(sample_index + 1, 5) as f32 + 0.8)
}

/// The per-column Z jitter from `eevee_volume_lib.bsl.hh:25-29` and `eevee_sampling_lib.bsl.hh:59-63`.
fn eevee_volume_froxel_jitter(x: u32, y: u32, offset: f32) -> f32 {
    let pixel_noise = fract_f32(0.067_110_56 * x as f32 + 0.005_837_15 * y as f32);
    fract_f32(offset + 52.982_918 * pixel_noise)
}

fn halton(mut index: u32, base: u32) -> f64 {
    let reciprocal_base = 1.0 / f64::from(base);
    let mut fraction = reciprocal_base;
    let mut value = 0.0;
    while index != 0 {
        value += f64::from(index % base) * fraction;
        index /= base;
        fraction *= reciprocal_base;
    }
    value
}

fn fract_f32(value: f32) -> f32 {
    value - value.floor()
}

#[cfg(test)]
mod volume_jitter_tests {
    use super::{
        FROXEL_COLUMN_CACHE_SLOTS, FroxelColumnCacheSlot, FroxelTextureLayout,
        RealtimeVolumeResolveCache, VolumeDepthMapping, VolumeProjection,
        eevee_volume_froxel_jitter, eevee_volume_jitter, eevee_volume_w,
    };

    #[test]
    fn volume_projection_jitter_matches_blender_halton_sequence() {
        let first = eevee_volume_jitter(0);
        assert!(
            first
                .into_iter()
                .all(|coordinate| coordinate.abs() <= 1.0e-6)
        );

        let second = eevee_volume_jitter(1);
        assert!((second[0] + 0.5).abs() < 1.0e-6);
        assert!((second[1] - 2.0 / 3.0).abs() < 1.0e-6);
        assert!(eevee_volume_w(0).abs() < 1.0e-6);
        assert!((eevee_volume_froxel_jitter(0, 0, eevee_volume_w(1)) - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn froxel_temperature_segments_compose_across_half_opacity() {
        let camera_prefix = (super::DVec3::ZERO, super::DVec3::ONE);
        let surface_prefix = (super::DVec3::splat(0.2), super::DVec3::splat(0.5));
        let far_prefix = (super::DVec3::splat(0.26), super::DVec3::splat(0.2));
        let (front_radiance, front_transmittance) =
            super::froxel_segment_from_prefix(camera_prefix, surface_prefix);
        let (behind_radiance, behind_transmittance) =
            super::froxel_segment_from_prefix(surface_prefix, far_prefix);
        let pass_through = 0.5;
        let composed = front_radiance + front_transmittance * pass_through * behind_radiance;
        let expected = super::DVec3::splat(0.23);
        assert!(composed.abs_diff_eq(expected, 1.0e-12));
        assert!((front_transmittance * behind_transmittance).abs_diff_eq(far_prefix.1, 1.0e-12));

        let mismatched = front_radiance + super::DVec3::splat(0.8) * pass_through * behind_radiance;
        assert!(mismatched.distance(expected) > 1.0e-3);
    }

    #[test]
    fn froxel_prefix_cache_isolated_by_image_render_sample() {
        let mut cache = RealtimeVolumeResolveCache {
            layout: FroxelTextureLayout::new(8, 8, 8),
            depth_mapping: VolumeDepthMapping::new(
                VolumeProjection::Orthographic,
                -0.1,
                -10.0,
                0.8,
            ),
            width: 8,
            height: 8,
            tile_size: 8,
            slice_count: 2,
            sample_count: 2,
            depth_start: 0.1,
            has_temperature_emission: true,
            column_slots: vec![FroxelColumnCacheSlot::default(); 2 * FROXEL_COLUMN_CACHE_SLOTS],
            column_prefix: vec![super::DVec3::ZERO; 2 * FROXEL_COLUMN_CACHE_SLOTS * 2],
            column_radiance: vec![super::DVec3::ZERO; 2 * FROXEL_COLUMN_CACHE_SLOTS * 2],
            column_clock: 0,
        };
        let first_slot = cache.replacement_column_slot(0).unwrap_or(usize::MAX);
        assert!(first_slot < cache.column_slots.len());
        assert!(
            cache
                .store_column(
                    first_slot,
                    3,
                    &[super::DVec3::ONE, super::DVec3::splat(0.5)],
                    &[super::DVec3::splat(0.25), super::DVec3::splat(0.125)],
                )
                .is_ok()
        );
        assert_eq!(cache.cached_column_slot(0, 3), Some(first_slot));
        assert!(cache.cached_column_slot(1, 3).is_none());

        let second_slot = cache.replacement_column_slot(1).unwrap_or(usize::MAX);
        assert!(second_slot < cache.column_slots.len());
        assert!(
            cache
                .store_column(
                    second_slot,
                    3,
                    &[super::DVec3::splat(0.25), super::DVec3::splat(0.125)],
                    &[super::DVec3::splat(0.5), super::DVec3::splat(0.25)],
                )
                .is_ok()
        );
        assert_eq!(
            cache.column_prefix(first_slot, 3),
            Some(&[super::DVec3::ONE, super::DVec3::splat(0.5)][..])
        );
        assert_eq!(
            cache.column_prefix(second_slot, 3),
            Some(&[super::DVec3::splat(0.25), super::DVec3::splat(0.125)][..])
        );
        assert_eq!(
            cache.column_radiance(first_slot, 3),
            Some(&[super::DVec3::splat(0.25), super::DVec3::splat(0.125)][..])
        );
        assert_eq!(
            cache.column_radiance(second_slot, 3),
            Some(&[super::DVec3::splat(0.5), super::DVec3::splat(0.25)][..])
        );
        let slots_ptr = cache.column_slots.as_ptr();
        let prefixes_ptr = cache.column_prefix.as_ptr();
        let radiance_ptr = cache.column_radiance.as_ptr();
        cache.clear_columns();
        assert_eq!(cache.column_slots.as_ptr(), slots_ptr);
        assert_eq!(cache.column_prefix.as_ptr(), prefixes_ptr);
        assert_eq!(cache.column_radiance.as_ptr(), radiance_ptr);
        assert!(cache.cached_column_slot(0, 3).is_none());
        assert!(cache.cached_column_slot(1, 3).is_none());
        assert_eq!(cache.sample_count, 2);
    }
}

fn realtime_world_probe_l1_refusal() -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "EEVEE realtime anisotropic volume scattering with a nonuniform world requires unavailable world-probe SH L1",
        serde_json::json!({
            "feature_id": "shader.volume.world_probe_l1.realtime_scattering",
        }),
    )
}

pub(super) fn prepare_realtime_volume_transport<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    geometry: &'a Geometry,
    images: &'a BTreeMap<String, ImageData>,
    warnings: &RefCell<Vec<serde_json::Value>>,
) -> Result<Option<RealtimeVolumeTransport<'a>>> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let (volumes, volume_temperatures) = collect_volumes(
        doc,
        snapshot,
        false,
        scene.unit.scale_length,
        images,
        warnings,
        crate::node_kernels::MathSemantics::Eevee,
    )?;
    let has_temperature_emission = volume_temperatures.iter().any(|temperature| {
        temperature
            .as_ref()
            .is_some_and(|temperature| temperature.blackbody > 1.0e-3)
    });
    let is_grey =
        |color: DVec3| color.x.total_cmp(&color.y).is_eq() && color.y.total_cmp(&color.z).is_eq();
    if volumes.iter().any(|volume| {
        volume.scattering && (!is_grey(volume.extinction_rgb_scale) || !is_grey(volume.albedo))
    }) {
        let warning = serde_json::json!({
            "code": "UNSUPPORTED_FEATURE",
            "feature_id": "shader.volume.spectral_extinction.realtime_scattering",
            "message": "Realtime EEVEE volume scattering does not match Blender 5.2.2's per-channel spectral in-scatter",
        });
        let mut recorded = warnings.borrow_mut();
        if !recorded.contains(&warning) {
            recorded.push(warning);
        }
    }
    let has_mesh_volumes = geometry_has_volume_material(geometry);
    if volumes.is_empty() && !has_mesh_volumes {
        return Ok(None);
    }
    let bvh = geometry.realtime_bvh();
    if has_mesh_volumes && bvh.is_none() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "realtime mesh volume tracing requires the geometry BVH",
        ));
    }
    let (world_ambient, world_nonuniform) = if let Some(world) = scene
        .world
        .as_ref()
        .and_then(|world_id| doc.worlds.get(world_id))
    {
        volume_ambient_l0(world, &doc.node_groups, images)?
    } else {
        (DVec3::ZERO, false)
    };
    if world_nonuniform
        && volumes.iter().any(|volume| {
            volume.scattering
                && volume
                    .lobes
                    .iter()
                    .any(|lobe| lobe.share > 0.0 && lobe.anisotropy != 0.0)
        })
    {
        return Err(realtime_world_probe_l1_refusal());
    }
    let (lights, camera_light_prepared) = collect_lights_with_camera_data(doc, snapshot);
    Ok(Some(RealtimeVolumeTransport {
        doc,
        snapshot,
        images,
        bvh,
        volumes,
        volume_temperatures,
        has_temperature_emission,
        lights,
        camera_light_prepared,
        world_ambient,
        world_nonuniform,
        meters_per_unit: scene.unit.scale_length,
        has_mesh_volumes,
        volumetric_shadows: scene.render.eevee_use_volumetric_shadows,
    }))
}

#[derive(Clone, Copy)]
struct RealtimeVolumeRay {
    origin: DVec3,
    direction: DVec3,
    start_distance: f64,
    max_distance: f64,
    footprint: CameraFootprint,
}

#[derive(Clone, Copy)]
struct RealtimeMeshQuery<'a, 'v> {
    transport: &'a RealtimeVolumeTransport<'v>,
    geometry: &'a TraceGeometry<'a>,
    bvh: &'a Bvh,
    scene: SceneRefs<'a, 'v>,
}

fn prepare_realtime_mesh_sample(
    query: RealtimeMeshQuery<'_, '_>,
    ray: RealtimeVolumeRay,
    scratch: &mut RealtimeVolumeScratch,
) -> Result<()> {
    let transport = query.transport;
    let RealtimeVolumeScratch {
        media,
        initial_media,
        hit_media,
        boundaries,
        froxel_media,
        froxel_initial_media,
        froxel_hit_media,
        froxel_mesh_boundary_hits,
        froxel_boundaries,
        evaluator_workspace,
        ..
    } = scratch;
    camera_media(
        transport.doc,
        query.geometry,
        query.bvh,
        ray.origin,
        transport.images,
        ray.footprint,
        query
            .scene
            .camera_matrices
            .map(|matrices| (&matrices.0, &matrices.1)),
        false,
        media,
        hit_media,
        evaluator_workspace,
        Some(&mut *froxel_media),
        Some(&mut *froxel_hit_media),
    )?;
    if transport.world_nonuniform
        && media.iter().any(|medium| {
            medium.params.volume_anisotropy != 0.0
                && medium
                    .params
                    .volume_scatter
                    .iter()
                    .any(|scatter| *scatter > 0.0)
        })
    {
        return Err(realtime_world_probe_l1_refusal());
    }
    initial_media.clone_from(media);
    froxel_initial_media.clone_from(froxel_media);
    boundaries.clear();
    froxel_boundaries.clear();
    froxel_mesh_boundary_hits.clear();
    let mut travelled = 0.0;
    let mut failure = None;
    query.bvh.walk_hits(
        query.geometry.triangles,
        WalkRay {
            origin: ray.origin,
            direction: ray.direction,
            epsilon: RAY_EPSILON,
            remaining: ray.max_distance,
            kind: Some(RayKind::Camera),
        },
        |hit, hit_origin| {
            let distance = travelled + hit.distance;
            match shadow_hit_into(
                transport.doc,
                query.geometry,
                hit,
                hit_origin,
                ray.direction,
                transport.images,
                Some(ray.footprint.at_distance(distance)),
                RayKind::Camera,
                query
                    .scene
                    .camera_matrices
                    .map(|matrices| (&matrices.0, &matrices.1)),
                false,
                None,
                hit_media,
                Some(&mut *froxel_hit_media),
                evaluator_workspace,
            ) {
                Ok(_) => (),
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            }
            if transport.world_nonuniform
                && hit_media.iter().any(|medium| {
                    medium.params.volume_anisotropy != 0.0
                        && medium
                            .params
                            .volume_scatter
                            .iter()
                            .any(|scatter| *scatter > 0.0)
                })
            {
                failure = Some(realtime_world_probe_l1_refusal());
                return false;
            }
            if !hit_media.is_empty() {
                boundaries.extend(hit_media.iter().copied().map(|medium| (distance, medium)));
            }
            if !froxel_hit_media.is_empty() {
                froxel_mesh_boundary_hits.extend(
                    froxel_hit_media
                        .iter()
                        .copied()
                        .map(|medium| (distance, medium)),
                );
            }
            travelled = distance + RAY_EPSILON;
            true
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

/// EEVEE's `light_shape_vector_get` evaluates an area light at the closest point on its emitting shape.
fn eevee_volume_light_sample(light: &PointLight, point: DVec3) -> Option<LightSample> {
    if !light.energy.is_finite() || light.energy <= 0.0 || light.volume_factor <= 0.0 {
        return None;
    }
    let area = if light.kind == LightType::Area {
        Some(light.area?)
    } else {
        None
    };
    if area.is_some_and(|shape| shape.half_u <= 0.0 || shape.half_v <= 0.0) {
        return None;
    }
    let directional = light.kind == LightType::Sun;
    let (direction, distance) = if directional {
        (-light.direction, f64::INFINITY)
    } else {
        let light_point = area.map_or(light.position, |shape| {
            let offset = point - light.position;
            let mut u = offset.dot(shape.axis_u);
            let mut v = offset.dot(shape.axis_v);
            if shape.ellipse {
                let radius = (u / shape.half_u).hypot(v / shape.half_v).max(1.0);
                u /= radius;
                v /= radius;
            } else {
                u = u.clamp(-shape.half_u, shape.half_u);
                v = v.clamp(-shape.half_v, shape.half_v);
            }
            light.position + shape.axis_u * u + shape.axis_v * v
        });
        let offset = light_point - point;
        let distance = offset.length();
        if !distance.is_finite() {
            return None;
        }
        (
            if distance > 0.0 {
                offset / distance
            } else {
                DVec3::ZERO
            },
            distance,
        )
    };
    let point_power = match light.kind {
        LightType::Sun => 1.0,
        LightType::Area => {
            let area = area?;
            let rectangle_area = 4.0 * area.half_u * area.half_v;
            let tmp =
                std::f64::consts::FRAC_PI_2 / (std::f64::consts::FRAC_PI_2 + rectangle_area.sqrt());
            let mrp_scaling = tmp + (1.0 - tmp) * std::f64::consts::FRAC_1_PI;
            std::f64::consts::FRAC_1_PI * mrp_scaling
        }
        LightType::Point | LightType::Spot => 1.0 / (4.0 * std::f64::consts::PI),
    };
    let distance_power = if directional {
        1.0
    } else {
        // EEVEE's area-size vector already holds half extents; volume shape radius is half its length.
        let shape_radius = area.map_or(light.radius.max(0.001), |shape| {
            (shape.half_u.hypot(shape.half_v) * 0.5).max(0.001)
        });
        let radius_squared = shape_radius * shape_radius;
        let distance_squared = distance * distance;
        let combined = distance_squared + radius_squared;
        let denominator = combined + distance * combined.sqrt();
        if denominator <= f64::MIN_POSITIVE {
            return None;
        }
        2.0 / denominator
    };
    let area_facing = area.map_or(1.0, |_| (-light.direction).dot(direction).clamp(0.0, 1.0));
    let attenuation = light.volume_fade(point)
        * if light.kind == LightType::Spot {
            light_attenuation(light, -direction)
        } else {
            1.0
        };
    let radiance = light.color
        * (light.energy
            * light.volume_factor
            * point_power
            * distance_power
            * area_facing
            * attenuation);
    (radiance.is_finite() && radiance.max_element() > 0.0).then_some(LightSample {
        direction,
        distance,
        radiance,
        pdf: 0.0,
    })
}
/// EEVEE stores each froxel's accumulated source in an `UFLOAT_11_11_10` image before integration.
fn eevee_volume_radiance_storage(radiance: DVec3) -> DVec3 {
    let radiance = DVec3::from_array(radiance.to_array().map(|channel| f64::from(channel as f32)));
    super::eevee_transmission::probe_radiance(radiance)
}

fn realtime_mesh_volume_integral(
    query: RealtimeMeshQuery<'_, '_>,
    ray: RealtimeVolumeRay,
    scratch: &mut RealtimeVolumeScratch,
    rng: &mut Pcg32,
    ambient_cache: &mut Option<DVec3>,
    include_sources: bool,
    skip_froxel_temperature_emission: bool,
) -> Result<(DVec3, DVec3)> {
    let transport = query.transport;
    let origin = ray.origin;
    let direction = ray.direction;
    let max_distance = ray.max_distance;
    let start_distance = ray.start_distance;
    let footprint = ray.footprint;
    let RealtimeVolumeScratch {
        media,
        initial_media,
        boundaries,
        evaluator_workspace,
        ..
    } = scratch;
    media.clone_from(initial_media);
    let has_mesh_medium = !media.is_empty()
        || boundaries
            .iter()
            .any(|(distance, _)| *distance <= max_distance);
    if !has_mesh_medium {
        let (radiance, transmittance) = realtime_voxel_integral(
            transport,
            ray,
            rng,
            ambient_cache,
            include_sources,
            skip_froxel_temperature_emission,
        )?;
        return Ok((radiance, transmittance));
    }

    // Scan the reusable stack in place, then undo the crossings instead of cloning it.
    media.reserve(boundaries.len());
    let mut mesh_entry = if media.is_empty() { f64::INFINITY } else { 0.0 };
    let mut mesh_exit = 0.0_f64;
    let mut event_index = 0;
    while event_index < boundaries.len() && boundaries[event_index].0 <= max_distance {
        let distance = boundaries[event_index].0;
        let was_inside = !media.is_empty();
        while event_index < boundaries.len()
            && boundaries[event_index].0.total_cmp(&distance).is_eq()
        {
            toggle_mesh_medium(media, &boundaries[event_index].1);
            event_index += 1;
        }
        if was_inside && media.is_empty() {
            mesh_exit = distance;
        } else if !was_inside && !media.is_empty() {
            mesh_entry = mesh_entry.min(distance);
        }
    }
    if !media.is_empty() {
        mesh_exit = max_distance;
    }
    let scanned_events = event_index;
    for (_, medium) in boundaries[..scanned_events].iter().rev() {
        toggle_mesh_medium(media, medium);
    }
    let mut voxel_entry = f64::INFINITY;
    let mut voxel_exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    for instance in transport
        .volumes
        .iter()
        .filter(|volume| volume.ray_mask & RayKind::Camera.bit() != 0)
    {
        let local_origin = instance.world_to_local.transform_point3(origin);
        let local_direction = instance.world_to_local.transform_vector3(direction);
        if let Some((entry, exit)) = intersect_volume_bounds(
            local_origin,
            local_direction,
            instance.bounds_min,
            instance.bounds_max,
            max_distance,
        ) {
            voxel_entry = voxel_entry.min(entry);
            voxel_exit = voxel_exit.max(exit);
            minimum_step = minimum_step.min(volume_sampling_step(instance, local_direction));
        }
    }
    let entry = mesh_entry.min(voxel_entry).max(start_distance);
    let exit = mesh_exit.max(voxel_exit).min(max_distance);
    if !entry.is_finite() || exit <= entry {
        let (radiance, transmittance) = realtime_voxel_integral(
            transport,
            ray,
            rng,
            ambient_cache,
            include_sources,
            skip_froxel_temperature_emission,
        )?;
        return Ok((radiance, transmittance));
    }
    let voxel_step = if minimum_step.is_finite() {
        let span = voxel_exit - voxel_entry;
        let count = (span / minimum_step).ceil().clamp(1.0, 64.0);
        span / count
    } else {
        f64::INFINITY
    };
    let mesh_step = if mesh_entry.is_finite() && mesh_exit > mesh_entry {
        (mesh_exit - mesh_entry) / 64.0
    } else {
        f64::INFINITY
    };
    let step = voxel_step.min(mesh_step).min((exit - entry) / 64.0);
    let meters_per_unit = transport.meters_per_unit;
    let ambient = if include_sources {
        realtime_world_ambient(transport, ambient_cache)
    } else {
        DVec3::ZERO
    };
    let mut integrator = RealtimeMeshIntegrator {
        query,
        origin,
        direction,
        meters_per_unit,
        ambient,
        footprint,
        voxel_entry,
        voxel_exit,
        step,
        rng,
        evaluator_workspace,
        include_sources,
        skip_froxel_temperature_emission,
        radiance: DVec3::ZERO,
        transmittance: DVec3::ONE,
    };
    event_index = 0;
    while event_index < boundaries.len() && boundaries[event_index].0 <= entry {
        let distance = boundaries[event_index].0;
        while event_index < boundaries.len()
            && boundaries[event_index].0.total_cmp(&distance).is_eq()
        {
            toggle_mesh_medium(media, &boundaries[event_index].1);
            event_index += 1;
        }
    }
    let mut position = entry;
    while position < exit {
        let next_mesh_boundary = boundaries
            .get(event_index)
            .map_or(exit, |(distance, _)| *distance)
            .min(exit);
        let next_voxel_boundary = if voxel_entry.is_finite() && position < voxel_entry {
            voxel_entry
        } else if voxel_exit.is_finite() && position < voxel_exit {
            voxel_exit
        } else {
            exit
        };
        let next = next_mesh_boundary.min(next_voxel_boundary);
        integrator.integrate_interval(media.as_slice(), position, next)?;
        while event_index < boundaries.len() && boundaries[event_index].0.total_cmp(&next).is_eq() {
            toggle_mesh_medium(media, &boundaries[event_index].1);
            event_index += 1;
        }
        position = next;
    }
    Ok((
        integrator.radiance,
        integrator.transmittance.clamp(DVec3::ZERO, DVec3::ONE),
    ))
}

struct RealtimeMeshIntegrator<'a, 'v, 'r> {
    query: RealtimeMeshQuery<'a, 'v>,
    origin: DVec3,
    direction: DVec3,
    meters_per_unit: f64,
    ambient: DVec3,
    footprint: CameraFootprint,
    voxel_entry: f64,
    voxel_exit: f64,
    step: f64,
    rng: &'r mut Pcg32,
    include_sources: bool,
    skip_froxel_temperature_emission: bool,
    evaluator_workspace: &'r mut EvaluatorWorkspace,
    radiance: DVec3,
    transmittance: DVec3,
}

impl RealtimeMeshIntegrator<'_, '_, '_> {
    fn integrate_interval(&mut self, media: &[MeshMedium], start: f64, end: f64) -> Result<()> {
        let length = end - start;
        if length <= 0.0 {
            return Ok(());
        }
        let steps = (length / self.step).ceil().clamp(1.0, 64.0) as usize;
        let step = length / steps as f64;
        let stack = stack_medium(media);
        let mesh_extinction = stack.map_or(DVec3::ZERO, |medium| medium_extinction(&medium));
        let mesh_emission = if self.include_sources {
            stack.map_or(DVec3::ZERO, |medium| {
                DVec3::from_array(medium.volume_emission)
            })
        } else {
            DVec3::ZERO
        };
        let mesh_scatter = if self.include_sources {
            stack.map_or(DVec3::ZERO, |medium| {
                DVec3::from_array(medium.volume_scatter)
            })
        } else {
            DVec3::ZERO
        };
        for index in 0..steps {
            let distance = start + (index as f64 + 0.5) * step;
            let point = self.origin + self.direction * distance;
            let mut voxel_extinction = DVec3::ZERO;
            let mut mesh_source = mesh_emission + self.ambient * mesh_scatter;
            let mut voxel_source = DVec3::ZERO;
            if (self.voxel_entry..=self.voxel_exit).contains(&distance) {
                for (volume_index, instance) in self
                    .query
                    .transport
                    .volumes
                    .iter()
                    .enumerate()
                    .filter(|(_, volume)| volume.ray_mask & RayKind::Camera.bit() != 0)
                {
                    let density = volume_density_at(instance, point)?;
                    let extinction = volume_extinction_rgb_density(instance, density);
                    voxel_extinction += extinction;
                    let scattering = if self.include_sources && instance.scattering {
                        extinction
                    } else {
                        DVec3::ZERO
                    };
                    if self.include_sources {
                        let temperature = self
                            .query
                            .transport
                            .volume_temperatures
                            .get(volume_index)
                            .and_then(Option::as_ref);
                        let emission =
                            if self.skip_froxel_temperature_emission && temperature.is_some() {
                                DVec3::ZERO
                            } else {
                                volume_emission_at(instance, temperature, point)?
                            };
                        voxel_source += emission + self.ambient * instance.albedo * scattering;
                    }
                    if self.include_sources && scattering.max_element() > 0.0 {
                        for light in &self.query.transport.lights {
                            if light.ray_mask & RayKind::VolumeScatter.bit() == 0 {
                                continue;
                            }
                            let gain = light.volume_factor * light.volume_fade(point);
                            if gain <= 0.0 {
                                continue;
                            }
                            let Some(sample) = sample_light(light, point, self.rng) else {
                                continue;
                            };
                            let visibility = if self.query.transport.volumetric_shadows {
                                march_volumes_with_budget(
                                    &self.query.transport.volumes,
                                    point,
                                    sample.direction,
                                    sample.distance,
                                    Some(self.rng),
                                    16,
                                    RayKind::Shadow,
                                )?
                            } else {
                                DVec3::ONE
                            };
                            let phase =
                                instance.phase_density(self.direction.dot(sample.direction));
                            voxel_source += instance.albedo
                                * sample.radiance
                                * (scattering * phase * visibility * gain);
                        }
                    }
                }
            }
            if self.include_sources && mesh_scatter != DVec3::ZERO {
                let scene = self.query.scene;
                for light in &self.query.transport.lights {
                    if light.ray_mask & RayKind::VolumeScatter.bit() == 0 {
                        continue;
                    }
                    let Some(sample) = eevee_volume_light_sample(light, point) else {
                        continue;
                    };
                    let visibility = if self.query.transport.volumetric_shadows {
                        shadow_transmittance_with_volumetric_shadows(
                            scene,
                            media,
                            ShadowRay {
                                point,
                                normal: DVec3::ZERO,
                                shading: DVec3::ZERO,
                                direction: sample.direction,
                                distance: sample.distance,
                                footprint: self.footprint.at_distance(distance),
                            },
                            u32::MAX,
                            true,
                            self.rng,
                            self.evaluator_workspace,
                        )?
                    } else {
                        DVec3::ONE
                    };
                    let cosine = self.direction.dot(sample.direction);
                    let mut scatter = DVec3::ZERO;
                    for medium in media {
                        scatter += DVec3::from_array(medium.params.volume_scatter)
                            * henyey_greenstein(cosine, medium.params.volume_anisotropy);
                    }
                    mesh_source += sample.radiance * visibility * scatter;
                }
            }
            let extinction = mesh_extinction + voxel_extinction * self.meters_per_unit;
            let attenuation = if self.include_sources {
                let source = eevee_volume_radiance_storage(
                    mesh_source + voxel_source * self.meters_per_unit,
                );
                let (attenuation_x, source_distance_x) =
                    homogeneous_segment_integral(extinction.x, step);
                let (attenuation_y, source_distance_y) =
                    homogeneous_segment_integral(extinction.y, step);
                let (attenuation_z, source_distance_z) =
                    homogeneous_segment_integral(extinction.z, step);
                let source_distance =
                    DVec3::new(source_distance_x, source_distance_y, source_distance_z);
                self.radiance += self.transmittance * source * source_distance;
                DVec3::new(attenuation_x, attenuation_y, attenuation_z)
            } else {
                DVec3::new(
                    homogeneous_segment_transmittance(extinction.x, step),
                    homogeneous_segment_transmittance(extinction.y, step),
                    homogeneous_segment_transmittance(extinction.z, step),
                )
            };
            self.transmittance *= attenuation;
        }
        Ok(())
    }
}

fn realtime_voxel_integral(
    transport: &RealtimeVolumeTransport<'_>,
    ray: RealtimeVolumeRay,
    rng: &mut Pcg32,
    ambient_cache: &mut Option<DVec3>,
    include_sources: bool,
    skip_froxel_temperature_emission: bool,
) -> Result<(DVec3, DVec3)> {
    let RealtimeVolumeRay {
        origin,
        direction,
        start_distance,
        max_distance,
        ..
    } = ray;
    let mut entry = f64::INFINITY;
    let mut exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    for instance in transport
        .volumes
        .iter()
        .filter(|volume| volume.ray_mask & RayKind::Camera.bit() != 0)
    {
        let local_origin = instance.world_to_local.transform_point3(origin);
        let local_direction = instance.world_to_local.transform_vector3(direction);
        if let Some((volume_entry, volume_exit)) = intersect_volume_bounds(
            local_origin,
            local_direction,
            instance.bounds_min,
            instance.bounds_max,
            max_distance,
        ) {
            entry = entry.min(volume_entry);
            exit = exit.max(volume_exit);
            minimum_step = minimum_step.min(volume_sampling_step(instance, local_direction));
        }
    }
    entry = entry.max(start_distance);
    if !entry.is_finite() || exit <= entry {
        return Ok((DVec3::ZERO, DVec3::ONE));
    }
    let segment_length = exit - entry;
    let step_count = (segment_length / minimum_step).ceil().clamp(1.0, 64.0) as usize;
    let step = segment_length / step_count as f64;
    let meters_per_unit = transport
        .volumes
        .first()
        .map_or(1.0, |instance| instance.meters_per_unit);
    let ambient = if include_sources {
        realtime_world_ambient(transport, ambient_cache)
    } else {
        DVec3::ZERO
    };
    let mut transmittance = DVec3::ONE;
    let mut radiance = DVec3::ZERO;
    for index in 0..step_count {
        let distance = entry + (index as f64 + 0.5) * step;
        let point = origin + direction * distance;
        let mut extinction = DVec3::ZERO;
        let mut source = DVec3::ZERO;
        for (volume_index, instance) in transport
            .volumes
            .iter()
            .enumerate()
            .filter(|(_, volume)| volume.ray_mask & RayKind::Camera.bit() != 0)
        {
            let density = volume_density_at(instance, point)?;
            let extinction_coefficient = volume_extinction_rgb_density(instance, density);
            extinction += extinction_coefficient;
            let scattering_coefficient = if include_sources && instance.scattering {
                extinction_coefficient
            } else {
                DVec3::ZERO
            };
            if include_sources {
                let temperature = transport
                    .volume_temperatures
                    .get(volume_index)
                    .and_then(Option::as_ref);
                source += if skip_froxel_temperature_emission && temperature.is_some() {
                    DVec3::ZERO
                } else {
                    volume_emission_at(instance, temperature, point)?
                };
                if instance.scattering {
                    source += ambient * instance.albedo * scattering_coefficient;
                    for light in &transport.lights {
                        if light.ray_mask & RayKind::VolumeScatter.bit() == 0 {
                            continue;
                        }
                        // EEVEE's volume in-scatter of a light is its power per class, `volume_factor`, times its influence.
                        let gain = light.volume_factor * light.volume_fade(point);
                        if gain <= 0.0 {
                            continue;
                        }
                        let Some(sample) = sample_light(light, point, rng) else {
                            continue;
                        };
                        let visibility = if transport.volumetric_shadows {
                            march_volumes_with_budget(
                                &transport.volumes,
                                point,
                                sample.direction,
                                sample.distance,
                                Some(rng),
                                16,
                                RayKind::Shadow,
                            )?
                        } else {
                            DVec3::ONE
                        };
                        let phase = instance.phase_density(direction.dot(sample.direction));
                        source += instance.albedo
                            * sample.radiance
                            * (scattering_coefficient * phase * visibility * gain);
                    }
                }
            }
        }
        let step_in_meters = step * meters_per_unit;
        let attenuation = if include_sources {
            let (attenuation_x, source_distance_x) =
                homogeneous_segment_integral(extinction.x, step_in_meters);
            let (attenuation_y, source_distance_y) =
                homogeneous_segment_integral(extinction.y, step_in_meters);
            let (attenuation_z, source_distance_z) =
                homogeneous_segment_integral(extinction.z, step_in_meters);
            let attenuation = DVec3::new(attenuation_x, attenuation_y, attenuation_z);
            let source_distance =
                DVec3::new(source_distance_x, source_distance_y, source_distance_z);
            let source = eevee_volume_radiance_storage(source);
            radiance += source * transmittance * source_distance;
            attenuation
        } else {
            DVec3::new(
                homogeneous_segment_transmittance(extinction.x, step_in_meters),
                homogeneous_segment_transmittance(extinction.y, step_in_meters),
                homogeneous_segment_transmittance(extinction.z, step_in_meters),
            )
        };
        transmittance *= attenuation;
    }
    Ok((radiance, transmittance.clamp(DVec3::ZERO, DVec3::ONE)))
}

fn realtime_world_ambient(
    transport: &RealtimeVolumeTransport<'_>,
    ambient_cache: &mut Option<DVec3>,
) -> DVec3 {
    if let Some(ambient) = *ambient_cache {
        return ambient;
    }
    let ambient = transport.world_ambient;
    *ambient_cache = Some(ambient);
    ambient
}

fn world_radiance(
    doc: &SceneDoc,
    scene_id: &Id,
    images: &BTreeMap<String, ImageData>,
    direction: DVec3,
    fallback: DVec3,
    path_traced: bool,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let Some(world) = doc
        .scenes
        .get(scene_id)
        .and_then(|scene| scene.world.as_ref())
        .and_then(|world_id| doc.worlds.get(world_id))
    else {
        return Ok(fallback);
    };
    crate::shader::evaluate_world_with_workspace(
        world,
        &doc.node_groups,
        images,
        direction,
        path_traced,
        evaluator_workspace,
    )
}

/// The scene a path reads, bundled so the volume and shadow code take one argument for it.
#[derive(Clone, Copy)]
struct SceneRefs<'a, 'v> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    geometry: &'a TraceGeometry<'a>,
    bvh: &'a Bvh,
    lights: &'a [PointLight],
    volumes: &'a [VolumeInstance<'v>],
    images: &'a BTreeMap<String, ImageData>,
    importance: Option<&'a WorldImportanceSampler>,
    camera_matrices: Option<&'a (DMat4, DMat4)>,
}

impl<'a> SceneRefs<'a, '_> {
    fn with_camera_matrices(mut self, camera_matrices: Option<&'a (DMat4, DMat4)>) -> Self {
        self.camera_matrices = camera_matrices;
        self
    }
}
fn camera_matrices_for_render(camera: &Camera) -> Result<Option<(DMat4, DMat4)>> {
    let view = super::raster::ViewBasis::from_camera(camera)?;
    Ok(view
        .camera_matrices()
        .map(|(camera_to_world, world_to_camera)| (*camera_to_world, *world_to_camera)))
}

/// One path's budgets: surface bounces, volume scatters (Cycles' `volume_bounces + 1` vertices take direct light), and
/// transparent passes.
#[derive(Clone, Copy)]
struct Bounces {
    surface: u32,
    volume: u32,
    transparent: u32,
}

/// A shadow ray from `point` along `direction` for `distance`. A surface receiver offsets its ray by its geometric `normal`
/// and its smooth `shading` normal; a volume scatter has neither and passes zero.
#[derive(Clone, Copy)]
struct ShadowRay {
    point: DVec3,
    normal: DVec3,
    shading: DVec3,
    direction: DVec3,
    distance: f64,
    /// The camera footprint at the ray's origin (Cycles' `sd->dP`, which a shadow ray carries unchanged).
    footprint: f64,
}

/// A volume scatter's receiver: its point, the incoming direction, its phase function, the per-light weights of the direct
/// light, and the environment along the incoming direction.
struct VolumeSite<'a> {
    point: DVec3,
    /// The scatter's camera footprint (Cycles' volume `sd->dP`), carried by its shadow rays.
    footprint: f64,
    incoming: DVec3,
    phase: &'a PhaseMixture,
    light_weights: &'a [f64],
    environment: DVec3,
}

/// A scatter's phase function: a mixture of Henyey-Greenstein lobes, one per volume closure (Cycles' phase closures), each
/// weighted by its scatter. The density is the normalised mixture of `_volume_shader_phase_eval_mis`, and a closure is
/// picked in proportion to its weight (`volume_shader_phase_pick`).
#[derive(Default)]
struct PhaseMixture {
    lobes: Vec<(f64, f64)>,
    total: f64,
}

impl PhaseMixture {
    /// Adds a closure of `weight` (its scatter) and `anisotropy`. A closure with no scatter takes no part.
    fn push(&mut self, weight: f64, anisotropy: f64) {
        if weight > 0.0 && weight.is_finite() {
            self.lobes.push((weight, anisotropy));
            self.total += weight;
        }
    }

    /// The normalised mixture density at `cosine`, the cosine between the incoming and the scattered direction.
    fn density(&self, cosine: f64) -> f64 {
        if self.total <= 0.0 {
            return 0.0;
        }
        self.lobes
            .iter()
            .map(|&(weight, anisotropy)| weight * henyey_greenstein(cosine, anisotropy))
            .sum::<f64>()
            / self.total
    }

    /// The anisotropy of a closure picked in proportion to its weight.
    fn pick(&self, rng: &mut Pcg32) -> Option<f64> {
        let mut selected = rng.next_f64() * self.total;
        for &(weight, anisotropy) in &self.lobes {
            if selected < weight {
                return Some(anisotropy);
            }
            selected -= weight;
        }
        self.lobes.last().map(|&(_, anisotropy)| anisotropy)
    }
}

/// Continues a path past a volume scatter. The direction is drawn from the lobe of the closure the event selected, and the
/// density the next MIS partner uses is the event's normalised mixture density.
fn sample_volume_continuation(
    incoming: DVec3,
    anisotropy: f64,
    phase: &PhaseMixture,
    rng: &mut Pcg32,
) -> (DVec3, f64) {
    let direction = sample_henyey_greenstein_direction(incoming, anisotropy, rng);
    let phase_pdf = phase.density(incoming.normalize_or_zero().dot(direction));
    (direction, phase_pdf)
}

/// The phase function at `distance` of a segment's media: each mesh medium is a closure weighted by its scatter, and each
/// anisotropy group of the voxel grid is one weighted by its scatter per unit length at `distance`.
fn event_phase(media: &[MeshMedium], grid: Option<&VoxelGrid>, distance: f64) -> PhaseMixture {
    let mut phase = PhaseMixture::default();
    for medium in media {
        phase.push(
            mean_scatter(&medium.params),
            medium.params.volume_anisotropy,
        );
    }
    if let Some(grid) = grid {
        for group in &grid.groups {
            let scatter = grid_scatter_at(grid, &group.scatter, distance).element_sum() / 3.0;
            phase.push(scatter * grid.meters_per_unit, group.anisotropy);
        }
    }
    phase
}

/// The density per unit distance of a scatter event in a mixed segment: the mesh stack's scatter density past the grid's
/// optical depth, plus the grid's scatter density past the stack's extinction (the analog first-event density of both).
fn mixed_scatter_pdf(
    stack: &BsdfParams,
    grid: &VoxelGrid,
    distance: f64,
    channel_pdf: DVec3,
) -> f64 {
    distance_scatter_pdf(stack, distance) * grid_survival_pdf(grid, distance, channel_pdf)
        + grid_scatter_pdf(grid, distance, channel_pdf) * (-mean_extinction(stack) * distance).exp()
}

/// The density per unit distance of a scatter event in a segment's media, which the distance sampler realises.
/// `channel_pdf` is the same hero-channel mixture used by voxel free-flight sampling,
/// so its marginal density is used for MIS.
fn segment_scatter_pdf(
    grid: Option<&VoxelGrid>,
    stack: Option<&BsdfParams>,
    distance: f64,
    channel_pdf: DVec3,
) -> f64 {
    match (grid, stack) {
        (Some(grid), Some(stack)) => mixed_scatter_pdf(stack, grid, distance, channel_pdf),
        (Some(grid), None) => grid_scatter_pdf(grid, distance, channel_pdf),
        (None, Some(stack)) => distance_scatter_pdf(stack, distance),
        (None, None) => 0.0,
    }
}

/// The per-channel transmittance of a segment's media to `distance`: the mesh stack's and the grid's.
fn segment_transmittance(
    grid: Option<&VoxelGrid>,
    stack: Option<&BsdfParams>,
    distance: f64,
) -> DVec3 {
    let mesh = stack.map_or(DVec3::ONE, |stack| medium_transmittance(stack, distance));
    let voxel = grid.map_or(DVec3::ONE, |grid| (-grid_depth(grid, distance)).exp());
    mesh * voxel
}

/// The camera-side scatter of a segment at `distance` towards a light at `cosine`: each closure's scatter times its own
/// Henyey-Greenstein phase, transmitted by the segment (Cycles' `volume_shader_eval` summed over the closures).
fn camera_scatter_phase(
    media: &[MeshMedium],
    grid: Option<&VoxelGrid>,
    distance: f64,
    cosine: f64,
    transmittance: DVec3,
) -> DVec3 {
    let mut scatter = DVec3::ZERO;
    for medium in media {
        scatter += DVec3::from_array(medium.params.volume_scatter)
            * henyey_greenstein(cosine, medium.params.volume_anisotropy);
    }
    if let Some(grid) = grid {
        for group in &grid.groups {
            scatter += grid_scatter_at(grid, &group.scatter, distance)
                * (grid.meters_per_unit * henyey_greenstein(cosine, group.anisotropy));
        }
    }
    scatter * transmittance
}

#[cfg(test)]
/// The volume and transport state of one path: the scene, the bounce budgets and the sampling state.
fn trace_path(
    scene: SceneRefs<'_, '_>,
    bounces: Bounces,
    origin: DVec3,
    direction: DVec3,
    footprint: CameraFootprint,
    environment: DVec3,
    film_transparent: bool,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(DVec3, f64)> {
    trace_path_with_temperatures(
        scene,
        &[],
        bounces,
        origin,
        direction,
        footprint,
        environment,
        film_transparent,
        rng,
        evaluator_workspace,
    )
}

fn trace_path_with_temperatures(
    scene: SceneRefs<'_, '_>,
    volume_temperatures: &[Option<VolumeTemperature>],
    bounces: Bounces,
    mut origin: DVec3,
    mut direction: DVec3,
    footprint: CameraFootprint,
    environment: DVec3,
    film_transparent: bool,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(DVec3, f64)> {
    let SceneRefs {
        doc,
        snapshot,
        geometry,
        bvh,
        lights,
        volumes,
        images,
        importance,
        camera_matrices,
    } = scene;
    let Bounces {
        surface: max_bounces,
        volume: volume_bounces,
        transparent: transparent_max_bounces,
    } = bounces;
    let mut radiance = DVec3::ZERO;
    let mut throughput = DVec3::ONE;
    let mut escaped = false;
    let mut previous = Scatter::Camera;
    let mut ray = RayKind::Camera;
    let mut bounce = 0u32;
    let mut transparent_bounce = 0u32;
    // Cycles' `transparent_max_bounce` is at least one: the first pass-through is always taken
    // (`intern/cycles/kernel/integrator/path_state.h`, the FIXME at `path_state_next`).
    let transparent_cap = transparent_max_bounces.max(1);
    let mut volume_bounds = 0u32;
    // The ray differentials of the path: `ray_dp` is the compact `ray.dP` and `ray_dd` the compact `ray.dD` (Cycles'
    // `INTEGRATOR_STATE(ray, dP)` and `dD`). A hit's footprint is `ray_dp + t · ray_dd`, and a surface bounce keeps that
    // footprint as the next ray's `dP`, widening `dD` only by the filter glossy scale.
    let mut ray_dp = footprint.dp;
    let mut ray_dd = footprint.dd;
    let widen_scale = filter_glossy_widen_scale(doc);
    let blur_glossy = filter_glossy(doc);
    // The smallest sampled pdf so far on this path (Cycles' `path.min_ray_pdf`, `FLT_MAX` at the start of a path and
    // lowered by each BSDF and phase sample, `shade_surface.h` and `shade_volume.h`). It drives the glossy blur.
    let mut min_ray_pdf = f64::MAX;
    // Cycles' `volume_bounce` limit (`path_state_next`): `volume_bounces + 1` scatter vertices take direct light.
    let volume_limit = volume_bounces.saturating_add(1);
    let mut volume_scatters = 0u32;
    // Set once the path's last scatter has taken its direct light. The ray then continues without scattering: its
    // segments only attenuate and emit, and it ends at its next surface, or in the environment, MIS weighted against
    // the scatter's phase sample (Cycles' `PATH_RAY_TERMINATE_AFTER_TRANSPARENT`).
    let mut terminal = false;
    // Set by the pass that reaches the cap: the next surface is shaded for emission and ends the
    // path (`PATH_RAY_TERMINATE_ON_NEXT_SURFACE`, `integrate_surface_terminate`).
    let mut terminate_on_next_surface = false;
    // A subsurface exit replaces the next closest hit: the probe's hit is shaded as the exit (`exit_hit`), a diffuse
    // surface built from the entry closure's parameters (`exit_entry`). `diffuse_ancestor` disables the Burley
    // closure after a diffuse bounce.
    let mut exit_hit: Option<BvhHit> = None;
    let mut exit_entry: Option<BsdfParams> = None;
    // Cycles' `PATH_RAY_SUBSURFACE_BACKFACING`: the entry was on a backfacing side, which flips the exit's normal.
    let mut exit_backfacing = false;
    let mut diffuse_ancestor = false;
    // The volume stack: the media the path is inside, innermost last. It starts with the mesh media that contain the
    // camera (Cycles' `volume_stack_init`).
    let mut media = Vec::new();
    let mut hit_media = Vec::new();
    camera_media(
        doc,
        geometry,
        bvh,
        origin,
        images,
        footprint,
        camera_matrices.map(|matrices| (&matrices.0, &matrices.1)),
        true,
        &mut media,
        &mut hit_media,
        evaluator_workspace,
        None,
        None,
    )?;
    loop {
        let hit = match exit_hit.take() {
            Some(exit) => Some(exit),
            None => closest_hit(
                geometry,
                bvh,
                origin,
                direction,
                RAY_EPSILON,
                f64::INFINITY,
                ray,
            ),
        };
        let surface_distance = hit.map_or(f64::INFINITY, |hit| hit.distance);
        // Camera rays see the lights that are visible to camera, and a volume scatter is light-sampled only. A surface
        // vertex's light hit needs its light sample as MIS partner, which a volume scatter on the way would drop, so a
        // scene with volumes does not count those hits.
        let receiver = match previous {
            Scatter::Camera if ray == RayKind::Camera => Some(origin),
            Scatter::Surface(vertex) if volumes.is_empty() => Some(vertex.point),
            Scatter::Delta(point) => Some(point),
            Scatter::Camera | Scatter::Surface(_) | Scatter::Volume(_) => None,
        };
        let light = receiver.and_then(|receiver| {
            nearest_light_hit(lights, origin, direction, receiver, surface_distance, ray)
        });
        let segment_limit = light.map_or(surface_distance, |light| light.distance);
        // The voxel volumes this segment can see: a volume whose object is hidden from the ray's kind passes the ray by
        // (Cycles' `intersect_volume_bounds` under the path's visibility).
        // The exit probe is a local intersection inside the object (`subsurface_disk`), not a light path: no volume
        // transport, emission or scatter applies along it, so it takes the neutral segment.
        let stack = if exit_entry.is_some() {
            None
        } else {
            stack_medium(&media)
        };
        let grid = if exit_entry.is_some() {
            None
        } else {
            voxel_grid(volumes, origin, direction, segment_limit, ray)?
        };
        let voxel_channel_pdf = grid.as_ref().map_or(DVec3::splat(1.0 / 3.0), |grid| {
            grid_channel_pdf(grid, throughput)
        });
        if grid.is_some() || stack.is_some() {
            let mut temperature_march = grid.as_ref().and_then(|grid| {
                TemperatureEmissionMarch::new(
                    volumes,
                    volume_temperatures,
                    origin,
                    direction,
                    grid,
                    rng,
                    ray,
                )
            });
            radiance += throughput
                * emission_until(
                    grid.as_ref(),
                    stack.as_ref(),
                    segment_limit,
                    temperature_march.as_mut(),
                )?;
        }
        // A segment holding both a voxel grid and a mesh medium is mixed: its nearest free flight is sampled analogously.
        let mixed = match (grid.as_ref(), stack) {
            (Some(grid), Some(medium)) if !terminal => Some((grid, medium)),
            _ => None,
        };
        // Equiangular direct light follows Cycles' `volume_stack_sample_method` over every medium of the segment: each voxel
        // grid's and each mesh medium's `volume_sampling`, combined by the same rule.
        let method = if terminal || exit_entry.is_some() {
            None
        } else {
            segment_sampling(
                grid.iter()
                    .flat_map(|grid| grid.samplings.iter().copied())
                    .chain(media.iter().map(|medium| medium.params.volume_sampling)),
            )
        };
        let budget = transparent_max_bounces.saturating_sub(transparent_bounce);
        let choices = equiangular_choices(lights, origin, method, rng);
        // Next-event estimation of the segment's media, for each light sampled equiangularly. The equiangular samples cover
        // the voxel grid's span alone in a voxel-only segment, where the grid has no scatter beyond it, and the whole segment
        // otherwise. Each closure's camera-side scatter is weighted by its own phase at the light's direction.
        let (span_start, span_length) = match (&grid, stack) {
            (Some(grid), None) => (grid.entry, grid_span(grid)),
            _ => (0.0, segment_limit),
        };
        if let Some(method) = method.filter(|_| span_length.is_finite()) {
            let span_origin = origin + direction * span_start;
            for (light, &(point, sampled)) in lights.iter().zip(&choices) {
                // A light hidden from volume scatter is not light-sampled in a medium (`SHADER_EXCLUDE_SCATTER`).
                if light.ray_mask & RayKind::VolumeScatter.bit() == 0 {
                    continue;
                }
                let Some(point) = point.filter(|_| sampled) else {
                    continue;
                };
                let Some((offset, pdf)) =
                    equiangular_sample(point, span_origin, direction, span_length, rng.next_f64())
                else {
                    continue;
                };
                let distance = span_start + offset;
                // Multiple importance sampling weights this sample against the distance strategy's density at the same
                // point (`volume_direct_scatter_mis`); equiangular sampling alone takes no weight.
                let mis = match method {
                    VolumeSampling::Equiangular => 1.0,
                    _ => {
                        2.0 * power_heuristic(
                            pdf,
                            segment_scatter_pdf(
                                grid.as_ref(),
                                stack.as_ref(),
                                distance,
                                voxel_channel_pdf,
                            ),
                        )
                    }
                };
                let sample_point = origin + direction * distance;
                let (nee, light_direction) = volume_light_nee(
                    scene,
                    &media,
                    light,
                    sample_point,
                    ray_dp + distance * ray_dd,
                    budget,
                    rng,
                    evaluator_workspace,
                )?;
                if nee.max_element() <= 0.0 {
                    continue;
                }
                let transmittance = segment_transmittance(grid.as_ref(), stack.as_ref(), distance);
                let scatter = camera_scatter_phase(
                    &media,
                    grid.as_ref(),
                    distance,
                    direction.dot(light_direction),
                    transmittance,
                );
                radiance += throughput * scatter * nee * (mis / pdf);
            }
        }
        if let Some((grid, medium)) = mixed {
            // The voxel hero-channel flight and mesh analog collision are sampled independently; the nearer event receives
            // the other medium's per-channel survival correction, and their densities form the distance MIS partner.
            let voxel = trace_volume_segment(Some(grid), origin, direction, throughput, rng, true);
            let voxel_distance = voxel
                .event
                .map(|event| (event.point - origin).dot(direction));
            let limit = voxel_distance.unwrap_or(segment_limit);
            let collision = sample_mesh_collision(&medium, limit, rng);
            if let Some(distance) = collision {
                throughput *= if voxel.stochastic {
                    grid_survival_weight(grid, distance, voxel.channel_pdf)
                } else {
                    (-grid_depth(grid, distance)).exp()
                };
                let scatter_rate = mean_scatter(&medium);
                let extinction_rate = mean_extinction(&medium);
                if scatter_rate <= 0.0 || rng.next_f64() * extinction_rate >= scatter_rate {
                    break;
                }
                throughput *= DVec3::from_array(medium.volume_scatter) / scatter_rate
                    * chromatic_correction(&medium, distance);
                let phase = event_phase(&media, Some(grid), distance);
                let Some(anisotropy) = phase.pick(rng) else {
                    break;
                };
                let event = VolumeEvent {
                    point: origin + direction * distance,
                    scattering_weight: DVec3::ZERO,
                };
                let weights = event_weights(
                    &choices,
                    method,
                    origin,
                    direction,
                    segment_limit,
                    distance,
                    segment_scatter_pdf(Some(grid), Some(&medium), distance, voxel.channel_pdf),
                );
                let event_footprint = ray_dp + (event.point - origin).length() * ray_dd;
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            footprint: event_footprint,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
                                true,
                                evaluator_workspace,
                            )?,
                        },
                        budget,
                        rng,
                        evaluator_workspace,
                    )?;
                volume_scatters += 1;
                terminal |= bounce >= max_bounces || volume_scatters >= volume_limit;
                let (next_direction, phase_pdf) =
                    sample_volume_continuation(direction, anisotropy, &phase, rng);
                direction = next_direction;
                origin = event.point + direction * RAY_EPSILON;
                ray_dp = event_footprint;
                min_ray_pdf = min_ray_pdf.min(phase_pdf);
                previous = Scatter::Volume(phase_pdf);
                ray = RayKind::VolumeScatter;
                bounce += 1;
                continue;
            }
            // No mesh collision before `limit`: the mesh's chromatic transmittance up to there, then the grid's own event
            // or the segment's end.
            throughput *= chromatic_correction(&medium, limit);
            if voxel.event.is_none() {
                throughput *= voxel.transmittance;
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
            if let Some(event) = voxel.event {
                if event.scattering_weight.max_element() <= 0.0 {
                    break;
                }
                throughput *= event.scattering_weight;
                let event_distance = (event.point - origin).dot(direction);
                let phase = event_phase(&media, Some(grid), event_distance);
                let Some(anisotropy) = phase.pick(rng) else {
                    break;
                };
                let weights = event_weights(
                    &choices,
                    method,
                    origin,
                    direction,
                    segment_limit,
                    event_distance,
                    segment_scatter_pdf(
                        Some(grid),
                        Some(&medium),
                        event_distance,
                        voxel.channel_pdf,
                    ),
                );
                let event_footprint = ray_dp + (event.point - origin).length() * ray_dd;
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            footprint: event_footprint,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
                                true,
                                evaluator_workspace,
                            )?,
                        },
                        budget,
                        rng,
                        evaluator_workspace,
                    )?;
                volume_scatters += 1;
                terminal |= bounce >= max_bounces || volume_scatters >= volume_limit;
                let (next_direction, phase_pdf) =
                    sample_volume_continuation(direction, anisotropy, &phase, rng);
                direction = next_direction;
                origin = event.point + direction * RAY_EPSILON;
                ray_dp = event_footprint;
                min_ray_pdf = min_ray_pdf.min(phase_pdf);
                previous = Scatter::Volume(phase_pdf);
                ray = RayKind::VolumeScatter;
                bounce += 1;
                continue;
            }
        } else if terminal {
            // The path's last scatter has taken its direct light: this ray only attenuates through the volumes it crosses.
            if let Some(grid) = grid.as_ref() {
                let segment =
                    trace_volume_segment(Some(grid), origin, direction, throughput, rng, false);
                throughput *= segment.transmittance;
            }
            if let Some(medium) = stack {
                throughput *= medium_transmittance(&medium, segment_limit);
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
        } else if let (Some(grid), None) = (grid.as_ref(), stack) {
            let segment =
                trace_volume_segment(Some(grid), origin, direction, throughput, rng, true);
            if let Some(event) = segment.event {
                if event.scattering_weight.max_element() <= 0.0 {
                    break;
                }
                throughput *= event.scattering_weight;
            } else {
                throughput *= segment.transmittance;
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
            if let Some(event) = segment.event {
                let event_distance = (event.point - origin).dot(direction);
                let phase = event_phase(&media, Some(grid), event_distance);
                let Some(anisotropy) = phase.pick(rng) else {
                    break;
                };
                let weights = event_weights(
                    &choices,
                    method,
                    origin + direction * grid.entry,
                    direction,
                    grid_span(grid),
                    event_distance - grid.entry,
                    segment_scatter_pdf(Some(grid), None, event_distance, segment.channel_pdf),
                );
                let event_footprint = ray_dp + (event.point - origin).length() * ray_dd;
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            footprint: event_footprint,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
                                true,
                                evaluator_workspace,
                            )?,
                        },
                        budget,
                        rng,
                        evaluator_workspace,
                    )?;
                volume_scatters += 1;
                terminal |= bounce >= max_bounces || volume_scatters >= volume_limit;
                let (next_direction, phase_pdf) =
                    sample_volume_continuation(direction, anisotropy, &phase, rng);
                direction = next_direction;
                origin = event.point + direction * RAY_EPSILON;
                ray_dp = event_footprint;
                min_ray_pdf = min_ray_pdf.min(phase_pdf);
                previous = Scatter::Volume(phase_pdf);
                ray = RayKind::VolumeScatter;
                bounce += 1;
                continue;
            }
        } else if let Some(medium) = stack {
            // The stack changes only at surfaces, so the stack's combined medium spans this whole segment, and its free
            // flight is spectral ratio tracking (`trace_mesh_medium`). A scatter picks its closure by scatter weight.
            // A mesh medium's emission is integrated in closed form over the whole segment, at the throughput the path
            // carries into it, independent of where it scatters (`volume_integrate_homogeneous`).
            match trace_mesh_medium(&medium, segment_limit, &mut throughput, rng) {
                MediumOutcome::Scatter(distance) => {
                    let phase = event_phase(&media, None, distance);
                    let Some(anisotropy) = phase.pick(rng) else {
                        break;
                    };
                    let event = VolumeEvent {
                        point: origin + direction * distance,
                        scattering_weight: DVec3::ZERO,
                    };
                    let weights = event_weights(
                        &choices,
                        method,
                        origin,
                        direction,
                        segment_limit,
                        distance,
                        segment_scatter_pdf(None, Some(&medium), distance, DVec3::splat(1.0 / 3.0)),
                    );
                    let event_footprint = ray_dp + (event.point - origin).length() * ray_dd;
                    radiance += throughput
                        * direct_volume_lighting(
                            scene,
                            &media,
                            &VolumeSite {
                                point: event.point,
                                footprint: event_footprint,
                                incoming: direction,
                                phase: &phase,
                                light_weights: &weights,
                                environment: world_radiance(
                                    doc,
                                    &snapshot.scene_id,
                                    images,
                                    direction,
                                    environment,
                                    true,
                                    evaluator_workspace,
                                )?,
                            },
                            budget,
                            rng,
                            evaluator_workspace,
                        )?;
                    volume_scatters += 1;
                    terminal |= bounce >= max_bounces || volume_scatters >= volume_limit;
                    let (next_direction, phase_pdf) =
                        sample_volume_continuation(direction, anisotropy, &phase, rng);
                    direction = next_direction;
                    origin = event.point + direction * RAY_EPSILON;
                    ray_dp = event_footprint;
                    min_ray_pdf = min_ray_pdf.min(phase_pdf);
                    previous = Scatter::Volume(phase_pdf);
                    ray = RayKind::VolumeScatter;
                    bounce += 1;
                    continue;
                }
                MediumOutcome::Survive => {}
                MediumOutcome::Absorb => break,
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
        }
        if let Some(light) = light {
            let weight = match previous {
                Scatter::Surface(vertex) => power_heuristic(vertex.bsdf_pdf, light.pdf),
                Scatter::Camera | Scatter::Delta(_) | Scatter::Volume(_) => 1.0,
            };
            if light.visible {
                radiance += throughput * light.radiance * weight;
            }
            origin += direction * (light.distance + RAY_EPSILON);
            // The origin moves past the light, so the footprint grows by the distance skipped: `dP + t · dD` measured
            // from the old origin is `dP' + t' · dD` from the new one, with `dP' = dP + skipped · dD`.
            ray_dp += (light.distance + RAY_EPSILON) * ray_dd;
            // A light is a transparent pass in Cycles (`integrate_light_forward`): it spends the
            // transparent budget, and the path ends at the cap, after the emission above.
            transparent_bounce += 1;
            if transparent_bounce >= transparent_max_bounces {
                break;
            }
            continue;
        }
        let Some(hit) = hit else {
            let environment_weight = match previous {
                Scatter::Camera | Scatter::Delta(_) => 1.0,
                Scatter::Surface(vertex) => power_heuristic(
                    vertex.bsdf_pdf,
                    environment_light_pdf(vertex.normal, direction, importance),
                ),
                Scatter::Volume(phase_pdf) => {
                    power_heuristic(phase_pdf, volume_sampling_pdf(direction, importance))
                }
            };
            // The transparent film leaves camera-background world radiance to the external compositor.
            if !film_transparent || !matches!(previous, Scatter::Camera) {
                let environment = world_radiance(
                    doc,
                    &snapshot.scene_id,
                    images,
                    direction,
                    environment,
                    true,
                    evaluator_workspace,
                )?;
                radiance += throughput * environment * environment_weight;
            }
            escaped = true;
            break;
        };
        let footprint_here = ray_dp + hit.distance * ray_dd;
        let mut surface = make_surface(
            doc,
            geometry,
            hit,
            origin,
            direction,
            images,
            footprint_here,
            camera_matrices,
            ray,
            evaluator_workspace,
        )?;
        // Cycles' glossy blur (`surface_shader_prepare_closures`) runs before the hit is shaded, so the direct light, the
        // bounce and the widening all see the blurred closures.
        if let Some(blur_alpha) = glossy_blur_alpha(blur_glossy, min_ray_pdf) {
            surface.closures = surface.closures.blurred(blur_alpha);
        }
        if let Some(entry) = exit_entry.take() {
            // The exit of a subsurface entry is a diffuse surface (`subsurface_shader_data_setup`), not the material it
            // hits. Cycles shades it along the exit hit's own normal: `subsurface_disk` sets the probe ray to `Ng[hit]`,
            // flipped for a backfacing entry, and `subsurface_scatter` reverses that ray, so the exit faces outward.
            if let Some(outward) = intrinsic_normal(geometry, hit.triangle_index) {
                let outward = if exit_backfacing { -outward } else { outward };
                if surface.normal.dot(outward) < 0.0 {
                    surface.normal = -surface.normal;
                    surface.geometric_normal = -surface.geometric_normal;
                }
            }
            // Cycles' diffuse closure has no view term (`bsdf_diffuse_setup`, `bsdf_diffuse_eval` tests only N·omega), so
            // the exit's view is its own normal, and the exit is the entry closure's diffuse lobe alone.
            let mut params = exit_params(&entry);
            params.normal = surface.normal;
            surface.closures = Closures::new(
                vec![HitClosure::new(
                    1.0,
                    &params,
                    surface.normal,
                    surface.geometric_normal,
                    surface.closures.bump_correction,
                )],
                surface.normal,
                surface.geometric_normal,
                surface.closures.bump_correction,
            );
            // The exit's volume is zero (`exit_params`), so the exit neither passes the path nor changes the stack.
            surface.media.clear();
            surface.volume_boundary = false;
        } else if diffuse_ancestor {
            // A Burley closure falls back to diffuse after a diffuse bounce (Cycles' `bssrdf_setup` with
            // `PATH_RAY_DIFFUSE_ANCESTOR`). The random walk keeps its closure.
            surface.closures.disable_burley_subsurface();
        }
        if surface.volume_boundary || !surface.media.is_empty() {
            // A medium surface passes the path and changes the volume stack: each of its volume closures enters the
            // medium of its object, or leaves the one it entered.
            for medium in &surface.media {
                toggle_mesh_medium(&mut media, medium);
            }
            volume_bounds += 1;
            if volume_bounds > VOLUME_BOUNDS_MAX {
                break;
            }
            // As a transparent pass, the origin moves to the surface, so its footprint is the surface's.
            ray_dp = surface.footprint;
            origin = offset_origin(surface.point, direction, surface.geometric_normal);
            continue;
        }
        // Emission is the closures' radiance weighted by their coverage, so a path that ends here still shows it.
        radiance += throughput * surface.closures.emission;
        if terminate_on_next_surface {
            // Cycles shades the surface for emission, then ends the path (`integrate_surface_terminate`).
            break;
        }
        // Light sampling covers every closure's reflective share at this vertex, whichever share is picked below.
        if surface.closures.reflect_total > 0.0 {
            radiance += throughput
                * direct_lighting(
                    scene,
                    &media,
                    &surface,
                    environment,
                    bounce < max_bounces && !terminal,
                    transparent_max_bounces.saturating_sub(transparent_bounce),
                    rng,
                    evaluator_workspace,
                )?;
        }
        match surface.closures.pick(rng.next_f64()) {
            // Nothing reflects or passes (emission only), so the path ends at this surface.
            None => break,
            Some(Pick::Pass { multiplier }) => {
                // A transparent event is not a regular bounce: it spends its own budget, `transparent_bounce`, and does
                // not advance `bounce` (Cycles `path_state.h:118-121`, which gives transparent events their own max). It
                // writes no MIS state (`shade_surface.h:609-611`), so the next light hit is MIS-weighted against the last
                // non-transparent vertex, `previous`. The pass carries the transparent tint scaled by its pick weight, so
                // its expected throughput is Cycles' `Σ w · (1 − α) · tint`.
                throughput *= multiplier;
                transparent_bounce += 1;
                terminate_on_next_surface |= transparent_bounce >= transparent_cap;
                // Cycles keeps `ray.P` and `dP` through a transparent pass and lets `t` grow: moving the origin to the
                // surface with `dP` set to the surface's footprint gives the same footprint at every later hit.
                ray_dp = surface.footprint;
                origin = offset_origin(surface.point, direction, surface.geometric_normal);
            }
            Some(Pick::Reflect(index)) => {
                if terminal {
                    break;
                }
                let draw = surface
                    .closures
                    .reflect(index, surface.geometric_normal, rng);
                // A subsurface entry takes no bounce, so it is taken at the budget too (`subsurface_bounce`, which
                // `path_state_next` does not count). The path leaves at the method's exit, keeping this vertex's count.
                if let Reflection::Subsurface { weight, method } = draw {
                    let entry = surface.closures.items[index];
                    let Some(exit) = subsurface_exit(
                        geometry,
                        bvh,
                        &surface,
                        &entry,
                        method,
                        throughput * weight,
                        rng,
                    ) else {
                        break;
                    };
                    // The exit's normal faces outward as Cycles' `subsurface_bounce` and `subsurface_scatter` set it: a
                    // Burley exit is oriented by its probe's entry (`subsurface_disk`), and a random walk exit faces
                    // along the walk's last flight, since Cycles reverses that ray at the exit point.
                    exit_backfacing = match method {
                        SubsurfaceMethod::Burley => intrinsic_normal(geometry, hit.triangle_index)
                            .is_some_and(|normal| normal.dot(direction) > 0.0),
                        _ => intrinsic_normal(geometry, exit.hit.triangle_index)
                            .is_some_and(|normal| normal.dot(exit.direction) < 0.0),
                    };
                    // Cycles' `subsurface_bounce` (`kernel/integrator/subsurface.h`, lines 90-91): the exit ray carries the
                    // entry's footprint as its `dP`, and a zero `dD`.
                    ray_dp = surface.footprint;
                    ray_dd = 0.0;
                    throughput = exit.throughput;
                    origin = exit.origin;
                    direction = exit.direction;
                    exit_hit = Some(exit.hit);
                    exit_entry = Some(entry.params);
                    previous = Scatter::Camera;
                    // The exit is a diffuse closure (`subsurface_scatter`), so a light it meets is a diffuse-class hit.
                    ray = RayKind::Diffuse;
                    continue;
                }
                if bounce >= max_bounces {
                    break;
                }
                // Cycles' singular pdf for the widening average, `1e6 · branch · share`, carried from the draw; a
                // continuous draw has none.
                let singular_weight = match draw {
                    Reflection::Delta { mis_weight, .. } => mis_weight,
                    _ => 0.0,
                };
                let (next_direction, weight) = match draw {
                    Reflection::Continuous {
                        direction,
                        weight,
                        pdf,
                        ray: lobe,
                    } => {
                        diffuse_ancestor |= lobe == RayKind::Diffuse;
                        min_ray_pdf = min_ray_pdf.min(pdf);
                        previous = Scatter::Surface(MisVertex {
                            bsdf_pdf: pdf,
                            normal: surface.normal,
                            point: surface.point,
                        });
                        ray = lobe;
                        (direction, weight)
                    }
                    Reflection::Delta {
                        direction,
                        weight,
                        ray: lobe,
                        pdf,
                        ..
                    } => {
                        min_ray_pdf = min_ray_pdf.min(pdf);
                        previous = Scatter::Delta(surface.point);
                        ray = lobe;
                        (direction, weight)
                    }
                    // Cycles' `LABEL_NONE`: a rejected draw ends the path. A subsurface draw was handled above.
                    Reflection::Absorbed | Reflection::Subsurface { .. } => break,
                };
                if !next_direction.is_finite() || !weight.is_finite() {
                    break;
                }
                throughput *= weight;
                if bounce >= 3 {
                    let survival = throughput.max_element().clamp(0.05, 0.95);
                    if rng.next_f64() >= survival {
                        break;
                    }
                    throughput /= survival;
                }
                // Cycles' `bsdf_widen_dD` after a non-transparent bounce (`kernel/closure/bsdf.h`): the next ray carries the
                // hit's footprint as `dP`, and `dD` widens to `widen_scale · sqrt(avg roughness²)` when that is larger.
                ray_dp = surface.footprint;
                ray_dd = widen_ray_dd(
                    ray_dd,
                    widen_scale,
                    surface.closures.average_roughness_squared(
                        next_direction.normalize_or_zero(),
                        singular_weight,
                    ),
                );
                direction = next_direction.normalize_or_zero();
                origin = offset_origin(surface.point, direction, surface.geometric_normal);
                bounce += 1;
            }
        }
    }
    if !radiance.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "path tracer produced a non-finite radiance value",
        ));
    }
    // Cycles writes film transparency only for a path that escapes to the background, as
    // `average(throughput)` (`intern/cycles/kernel/integrator/shade_background.h`). Every other
    // termination leaves the pixel covered.
    let alpha = if escaped {
        1.0 - throughput.element_sum() / 3.0
    } else {
        1.0
    };
    Ok((radiance, alpha.clamp(0.0, 1.0)))
}

#[expect(
    clippy::too_many_arguments,
    reason = "center render passes need the resolved scene and independent selection ray"
)]
fn write_center_passes(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    volumes: &[VolumeInstance<'_>],
    images: &BTreeMap<String, ImageData>,
    camera: &Camera,
    width: u32,
    height: u32,
    film_transparent: bool,
    output: &mut RasterOutput,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<()> {
    let has_volume_only = geometry
        .prepared
        .iter()
        .any(PreparedMaterial::is_volume_only);

    // The camera's frame is built once for the pass, as the render builds each eye's frame once.
    let frame = camera_frame(camera, width, height)?;
    let side_by_side = camera.stereo_mode == StereoMode::SideBySide;
    let left_camera = if side_by_side {
        camera_eye(camera, -1.0)
    } else {
        *camera
    };
    let left_camera_matrices = camera_matrices_for_render(&left_camera)?;
    let right_camera_matrices = if side_by_side {
        let right_camera = camera_eye(camera, 1.0);
        camera_matrices_for_render(&right_camera)?
    } else {
        None
    };
    for y in 0..height {
        for x in 0..width {
            let index = usize::try_from(y)
                .ok()
                .and_then(|row| row.checked_mul(usize::try_from(width).ok()?))
                .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "pixel index exceeds platform limits",
                    )
                })?;
            // A pixel with no ray has no volume to attenuate, so its path-pass value stands.
            let Some((origin, direction)) =
                ray_for_sample_in_frame(&frame, camera, width, height, x, y, 0.5, 0.5)?
            else {
                continue;
            };
            let hit = closest_hit(
                geometry,
                bvh,
                origin,
                direction,
                RAY_EPSILON,
                f64::INFINITY,
                RayKind::Camera,
            );
            let segment_limit = hit.map_or(f64::INFINITY, |hit| hit.distance);
            let volume_transmittance = march_volumes(
                volumes,
                origin,
                direction,
                segment_limit,
                None,
                RayKind::Camera,
            )?;
            let Some(hit) = hit else {
                if film_transparent {
                    let alpha = 1.0 - volume_transmittance.element_sum() / 3.0;
                    output.linear_rgba[index][3] = finite_f32(alpha);
                    output.rgba[index * 4 + 3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                } else {
                    output.linear_rgba[index][3] = 1.0;
                    output.rgba[index * 4 + 3] = 255;
                }
                continue;
            };
            let footprint = camera_footprint(
                &frame,
                camera,
                width,
                height,
                x,
                y,
                0.5,
                0.5,
                (origin, direction),
            )?;
            let camera_matrices = if side_by_side && x >= width / 2 {
                right_camera_matrices.as_ref()
            } else {
                left_camera_matrices.as_ref()
            };
            let surface = make_surface(
                doc,
                geometry,
                hit,
                origin,
                direction,
                images,
                footprint.at_distance(hit.distance),
                camera_matrices,
                RayKind::Camera,
                evaluator_workspace,
            )?;
            if surface.closures.coverage <= 0.0 {
                continue;
            }
            output.ids[index] = surface.object_index;
            output.elements[index] = surface.element_index;
            output.depths[index] = finite_f32(camera.depth_of(surface.point));
            output.normals[index] = surface.normal.to_array().map(finite_f32);
            output.ambient_occlusion[index] = finite_f32(super::raster::ambient_visibility(
                bvh,
                geometry.triangles,
                geometry.prepared,
                surface.point,
                surface.geometric_normal,
                surface.normal,
                1.0,
                has_volume_only,
            ));
            output.albedo[index] = surface.closures.albedo.map(finite_f32);
            // The emission pass is the beauty's emission term: the coat attenuates it per channel.
            let emission = surface.closures.emission_pass;
            output.emission[index] = [
                finite_f32(emission.x),
                finite_f32(emission.y),
                finite_f32(emission.z),
                1.0,
            ];
            if film_transparent {
                let coverage = surface.closures.coverage;
                let alpha =
                    coverage + (1.0 - coverage) * (1.0 - volume_transmittance.element_sum() / 3.0);
                output.linear_rgba[index][3] = finite_f32(alpha);
                output.rgba[index * 4 + 3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    Ok(())
}

/// Source and destination RGB are associated radiance, including additive camera-visible emission.
fn overlay_over_rgba(source: [f32; 4], destination: [f32; 4], film_transparent: bool) -> [f64; 4] {
    let source_alpha = f64::from(source[3]).clamp(0.0, 1.0);
    let destination_alpha = f64::from(destination[3]).clamp(0.0, 1.0);
    let inverse_source_alpha = 1.0 - source_alpha;
    let alpha = if film_transparent {
        source_alpha + destination_alpha * inverse_source_alpha
    } else {
        1.0
    };
    let rgb: [f64; 3] = std::array::from_fn(|channel| {
        f64::from(source[channel]) + f64::from(destination[channel]) * inverse_source_alpha
    });
    [rgb[0], rgb[1], rgb[2], alpha]
}

/// Converts associated render RGB to straight bytes; alpha-zero additive radiance stays observable.
fn display_rgb_from_associated(rgb: [f64; 3], alpha: f64) -> [f64; 3] {
    if alpha > 0.0 {
        rgb.map(|channel| channel / alpha)
    } else {
        rgb
    }
}

#[cfg(test)]
mod overlay_tests {
    use std::collections::BTreeMap;

    use super::{
        AreaShape, Bounces, Camera, CameraFootprint, DMat4, DVec3, EvaluatorWorkspace, Geometry,
        Pcg32, PointLight, RasterOutput, RayKind, SceneRefs, TraceGeometry,
        display_rgb_from_associated, overlay_over_rgba, overlay_rasterized_lines, to_srgb_byte,
        trace_path,
    };
    use crate::{
        eval::{EvaluationContext, Snapshot},
        model::{LightType, SceneDoc},
        render::{
            bvh::Bvh,
            raster::{Line, Projection, StereoMode},
        },
    };

    #[test]
    fn associated_line_over_opaque_destination_keeps_full_coverage_contribution() {
        assert_eq!(
            overlay_over_rgba([0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0], true),
            [0.5, 0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn associated_line_over_fractional_destination_preserves_associated_energy() {
        assert_eq!(
            overlay_over_rgba([0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.25, 0.25], true),
            [0.5, 0.0, 0.125, 0.625]
        );
    }

    #[test]
    fn associated_line_over_transparent_destination_keeps_additive_radiance() {
        assert_eq!(
            overlay_over_rgba([0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 1.25, 0.0], true),
            [0.5, 0.0, 0.625, 0.5]
        );
    }

    #[test]
    fn zero_alpha_associated_buffer_keeps_additive_radiance() {
        assert_eq!(
            overlay_over_rgba([0.0; 4], [1.25, 0.5, 0.25, 0.0], true),
            [1.25, 0.5, 0.25, 0.0]
        );
    }

    #[test]
    fn physical_light_survives_fractional_stroke_in_producer_overlay_and_byte_encoding()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let camera = Camera {
            position: DVec3::ZERO,
            target: -DVec3::Z,
            up: DVec3::Y,
            near: 0.1,
            far: 10.0,
            projection: Projection::Orthographic { height: 2.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.0,
        };
        let geometry = Geometry {
            triangles: Vec::new(),
            lines: vec![Line {
                start: DVec3::new(0.0, 0.0, -1.0),
                end: DVec3::new(0.0, 0.0, -1.0),
                radius_start: 0.1,
                radius_end: 0.1,
                color: [0.0, 0.0, 0.0, 0.5],
                object_index: 1,
                element_index: 1,
                ray_mask: RayKind::Camera.bit(),
            }],
            wire_edges: Vec::new(),
            wire_points: Vec::new(),
            materials: Vec::new(),
            prepared: Vec::new(),
            objects: Vec::new(),
            elements: Vec::new(),
            bounds: None,
            warnings: Vec::new(),
            shader_warnings: std::cell::RefCell::new(Vec::new()),
            shadows: None,
        };
        let trace_geometry = TraceGeometry::of(&geometry);
        let bvh = Bvh::build(&geometry.triangles)?;
        let images = BTreeMap::new();
        let light = PointLight {
            position: DVec3::new(0.0, 0.0, -3.0),
            direction: DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Area,
            color: DVec3::X,
            energy: 1.0,
            radius: 0.0,
            soft_falloff: false,
            spot_size: 1.0,
            spot_blend: 0.0,
            area: Some(AreaShape {
                axis_u: DVec3::X,
                axis_v: DVec3::Y,
                half_u: 0.5,
                half_v: 0.5,
                ellipse: false,
                area: 1.0,
            }),
            spread: None,
            use_mis: true,
            volume_factor: 0.0,
            volume_radius_squared: 0.0,
            ray_mask: RayKind::Camera.bit(),
            camera_shape_eligible: true,
        };
        let lights = [light];
        let scene = SceneRefs {
            doc: &doc,
            snapshot: &snapshot,
            geometry: &trace_geometry,
            bvh: &bvh,
            lights: &lights,
            volumes: &[],
            images: &images,
            importance: None,
            camera_matrices: None,
        };
        let world_only_scene = SceneRefs {
            lights: &[],
            ..scene
        };
        let transparent_environment = DVec3::new(0.0, 8.0, 0.0);
        let mut environment_rng = Pcg32::for_sample(7, 0, 0, 1);
        let mut environment_workspace = EvaluatorWorkspace::default();
        let (transparent_world, transparent_alpha) = trace_path(
            world_only_scene,
            Bounces {
                surface: 0,
                volume: 0,
                transparent: 2,
            },
            DVec3::ZERO,
            -DVec3::Z,
            CameraFootprint::default(),
            transparent_environment,
            true,
            &mut environment_rng,
            &mut environment_workspace,
        )?;
        assert_eq!(transparent_world, DVec3::ZERO);
        assert_eq!(transparent_alpha, 0.0);
        let mut opaque_environment_rng = Pcg32::for_sample(7, 0, 0, 2);
        let mut opaque_environment_workspace = EvaluatorWorkspace::default();
        let (opaque_world, _) = trace_path(
            world_only_scene,
            Bounces {
                surface: 0,
                volume: 0,
                transparent: 2,
            },
            DVec3::ZERO,
            -DVec3::Z,
            CameraFootprint::default(),
            transparent_environment,
            false,
            &mut opaque_environment_rng,
            &mut opaque_environment_workspace,
        )?;
        assert_eq!(opaque_world, transparent_environment);
        assert_eq!(
            overlay_over_rgba(
                [0.0; 4],
                [
                    opaque_world.x as f32,
                    opaque_world.y as f32,
                    opaque_world.z as f32,
                    0.0,
                ],
                false
            ),
            [0.0, 8.0, 0.0, 1.0]
        );
        let mut rng = Pcg32::for_sample(7, 0, 0, 0);
        let mut workspace = EvaluatorWorkspace::default();
        let (radiance, alpha) = trace_path(
            scene,
            Bounces {
                surface: 0,
                volume: 0,
                transparent: 2,
            },
            DVec3::ZERO,
            -DVec3::Z,
            CameraFootprint::default(),
            DVec3::new(0.0, 8.0, 0.0),
            true,
            &mut rng,
            &mut workspace,
        )?;
        assert_eq!(alpha, 0.0);
        assert!(radiance.x > 0.0);
        assert_eq!(radiance.y, 0.0);
        assert_eq!(radiance.z, 0.0);

        // In a 1×1 image the optical center lies at the pixel corner, outside the minimum 0.5-pixel stroke radius.
        // A 2×2 frame puts it at pixel (0, 0)'s center, giving that pixel full stroke coverage.
        let mut output = RasterOutput {
            rgba: vec![0; 16],
            linear_rgba: vec![[radiance.x as f32, radiance.y as f32, radiance.z as f32, 0.0]; 4],
            linear_rgba_associated: true,
            ids: vec![0; 4],
            depths: vec![0.0; 4],
            elements: vec![0; 4],
            normals: vec![[0.0; 3]; 4],
            albedo: vec![[0.0; 4]; 4],
            emission: vec![[0.0; 4]; 4],
            ambient_occlusion: vec![1.0; 4],
        };
        overlay_rasterized_lines(&geometry, &camera, 2, 2, 0, true, &mut output)?;

        let pixel = output.linear_rgba[0];
        let expected_red = radiance.x * 0.5;
        let actual_red = f64::from(pixel[0]);
        assert!(
            (actual_red - expected_red).abs() < 1.0e-6,
            "full-coverage 0.5-alpha stroke should halve associated red radiance: actual={actual_red}, expected={expected_red}, difference={}",
            (actual_red - expected_red).abs()
        );
        assert_eq!(pixel[1], 0.0);
        assert_eq!(pixel[2], 0.0);
        assert_eq!(pixel[3], 0.5);
        assert_eq!(output.ids[0], 1);
        let display_rgb = display_rgb_from_associated(
            [
                f64::from(pixel[0]),
                f64::from(pixel[1]),
                f64::from(pixel[2]),
            ],
            f64::from(pixel[3]),
        );
        assert_eq!(output.rgba[0], to_srgb_byte(display_rgb[0]));
        assert_eq!(&output.rgba[1..4], &[0, 0, 128]);
        Ok(())
    }
}

fn overlay_rasterized_lines(
    geometry: &Geometry,
    camera: &Camera,
    width: u32,
    height: u32,
    seed: u32,
    film_transparent: bool,
    output: &mut RasterOutput,
) -> Result<()> {
    if geometry.lines.is_empty() {
        return Ok(());
    }
    let overlay = rasterize_with_lines(
        &[],
        &geometry.lines,
        camera,
        width,
        height,
        Mode::Beauty,
        [0, 0, 0, 0],
        None,
        1,
        seed,
        // One sample per pixel: EEVEE's film filter is off, so the overlay is drawn unfiltered.
        0.0,
        1.0,
        None,
        geometry.prepared.as_slice(),
        None,
        None,
        false,
    )?;
    for index in 0..output.ids.len() {
        let Some(object_index) = overlay.ids.get(index).copied().filter(|value| *value != 0) else {
            continue;
        };
        let Some(line_depth) = overlay.depths.get(index).copied() else {
            continue;
        };
        if output.ids.get(index).is_some_and(|value| *value != 0)
            && output
                .depths
                .get(index)
                .is_some_and(|depth| line_depth >= *depth)
        {
            continue;
        }
        let Some(source) = overlay.linear_rgba.get(index).copied() else {
            continue;
        };
        let Some(destination) = output.linear_rgba.get(index).copied() else {
            continue;
        };
        let source_alpha = f64::from(source[3]).clamp(0.0, 1.0);
        let color = overlay_over_rgba(source, destination, film_transparent);
        let alpha = color[3];
        if let Some(pixel) = output.linear_rgba.get_mut(index) {
            *pixel = color.map(finite_f32);
        }
        if let Some(value) = output.ids.get_mut(index) {
            *value = object_index;
        }
        if let Some(value) = output.depths.get_mut(index) {
            *value = line_depth;
        }
        if let Some(value) = overlay.elements.get(index).copied()
            && let Some(element) = output.elements.get_mut(index)
        {
            *element = value;
        }
        if let Some(value) = overlay.normals.get(index).copied()
            && let Some(normal) = output.normals.get_mut(index)
        {
            *normal = value;
        }
        if let Some(albedo) = output.albedo.get_mut(index) {
            let straight = if overlay.linear_rgba_associated {
                if source_alpha > 0.0 {
                    [
                        f64::from(source[0]) / source_alpha,
                        f64::from(source[1]) / source_alpha,
                        f64::from(source[2]) / source_alpha,
                    ]
                } else {
                    [0.0; 3]
                }
            } else {
                [
                    f64::from(source[0]),
                    f64::from(source[1]),
                    f64::from(source[2]),
                ]
            };
            *albedo = [
                finite_f32(straight[0]),
                finite_f32(straight[1]),
                finite_f32(straight[2]),
                source_alpha as f32,
            ];
        }
        if let Some(emission) = output.emission.get_mut(index) {
            *emission = [0.0; 4];
        }
        if let Some(ao) = output.ambient_occlusion.get_mut(index) {
            *ao = 1.0;
        }
        let byte_index = index.saturating_mul(4);
        if let Some(bytes) = output
            .rgba
            .get_mut(byte_index..byte_index.saturating_add(4))
        {
            let display_rgb = display_rgb_from_associated([color[0], color[1], color[2]], alpha);
            bytes.copy_from_slice(&[
                to_srgb_byte(display_rgb[0]),
                to_srgb_byte(display_rgb[1]),
                to_srgb_byte(display_rgb[2]),
                (alpha * 255.0).round() as u8,
            ]);
        }
    }
    Ok(())
}

/// The nearest hit a ray of kind `ray` can see; objects hidden from that kind are passed through.
fn closest_hit(
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
    ray: RayKind,
) -> Option<BvhHit> {
    bvh.closest_hit_where(
        geometry.triangles,
        origin,
        direction,
        t_min,
        t_max,
        &|mask| mask & ray.bit() != 0,
    )
}

/// The shading of one hit for the transparent pass of a realtime shadow ray (`render/scene.rs`). Each closure carries its
/// alpha after the material's alpha mode, as the path tracer's closures do, so a realtime shadow and a path-traced one
/// read the same transparency. The pass reads neither the displacement gradient nor the layers, so only the material
/// evaluation runs.
pub(super) fn hit_params(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    camera_matrices: Option<(&DMat4, &DMat4)>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<SurfaceShading> {
    with_hit_frame(
        doc,
        geometry,
        hit,
        origin,
        direction,
        images,
        None,
        camera_matrices,
        false,
        None,
        |frame| {
            if let Some(node_id) = frame.prepared.realtime_transparency_bump() {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!(
                        "Bump node `{node_id}` affects realtime transparent-shadow coverage through its Height samples"
                    ),
                    serde_json::json!({
                        "feature_id": "shader.node.Bump.realtime_transparent_shadow_height",
                        "node_id": node_id.as_str(),
                    }),
                ));
            }
            let context = frame.context.with_opacity_only();
            let mut shading = evaluate_surface_with_workspace_semantics(
                frame.material,
                frame.prepared,
                &context,
                evaluator_workspace,
                crate::node_kernels::MathSemantics::Eevee,
            )?;
            for closure in &mut shading.closures {
                apply_hit_alpha(frame.material, &mut closure.params);
            }
            Ok(shading)
        },
    )
}

/// A hit's position, normals, and the evaluation context of its material. The context borrows its layer tables, so a
/// frame exists only inside [`with_hit_frame`].
struct HitFrame<'a> {
    point: DVec3,
    shadow_point: DVec3,
    back_facing: bool,
    geometric_normal: DVec3,
    normal: DVec3,
    object: &'a ObjectRecord,
    triangle: &'a Triangle,
    material: &'a Material,
    prepared: &'a PreparedMaterial,
    context: HitContext<'a>,
}

/// Builds the frame of `hit` and calls `body` with it: the one place a hit's geometry, UVs and context are set up.
fn with_hit_frame<T>(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    footprint: Option<f64>,
    camera_matrices: Option<(&DMat4, &DMat4)>,
    path_traced: bool,
    light_falloff_distance: Option<f64>,
    body: impl FnOnce(&HitFrame<'_>) -> Result<T>,
) -> Result<T> {
    let triangle = geometry.triangles.get(hit.triangle_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "BVH returned an invalid triangle index",
        )
    })?;
    let point = origin + direction * hit.distance;
    let shadow_point = smooth_shadow_origin(triangle, point, hit.barycentric);
    let mut geometric_normal = (triangle.positions[1] - triangle.positions[0])
        .cross(triangle.positions[2] - triangle.positions[0])
        .normalize_or_zero();
    // Cycles' `SD_BACKFACING`: the geometric normal faces away from the ray.
    let back_facing = geometric_normal.dot(-direction) < 0.0;
    if back_facing {
        geometric_normal = -geometric_normal;
    }
    // Smooth triangles light with the blended corner normal on the geometric normal's side.
    let normal = match triangle.normals {
        None => geometric_normal,
        Some(corners) => {
            let shading = blend_corner_normals(corners, hit.barycentric, geometric_normal);
            if shading.dot(geometric_normal) < 0.0 {
                -shading
            } else {
                shading
            }
        }
    };

    let object = object_for_index(geometry.objects, triangle.object_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "triangle references an unmapped object",
        )
    })?;
    let material = geometry
        .materials
        .get(triangle.material_index)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "triangle references an unmapped material",
            )
        })?;
    let prepared = geometry
        .prepared
        .get(triangle.material_index)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "triangle references an unmapped material",
            )
        })?;
    let element = triangle
        .element_index
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| geometry.elements.get(index));
    let uv_layers = HitUvLayers {
        names: &object.uv_layers,
        corners: &triangle.layer_uvs,
        positions: triangle.positions,
        barycentric: hit.barycentric,
        render: element.map_or(object.uv_render, |element| element.uv_render),
        available_mask: element.and_then(|element| element.uv_available_mask),
    };
    let mut tables = LayerTables::new();
    let mut context = layered_context(
        doc,
        images,
        geometry.warnings,
        object,
        triangle,
        HitPose {
            point,
            object: object_space(object, point),
            normal,
            geometric_normal,
            back_facing,
            path_traced,
            view: -direction,
            barycentric: hit.barycentric,
        },
        None,
        uv_layers,
        &mut tables,
    )?;
    if let Some((camera_to_world, world_to_camera)) = camera_matrices {
        context.set_camera_matrices(*camera_to_world, *world_to_camera);
    }
    context.geometric_normal = geometric_normal;
    context.back_facing = back_facing;
    context.realtime_differentials = None;
    context.light_falloff_distance = light_falloff_distance;
    context.path_traced = path_traced;
    let context = if path_traced {
        match footprint {
            Some(dp) => context.with_footprint(HitFootprint {
                dp,
                geometric_normal,
            }),
            None => context,
        }
    } else {
        context
    };
    body(&HitFrame {
        point,
        shadow_point,
        back_facing,
        geometric_normal,
        normal,
        object,
        triangle,
        material,
        prepared,
        context,
    })
}

/// The alpha of a surface after its material's alpha mode. The graph is the only source of transparency for a material
/// with one (Cycles ignores the render method).
fn apply_hit_alpha(material: &Material, params: &mut BsdfParams) {
    params.alpha = match material.alpha_mode.as_str() {
        "opaque" if material.node_tree.is_none() => 1.0,
        "clip" => {
            if params.alpha >= material.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
        _ => params.alpha.clamp(0.0, 1.0),
    };
}

/// The media a hit's Volume output bounds: one per closure that scatters, absorbs, or emits, its coefficients
/// weighted by the closure's weight, since Cycles sums them (`volume_shader_eval`).
fn surface_media(object_index: u32, volume: &[ShaderClosure]) -> Vec<MeshMedium> {
    let mut media = Vec::new();
    surface_media_into(object_index, volume, &mut media);
    media
}

fn surface_media_into(object_index: u32, volume: &[ShaderClosure], media: &mut Vec<MeshMedium>) {
    media.clear();
    media.reserve(volume.len());
    for (closure, shader) in volume.iter().enumerate() {
        let weight = shader.weight;
        let mut params = shader.params;
        params.volume_scatter = params.volume_scatter.map(|value| weight * value);
        params.volume_absorption = params.volume_absorption.map(|value| weight * value);
        params.volume_emission = params.volume_emission.map(|value| weight * value);
        if params.has_medium() || params.volume_emission.iter().any(|value| *value != 0.0) {
            media.push(MeshMedium {
                object_index,
                closure,
                params,
            });
        }
    }
}

fn surface_froxel_media_into(
    object: &ObjectRecord,
    triangle: &Triangle,
    volume: &[ShaderClosure],
    media: &mut Vec<FroxelMeshMedium>,
) {
    media.clear();
    media.reserve(volume.len());
    let generated_per_local = triangle.generated.per_local;
    let local_origin = object_space(object, triangle.positions[0]);
    let generated_origin = triangle.generated.corners[0]
        - DVec3::new(
            local_origin.x * generated_per_local.x,
            local_origin.y * generated_per_local.y,
            local_origin.z * generated_per_local.z,
        );
    for closure in 0..volume.len() {
        media.push(FroxelMeshMedium {
            object_index: triangle.object_index,
            material_index: triangle.material_index,
            closure,
            generated_origin,
            generated_per_local,
        });
    }
}

fn make_surface(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    footprint: f64,
    camera_matrices: Option<&(DMat4, DMat4)>,
    ray_kind: RayKind,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<Surface> {
    let triangle_index = hit.triangle_index;
    with_hit_frame(
        doc,
        geometry,
        hit,
        origin,
        direction,
        images,
        Some(footprint),
        camera_matrices.map(|matrices| (&matrices.0, &matrices.1)),
        true,
        Some(hit.distance),
        |frame| {
            let shading = evaluate_surface_with_workspace(
                frame.material,
                frame.prepared,
                &frame.context,
                evaluator_workspace,
            )?;
            // A displacement height field perturbs every closure's normal, as Cycles' bump displaces `sd->N` before
            // the closures are built.
            let gradient = if shading.displacement_height != 0.0
                && frame.material.displacement_method != "displacement"
            {
                Some(height_gradient(&frame.context, frame.normal, |sample| {
                    evaluate_displacement_with_workspace_semantics(
                        frame.material,
                        frame.prepared,
                        sample,
                        evaluator_workspace,
                        crate::node_kernels::MathSemantics::Cycles,
                    )
                })?)
            } else {
                None
            };
            // Cycles' bump from displacement writes the bumped normal into `sd->N` before the closures are built
            // (`scene/shader_graph.cpp`, `bump_from_displacement`), so the shading-side test and the diffuse smoothing
            // measure against the bumped normal, not the undisplaced one.
            let smooth = match gradient {
                Some(shift) => {
                    let bumped = (frame.normal - shift).normalize_or_zero();
                    let bumped = if bumped == DVec3::ZERO {
                        frame.normal
                    } else {
                        bumped
                    };
                    if bumped.dot(-direction) < 0.0 {
                        -bumped
                    } else {
                        bumped
                    }
                }
                None => frame.normal,
            };
            let mut items = Vec::with_capacity(shading.closures.len());
            for closure in shading
                .closures
                .iter()
                .filter(|closure| closure.weight > 0.0)
            {
                let mut params = closure.params;
                apply_hit_alpha(frame.material, &mut params);
                params.back_facing = frame.back_facing;
                let perturbed = gradient.map_or(params.normal, |shift| params.normal - shift);
                let mut normal = if perturbed.is_finite() {
                    perturbed.normalize_or_zero()
                } else {
                    frame.normal
                };
                if normal.dot(-direction) < 0.0 {
                    normal = -normal;
                }
                params.normal = normal;
                let mut item = HitClosure::new(
                    closure.weight,
                    &params,
                    -direction,
                    frame.geometric_normal,
                    frame.material.use_bump_map_correction,
                );
                if ray_kind != RayKind::Camera && params.thin_wall {
                    let transparent =
                        thin_glass_shadow_transparency(&params, &item.layers) * params.alpha;
                    if transparent.max_element() > 0.0 {
                        item.pass_tint += transparent;
                        item.params.thin_glass_transparent = true;
                        item.layers = layers_at(
                            &item.params,
                            normal,
                            -direction,
                            frame.geometric_normal,
                            frame.material.use_bump_map_correction,
                        );
                    }
                }
                items.push(item);
            }
            // The dominant closure's normal is the surface's reference normal, for light sampling and the normal pass.
            let normal = heaviest(&items, |closure| closure.weight)
                .map_or(frame.normal, |closure| closure.params.normal);
            Ok(Surface {
                point: frame.point,
                normal,
                geometric_normal: frame.geometric_normal,
                shadow_point: frame.shadow_point,
                closures: Closures::new(
                    items,
                    smooth,
                    frame.geometric_normal,
                    frame.material.use_bump_map_correction,
                ),
                volume_boundary: frame.prepared.is_volume_only(),
                media: surface_media(frame.triangle.object_index, &shading.volume),
                object_index: frame.triangle.object_index,
                element_index: frame.triangle.element_index,
                triangle_index,
                footprint,
            })
        },
    )
}

/// The media and shadow transparency of one hit, which is all a shadow ray or the camera's medium probe reads from it.
/// Displacement gradients are not evaluated here; smooth Thin Wall transmission uses its Principled shading normal.
/// Consequently a warning or error produced only by gradient samples is still raised on path and camera hits.
fn shadow_hit(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    footprint: Option<f64>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(Vec<MeshMedium>, DVec3, bool)> {
    let mut media = Vec::new();
    let (transparency, volume_boundary) = shadow_hit_into(
        doc,
        geometry,
        hit,
        origin,
        direction,
        images,
        footprint,
        RayKind::Shadow,
        None,
        true,
        Some(hit.distance),
        &mut media,
        None,
        evaluator_workspace,
    )?;
    Ok((media, transparency, volume_boundary))
}

fn shadow_hit_into(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    footprint: Option<f64>,
    ray_kind: RayKind,
    camera_matrices: Option<(&DMat4, &DMat4)>,
    path_traced: bool,
    light_falloff_distance: Option<f64>,
    media: &mut Vec<MeshMedium>,
    froxel_media: Option<&mut Vec<FroxelMeshMedium>>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(DVec3, bool)> {
    with_hit_frame(
        doc,
        geometry,
        hit,
        origin,
        direction,
        images,
        footprint,
        camera_matrices,
        path_traced,
        light_falloff_distance,
        |frame| {
            let shading = evaluate_surface_with_workspace(
                frame.material,
                frame.prepared,
                &frame.context,
                evaluator_workspace,
            )?;
            // The closures `make_surface` keeps (weight above zero), with alpha after the material's alpha mode, as `Closures::new`
            // reads them.
            let mut transparency = shadow_transparency(
                shading
                    .closures
                    .iter()
                    .filter(|closure| closure.weight > 0.0)
                    .map(|closure| {
                        let mut params = closure.params;
                        apply_hit_alpha(frame.material, &mut params);
                        (closure.weight, params.alpha, transparent_tint(&params))
                    }),
            );
            if ray_kind == RayKind::Shadow {
                let view = -direction;
                for closure in &shading.closures {
                    if closure.weight <= 0.0 || !closure.params.thin_wall {
                        continue;
                    }
                    let mut params = closure.params;
                    apply_hit_alpha(frame.material, &mut params);
                    params.back_facing = frame.back_facing;
                    let normal = if params.normal.dot(view) < 0.0 {
                        -params.normal
                    } else {
                        params.normal
                    };
                    params.normal = normal;
                    let layers = layers_at(
                        &params,
                        normal,
                        view,
                        frame.geometric_normal,
                        frame.material.use_bump_map_correction,
                    );
                    transparency += thin_glass_shadow_transparency(&params, &layers)
                        * (closure.weight * params.alpha);
                }
            }
            let volume_boundary = frame.prepared.is_volume_only();
            surface_media_into(frame.triangle.object_index, &shading.volume, media);
            if let Some(froxel_media) = froxel_media {
                froxel_media.clear();
                if frame.prepared.has_volume_output()
                    || (frame.material.node_tree.is_none()
                        && frame.material.volume_density.is_finite()
                        && frame.material.volume_density > 0.0)
                {
                    surface_froxel_media_into(
                        frame.object,
                        frame.triangle,
                        &shading.volume,
                        froxel_media,
                    );
                }
            }
            Ok((transparency, volume_boundary))
        },
    )
}

pub(super) fn object_for_index(
    objects: &[ObjectRecord],
    object_index: u32,
) -> Option<&ObjectRecord> {
    object_index
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| objects.get(index))
}

fn direct_lighting(
    scene: SceneRefs<'_, '_>,
    media: &[MeshMedium],
    surface: &Surface,
    environment: DVec3,
    // Whether the path samples a BSDF continuation here. Only then does a BSDF ray pair with the
    // light sample for MIS; at the last vertex the light sample is unweighted.
    scatters: bool,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let SceneRefs {
        doc,
        snapshot,
        lights,
        volumes,
        images,
        importance,
        ..
    } = scene;
    let normal = surface.normal;
    let geometric = surface.geometric_normal;
    let closures = &surface.closures;
    let mut contribution = DVec3::ZERO;
    for light in lights {
        let Some(sample) = sample_light(light, surface.point, rng) else {
            continue;
        };
        // Every closure's own lobes and cosine (`Closures::bsdf_eval`), keeping the lobes this light's ray mask admits, as
        // Cycles drops a closure from a light that excludes its kind. Below the surface only a glass lobe contributes, so
        // a direction that no closure reaches skips the shadow ray.
        let bsdf = closures.bsdf_eval(sample.direction, None, light.ray_mask);
        if bsdf.max_element() <= 0.0 || sample.radiance.max_element() <= 0.0 {
            continue;
        }
        let visibility = shadow_transmittance(
            scene,
            media,
            ShadowRay {
                point: surface.shadow_point,
                normal: geometric,
                shading: closures.smooth,
                direction: sample.direction,
                distance: sample.distance,
                footprint: surface.footprint,
            },
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        if visibility.max_element() <= 0.0 {
            continue;
        }
        // A BSDF ray reaches a hittable light only when the path continues past this vertex and no volume can
        // scatter it first (`trace_path`).
        let mis = if scatters && volumes.is_empty() && light.hittable() {
            power_heuristic(sample.pdf, closures.continuous_pdf(sample.direction))
        } else {
            1.0
        };
        contribution += bsdf * sample.radiance * (visibility * mis);
    }
    // The environment's cosine strategy follows the dominant closure's normal, as `environment_light_pdf` takes it.
    let direction = match importance {
        Some(sampler) if rng.next_f64() >= 0.5 => sampler.sample(rng).0,
        _ => sample_cosine_hemisphere(normal, rng).0,
    };
    let light_pdf = environment_light_pdf(normal, direction, importance);
    // The importance sampler can draw below the surface, which only a glass lobe reaches.
    let bsdf = if light_pdf > 0.0 {
        closures.bsdf_eval(direction, None, RayKind::ALL_MASK)
    } else {
        DVec3::ZERO
    };
    if bsdf.max_element() > 0.0 {
        let environment_sample = world_radiance(
            doc,
            &snapshot.scene_id,
            images,
            direction,
            environment,
            true,
            evaluator_workspace,
        )?;
        let visibility = shadow_transmittance(
            scene,
            media,
            ShadowRay {
                point: surface.shadow_point,
                normal: geometric,
                shading: closures.smooth,
                direction,
                distance: f64::INFINITY,
                footprint: surface.footprint,
            },
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        let weight = if scatters {
            power_heuristic(light_pdf, closures.continuous_pdf(direction))
        } else {
            1.0
        };
        contribution += bsdf * environment_sample * (visibility * (weight / light_pdf));
    }
    Ok(contribution)
}

/// The direct light of a volume scatter: each light's next-event estimate weighted by the scatter's phase function, and one
/// environment sample with MIS against the phase function.
fn direct_volume_lighting(
    scene: SceneRefs<'_, '_>,
    media: &[MeshMedium],
    site: &VolumeSite<'_>,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let mut contribution = DVec3::ZERO;
    for (index, light) in scene.lights.iter().enumerate() {
        // Cycles' volume light sampling skips a light hidden from volume scatter (`SHADER_EXCLUDE_SCATTER`, `shade_volume.h`).
        if light.ray_mask & RayKind::VolumeScatter.bit() == 0 {
            continue;
        }
        let weight = site.light_weights.get(index).copied().unwrap_or(1.0);
        if weight <= 0.0 {
            continue;
        }
        let (radiance, direction) = volume_light_nee(
            scene,
            media,
            light,
            site.point,
            site.footprint,
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        contribution += radiance * (site.phase.density(site.incoming.dot(direction)) * weight);
    }
    let (direction, pdf) = if let Some(sampler) = scene.importance {
        sampler.sample(rng)
    } else {
        sample_uniform_sphere(rng)
    };
    if pdf > 0.0 {
        let incoming = world_radiance(
            scene.doc,
            &scene.snapshot.scene_id,
            scene.images,
            direction,
            site.environment,
            true,
            evaluator_workspace,
        )?;
        let visibility = shadow_transmittance(
            scene,
            media,
            ShadowRay {
                point: site.point,
                normal: DVec3::ZERO,
                shading: DVec3::ZERO,
                direction,
                distance: f64::INFINITY,
                footprint: site.footprint,
            },
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        let phase = site.phase.density(site.incoming.dot(direction));
        let weight = power_heuristic(pdf, phase);
        contribution += incoming * (phase * visibility * weight / pdf);
    }
    Ok(contribution)
}

fn sample_uniform_sphere(rng: &mut Pcg32) -> (DVec3, f64) {
    let cosine = 1.0 - 2.0 * rng.next_f64();
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    let angle = std::f64::consts::TAU * rng.next_f64();
    (
        DVec3::new(sine * angle.cos(), sine * angle.sin(), cosine),
        1.0 / (4.0 * std::f64::consts::PI),
    )
}

fn power_heuristic(first_pdf: f64, second_pdf: f64) -> f64 {
    let first_squared = first_pdf * first_pdf;
    let second_squared = second_pdf * second_pdf;
    first_squared / (first_squared + second_squared).max(f64::MIN_POSITIVE)
}

/// The density with which `direct_volume_lighting` and the environment miss sample a direction: the
/// importance map's density, or the uniform sphere's when there is no map.
fn volume_sampling_pdf(direction: DVec3, importance: Option<&WorldImportanceSampler>) -> f64 {
    importance.map_or(1.0 / (4.0 * std::f64::consts::PI), |sampler| {
        sampler.pdf(direction)
    })
}

/// Blender's `cycles.volume_sampling` for a segment, from the volumes it crosses, as Cycles'
/// `volume_stack_sample_method` combines them: multiple importance sampling if a volume asks for it or asks for
/// both distance and equiangular, else the one strategy they share. `None` when no volume is crossed.
fn segment_sampling(
    samplings: impl IntoIterator<Item = crate::model::VolumeSampling>,
) -> Option<crate::model::VolumeSampling> {
    use crate::model::VolumeSampling;
    let mut distance = false;
    let mut equiangular = false;
    for sampling in samplings {
        match sampling {
            VolumeSampling::MultipleImportance => return Some(VolumeSampling::MultipleImportance),
            VolumeSampling::Distance => distance = true,
            VolumeSampling::Equiangular => equiangular = true,
        }
    }
    match (distance, equiangular) {
        (true, true) => Some(VolumeSampling::MultipleImportance),
        (false, true) => Some(VolumeSampling::Equiangular),
        (true, false) => Some(VolumeSampling::Distance),
        (false, false) => None,
    }
}

/// The point that a light's equiangular sampling aims at (Cycles' `light_sample_from_volume_segment`): a light
/// sample from the segment's origin, so the sampled light position along its direction. A sun has no position
/// and takes distance sampling only.
fn equiangular_point(light: &PointLight, origin: DVec3, rng: &mut Pcg32) -> Option<DVec3> {
    if light.kind == LightType::Sun {
        return None;
    }
    let sample = sample_light(light, origin, rng)?;
    sample
        .distance
        .is_finite()
        .then(|| origin + sample.direction * sample.distance)
}

/// The equiangular point and sampling choice of each light for one segment (Cycles'
/// `volume_direct_sample_method`): under equiangular sampling every light with a point is sampled equiangularly, under
/// multiple importance sampling each is with probability one half, and under distance sampling none is. A light with
/// no point (a sun, or no sample) takes distance sampling only.
fn equiangular_choices(
    lights: &[PointLight],
    origin: DVec3,
    method: Option<crate::model::VolumeSampling>,
    rng: &mut Pcg32,
) -> Vec<(Option<DVec3>, bool)> {
    use crate::model::VolumeSampling;
    lights
        .iter()
        .map(|light| {
            let point = match method {
                Some(VolumeSampling::Equiangular | VolumeSampling::MultipleImportance) => {
                    equiangular_point(light, origin, rng)
                }
                Some(VolumeSampling::Distance) | None => None,
            };
            let sampled = match (method, point) {
                (Some(VolumeSampling::Equiangular), Some(_)) => true,
                (Some(VolumeSampling::MultipleImportance), Some(_)) => rng.next_f64() < 0.5,
                _ => false,
            };
            (point, sampled)
        })
        .collect()
}

/// The geometry of Cycles' equiangular sampling (`volume_equiangular_sample`): the light's offset along the ray,
/// its distance from the ray, and the angle that the segment `[0, distance]` subtends at the light.
struct Equiangular {
    delta: f64,
    perpendicular: f64,
    theta_a: f64,
    theta_span: f64,
}

fn equiangular_geometry(
    light: DVec3,
    origin: DVec3,
    direction: DVec3,
    distance: f64,
) -> Option<Equiangular> {
    let delta = (light - origin).dot(direction);
    let perpendicular = (light - origin - direction * delta).length();
    if perpendicular <= 0.0 || !perpendicular.is_finite() {
        return None;
    }
    let theta_a = (-delta).atan2(perpendicular);
    let theta_b = (distance - delta).atan2(perpendicular);
    Some(Equiangular {
        delta,
        perpendicular,
        theta_a,
        theta_span: theta_b - theta_a,
    })
}

/// Cycles' equiangular sample of `[0, distance]` (`volume_equiangular_sample`): a distance along the ray and its
/// density per unit distance. `None` when the light lies on the ray.
fn equiangular_sample(
    light: DVec3,
    origin: DVec3,
    direction: DVec3,
    distance: f64,
    xi: f64,
) -> Option<(f64, f64)> {
    let geometry = equiangular_geometry(light, origin, direction, distance)?;
    if geometry.theta_span < 1.0e-6 {
        // Cycles samples uniformly when the segment subtends almost no angle.
        if distance <= 0.0 || !distance.is_finite() {
            return None;
        }
        return Some((xi * distance, 1.0 / distance));
    }
    let offset = geometry.perpendicular * (geometry.theta_a + xi * geometry.theta_span).tan();
    let pdf = geometry.perpendicular
        / (geometry.theta_span
            * (geometry.perpendicular * geometry.perpendicular + offset * offset));
    Some(((geometry.delta + offset).clamp(0.0, distance), pdf))
}

/// The density per unit distance of `equiangular_sample` at `t` (`volume_equiangular_pdf`).
fn equiangular_pdf(light: DVec3, origin: DVec3, direction: DVec3, distance: f64, t: f64) -> f64 {
    let Some(geometry) = equiangular_geometry(light, origin, direction, distance) else {
        return 0.0;
    };
    if geometry.theta_span < 1.0e-6 {
        return if distance <= 0.0 || !distance.is_finite() {
            0.0
        } else {
            1.0 / distance
        };
    }
    let offset = t - geometry.delta;
    geometry.perpendicular
        / (geometry.theta_span
            * (geometry.perpendicular * geometry.perpendicular + offset * offset))
}

/// The density per unit distance of the first real event of `trace_mesh_medium` being a scatter at `t`: the
/// channel-mean scatter rate times the survival at the channel-mean extinction. Its weight in the MIS of
/// `volume_direct_scatter_mis` needs only to be a positive function of `t`; the scatter's estimator is already
/// weighted by the path, so the MIS stays unbiased for any such function.
fn distance_scatter_pdf(medium: &BsdfParams, t: f64) -> f64 {
    let scatter = DVec3::from_array(medium.volume_scatter);
    let extinction = scatter + DVec3::from_array(medium.volume_absorption);
    scatter.element_sum() / 3.0 * (-extinction.element_sum() / 3.0 * t).exp()
}

/// One point light's next-event estimate at `point`: the sample's radiance through its shadow transmittance, and the
/// sample's direction. The caller weights the estimate by the phase function at that direction.
fn volume_light_nee(
    scene: SceneRefs<'_, '_>,
    media: &[MeshMedium],
    light: &PointLight,
    point: DVec3,
    footprint: f64,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(DVec3, DVec3)> {
    let Some(sample) = sample_light(light, point, rng) else {
        return Ok((DVec3::ZERO, DVec3::ZERO));
    };
    if sample.radiance.max_element() <= 0.0 {
        return Ok((DVec3::ZERO, sample.direction));
    }
    let visibility = shadow_transmittance(
        scene,
        media,
        ShadowRay {
            point,
            normal: DVec3::ZERO,
            shading: DVec3::ZERO,
            direction: sample.direction,
            distance: sample.distance,
            footprint,
        },
        transparent_budget,
        rng,
        evaluator_workspace,
    )?;
    Ok((sample.radiance * visibility, sample.direction))
}

/// The transmittance of a shadow ray: opaque hits block it, transparent surfaces and medium boundaries pass it, and each
/// medium it crosses attenuates it.
fn shadow_transmittance(
    scene: SceneRefs<'_, '_>,
    media: &[MeshMedium],
    ray: ShadowRay,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    shadow_transmittance_with_volumetric_shadows(
        scene,
        media,
        ray,
        transparent_budget,
        true,
        rng,
        evaluator_workspace,
    )
}

fn shadow_transmittance_with_volumetric_shadows(
    scene: SceneRefs<'_, '_>,
    media: &[MeshMedium],
    ray: ShadowRay,
    transparent_budget: u32,
    volumetric_shadows: bool,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let SceneRefs {
        doc,
        geometry,
        bvh,
        volumes,
        images,
        ..
    } = scene;
    let ShadowRay {
        point,
        normal,
        shading,
        direction,
        distance,
        footprint,
    } = ray;
    let mut transparent_hits = 0u32;
    let mut blocked = false;
    let mut failure = None;
    // The shadow ray starts in the receiver's volume stack and crosses each medium by its transmittance.
    let mut shadow_media = if volumetric_shadows {
        media.to_vec()
    } else {
        Vec::new()
    };
    let mut transmittance = DVec3::ONE;
    let remaining = if distance.is_finite() {
        (distance - RAY_EPSILON).max(0.0)
    } else {
        f64::INFINITY
    };
    let (origin, remaining) = bvh.walk_hits(
        geometry.triangles,
        WalkRay {
            origin: shadow_origin(point, direction, normal, shading),
            direction,
            epsilon: RAY_EPSILON,
            remaining,
            kind: Some(RayKind::Shadow),
        },
        |hit, origin| {
            let (media, transparency, volume_boundary) = match shadow_hit(
                doc,
                geometry,
                hit,
                origin,
                direction,
                images,
                Some(footprint),
                evaluator_workspace,
            ) {
                Ok(shadow) => shadow,
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            };
            if volumetric_shadows && !volumes.is_empty() {
                match march_volumes(
                    volumes,
                    origin,
                    direction,
                    hit.distance,
                    Some(rng),
                    RayKind::Shadow,
                ) {
                    Ok(voxel_transmittance) => transmittance *= voxel_transmittance,
                    Err(error) => {
                        failure = Some(error);
                        return false;
                    }
                }
            }
            if volumetric_shadows && let Some(stack) = stack_medium(&shadow_media) {
                transmittance *= medium_transmittance(&stack, hit.distance);
            }
            if volumetric_shadows {
                for medium in &media {
                    toggle_mesh_medium(&mut shadow_media, medium);
                }
            }
            if volume_boundary {
                return true;
            }
            // Cycles counts each transparent surface a shadow ray crosses against the path's remaining budget and blocks
            // the ray past it (`intern/cycles/kernel/bvh/intersect_filter.h`).
            transparent_hits += 1;
            if transparent_hits > transparent_budget {
                blocked = true;
                return false;
            }
            // Shadow transparency is the closures' transparent share (`surface_shader_transparency`); ordinary glass
            // blocks, while Cycles' smooth Thin Wall transmission becomes a transparent closure for non-camera rays
            // (`bsdf_thin_glass_transmission_setup`), which `shadow_hit` adds here.
            transmittance *= transparency;
            if transmittance.max_element() <= 1.0e-5 {
                blocked = true;
                return false;
            }
            true
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    if blocked {
        return Ok(DVec3::ZERO);
    }
    // The last segment, to the light or to infinity, lies in the stack's combined medium.
    let volume_transmittance = if volumetric_shadows {
        if let Some(stack) = stack_medium(&shadow_media) {
            transmittance *= medium_transmittance(&stack, remaining);
        }
        march_volumes(
            volumes,
            origin,
            direction,
            remaining,
            Some(rng),
            RayKind::Shadow,
        )?
    } else {
        DVec3::ONE
    };
    Ok(transmittance.clamp(DVec3::ZERO, DVec3::ONE) * volume_transmittance)
}

fn sample_light(light: &PointLight, point: DVec3, rng: &mut Pcg32) -> Option<LightSample> {
    if !light.energy.is_finite() || light.energy <= 0.0 {
        return None;
    }
    if light.kind == LightType::Sun {
        let direction = (-light.direction).normalize_or_zero();
        if direction == DVec3::ZERO {
            return None;
        }
        return Some(LightSample {
            direction,
            distance: f64::INFINITY,
            radiance: light.color * light.energy,
            pdf: 0.0,
        });
    }
    if light.kind == LightType::Area {
        return sample_area_light(light, point, rng);
    }
    // Point and spot lights, as `point_light_sample` and `spot_light_sample` in
    // `intern/cycles/kernel/light/`. Each returned `radiance` is the sample's weight, the emitter
    // radiance over the sampling density. For a light with a radius, `radiance * pdf` is the emitter
    // radiance, and `pdf` is that density in solid angle for the MIS weight of a BSDF hit. A
    // zero-radius point is a delta light, with `pdf` 0.
    let offset_to_centre = light.position - point;
    let distance_squared = offset_to_centre.length_squared();
    if !distance_squared.is_finite() {
        return None;
    }
    let radius = light.radius;
    let is_sphere = radius > 0.0 && !light.soft_falloff;
    let inside_sphere = is_sphere && distance_squared <= radius * radius;
    if distance_squared <= f64::EPSILON && !inside_sphere {
        return None;
    }
    let distance_to_centre = distance_squared.sqrt();
    // Unit vectors from the receiver to the light centre, and from the light centre to the receiver.
    let toward_centre = if distance_to_centre > 0.0 {
        offset_to_centre / distance_to_centre
    } else {
        DVec3::Z
    };
    let from_centre = -toward_centre;
    if radius <= 0.0 {
        // A zero-radius point: radiant intensity P / (4π), i.e. `eval_fac = 1 / (4π)`.
        let direction = toward_centre;
        let intensity = point_intensity(light.energy);
        return Some(LightSample {
            direction,
            distance: distance_to_centre,
            radiance: light.color
                * (intensity / distance_squared * light_attenuation(light, -direction)),
            pdf: 0.0,
        });
    }
    let emitter = emitter_radiance(light.energy, radius);
    if light.soft_falloff {
        // Disk of radius `r` facing the receiver (`is_sphere = false`): a uniform disk point with
        // area pdf `1 / (π r²)`, so the weight is `eval_fac · π r² · cos θ_light / t²`.
        let (tangent, bitangent) = basis(from_centre);
        let disk_radius = radius * rng.next_f64().sqrt();
        let phi = std::f64::consts::TAU * rng.next_f64();
        let sample_point =
            light.position + (tangent * phi.cos() + bitangent * phi.sin()) * disk_radius;
        let offset = sample_point - point;
        let distance = offset.length();
        if !distance.is_finite() || distance <= f64::EPSILON {
            return None;
        }
        let direction = offset / distance;
        let cos_light = from_centre.dot(-direction);
        if cos_light <= 0.0 {
            return None;
        }
        let weight =
            emitter * std::f64::consts::PI * radius * radius * cos_light / (distance * distance);
        return Some(LightSample {
            direction,
            distance,
            radiance: light.color * (weight * light_attenuation(light, -direction)),
            pdf: distance * distance / (std::f64::consts::PI * radius * radius * cos_light),
        });
    }
    // Sphere (`is_sphere = true`).
    if distance_squared > radius * radius {
        // Uniform cone over the visible cap, pdf `1 / (2π (1 − cos θ_max))`.
        let one_minus_cos = sphere_one_minus_cos(radius * radius, distance_squared);
        let cosine = 1.0 - one_minus_cos * rng.next_f64();
        let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
        let phi = std::f64::consts::TAU * rng.next_f64();
        let (tangent, bitangent) = basis(toward_centre);
        let direction =
            toward_centre * cosine + (tangent * phi.cos() + bitangent * phi.sin()) * sine;
        // Law of cosines: the first hit with the sphere along `direction`.
        let distance = distance_to_centre * cosine
            - (radius * radius - distance_squared + distance_squared * cosine * cosine)
                .max(0.0)
                .sqrt();
        let solid_angle = 2.0 * std::f64::consts::PI * one_minus_cos;
        return Some(LightSample {
            direction,
            distance,
            radiance: light.color * (emitter * solid_angle * light_attenuation(light, -direction)),
            pdf: 1.0 / solid_angle,
        });
    }
    // Receiver inside the sphere: uniform directions (pdf `1 / 4π`), and the spot cone does not
    // attenuate, as in `spot_light_sample`'s inside branch.
    let (direction, pdf) = sample_uniform_sphere(rng);
    let cosine = direction.dot(toward_centre);
    let distance = distance_to_centre * cosine
        + (radius * radius - distance_squared + distance_squared * cosine * cosine)
            .max(0.0)
            .sqrt();
    Some(LightSample {
        direction,
        distance,
        radiance: light.color * (emitter * 4.0 * std::f64::consts::PI),
        pdf,
    })
}

/// Samples a point over an area light's region and returns its direct-light estimator.
///
/// Blender's area light has radiance `L = strength · invarea / π`, with `invarea = 1 / area` for
/// a non-normalised area (`intern/cycles/scene/light.cpp`, `AreaLight::copy_to_kernel`), and its
/// kernel evaluates `eval_fac = M_1_PI_F * invarea` (`intern/cycles/kernel/light/area.h`,
/// `area_light_eval`). The light emits from its front side only (`area_light_sample` rejects
/// points behind the plane). Uniform area sampling has pdf `1 / area`, and converting to solid
/// angle gives pdf `d² / (area · cos_light)`, so the estimator is `L · area · cos_light / d²`.
///
/// A spread draws the point from the region the cone reaches (`spread_region`) and scales `L` by
/// `spread_attenuation`, as `area_light_eval` does. The pdf is over that region, the one a
/// BSDF-sampled ray is weighted with in `light_hit`.
fn sample_area_light(light: &PointLight, point: DVec3, rng: &mut Pcg32) -> Option<LightSample> {
    let shape = light.area?;
    if shape.area <= 0.0 {
        return None;
    }
    let (centre, region) = match light.spread {
        None => (light.position, shape),
        Some(spread) if spread.is_zero() => return sample_spread_line(light, &shape, point),
        Some(spread) => spread_region(&shape, light.position, light.direction, spread, point)?,
    };
    if region.area <= 0.0 {
        return None;
    }
    let (local_u, local_v) = if region.ellipse {
        let radius = rng.next_f64().sqrt();
        let angle = rng.next_f64() * std::f64::consts::TAU;
        (radius * angle.cos(), radius * angle.sin())
    } else {
        (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0)
    };
    let target = centre
        + region.axis_u * (region.half_u * local_u)
        + region.axis_v * (region.half_v * local_v);
    // A disk region may overhang the emitter, which emits nothing outside its shape.
    if !inside_emitter(&shape, light.position, target) {
        return None;
    }
    let offset = target - point;
    let distance_squared = offset.length_squared();
    if distance_squared <= f64::EPSILON || !distance_squared.is_finite() {
        return None;
    }
    let distance = distance_squared.sqrt();
    let direction = offset / distance;
    let cos_light = light.direction.dot(-direction);
    if cos_light <= 0.0 {
        return None;
    }
    let attenuation = light.spread.map_or(1.0, |spread| {
        spread_attenuation(spread, -direction, light.direction)
    });
    if attenuation <= 0.0 {
        return None;
    }
    let emitted = light.color * (area_radiance(light.energy, shape.area) * attenuation);
    Some(LightSample {
        direction,
        distance,
        radiance: emitted * (region.area * cos_light / distance_squared),
        pdf: distance_squared / (region.area * cos_light),
    })
}

/// A zero spread emits along the normal alone, so a receiver sees the emitter only at its foot on
/// the emitter's plane, if that foot is on the emitter. Cycles samples that point with pdf 1
/// (`area_light_eval`, `tan_half_spread == 0`), so it is a delta here, as the sun is, weighted by
/// Cycles' `eval_fac` on the normal: `π` times the radiance. A Blender render of a zero spread
/// lights exactly the receivers under the emitter.
fn sample_spread_line(light: &PointLight, shape: &AreaShape, point: DVec3) -> Option<LightSample> {
    let distance = light.direction.dot(point - light.position);
    if distance <= 0.0 {
        return None;
    }
    let foot = point - distance * light.direction;
    if !inside_emitter(shape, light.position, foot) {
        return None;
    }
    Some(LightSample {
        direction: -light.direction,
        distance,
        radiance: light.color * (area_radiance(light.energy, shape.area) * std::f64::consts::PI),
        pdf: 0.0,
    })
}

/// `1 − cos θ` of the cone that a sphere of squared radius `radius_squared` subtends from a point
/// at squared distance `distance_squared`, in the stable `sin_sqr_to_one_minus_cos` form.
fn sphere_one_minus_cos(radius_squared: f64, distance_squared: f64) -> f64 {
    let sin_squared = radius_squared / distance_squared;
    // `1 - sqrt(1 - x)` cancels for tiny x.
    sin_squared / (1.0 + (1.0 - sin_squared).clamp(0.0, 1.0).sqrt())
}

/// The spot cone's factor for `light` towards `to_point` (the light-to-receiver direction). Every
/// other light type has no angular falloff.
fn light_attenuation(light: &PointLight, to_point: DVec3) -> f64 {
    if light.kind == LightType::Spot {
        spot_attenuation(
            light.direction.dot(to_point),
            light.spot_size,
            light.spot_blend,
        )
    } else {
        1.0
    }
}

/// The nearest crossing of a ray with a hittable light in `(RAY_EPSILON, limit)`. `receiver` is the
/// vertex that light sampling starts from: it fixes the disk orientation, the sphere's inside and
/// outside cases, and the sampling density the MIS weight needs. A light the path has just crossed
/// is never found again: its crossing lies behind the advanced origin, so its distance is negative.
fn nearest_light_hit(
    lights: &[PointLight],
    origin: DVec3,
    direction: DVec3,
    receiver: DVec3,
    limit: f64,
    ray: RayKind,
) -> Option<LightHit> {
    let mut nearest: Option<LightHit> = None;
    for light in lights.iter().filter(|light| light.intersects(ray)) {
        let closest = nearest.map_or(limit, |hit| hit.distance);
        if let Some((distance, radiance, pdf)) =
            light_hit(light, origin, direction, receiver, closest)
        {
            nearest = Some(LightHit {
                distance,
                radiance,
                pdf,
                visible: light.ray_mask & ray.bit() != 0,
            });
        }
    }
    nearest
}

/// Intersects a ray with a plane and returns the distance and world-space hit point.
/// One-light plane crossing shared by path-tracer emitters and EEVEE's camera-shape display.
fn ray_plane_intersection(
    origin: DVec3,
    direction: DVec3,
    center: DVec3,
    normal: DVec3,
    cos_light: f64,
    limit: f64,
) -> Option<(f64, DVec3)> {
    if cos_light.abs() <= f64::MIN_POSITIVE {
        return None;
    }
    let distance = (origin - center).dot(normal) / cos_light;
    if !(distance > RAY_EPSILON && distance < limit) {
        return None;
    }
    Some((distance, origin + direction * distance))
}

/// Intersects a ray with a hittable light, as Cycles' `area_light_intersect`, `point_light_intersect`
/// and `spot_light_intersect`. Returns distance, emitter radiance, and solid-angle density from `receiver`.
/// Area and soft-falloff disks are one-sided; a hard-falloff sphere emits from every side.
fn light_hit(
    light: &PointLight,
    origin: DVec3,
    direction: DVec3,
    receiver: DVec3,
    limit: f64,
) -> Option<(f64, DVec3, f64)> {
    if let Some(shape) = light.area {
        let cos_light = -light.direction.dot(direction);
        if cos_light <= 0.0 {
            return None;
        }
        let (distance, hit) = ray_plane_intersection(
            origin,
            direction,
            light.position,
            light.direction,
            cos_light,
            limit,
        )?;
        if !inside_emitter(&shape, light.position, hit) {
            return None;
        }
        // The region that light sampling draws from for `receiver`, and the spread factor towards it.
        let (region_area, attenuation) = match light.spread {
            None => (shape.area, 1.0),
            Some(spread) => {
                match spread_region(&shape, light.position, light.direction, spread, receiver) {
                    Some((_, region)) => (
                        region.area,
                        spread_attenuation(
                            spread,
                            (receiver - hit).normalize_or_zero(),
                            light.direction,
                        ),
                    ),
                    None => (0.0, 0.0),
                }
            }
        };
        if region_area <= 0.0 || attenuation <= 0.0 {
            // The cone does not reach the receiver, so the crossing emits nothing. It is still a
            // pass: Cycles finds an emitter by its shape, not by what it emits.
            return Some((distance, DVec3::ZERO, 0.0));
        }
        let radiance = light.color * (area_radiance(light.energy, shape.area) * attenuation);
        // The sampling density is per solid angle from the receiver, not from the advanced origin.
        let pdf = (hit - receiver).length_squared() / (region_area * cos_light);
        return Some((distance, radiance, pdf));
    }
    let radius = light.radius;
    let radius_squared = radius * radius;
    let offset_to_receiver = receiver - light.position;
    let distance_squared = offset_to_receiver.length_squared();
    let emitted = emitter_radiance(light.energy, radius);
    if light.soft_falloff {
        // The disk faces the receiver, as `point_light_sample`'s disk branch.
        let normal = offset_to_receiver.normalize_or_zero();
        let cos_light = -normal.dot(direction);
        if cos_light <= 0.0 {
            return None;
        }
        let (distance, hit) =
            ray_plane_intersection(origin, direction, light.position, normal, cos_light, limit)?;
        if (hit - light.position).length_squared() > radius_squared {
            return None;
        }
        let radiance = light.color * (emitted * light_attenuation(light, -direction));
        let pdf = (origin + direction * distance - receiver).length_squared()
            / (std::f64::consts::PI * radius_squared * cos_light);
        return Some((distance, radiance, pdf));
    }
    // The sphere: its first crossing from outside, or its far crossing from inside.
    let offset = origin - light.position;
    let half_b = offset.dot(direction);
    let discriminant = half_b * half_b - (offset.length_squared() - radius_squared);
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let inside = distance_squared <= radius_squared;
    let distance = if inside {
        -half_b + root
    } else {
        -half_b - root
    };
    if !(distance > RAY_EPSILON && distance < limit) {
        return None;
    }
    if inside {
        // Uniform directions, and no spot attenuation, as `sample_light`'s inside branch.
        return Some((
            distance,
            light.color * emitted,
            1.0 / (4.0 * std::f64::consts::PI),
        ));
    }
    let solid_angle =
        2.0 * std::f64::consts::PI * sphere_one_minus_cos(radius_squared, distance_squared);
    let radiance = light.color * (emitted * light_attenuation(light, -direction));
    Some((distance, radiance, 1.0 / solid_angle))
}

/// Surface vertex context kept for MIS at the next environment miss or light hit. `bsdf_pdf` is the
/// density of the continuous lobe mixture that generated the direction, and `point` is the vertex
/// that light sampling would have used.
#[derive(Clone, Copy)]
struct MisVertex {
    bsdf_pdf: f64,
    normal: DVec3,
    point: DVec3,
}

/// What scattered the path into its current segment, for the environment and light weights.
#[derive(Clone, Copy)]
enum Scatter {
    /// The camera ray. A light it meets counts only when the light is visible to camera rays.
    Camera,
    /// A delta (sharp glass reflection or refraction) bounce at `point`: light sampling cannot reach it, so a
    /// light hit is unweighted. The light is evaluated from `point`, as Cycles uses the segment's
    /// `ray_P`.
    Delta(DVec3),
    /// A continuous surface bounce, weighted against light sampling by its lobe-mixture density.
    Surface(MisVertex),
    /// A volume phase bounce. `phase_pdf` is the Henyey-Greenstein density of the direction it
    /// scattered into, for the MIS weight of an environment hit (`direct_volume_lighting` weighs the
    /// same direction against the environment sampling density). Lights are light-sampled only.
    Volume(f64),
}

/// A direction drawn from one closure's own lobes, before the closure set weights it. `Continuous` is a lobe-mixture
/// draw of that closure, carrying the ray kind of its lobe (`PATH_RAY_*`). `Delta` is a singular glass direction with its
/// weight over its branch probability and its ray kind. `Subsurface` enters through the closure's subsurface lobe, with
/// its weight over that branch probability. `Absorbed` is a draw Cycles rejects (`LABEL_NONE`).
enum LobeDraw {
    Continuous {
        direction: DVec3,
        ray: RayKind,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
        ray: RayKind,
        /// The lobe mixture's pdf of the draw, Cycles' `bsdf_pdf` of a singular closure: its `1e6` weight plus the
        /// other lobes' densities at `direction`. It lowers `min_ray_pdf`.
        pdf: f64,
        /// Cycles' singular pdf and sample weight for the widening average, `1e6 · branch · share`.
        mis_weight: f64,
    },
    Subsurface {
        weight: DVec3,
        method: SubsurfaceMethod,
    },
    Absorbed,
}

/// Draws a direction from one closure's lobes, as Cycles' `bsdf_sample` draws its closure: the branch by the closure's
/// own lobe probabilities, then that lobe's sampler. Only the closure set weights the result.
fn sample_lobes(
    params: &BsdfParams,
    layers: &Layers,
    geometric: DVec3,
    rng: &mut Pcg32,
) -> LobeDraw {
    let normal = layers.normal;
    let view = layers.view;
    let probabilities = lobe_probabilities(params, layers);
    let choice = rng.next_f64();
    // Thin Wall replaces a BSSRDF with diffuse reflection and cosine-sampled translucent transmission.
    if choice < probabilities.subsurface {
        if params.thin_wall {
            let direction = sample_cosine_hemisphere(-normal, rng).0;
            if direction.dot(geometric) >= 0.0 {
                return LobeDraw::Absorbed;
            }
            return LobeDraw::Continuous {
                direction,
                ray: RayKind::Diffuse,
            };
        }
        let method = params.subsurface_method;
        let weight =
            subsurface_closure_weight(params, layers.energy_roughness, normal, view, layers.lower)
                / probabilities.subsurface.max(1.0e-8);
        return LobeDraw::Subsurface { weight, method };
    }
    let transmission_end = probabilities.subsurface + probabilities.transmission;
    if choice < transmission_end {
        let rands = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
        return match sample_glass(params, layers, geometric, probabilities.transmission, rands) {
            GlassSample::Absorbed => LobeDraw::Absorbed,
            GlassSample::Singular {
                direction,
                weight,
                mis_weight,
            } => LobeDraw::Delta {
                direction,
                weight,
                ray: glass_ray(normal, direction),
                pdf: mis_weight + continuous_density(params, layers, direction),
                mis_weight,
            },
            GlassSample::Rough { direction } => LobeDraw::Continuous {
                direction,
                ray: glass_ray(normal, direction),
            },
        };
    }
    let diffuse_end = transmission_end + probabilities.diffuse;
    let sheen_end = diffuse_end + probabilities.sheen;
    let coat_end = sheen_end + probabilities.coat;

    // Cycles' sheen samples are `LABEL_REFLECT | LABEL_DIFFUSE` (`bsdf_sheen.h`), so they set the diffuse ancestor.
    let (direction, ray) = if choice < diffuse_end {
        (sample_cosine_hemisphere(normal, rng).0, RayKind::Diffuse)
    } else if choice < sheen_end {
        let first = rng.next_f64();
        let second = rng.next_f64();
        (layers.sample_sheen(first, second), RayKind::Diffuse)
    } else if choice < coat_end {
        let half_vector = sample_ggx_half(layers.coat_normal, params.coat_roughness, rng);
        (
            reflect(-view, half_vector).normalize_or_zero(),
            RayKind::Glossy,
        )
    } else {
        let half_vector = sample_specular_ggx_half(
            layers.specular_microfacet,
            layers.specular,
            view,
            rng.next_f64(),
            rng.next_f64(),
        );
        (
            reflect(-view, half_vector).normalize_or_zero(),
            RayKind::Glossy,
        )
    };
    // Cycles keeps a reflection sample only above the geometric normal: `bsdf_diffuse.h:65-71` zeroes it otherwise, and
    // the microfacet sampler rejects the same case (`bsdf_microfacet.h:863`). The check is in sampling only; the
    // evaluation keeps the lobe, and the shadow ray already blocks the direct term below the surface.
    if direction.dot(geometric) <= 0.0 {
        return LobeDraw::Absorbed;
    }
    LobeDraw::Continuous { direction, ray }
}

/// The ray kind of a glass draw, by the side of the surface it leaves to. A transmitted draw is a transmission; a
/// reflection, including the total internal reflection that `sample_glass` returns as a singular reflection, is a glossy
/// draw (`integrator/path_state.h`, singular labels take the glossy kind).
fn glass_ray(normal: DVec3, direction: DVec3) -> RayKind {
    if normal.dot(direction) < 0.0 {
        RayKind::Transmission
    } else {
        RayKind::Glossy
    }
}

/// The exit of a subsurface entry, by the method the entry sampled (Cycles' `subsurface_bounce` and
/// `subsurface_scatter`). `entry` is the closure whose subsurface lobe was sampled, and `throughput` the path throughput
/// with the entry weight already applied.
fn subsurface_exit(
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    surface: &Surface,
    entry: &HitClosure,
    method: SubsurfaceMethod,
    throughput: DVec3,
    rng: &mut Pcg32,
) -> Option<SubsurfaceExit> {
    match method {
        SubsurfaceMethod::Burley => burley_exit(geometry, bvh, surface, entry, throughput, rng),
        SubsurfaceMethod::RandomWalk => random_walk_exit(
            geometry,
            bvh,
            surface,
            entry,
            WalkMethod::RandomWalk,
            throughput,
            rng,
        ),
        SubsurfaceMethod::RandomWalkSkin => random_walk_exit(
            geometry,
            bvh,
            surface,
            entry,
            WalkMethod::Skin,
            throughput,
            rng,
        ),
        SubsurfaceMethod::RandomWalkLegacy => random_walk_exit(
            geometry,
            bvh,
            surface,
            entry,
            WalkMethod::Legacy,
            throughput,
            rng,
        ),
    }
}

/// Cycles' random walk from an entry (`subsurface_entry_bounce`, `subsurface_random_walk`). The walk starts exactly
/// at the surface point, as `subsurface_bounce` sets `ray.P = sd->P`, and on its first bounce it does not hit the
/// triangle it entered through (`ray.self.prim`). It ends at the object's surface, shaded as the diffuse exit. The
/// entry is the refraction through an interface of the surface's roughness squared, or for the skin a diffuse lobe
/// or a refraction through a rough interface (`skin_entry_direction`).
fn random_walk_exit(
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    surface: &Surface,
    closure: &HitClosure,
    method: WalkMethod,
    throughput: DVec3,
    rng: &mut Pcg32,
) -> Option<SubsurfaceExit> {
    let params = &closure.params;
    let (u, v) = (rng.next_f64(), rng.next_f64());
    // The entry's GGX interface has the clamped roughness squared (`bssrdf->alpha = sqr(roughness)`), and refracts at the
    // specular layer's eta (`bssrdf->ior = eta`, `svm/closure.h`). The skin refracts at its Subsurface IOR instead.
    let roughness = params.roughness.clamp(0.0, 1.0);
    let direction = match method {
        WalkMethod::RandomWalk | WalkMethod::Legacy => entry_direction(
            closure.layers.specular,
            closure.layers.view,
            roughness * roughness,
            specular_eta(params),
            u,
            v,
        ),
        WalkMethod::Skin => skin_entry_direction(
            closure.layers.specular,
            closure.layers.view,
            params.subsurface_ior,
            u,
            v,
        ),
    }?;
    if surface.geometric_normal.dot(direction) >= 0.0 {
        return None;
    }
    let albedo = base_color(params);
    let anisotropy = walk_anisotropy(method, params.subsurface_anisotropy);
    // A channel below the minimum radius has no closure (`bssrdf_setup`), so its radius is zero.
    let mask = channel_mask(raw_radius(params));
    let radius = walk_radius(params, albedo, method);
    let radius = DVec3::from_array(std::array::from_fn(|channel| {
        if mask[channel] { radius[channel] } else { 0.0 }
    }));
    let entry = WalkEntry {
        point: surface.point,
        normal: closure.layers.specular,
        direction,
        albedo,
        radius,
        anisotropy,
        van_de_hulst: walk_uses_van_de_hulst(method, anisotropy),
        entry_triangle: surface.triangle_index,
        throughput,
        epsilon: RAY_EPSILON,
    };
    let mut draw = || rng.next_f64();
    walk(
        geometry.triangles,
        bvh,
        surface.object_index,
        &entry,
        &mut draw,
    )
}

/// The geometric normal of a triangle from its winding, as Cycles' `Ng` is: the cross product of its edges, outward
/// for a mesh wound outward. It is not flipped towards any ray, so a subsurface exit can tell its side of the surface.
fn intrinsic_normal(geometry: &TraceGeometry<'_>, triangle_index: usize) -> Option<DVec3> {
    let triangle = geometry.triangles.get(triangle_index)?;
    Some(
        (triangle.positions[1] - triangle.positions[0])
            .cross(triangle.positions[2] - triangle.positions[0])
            .normalize_or_zero(),
    )
}

/// Probes the object of `surface` for the exit of the Burley profile, as Cycles' `subsurface_disk` does: a disk
/// sample of the profile on one of the three axes of the surface, a ray through the object along that axis, and an
/// importance resample among the hits of the ray on the object. At most `MAX_HITS` hits are kept, by Cycles'
/// reservoir (`subsurface_disk.h`, `scene_intersect_local` with `max_hits`), and the weight of a probe that saw more
/// is scaled by `num_hits / max_hits`. `throughput` includes the entry weight.
fn burley_exit(
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    surface: &Surface,
    entry: &HitClosure,
    throughput: DVec3,
    rng: &mut Pcg32,
) -> Option<SubsurfaceExit> {
    let radius = burley_radius(&entry.params);
    let mask = channel_mask(raw_radius(&entry.params));
    // One draw picks the channel and, remapped, the radius (`bssrdf_sample`).
    let (channel, xi_radius) = pick_channel(mask, rng.next_f64())?;
    let mean_free_path = radius.to_array()[channel];
    let (disk_radius, height) = sample_radius(mean_free_path, xi_radius);
    // The second draw picks the disk's axis and phase (`subsurface_disk`).
    let axis_draw = rng.next_f64();
    let normal = surface.geometric_normal;
    // Cycles' `make_orthonormals(disk_N, &disk_T, &disk_B)`: the tangent and bitangent the T and B probes travel along.
    // Their hit distribution is not symmetric about the normal, so the basis changes the exit weights.
    let (tangent, bitangent) = make_orthonormals(normal);
    // The normal axis is probed with half the probability, the others with a quarter each.
    let (disk_normal, disk_tangent, disk_bitangent, pick_pdf, phase_draw) = if axis_draw < 0.5 {
        (
            normal,
            tangent,
            bitangent,
            [0.5, 0.25, 0.25],
            axis_draw * 2.0,
        )
    } else if axis_draw < 0.75 {
        (
            tangent,
            normal,
            bitangent,
            [0.25, 0.5, 0.25],
            (axis_draw - 0.5) * 4.0,
        )
    } else {
        (
            bitangent,
            tangent,
            normal,
            [0.25, 0.25, 0.5],
            (axis_draw - 0.75) * 4.0,
        )
    };
    let phase = std::f64::consts::TAU * phase_draw;
    let disk_point =
        (disk_radius * phase.cos()) * disk_tangent + (disk_radius * phase.sin()) * disk_bitangent;
    let origin = surface.point + disk_normal * height + disk_point;
    let direction = -disk_normal;
    // Cycles records at most `MAX_HITS` hits of the object as a reservoir (`scene_intersect_local`, `kernel/bvh/local.h`),
    // with a generator seeded from the path's random numbers (`lcg_state_init` in `subsurface_disk`).
    let mut probe = ProbeHits::new(rng.next_u32());
    bvh.walk_hits(
        geometry.triangles,
        WalkRay {
            origin,
            direction,
            epsilon: RAY_EPSILON,
            remaining: 2.0 * height,
            kind: None,
        },
        |hit, from| {
            if geometry
                .triangles
                .get(hit.triangle_index)
                .is_some_and(|triangle| triangle.object_index == surface.object_index)
            {
                // The hit's distance along the probe: the key of Cycles' duplicate test and of its sort.
                let t = (from - origin).dot(direction) + hit.distance;
                probe.record(t, (hit, from));
            }
            true
        },
    );
    let (seen, mut found) = probe.finish();
    // The recorded hits are sorted by distance (`sort_intersections_and_normals`), so the resample has a fixed order.
    // The empty slots of the fixed array sort last.
    found.sort_by(|a, b| match (a, b) {
        (Some(a), Some(b)) => a.0.total_cmp(&b.0),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    // Hits the reservoir replaced still count: `w *= num_hits / max_hits` (`subsurface_disk`).
    let reservoir_scale = if seen > MAX_HITS {
        seen as f64 / MAX_HITS as f64
    } else {
        1.0
    };
    let disk_pdf = pdf_mean(radius, mask, disk_radius);
    if disk_pdf <= 0.0 || found.iter().all(Option::is_none) {
        return None;
    }
    let mut weighted: [Option<(BvhHit, DVec3, DVec3)>; MAX_HITS] = [None; MAX_HITS];
    let mut count = 0;
    let mut total = 0.0;
    for (_, (hit, from)) in found.into_iter().flatten() {
        let Some(ng) = intrinsic_normal(geometry, hit.triangle_index) else {
            continue;
        };
        let point = from + direction * hit.distance;
        let pdf_n = pick_pdf[0] * disk_normal.dot(ng).abs();
        let pdf_t = pick_pdf[1] * disk_tangent.dot(ng).abs();
        let pdf_b = pick_pdf[2] * disk_bitangent.dot(ng).abs();
        // The power-heuristic weight of the normal axis over the three axes (`subsurface_disk`).
        let denominator = pdf_n * pdf_n + pdf_t * pdf_t + pdf_b * pdf_b;
        let mis = if denominator > 0.0 {
            pdf_n / denominator
        } else {
            0.0
        };
        let distance = (point - surface.point).length();
        let weight = burley_eval(radius, mask, distance) / disk_pdf * mis * reservoir_scale;
        total += weight.abs().element_sum() / 3.0;
        weighted[count] = Some((hit, from, weight));
        count += 1;
    }
    if total <= 0.0 {
        return None;
    }
    // Importance resampling: one hit, chosen in proportion to its mean weight.
    let target = rng.next_f64() * total;
    let mut partial = 0.0;
    for (hit, from, weight) in weighted.into_iter().take(count).flatten() {
        let average = weight.abs().element_sum() / 3.0;
        partial += average;
        if average > 0.0 && target < partial {
            return Some(SubsurfaceExit {
                origin: from,
                direction,
                hit,
                throughput: throughput * (weight * (total / average)),
            });
        }
    }
    None
}

/// Density with which a direction is light-sampled at a surface vertex. The cosine strategy is
/// mixed 50/50 with the importance strategy when a world sampler exists, exactly as
/// `direct_lighting` draws.
fn environment_light_pdf(
    normal: DVec3,
    direction: DVec3,
    importance: Option<&WorldImportanceSampler>,
) -> f64 {
    let cosine_pdf = normal.dot(direction).max(0.0) / std::f64::consts::PI;
    importance.map_or(cosine_pdf, |sampler| {
        0.5 * cosine_pdf + 0.5 * sampler.pdf(direction)
    })
}

fn sample_ggx_half(normal: DVec3, roughness: f64, rng: &mut Pcg32) -> DVec3 {
    let alpha = roughness.clamp(0.02, 1.0).powi(2);
    let alpha_squared = alpha * alpha;
    let first = rng.next_f64();
    let second = rng.next_f64();
    let phi = std::f64::consts::TAU * first;
    let cosine = ((1.0 - second) / (1.0 + (alpha_squared - 1.0) * second)).sqrt();
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    let (tangent, bitangent) = basis(normal);
    (tangent * (sine * phi.cos()) + bitangent * (sine * phi.sin()) + normal * cosine)
        .normalize_or_zero()
}

fn sample_cosine_hemisphere(normal: DVec3, rng: &mut Pcg32) -> (DVec3, f64) {
    let radius = rng.next_f64().sqrt();
    let angle = std::f64::consts::TAU * rng.next_f64();
    let local_x = radius * angle.cos();
    let local_y = radius * angle.sin();
    let local_z = (1.0 - radius * radius).max(0.0).sqrt();
    let (tangent, bitangent) = basis(normal);
    let direction =
        (tangent * local_x + bitangent * local_y + normal * local_z).normalize_or_zero();
    (
        direction,
        normal.dot(direction).max(0.0) / std::f64::consts::PI,
    )
}

fn basis(normal: DVec3) -> (DVec3, DVec3) {
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent).normalize_or_zero();
    (tangent, bitangent)
}

fn reflect(direction: DVec3, normal: DVec3) -> DVec3 {
    direction - 2.0 * direction.dot(normal) * normal
}

fn collect_lights(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<PointLight> {
    collect_lights_impl(doc, snapshot, None)
}

fn collect_lights_with_camera_data(
    doc: &SceneDoc,
    snapshot: &Snapshot,
) -> (Vec<PointLight>, Vec<CameraLightPrepared>) {
    let mut camera_light_prepared = Vec::new();
    let lights = collect_lights_impl(doc, snapshot, Some(&mut camera_light_prepared));
    (lights, camera_light_prepared)
}

fn collect_lights_impl(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    mut camera_light_prepared: Option<&mut Vec<CameraLightPrepared>>,
) -> Vec<PointLight> {
    // EEVEE's light threshold sets the volume influence radius of a local light; the path tracer does not read it.
    let threshold = doc
        .scenes
        .get(&snapshot.scene_id)
        .map_or(0.01, |scene| scene.render.eevee_light_threshold);
    doc.nodes
        .iter()
        .filter(|(node_id, node)| {
            node.kind == "light"
                && snapshot.node_is_visible(node_id, node)
                && crate::exchange::evaluated_visibility(snapshot, node_id, node).1
        })
        .filter_map(|(node_id, node)| {
            let evaluated = snapshot.nodes.get(node_id)?;
            let data = evaluated.evaluated_light.as_ref().or_else(|| {
                node.data
                    .as_ref()
                    .and_then(|data_id| doc.data_blocks.get(data_id))
                    .and_then(|data| data.light.as_ref())
            })?;
            let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
            let color = DVec3::from_array(data.color).max(DVec3::ZERO);
            let energy = data.energy.max(0.0);
            let area = (data.light_type == LightType::Area).then(|| area_shape(data, &matrix));
            let volume_radius_squared = if data.light_type == LightType::Sun {
                f64::INFINITY
            } else {
                local_light_influence_radius_squared(data, data.volume_factor, threshold)
            };
            let local_influence_radius_squared = if data.light_type == LightType::Sun {
                0.0
            } else {
                local_light_influence_radius_squared(
                    data,
                    data.diffuse_factor.max(data.specular_factor),
                    threshold,
                )
                .max(volume_radius_squared)
            };
            let camera_shape_eligible = if data.light_type == LightType::Sun {
                true
            } else {
                // EEVEE's culler compares the product of half-extents (`eevee_light.cc:219-235`).
                let area_shape_has_power_area =
                    area.is_none_or(|shape| shape.half_u * shape.half_v >= 1.0e-5);
                area_shape_has_power_area && local_influence_radius_squared > 0.0
            };
            if let Some(prepared) = camera_light_prepared.as_mut() {
                prepared.push(CameraLightPrepared {
                    influence_radius_max: if data.light_type == LightType::Sun {
                        0.0
                    } else {
                        local_influence_radius_squared.max(0.0).sqrt()
                    },
                    area_display: camera_shape_eligible
                        .then(|| area.map(camera_light_display_area))
                        .flatten(),
                });
            }
            Some(PointLight {
                position: matrix.transform_point3(DVec3::ZERO),
                direction: matrix.transform_vector3(-DVec3::Z).normalize_or_zero(),
                world_transform: matrix,
                kind: data.light_type,
                color,
                energy,
                radius: data.radius.max(0.0),
                soft_falloff: data.soft_falloff,
                spot_size: data.spot_size,
                spot_blend: data.spot_blend,
                area,
                spread: Spread::new(data.spread),
                use_mis: data.use_multiple_importance_sampling,
                volume_factor: data.volume_factor.max(0.0),
                volume_radius_squared,
                ray_mask: node.resolved_ray_visibility().mask(),
                camera_shape_eligible,
            })
        })
        .collect()
}

/// The bounds of the index box `[low, high]` through an affine index-to-local matrix: the box of its eight corners. An
/// affine transform of an axis-aligned box is a box only up to its corners, so the bounds take the extremes of all eight.
fn index_box_bounds(index_to_local: DMat4, low: DVec3, high: DVec3) -> (DVec3, DVec3) {
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    for corner in 0..8 {
        let index = DVec3::new(
            if corner & 1 == 0 { low.x } else { high.x },
            if corner & 2 == 0 { low.y } else { high.y },
            if corner & 4 == 0 { low.z } else { high.z },
        );
        let point = index_to_local.transform_point3(index);
        minimum = minimum.min(point);
        maximum = maximum.max(point);
    }
    (minimum, maximum)
}

/// The volume closures of a material's Volume output summed as one parameter set: Cycles sums the closures weighted by
/// their weight (`volume_shader_eval`). The sampling and interpolation settings are the heaviest closure's, which a single
/// closure has exactly. The phase is not summed; each closure keeps its own lobe (`scatter_lobes`). `None` when the output
/// has no closure.
fn volume_medium(volume: &[ShaderClosure]) -> Option<BsdfParams> {
    let dominant = heaviest(volume, |closure| closure.weight)?;
    let mut medium = dominant.params;
    // The phase is per closure (`scatter_lobes`), so the summed medium has no anisotropy of its own.
    medium.volume_anisotropy = 0.0;
    medium.volume_scatter = weighted_volume_channels(volume, |params| params.volume_scatter);
    medium.volume_absorption = weighted_volume_channels(volume, |params| params.volume_absorption);
    medium.volume_emission = weighted_volume_channels(volume, |params| params.volume_emission);
    Some(medium)
}

/// The phase lobes of a voxel volume's Volume closures: each closure's share of the volume's scatter, its weight times its
/// mean scatter over the total, with its own anisotropy. A volume without closures scatters isotropically.
fn scatter_lobes(volume: &[ShaderClosure]) -> Vec<ScatterLobe> {
    if volume.is_empty() {
        return vec![ScatterLobe {
            share: 1.0,
            anisotropy: 0.0,
        }];
    }
    let scatter: Vec<f64> = volume
        .iter()
        .map(|closure| closure.weight * mean_scatter(&closure.params))
        .collect();
    let total: f64 = scatter.iter().sum();
    volume
        .iter()
        .zip(scatter)
        .map(|(closure, scatter)| ScatterLobe {
            share: if total > 0.0 { scatter / total } else { 0.0 },
            anisotropy: closure.params.volume_anisotropy,
        })
        .collect()
}

/// `Σ weight · channel` over the volume closures, per colour channel.
fn weighted_volume_channels(
    volume: &[ShaderClosure],
    channel: impl Fn(&BsdfParams) -> [f64; 3],
) -> [f64; 3] {
    let mut total = [0.0; 3];
    for closure in volume {
        for (total, value) in total.iter_mut().zip(channel(&closure.params)) {
            *total += closure.weight * value;
        }
    }
    total
}

fn collect_volumes<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    path_traced: bool,
    meters_per_unit: f64,
    images: &BTreeMap<String, ImageData>,
    warnings: &RefCell<Vec<serde_json::Value>>,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<(Vec<VolumeInstance<'a>>, Vec<Option<VolumeTemperature>>)> {
    let mut volumes = Vec::new();
    let mut volume_temperatures = Vec::new();
    let mut evaluator_workspace = EvaluatorWorkspace::default();
    for (node_id, node) in &doc.nodes {
        if !snapshot.node_is_visible(node_id, node)
            || !crate::exchange::evaluated_visibility(snapshot, node_id, node).1
        {
            continue;
        }
        let Some(data_id) = node.data.as_ref() else {
            continue;
        };
        let Some(data_block) = doc.data_blocks.get(data_id) else {
            continue;
        };
        let Some(volume) = snapshot
            .volume_data
            .get(data_id)
            .or(data_block.volume.as_ref())
        else {
            continue;
        };
        let Some(evaluated) = snapshot.nodes.get(node_id) else {
            continue;
        };
        let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
        let (bounds_min, bounds_max, voxel_size, density_range) =
            if let Some(decoded) = volume.decoded_vdb.as_ref() {
                let Some(grid) = decoded.density_grid() else {
                    continue;
                };
                let Some((bounds_min, bounds_max)) = grid.active_bbox_world_affine() else {
                    continue;
                };
                (
                    bounds_min,
                    bounds_max,
                    grid.transform.affine_voxel_size().min_element(),
                    None,
                )
            } else {
                let Some(grid) = volume.grids.first() else {
                    continue;
                };
                let index_to_local = grid.checked_index_to_local_matrix()?;
                let last = DVec3::from_array(
                    grid.dims
                        .map(|dimension| f64::from(dimension.saturating_sub(1))),
                );
                let (bounds_min, bounds_max) = index_box_bounds(index_to_local, DVec3::ZERO, last);
                let density_range = grid.values.as_ref().and_then(|values| {
                    let mut values = values.iter().copied();
                    let first = values.next()?;
                    if !first.is_finite() {
                        return None;
                    }
                    let mut minimum = first;
                    let mut maximum = first;
                    for value in values {
                        if !value.is_finite() {
                            return None;
                        }
                        minimum = minimum.min(value);
                        maximum = maximum.max(value);
                    }
                    Some((f64::from(minimum), f64::from(maximum)))
                });
                (bounds_min, bounds_max, grid.voxel_size, density_range)
            };
        let material = snapshot
            .slot_material(node_id, node, Some(data_block), 0)
            .and_then(|material_id| doc.materials.get(material_id));
        if let Some(material) = material {
            ensure_supported_volume_graph(material, &doc.node_groups, volume)?;
        }
        let prepared = material.map(|material| PreparedMaterial::new(material, &doc.node_groups));
        let mut temperature_captures = Vec::new();
        let evaluated_material = if let (Some(material), Some(prepared)) =
            (material, prepared.as_ref())
        {
            let local_center = (bounds_min + bounds_max) * 0.5;
            let temperature_capture = RefCell::new(Vec::new());
            let mut context = HitContext::new(&doc.node_groups)
                .with_images(images)
                .with_volume_data(volume)
                .with_volume_density_grid()
                .with_volume_temperature_capture(&temperature_capture)
                .with_warnings(warnings)
                .with_path_traced(path_traced);
            context.object = local_center;
            context.object_to_world = matrix;
            let world_to_object = matrix.inverse();
            context.world_to_object = if world_to_object
                .to_cols_array()
                .iter()
                .all(|value| value.is_finite())
            {
                world_to_object
            } else {
                DMat4::IDENTITY
            };
            context.object_color = object_info_color(node);
            context.object_pass_index = object_info_pass_index(node);
            context.object_random_number = snapshot
                .instance_paths
                .get(node_id)
                .and_then(|paths| paths.first())
                .is_none_or(Vec::is_empty)
                .then(|| f64::from(crate::blender_rng::cycles_object_random_number(&node.name)));
            context.material_pass_index = material.pass_index.map(f64::from);
            context.path_traced = path_traced;
            context.light_falloff_distance = path_traced.then_some(0.0);
            // Cycles maps a volume's Generated as `(P - min) * safe_divide(1, size)` (`blender/volume.cpp`), so a
            // zero-thickness axis maps to 0, where a mesh's degenerate axis maps to 0.5.
            let size = bounds_max - bounds_min;
            let inverse_size = DVec3::from_array(size.to_array().map(|extent| {
                if crate::float::equal_f64(extent, 0.0) {
                    0.0
                } else {
                    1.0 / extent
                }
            }));
            context.generated = (local_center - bounds_min) * inverse_size;
            context.generated_per_local = inverse_size;
            context.position = matrix.transform_point3(local_center);
            let shading = evaluate_surface_with_workspace_semantics(
                material,
                prepared,
                &context,
                &mut evaluator_workspace,
                math_semantics,
            )?;
            temperature_captures = std::mem::take(&mut *temperature_capture.borrow_mut());
            Some(shading)
        } else {
            None
        };
        let has_volume_output = prepared
            .as_ref()
            .is_some_and(PreparedMaterial::has_volume_output);
        let simple_volume = material.is_some_and(|material| {
            material.node_tree.is_none()
                && material.volume_density.is_finite()
                && material.volume_density > 0.0
        });
        let lobes = match &evaluated_material {
            Some(shading) => scatter_lobes(&shading.volume),
            None => scatter_lobes(&[]),
        };
        let params = evaluated_material
            .as_ref()
            .and_then(|shading| volume_medium(&shading.volume));
        // Cycles' absolute coefficients (`BsdfParams`): a material volume's σs and σa, or, for a
        // material without a volume output, unit grey absorption. No material is the Principled
        // Volume default (colour 0.5, density 1).
        let (scatter, absorption) = match params {
            Some(params) if has_volume_output || simple_volume => {
                (params.volume_scatter, params.volume_absorption)
            }
            Some(_) => ([0.0; 3], [1.0; 3]),
            None => ([0.5; 3], [0.5; 3]),
        };
        let extinction: [f64; 3] = std::array::from_fn(|index| scatter[index] + absorption[index]);
        let extinction_rgb_scale = DVec3::from_array(extinction);
        let albedo = DVec3::from_array(std::array::from_fn(|index| {
            if extinction[index] > 0.0 {
                scatter[index] / extinction[index]
            } else {
                0.0
            }
        }))
        .clamp(DVec3::ZERO, DVec3::ONE);
        let scattering = scatter.iter().any(|weight| *weight > 0.0);
        let emission = if has_volume_output || simple_volume {
            params.map_or(DVec3::ZERO, |params| {
                DVec3::from_array(params.volume_emission)
            })
        } else {
            DVec3::ZERO
        };
        let interpolation = params.map_or(crate::model::VolumeInterpolation::default(), |params| {
            params.volume_interpolation
        });
        let mut volume_temperature = None;
        if let Some(shading) = &evaluated_material {
            let mut active_capture = None;
            for capture in &temperature_captures {
                if !capture.attribute.is_empty()
                    && volume_attribute_is_present(&capture.attribute, volume)?
                {
                    if active_capture.is_some() {
                        return Err(volume_temperature_graph_error(node_id));
                    }
                    active_capture = Some(capture);
                }
            }
            if let Some(capture) = active_capture {
                if shading.volume.len() != 1 || temperature_captures.len() != 1 {
                    return Err(volume_temperature_graph_error(node_id));
                }
                let mut temperature = VolumeTemperature {
                    attribute: capture.attribute.clone(),
                    temperature: capture.temperature,
                    blackbody: capture.blackbody,
                    tint: capture.tint,
                    closure_weight: shading.volume[0].weight,
                    center_emission: DVec3::ZERO,
                    interpolation,
                    path_traced,
                };
                // Cycles evaluates the material closure with mapped VDB taps at the center; subtract
                // that baseline when replacing it with per-step temperature samples.
                temperature.center_emission = volume_temperature_emission_at(
                    &temperature,
                    volume,
                    (bounds_min + bounds_max) * 0.5,
                    path_traced,
                )?;
                volume_temperature = Some(temperature);
            }
        }
        // EEVEE's volume mesh bounds pad active grids by half a voxel (`BKE_volume_grid_bounds`); Cycles extends them
        // by the sampler's full support so rays do not stop before the interpolation shell.
        let pad_voxels = if path_traced {
            match interpolation {
                crate::model::VolumeInterpolation::Linear => 1.0,
                crate::model::VolumeInterpolation::Cubic => 2.0,
            }
        } else {
            0.5
        };
        let (bounds_min, bounds_max) = if let Some(decoded) = volume.decoded_vdb.as_ref() {
            let Some(grid) = decoded.density_grid() else {
                continue;
            };
            let Some(bounds) = grid.active_bbox_world_padded_affine(DVec3::splat(pad_voxels))
            else {
                continue;
            };
            bounds
        } else {
            let Some(grid) = volume.grids.first() else {
                continue;
            };
            // The index box, padded per axis, then transformed by the grid's index-to-local matrix (affine grids
            // included), as the sampler reads it.
            let last = DVec3::from_array(
                grid.dims
                    .map(|dimension| f64::from(dimension.saturating_sub(1))),
            );
            index_box_bounds(
                grid.checked_index_to_local_matrix()?,
                DVec3::splat(-pad_voxels),
                last + DVec3::splat(pad_voxels),
            )
        };
        volumes.push(VolumeInstance {
            volume,
            world_to_local: matrix.inverse(),
            bounds_min,
            bounds_max,
            voxel_size,
            meters_per_unit,
            density_range,
            albedo,
            lobes,
            emission,
            scattering,
            extinction_rgb_scale,
            sampling: params.map_or(crate::model::VolumeSampling::default(), |params| {
                params.volume_sampling
            }),
            interpolation,
            index_from_local: if volume.decoded_vdb.is_some() {
                None
            } else {
                Some(local_to_index_matrix(volume)?)
            },
            path_traced,
            ray_mask: node.resolved_ray_visibility().mask(),
        });
        volume_temperatures.push(volume_temperature);
    }
    Ok((volumes, volume_temperatures))
}
fn volume_temperature_emission_at(
    temperature: &VolumeTemperature,
    volume: &VolumeData,
    local_position: DVec3,
    map_vdb_taps: bool,
) -> Result<DVec3> {
    let mut map = |attribute: f64| {
        blackbody_emission(
            (temperature.temperature * attribute.max(0.0) as f32).max(0.0),
            temperature.blackbody,
            temperature.tint,
            temperature.path_traced,
        )
    };
    let sampled = if temperature.attribute.is_empty() {
        None
    } else if map_vdb_taps {
        crate::geom::volume::sample_scalar_attribute_mapped_checked(
            &temperature.attribute,
            local_position,
            volume,
            temperature.interpolation,
            &mut map,
        )?
    } else {
        crate::geom::volume::sample_scalar_attribute_realtime_checked(
            &temperature.attribute,
            local_position,
            volume,
            temperature.interpolation,
        )?
        .map(map)
    };
    Ok(sampled.unwrap_or_else(|| {
        if temperature.path_traced {
            blackbody_emission(
                temperature.temperature.max(0.0),
                temperature.blackbody,
                temperature.tint,
                true,
            )
        } else {
            DVec3::ZERO
        }
    }) * temperature.closure_weight)
}

fn blackbody_emission(
    temperature: f32,
    blackbody: f32,
    tint: [f32; 3],
    path_traced: bool,
) -> DVec3 {
    let temperature_squared = temperature * temperature;
    let temperature_fourth = temperature_squared * temperature_squared;
    let sigma = 5.670_373e-8_f32 * 1.0e-6_f32 / std::f32::consts::PI;
    let intensity = sigma * blackbody.mul_add(temperature_fourth - 1.0, 1.0);
    let minimum_intensity = if path_traced { 0.0 } else { 1.0e-5 };
    if intensity.partial_cmp(&minimum_intensity) != Some(std::cmp::Ordering::Greater) {
        return DVec3::ZERO;
    }
    let color = if path_traced {
        crate::shader::blackbody::blackbody_rgb(temperature)
    } else {
        crate::shader::converters::eevee_blackbody_rgb(temperature)
    };
    DVec3::new(
        f64::from(color[0] * tint[0] * intensity),
        f64::from(color[1] * tint[1] * intensity),
        f64::from(color[2] * tint[2] * intensity),
    )
}

#[cfg(test)]
mod volume_blackbody_tests {
    use super::*;

    #[test]
    fn realtime_zero_temperature_emits_nothing() {
        assert_eq!(blackbody_emission(0.0, 0.2, [1.0; 3], false), DVec3::ZERO);
    }
}

fn volume_emission_at(
    instance: &VolumeInstance<'_>,
    temperature: Option<&VolumeTemperature>,
    world_position: DVec3,
) -> Result<DVec3> {
    let Some(temperature) = temperature else {
        return Ok(instance.emission);
    };
    let local_position = instance.world_to_local.transform_point3(world_position);
    let sampled = volume_temperature_emission_at(
        temperature,
        instance.volume,
        local_position,
        temperature.path_traced,
    )?;
    Ok(instance.emission + sampled - temperature.center_emission)
}

fn volume_temperature_graph_error(node_id: &crate::model::Id) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "per-sample blackbody temperature evaluation requires a single Principled Volume closure",
        serde_json::json!({"feature_id":"shader.volume.spatially_varying_graph","node_id":node_id}),
    )
}

/// Reject volume graph sources that the volume path does not evaluate per step.
fn ensure_supported_volume_graph(
    material: &Material,
    node_groups: &crate::model::Registry<crate::graph::NodeGroup>,
    volume: &VolumeData,
) -> Result<()> {
    let Some(graph_id) = &material.node_tree else {
        return Ok(());
    };
    let Some(group) = node_groups.get(graph_id) else {
        return Ok(());
    };
    let mut active_output = None;
    let mut fallback_output = None;
    for (id, node) in &group.nodes {
        if !matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
            continue;
        }
        if node
            .properties
            .get("is_active_output")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            active_output = Some(id);
            break;
        }
        fallback_output.get_or_insert(id);
    }
    let Some(output) = active_output.or(fallback_output) else {
        return Ok(());
    };
    let mut pending: Vec<Id> = group
        .links
        .iter()
        .filter(|link| link.to_node == *output && link.to_socket == "Volume")
        .map(|link| link.from_node.clone())
        .collect();
    let mut visited = BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if !visited.insert(node_id.clone()) {
            continue;
        }
        let Some(node) = group.nodes.get(&node_id) else {
            continue;
        };
        if volume_graph_node_is_spatial(&node.node_type, group, &node_id) {
            return Err(spatial_volume_graph_error(&node_id, &node.node_type));
        }
        if matches!(
            node.node_type.as_str(),
            "ShaderNodeVolumePrincipled" | "VolumePrincipled" | "PrincipledVolume"
        ) {
            let density_active = volume_socket_is_active(group, &node_id, node, "Density", 1.0);
            if density_active {
                for (socket, default) in [("Density Attribute", "density"), ("Color Attribute", "")]
                {
                    let attribute = node
                        .inputs
                        .get(socket)
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(default);
                    let density_source_present = volume
                        .decoded_vdb
                        .as_ref()
                        .is_some_and(|decoded| decoded.density_grid().is_some())
                        || !volume.grids.is_empty();
                    let unsupported = if socket == "Density Attribute" {
                        attribute != "density" && density_source_present
                    } else {
                        !attribute.is_empty() && volume_attribute_is_present(attribute, volume)?
                    };
                    if unsupported {
                        return Err(spatial_volume_graph_error(
                            &node_id,
                            &format!(
                                "{}:{}",
                                socket.replace(' ', "_"),
                                if attribute.is_empty() {
                                    "<empty>"
                                } else {
                                    attribute
                                }
                            ),
                        ));
                    }
                }
            }
        }
        pending.extend(
            group
                .links
                .iter()
                .filter(|link| link.to_node == node_id)
                .map(|link| link.from_node.clone()),
        );
    }
    Ok(())
}

fn volume_socket_is_active(
    group: &crate::graph::NodeGroup,
    node_id: &Id,
    node: &crate::graph::GraphNode,
    socket: &str,
    default: f64,
) -> bool {
    group
        .links
        .iter()
        .any(|link| link.to_node == *node_id && link.to_socket == socket)
        || node
            .inputs
            .get(socket)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default)
            > 0.0
}

fn volume_graph_node_is_spatial(
    node_type: &str,
    group: &crate::graph::NodeGroup,
    node_id: &Id,
) -> bool {
    if matches!(
        node_type,
        "ShaderNodeTexCoord"
            | "ShaderNodeUVMap"
            | "ShaderNodeNewGeometry"
            | "ShaderNodeGeometry"
            | "Geometry"
            | "ShaderNodeAttribute"
            | "ShaderNodeVertexColor"
            | "Attribute"
            | "VertexColor"
    ) {
        return true;
    }
    if !matches!(
        node_type,
        "ShaderNodeTexImage"
            | "ShaderNodeTexEnvironment"
            | "ShaderNodeTexNoise"
            | "ShaderNodeTexVoronoi"
            | "ShaderNodeTexGradient"
            | "ShaderNodeTexChecker"
            | "ShaderNodeTexWave"
            | "ShaderNodeTexBrick"
            | "ShaderNodeTexMagic"
            | "ShaderNodeTexWhiteNoise"
            | "ShaderNodeTexGabor"
            | "ShaderNodeTexMusgrave"
            | "ShaderNodeTexPointDensity"
            | "ShaderNodeMapping"
    ) {
        return false;
    }
    !group
        .links
        .iter()
        .any(|link| link.to_node == *node_id && link.to_socket == "Vector")
}

fn spatial_volume_graph_error(node_id: &Id, source: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "volume graph source `{source}` is unsupported by Cycles path-traced per-step volume evaluation"
        ),
        serde_json::json!({"feature_id":"shader.volume.spatially_varying_graph","node_id":node_id,"source":source}),
    )
}
#[cfg(test)]
mod volume_shader_context_tests {
    use std::{cell::RefCell, collections::BTreeMap};

    use glam::DVec3;
    use serde_json::json;

    use crate::{
        eval::{EvaluationContext, EvaluationPurpose, Snapshot},
        geom::volume::{VolumeData, VolumeGrid},
        graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
        model::{DataBlock, Id, Material, Node, SceneDoc},
    };

    use super::collect_volumes;

    #[test]
    fn realtime_volume_uses_eevee_falloff_and_volume_object_info() -> crate::error::Result<()> {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let graph_id = Id::from_static("volume_shader");
        let material_id = Id::from_static("volume_material");
        let data_id = Id::from_static("volume_data");
        let object_id = Id::from_static("volume_object");
        let mut group = NodeGroup::new("Volume shader context", GraphKind::Shader);
        for (id, node_type) in [
            ("info", "ShaderNodeObjectInfo"),
            ("falloff", "ShaderNodeLightFalloff"),
            ("alpha_scatter", "ShaderNodeVolumeScatter"),
            ("object_scatter", "ShaderNodeVolumeScatter"),
            ("random_scatter", "ShaderNodeVolumeScatter"),
            ("material_scatter", "ShaderNodeVolumeScatter"),
            ("falloff_scatter", "ShaderNodeVolumeScatter"),
            ("add_alpha_object", "ShaderNodeAddShader"),
            ("add_random_material", "ShaderNodeAddShader"),
            ("add_object_info", "ShaderNodeAddShader"),
            ("add_falloff", "ShaderNodeAddShader"),
            ("output", "ShaderNodeOutputMaterial"),
        ] {
            let _ = group
                .nodes
                .insert(Id::from_static(id), GraphNode::new(node_type));
        }
        if let Some(node) = group.nodes.get_mut(&Id::from_static("falloff")) {
            let _ = node.inputs.insert("Strength".to_owned(), json!(5.0));
            let _ = node.inputs.insert("Smooth".to_owned(), json!(0.0));
        }
        if let Some(node) = group.nodes.get_mut(&Id::from_static("output")) {
            let _ = node
                .properties
                .insert("is_active_output".to_owned(), json!(true));
        }
        for (density, scatter) in [
            ("Alpha", "alpha_scatter"),
            ("Object Index", "object_scatter"),
            ("Random", "random_scatter"),
            ("Material Index", "material_scatter"),
        ] {
            group.links.push(GraphLink {
                from_node: Id::from_static("info"),
                from_socket: density.to_owned(),
                to_node: Id::from_static(scatter),
                to_socket: "Density".to_owned(),
            });
            group.links.push(GraphLink {
                from_node: Id::from_static("info"),
                from_socket: "Color".to_owned(),
                to_node: Id::from_static(scatter),
                to_socket: "Color".to_owned(),
            });
        }
        group.links.push(GraphLink {
            from_node: Id::from_static("falloff"),
            from_socket: "Linear".to_owned(),
            to_node: Id::from_static("falloff_scatter"),
            to_socket: "Density".to_owned(),
        });
        for (first, second, add) in [
            ("alpha_scatter", "object_scatter", "add_alpha_object"),
            ("random_scatter", "material_scatter", "add_random_material"),
            ("add_alpha_object", "add_random_material", "add_object_info"),
            ("add_object_info", "falloff_scatter", "add_falloff"),
        ] {
            group.links.push(GraphLink {
                from_node: Id::from_static(first),
                from_socket: "Volume".to_owned(),
                to_node: Id::from_static(add),
                to_socket: "Shader".to_owned(),
            });
            group.links.push(GraphLink {
                from_node: Id::from_static(second),
                from_socket: "Volume".to_owned(),
                to_node: Id::from_static(add),
                to_socket: "Shader_001".to_owned(),
            });
        }
        group.links.push(GraphLink {
            from_node: Id::from_static("add_falloff"),
            from_socket: "Shader".to_owned(),
            to_node: Id::from_static("output"),
            to_socket: "Volume".to_owned(),
        });
        let _ = doc.node_groups.insert(graph_id.clone(), group);

        let material = Material {
            name: "Volume object-info context".to_owned(),
            pass_index: Some(11),
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let _ = doc.materials.insert(material_id.clone(), material);

        let grid = VolumeGrid {
            dims: [2, 2, 2],
            values: Some(vec![1.0; 8]),
            ..VolumeGrid::default()
        };
        let mut volume = VolumeData::default();
        volume.grids.push(grid);
        let data = DataBlock {
            data_type: "volume".to_owned(),
            mesh: None,
            volume: Some(volume),
            ..DataBlock::default()
        };
        let _ = doc.data_blocks.insert(data_id.clone(), data);

        let mut object = Node {
            name: "Volume Info Target".to_owned(),
            kind: "volume".to_owned(),
            data: Some(data_id),
            materials: vec![Some(material_id)],
            ..Node::default()
        };
        let _ = object.properties.insert(
            "blender_object_metadata".to_owned(),
            json!({"rna_properties": {
                "color": [0.2, 0.4, 0.6, 0.7],
                "pass_index": 29
            }}),
        );
        let _ = doc.nodes.insert(object_id.clone(), object);
        if let Some(root) = doc.collections.get_mut(&Id::from_static("collection_root")) {
            root.objects.push(object_id);
        }

        let snapshot = Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                scene_id: Some(doc.active_scene.clone()),
                purpose: EvaluationPurpose::Render,
                ..EvaluationContext::default()
            },
        )?;
        let expected_random = f64::from(crate::blender_rng::cycles_object_random_number(
            "Volume Info Target",
        ));
        let object_info_density = 0.7 + 29.0 + 11.0 + expected_random;
        let images = BTreeMap::new();
        let warnings = RefCell::new(Vec::new());
        for (path_traced, falloff) in [(false, 5.0), (true, 0.0)] {
            let (volumes, _) = collect_volumes(
                &doc,
                &snapshot,
                path_traced,
                1.0,
                &images,
                &warnings,
                if path_traced {
                    crate::node_kernels::MathSemantics::Cycles
                } else {
                    crate::node_kernels::MathSemantics::Eevee
                },
            )?;
            assert_eq!(volumes.len(), 1);
            let volume = &volumes[0];
            for (channel, color) in [0.2, 0.4, 0.6].into_iter().enumerate() {
                let expected_extinction = color * object_info_density + 0.8 * falloff;
                assert!(
                    (volume.extinction_rgb_scale[channel] - expected_extinction).abs() < 1.0e-5,
                    "volume channel {channel}: {:?}",
                    volume.extinction_rgb_scale
                );
            }
            assert_eq!(volume.albedo, DVec3::ONE);
        }
        Ok(())
    }
}

fn homogeneous_segment_transmittance(extinction: f64, distance: f64) -> f64 {
    (-extinction.max(0.0) * distance.max(0.0)).exp()
}

fn homogeneous_segment_integral(extinction: f64, distance: f64) -> (f64, f64) {
    let extinction = extinction.max(0.0);
    let distance = distance.max(0.0);
    let transmittance = homogeneous_segment_transmittance(extinction, distance);
    let source_distance = if extinction > 0.0 {
        -(-extinction * distance).exp_m1() / extinction
    } else {
        distance
    };
    (transmittance, source_distance)
}

/// A closed mesh's medium on the path's volume stack, as Cycles' object volume stack: the coefficients of one volume
/// closure of the surface that entered it, keyed by the object and the closure's index. The same key leaves it.
#[derive(Clone, Copy)]
struct MeshMedium {
    object_index: u32,
    closure: usize,
    params: BsdfParams,
}

#[derive(Clone, Copy)]
struct FroxelMeshMedium {
    object_index: u32,
    material_index: usize,
    closure: usize,
    generated_origin: DVec3,
    generated_per_local: DVec3,
}

/// How a path leaves one segment of a mesh medium (`trace_mesh_medium`).
enum MediumOutcome {
    /// A real scattering event at this distance along the ray.
    Scatter(f64),
    /// The path crosses the segment, its throughput carrying the transmittance.
    Survive,
    /// The path is absorbed in the medium, or exhausts its tentative collisions.
    Absorb,
}

/// The tentative-collision budget of one medium segment. Exhausting it absorbs the path; the tail of the
/// null-collision chain that this cuts off is negligible for the densities a rendered medium has.
const MAX_MEDIUM_EVENTS: u32 = 4096;

/// Spectral ratio tracking over a medium segment of length `distance`
/// (`intern/cycles/kernel/integrator/shade_volume.h`, `volume_integrate_heterogeneous`). A medium with no
/// scattering is attenuated deterministically, as Cycles does (`Absorption only`): each channel takes
/// `exp(-σa·distance)` and no event is drawn. Otherwise tentative collisions come at the majorant `max_c σt_c`.
/// Each is a scatter, a null collision or an absorption, chosen with the channel-mean probabilities, and the
/// throughput takes each channel's own branch weight. Every channel's transmittance and single-scatter integral
/// is then exact in expectation.
fn trace_mesh_medium(
    medium: &BsdfParams,
    distance: f64,
    throughput: &mut DVec3,
    rng: &mut Pcg32,
) -> MediumOutcome {
    let scatter = DVec3::from_array(medium.volume_scatter);
    let absorption = DVec3::from_array(medium.volume_absorption);
    if scatter == DVec3::ZERO {
        // A channel with no absorption stays at one, which keeps an infinite segment from giving 0 · ∞.
        *throughput *= DVec3::from_array(absorption.to_array().map(|sigma| {
            if sigma > 0.0 {
                (-sigma * distance).exp()
            } else {
                1.0
            }
        }));
        return MediumOutcome::Survive;
    }
    let extinction = scatter + absorption;
    let majorant = extinction.max_element();
    if majorant <= 0.0 {
        return MediumOutcome::Survive;
    }
    let null = DVec3::splat(majorant) - extinction;
    // The channel means of the three branch weights sum to the majorant, so they are the branch probabilities
    // over it.
    let mean_scatter = scatter.element_sum() / 3.0;
    let mean_null = null.element_sum() / 3.0;
    let mut position = 0.0;
    for _ in 0..MAX_MEDIUM_EVENTS {
        position += -rng.next_f64().max(f64::MIN_POSITIVE).ln() / majorant;
        if position >= distance {
            // No tentative collision before the boundary: each channel carries exp(-σt·d) over the probability
            // exp(-σm·d) of that event, which is exp((σm - σt)·d).
            *throughput *=
                DVec3::from_array(null.to_array().map(|weight| (weight * distance).exp()));
            return MediumOutcome::Survive;
        }
        let choice = rng.next_f64() * majorant;
        if choice < mean_scatter {
            *throughput *= scatter / mean_scatter;
            return MediumOutcome::Scatter(position);
        }
        if choice < mean_scatter + mean_null {
            *throughput *= null / mean_null;
            continue;
        }
        *throughput = DVec3::ZERO;
        return MediumOutcome::Absorb;
    }
    *throughput = DVec3::ZERO;
    MediumOutcome::Absorb
}

/// Pushes the medium of a volume closure when the path enters it, or pops it when the path leaves it. The key is the
/// object and the closure's index, so each closure of a surface toggles its own medium.
fn toggle_mesh_medium(media: &mut Vec<MeshMedium>, entry: &MeshMedium) {
    if let Some(position) = media.iter().rposition(|medium| {
        medium.object_index == entry.object_index && medium.closure == entry.closure
    }) {
        media.remove(position);
    } else {
        media.push(*entry);
    }
}

fn toggle_froxel_mesh_medium(media: &mut Vec<FroxelMeshMedium>, entry: &FroxelMeshMedium) {
    if let Some(position) = media.iter().rposition(|medium| {
        medium.object_index == entry.object_index
            && medium.material_index == entry.material_index
            && medium.closure == entry.closure
    }) {
        media.remove(position);
    } else {
        media.push(*entry);
    }
}

/// The coefficients of the whole volume stack, summed as Cycles evaluates the media a segment is inside
/// (`volume_shader_eval` over `volume_stack_read`): scatter and absorption add up over the media, and the sampling is the
/// stack's combined method (`volume_stack_sample_method`). The phase is not combined: each medium is its own closure, as
/// `PhaseMixture` holds it, so the combined anisotropy is unused and set to zero. `None` when no medium is inside.
fn stack_medium(media: &[MeshMedium]) -> Option<BsdfParams> {
    let first = media.first()?;
    let mut combined = first.params;
    let mut scatter = DVec3::ZERO;
    let mut absorption = DVec3::ZERO;
    let mut emission = DVec3::ZERO;
    for medium in media {
        scatter += DVec3::from_array(medium.params.volume_scatter);
        absorption += DVec3::from_array(medium.params.volume_absorption);
        emission += DVec3::from_array(medium.params.volume_emission);
    }
    combined.volume_scatter = scatter.to_array();
    combined.volume_absorption = absorption.to_array();
    combined.volume_emission = emission.to_array();
    combined.volume_anisotropy = 0.0;
    combined.volume_sampling =
        segment_sampling(media.iter().map(|medium| medium.params.volume_sampling))
            .unwrap_or(first.params.volume_sampling);
    Some(combined)
}

/// The mesh media that contain `origin`, as Cycles' `volume_stack_init` seeds the stack. A ray from the origin toggles
/// each medium it crosses, so a medium's object is in the stack when the ray crosses its surfaces an odd number of
/// times, that is, when the origin is inside it. The probe direction is fixed and oblique, so it avoids axis-aligned edges.
fn camera_media(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    origin: DVec3,
    images: &BTreeMap<String, ImageData>,
    footprint: CameraFootprint,
    camera_matrices: Option<(&DMat4, &DMat4)>,
    path_traced: bool,
    inside: &mut Vec<MeshMedium>,
    hit_media: &mut Vec<MeshMedium>,
    evaluator_workspace: &mut EvaluatorWorkspace,
    mut froxel_inside: Option<&mut Vec<FroxelMeshMedium>>,
    mut froxel_hit_media: Option<&mut Vec<FroxelMeshMedium>>,
) -> Result<()> {
    let probe = DVec3::new(
        0.267_261_241_912_424_4,
        0.534_522_483_824_848_8,
        0.801_783_725_737_273_2,
    );
    inside.clear();
    if let Some(froxel_inside) = froxel_inside.as_deref_mut() {
        froxel_inside.clear();
    }
    let mut failure = None;
    bvh.walk_hits(
        geometry.triangles,
        WalkRay {
            origin,
            direction: probe,
            epsilon: RAY_EPSILON,
            remaining: f64::INFINITY,
            // Cycles builds the camera's volume stack with the camera ray's visibility (`integrator_intersect_volume_stack`).
            kind: Some(RayKind::Camera),
        },
        |hit, hit_origin| {
            match shadow_hit_into(
                doc,
                geometry,
                hit,
                hit_origin,
                probe,
                images,
                Some(footprint.at_distance(hit.distance)),
                RayKind::Camera,
                camera_matrices,
                path_traced,
                path_traced.then_some(0.0),
                hit_media,
                froxel_hit_media.as_deref_mut(),
                evaluator_workspace,
            ) {
                Ok(_) => (),
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            }
            for medium in hit_media.iter() {
                toggle_mesh_medium(inside, medium);
            }
            if let (Some(froxel_inside), Some(froxel_hits)) =
                (froxel_inside.as_deref_mut(), froxel_hit_media.as_deref())
            {
                for medium in froxel_hits {
                    toggle_froxel_mesh_medium(froxel_inside, medium);
                }
            }
            true
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

/// The channel-mean scatter and extinction of a medium: the rates of Cycles' grey majorant transport.
fn mean_scatter(medium: &BsdfParams) -> f64 {
    DVec3::from_array(medium.volume_scatter).element_sum() / 3.0
}

fn mean_extinction(medium: &BsdfParams) -> f64 {
    (DVec3::from_array(medium.volume_scatter) + DVec3::from_array(medium.volume_absorption))
        .element_sum()
        / 3.0
}

/// The distance of the medium's first collision before `limit`, sampled analogously at the channel-mean extinction
/// (`None` past the limit). A collision is a scatter or an absorption, chosen by the channel-mean scatter fraction.
fn sample_mesh_collision(medium: &BsdfParams, limit: f64, rng: &mut Pcg32) -> Option<f64> {
    let extinction = mean_extinction(medium);
    if extinction <= 0.0 {
        return None;
    }
    let distance = -rng.next_f64().max(f64::MIN_POSITIVE).ln() / extinction;
    (distance < limit).then_some(distance)
}

/// The per-channel weight of a grey analog free flight over `distance`, for a medium sampled at its channel-mean
/// extinction: `exp(-(σt_c - σ̄) · distance)`. The analog event and survival probabilities are the channel-mean ones, so
/// this factor carries the chromatic transmittance (Cycles' spectral analog, `volume_integrate_heterogeneous`). An
/// infinite distance has no survival in any channel that extinguishes more than the mean.
fn chromatic_correction(medium: &BsdfParams, distance: f64) -> DVec3 {
    let extinction =
        DVec3::from_array(medium.volume_scatter) + DVec3::from_array(medium.volume_absorption);
    let mean = extinction.element_sum() / 3.0;
    DVec3::from_array(extinction.to_array().map(|sigma| {
        if distance.is_finite() {
            (-(sigma - mean) * distance).exp()
        } else if sigma > mean {
            0.0
        } else {
            1.0
        }
    }))
}

/// The length of a voxel grid's sampled span along its ray, from `grid.entry`.
fn grid_span(grid: &VoxelGrid) -> f64 {
    grid.extinction.len() as f64 * grid.step
}

/// The per-channel transmittance `exp(-σt·length)` of a homogeneous medium. An infinite length is zero wherever
/// the medium has extinction.
fn medium_transmittance(params: &BsdfParams, length: f64) -> DVec3 {
    let extinction =
        DVec3::from_array(params.volume_scatter) + DVec3::from_array(params.volume_absorption);
    DVec3::from_array(extinction.to_array().map(|sigma| {
        if sigma <= 0.0 {
            1.0
        } else if length.is_finite() {
            (-sigma * length.max(0.0)).exp()
        } else {
            0.0
        }
    }))
}

fn volume_extinction_rgb_density(instance: &VolumeInstance<'_>, density: f64) -> DVec3 {
    instance.extinction_rgb_scale * density.max(0.0)
}

/// The weight of each light's direct light at a distance-sampled scatter event (`volume_direct_scatter_mis`): zero
/// for a light sampled equiangularly, the multiple importance sampling weight against its equiangular pdf under
/// multiple importance sampling, and one otherwise.
fn event_weights(
    choices: &[(Option<DVec3>, bool)],
    method: Option<crate::model::VolumeSampling>,
    origin: DVec3,
    direction: DVec3,
    segment_limit: f64,
    distance: f64,
    scatter_pdf: f64,
) -> Vec<f64> {
    use crate::model::VolumeSampling;
    choices
        .iter()
        .map(|&(point, sampled)| match (sampled, method, point) {
            (true, _, _) => 0.0,
            (false, Some(VolumeSampling::MultipleImportance), Some(point)) => {
                2.0 * power_heuristic(
                    scatter_pdf,
                    equiangular_pdf(point, origin, direction, segment_limit, distance),
                )
            }
            _ => 1.0,
        })
        .collect()
}
/// A segment's per-step RGB extinction and scatter, with static and temperature-grid emission sources.
struct VoxelGrid {
    entry: f64,
    exit: f64,
    step: f64,
    object_step_size: f64,
    meters_per_unit: f64,
    /// Per-channel extinction per meter of each step.
    extinction: Vec<DVec3>,
    /// Emission per meter of each step.
    emission: Vec<DVec3>,
    /// Scatter per meter and channel of each step, summed over the volumes.
    scatter: Vec<DVec3>,
    /// Grid-wide single-scattering albedo used to select the hero channel.
    channel_albedo: DVec3,
    /// All per-step extinction and scattering coefficients have identical RGB channels.
    achromatic: bool,
    /// The scatter of the volumes with each anisotropy, so every phase is evaluated with its own volumes.
    groups: Vec<ScatterGroup>,
    stochastic: bool,
    /// Blender's `cycles.volume_sampling` of each volume that the segment crosses.
    samplings: Vec<crate::model::VolumeSampling>,
}

/// The scatter per meter and channel of the volumes with one anisotropy, per step of a [`VoxelGrid`].
struct ScatterGroup {
    anisotropy: f64,
    scatter: Vec<DVec3>,
}

fn voxel_grid(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    ray: RayKind,
) -> Result<Option<VoxelGrid>> {
    let mut entry = f64::INFINITY;
    let mut exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    let mut stochastic = false;
    let mut samplings = Vec::new();
    for instance in volumes {
        if instance.ray_mask & ray.bit() == 0 {
            continue;
        }
        if !instance.voxel_size.is_finite()
            || instance.voxel_size <= 0.0
            || !instance.meters_per_unit.is_finite()
            || instance.meters_per_unit <= 0.0
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "volume scale or voxel size is invalid",
            ));
        }
        let local_origin = instance.world_to_local.transform_point3(origin);
        let local_direction = instance.world_to_local.transform_vector3(direction);
        if let Some((volume_entry, volume_exit)) = intersect_volume_bounds(
            local_origin,
            local_direction,
            instance.bounds_min,
            instance.bounds_max,
            max_distance,
        ) {
            entry = entry.min(volume_entry);
            exit = exit.max(volume_exit);
            minimum_step = minimum_step.min(volume_sampling_step(instance, local_direction));
            stochastic |= instance.scattering
                && instance.albedo.max_element() > 0.0
                && instance.extinction_rgb_scale.max_element() > 0.0;
            samplings.push(instance.sampling);
        }
    }
    if !entry.is_finite() || exit <= entry {
        return Ok(None);
    }
    let segment_length = exit - entry;
    let steps = (segment_length / minimum_step)
        .ceil()
        .clamp(1.0, MAX_VOLUME_STEPS as f64) as usize;
    let step = segment_length / steps as f64;
    let meters_per_unit = volumes
        .iter()
        .find(|instance| instance.ray_mask & ray.bit() != 0)
        .map_or(1.0, |instance| instance.meters_per_unit);
    let mut groups: Vec<ScatterGroup> = Vec::new();
    for instance in volumes
        .iter()
        .filter(|instance| instance.ray_mask & ray.bit() != 0)
    {
        for lobe in &instance.lobes {
            if !groups
                .iter()
                .any(|group| crate::float::equal_f64(group.anisotropy, lobe.anisotropy))
            {
                groups.push(ScatterGroup {
                    anisotropy: lobe.anisotropy,
                    scatter: vec![DVec3::ZERO; steps],
                });
            }
        }
    }
    let mut extinction = Vec::with_capacity(steps);
    let mut emission = Vec::with_capacity(steps);
    let mut scatter = vec![DVec3::ZERO; steps];
    let mut total_extinction = DVec3::ZERO;
    let mut total_scatter = DVec3::ZERO;
    let mut achromatic = true;
    for (index, step_total) in scatter.iter_mut().enumerate() {
        let point = origin + direction * (entry + (index as f64 + 0.5) * step);
        let mut step_extinction = DVec3::ZERO;
        let mut step_emission = DVec3::ZERO;
        for instance in volumes {
            if instance.ray_mask & ray.bit() == 0 {
                continue;
            }
            let density = volume_density_at(instance, point)?;
            let coefficient = volume_extinction_rgb_density(instance, density);
            let step_scatter = instance.albedo * coefficient;
            step_extinction += coefficient;
            step_emission += instance.emission;
            *step_total += step_scatter;
            total_extinction += coefficient;
            total_scatter += step_scatter;
            for lobe in &instance.lobes {
                if let Some(group) = groups
                    .iter_mut()
                    .find(|group| crate::float::equal_f64(group.anisotropy, lobe.anisotropy))
                {
                    group.scatter[index] += step_scatter * lobe.share;
                }
            }
        }
        achromatic &= step_extinction.x.total_cmp(&step_extinction.y).is_eq()
            && step_extinction.y.total_cmp(&step_extinction.z).is_eq()
            && step_total.x.total_cmp(&step_total.y).is_eq()
            && step_total.y.total_cmp(&step_total.z).is_eq();
        extinction.push(step_extinction);
        emission.push(step_emission);
    }
    let extinction_channels = total_extinction.to_array();
    let scatter_channels = total_scatter.to_array();
    let channel_albedo = DVec3::from_array(std::array::from_fn(|index| {
        if extinction_channels[index] > 0.0 {
            scatter_channels[index] / extinction_channels[index]
        } else {
            1.0
        }
    }))
    .clamp(DVec3::ZERO, DVec3::ONE);
    Ok(Some(VoxelGrid {
        entry,
        exit,
        step,
        object_step_size: minimum_step,
        meters_per_unit,
        extinction,
        emission,
        scatter,
        channel_albedo,
        achromatic,
        groups,
        stochastic,
        samplings,
    }))
}

struct TemperatureEmissionMarch<'a, 'volume> {
    volumes: &'a [VolumeInstance<'volume>],
    volume_temperatures: &'a [Option<VolumeTemperature>],
    origin: DVec3,
    direction: DVec3,
    ray: RayKind,
    ray_tmin: f32,
    ray_tmax: f32,
    step_size: f32,
    shade_offset: f32,
    offset: f32,
    step: usize,
    segment_max: f32,
}

impl<'a, 'volume> TemperatureEmissionMarch<'a, 'volume> {
    fn new(
        volumes: &'a [VolumeInstance<'volume>],
        volume_temperatures: &'a [Option<VolumeTemperature>],
        origin: DVec3,
        direction: DVec3,
        grid: &VoxelGrid,
        rng: &Pcg32,
        ray: RayKind,
    ) -> Option<Self> {
        if !volumes.iter().enumerate().any(|(index, instance)| {
            instance.ray_mask & ray.bit() != 0
                && volume_temperatures.get(index).is_some_and(Option::is_some)
        }) {
            return None;
        }

        let ray_tmin = grid.entry as f32;
        let ray_tmax = grid.exit as f32;
        let ray_length = ray_tmax - ray_tmin;
        let mut step_size = (grid.object_step_size as f32).min(ray_length);
        if ray_length > CYCLES_VOLUME_MAX_STEPS as f32 * step_size {
            step_size = ray_length / CYCLES_VOLUME_MAX_STEPS as f32;
        }
        // Keep the Cycles step dimensions from advancing Potter's transport RNG stream.
        let mut offset_rng = (*rng).clone();
        let shade_offset = (offset_rng.next_f64() as f32).min(1.0 - f32::EPSILON);
        let offset = (offset_rng.next_f64() as f32).min(1.0 - f32::EPSILON);
        Some(Self {
            volumes,
            volume_temperatures,
            origin,
            direction,
            ray,
            ray_tmin,
            ray_tmax,
            step_size,
            shade_offset,
            offset,
            step: 0,
            segment_max: ray_tmin,
        })
    }

    fn next(&mut self) -> Result<Option<TemperatureEmissionStep>> {
        if self.step == CYCLES_VOLUME_MAX_STEPS || self.segment_max >= self.ray_tmax {
            return Ok(None);
        }
        let segment_min = self.segment_max;
        self.segment_max = self
            .ray_tmax
            .min(self.ray_tmin + (self.step as f32 + self.offset) * self.step_size);
        let shade_t = segment_min + (self.segment_max - segment_min) * self.shade_offset;
        self.step += 1;
        let point = self.origin + self.direction * f64::from(shade_t);
        let mut emission = DVec3::ZERO;
        for (volume_index, instance) in self.volumes.iter().enumerate() {
            if instance.ray_mask & self.ray.bit() == 0 {
                continue;
            }
            if let Some(temperature) = self
                .volume_temperatures
                .get(volume_index)
                .and_then(Option::as_ref)
            {
                emission +=
                    volume_emission_at(instance, Some(temperature), point)? - instance.emission;
            }
        }
        Ok(Some(TemperatureEmissionStep {
            start: f64::from(segment_min),
            end: f64::from(self.segment_max),
            emission,
        }))
    }
}

#[derive(Clone, Copy)]
struct TemperatureEmissionStep {
    start: f64,
    end: f64,
    emission: DVec3,
}

/// The optical depth, in meters, from the segment origin to `distance` along the grid's steps. Each step is
/// linear in its own distance, as the sampler accumulates it.
fn grid_depth(grid: &VoxelGrid, distance: f64) -> DVec3 {
    let mut depth = DVec3::ZERO;
    for (index, extinction) in grid.extinction.iter().enumerate() {
        let start = grid.entry + index as f64 * grid.step;
        if distance <= start {
            break;
        }
        let covered = distance.min(start + grid.step) - start;
        depth += *extinction * (covered * grid.meters_per_unit);
    }
    depth
}

/// The grid step containing `distance`, or `None` off the grid.
fn grid_step_index(grid: &VoxelGrid, distance: f64) -> Option<usize> {
    let offset = (distance - grid.entry) / grid.step;
    (offset >= 0.0 && offset < grid.extinction.len() as f64).then_some(offset as usize)
}

/// The scatter per meter and channel of a per-step table at `distance`, zero off the grid.
fn grid_scatter_at(grid: &VoxelGrid, values: &[DVec3], distance: f64) -> DVec3 {
    grid_step_index(grid, distance).map_or(DVec3::ZERO, |index| values[index])
}

fn grid_channel_pdf(grid: &VoxelGrid, throughput: DVec3) -> DVec3 {
    volume_channel_pdf(grid.channel_albedo, throughput)
}

fn volume_channel_pdf(albedo: DVec3, throughput: DVec3) -> DVec3 {
    let weights = (albedo * throughput).abs();
    let sum = weights.element_sum();
    if (1.0 - sum) < 1.0 {
        weights / sum
    } else {
        DVec3::splat(1.0 / 3.0)
    }
}

fn grid_survival_pdf(grid: &VoxelGrid, distance: f64, channel_pdf: DVec3) -> f64 {
    channel_pdf.dot((-grid_depth(grid, distance)).exp())
}
fn grid_survival_weight(grid: &VoxelGrid, distance: f64, channel_pdf: DVec3) -> DVec3 {
    let transmittance = (-grid_depth(grid, distance)).exp();
    let survival_pdf = grid_survival_pdf(grid, distance, channel_pdf);
    if survival_pdf > 0.0 {
        transmittance / survival_pdf
    } else {
        DVec3::ZERO
    }
}

/// The density per unit distance of grid scatter under the hero-channel mixture: per-channel scatter times
/// transmittance, weighted by the channel PDF.
fn grid_scatter_pdf(grid: &VoxelGrid, distance: f64, channel_pdf: DVec3) -> f64 {
    let scatter = grid_scatter_at(grid, &grid.scatter, distance) * grid.meters_per_unit;
    channel_pdf.dot(scatter * (-grid_depth(grid, distance)).exp())
}

/// The per-channel extinction of a medium: its scatter plus its absorption, per unit length.
fn medium_extinction(medium: &BsdfParams) -> DVec3 {
    DVec3::from_array(medium.volume_scatter) + DVec3::from_array(medium.volume_absorption)
}

/// One piece of a segment's emission, `length` scene units long: `source` per unit length, weighted by the transmittance
/// `exponent` of everything before it and by the piece's own per-channel extinction `rate`. Its integral is Cycles' closed
/// form `E · (1 - exp(-σ·length)) / σ` (`volume_emission_integrate`), which is `E · length` as σ goes to zero.
fn add_emission_piece(
    emitted: &mut DVec3,
    exponent: &mut DVec3,
    length: f64,
    rate: DVec3,
    source: DVec3,
) {
    if source != DVec3::ZERO {
        let shape = DVec3::from_array(rate.to_array().map(|rate| {
            if rate.abs() > 1.0e-12 {
                (1.0 - (-rate * length).exp()) / rate
            } else {
                length
            }
        }));
        *emitted += source * (-*exponent).exp() * shape;
    }
    *exponent += rate * length;
}

/// The exact RGB emission integral to `end`: mesh coefficients are constant, while voxel extinction and each emission
/// source are piecewise constant over their respective steps.
fn emission_until(
    grid: Option<&VoxelGrid>,
    mesh: Option<&BsdfParams>,
    end: f64,
    mut temperature_march: Option<&mut TemperatureEmissionMarch<'_, '_>>,
) -> Result<DVec3> {
    let mesh_rate = mesh.map_or(DVec3::ZERO, medium_extinction);
    let mesh_source = mesh.map_or(DVec3::ZERO, |medium| {
        DVec3::from_array(medium.volume_emission)
    });
    let mut emitted = DVec3::ZERO;
    let mut exponent = DVec3::ZERO;
    let mut position = 0.0_f64;
    let mut temperature_step = match temperature_march.as_deref_mut() {
        Some(march) => march.next()?,
        None => None,
    };
    if let Some(grid) = grid {
        for (index, extinction) in grid.extinction.iter().enumerate() {
            let start = grid.entry + index as f64 * grid.step;
            let gap_end = start.min(end);
            if gap_end > position {
                add_emission_piece(
                    &mut emitted,
                    &mut exponent,
                    gap_end - position,
                    mesh_rate,
                    mesh_source,
                );
                position = gap_end;
            }
            let grid_end = (start + grid.step).min(end);
            if grid_end > position {
                let grid_rate = *extinction * grid.meters_per_unit;
                let grid_source = grid.emission[index] * grid.meters_per_unit;
                while position < grid_end {
                    while temperature_step.is_some_and(|step| step.end <= position) {
                        temperature_step = match temperature_march.as_deref_mut() {
                            Some(march) => march.next()?,
                            None => None,
                        };
                    }
                    let (temperature_source, piece_end) = match temperature_step {
                        Some(step) if position < step.start => {
                            (DVec3::ZERO, grid_end.min(step.start))
                        }
                        Some(step) => {
                            (step.emission * grid.meters_per_unit, grid_end.min(step.end))
                        }
                        None => (DVec3::ZERO, grid_end),
                    };
                    add_emission_piece(
                        &mut emitted,
                        &mut exponent,
                        piece_end - position,
                        mesh_rate + grid_rate,
                        grid_source + mesh_source + temperature_source,
                    );
                    position = piece_end;
                }
            }
            if position >= end {
                return Ok(emitted);
            }
        }
    }
    if end > position {
        add_emission_piece(
            &mut emitted,
            &mut exponent,
            end - position,
            mesh_rate,
            mesh_source,
        );
    }
    Ok(emitted)
}

/// Samples one RGB hero channel, then samples a free flight in that channel. The event's RGB
/// scattering weight divides the per-channel scatter/transmittance by the mixture distance PDF.
fn trace_volume_segment(
    grid: Option<&VoxelGrid>,
    origin: DVec3,
    direction: DVec3,
    throughput: DVec3,
    rng: &mut Pcg32,
    events: bool,
) -> VolumeSegment {
    let Some(grid) = grid else {
        return VolumeSegment {
            transmittance: DVec3::ONE,
            event: None,
            stochastic: false,
            channel_pdf: DVec3::splat(1.0 / 3.0),
        };
    };
    let channel_pdf = grid_channel_pdf(grid, throughput);
    let channel = if grid.stochastic && events && !grid.achromatic {
        let sample = rng.next_f64();
        if sample < channel_pdf.x {
            0
        } else if sample < channel_pdf.x + channel_pdf.y {
            1
        } else {
            2
        }
    } else {
        0
    };
    let free_flight = if grid.stochastic && events {
        -rng.next_f64().max(f64::MIN_POSITIVE).ln()
    } else {
        f64::INFINITY
    };
    let step_in_meters = grid.step * grid.meters_per_unit;
    let mut accumulated_depth = 0.0;
    for (index, extinction) in grid.extinction.iter().enumerate() {
        let sigma_t = extinction.to_array()[channel];
        if sigma_t <= 0.0 {
            continue;
        }
        let start = grid.entry + index as f64 * grid.step;
        let step_depth = sigma_t * step_in_meters;
        if free_flight <= accumulated_depth + step_depth {
            let traveled_meters =
                ((free_flight - accumulated_depth) / sigma_t).clamp(0.0, step_in_meters);
            let event_distance = start + traveled_meters / grid.meters_per_unit;
            let point = origin + direction * event_distance;
            let transmittance = (-grid_depth(grid, event_distance)).exp();
            let scatter = grid_scatter_at(grid, &grid.scatter, event_distance);
            let scatter_probability = if sigma_t > 0.0 {
                scatter.to_array()[channel] / sigma_t
            } else {
                0.0
            };
            let scattering_weight = if rng.next_f64() < scatter_probability {
                let scatter_pdf = channel_pdf.dot(scatter * grid.meters_per_unit * transmittance);
                if scatter_pdf > 0.0 {
                    scatter * (grid.meters_per_unit / scatter_pdf) * transmittance
                } else {
                    DVec3::ZERO
                }
            } else {
                DVec3::ZERO
            };
            return VolumeSegment {
                transmittance: DVec3::ONE,
                event: Some(VolumeEvent {
                    point,
                    scattering_weight,
                }),
                stochastic: grid.stochastic,
                channel_pdf,
            };
        }
        accumulated_depth += step_depth;
    }
    let distance = grid.entry + grid_span(grid);
    let transmittance = (-grid_depth(grid, distance)).exp();
    let survival_pdf = if grid.stochastic && events {
        grid_survival_pdf(grid, distance, channel_pdf)
    } else {
        1.0
    };
    VolumeSegment {
        transmittance: if survival_pdf > 0.0 {
            transmittance / survival_pdf
        } else {
            DVec3::ZERO
        },
        event: None,
        stochastic: grid.stochastic && events,
        channel_pdf,
    }
}

fn volume_contains_point(instance: &VolumeInstance<'_>, world_position: DVec3) -> bool {
    let local = instance.world_to_local.transform_point3(world_position);
    local.x >= instance.bounds_min.x
        && local.y >= instance.bounds_min.y
        && local.z >= instance.bounds_min.z
        && local.x <= instance.bounds_max.x
        && local.y <= instance.bounds_max.y
        && local.z <= instance.bounds_max.z
}

fn volume_density_at(instance: &VolumeInstance<'_>, world_position: DVec3) -> Result<f64> {
    let local_position = instance.world_to_local.transform_point3(world_position);
    let density = instance_density(instance, local_position);
    if !density.is_finite() {
        instance_density_checked(instance, local_position)?;
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "volume density is not finite",
        ));
    }
    Ok(density.max(0.0))
}

/// The density of a volume instance at a volume-local position. An inline grid samples at its index coordinate, through the
/// inverse cached on the instance; a decoded VDB samples at its local position.
fn instance_density(instance: &VolumeInstance<'_>, local_position: DVec3) -> f64 {
    instance_density_checked(instance, local_position).unwrap_or(f64::NAN)
}

/// `instance_density` with its error, which reports why a density is not finite.
fn instance_density_checked(instance: &VolumeInstance<'_>, local_position: DVec3) -> Result<f64> {
    match instance.index_from_local {
        Some(index_from_local) => {
            let coordinate = index_from_local.transform_point3(local_position);
            if instance.path_traced {
                sample_index_density_checked_with(
                    coordinate,
                    instance.volume,
                    instance.interpolation,
                )
            } else {
                sample_index_density_realtime_checked_with(
                    coordinate,
                    instance.volume,
                    instance.interpolation,
                )
            }
        }
        None if instance.path_traced => {
            sample_density_checked_with(local_position, instance.volume, instance.interpolation)
        }
        None => sample_density_realtime_checked_with(
            local_position,
            instance.volume,
            instance.interpolation,
        ),
    }
}

/// The ray-march step in meters along a volume-local direction. Inline grids use half-voxel steps; decoded VDBs use
/// one affine voxel, as Blender's volume renderers do.
fn volume_sampling_step(instance: &VolumeInstance<'_>, local_direction: DVec3) -> f64 {
    match instance.index_from_local {
        Some(index_from_local) => {
            let index_speed = index_from_local
                .transform_vector3(local_direction)
                .length()
                .max(f64::MIN_POSITIVE);
            0.5 / index_speed
        }
        None => instance.voxel_size / local_direction.length().max(f64::MIN_POSITIVE),
    }
}

fn march_volumes(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    rng: Option<&mut Pcg32>,
    ray: RayKind,
) -> Result<DVec3> {
    march_volumes_with_budget(
        volumes,
        origin,
        direction,
        max_distance,
        rng,
        MAX_VOLUME_STEPS,
        ray,
    )
}

fn march_volumes_with_budget(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    mut rng: Option<&mut Pcg32>,
    step_budget: usize,
    ray: RayKind,
) -> Result<DVec3> {
    let step_budget = step_budget.max(1);
    let mut transmittance = DVec3::ONE;
    for instance in volumes {
        if instance.ray_mask & ray.bit() == 0 {
            continue;
        }
        if !instance.voxel_size.is_finite() || instance.voxel_size <= 0.0 {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "volume voxel size is invalid",
            ));
        }
        if !instance.meters_per_unit.is_finite() || instance.meters_per_unit <= 0.0 {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "volume scene unit scale is invalid",
            ));
        }
        let local_origin = instance.world_to_local.transform_point3(origin);
        let local_direction = instance.world_to_local.transform_vector3(direction);
        let Some((entry, exit)) = intersect_volume_bounds(
            local_origin,
            local_direction,
            instance.bounds_min,
            instance.bounds_max,
            max_distance,
        ) else {
            continue;
        };
        let segment_length = (exit - entry).max(0.0);
        if segment_length <= 0.0 {
            continue;
        }
        // Use ratio tracking for heterogeneous grids; constant density remains exact.
        if let (Some(random), Some((minimum_density, maximum_density))) =
            (rng.as_deref_mut(), instance.density_range)
        {
            let majorant = instance.extinction_rgb_scale.max_element() * maximum_density;
            if majorant > 0.0 && (maximum_density - minimum_density).abs() > f64::EPSILON {
                let mut distance = entry;
                for _ in 0..step_budget {
                    let free_flight = -random.next_f64().max(f64::MIN_POSITIVE).ln()
                        / (majorant * instance.meters_per_unit);
                    distance += free_flight;
                    if distance >= exit {
                        break;
                    }
                    let local_position = local_origin + local_direction * distance;
                    let density = instance_density(instance, local_position);
                    if !density.is_finite() {
                        instance_density_checked(instance, local_position)?;
                        return Err(PotError::new(
                            ErrorCode::EvaluationFailed,
                            "volume density is not finite",
                        ));
                    }
                    transmittance *= DVec3::ONE
                        - volume_extinction_rgb_density(instance, density)
                            .min(DVec3::splat(majorant))
                            / majorant;
                    if transmittance.max_element() <= 1.0e-6 {
                        break;
                    }
                }
            } else if majorant > 0.0 {
                transmittance *= (-volume_extinction_rgb_density(instance, maximum_density)
                    * segment_length
                    * instance.meters_per_unit)
                    .exp();
            }
            if transmittance.max_element() <= 1.0e-6 {
                break;
            }
            continue;
        }
        let step_world = volume_sampling_step(instance, local_direction)
            .max(segment_length / step_budget as f64);
        let steps = (segment_length / step_world)
            .ceil()
            .clamp(1.0, step_budget as f64) as usize;
        let step = segment_length / steps as f64;
        for sample_index in 0..steps {
            let distance = entry + (sample_index as f64 + 0.5) * step;
            let local_position = local_origin + local_direction * distance;
            let density = instance_density(instance, local_position);
            if !density.is_finite() {
                instance_density_checked(instance, local_position)?;
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "volume density is not finite",
                ));
            }
            let attenuation = (-volume_extinction_rgb_density(instance, density)
                * step
                * instance.meters_per_unit)
                .exp();
            transmittance *= attenuation;
            if transmittance.max_element() <= 1.0e-6 {
                break;
            }
        }
        if transmittance.max_element() <= 1.0e-6 {
            break;
        }
    }
    Ok(transmittance.clamp(DVec3::ZERO, DVec3::ONE))
}

fn sample_henyey_greenstein_direction(incoming: DVec3, anisotropy: f64, rng: &mut Pcg32) -> DVec3 {
    let g = anisotropy.clamp(-0.999, 0.999);
    let uniform = rng.next_f64();
    let cosine = if g.abs() < 1.0e-3 {
        1.0 - 2.0 * uniform
    } else {
        let ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * uniform);
        ((1.0 + g * g - ratio * ratio) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    let angle = std::f64::consts::TAU * rng.next_f64();
    let (tangent, bitangent) = basis(incoming.normalize_or_zero());
    (tangent * (sine * angle.cos())
        + bitangent * (sine * angle.sin())
        + incoming.normalize_or_zero() * cosine)
        .normalize_or_zero()
}

fn henyey_greenstein(cosine: f64, anisotropy: f64) -> f64 {
    let g = anisotropy.clamp(-0.999, 0.999);
    let denominator = (1.0 + g * g - 2.0 * g * cosine.clamp(-1.0, 1.0)).max(f64::MIN_POSITIVE);
    (1.0 - g * g) / (4.0 * std::f64::consts::PI * denominator.powf(1.5))
}

fn intersect_volume_bounds(
    origin: DVec3,
    direction: DVec3,
    minimum: DVec3,
    maximum: DVec3,
    max_distance: f64,
) -> Option<(f64, f64)> {
    let mut near = 0.0_f64;
    let mut far = max_distance;
    for axis in 0..3 {
        let origin_axis = origin[axis];
        let direction_axis = direction[axis];
        if direction_axis.abs() <= f64::EPSILON {
            if origin_axis < minimum[axis] || origin_axis > maximum[axis] {
                return None;
            }
            continue;
        }
        let reciprocal = 1.0 / direction_axis;
        let first = (minimum[axis] - origin_axis) * reciprocal;
        let second = (maximum[axis] - origin_axis) * reciprocal;
        near = near.max(first.min(second));
        far = far.min(first.max(second));
        if far <= near {
            return None;
        }
    }
    (far > near).then_some((near, far))
}

pub(super) fn load_images(
    doc: &SceneDoc,
    root: &Path,
    staged_assets: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<BTreeMap<String, ImageData>> {
    let mut references = BTreeMap::<String, ImageInterpolation>::new();
    for material in doc.materials.values() {
        for texture in [
            material.base_color_texture.as_ref(),
            material.roughness_texture.as_ref(),
            material.metallic_texture.as_ref(),
            material.normal_texture.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            references
                .entry(texture.image.to_string())
                .or_insert(texture.interpolation);
        }
    }
    for group in doc.node_groups.values() {
        for node in group.nodes.values() {
            let Some(image_id) = node
                .properties
                .get("image")
                .or_else(|| node.properties.get("image_id"))
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let interpolation = match node
                .properties
                .get("interpolation")
                .and_then(serde_json::Value::as_str)
            {
                Some("closest") => ImageInterpolation::Closest,
                _ => ImageInterpolation::Linear,
            };
            references
                .entry(image_id.to_owned())
                .or_insert(interpolation);
        }
    }
    let mut images = BTreeMap::new();
    for (image_id, interpolation) in references {
        let id = Id::new(image_id.clone())?;
        let image = doc.images.get(&id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("image `{image_id}` does not exist"),
                serde_json::json!({ "image_id": image_id }),
            )
        })?;
        let image_data = if let Some(staged_assets) = staged_assets {
            crate::image::load_image_data_with_staged(image, root, interpolation, staged_assets)?
        } else {
            crate::image::load_image_data(image, root, interpolation)?
        };
        images.insert(image_id, image_data);
    }

    Ok(images)
}

fn offset_origin(point: DVec3, direction: DVec3, normal: DVec3) -> DVec3 {
    let sign = if direction.dot(normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    point + normal * (sign * RAY_EPSILON * (1.0 + point.abs().max_element()))
}

/// The start of a shadow ray leaving a surface point, as Cycles' `shadow_ray_offset` (`kernel/light/sample.h`). The
/// geometric normal is flipped to the transmission side when the light lies behind the smooth shading normal
/// (`dot(N, L) < 0`), and the start lies on that side. A ray that leaves the front of the normal it is measured against
/// does not meet its own triangle (Cycles' `skip_self`); a ray that crosses that normal, a light below the geometric
/// surface, meets the surface and is blocked unless it is transparent.
fn shadow_origin(point: DVec3, direction: DVec3, normal: DVec3, shading: DVec3) -> DVec3 {
    let front = if shading.dot(direction) < 0.0 {
        -normal
    } else {
        normal
    };
    point + front * (RAY_EPSILON * (1.0 + point.abs().max_element()))
}

fn allocate<T: Clone>(length: usize, value: T, label: &str) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            format!("{label} exceeds available memory"),
        )
    })?;
    output.resize(length, value);
    Ok(output)
}

fn finite_f32(value: f64) -> f32 {
    if value.is_finite() {
        value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
    } else {
        0.0
    }
}

fn to_srgb_byte(value: f64) -> u8 {
    let linear = value.clamp(0.0, 1.0);
    let srgb = if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
#[cfg(test)]
mod volume_extinction_tests {
    #![expect(
        clippy::expect_used,
        reason = "sphere-light regression samples must exist"
    )]
    use super::*;
    use crate::geom::volume::{VolumeData, VolumeGrid};

    #[test]
    fn density_one_per_meter_obeys_beer_lambert_after_scene_unit_conversion()
    -> crate::error::Result<()> {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 3, 3],
                values: Some(vec![1.0; 27]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let instance = VolumeInstance {
            volume: &volume,
            world_to_local: DMat4::IDENTITY,
            bounds_min: DVec3::ZERO,
            bounds_max: DVec3::splat(2.0),
            voxel_size: 1.0,
            meters_per_unit: 0.5,
            density_range: Some((1.0, 1.0)),
            albedo: DVec3::ZERO,
            lobes: scatter_lobes(&[]),
            emission: DVec3::ZERO,
            scattering: false,
            extinction_rgb_scale: DVec3::ONE,
            sampling: crate::model::VolumeSampling::MultipleImportance,
            interpolation: crate::model::VolumeInterpolation::Linear,
            index_from_local: None,
            path_traced: true,
            ray_mask: crate::model::RayVisibility::default().mask(),
        };
        let mut rng = Pcg32::for_sample(0, 0, 0, 0);
        let transmittance = march_volumes(
            &[instance],
            DVec3::new(-1.0, 1.0, 1.0),
            DVec3::X,
            f64::INFINITY,
            Some(&mut rng),
            RayKind::Shadow,
        )?;

        assert!(
            (transmittance - DVec3::splat((-1.0_f64).exp()))
                .abs()
                .max_element()
                <= 1.0e-12,
            "2 scene units at 0.5 m/unit should attenuate by one meter"
        );
        Ok(())
    }
    #[test]
    fn a_volume_hidden_from_camera_rays_still_shadows() -> crate::error::Result<()> {
        // Cycles tests a volume's bounds per ray kind: a volume hidden from the camera passes camera rays, and one hidden
        // from shadows passes shadow rays. Both are checked on the same unit-density slab (`visible_camera` off,
        // `visible_shadow` on, then the reverse).
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 3, 3],
                values: Some(vec![1.0; 27]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let instance = |ray_mask: u8| VolumeInstance {
            volume: &volume,
            world_to_local: DMat4::IDENTITY,
            bounds_min: DVec3::ZERO,
            bounds_max: DVec3::splat(2.0),
            voxel_size: 1.0,
            meters_per_unit: 1.0,
            density_range: Some((1.0, 1.0)),
            albedo: DVec3::ZERO,
            lobes: vec![ScatterLobe {
                share: 1.0,
                anisotropy: 0.0,
            }],
            emission: DVec3::ZERO,
            scattering: false,
            extinction_rgb_scale: DVec3::ONE,
            sampling: crate::model::VolumeSampling::default(),
            interpolation: crate::model::VolumeInterpolation::Linear,
            index_from_local: None,
            path_traced: true,
            ray_mask,
        };
        let shadow_only = RayKind::Shadow.bit();
        let camera_only = RayKind::Camera.bit();
        let march = |volumes: &[VolumeInstance<'_>], ray: RayKind| {
            let mut rng = Pcg32::for_sample(0, 0, 0, 0);
            march_volumes(
                volumes,
                DVec3::new(-1.0, 1.0, 1.0),
                DVec3::X,
                f64::INFINITY,
                Some(&mut rng),
                ray,
            )
        };
        let hidden_from_camera = [instance(shadow_only)];
        assert!(
            (march(&hidden_from_camera, RayKind::Camera)? - DVec3::ONE)
                .abs()
                .max_element()
                <= 1.0e-12,
            "a volume hidden from camera rays must pass them"
        );
        assert!(
            march(&hidden_from_camera, RayKind::Shadow)?.max_element() < 0.9,
            "a volume visible to shadows must shadow"
        );
        let hidden_from_shadow = [instance(camera_only)];
        assert!(
            (march(&hidden_from_shadow, RayKind::Shadow)? - DVec3::ONE)
                .abs()
                .max_element()
                <= 1.0e-12,
            "a volume hidden from shadows must pass shadow rays"
        );
        Ok(())
    }

    #[test]
    fn henyey_greenstein_samples_match_the_phase_pdf() {
        const BIN_COUNT: usize = 10;
        const SAMPLE_COUNT: usize = 100_000;
        let anisotropy = 0.35;
        let mut rng = Pcg32::for_sample(0x1234, 0, 0, 0);
        let mut bins = [0_usize; BIN_COUNT];
        for _ in 0..SAMPLE_COUNT {
            let direction = sample_henyey_greenstein_direction(DVec3::Z, anisotropy, &mut rng);
            let cosine = direction.z.clamp(-1.0, 1.0);
            let bin =
                ((cosine.midpoint(1.0) * BIN_COUNT as f64).floor() as usize).min(BIN_COUNT - 1);
            bins[bin] += 1;
        }
        for (index, observed) in bins.into_iter().enumerate() {
            let lower = -1.0 + 2.0 * index as f64 / BIN_COUNT as f64;
            let upper = -1.0 + 2.0 * (index + 1) as f64 / BIN_COUNT as f64;
            let probability = hg_cdf(upper, anisotropy) - hg_cdf(lower, anisotropy);
            let expected = SAMPLE_COUNT as f64 * probability;
            let tolerance =
                6.0 * (SAMPLE_COUNT as f64 * probability * (1.0 - probability)).sqrt() + 3.0;
            assert!(
                (observed as f64 - expected).abs() <= tolerance,
                "HG bin {index}: observed {observed}, expected {expected:.1} ± {tolerance:.1}"
            );
        }
    }

    #[test]
    fn thin_sun_lit_slab_matches_the_first_order_single_scatter_solution() {
        let extinction = 0.002;
        let albedo = 0.5;
        let light = DVec3::splat(3.0);
        let phase = henyey_greenstein(0.4, 0.0);
        let distance = 0.1;
        let (_, integral) = homogeneous_segment_integral(extinction, distance);
        let single_scatter = light * (extinction * albedo * phase * integral);
        let first_order = light * (extinction * albedo * phase * distance);
        assert!((single_scatter - first_order).length() / first_order.length() < 1.0e-4);
    }

    #[test]
    fn zero_albedo_is_exact_beer_lambert_absorption() {
        let extinction = 0.7;
        let distance = 1.25;
        let (transmittance, integral) = homogeneous_segment_integral(extinction, distance);
        let scatter = DVec3::splat(extinction * 0.0 * integral);
        assert_eq!(transmittance, (-extinction * distance).exp());
        assert_eq!(scatter, DVec3::ZERO);
    }

    #[test]
    fn unit_albedo_preserves_uniform_white_environment_energy() {
        let extinction = 0.7;
        let distance = 2.0;
        let (transmittance, _) = homogeneous_segment_integral(extinction, distance);
        let environment = DVec3::ONE;
        let single_scatter = environment * (1.0 - transmittance);
        let outgoing = environment * transmittance + single_scatter;
        assert!((outgoing - environment).length() <= 1.0e-12);
    }

    #[test]
    fn area_light_estimator_matches_power_over_pi_distance_squared_and_faces_one_way() {
        const SAMPLE_COUNT: usize = 10_000;
        let power = 1_000.0;
        let distance = 10.0;
        // A unit square at distance 10 subtends 0.25% less than 1 / d², within the 1% tolerance.
        // Sampling noise at this sample count is far smaller.
        let light = PointLight {
            position: DVec3::ZERO,
            direction: DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Area,
            color: DVec3::ONE,
            energy: power,
            radius: 0.0,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
            ray_mask: crate::model::RayVisibility::default().mask(),
            camera_shape_eligible: true,
            spread: None,
            spot_size: 0.0,
            spot_blend: 0.0,
            area: Some(AreaShape {
                axis_u: DVec3::X,
                axis_v: DVec3::Y,
                half_u: 0.5,
                half_v: 0.5,
                ellipse: false,
                area: 1.0,
            }),
        };
        let mut rng = Pcg32::for_sample(0x5eed, 0, 0, 0);
        let front = DVec3::new(0.0, 0.0, distance);
        let mut sum = 0.0;
        for _ in 0..SAMPLE_COUNT {
            sum +=
                sample_area_light(&light, front, &mut rng).map_or(0.0, |sample| sample.radiance.x);
        }
        let mean = sum / SAMPLE_COUNT as f64;
        let expected = power / (std::f64::consts::PI * distance * distance);
        assert!(
            (mean - expected).abs() <= 0.01 * expected,
            "area light mean {mean:.6} should match P/(pi*d^2) = {expected:.6} within 1%"
        );
        let behind = DVec3::new(0.0, 0.0, -distance);
        for _ in 0..SAMPLE_COUNT {
            assert!(sample_area_light(&light, behind, &mut rng).is_none());
        }
    }

    #[test]
    fn distant_tiny_sphere_converges_to_point_intensity() {
        let light = PointLight {
            position: DVec3::ZERO,
            direction: DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Point,
            color: DVec3::ONE,
            energy: 4.0 * std::f64::consts::PI,
            radius: 1.0e-6,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
            ray_mask: crate::model::RayVisibility::default().mask(),
            camera_shape_eligible: true,
            spread: None,
            spot_size: std::f64::consts::FRAC_PI_4,
            spot_blend: 0.15,
            area: None,
        };
        let distance = 1_000.0;
        let mut rng = Pcg32::for_sample(0x51e, 0, 0, 0);
        let sample = sample_light(&light, DVec3::new(0.0, 0.0, distance), &mut rng)
            .expect("a distant sphere remains visible");
        let expected = 1.0 / (distance * distance);
        assert!(sample.radiance.x > 0.0, "tiny sphere radiance is not zero");
        assert!(
            (sample.radiance.x - expected).abs() <= 1.0e-8 * expected,
            "tiny sphere radiance {} should approach point intensity {expected}",
            sample.radiance.x
        );
    }

    #[test]
    fn sphere_sample_near_its_centre_has_a_finite_exit_distance() {
        let light = PointLight {
            position: DVec3::ZERO,
            direction: DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Point,
            color: DVec3::ONE,
            energy: 100.0,
            radius: 1.0,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
            ray_mask: crate::model::RayVisibility::default().mask(),
            camera_shape_eligible: true,
            spread: None,
            spot_size: std::f64::consts::FRAC_PI_4,
            spot_blend: 0.15,
            area: None,
        };
        let mut rng = Pcg32::for_sample(0x51e, 0, 0, 0);
        let sample = sample_light(&light, DVec3::new(1.0e-9, 0.0, 0.0), &mut rng)
            .expect("a point inside a sphere is lit");
        assert!(sample.distance.is_finite() && (0.0..=2.0).contains(&sample.distance));
        assert!(sample.radiance.is_finite() && sample.radiance.x > 0.0);
    }

    fn hg_cdf(cosine: f64, anisotropy: f64) -> f64 {
        if anisotropy.abs() <= f64::EPSILON {
            return cosine.midpoint(1.0);
        }
        let anisotropy = anisotropy.clamp(-0.999, 0.999);
        let denominator = (1.0 + anisotropy * anisotropy - 2.0 * anisotropy * cosine).sqrt();
        (1.0 - anisotropy * anisotropy) / (2.0 * anisotropy)
            * (denominator.recip() - 1.0 / (1.0 + anisotropy))
    }
}

#[cfg(test)]
mod sampling_tests {
    use std::f64::consts::{FRAC_PI_2, TAU};

    use super::super::principled_bsdf::evaluate_masked;
    use super::*;
    use crate::model::Material;

    #[test]
    fn a_voxel_volume_keeps_one_phase_lobe_per_closure() {
        // Each closure's share of the scatter is its weight times its mean scatter over the total: 1 and 3 of 4.
        let material = Material::default();
        let mut forward = BsdfParams::from_material(&material, DVec3::Z);
        forward.volume_scatter = [1.0; 3];
        forward.volume_anisotropy = 0.8;
        let mut backward = forward;
        backward.volume_scatter = [3.0; 3];
        backward.volume_anisotropy = -0.5;
        let lobes = scatter_lobes(&[
            ShaderClosure {
                weight: 1.0,
                params: forward,
            },
            ShaderClosure {
                weight: 1.0,
                params: backward,
            },
        ]);
        assert_eq!(lobes.len(), 2);
        assert_eq!((lobes[0].share, lobes[0].anisotropy), (0.25, 0.8));
        assert_eq!((lobes[1].share, lobes[1].anisotropy), (0.75, -0.5));
        // A volume without closures scatters isotropically.
        let isotropic = scatter_lobes(&[]);
        assert_eq!(isotropic.len(), 1);
        assert_eq!((isotropic[0].share, isotropic[0].anisotropy), (1.0, 0.0));
    }

    /// The mean of a reflective draw's weight (`f · cos / p_mix`, from `Closures::reflect`) over the sampler's own directions is the
    /// integral of `f · cos` over the hemisphere, if the mixture density the sampler reports matches
    /// how it draws the directions. At metallic 0.5 all three branches contribute, so a wrong
    /// density in any of them biases the mean. The reference is a midpoint quadrature of
    /// `evaluate · cos` at two resolutions; its disagreement is part of the tolerance.
    #[test]
    fn sampled_bsdf_weight_mean_matches_the_evaluate_cos_quadrature() {
        let material = Material {
            base_color: [0.8, 0.5, 0.3, 1.0],
            metallic: 0.5,
            roughness: 0.3,
            ..Material::default()
        };
        let params = BsdfParams::from_material(&material, DVec3::Z);
        let normal = DVec3::Z;
        let view = DVec3::new(0.3, 0.1, 0.95).normalize();

        let quadrature = |n_theta: usize, n_phi: usize| -> DVec3 {
            let d_theta = FRAC_PI_2 / n_theta as f64;
            let d_phi = TAU / n_phi as f64;
            let mut total = DVec3::ZERO;
            for i in 0..n_theta {
                let theta = (i as f64 + 0.5) * d_theta;
                for j in 0..n_phi {
                    let phi = (j as f64 + 0.5) * d_phi;
                    let light = DVec3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    );
                    let cosine = light.dot(normal);
                    total += evaluate_masked(
                        &params,
                        &layers_at(&params, normal, view, normal, true),
                        light,
                        RayKind::ALL_MASK,
                    ) * (cosine * theta.sin() * d_theta * d_phi);
                }
            }
            total
        };
        let coarse = quadrature(400, 200);
        let fine = quadrature(800, 400);

        let samples = 400_000_usize;
        let mut rng = Pcg32::for_sample(1, 0, 0, 0);
        let closures = Closures::new(
            vec![HitClosure::new(1.0, &params, view, normal, true)],
            normal,
            normal,
            true,
        );
        let mut sum = DVec3::ZERO;
        let mut sum_squares = DVec3::ZERO;
        for _ in 0..samples {
            let weight = match closures.reflect(0, normal, &mut rng) {
                Reflection::Continuous { weight, .. } | Reflection::Delta { weight, .. } => weight,
                Reflection::Absorbed => DVec3::ZERO,
                Reflection::Subsurface { .. } => panic!("the test closure has no subsurface lobe"),
            };
            sum += weight;
            sum_squares += weight * weight;
        }
        let count = samples as f64;
        let mean = sum / count;
        let variance = (sum_squares / count - mean * mean).max(DVec3::ZERO);
        let stderr = (variance / count).sqrt();
        let reference = fine;
        let quadrature_error = (coarse - fine).abs();
        let tolerance = stderr * 4.0 + quadrature_error;
        let delta = (mean - reference).abs();
        assert!(
            delta.cmple(tolerance).all(),
            "sampled mean {mean:?} vs quadrature {reference:?} (tolerance {tolerance:?})"
        );
    }

    /// A BSDF ray that reaches a light must see the crossing, emitter radiance, and density that
    /// light sampling gives that direction. Otherwise the two MIS strategies do not partition the
    /// same integral, and the weights bias the image.
    #[test]
    fn light_hits_agree_with_light_samples_in_distance_radiance_and_density() {
        let receiver = DVec3::ZERO;
        let area = |half_u: f64, half_v: f64, ellipse: bool| AreaShape {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u,
            half_v,
            ellipse,
            area: if ellipse {
                std::f64::consts::PI * half_u * half_v
            } else {
                4.0 * half_u * half_v
            },
        };
        let light =
            |kind: LightType, position: DVec3, radius: f64, soft_falloff: bool| PointLight {
                position,
                direction: -DVec3::Z,
                world_transform: DMat4::IDENTITY,
                kind,
                color: DVec3::new(1.0, 0.5, 0.25),
                energy: 50.0,
                radius,
                soft_falloff,
                spot_size: 1.2,
                spot_blend: 0.3,
                area: None,
                use_mis: true,
                volume_factor: 1.0,
                volume_radius_squared: f64::INFINITY,
                ray_mask: crate::model::RayVisibility::default().mask(),
                camera_shape_eligible: true,
                spread: None,
            };
        let above = DVec3::new(0.2, -0.1, 3.0);
        let cases = [
            (
                "rectangle",
                PointLight {
                    area: Some(area(0.6, 0.3, false)),
                    ..light(LightType::Area, above, 0.0, false)
                },
            ),
            (
                "ellipse",
                PointLight {
                    area: Some(area(0.6, 0.3, true)),
                    ..light(LightType::Area, above, 0.0, false)
                },
            ),
            (
                "spread rectangle",
                PointLight {
                    area: Some(area(0.6, 0.3, false)),
                    spread: Spread::new(0.3),
                    ..light(LightType::Area, above, 0.0, false)
                },
            ),
            (
                "spread disk",
                PointLight {
                    area: Some(area(0.6, 0.6, true)),
                    spread: Spread::new(1.0),
                    ..light(LightType::Area, above, 0.0, false)
                },
            ),
            ("soft disk", light(LightType::Point, above, 0.5, true)),
            ("spot disk", light(LightType::Spot, above, 0.5, true)),
            ("sphere", light(LightType::Point, above, 0.5, false)),
            ("spot sphere", light(LightType::Spot, above, 0.5, false)),
            (
                "receiver inside a sphere",
                light(LightType::Point, DVec3::new(0.0, 0.0, 0.2), 0.5, false),
            ),
        ];
        let mut rng = Pcg32::for_sample(0x41e, 0, 0, 0);
        for (label, light) in cases {
            let mut checked = 0;
            for _ in 0..200 {
                let Some(sample) = sample_light(&light, receiver, &mut rng) else {
                    continue;
                };
                let Some((distance, radiance, pdf)) =
                    light_hit(&light, receiver, sample.direction, receiver, f64::INFINITY)
                else {
                    panic!("{label}: a sampled direction must hit its light");
                };
                assert!(
                    (distance - sample.distance).abs() <= 1.0e-7 * sample.distance,
                    "{label}: hit at {distance}, sampled at {}",
                    sample.distance
                );
                assert!(
                    (pdf - sample.pdf).abs() <= 1.0e-7 * sample.pdf,
                    "{label}: hit density {pdf}, sampled density {}",
                    sample.pdf
                );
                let emitted = sample.radiance * sample.pdf;
                let scale = radiance
                    .abs()
                    .max_element()
                    .max(emitted.abs().max_element());
                assert!(
                    (radiance - emitted).abs().max_element() <= 1.0e-7 * scale,
                    "{label}: hit radiance {radiance:?}, sampled radiance × pdf {emitted:?}"
                );
                checked += 1;
            }
            assert!(checked >= 100, "{label}: only {checked} samples were lit");
        }
    }

    /// Light sampling over a spread light reaches every point the cone lights: its mean irradiance
    /// matches a brute-force grid over the whole emitter, on and off the cone's axis.
    #[test]
    fn spread_light_samples_match_the_brute_force_irradiance() {
        let centre = DVec3::new(0.0, 0.0, 2.0);
        let half = 0.5;
        let normal = -DVec3::Z;
        for spread_value in [0.3, 1.0] {
            let Some(spread) = Spread::new(spread_value) else {
                panic!("a spread of {spread_value} is below π");
            };
            for ellipse in [false, true] {
                let shape = AreaShape::new(DVec3::X, DVec3::Y, half, half, ellipse);
                let light = PointLight {
                    position: centre,
                    direction: normal,
                    world_transform: DMat4::IDENTITY,
                    kind: LightType::Area,
                    color: DVec3::ONE,
                    energy: std::f64::consts::PI * shape.area,
                    radius: 0.0,
                    soft_falloff: false,
                    spot_size: 0.0,
                    spot_blend: 0.0,
                    area: Some(shape),
                    spread: Some(spread),
                    ray_mask: crate::model::RayVisibility::default().mask(),
                    camera_shape_eligible: true,
                    use_mis: true,
                    volume_factor: 1.0,
                    volume_radius_squared: f64::INFINITY,
                };
                for receiver in [
                    DVec3::ZERO,
                    DVec3::new(0.25, 0.1, 0.0),
                    DVec3::new(0.7, -0.2, 0.0),
                ] {
                    let label =
                        format!("spread {spread_value}, ellipse {ellipse}, receiver {receiver:?}");
                    // Midpoint grid over the emitter, with the receiver's cosine on the floor (+Z).
                    let cells: u32 = 400;
                    let step = 2.0 * half / f64::from(cells);
                    let mut brute = 0.0;
                    for i in 0..cells {
                        for j in 0..cells {
                            let x = -half + (f64::from(i) + 0.5) * step;
                            let y = -half + (f64::from(j) + 0.5) * step;
                            if ellipse && x * x + y * y > half * half {
                                continue;
                            }
                            let to_light = centre + DVec3::new(x, y, 0.0) - receiver;
                            let distance_squared = to_light.length_squared();
                            let direction = to_light / distance_squared.sqrt();
                            let cos_light = normal.dot(-direction);
                            let cos_receiver = DVec3::Z.dot(direction);
                            if cos_light <= 0.0 || cos_receiver <= 0.0 {
                                continue;
                            }
                            brute += spread_attenuation(spread, -direction, normal)
                                * cos_light
                                * cos_receiver
                                / distance_squared
                                * step
                                * step;
                        }
                    }
                    let mut rng = Pcg32::for_sample(0x5be, 0, 0, 0);
                    let samples: u32 = 200_000;
                    let (mut sum, mut sum_squares) = (0.0, 0.0);
                    for _ in 0..samples {
                        let weight = sample_light(&light, receiver, &mut rng)
                            .map_or(0.0, |sample| {
                                sample.radiance.x * DVec3::Z.dot(sample.direction).max(0.0)
                            });
                        sum += weight;
                        sum_squares += weight * weight;
                    }
                    let count = f64::from(samples);
                    let mean = sum / count;
                    let stderr = ((sum_squares / count - mean * mean).max(0.0) / count).sqrt();
                    assert!(brute > 0.0, "{label}: the emitter must light the receiver");
                    assert!(
                        (mean - brute).abs() <= 4.0 * stderr + 0.01 * brute,
                        "{label}: sampled irradiance {mean} against brute force {brute} \
                         (standard error {stderr})"
                    );
                }
            }
        }
    }

    /// A zero spread emits along its normal alone: a receiver in front sees the emitter at the
    /// point straight above it, as a delta light with Cycles' radiance on the normal.
    #[test]
    fn zero_spread_emits_along_the_normal_only() {
        let shape = AreaShape::new(DVec3::X, DVec3::Y, 0.5, 0.5, false);
        let light = PointLight {
            position: DVec3::new(0.0, 0.0, 2.0),
            direction: -DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Area,
            color: DVec3::ONE,
            energy: std::f64::consts::PI * shape.area,
            radius: 0.0,
            soft_falloff: false,
            spot_size: 0.0,
            spot_blend: 0.0,
            area: Some(shape),
            spread: Spread::new(0.0),
            ray_mask: crate::model::RayVisibility::default().mask(),
            camera_shape_eligible: true,
            use_mis: true,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
        };
        let mut rng = Pcg32::for_sample(0x5e0, 0, 0, 0);
        let Some(sample) = sample_light(&light, DVec3::new(0.4, -0.3, 0.5), &mut rng) else {
            panic!("a receiver in front is lit");
        };
        assert!((sample.direction - DVec3::Z).length() <= 1.0e-12);
        assert!((sample.distance - 1.5).abs() <= 1.0e-12);
        assert_eq!(sample.pdf, 0.0);
        assert!((sample.radiance.x - std::f64::consts::PI).abs() <= 1.0e-12);
        assert!(
            !light.hittable(),
            "a zero spread is never hit by a BSDF ray"
        );
        // Past the emitter's edge, the foot of the receiver is off the emitter, which emits nothing there.
        assert!(sample_light(&light, DVec3::new(0.8, 0.0, 0.5), &mut rng).is_none());
    }

    /// A light the path has just crossed is never found again: the origin is advanced past it, so
    /// each shape's crossing lies behind the origin. Covers the one-sided disk, the one-sided
    /// rectangle, a sphere seen from outside, and a receiver inside a sphere.
    #[test]
    fn a_crossed_light_is_not_hit_again() {
        let light = |kind: LightType, area: Option<AreaShape>, radius: f64, soft_falloff: bool| {
            PointLight {
                position: DVec3::new(0.0, 0.0, 2.0),
                direction: -DVec3::Z,
                kind,
                world_transform: DMat4::IDENTITY,
                color: DVec3::ONE,
                energy: 10.0,
                radius,
                soft_falloff,
                spot_size: 0.0,
                spot_blend: 0.0,
                area,
                use_mis: true,
                volume_factor: 1.0,
                volume_radius_squared: f64::INFINITY,
                ray_mask: crate::model::RayVisibility::default().mask(),
                camera_shape_eligible: true,
                spread: None,
            }
        };
        let rectangle = AreaShape {
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            half_u: 0.5,
            half_v: 0.5,
            ellipse: false,
            area: 1.0,
        };
        let cases = [
            (
                "rectangle",
                light(LightType::Area, Some(rectangle), 0.0, false),
                DVec3::ZERO,
            ),
            (
                "soft disk",
                light(LightType::Point, None, 0.5, true),
                DVec3::ZERO,
            ),
            (
                "sphere from outside",
                light(LightType::Point, None, 0.5, false),
                DVec3::ZERO,
            ),
            (
                "sphere from inside",
                light(LightType::Point, None, 0.5, false),
                DVec3::new(0.0, 0.0, 1.8),
            ),
        ];
        for (label, light, receiver) in cases {
            let direction = DVec3::Z;
            let first = light_hit(&light, receiver, direction, receiver, f64::INFINITY)
                .unwrap_or_else(|| panic!("{label}: the first crossing must be hit"));
            let advanced = receiver + direction * (first.0 + RAY_EPSILON);
            assert!(
                light_hit(&light, advanced, direction, receiver, f64::INFINITY).is_none(),
                "{label}: a crossed light must not be found again"
            );
        }
    }

    /// The sampling density of a light hit is per solid angle from the receiver. An origin past
    /// the receiver shortens the ray distance, but not the density: the rectangle at height 2 is
    /// 2 m from the receiver, so its density is `d² / (A cos) = 4` even though the origin is 1 m
    /// away.
    #[test]
    fn light_hit_density_is_measured_from_the_receiver_past_an_advanced_origin() {
        let light = PointLight {
            position: DVec3::new(0.0, 0.0, 2.0),
            direction: -DVec3::Z,
            kind: LightType::Area,
            world_transform: DMat4::IDENTITY,
            color: DVec3::ONE,
            energy: 10.0,
            radius: 0.0,
            soft_falloff: false,
            spot_size: 0.0,
            spot_blend: 0.0,
            area: Some(AreaShape {
                axis_u: DVec3::X,
                axis_v: DVec3::Y,
                half_u: 0.5,
                half_v: 0.5,
                ellipse: false,
                area: 1.0,
            }),
            use_mis: true,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
            ray_mask: crate::model::RayVisibility::default().mask(),
            camera_shape_eligible: true,
            spread: None,
        };
        let receiver = DVec3::ZERO;
        let origin = DVec3::new(0.0, 0.0, 1.0);
        let Some((distance, _, pdf)) = light_hit(&light, origin, DVec3::Z, receiver, f64::INFINITY)
        else {
            panic!("the advanced origin still sees the light");
        };
        assert!(
            (distance - 1.0).abs() < 1.0e-12,
            "distance {distance} from the origin"
        );
        assert!(
            (pdf - 4.0).abs() < 1.0e-12,
            "density {pdf} is not measured from the receiver"
        );
    }

    #[test]
    fn bent_closure_normal_fails_only_when_the_light_is_on_the_other_side_of_the_smooth_surface() {
        let smooth = DVec3::Z;
        let bent = DVec3::new(0.6, 0.0, 0.8);
        // Above both the smooth surface and the bent normal: the three signs agree.
        assert!(shading_side_passes(
            smooth,
            bent,
            DVec3::new(0.643, 0.0, 0.766)
        ));
        // Below the smooth surface (`cos_smooth < 0`) but above the bent normal (`cos_bent > 0`): the product is
        // negative, so the closure is zero (Cycles `bump_shadowing_term`, `cosNsI * cosNsN * cosNI < 0`).
        assert!(!shading_side_passes(
            smooth,
            bent,
            DVec3::new(0.9, 0.0, -0.1).normalize()
        ));
        // A closure on the smooth normal passes for every light.
        assert!(shading_side_passes(
            smooth,
            smooth,
            DVec3::new(0.9, 0.0, -0.1)
        ));
    }

    #[test]
    fn diffuse_smoothing_is_conty_s_ggx_shadowing_at_grazing_light_and_one_otherwise() {
        let smooth = DVec3::Z;
        let bent = DVec3::new(0.6, 0.0, 0.8);
        // The light at cos 0.5 from the smooth normal and the bend's cos 0.8: alpha² = (1/0.64 - 1) / 8 = 0.0703125,
        // tan²(light) = 3, Λ = ½(√(1 + 0.2109375) - 1) = 0.0502131, G = 1 / (1 + Λ) = 0.95219.
        let grazing = DVec3::new(0.866_025_403_784_438_6, 0.0, 0.5);
        let smoothing = diffuse_smoothing(smooth, bent, grazing);
        assert!(
            (smoothing - 0.95219).abs() < 1.0e-4,
            "grazing smoothing {smoothing}"
        );
        // The light straight above is not shadowed, and a flat closure is never smoothed.
        assert!((diffuse_smoothing(smooth, bent, DVec3::Z) - 1.0).abs() < 1.0e-12);
        assert_eq!(diffuse_smoothing(smooth, smooth, grazing), 1.0);
    }

    #[test]
    fn a_transmitted_shadow_ray_starts_behind_the_surface_and_a_reflected_one_in_front() {
        // Cycles' `shadow_ray_offset` flips Ng when `dot(N, L) < 0`, and the start lies on the flipped side. A light
        // below a glass plane therefore starts the ray below the plane, so it does not meet its own glass (a rough
        // glass lit from behind is not blocked by itself, as the closure-list review found).
        let point = DVec3::ZERO;
        let normal = DVec3::Z;
        let below = DVec3::new(0.6, 0.0, -0.8);
        let transmitted = shadow_origin(point, below, normal, normal);
        assert!(
            transmitted.z < 0.0,
            "transmitted ray starts at {transmitted:?}"
        );
        // A light above a reflecting plane starts in front of it.
        let above = DVec3::new(0.6, 0.0, 0.8);
        assert!(shadow_origin(point, above, normal, normal).z > 0.0);
    }

    #[test]
    fn a_ray_crossing_the_geometric_plane_starts_in_front_and_meets_its_surface() {
        // The light is behind the geometric normal but in front of a bent shading normal: `dot(N, L) > 0`, so the normal
        // is not flipped, the start lies in front, and the ray crosses the plane, which blocks it.
        let point = DVec3::ZERO;
        let normal = DVec3::Z;
        let bent = DVec3::new(0.9, 0.0, 0.435_889_894_354_067_4).normalize();
        let light = DVec3::new(1.0, 0.0, -0.2).normalize();
        assert!(bent.dot(light) > 0.0 && normal.dot(light) < 0.0);
        let start = shadow_origin(point, light, normal, bent);
        assert!(
            start.z > 0.0 && light.z < 0.0,
            "ray from {start:?} does not cross the plane"
        );
    }
}

#[cfg(test)]
mod band_tests {
    use std::cell::RefCell;

    use serde_json::{Value, json};

    use super::{band_rows, merge_shader_warnings};

    /// Records a warning as a worker's sink does: a warning with a material/input once per pair, any other warning once
    /// per equal value (`layer_index`, `color_layer_missing`).
    fn record(sink: &mut Vec<Value>, warning: Value) {
        let duplicate = match warning["input"].as_str() {
            Some(input) => sink.iter().any(|existing| {
                existing["material"] == warning["material"]
                    && existing["input"].as_str() == Some(input)
            }),
            None => sink.contains(&warning),
        };
        if !duplicate {
            sink.push(warning);
        }
    }

    #[test]
    fn bands_cover_every_row_once_in_order() {
        for height in [1_usize, 2, 7, 64, 192] {
            for band_count in [0_usize, 1, 2, 3, 6, 64, 500] {
                let bands = band_rows(height, band_count);
                assert!(bands.len() <= band_count.max(1));
                let mut next = 0;
                for band in &bands {
                    assert_eq!(band.start, next);
                    assert!(!band.is_empty());
                    next = band.end;
                }
                assert_eq!(next, height, "height {height}, band count {band_count}");
            }
        }
    }

    #[test]
    fn merged_band_warnings_equal_the_single_band_warnings() {
        // A pixel-ordered stream in which each warning recurs in every band, as a shared material does.
        let stream: Vec<Value> = (0..48)
            .map(|index| match index % 4 {
                0 => json!({"code": "UNSUPPORTED_FEATURE", "feature_id": "shader.advanced",
                    "material": "a", "input": "Unsupported Node", "message": "a"}),
                1 => json!({"code": "UNSUPPORTED_FEATURE", "feature_id": "shader.advanced",
                    "material": "b", "input": "Unsupported Node", "message": "b"}),
                2 => json!({"code": "uv.layer_missing", "layer": "UV2",
                    "message": "UV layer `UV2` is not on this mesh; the render layer was used"}),
                _ => json!({"code": "mesh.color_layer_missing", "layer": "Col",
                    "message": "color layer `Col` is not on this mesh; opaque black was used"}),
            })
            .collect();
        let mut single = Vec::new();
        for warning in &stream {
            record(&mut single, warning.clone());
        }
        assert_eq!(
            single.len(),
            4,
            "the single-band render records each distinct warning once"
        );
        for band_count in [1_usize, 2, 3, 5, 7, 48] {
            let sink = RefCell::new(Vec::new());
            for rows in band_rows(stream.len(), band_count) {
                let mut band_sink = Vec::new();
                for warning in &stream[rows] {
                    record(&mut band_sink, warning.clone());
                }
                merge_shader_warnings(&sink, band_sink);
            }
            assert_eq!(sink.into_inner(), single, "band count {band_count}");
        }
    }
}
