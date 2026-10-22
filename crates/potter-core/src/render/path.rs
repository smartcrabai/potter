use std::{
    cell::RefCell,
    collections::BTreeMap,
    ops::{Deref, Range},
    path::Path,
};

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::volume::{
        VolumeData, local_to_index_matrix, sample_density_checked_with, sample_density_with,
        sample_index_density_checked_with, sample_index_density_with,
    },
    image::{ImageData, ImageInterpolation},
    model::{Id, LightType, Material, SceneDoc, VolumeSampling},
    shader::{
        BsdfParams, EvaluatorWorkspace, HitContext, HitUvLayers, PreparedMaterial, ShaderClosure,
        SubsurfaceMethod, SurfaceShading, equirectangular_uv, evaluate_displacement_with_workspace,
        evaluate_surface_with_workspace, heaviest, height_gradient,
    },
};

use super::principled_bsdf::{
    GlassSample, Layers, base_color, continuous_density, diffuse_coefficient,
    evaluate as evaluate_bsdf, layers_at, lobe_probabilities, make_orthonormals, sample_glass,
    sheen_coefficient, specular_eta, subsurface_closure_weight,
};
use super::subsurface::{
    MAX_HITS, ProbeHits, SubsurfaceExit, burley_radius, channel_mask, eval as burley_eval,
    exit_params, pdf_mean, pick_channel, raw_radius, sample_radius,
};
use super::subsurface_walk::{
    WalkEntry, WalkMethod, entry_direction, skin_entry_direction, walk, walk_anisotropy,
    walk_radius, walk_uses_van_de_hulst,
};
use super::{
    bvh::{Bvh, BvhHit},
    light_model::{
        AreaShape, Spread, area_radiance, area_shape, emitter_radiance, influence_fade,
        inside_emitter, local_light_influence_radius_squared, point_intensity, spot_attenuation,
        spread_attenuation, spread_region,
    },
    raster::{
        Camera, CameraFrame, Mode, RasterOutput, StereoMode, Triangle, blend_corner_normals,
        camera_eye, camera_frame, rasterize_with_lines, ray_for_sample, ray_for_sample_in_frame,
        smooth_shadow_origin,
    },
    scene::{
        ElementRecord, Geometry, HitPose, LayerTables, ObjectRecord, layered_context, object_space,
    },
};

const RAY_EPSILON: f64 = 1.0e-5;
const MAX_VOLUME_STEPS: usize = 2048;
/// Cycles' `VOLUME_BOUNDS_MAX` (`intern/cycles/kernel/types.h`): volume-only boundary crossings
/// have their own counter, and the path ends past this many.
const VOLUME_BOUNDS_MAX: u32 = 1024;

#[derive(Clone, Copy)]
struct PointLight {
    position: DVec3,
    direction: DVec3,
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

    /// Whether a BSDF-sampled ray can hit this light (Cycles' `SHADER_USE_MIS`, set by
    /// `AreaLight::copy_to_kernel` for an emitter with area and a spread above zero, and by
    /// `PointLight::copy_to_kernel` for a radius). Zero-radius point and spot lights are delta
    /// lights, and the sun is never hit.
    fn hittable(&self) -> bool {
        self.use_mis
            && match self.kind {
                LightType::Area => {
                    self.area.is_some_and(|shape| shape.area > 0.0)
                        && self.spread.is_none_or(|spread| !spread.is_zero())
                }
                LightType::Point | LightType::Spot => self.radius > 0.0,
                LightType::Sun => false,
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
    /// Channel-mean σt per unit grid density (absolute, as `BsdfParams`'s coefficients).
    extinction_scale: f64,
    /// Blender's `cycles.volume_sampling` of the volume's material.
    sampling: crate::model::VolumeSampling,
    /// Blender's `cycles.volume_interpolation` of the volume's material.
    interpolation: crate::model::VolumeInterpolation,
    /// The inverse of an inline grid's index-to-local matrix, built once per volume (`local_to_index_matrix`). `None` for a
    /// decoded VDB, which samples at its local position.
    index_from_local: Option<DMat4>,
}

#[derive(Clone, Copy)]
struct VolumeEvent {
    point: DVec3,
    albedo: DVec3,
}

struct VolumeSegment {
    transmittance: f64,
    event: Option<VolumeEvent>,
    stochastic: bool,
}

/// One closure of a hit, weighted as Cycles weights closures (`closure_alloc`): `weight` is the share of the hit the
/// closure takes, and `params` is the closure's own shading, with its alpha after the material's alpha mode, its own
/// normal, and its layers set up for the hit's view.
#[derive(Clone, Copy)]
struct HitClosure {
    weight: f64,
    params: BsdfParams,
    layers: Layers,
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

    /// Selection weight of the pass-through share, `w · (1 − α) · mean(tint)` (Cycles' sample weight of its Transparent
    /// closure, `bsdf_transparent_setup`).
    fn pass_weight(&self) -> f64 {
        self.weight * (1.0 - self.params.alpha) * transparent_tint(&self.params).element_sum() / 3.0
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

    /// The throughput of a pass through `closure`: its tint times `S / mean(tint)`. The pick probability
    /// `w·(1−α)·mean(tint) / S` then leaves the expected throughput `w·(1−α)·tint`.
    fn pass_multiplier(&self, closure: &HitClosure) -> DVec3 {
        let tint = transparent_tint(&closure.params);
        tint * (self.selection_total / (tint.element_sum() / 3.0))
    }

    /// The reflective share at `light`: `Σ w·α·f·|cos|` over the closures, each with its own lobes, normal and cosine
    /// (Cycles' `surface_shader_bsdf_eval`). `sampled` is the closure a draw picked and whether that draw was diffuse. Cycles'
    /// `bsdf_sample` evaluates the picked closure with `is_eval = false`, so only its diffuse draw keeps the shading-side
    /// test; every other closure takes it (`bump_shadowing_term`, `kernel/closure/bsdf.h`).
    fn bsdf_eval(&self, light: DVec3, sampled: Option<(usize, bool)>) -> DVec3 {
        self.items
            .iter()
            .enumerate()
            .fold(DVec3::ZERO, |sum, (index, closure)| {
                let is_eval = match sampled {
                    Some((picked, diffuse)) if picked == index => diffuse,
                    _ => true,
                };
                sum + self.closure_eval(closure, light, is_eval)
            })
    }

    /// One closure's reflective term at `light`, `w·α·f·|cos|`, with its diffuse and sheen lobes smoothed at grazing light.
    fn closure_eval(&self, closure: &HitClosure, light: DVec3, is_eval: bool) -> DVec3 {
        let reflect = closure.reflect_weight();
        if reflect <= 0.0 {
            return DVec3::ZERO;
        }
        let (params, layers) = (&closure.params, &closure.layers);
        if is_eval && !shading_side_passes(self.smooth, layers.normal, light) {
            return DVec3::ZERO;
        }
        let cosine = layers.normal.dot(light).abs();
        // The diffuse and sheen lobes sit in the above-shading share of `evaluate`, so their smoothing applies to that
        // share only; the specular and glass lobes take none.
        let above = layers.normal.dot(light) > 0.0;
        let diffuse = if above {
            diffuse_coefficient(params, layers)
        } else {
            DVec3::ZERO
        };
        let sheen = if above {
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
        let shaded = evaluate_bsdf(params, layers, light) + diffuse_correction + sheen_correction;
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
            LobeDraw::Delta { direction, weight } => Reflection::Delta {
                direction,
                weight: weight * self.selection_total,
            },
            LobeDraw::Subsurface { weight, method } => Reflection::Subsurface {
                weight: weight * self.selection_total,
                method,
            },
            LobeDraw::Continuous { direction, diffuse } => {
                let pdf = self.continuous_pdf(direction);
                Reflection::Continuous {
                    direction,
                    weight: self.bsdf_eval(direction, Some((index, diffuse))) / pdf.max(1.0e-8),
                    pdf,
                    diffuse,
                }
            }
        }
    }

    /// Cycles' falling back to diffuse after a diffuse bounce (`bssrdf_setup` with `PATH_RAY_DIFFUSE_ANCESTOR`): a
    /// Burley closure loses its subsurface weight. The random walk keeps its closure.
    fn disable_burley_subsurface(&mut self) {
        let burley = |closure: &HitClosure| {
            closure.params.subsurface_method == SubsurfaceMethod::Burley
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
                if params.subsurface_method == SubsurfaceMethod::Burley {
                    params.subsurface_weight = 0.0;
                }
                HitClosure::new(
                    closure.weight,
                    &params,
                    closure.layers.view,
                    self.geometric,
                    self.bump_correction,
                )
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
    object_index: u32,
    element_index: u32,
    /// The triangle the hit lies on, which a random walk entering through it must not hit first (`ray.self.prim`).
    triangle_index: usize,
}

/// The result of a reflective draw of the closure set. `Continuous` carries its throughput weight, the mixture density
/// that MIS weighs light sampling against, and whether it is a diffuse draw (a Burley subsurface closure falls back to
/// diffuse after one). `Delta` is a sharp glass direction. `Subsurface` enters the surface through the subsurface lobe of
/// its method; the path leaves at that method's exit. `Absorbed` ends the path.
enum Reflection {
    Continuous {
        direction: DVec3,
        weight: DVec3,
        pdf: f64,
        diffuse: bool,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
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
/// `light_eval_from_intersection`. `radiance` is before the MIS weight.
#[derive(Clone, Copy)]
struct LightHit {
    distance: f64,
    radiance: DVec3,
    pdf: f64,
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
const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

/// The read-only geometry a path trace reads. It has no interior mutability, so every worker shares it.
#[derive(Clone, Copy)]
struct SharedGeometry<'a> {
    triangles: &'a [Triangle],
    materials: &'a [Material],
    prepared: &'a [PreparedMaterial],
    objects: &'a [ObjectRecord],
    elements: &'a [ElementRecord],
}

impl<'a> SharedGeometry<'a> {
    fn of(geometry: &'a Geometry) -> Self {
        Self {
            triangles: geometry.triangles.as_slice(),
            materials: geometry.materials.as_slice(),
            prepared: geometry.prepared.as_slice(),
            objects: geometry.objects.as_slice(),
            elements: geometry.elements.as_slice(),
        }
    }
}

/// A trace's view of the geometry: the shared part, and the sink its shading warnings are recorded in. Each worker
/// records into its own sink, and the sinks are merged in pixel order once the workers finish.
#[derive(Clone, Copy)]
struct TraceGeometry<'a> {
    shared: SharedGeometry<'a>,
    warnings: &'a RefCell<Vec<serde_json::Value>>,
}

impl<'a> TraceGeometry<'a> {
    fn of(geometry: &'a Geometry) -> Self {
        Self {
            shared: SharedGeometry::of(geometry),
            warnings: &geometry.shader_warnings,
        }
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
    images: &'a BTreeMap<String, ImageData>,
    importance: Option<&'a WorldImportanceSampler>,
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
                let sample_x = rng.next_f64();
                let sample_y = rng.next_f64();
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
                    // A sample with no ray (outside a fisheye's circle, or past its field of view) is opaque black: Blender's
                    // film writes a pixel with no ray as black with alpha 1 under either film (`probe_transparent_film_fisheye.py`).
                    let (left, left_alpha) = match left_ray {
                        Some((left_origin, left_direction)) => trace_path(
                            scene,
                            bounces,
                            left_origin,
                            left_direction,
                            job.environment,
                            &mut left_rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    };
                    let (right, right_alpha) = match right_ray {
                        Some((right_origin, right_direction)) => trace_path(
                            scene,
                            bounces,
                            right_origin,
                            right_direction,
                            job.environment,
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
                    match ray_for_sample_in_frame(
                        frame, camera, job.width, job.height, x, y, sample_x, sample_y,
                    )? {
                        Some((origin, direction)) => trace_path(
                            scene,
                            bounces,
                            origin,
                            direction,
                            job.environment,
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
            rgba[index * 4..index * 4 + 4].copy_from_slice(&[
                to_srgb_byte(color.x),
                to_srgb_byte(color.y),
                to_srgb_byte(color.z),
                (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
    }
    Ok(())
}

/// The contiguous row bands that split `height` rows across at most `band_count` workers, in row order. Each band has
/// `ceil(height / band_count)` rows except the last, and no band is empty.
fn band_rows(height: usize, band_count: usize) -> Vec<Range<usize>> {
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
fn merge_shader_warnings(sink: &RefCell<Vec<serde_json::Value>>, warnings: Vec<serde_json::Value>) {
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
    let volumes = collect_volumes(
        doc,
        snapshot,
        scene.unit.scale_length,
        images,
        &geometry.shader_warnings,
    )?;
    let world_importance = WorldImportanceSampler::build(doc, &snapshot.scene_id, images)?;
    let mut output = RasterOutput {
        rgba: allocate(rgba_len, 0_u8, "RGBA image")?,
        linear_rgba: allocate(pixel_count, [0.0_f32; 4], "linear RGBA image")?,
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
            images,
            importance: world_importance.as_ref(),
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
pub(super) fn composite_realtime_volumes(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    camera: &Camera,
    width: u32,
    height: u32,
    seed: u32,
    film_transparent: bool,
    images: &BTreeMap<String, ImageData>,
    output: &mut RasterOutput,
    warnings: &RefCell<Vec<serde_json::Value>>,
) -> Result<()> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let volumes = collect_volumes(doc, snapshot, scene.unit.scale_length, images, warnings)?;
    if volumes.is_empty() {
        return Ok(());
    }
    let lights = collect_lights(doc, snapshot);
    let fallback = scene
        .world
        .as_ref()
        .and_then(|world_id| doc.worlds.get(world_id))
        .map_or(DVec3::ZERO, |world| {
            DVec3::from_array(world.color) * world.strength
        });
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    for y in 0..height {
        let row = usize::try_from(y)
            .ok()
            .and_then(|row| row.checked_mul(width_usize))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "pixel row exceeds platform limits",
                )
            })?;
        for x in 0..width {
            let column = usize::try_from(x).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "pixel column exceeds platform limits",
                )
            })?;
            let index = row.checked_add(column).ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "pixel index exceeds platform limits",
                )
            })?;
            let Some((origin, direction)) = ray_for_sample(camera, width, height, x, y, 0.5, 0.5)?
            else {
                continue;
            };
            let max_distance = if output.ids[index] == 0 {
                f64::INFINITY
            } else {
                let projected_depth = f64::from(output.depths[index]);
                camera.ray_distance_for_depth(direction, projected_depth)
            };
            let mut rng = Pcg32::for_sample(seed, x, y, 0);
            let (volume_radiance, transmittance) = realtime_volume_integral(
                doc,
                snapshot,
                images,
                &lights,
                &volumes,
                origin,
                direction,
                max_distance,
                fallback,
                &mut rng,
            )?;
            if transmittance >= 1.0 && volume_radiance.max_element() <= 0.0 {
                continue;
            }
            let previous = output.linear_rgba[index];
            let red = f64::from(previous[0]) * transmittance + volume_radiance.x;
            let green = f64::from(previous[1]) * transmittance + volume_radiance.y;
            let blue = f64::from(previous[2]) * transmittance + volume_radiance.z;
            let alpha = if film_transparent {
                1.0 - transmittance + f64::from(previous[3]) * transmittance
            } else {
                1.0
            };
            output.linear_rgba[index] = [
                finite_f32(red),
                finite_f32(green),
                finite_f32(blue),
                finite_f32(alpha),
            ];
            let byte_index = index * 4;
            output.rgba[byte_index..byte_index + 4].copy_from_slice(&[
                to_srgb_byte(red),
                to_srgb_byte(green),
                to_srgb_byte(blue),
                (alpha * 255.0).round() as u8,
            ]);
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the realtime volume integrator needs explicit world and scene state"
)]
fn realtime_volume_integral(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    images: &BTreeMap<String, ImageData>,
    lights: &[PointLight],
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    environment: DVec3,
    rng: &mut Pcg32,
) -> Result<(DVec3, f64)> {
    let mut entry = f64::INFINITY;
    let mut exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    for instance in volumes {
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
            minimum_step = minimum_step
                .min(instance.voxel_size / local_direction.length().max(f64::MIN_POSITIVE) * 0.5);
        }
    }
    if !entry.is_finite() || exit <= entry {
        return Ok((DVec3::ZERO, 1.0));
    }
    let segment_length = exit - entry;
    let step_count = (segment_length / minimum_step).ceil().clamp(1.0, 64.0) as usize;
    let step = segment_length / step_count as f64;
    let meters_per_unit = volumes
        .first()
        .map_or(1.0, |instance| instance.meters_per_unit);
    let ambient = world_radiance(doc, &snapshot.scene_id, images, -direction, environment)?;
    let mut transmittance = 1.0;
    let mut radiance = DVec3::ZERO;
    for index in 0..step_count {
        let distance = entry + (index as f64 + 0.5) * step;
        let point = origin + direction * distance;
        let mut extinction = 0.0;
        let mut source = DVec3::ZERO;
        for instance in volumes {
            let density = volume_density_at(instance, point)?;
            let extinction_coefficient = volume_extinction_density(instance, density);
            let scattering_coefficient = if instance.scattering {
                extinction_coefficient
            } else {
                0.0
            };
            extinction += extinction_coefficient;
            source += instance.emission;
            if instance.scattering {
                source += ambient * instance.albedo * scattering_coefficient;
                for light in lights {
                    // EEVEE's volume in-scatter of a light is its power per class, `volume_factor`, times its influence.
                    let gain = light.volume_factor * light.volume_fade(point);
                    if gain <= 0.0 {
                        continue;
                    }
                    let Some(sample) = sample_light(light, point, rng) else {
                        continue;
                    };
                    let visibility = march_volumes_with_budget(
                        volumes,
                        point,
                        sample.direction,
                        sample.distance,
                        Some(rng),
                        16,
                    )?;
                    let phase = instance.phase_density(direction.dot(sample.direction));
                    source += instance.albedo
                        * sample.radiance
                        * (scattering_coefficient * phase * visibility * gain);
                }
            }
        }
        let step_in_meters = step * meters_per_unit;
        let (attenuation, source_distance) =
            homogeneous_segment_integral(extinction, step_in_meters);
        radiance += source * (transmittance * source_distance);
        transmittance *= attenuation;
    }
    Ok((radiance, transmittance.clamp(0.0, 1.0)))
}

fn world_radiance(
    doc: &SceneDoc,
    scene_id: &Id,
    images: &BTreeMap<String, ImageData>,
    direction: DVec3,
    fallback: DVec3,
) -> Result<DVec3> {
    let Some(world) = doc
        .scenes
        .get(scene_id)
        .and_then(|scene| scene.world.as_ref())
        .and_then(|world_id| doc.worlds.get(world_id))
    else {
        return Ok(fallback);
    };
    crate::shader::evaluate_world(world, &doc.node_groups, images, direction)
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
}

/// A volume scatter's receiver: its point, the incoming direction, its phase function, the per-light weights of the direct
/// light, and the environment along the incoming direction.
struct VolumeSite<'a> {
    point: DVec3,
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
fn mixed_scatter_pdf(stack: &BsdfParams, grid: &VoxelGrid, distance: f64) -> f64 {
    distance_scatter_pdf(stack, distance) * (-grid_depth(grid, distance)).exp()
        + grid_scatter_pdf(grid, distance) * (-mean_extinction(stack) * distance).exp()
}

/// The density per unit distance of a scatter event in a segment's media, which the distance sampler realises: the MIS
/// partner of the equiangular samples and the density of distance-sampled events.
fn segment_scatter_pdf(grid: Option<&VoxelGrid>, stack: Option<&BsdfParams>, distance: f64) -> f64 {
    match (grid, stack) {
        (Some(grid), Some(stack)) => mixed_scatter_pdf(stack, grid, distance),
        (Some(grid), None) => grid_scatter_pdf(grid, distance),
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
    let voxel = grid.map_or(1.0, |grid| (-grid_depth(grid, distance)).exp());
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

/// The volume and transport state of one path: the scene, the bounce budgets and the sampling state.
fn trace_path(
    scene: SceneRefs<'_, '_>,
    bounces: Bounces,
    mut origin: DVec3,
    mut direction: DVec3,
    environment: DVec3,
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
    let mut bounce = 0u32;
    let mut transparent_bounce = 0u32;
    // Cycles' `transparent_max_bounce` is at least one: the first pass-through is always taken
    // (`intern/cycles/kernel/integrator/path_state.h`, the FIXME at `path_state_next`).
    let transparent_cap = transparent_max_bounces.max(1);
    let mut volume_bounds = 0u32;
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
    let mut media = camera_media(doc, geometry, bvh, origin, images, evaluator_workspace)?;
    loop {
        let hit = match exit_hit.take() {
            Some(exit) => Some(exit),
            None => closest_hit(geometry, bvh, origin, direction, RAY_EPSILON, f64::INFINITY),
        };
        let surface_distance = hit.map_or(f64::INFINITY, |hit| hit.distance);
        // Camera rays do not see lights, and a volume scatter is light-sampled only. A surface
        // vertex's light hit needs its light sample as MIS partner, which a volume scatter on the
        // way would drop, so a scene with volumes does not count those hits.
        let receiver = match previous {
            Scatter::Surface(vertex) if volumes.is_empty() => Some(vertex.point),
            Scatter::Delta(point) => Some(point),
            Scatter::Camera | Scatter::Surface(_) | Scatter::Volume(_) => None,
        };
        let light = receiver.and_then(|receiver| {
            nearest_light_hit(lights, origin, direction, receiver, surface_distance)
        });
        let segment_limit = light.map_or(surface_distance, |light| light.distance);
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
            voxel_grid(volumes, origin, direction, segment_limit)?
        };
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
                            segment_scatter_pdf(grid.as_ref(), stack.as_ref(), distance),
                        )
                    }
                };
                let sample_point = origin + direction * distance;
                let (nee, light_direction) = volume_light_nee(
                    scene,
                    &media,
                    light,
                    sample_point,
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
            // The grid's stochastic free flight and the mesh's analog collision are sampled independently, and the nearer
            // event is taken. The grid's emission runs up to that event, weighted by the mesh's chromatic excess, before the
            // throughput takes any transmittance (`emission_until`). Analog sampling is exact for the event, so the
            // mesh's chromatic transmittance is its only extra weight (`chromatic_correction`).
            let voxel = trace_volume_segment(Some(grid), volumes, origin, direction, rng, true)?;
            let voxel_distance = voxel
                .event
                .map(|event| (event.point - origin).dot(direction));
            let limit = voxel_distance.unwrap_or(segment_limit);
            let collision = sample_mesh_collision(&medium, limit, rng);
            let emission_end = collision.unwrap_or(limit);
            radiance += throughput
                * emission_until(
                    Some(grid),
                    voxel.stochastic,
                    Some(&medium),
                    true,
                    emission_end,
                );
            if let Some(distance) = collision {
                if !voxel.stochastic {
                    throughput *= (-grid_depth(grid, distance)).exp();
                }
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
                    albedo: DVec3::ZERO,
                };
                let weights = event_weights(
                    &choices,
                    method,
                    origin,
                    direction,
                    segment_limit,
                    distance,
                    segment_scatter_pdf(Some(grid), Some(&medium), distance),
                );
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
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
                previous = Scatter::Volume(phase_pdf);
                bounce += 1;
                continue;
            }
            // No mesh collision before `limit`: the mesh's chromatic transmittance up to there, then the grid's own event
            // or the segment's end.
            throughput *= chromatic_correction(&medium, limit);
            if !voxel.stochastic {
                throughput *= voxel.transmittance;
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
            if let Some(event) = voxel.event {
                let scatter_probability = (event.albedo.x + event.albedo.y + event.albedo.z) / 3.0;
                if scatter_probability <= 0.0 || rng.next_f64() >= scatter_probability {
                    break;
                }
                throughput *= event.albedo / scatter_probability;
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
                    segment_scatter_pdf(Some(grid), Some(&medium), event_distance),
                );
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
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
                previous = Scatter::Volume(phase_pdf);
                bounce += 1;
                continue;
            }
        } else if terminal {
            // The path's last scatter has taken its direct light: this ray only attenuates through the volumes it crosses.
            radiance += throughput
                * emission_until(grid.as_ref(), false, stack.as_ref(), false, segment_limit);
            if let Some(grid) = grid.as_ref() {
                let segment =
                    trace_volume_segment(Some(grid), volumes, origin, direction, rng, false)?;
                throughput *= segment.transmittance;
            }
            if let Some(medium) = stack {
                throughput *= medium_transmittance(&medium, segment_limit);
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
        } else if let (Some(grid), None) = (grid.as_ref(), stack) {
            let segment = trace_volume_segment(Some(grid), volumes, origin, direction, rng, true)?;
            let emission_end = segment
                .event
                .map_or(segment_limit, |event| (event.point - origin).dot(direction));
            radiance += throughput
                * emission_until(Some(grid), segment.stochastic, None, false, emission_end);
            if !segment.stochastic {
                throughput *= segment.transmittance;
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
            if let Some(event) = segment.event {
                let scatter_probability = (event.albedo.x + event.albedo.y + event.albedo.z) / 3.0;
                if scatter_probability <= 0.0 || rng.next_f64() >= scatter_probability {
                    break;
                }
                throughput *= event.albedo / scatter_probability;
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
                    segment_scatter_pdf(Some(grid), None, event_distance),
                );
                radiance += throughput
                    * direct_volume_lighting(
                        scene,
                        &media,
                        &VolumeSite {
                            point: event.point,
                            incoming: direction,
                            phase: &phase,
                            light_weights: &weights,
                            environment: world_radiance(
                                doc,
                                &snapshot.scene_id,
                                images,
                                direction,
                                environment,
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
                previous = Scatter::Volume(phase_pdf);
                bounce += 1;
                continue;
            }
        } else if let Some(medium) = stack {
            // The stack changes only at surfaces, so the stack's combined medium spans this whole segment, and its free
            // flight is spectral ratio tracking (`trace_mesh_medium`). A scatter picks its closure by scatter weight.
            // A mesh medium's emission is integrated in closed form over the whole segment, at the throughput the path
            // carries into it, independent of where it scatters (`volume_integrate_homogeneous`).
            radiance +=
                throughput * emission_until(None, false, Some(&medium), false, segment_limit);
            match trace_mesh_medium(&medium, segment_limit, &mut throughput, rng) {
                MediumOutcome::Scatter(distance) => {
                    let phase = event_phase(&media, None, distance);
                    let Some(anisotropy) = phase.pick(rng) else {
                        break;
                    };
                    let event = VolumeEvent {
                        point: origin + direction * distance,
                        albedo: DVec3::ZERO,
                    };
                    let weights = event_weights(
                        &choices,
                        method,
                        origin,
                        direction,
                        segment_limit,
                        distance,
                        segment_scatter_pdf(None, Some(&medium), distance),
                    );
                    radiance += throughput
                        * direct_volume_lighting(
                            scene,
                            &media,
                            &VolumeSite {
                                point: event.point,
                                incoming: direction,
                                phase: &phase,
                                light_weights: &weights,
                                environment: world_radiance(
                                    doc,
                                    &snapshot.scene_id,
                                    images,
                                    direction,
                                    environment,
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
                    previous = Scatter::Volume(phase_pdf);
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
            radiance += throughput * light.radiance * weight;
            origin += direction * (light.distance + RAY_EPSILON);
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
            let environment =
                world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
            radiance += throughput * environment * environment_weight;
            escaped = true;
            break;
        };
        let mut surface = make_surface(
            doc,
            geometry,
            hit,
            origin,
            direction,
            images,
            evaluator_workspace,
        )?;
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
        } else if diffuse_ancestor {
            // A Burley closure falls back to diffuse after a diffuse bounce (Cycles' `bssrdf_setup` with
            // `PATH_RAY_DIFFUSE_ANCESTOR`). The random walk keeps its closure.
            surface.closures.disable_burley_subsurface();
        }
        if !surface.media.is_empty() {
            // A medium surface passes the path and changes the volume stack: each of its volume closures enters the
            // medium of its object, or leaves the one it entered.
            for medium in &surface.media {
                toggle_mesh_medium(&mut media, medium);
            }
            volume_bounds += 1;
            if volume_bounds > VOLUME_BOUNDS_MAX {
                break;
            }
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
                    throughput = exit.throughput;
                    origin = exit.origin;
                    direction = exit.direction;
                    exit_hit = Some(exit.hit);
                    exit_entry = Some(entry.params);
                    previous = Scatter::Camera;
                    continue;
                }
                if bounce >= max_bounces {
                    break;
                }
                let (next_direction, weight) = match draw {
                    Reflection::Continuous {
                        direction,
                        weight,
                        pdf,
                        diffuse,
                    } => {
                        diffuse_ancestor |= diffuse;
                        previous = Scatter::Surface(MisVertex {
                            bsdf_pdf: pdf,
                            normal: surface.normal,
                            point: surface.point,
                        });
                        (direction, weight)
                    }
                    Reflection::Delta { direction, weight } => {
                        previous = Scatter::Delta(surface.point);
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
    Ok((radiance.max(DVec3::ZERO), alpha.clamp(0.0, 1.0)))
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
            let Some((origin, direction)) = ray_for_sample(camera, width, height, x, y, 0.5, 0.5)?
            else {
                continue;
            };
            let hit = closest_hit(geometry, bvh, origin, direction, RAY_EPSILON, f64::INFINITY);
            let segment_limit = hit.map_or(f64::INFINITY, |hit| hit.distance);
            let volume_transmittance =
                march_volumes(volumes, origin, direction, segment_limit, None)?;
            let Some(hit) = hit else {
                if film_transparent {
                    let alpha = 1.0 - volume_transmittance;
                    output.linear_rgba[index][3] = finite_f32(alpha);
                    output.rgba[index * 4 + 3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                } else {
                    output.linear_rgba[index][3] = 1.0;
                    output.rgba[index * 4 + 3] = 255;
                }
                continue;
            };
            let surface = make_surface(
                doc,
                geometry,
                hit,
                origin,
                direction,
                images,
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
                surface.point,
                surface.geometric_normal,
                surface.normal,
                1.0,
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
                let alpha = coverage + (1.0 - coverage) * (1.0 - volume_transmittance);
                output.linear_rgba[index][3] = finite_f32(alpha);
                output.rgba[index * 4 + 3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    Ok(())
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
        1,
        seed,
        None,
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
        let destination_alpha = f64::from(destination[3]).clamp(0.0, 1.0);
        let inverse_alpha = 1.0 - source_alpha;
        let color = [
            f64::from(source[0]) * source_alpha + f64::from(destination[0]) * inverse_alpha,
            f64::from(source[1]) * source_alpha + f64::from(destination[1]) * inverse_alpha,
            f64::from(source[2]) * source_alpha + f64::from(destination[2]) * inverse_alpha,
        ];
        let alpha = if film_transparent {
            source_alpha + destination_alpha * inverse_alpha
        } else {
            1.0
        };
        if let Some(pixel) = output.linear_rgba.get_mut(index) {
            *pixel = [
                finite_f32(color[0]),
                finite_f32(color[1]),
                finite_f32(color[2]),
                finite_f32(alpha),
            ];
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
            *albedo = [source[0], source[1], source[2], source_alpha as f32];
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
            bytes.copy_from_slice(&[
                to_srgb_byte(color[0]),
                to_srgb_byte(color[1]),
                to_srgb_byte(color[2]),
                (alpha * 255.0).round() as u8,
            ]);
        }
    }
    Ok(())
}

fn closest_hit(
    geometry: &TraceGeometry<'_>,
    bvh: &Bvh,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
) -> Option<BvhHit> {
    bvh.closest_hit_two_sided(geometry.triangles, origin, direction, t_min, t_max)
}

/// The shading of one hit for the transparent pass of a realtime shadow ray (`render/scene.rs`). Each closure carries its
/// alpha after the material's alpha mode, as the path tracer's closures do, so a realtime shadow and a path-traced one
/// read the same transparency. The pass reads neither the displacement gradient nor the layers, so only the material
/// evaluation runs.
pub(super) fn hit_params(
    doc: &SceneDoc,
    geometry: &Geometry,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<SurfaceShading> {
    with_hit_frame(
        doc,
        &TraceGeometry::of(geometry),
        hit,
        origin,
        direction,
        images,
        |frame| {
            let mut shading = evaluate_surface_with_workspace(
                frame.material,
                frame.prepared,
                &frame.context,
                evaluator_workspace,
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
    let context = layered_context(
        doc,
        images,
        geometry.warnings,
        object,
        triangle,
        HitPose {
            point,
            object: object_space(object, point),
            normal,
            view: -direction,
            barycentric: hit.barycentric,
        },
        None,
        uv_layers,
        &mut tables,
    )?;
    body(&HitFrame {
        point,
        shadow_point,
        back_facing,
        geometric_normal,
        normal,
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

/// The media a hit's Volume output bounds: one per volume closure that scatters or absorbs light, its coefficients
/// weighted by the closure's weight, since Cycles sums the closures' coefficients (`volume_shader_eval`).
fn surface_media(object_index: u32, volume: &[ShaderClosure]) -> Vec<MeshMedium> {
    volume
        .iter()
        .enumerate()
        .filter_map(|(closure, shader)| {
            let weight = shader.weight;
            let mut params = shader.params;
            params.volume_scatter = params.volume_scatter.map(|value| weight * value);
            params.volume_absorption = params.volume_absorption.map(|value| weight * value);
            params.volume_emission = params.volume_emission.map(|value| weight * value);
            params.has_medium().then_some(MeshMedium {
                object_index,
                closure,
                params,
            })
        })
        .collect()
}

fn make_surface(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<Surface> {
    let triangle_index = hit.triangle_index;
    with_hit_frame(doc, geometry, hit, origin, direction, images, |frame| {
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
                evaluate_displacement_with_workspace(
                    frame.material,
                    frame.prepared,
                    sample,
                    evaluator_workspace,
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
            items.push(HitClosure::new(
                closure.weight,
                &params,
                -direction,
                frame.geometric_normal,
                frame.material.use_bump_map_correction,
            ));
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
            media: surface_media(frame.triangle.object_index, &shading.volume),
            object_index: frame.triangle.object_index,
            element_index: frame.triangle.element_index,
            triangle_index,
        })
    })
}

/// The media and shadow transparency of one hit, which is all a shadow ray or the camera's medium probe reads from it.
/// Neither depends on the normal, so the displacement gradient is not evaluated here. Consequently a warning, or an error,
/// that only the gradient's samples produce is not raised from a shadow ray; path and camera hits still raise it.
fn shadow_hit(
    doc: &SceneDoc,
    geometry: &TraceGeometry<'_>,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(Vec<MeshMedium>, DVec3)> {
    with_hit_frame(doc, geometry, hit, origin, direction, images, |frame| {
        let shading = evaluate_surface_with_workspace(
            frame.material,
            frame.prepared,
            &frame.context,
            evaluator_workspace,
        )?;
        // The closures `make_surface` keeps (weight above zero), with alpha after the material's alpha mode, as `Closures::new`
        // reads them.
        let transparency = shadow_transparency(
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
        Ok((
            surface_media(frame.triangle.object_index, &shading.volume),
            transparency,
        ))
    })
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
        // Every closure's own lobes and cosine (`Closures::bsdf_eval`). Below the surface only a glass lobe
        // contributes, so a direction that no closure reaches skips the shadow ray.
        let bsdf = closures.bsdf_eval(sample.direction, None);
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
        closures.bsdf_eval(direction, None)
    } else {
        DVec3::ZERO
    };
    if bsdf.max_element() > 0.0 {
        let environment_sample =
            world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        let visibility = shadow_transmittance(
            scene,
            media,
            ShadowRay {
                point: surface.shadow_point,
                normal: geometric,
                shading: closures.smooth,
                direction,
                distance: f64::INFINITY,
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
        let weight = site.light_weights.get(index).copied().unwrap_or(1.0);
        if weight <= 0.0 {
            continue;
        }
        let (radiance, direction) = volume_light_nee(
            scene,
            media,
            light,
            site.point,
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
    } = ray;
    let mut transparent_hits = 0u32;
    let mut blocked = false;
    let mut failure = None;
    // The shadow ray starts in the receiver's volume stack and crosses each medium by its transmittance.
    let mut shadow_media = media.to_vec();
    let mut transmittance = DVec3::ONE;
    let remaining = if distance.is_finite() {
        (distance - RAY_EPSILON).max(0.0)
    } else {
        f64::INFINITY
    };
    let (origin, remaining) = bvh.walk_hits(
        geometry.triangles,
        shadow_origin(point, direction, normal, shading),
        direction,
        RAY_EPSILON,
        remaining,
        |hit, origin| {
            let (media, transparency) = match shadow_hit(
                doc,
                geometry,
                hit,
                origin,
                direction,
                images,
                evaluator_workspace,
            ) {
                Ok(shadow) => shadow,
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            };
            if let Some(stack) = stack_medium(&shadow_media) {
                transmittance *= medium_transmittance(&stack, hit.distance);
            }
            if !media.is_empty() {
                // A medium boundary passes the shadow ray and changes the volume stack, as it does for a path. It is not a
                // transparent surface, so it does not count against the budget, and its opaque alpha does not block.
                for medium in &media {
                    toggle_mesh_medium(&mut shadow_media, medium);
                }
                return true;
            }
            // Cycles counts each transparent surface a shadow ray crosses against the path's remaining budget and blocks
            // the ray past it (`intern/cycles/kernel/bvh/intersect_filter.h`).
            transparent_hits += 1;
            if transparent_hits > transparent_budget {
                blocked = true;
                return false;
            }
            // Cycles' shadow transparency is the closures' transparent extinction, `Σ w · (1 − α) · tint`
            // (`kernel/integrator/surface_shader.h`, `surface_shader_transparency`). A glass or other transmissive
            // closure has none, so it blocks the shadow ray: its shader lacks `SD_HAS_TRANSPARENT_SHADOW`
            // (`scene/shader.cpp`, `kernel/bvh/intersect_filter.h`).
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
    if let Some(stack) = stack_medium(&shadow_media) {
        transmittance *= medium_transmittance(&stack, remaining);
    }
    let volume_transmittance = march_volumes(volumes, origin, direction, remaining, Some(rng))?;
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
) -> Option<LightHit> {
    let mut nearest: Option<LightHit> = None;
    for light in lights.iter().filter(|light| light.hittable()) {
        let closest = nearest.map_or(limit, |hit| hit.distance);
        if let Some((distance, radiance, pdf)) =
            light_hit(light, origin, direction, receiver, closest)
        {
            nearest = Some(LightHit {
                distance,
                radiance,
                pdf,
            });
        }
    }
    nearest
}

/// Intersects a ray with a hittable light, as Cycles' `area_light_intersect`, `point_light_intersect`
/// and `spot_light_intersect`. Returns the distance, the emitter radiance, and the solid-angle
/// density that `sample_light` gives this direction from `receiver`. Area lights and the disk of a
/// soft-falloff point or spot light are one-sided (they emit towards the receiver); a sphere emits
/// from every side.
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
        let distance = (origin - light.position).dot(light.direction) / cos_light;
        if !(distance > RAY_EPSILON && distance < limit) {
            return None;
        }
        let hit = origin + direction * distance;
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
        let distance = (origin - light.position).dot(normal) / cos_light;
        if !(distance > RAY_EPSILON && distance < limit) {
            return None;
        }
        let hit = origin + direction * distance - light.position;
        if hit.length_squared() > radius_squared {
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
    /// The camera ray. Lights are not visible to camera rays.
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
/// draw of that closure, flagged diffuse for a diffuse-branch draw. `Delta` is a singular glass direction with its weight
/// over its branch probability. `Subsurface` enters through the closure's subsurface lobe, with its weight over that
/// branch probability. `Absorbed` is a draw Cycles rejects (`LABEL_NONE`).
enum LobeDraw {
    Continuous {
        direction: DVec3,
        diffuse: bool,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
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
    // The subsurface lobe's probability is zero for a closure with no Subsurface Weight, so `choice` never reaches the
    // branch without one. Every method is evaluated, so the branch needs no method check.
    if choice < probabilities.subsurface {
        let method = params.subsurface_method;
        let weight = subsurface_closure_weight(params, normal, view, layers.lower)
            / probabilities.subsurface.max(1.0e-8);
        return LobeDraw::Subsurface { weight, method };
    }
    let transmission_end = probabilities.subsurface + probabilities.transmission;
    if choice < transmission_end {
        let rands = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
        return match sample_glass(params, layers, geometric, probabilities.transmission, rands) {
            GlassSample::Absorbed => LobeDraw::Absorbed,
            GlassSample::Singular { direction, weight } => LobeDraw::Delta { direction, weight },
            GlassSample::Rough { direction } => LobeDraw::Continuous {
                direction,
                diffuse: false,
            },
        };
    }
    let diffuse_end = transmission_end + probabilities.diffuse;
    let sheen_end = diffuse_end + probabilities.sheen;
    let coat_end = sheen_end + probabilities.coat;
    // Cycles' sheen samples are `LABEL_REFLECT | LABEL_DIFFUSE` (`bsdf_sheen.h`), so they set the diffuse ancestor.
    let (direction, diffuse) = if choice < diffuse_end {
        (sample_cosine_hemisphere(normal, rng).0, true)
    } else if choice < sheen_end {
        let first = rng.next_f64();
        let second = rng.next_f64();
        (layers.sample_sheen(first, second), true)
    } else if choice < coat_end {
        let half_vector = sample_ggx_half(layers.coat_normal, params.coat_roughness, rng);
        (reflect(-view, half_vector).normalize_or_zero(), false)
    } else {
        let half_vector = sample_ggx_half(layers.specular, params.roughness, rng);
        (reflect(-view, half_vector).normalize_or_zero(), false)
    };
    // Cycles keeps a reflection sample only above the geometric normal: `bsdf_diffuse.h:65-71` zeroes it otherwise, and
    // the microfacet sampler rejects the same case (`bsdf_microfacet.h:863`). The check is in sampling only; the
    // evaluation keeps the lobe, and the shadow ray already blocks the direct term below the surface.
    if direction.dot(geometric) <= 0.0 {
        return LobeDraw::Absorbed;
    }
    LobeDraw::Continuous { direction, diffuse }
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
        origin,
        direction,
        RAY_EPSILON,
        2.0 * height,
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
    // EEVEE's light threshold sets the volume influence radius of a local light; the path tracer does not read it.
    let threshold = doc
        .scenes
        .get(&snapshot.scene_id)
        .map_or(0.01, |scene| scene.render.eevee_light_threshold);
    doc.nodes
        .iter()
        .filter(|(_, node)| node.kind == "light" && node.visible && node.render_visible)
        .filter_map(|(node_id, node)| {
            let data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))?
                .light
                .as_ref()?;
            let evaluated = snapshot.nodes.get(node_id)?;
            let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
            Some(PointLight {
                position: matrix.transform_point3(DVec3::ZERO),
                direction: matrix.transform_vector3(-DVec3::Z).normalize_or_zero(),
                kind: data.light_type,
                color: DVec3::from_array(data.color).max(DVec3::ZERO),
                energy: data.energy.max(0.0),
                radius: data.radius.max(0.0),
                soft_falloff: data.soft_falloff,
                spot_size: data.spot_size,
                spot_blend: data.spot_blend,
                area: (data.light_type == LightType::Area).then(|| area_shape(data, &matrix)),
                spread: Spread::new(data.spread),
                use_mis: data.use_multiple_importance_sampling,
                volume_factor: data.volume_factor.max(0.0),
                volume_radius_squared: if data.light_type == LightType::Sun {
                    f64::INFINITY
                } else {
                    local_light_influence_radius_squared(data, data.volume_factor, threshold)
                },
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
    meters_per_unit: f64,
    images: &BTreeMap<String, ImageData>,
    warnings: &RefCell<Vec<serde_json::Value>>,
) -> Result<Vec<VolumeInstance<'a>>> {
    let mut volumes = Vec::new();
    let mut evaluator_workspace = EvaluatorWorkspace::default();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible {
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
                let Some((bounds_min, bounds_max)) = grid.active_bbox_world() else {
                    continue;
                };
                (
                    bounds_min,
                    bounds_max,
                    grid.voxel_size().min_element(),
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
        let evaluated_material = if let Some(material) = material {
            let local_center = (bounds_min + bounds_max) * 0.5;
            let mut context = HitContext::new(&doc.node_groups)
                .with_images(images)
                .with_warnings(warnings);
            context.object = local_center;
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
            Some(evaluate_surface_with_workspace(
                material,
                &PreparedMaterial::new(material, &doc.node_groups),
                &context,
                &mut evaluator_workspace,
            )?)
        } else {
            None
        };
        let has_volume_output = material
            .and_then(|material| material.node_tree.as_ref())
            .and_then(|graph_id| doc.node_groups.get(graph_id))
            .is_some_and(|group| {
                group.nodes.iter().any(|(node_id, node)| {
                    matches!(
                        node.node_type.as_str(),
                        "OutputMaterial" | "ShaderNodeOutputMaterial"
                    ) && group
                        .links
                        .iter()
                        .any(|link| link.to_node == *node_id && link.to_socket == "Volume")
                })
            });
        let simple_volume = material.is_some_and(|material| {
            material.node_tree.is_none()
                && material.volume_density.is_finite()
                && material.volume_density > 0.0
        });
        let lobes = match &evaluated_material {
            Some(shading) => scatter_lobes(&shading.volume),
            None => scatter_lobes(&[]),
        };
        let params = evaluated_material.and_then(|shading| volume_medium(&shading.volume));
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
        // Voxel transport is scalar: extinction is the channel mean of σt per unit grid density, and
        // the albedo is the per-channel σs / σt. That is exact for grey media only.
        let extinction: [f64; 3] = std::array::from_fn(|index| scatter[index] + absorption[index]);
        let extinction_scale = extinction.iter().sum::<f64>() / 3.0;
        let albedo = DVec3::from_array(std::array::from_fn(|index| {
            if extinction[index] > 0.0 {
                scatter[index] / extinction[index]
            } else {
                0.0
            }
        }))
        .clamp(DVec3::ZERO, DVec3::ONE);
        let scattering = scatter.iter().any(|weight| *weight > 0.0);
        if (has_volume_output || simple_volume)
            && extinction
                .iter()
                .any(|weight| (weight - extinction_scale).abs() > 1.0e-12)
        {
            warnings.borrow_mut().push(serde_json::json!({
                "code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.volume.spectral_extinction",
                "node_id": node_id.to_string(),
                "message": format!(
                    "Volume '{node_id}' has per-channel extinction; voxel volumes transport the channel-mean extinction, so the colour of its absorption and scattering is approximated"
                ),
            }));
        }
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
        // A sample reads the neighbours of its cell: linear reads one voxel beyond the box and cubic two. Cycles pads a
        // volume's mesh by the same counts (`scene/volume.cpp`, `create_volume_mesh`), so a ray does not stop at the active
        // box and lose that shell. The pad is per axis in index space: a VDB pads its index box before it is transformed.
        let pad_voxels = match interpolation {
            crate::model::VolumeInterpolation::Linear => 1.0,
            crate::model::VolumeInterpolation::Cubic => 2.0,
        };
        let (bounds_min, bounds_max) = if let Some(decoded) = volume.decoded_vdb.as_ref() {
            let Some(grid) = decoded.density_grid() else {
                continue;
            };
            let Some(bounds) = grid.active_bbox_world_padded(DVec3::splat(pad_voxels)) else {
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
            extinction_scale,
            sampling: params.map_or(crate::model::VolumeSampling::default(), |params| {
                params.volume_sampling
            }),
            interpolation,
            index_from_local: if volume.decoded_vdb.is_some() {
                None
            } else {
                Some(local_to_index_matrix(volume)?)
            },
        });
    }
    Ok(volumes)
}

fn homogeneous_segment_integral(extinction: f64, distance: f64) -> (f64, f64) {
    let extinction = extinction.max(0.0);
    let distance = distance.max(0.0);
    let transmittance = (-extinction * distance).exp();
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
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<Vec<MeshMedium>> {
    let probe = DVec3::new(
        0.267_261_241_912_424_4,
        0.534_522_483_824_848_8,
        0.801_783_725_737_273_2,
    );
    let mut inside: Vec<MeshMedium> = Vec::new();
    let mut failure = None;
    bvh.walk_hits(
        geometry.triangles,
        origin,
        probe,
        RAY_EPSILON,
        f64::INFINITY,
        |hit, hit_origin| {
            let (media, _) = match shadow_hit(
                doc,
                geometry,
                hit,
                hit_origin,
                probe,
                images,
                evaluator_workspace,
            ) {
                Ok(shadow) => shadow,
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            };
            for medium in &media {
                toggle_mesh_medium(&mut inside, medium);
            }
            true
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(inside)
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

fn volume_extinction_density(instance: &VolumeInstance<'_>, density: f64) -> f64 {
    density.max(0.0) * instance.extinction_scale
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
/// One voxel segment's sampling grid, as `trace_volume_segment` samples it: the steps across the volumes' bounds
/// along the ray, with each step's grey extinction, emission and scatter per channel at its midpoint. The
/// equiangular direct light reads the same grid, so its transmittance and scatter densities are the sampler's own.
struct VoxelGrid {
    entry: f64,
    step: f64,
    meters_per_unit: f64,
    /// Grey extinction per meter of each step.
    extinction: Vec<f64>,
    /// Emission per meter of each step.
    emission: Vec<DVec3>,
    /// Scatter per meter and channel of each step, summed over the volumes.
    scatter: Vec<DVec3>,
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
) -> Result<Option<VoxelGrid>> {
    let mut entry = f64::INFINITY;
    let mut exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    let mut stochastic = false;
    let mut samplings = Vec::new();
    for instance in volumes {
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
            let local_speed = local_direction.length().max(f64::MIN_POSITIVE);
            minimum_step = minimum_step.min(instance.voxel_size / local_speed * 0.5);
            stochastic |= instance.scattering
                && instance.albedo.max_element() > 0.0
                && instance.extinction_scale > 0.0;
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
        .first()
        .map_or(1.0, |instance| instance.meters_per_unit);
    let mut groups: Vec<ScatterGroup> = Vec::new();
    for lobe in volumes.iter().flat_map(|instance| instance.lobes.iter()) {
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
    let mut extinction = Vec::with_capacity(steps);
    let mut emission = Vec::with_capacity(steps);
    let mut scatter = vec![DVec3::ZERO; steps];
    for (index, step_total) in scatter.iter_mut().enumerate() {
        let point = origin + direction * (entry + (index as f64 + 0.5) * step);
        let mut step_extinction = 0.0;
        let mut step_emission = DVec3::ZERO;
        for instance in volumes {
            let density = volume_density_at(instance, point)?;
            let coefficient = volume_extinction_density(instance, density);
            let step_scatter = instance.albedo * coefficient;
            step_extinction += coefficient;
            step_emission += instance.emission;
            *step_total += step_scatter;
            for lobe in &instance.lobes {
                if let Some(group) = groups
                    .iter_mut()
                    .find(|group| crate::float::equal_f64(group.anisotropy, lobe.anisotropy))
                {
                    group.scatter[index] += step_scatter * lobe.share;
                }
            }
        }
        extinction.push(step_extinction);
        emission.push(step_emission);
    }
    Ok(Some(VoxelGrid {
        entry,
        step,
        meters_per_unit,
        extinction,
        emission,
        scatter,
        groups,
        stochastic,
        samplings,
    }))
}

/// The optical depth, in meters, from the segment origin to `distance` along the grid's steps. Each step is
/// linear in its own distance, as the sampler accumulates it.
fn grid_depth(grid: &VoxelGrid, distance: f64) -> f64 {
    let mut depth = 0.0;
    for (index, extinction) in grid.extinction.iter().enumerate() {
        let start = grid.entry + index as f64 * grid.step;
        if distance <= start {
            break;
        }
        let covered = distance.min(start + grid.step) - start;
        depth += extinction * covered * grid.meters_per_unit;
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

/// The density per unit distance, of the scatter events of the grid at `distance`, that the distance sampler
/// realises: the channel-mean scatter per meter, transmitted over the depth to `distance`. It is a positive
/// function of `distance` for the MIS weights, and the estimator is unbiased for any such function.
fn grid_scatter_pdf(grid: &VoxelGrid, distance: f64) -> f64 {
    let scatter = grid_scatter_at(grid, &grid.scatter, distance).element_sum() / 3.0;
    scatter * grid.meters_per_unit * (-grid_depth(grid, distance)).exp()
}

/// The per-channel extinction of a medium: its scatter plus its absorption, per unit length.
fn medium_extinction(medium: &BsdfParams) -> DVec3 {
    DVec3::from_array(medium.volume_scatter) + DVec3::from_array(medium.volume_absorption)
}

/// The per-channel extinction above the channel mean of a mesh medium: the exponent that grey analog sampling leaves to
/// the channels (`chromatic_correction` is its exponential).
fn mesh_chromatic(medium: &BsdfParams) -> DVec3 {
    let extinction = medium_extinction(medium);
    extinction - DVec3::splat(extinction.element_sum() / 3.0)
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

/// The emission of a segment from its origin to `end`, per channel. A voxel grid contributes its per-step emission and
/// extinction, and a mesh medium its constant emission and extinction, all in one walk over the pieces. A grid that is
/// `grid_analog` (stochastic) takes no transmittance weight of its own: its collisions end the emission
/// (`trace_volume_segment`). A mesh that is `mesh_analog` carries only its chromatic excess over the channel mean, since the
/// analog survival of the mean is the sampling (`mesh_chromatic`). A deterministic medium carries its full transmittance,
/// as Cycles' homogeneous integrator does (`volume_integrate_homogeneous`).
fn emission_until(
    grid: Option<&VoxelGrid>,
    grid_analog: bool,
    mesh: Option<&BsdfParams>,
    mesh_analog: bool,
    end: f64,
) -> DVec3 {
    let mesh_rate = mesh.map_or(DVec3::ZERO, |medium| {
        if mesh_analog {
            mesh_chromatic(medium)
        } else {
            medium_extinction(medium)
        }
    });
    let mesh_source = mesh.map_or(DVec3::ZERO, |medium| {
        DVec3::from_array(medium.volume_emission)
    });
    let mut emitted = DVec3::ZERO;
    let mut exponent = DVec3::ZERO;
    let mut position = 0.0_f64;
    if let Some(grid) = grid {
        for (index, extinction) in grid.extinction.iter().enumerate() {
            let start = grid.entry + index as f64 * grid.step;
            // The mesh alone, from the last piece to this step's start.
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
            let covered = (start + grid.step).min(end) - position;
            if covered > 0.0 {
                let grid_rate = if grid_analog {
                    0.0
                } else {
                    extinction * grid.meters_per_unit
                };
                let grid_source = grid.emission[index] * grid.meters_per_unit;
                add_emission_piece(
                    &mut emitted,
                    &mut exponent,
                    covered,
                    mesh_rate + DVec3::splat(grid_rate),
                    grid_source + mesh_source,
                );
                position += covered;
            }
            if position >= end {
                return emitted;
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
    emitted
}

/// Trace one voxel segment's free flight on its grid with the distance sampler's events, and its transmittance
/// (`volume_integrate_heterogeneous`, with the grid's grey transport). The caller integrates the emission up to the event
/// with `emission_until`.
fn trace_volume_segment(
    grid: Option<&VoxelGrid>,
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    rng: &mut Pcg32,
    // False for a ray that may not collide any more: its segment only attenuates.
    events: bool,
) -> Result<VolumeSegment> {
    let Some(grid) = grid else {
        return Ok(VolumeSegment {
            transmittance: 1.0,
            event: None,
            stochastic: false,
        });
    };
    let free_flight = if grid.stochastic && events {
        -rng.next_f64().max(f64::MIN_POSITIVE).ln()
    } else {
        f64::INFINITY
    };
    let step_in_meters = grid.step * grid.meters_per_unit;
    let mut accumulated_depth = 0.0;
    for (index, &extinction) in grid.extinction.iter().enumerate() {
        if extinction <= 0.0 {
            continue;
        }
        let start = grid.entry + index as f64 * grid.step;
        let step_depth = extinction * step_in_meters;
        let event_in_step = free_flight <= accumulated_depth + step_depth;
        let traveled_meters = if event_in_step {
            ((free_flight - accumulated_depth) / extinction).clamp(0.0, step_in_meters)
        } else {
            step_in_meters
        };
        accumulated_depth += extinction * traveled_meters;
        if event_in_step {
            let event_point = origin + direction * (start + traveled_meters / grid.meters_per_unit);
            let midpoint = origin + direction * (start + grid.step * 0.5);
            let selected = rng.next_f64() * extinction;
            let mut weight = 0.0;
            let mut chosen = None;
            for instance in volumes {
                let density = volume_density_at(instance, midpoint)?;
                weight += volume_extinction_density(instance, density);
                if selected <= weight {
                    chosen = Some(instance);
                    break;
                }
            }
            let event = chosen.map_or(
                VolumeEvent {
                    point: event_point,
                    albedo: DVec3::ZERO,
                },
                |instance| VolumeEvent {
                    point: event_point,
                    albedo: if instance.scattering {
                        instance.albedo
                    } else {
                        DVec3::ZERO
                    },
                },
            );
            return Ok(VolumeSegment {
                transmittance: (-accumulated_depth).exp(),
                event: Some(event),
                stochastic: grid.stochastic,
            });
        }
    }
    Ok(VolumeSegment {
        transmittance: (-accumulated_depth).exp(),
        event: None,
        stochastic: grid.stochastic,
    })
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
    match instance.index_from_local {
        Some(index_from_local) => sample_index_density_with(
            index_from_local.transform_point3(local_position),
            instance.volume,
            instance.interpolation,
        ),
        None => sample_density_with(local_position, instance.volume, instance.interpolation),
    }
}

/// `instance_density` with its error, which reports why a density is not finite.
fn instance_density_checked(instance: &VolumeInstance<'_>, local_position: DVec3) -> Result<f64> {
    match instance.index_from_local {
        Some(index_from_local) => sample_index_density_checked_with(
            index_from_local.transform_point3(local_position),
            instance.volume,
            instance.interpolation,
        ),
        None => {
            sample_density_checked_with(local_position, instance.volume, instance.interpolation)
        }
    }
}

fn march_volumes(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    rng: Option<&mut Pcg32>,
) -> Result<f64> {
    march_volumes_with_budget(
        volumes,
        origin,
        direction,
        max_distance,
        rng,
        MAX_VOLUME_STEPS,
    )
}

fn march_volumes_with_budget(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    mut rng: Option<&mut Pcg32>,
    step_budget: usize,
) -> Result<f64> {
    let step_budget = step_budget.max(1);
    let mut transmittance = 1.0;
    for instance in volumes {
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
            let majorant = volume_extinction_density(instance, maximum_density);
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
                    transmittance *=
                        1.0 - volume_extinction_density(instance, density).min(majorant) / majorant;
                    if transmittance <= 1.0e-6 {
                        break;
                    }
                }
            } else if majorant > 0.0 {
                transmittance *= (-majorant * segment_length * instance.meters_per_unit).exp();
            }
            if transmittance <= 1.0e-6 {
                break;
            }
            continue;
        }
        let local_speed = local_direction.length().max(f64::MIN_POSITIVE);
        let step_world =
            (instance.voxel_size / local_speed * 0.5).max(segment_length / step_budget as f64);
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
            let attenuation =
                (-volume_extinction_density(instance, density) * step * instance.meters_per_unit)
                    .exp();
            transmittance *= attenuation;
            if transmittance <= 1.0e-6 {
                break;
            }
        }
        if transmittance <= 1.0e-6 {
            break;
        }
    }
    Ok(transmittance)
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
            extinction_scale: 1.0,
            sampling: crate::model::VolumeSampling::MultipleImportance,
            interpolation: crate::model::VolumeInterpolation::Linear,
            index_from_local: None,
        };
        let mut rng = Pcg32::for_sample(0, 0, 0, 0);
        let transmittance = march_volumes(
            &[instance],
            DVec3::new(-1.0, 1.0, 1.0),
            DVec3::X,
            f64::INFINITY,
            Some(&mut rng),
        )?;

        assert!(
            (transmittance - (-1.0_f64).exp()).abs() <= 1.0e-12,
            "2 scene units at 0.5 m/unit should attenuate by one meter"
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
            kind: LightType::Area,
            color: DVec3::ONE,
            energy: power,
            radius: 0.0,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
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
            kind: LightType::Point,
            color: DVec3::ONE,
            energy: 4.0 * std::f64::consts::PI,
            radius: 1.0e-6,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
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
            kind: LightType::Point,
            color: DVec3::ONE,
            energy: 100.0,
            radius: 1.0,
            soft_falloff: false,
            use_mis: false,
            volume_factor: 1.0,
            volume_radius_squared: f64::INFINITY,
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
                    total += evaluate_bsdf(
                        &params,
                        &layers_at(&params, normal, view, normal, true),
                        light,
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
                    kind: LightType::Area,
                    color: DVec3::ONE,
                    energy: std::f64::consts::PI * shape.area,
                    radius: 0.0,
                    soft_falloff: false,
                    spot_size: 0.0,
                    spot_blend: 0.0,
                    area: Some(shape),
                    spread: Some(spread),
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
            kind: LightType::Area,
            color: DVec3::ONE,
            energy: std::f64::consts::PI * shape.area,
            radius: 0.0,
            soft_falloff: false,
            spot_size: 0.0,
            spot_blend: 0.0,
            area: Some(shape),
            spread: Spread::new(0.0),
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

    /// Records one warning as a worker's sink does: a Principled input once per (material, input)
    /// (`record_unevaluated_inputs`), any other warning once per equal value (`layer_index`, `color_layer_missing`).
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
                0 => json!({"code": "UNSUPPORTED_FEATURE", "feature_id": "shader.principled.anisotropic",
                    "material": "a", "input": "Anisotropic", "message": "a"}),
                1 => json!({"code": "UNSUPPORTED_FEATURE", "feature_id": "shader.principled.anisotropic",
                    "material": "b", "input": "Anisotropic", "message": "b"}),
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
