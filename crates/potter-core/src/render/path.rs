use std::{cell::RefCell, collections::BTreeMap, path::Path};

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::volume::{VolumeData, sample_density_checked_with, sample_density_with},
    image::{ImageData, ImageInterpolation},
    model::{Id, LightType, Material, SceneDoc, VolumeSampling},
    shader::{
        BsdfParams, EvaluatorWorkspace, HitContext, HitUvLayers, PreparedMaterial,
        equirectangular_uv, evaluate_surface_with_workspace, height_gradient,
    },
};

use super::principled_bsdf::{
    Layers, continuous_density, evaluate as evaluate_bsdf, layers, lobe_probabilities,
};
use super::{
    bvh::{Bvh, BvhHit},
    light_model::{
        AreaShape, Spread, area_radiance, area_shape, emitter_radiance, inside_emitter,
        point_intensity, spot_attenuation, spread_attenuation, spread_region,
    },
    raster::{
        Camera, Mode, RasterOutput, StereoMode, Triangle, blend_corner_normals, camera_eye,
        rasterize_with_lines, ray_for_sample, smooth_shadow_origin,
    },
    scene::{Geometry, HitPose, LayerTables, ObjectRecord, layered_context, object_space},
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
}

impl PointLight {
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

#[derive(Clone, Copy)]
struct VolumeInstance<'a> {
    volume: &'a VolumeData,
    world_to_local: DMat4,
    bounds_min: DVec3,
    bounds_max: DVec3,
    voxel_size: f64,
    meters_per_unit: f64,
    density_range: Option<(f64, f64)>,
    albedo: DVec3,
    anisotropy: f64,
    emission: DVec3,
    scattering: bool,
    /// Channel-mean σt per unit grid density (absolute, as `BsdfParams`'s coefficients).
    extinction_scale: f64,
    /// Blender's `cycles.volume_sampling` of the volume's material.
    sampling: crate::model::VolumeSampling,
    /// Blender's `cycles.volume_interpolation` of the volume's material.
    interpolation: crate::model::VolumeInterpolation,
}

#[derive(Clone, Copy)]
struct VolumeEvent {
    point: DVec3,
    albedo: DVec3,
    anisotropy: f64,
}

struct VolumeSegment {
    transmittance: f64,
    emission: DVec3,
    event: Option<VolumeEvent>,
    stochastic: bool,
}

#[derive(Clone, Copy)]
struct Surface {
    point: DVec3,
    normal: DVec3,
    /// Unit normal on the viewer's side of the triangle; offsets ray origins to avoid self-hits.
    geometric_normal: DVec3,
    /// Origin of shadow rays: `point` on flat triangles, lifted onto the smooth surface otherwise.
    shadow_point: DVec3,
    params: BsdfParams,
    /// The sheen and coat layers at this hit, set up once for the hit's view direction.
    layers: Layers,
    object_index: u32,
    element_index: u32,
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

    for y in 0..height {
        for x in 0..width {
            let index = usize::try_from(y)
                .ok()
                .and_then(|row| row.checked_mul(width_usize))
                .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "pixel index exceeds platform limits",
                    )
                })?;
            let mut sum = DVec3::ZERO;
            let mut alpha_sum = 0.0;
            for sample_index in 0..samples {
                let mut rng = Pcg32::for_sample(seed, x, y, sample_index);
                let sample_x = rng.next_f64();
                let sample_y = rng.next_f64();
                let (radiance, alpha) = if camera.stereo_mode == StereoMode::Anaglyph {
                    let left_camera = camera_eye(camera, -1.0);
                    let right_camera = camera_eye(camera, 1.0);
                    let left_ray =
                        ray_for_sample(&left_camera, width, height, x, y, sample_x, sample_y)?;
                    let right_ray =
                        ray_for_sample(&right_camera, width, height, x, y, sample_x, sample_y)?;
                    let mut left_rng = rng.clone();
                    let mut right_rng = rng.clone();
                    // A sample with no ray (outside a fisheye's circle, or past its field of view) is opaque black: Blender's
                    // film writes a pixel with no ray as black with alpha 1 under either film (`probe_transparent_film_fisheye.py`).
                    let (left, left_alpha) = match left_ray {
                        Some((left_origin, left_direction)) => trace_path(
                            doc,
                            snapshot,
                            geometry,
                            &bvh,
                            &lights,
                            &volumes,
                            images,
                            world_importance.as_ref(),
                            left_origin,
                            left_direction,
                            environment,
                            max_bounces,
                            transparent_max_bounces,
                            &mut left_rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    };
                    let (right, right_alpha) = match right_ray {
                        Some((right_origin, right_direction)) => trace_path(
                            doc,
                            snapshot,
                            geometry,
                            &bvh,
                            &lights,
                            &volumes,
                            images,
                            world_importance.as_ref(),
                            right_origin,
                            right_direction,
                            environment,
                            max_bounces,
                            transparent_max_bounces,
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
                    match ray_for_sample(camera, width, height, x, y, sample_x, sample_y)? {
                        Some((origin, direction)) => trace_path(
                            doc,
                            snapshot,
                            geometry,
                            &bvh,
                            &lights,
                            &volumes,
                            images,
                            world_importance.as_ref(),
                            origin,
                            direction,
                            environment,
                            max_bounces,
                            transparent_max_bounces,
                            &mut rng,
                            &mut evaluator_workspace,
                        )?,
                        None => (DVec3::ZERO, 1.0),
                    }
                };
                sum += radiance;
                alpha_sum += alpha;
            }
            let sample_scale = 1.0 / f64::from(samples);
            let color = sum * sample_scale;
            let alpha = if film_transparent {
                alpha_sum * sample_scale
            } else {
                1.0
            };
            output.linear_rgba[index] = [
                finite_f32(color.x),
                finite_f32(color.y),
                finite_f32(color.z),
                finite_f32(alpha),
            ];
            output.rgba[index * 4..index * 4 + 4].copy_from_slice(&[
                to_srgb_byte(color.x),
                to_srgb_byte(color.y),
                to_srgb_byte(color.z),
                (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
    }

    write_center_passes(
        doc,
        snapshot,
        geometry,
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
            source += instance.emission * density;
            if instance.scattering {
                source += ambient * instance.albedo * scattering_coefficient;
                for light in lights {
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
                    let phase =
                        henyey_greenstein(direction.dot(sample.direction), instance.anisotropy);
                    source += instance.albedo
                        * sample.radiance
                        * (scattering_coefficient * phase * visibility);
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

#[expect(
    clippy::too_many_arguments,
    reason = "one path's ray, scene state, sampling, and world context are explicit"
)]
fn trace_path(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    bvh: &Bvh,
    lights: &[PointLight],
    volumes: &[VolumeInstance<'_>],
    images: &BTreeMap<String, ImageData>,
    importance: Option<&WorldImportanceSampler>,
    mut origin: DVec3,
    mut direction: DVec3,
    environment: DVec3,
    max_bounces: u32,
    transparent_max_bounces: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<(DVec3, f64)> {
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
    // Set by the pass that reaches the cap: the next surface is shaded for emission and ends the
    // path (`PATH_RAY_TERMINATE_ON_NEXT_SURFACE`, `integrate_surface_terminate`).
    let mut terminate_on_next_surface = false;
    // The volume stack: the media the path is inside, innermost last (Cycles' object volume stack).
    let mut media: Vec<MeshMedium> = Vec::new();
    loop {
        let hit = closest_hit(geometry, bvh, origin, direction, RAY_EPSILON, f64::INFINITY);
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
        let environment = world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        // Voxel volumes sample their grid by free flight. Under their material's `volume_sampling` their direct light is
        // equiangular about each light's point, as the mesh medium's is, unless a mesh medium is on top: its direct light
        // covers the whole segment, so voxel events then take distance sampling in full.
        let grid = voxel_grid(volumes, origin, direction, segment_limit)?;
        let voxel_method = match (&grid, media.is_empty()) {
            (Some(grid), true) => segment_sampling(grid.samplings.iter().copied()),
            _ => None,
        };
        let budget = transparent_max_bounces.saturating_sub(transparent_bounce);
        let choices = equiangular_choices(lights, origin, voxel_method, rng);
        if let (Some(method), Some(grid)) = (voxel_method, grid.as_ref()) {
            for (light, &(point, sampled)) in lights.iter().zip(&choices) {
                let Some(point) = point.filter(|_| sampled) else {
                    continue;
                };
                let Some((distance, pdf)) =
                    equiangular_sample(point, origin, direction, segment_limit, rng.next_f64())
                else {
                    continue;
                };
                let transmitted = grid.meters_per_unit * (-grid_depth(grid, distance)).exp();
                let mis = match method {
                    VolumeSampling::Equiangular => 1.0,
                    _ => 2.0 * power_heuristic(pdf, grid_scatter_pdf(grid, distance)),
                };
                let sample_point = origin + direction * distance;
                for group in &grid.groups {
                    let camera = grid_scatter_at(grid, &group.scatter, distance) * transmitted;
                    let nee = volume_light_nee(
                        doc,
                        snapshot,
                        geometry,
                        bvh,
                        volumes,
                        &media,
                        images,
                        light,
                        sample_point,
                        direction,
                        group.anisotropy,
                        budget,
                        rng,
                        evaluator_workspace,
                    )?;
                    radiance += throughput * camera * nee * (mis / pdf);
                }
            }
        }
        let volume_segment = trace_volume_segment(grid.as_ref(), volumes, origin, direction, rng)?;
        radiance += throughput * volume_segment.emission;
        if !volume_segment.stochastic {
            throughput *= volume_segment.transmittance;
        }
        if throughput.max_element() <= 1.0e-8 {
            break;
        }
        if let Some(event) = volume_segment.event {
            let scatter_probability = (event.albedo.x + event.albedo.y + event.albedo.z) / 3.0;
            if scatter_probability <= 0.0 || rng.next_f64() >= scatter_probability {
                break;
            }
            throughput *= event.albedo / scatter_probability;
            let event_distance = (event.point - origin).dot(direction);
            let (event_method, scatter_pdf) = match (voxel_method, grid.as_ref()) {
                (Some(method), Some(grid)) => {
                    (Some(method), grid_scatter_pdf(grid, event_distance))
                }
                _ => (None, 0.0),
            };
            let weights = event_weights(
                &choices,
                event_method,
                origin,
                direction,
                segment_limit,
                event_distance,
                scatter_pdf,
            );
            radiance += throughput
                * direct_volume_lighting(
                    doc,
                    snapshot,
                    geometry,
                    bvh,
                    lights,
                    &weights,
                    volumes,
                    &media,
                    images,
                    importance,
                    event,
                    direction,
                    environment,
                    budget,
                    rng,
                    evaluator_workspace,
                )?;
            if bounce == max_bounces {
                break;
            }
            let incoming = direction;
            direction = sample_henyey_greenstein_direction(incoming, event.anisotropy, rng);
            let phase_pdf = henyey_greenstein(
                incoming.normalize_or_zero().dot(direction),
                event.anisotropy,
            );
            origin = event.point + direction * RAY_EPSILON;
            previous = Scatter::Volume(phase_pdf);
            bounce += 1;
            continue;
        }
        // The stack changes only at surfaces, so the top medium spans this whole segment. Its direct light follows
        // its material's `volume_sampling` (Cycles' `volume_direct_sample_method`): equiangular sampling about each
        // light's point, distance sampling at the scatter event, or both as multiple importance sampling. Free flight
        // is spectral ratio tracking (`trace_mesh_medium`).
        if let Some(medium) = media.last().map(|medium| medium.params) {
            use crate::model::VolumeSampling;
            let budget = transparent_max_bounces.saturating_sub(transparent_bounce);
            let method = segment_sampling([medium.volume_sampling]);
            let choices = equiangular_choices(lights, origin, method, rng);
            for (light, &(point, sampled)) in lights.iter().zip(&choices) {
                let Some(point) = point.filter(|_| sampled) else {
                    continue;
                };
                let Some((distance, pdf)) =
                    equiangular_sample(point, origin, direction, segment_limit, rng.next_f64())
                else {
                    continue;
                };
                let nee = volume_light_nee(
                    doc,
                    snapshot,
                    geometry,
                    bvh,
                    volumes,
                    &media,
                    images,
                    light,
                    origin + direction * distance,
                    direction,
                    medium.volume_anisotropy,
                    budget,
                    rng,
                    evaluator_workspace,
                )?;
                // Multiple importance sampling weights this sample against the distance strategy's density at
                // the same point (`volume_direct_scatter_mis`); equiangular sampling alone takes no weight.
                let mis = match method {
                    Some(VolumeSampling::Equiangular) => 1.0,
                    _ => 2.0 * power_heuristic(pdf, distance_scatter_pdf(&medium, distance)),
                };
                radiance += throughput * camera_scatter(&medium, distance) * nee * (mis / pdf);
            }
            match trace_mesh_medium(&medium, segment_limit, &mut throughput, rng) {
                MediumOutcome::Scatter(distance) => {
                    let event = VolumeEvent {
                        point: origin + direction * distance,
                        albedo: DVec3::ZERO,
                        anisotropy: medium.volume_anisotropy,
                    };
                    // A light sampled equiangularly takes no weight here. Under multiple importance sampling, a
                    // light with an equiangular point is weighted against its equiangular pdf at the event, and
                    // every other light takes the event in full.
                    let weights: Vec<f64> = lights
                        .iter()
                        .zip(&choices)
                        .map(|(_, &(point, sampled))| match (sampled, method, point) {
                            (true, _, _) => 0.0,
                            (false, Some(VolumeSampling::MultipleImportance), Some(point)) => {
                                2.0 * power_heuristic(
                                    distance_scatter_pdf(&medium, distance),
                                    equiangular_pdf(
                                        point,
                                        origin,
                                        direction,
                                        segment_limit,
                                        distance,
                                    ),
                                )
                            }
                            _ => 1.0,
                        })
                        .collect();
                    radiance += throughput
                        * direct_volume_lighting(
                            doc,
                            snapshot,
                            geometry,
                            bvh,
                            lights,
                            &weights,
                            volumes,
                            &media,
                            images,
                            importance,
                            event,
                            direction,
                            environment,
                            budget,
                            rng,
                            evaluator_workspace,
                        )?;
                    if bounce == max_bounces {
                        break;
                    }
                    let incoming = direction;
                    direction = sample_henyey_greenstein_direction(incoming, event.anisotropy, rng);
                    let phase_pdf = henyey_greenstein(
                        incoming.normalize_or_zero().dot(direction),
                        event.anisotropy,
                    );
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
            radiance += throughput * environment * environment_weight;
            escaped = true;
            break;
        };
        let surface = make_surface(
            doc,
            snapshot,
            geometry,
            hit,
            origin,
            direction,
            images,
            evaluator_workspace,
        )?;
        if surface.params.has_medium() {
            // A medium surface passes the path and changes the volume stack: it enters the medium of its
            // object, or leaves the one it entered.
            toggle_mesh_medium(&mut media, surface.object_index, &surface.params);
            volume_bounds += 1;
            if volume_bounds > VOLUME_BOUNDS_MAX {
                break;
            }
            origin = offset_origin(surface.point, direction, surface.geometric_normal);
            continue;
        }
        if terminate_on_next_surface {
            // Cycles shades the surface for emission, then ends the path (`integrate_surface_terminate`).
            radiance += throughput
                * DVec3::from_array(surface.params.emission_color)
                * surface.params.emission_strength
                * surface.layers.lower();
            break;
        }
        if rng.next_f64() >= surface.params.alpha.clamp(0.0, 1.0) {
            // A pass carries the Transparent closure's colour: the tint is its mean, weighted by the pass chance
            // `1 - alpha`, so the expected throughput is Cycles' `(1 - alpha) * tint`.
            throughput *= DVec3::from_array(surface.params.transparent_tint);
            transparent_bounce += 1;
            terminate_on_next_surface |= transparent_bounce >= transparent_cap;
            origin = offset_origin(surface.point, direction, surface.geometric_normal);
            continue;
        }
        radiance += throughput
            * DVec3::from_array(surface.params.emission_color)
            * surface.params.emission_strength
            * surface.layers.lower();
        radiance += throughput
            * direct_lighting(
                doc,
                snapshot,
                geometry,
                bvh,
                lights,
                volumes,
                &media,
                images,
                importance,
                &surface,
                environment,
                bounce < max_bounces,
                transparent_max_bounces.saturating_sub(transparent_bounce),
                rng,
                evaluator_workspace,
            )?;
        if bounce == max_bounces {
            break;
        }
        let (next_direction, weight) = match sample_bsdf(&surface.params, &surface.layers, rng) {
            BsdfSample::Continuous {
                direction,
                weight,
                pdf,
            } => {
                previous = Scatter::Surface(MisVertex {
                    bsdf_pdf: pdf,
                    normal: surface.normal,
                    point: surface.point,
                });
                (direction, weight)
            }
            BsdfSample::Delta { direction, weight } => {
                previous = Scatter::Delta(surface.point);
                (direction, weight)
            }
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
    snapshot: &Snapshot,
    geometry: &Geometry,
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
                snapshot,
                geometry,
                hit,
                origin,
                direction,
                images,
                evaluator_workspace,
            )?;
            if surface.params.alpha <= 0.0 {
                continue;
            }
            output.ids[index] = surface.object_index;
            output.elements[index] = surface.element_index;
            output.depths[index] = finite_f32(camera.depth_of(surface.point));
            output.normals[index] = surface.normal.to_array().map(finite_f32);
            output.ambient_occlusion[index] = finite_f32(super::raster::ambient_visibility(
                bvh,
                &geometry.triangles,
                surface.point,
                surface.geometric_normal,
                surface.normal,
                1.0,
            ));
            output.albedo[index] = surface.params.base_color.map(finite_f32);
            // The emission pass is the beauty's emission term: the coat attenuates it per channel.
            let emission = DVec3::from_array(surface.params.emission_color)
                * surface.params.emission_strength
                * surface.layers.lower();
            output.emission[index] = [
                finite_f32(emission.x),
                finite_f32(emission.y),
                finite_f32(emission.z),
                1.0,
            ];
            if film_transparent {
                let alpha = surface.params.alpha
                    + (1.0 - surface.params.alpha) * (1.0 - volume_transmittance);
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
    geometry: &Geometry,
    bvh: &Bvh,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
) -> Option<BvhHit> {
    bvh.closest_hit_two_sided(&geometry.triangles, origin, direction, t_min, t_max)
}

/// The shading parameters at one hit, for the transparent pass of a realtime shadow ray (`render/scene.rs`). They
/// are the path tracer's, so a realtime shadow and a path-traced one read the same alpha and tint. The pass reads
/// neither the displacement gradient nor the coat layers, so only the material evaluation runs.
pub(super) fn hit_params(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<BsdfParams> {
    with_hit_frame(
        doc,
        snapshot,
        geometry,
        hit,
        origin,
        direction,
        images,
        |frame| {
            let mut params = evaluate_surface_with_workspace(
                frame.material,
                frame.prepared,
                &frame.context,
                evaluator_workspace,
            )?;
            apply_hit_alpha(frame.material, &mut params);
            Ok(params)
        },
    )
}

/// A hit's position, normals, and the evaluation context of its material. The context borrows its layer tables, so a
/// frame exists only inside [`with_hit_frame`].
struct HitFrame<'a> {
    point: DVec3,
    shadow_point: DVec3,
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
    snapshot: &Snapshot,
    geometry: &Geometry,
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
    if geometric_normal.dot(-direction) < 0.0 {
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

    let object = object_for_index(&geometry.objects, triangle.object_index).ok_or_else(|| {
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
        &geometry.shader_warnings,
        object,
        triangle,
        HitPose {
            point,
            object: object_space(snapshot, object, point),
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

fn make_surface(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<Surface> {
    with_hit_frame(
        doc,
        snapshot,
        geometry,
        hit,
        origin,
        direction,
        images,
        |frame| {
            let mut params = evaluate_surface_with_workspace(
                frame.material,
                frame.prepared,
                &frame.context,
                evaluator_workspace,
            )?;
            apply_hit_alpha(frame.material, &mut params);
            let normal = frame.normal;
            if params.displacement_height != 0.0
                && frame.material.displacement_method != "displacement"
            {
                let gradient = height_gradient(&frame.context, normal, |sample| {
                    evaluate_surface_with_workspace(
                        frame.material,
                        frame.prepared,
                        sample,
                        evaluator_workspace,
                    )
                    .map(|surface| surface.displacement_height)
                })?;
                params.normal = normal - gradient;
            }
            params.normal = if params.normal.is_finite() {
                params.normal.normalize_or_zero()
            } else {
                normal
            };
            if params.normal.dot(-direction) < 0.0 {
                params.normal = -params.normal;
            }
            let hit_layers = layers(&params, params.normal, -direction);
            Ok(Surface {
                point: frame.point,
                normal: params.normal,
                geometric_normal: frame.geometric_normal,
                shadow_point: frame.shadow_point,
                params,
                layers: hit_layers,
                object_index: frame.triangle.object_index,
                element_index: frame.triangle.element_index,
            })
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

#[expect(
    clippy::too_many_arguments,
    reason = "direct-light sampling needs full material, shadow, and medium context"
)]
fn direct_lighting(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    bvh: &Bvh,
    lights: &[PointLight],
    volumes: &[VolumeInstance<'_>],
    media: &[MeshMedium],
    images: &BTreeMap<String, ImageData>,
    importance: Option<&WorldImportanceSampler>,
    surface: &Surface,
    environment: DVec3,
    // Whether the path samples a BSDF continuation here. Only then does a BSDF ray pair with the
    // light sample for MIS; at the last vertex the light sample is unweighted.
    scatters: bool,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let normal = surface.normal;
    let geometric = surface.geometric_normal;
    let mut contribution = DVec3::ZERO;
    for light in lights {
        let Some(sample) = sample_light(light, surface.point, rng) else {
            continue;
        };
        let cosine = normal.dot(sample.direction).max(0.0);
        if cosine <= 0.0 || sample.radiance.max_element() <= 0.0 {
            continue;
        }
        let visibility = shadow_transmittance(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            media,
            images,
            surface.shadow_point,
            geometric,
            sample.direction,
            sample.distance,
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        if visibility.max_element() <= 0.0 {
            continue;
        }
        // A BSDF ray reaches a hittable light only when the path continues past this vertex and
        // no volume can scatter it first (`trace_path`).
        let mis = if scatters && volumes.is_empty() && light.hittable() {
            power_heuristic(
                sample.pdf,
                continuous_density(&surface.params, &surface.layers, sample.direction),
            )
        } else {
            1.0
        };
        contribution += evaluate_bsdf(&surface.params, &surface.layers, sample.direction)
            * sample.radiance
            * (cosine * visibility * mis);
    }
    let direction = match importance {
        Some(sampler) if rng.next_f64() >= 0.5 => sampler.sample(rng).0,
        _ => sample_cosine_hemisphere(normal, rng).0,
    };
    let cosine = normal.dot(direction).max(0.0);
    let light_pdf = environment_light_pdf(normal, direction, importance);
    if cosine > 0.0 && light_pdf > 0.0 {
        let environment_sample =
            world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        let visibility = shadow_transmittance(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            media,
            images,
            surface.shadow_point,
            geometric,
            direction,
            f64::INFINITY,
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        let bsdf_pdf = continuous_density(&surface.params, &surface.layers, direction);
        let weight = if scatters {
            power_heuristic(light_pdf, bsdf_pdf)
        } else {
            1.0
        };
        contribution += evaluate_bsdf(&surface.params, &surface.layers, direction)
            * environment_sample
            * (cosine * visibility * weight / light_pdf);
    }
    Ok(contribution)
}

#[expect(
    clippy::too_many_arguments,
    reason = "volume direct lighting needs explicit transport and world state"
)]
fn direct_volume_lighting(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    bvh: &Bvh,
    lights: &[PointLight],
    light_weights: &[f64],
    volumes: &[VolumeInstance<'_>],
    media: &[MeshMedium],
    images: &BTreeMap<String, ImageData>,
    importance: Option<&WorldImportanceSampler>,
    event: VolumeEvent,
    incoming_direction: DVec3,
    environment: DVec3,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let mut contribution = DVec3::ZERO;
    for (index, light) in lights.iter().enumerate() {
        let weight = light_weights.get(index).copied().unwrap_or(1.0);
        if weight <= 0.0 {
            continue;
        }
        contribution += volume_light_nee(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            media,
            images,
            light,
            event.point,
            incoming_direction,
            event.anisotropy,
            transparent_budget,
            rng,
            evaluator_workspace,
        )? * weight;
    }
    let (direction, pdf) = if let Some(sampler) = importance {
        sampler.sample(rng)
    } else {
        sample_uniform_sphere(rng)
    };
    if pdf > 0.0 {
        let incoming = world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        let visibility = shadow_transmittance(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            media,
            images,
            event.point,
            DVec3::ZERO,
            direction,
            f64::INFINITY,
            transparent_budget,
            rng,
            evaluator_workspace,
        )?;
        let phase = henyey_greenstein(incoming_direction.dot(direction), event.anisotropy);
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

/// The camera-side scattered throughput of a homogeneous medium at `t`, per channel: σs · exp(-σt · t).
fn camera_scatter(medium: &BsdfParams, t: f64) -> DVec3 {
    let scatter = DVec3::from_array(medium.volume_scatter);
    let extinction = scatter + DVec3::from_array(medium.volume_absorption);
    scatter * DVec3::from_array(extinction.to_array().map(|sigma| (-sigma * t).exp()))
}

/// One point light's next-event estimate at `point` for an incoming travel direction: the sample's radiance, the
/// Henyey-Greenstein phase and the shadow transmittance, without any MIS weight.
#[expect(
    clippy::too_many_arguments,
    reason = "a light's estimate needs the scene, the medium stack, and the sampling state"
)]
fn volume_light_nee(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    bvh: &Bvh,
    volumes: &[VolumeInstance<'_>],
    media: &[MeshMedium],
    images: &BTreeMap<String, ImageData>,
    light: &PointLight,
    point: DVec3,
    incoming_direction: DVec3,
    anisotropy: f64,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let Some(sample) = sample_light(light, point, rng) else {
        return Ok(DVec3::ZERO);
    };
    if sample.radiance.max_element() <= 0.0 {
        return Ok(DVec3::ZERO);
    }
    let visibility = shadow_transmittance(
        doc,
        snapshot,
        geometry,
        bvh,
        volumes,
        media,
        images,
        point,
        DVec3::ZERO,
        sample.direction,
        sample.distance,
        transparent_budget,
        rng,
        evaluator_workspace,
    )?;
    let phase = henyey_greenstein(incoming_direction.dot(sample.direction), anisotropy);
    Ok(sample.radiance * (phase * visibility))
}

#[expect(
    clippy::too_many_arguments,
    reason = "shadow traversal needs exact scene visibility, materials, and medium state"
)]
fn shadow_transmittance(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    bvh: &Bvh,
    volumes: &[VolumeInstance<'_>],
    media: &[MeshMedium],
    images: &BTreeMap<String, ImageData>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
    transparent_budget: u32,
    rng: &mut Pcg32,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
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
        &geometry.triangles,
        offset_origin(point, direction, normal),
        direction,
        RAY_EPSILON,
        remaining,
        |hit, origin| {
            let surface = match make_surface(
                doc,
                snapshot,
                geometry,
                hit,
                origin,
                direction,
                images,
                evaluator_workspace,
            ) {
                Ok(surface) => surface,
                Err(error) => {
                    failure = Some(error);
                    return false;
                }
            };
            if let Some(medium) = shadow_media.last() {
                transmittance *= medium_transmittance(&medium.params, hit.distance);
            }
            if surface.params.has_medium() {
                // A medium surface passes the shadow ray and changes the stack, as it does for a path. It is
                // not a transparent surface, so it does not count against the budget.
                toggle_mesh_medium(&mut shadow_media, surface.object_index, &surface.params);
            } else {
                // Cycles counts each transparent surface a shadow ray crosses against the path's remaining
                // budget and blocks the ray past it (`intern/cycles/kernel/bvh/intersect_filter.h`).
                transparent_hits += 1;
                if transparent_hits > transparent_budget {
                    blocked = true;
                    return false;
                }
                let alpha = surface.params.alpha.clamp(0.0, 1.0);
                let transmission = surface.params.transmission.clamp(0.0, 1.0);
                if alpha >= 1.0 && transmission <= 0.0 {
                    blocked = true;
                    return false;
                }
                // Each transparent share passes its Transparent colour, `(1 - alpha) * tint` as Cycles' closure
                // weight (`bsdf_transparent.h`). A glass share passes its transmission, tinted by the base colour.
                let base_tint = DVec3::new(
                    surface.params.base_color[0],
                    surface.params.base_color[1],
                    surface.params.base_color[2],
                )
                .max_element()
                .clamp(0.0, 1.0);
                transmittance *= DVec3::from_array(surface.params.transparent_tint) * (1.0 - alpha)
                    + DVec3::splat(alpha * transmission * base_tint);
            }
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
    // The last segment, to the light or to infinity, lies in the top medium of the stack.
    if let Some(medium) = shadow_media.last() {
        transmittance *= medium_transmittance(&medium.params, remaining);
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
    /// A delta (specular transmission) bounce at `point`: light sampling cannot reach it, so a
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

/// Outcome of sampling the Principled BSDF. `Continuous` carries the exact lobe-mixture density
/// that MIS needs; `Delta` is the specular transmission lobe, which light sampling cannot reach.
enum BsdfSample {
    Continuous {
        direction: DVec3,
        weight: DVec3,
        pdf: f64,
    },
    Delta {
        direction: DVec3,
        weight: DVec3,
    },
}

fn sample_bsdf(params: &BsdfParams, layers: &Layers, rng: &mut Pcg32) -> BsdfSample {
    let (normal, view) = (layers.normal, layers.view);
    let probabilities = lobe_probabilities(params, layers);
    let choice = rng.next_f64();
    if choice < probabilities.transmission {
        let direction = refract(-view, normal, 1.0 / params.ior.max(1.0e-3))
            .unwrap_or_else(|| reflect(-view, normal));
        let tint = DVec3::new(
            params.base_color[0],
            params.base_color[1],
            params.base_color[2],
        );
        let weight = tint
            * (params.reflectance.clamp(0.0, 1.0) * params.transmission.clamp(0.0, 1.0)
                / probabilities.transmission.max(1.0e-8))
            * layers.lower();
        return BsdfSample::Delta { direction, weight };
    }
    let diffuse_end = probabilities.transmission + probabilities.diffuse;
    let sheen_end = diffuse_end + probabilities.sheen;
    let coat_end = sheen_end + probabilities.coat;
    let direction = if choice < diffuse_end {
        sample_cosine_hemisphere(normal, rng).0
    } else if choice < sheen_end {
        let first = rng.next_f64();
        let second = rng.next_f64();
        layers.sample_sheen(first, second)
    } else if choice < coat_end {
        let half_vector =
            sample_ggx_half(params.coat_normal_or_shading(), params.coat_roughness, rng);
        reflect(-view, half_vector).normalize_or_zero()
    } else {
        let half_vector = sample_ggx_half(normal, params.roughness, rng);
        reflect(-view, half_vector).normalize_or_zero()
    };
    // Estimator `f cos / p_mix`: the density is the lobe mixture, not the branch's own lobe.
    let pdf = continuous_density(params, layers, direction);
    let cosine = normal.dot(direction).max(0.0);
    let weight = evaluate_bsdf(params, layers, direction) * (cosine / pdf.max(1.0e-8));
    BsdfSample::Continuous {
        direction,
        weight,
        pdf,
    }
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

fn refract(direction: DVec3, normal: DVec3, eta: f64) -> Option<DVec3> {
    let cosine = (-direction).dot(normal).clamp(-1.0, 1.0);
    let discriminant = 1.0 - eta * eta * (1.0 - cosine * cosine);
    (discriminant >= 0.0).then(|| eta * direction + (eta * cosine - discriminant.sqrt()) * normal)
}

fn collect_lights(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<PointLight> {
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
            })
        })
        .collect()
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
                let bounds_max = DVec3::from_array(
                    grid.dims
                        .map(|dimension| f64::from(dimension.saturating_sub(1)) * grid.voxel_size),
                ) + grid.origin;
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
                (grid.origin, bounds_max, grid.voxel_size, density_range)
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
            context.generated = local_center;
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
        let params = evaluated_material;
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
        volumes.push(VolumeInstance {
            volume,
            world_to_local: matrix.inverse(),
            bounds_min,
            bounds_max,
            voxel_size,
            meters_per_unit,
            density_range,
            albedo,
            anisotropy: params.map_or(0.0, |params| params.volume_anisotropy),
            emission,
            scattering,
            extinction_scale,
            sampling: params.map_or(crate::model::VolumeSampling::default(), |params| {
                params.volume_sampling
            }),
            interpolation: params.map_or(crate::model::VolumeInterpolation::default(), |params| {
                params.volume_interpolation
            }),
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

/// A closed mesh's medium on the path's volume stack, as Cycles' object volume stack: the coefficients of the
/// surface that entered it. A surface of the same object leaves it.
#[derive(Clone, Copy)]
struct MeshMedium {
    object_index: u32,
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

/// Spectral ratio tracking over a homogeneous medium segment of length `distance`
/// (`intern/cycles/kernel/integrator/shade_volume.h`, `volume_integrate_heterogeneous`). Tentative collisions
/// come at the majorant `max_c σt_c`. Each is a scatter, a null collision or an absorption, chosen with the
/// channel-mean probabilities, and the throughput takes each channel's own branch weight. Every channel's
/// transmittance and single-scatter integral is then exact in expectation.
fn trace_mesh_medium(
    medium: &BsdfParams,
    distance: f64,
    throughput: &mut DVec3,
    rng: &mut Pcg32,
) -> MediumOutcome {
    let scatter = DVec3::from_array(medium.volume_scatter);
    let absorption = DVec3::from_array(medium.volume_absorption);
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

/// Pushes the medium of `object_index` when the path enters it, or pops it when the path leaves it.
fn toggle_mesh_medium(media: &mut Vec<MeshMedium>, object_index: u32, params: &BsdfParams) {
    if let Some(position) = media
        .iter()
        .rposition(|medium| medium.object_index == object_index)
    {
        media.remove(position);
    } else {
        media.push(MeshMedium {
            object_index,
            params: *params,
        });
    }
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
    for instance in volumes {
        if !groups
            .iter()
            .any(|group| crate::float::equal_f64(group.anisotropy, instance.anisotropy))
        {
            groups.push(ScatterGroup {
                anisotropy: instance.anisotropy,
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
            step_emission += instance.emission * density;
            *step_total += step_scatter;
            if let Some(group) = groups
                .iter_mut()
                .find(|group| crate::float::equal_f64(group.anisotropy, instance.anisotropy))
            {
                group.scatter[index] += step_scatter;
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

/// Trace one voxel segment's free flight on its grid, with the distance sampler's events and emission
/// (`volume_integrate_heterogeneous`, with the grid's grey transport).
fn trace_volume_segment(
    grid: Option<&VoxelGrid>,
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    rng: &mut Pcg32,
) -> Result<VolumeSegment> {
    let Some(grid) = grid else {
        return Ok(VolumeSegment {
            transmittance: 1.0,
            emission: DVec3::ZERO,
            event: None,
            stochastic: false,
        });
    };
    let free_flight = if grid.stochastic {
        -rng.next_f64().max(f64::MIN_POSITIVE).ln()
    } else {
        f64::INFINITY
    };
    let step_in_meters = grid.step * grid.meters_per_unit;
    let mut accumulated_depth = 0.0;
    let mut emitted = DVec3::ZERO;
    for (index, &extinction) in grid.extinction.iter().enumerate() {
        if extinction <= 0.0 {
            continue;
        }
        let start = grid.entry + index as f64 * grid.step;
        let source = grid.emission[index];
        let step_depth = extinction * step_in_meters;
        let event_in_step = free_flight <= accumulated_depth + step_depth;
        let traveled_meters = if event_in_step {
            ((free_flight - accumulated_depth) / extinction).clamp(0.0, step_in_meters)
        } else {
            step_in_meters
        };
        let (_, source_distance) = homogeneous_segment_integral(extinction, traveled_meters);
        emitted += source * (-accumulated_depth).exp() * source_distance;
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
                    anisotropy: 0.0,
                },
                |instance| VolumeEvent {
                    point: event_point,
                    albedo: if instance.scattering {
                        instance.albedo
                    } else {
                        DVec3::ZERO
                    },
                    anisotropy: instance.anisotropy,
                },
            );
            return Ok(VolumeSegment {
                transmittance: (-accumulated_depth).exp(),
                emission: emitted,
                event: Some(event),
                stochastic: grid.stochastic,
            });
        }
    }
    Ok(VolumeSegment {
        transmittance: (-accumulated_depth).exp(),
        emission: emitted,
        event: None,
        stochastic: grid.stochastic,
    })
}

fn volume_density_at(instance: &VolumeInstance<'_>, world_position: DVec3) -> Result<f64> {
    let local_position = instance.world_to_local.transform_point3(world_position);
    let density = sample_density_with(local_position, instance.volume, instance.interpolation);
    if !density.is_finite() {
        sample_density_checked_with(local_position, instance.volume, instance.interpolation)?;
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "volume density is not finite",
        ));
    }
    Ok(density.max(0.0))
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
                    let density = sample_density_with(
                        local_position,
                        instance.volume,
                        instance.interpolation,
                    );
                    if !density.is_finite() {
                        sample_density_checked_with(
                            local_position,
                            instance.volume,
                            instance.interpolation,
                        )?;
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
            let density =
                sample_density_with(local_position, instance.volume, instance.interpolation);
            if !density.is_finite() {
                sample_density_checked_with(
                    local_position,
                    instance.volume,
                    instance.interpolation,
                )?;
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
            anisotropy: 0.0,
            emission: DVec3::ZERO,
            scattering: false,
            extinction_scale: 1.0,
            sampling: crate::model::VolumeSampling::MultipleImportance,
            interpolation: crate::model::VolumeInterpolation::Linear,
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

    /// The mean of `sample_bsdf`'s weight (`f · cos / p_mix`) over the sampler's own directions is the
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
                    total += evaluate_bsdf(&params, &layers(&params, normal, view), light)
                        * (cosine * theta.sin() * d_theta * d_phi);
                }
            }
            total
        };
        let coarse = quadrature(400, 200);
        let fine = quadrature(800, 400);

        let samples = 400_000_usize;
        let mut rng = Pcg32::for_sample(1, 0, 0, 0);
        let mut sum = DVec3::ZERO;
        let mut sum_squares = DVec3::ZERO;
        for _ in 0..samples {
            let weight = match sample_bsdf(&params, &layers(&params, normal, view), &mut rng) {
                BsdfSample::Continuous { weight, .. } | BsdfSample::Delta { weight, .. } => weight,
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
}
