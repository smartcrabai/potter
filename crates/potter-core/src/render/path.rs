use std::{collections::BTreeMap, path::Path};

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::volume::{VolumeData, sample_density, sample_density_checked},
    image::{ImageData, ImageInterpolation},
    model::{Id, LightType, SceneDoc},
    shader::{BsdfParams, HitContext, evaluate_surface},
};

use super::{
    bvh::{Bvh, BvhHit},
    raster::{
        Camera, Mode, RasterOutput, StereoMode, camera_eye, rasterize_with_lines, ray_for_sample,
    },
    scene::{Geometry, ObjectRecord},
};

const RAY_EPSILON: f64 = 1.0e-5;
const MAX_VOLUME_STEPS: usize = 2048;
const MAX_SHADOW_LAYERS: usize = 16;
// Calibrated against the Blender-gated Cycles VDB comparison; surface lighting uses its own energy convention.
const VOLUME_DIRECT_LIGHT_SCALE: f64 = 0.75;

#[derive(Clone, Copy)]
struct PointLight {
    position: DVec3,
    direction: DVec3,
    kind: LightType,
    color: DVec3,
    energy: f64,
    radius: f64,
    spot_size: f64,
    spot_blend: f64,
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
    density_scale: f64,
    albedo: DVec3,
    anisotropy: f64,
    emission: DVec3,
    scattering: bool,
    absorption_tint: Option<DVec3>,
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
    params: BsdfParams,
    object_index: u32,
    element_index: u32,
    distance: f64,
}

#[derive(Clone, Copy)]
struct LightSample {
    direction: DVec3,
    distance: f64,
    radiance: DVec3,
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
                let pixel = image.pixels.get(pixel_index).ok_or_else(|| {
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
        let longitude = (u - 0.5) * std::f64::consts::TAU;
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
        let u = (direction.y.atan2(direction.x) / std::f64::consts::TAU + 0.5).rem_euclid(1.0);
        let v = direction.z.clamp(-1.0, 1.0).acos() / std::f64::consts::PI;
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
    let lights = collect_lights(doc, snapshot);
    let volumes = collect_volumes(doc, snapshot, scene.unit.scale_length, images)?;
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
                    let (left_origin, left_direction) =
                        ray_for_sample(&left_camera, width, height, x, y, sample_x, sample_y)?;
                    let (right_origin, right_direction) =
                        ray_for_sample(&right_camera, width, height, x, y, sample_x, sample_y)?;
                    let mut left_rng = rng.clone();
                    let mut right_rng = rng.clone();
                    let (left, left_alpha) = trace_path(
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
                        &mut left_rng,
                    )?;
                    let (right, right_alpha) = trace_path(
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
                        &mut right_rng,
                    )?;
                    (
                        DVec3::new(left.x, right.y, right.z),
                        f64::midpoint(left_alpha, right_alpha),
                    )
                } else {
                    let (origin, direction) =
                        ray_for_sample(camera, width, height, x, y, sample_x, sample_y)?;
                    trace_path(
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
                        &mut rng,
                    )?
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
) -> Result<()> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let volumes = collect_volumes(doc, snapshot, scene.unit.scale_length, images)?;
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
    let camera_forward = (camera.target - camera.position).normalize_or_zero();
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
            let (origin, direction) = ray_for_sample(camera, width, height, x, y, 0.5, 0.5)?;
            let max_distance = if output.ids[index] == 0 {
                f64::INFINITY
            } else {
                let projected_depth = f64::from(output.depths[index]);
                projected_depth / direction.dot(camera_forward).max(f64::MIN_POSITIVE)
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
            let density_scaled = density * instance.density_scale.max(0.0);
            let scattering_coefficient = if instance.scattering {
                density_scaled
            } else {
                0.0
            };
            extinction += volume_extinction_density(instance, density);
            source += instance.emission * density_scaled;
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
    rng: &mut Pcg32,
) -> Result<(DVec3, f64)> {
    let mut radiance = DVec3::ZERO;
    let mut throughput = DVec3::ONE;
    let mut alpha = 0.0;
    let mut previous_diffuse = false;
    for bounce in 0..=max_bounces {
        let hit = closest_hit(geometry, bvh, origin, direction, RAY_EPSILON, f64::INFINITY);
        let segment_limit = hit.map_or(f64::INFINITY, |hit| hit.distance);
        let environment = world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        let volume_segment = trace_volume_segment(volumes, origin, direction, segment_limit, rng)?;
        radiance += throughput * volume_segment.emission;
        if !volume_segment.stochastic {
            throughput *= volume_segment.transmittance;
        }
        if throughput.max_element() <= 1.0e-8 {
            break;
        }
        if bounce == 0 {
            alpha = 1.0 - volume_segment.transmittance;
        }
        if let Some(event) = volume_segment.event {
            let scatter_probability = (event.albedo.x + event.albedo.y + event.albedo.z) / 3.0;
            if scatter_probability <= 0.0 || rng.next_f64() >= scatter_probability {
                break;
            }
            throughput *= event.albedo / scatter_probability;
            radiance += throughput
                * direct_volume_lighting(
                    doc,
                    snapshot,
                    geometry,
                    bvh,
                    lights,
                    volumes,
                    images,
                    importance,
                    event,
                    direction,
                    environment,
                    rng,
                )?;
            if bounce == max_bounces {
                break;
            }
            direction = sample_henyey_greenstein_direction(direction, event.anisotropy, rng);
            origin = event.point + direction * RAY_EPSILON;
            previous_diffuse = true;
            continue;
        }
        let Some(hit) = hit else {
            let environment_weight = if previous_diffuse { 0.5 } else { 1.0 };
            radiance += throughput * environment * environment_weight;
            break;
        };
        let surface = make_surface(doc, snapshot, geometry, hit, origin, direction, images)?;
        if surface.params.volume_density > 0.0
            && let Some(boundary) = closest_hit(
                geometry,
                bvh,
                surface.point + direction * RAY_EPSILON,
                direction,
                RAY_EPSILON,
                f64::INFINITY,
            )
        {
            let exits_medium = geometry
                .triangles
                .get(boundary.triangle_index)
                .is_some_and(|triangle| triangle.object_index == surface.object_index);
            let distance = boundary.distance.max(0.0);
            let scalar_transmittance = (-surface.params.volume_density * distance).exp();
            let transmittance = if surface.params.volume_scattering {
                DVec3::splat(scalar_transmittance)
            } else {
                let absorption = DVec3::from_array(surface.params.volume_color)
                    .clamp(DVec3::splat(1.0e-8), DVec3::ONE);
                DVec3::new(
                    absorption.x.powf(surface.params.volume_density * distance),
                    absorption.y.powf(surface.params.volume_density * distance),
                    absorption.z.powf(surface.params.volume_density * distance),
                )
            };
            if surface.params.volume_scattering {
                let phase = henyey_greenstein(-1.0, surface.params.volume_anisotropy)
                    * (4.0 * std::f64::consts::PI);
                radiance += throughput
                    * environment
                    * DVec3::from_array(surface.params.volume_color)
                    * ((1.0 - scalar_transmittance) * phase);
            }
            throughput *= transmittance;
            if bounce == 0 {
                alpha = 1.0 - transmittance.element_sum() / 3.0;
            }
            if exits_medium {
                origin = surface.point + direction * (distance + 2.0 * RAY_EPSILON);
            } else {
                origin = surface.point + direction * (distance - 2.0 * RAY_EPSILON);
            }
            if throughput.max_element() <= 1.0e-8 {
                break;
            }
            continue;
        }
        if bounce == 0 {
            alpha = 1.0;
        }
        if rng.next_f64() >= surface.params.alpha.clamp(0.0, 1.0) {
            origin = offset_origin(surface.point, direction, surface.normal);
            continue;
        }
        radiance += throughput
            * DVec3::from_array(surface.params.emission_color)
            * surface.params.emission_strength.max(0.0);
        radiance += throughput
            * direct_lighting(
                doc,
                snapshot,
                geometry,
                bvh,
                lights,
                volumes,
                images,
                importance,
                &surface,
                direction,
                environment,
                rng,
            )?;
        if bounce == max_bounces {
            break;
        }
        let (next_direction, weight, diffuse_event) = sample_bsdf(&surface.params, -direction, rng);
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
        previous_diffuse = diffuse_event;
        direction = next_direction.normalize_or_zero();
        origin = offset_origin(surface.point, direction, surface.normal);
    }
    if !radiance.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "path tracer produced a non-finite radiance value",
        ));
    }
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
) -> Result<()> {
    let camera_forward = (camera.target - camera.position).normalize_or_zero();
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
            let (origin, direction) = ray_for_sample(camera, width, height, x, y, 0.5, 0.5)?;
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
            let surface = make_surface(doc, snapshot, geometry, hit, origin, direction, images)?;
            if surface.params.alpha <= 0.0 {
                continue;
            }
            output.ids[index] = surface.object_index;
            output.elements[index] = surface.element_index;
            output.depths[index] =
                finite_f32((surface.point - camera.position).dot(camera_forward));
            output.normals[index] = surface.normal.to_array().map(finite_f32);
            output.ambient_occlusion[index] = finite_f32(super::raster::ambient_visibility(
                bvh,
                &geometry.triangles,
                surface.point,
                surface.normal,
                1.0,
            ));
            output.albedo[index] = surface.params.base_color.map(finite_f32);
            output.emission[index] = [
                finite_f32(surface.params.emission_color[0] * surface.params.emission_strength),
                finite_f32(surface.params.emission_color[1] * surface.params.emission_strength),
                finite_f32(surface.params.emission_color[2] * surface.params.emission_strength),
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

fn make_surface(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
    hit: BvhHit,
    origin: DVec3,
    direction: DVec3,
    images: &BTreeMap<String, ImageData>,
) -> Result<Surface> {
    let triangle = geometry.triangles.get(hit.triangle_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "BVH returned an invalid triangle index",
        )
    })?;
    let point = origin + direction * hit.distance;
    let mut normal = (triangle.positions[1] - triangle.positions[0])
        .cross(triangle.positions[2] - triangle.positions[0])
        .normalize_or_zero();
    if normal.dot(-direction) < 0.0 {
        normal = -normal;
    }
    let uv = [
        triangle.uv[0][0] * hit.barycentric[0]
            + triangle.uv[1][0] * hit.barycentric[1]
            + triangle.uv[2][0] * hit.barycentric[2],
        triangle.uv[0][1] * hit.barycentric[0]
            + triangle.uv[1][1] * hit.barycentric[1]
            + triangle.uv[2][1] * hit.barycentric[2],
    ];
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
    let matrix = snapshot
        .nodes
        .get(&object.node_id)
        .map_or(DMat4::IDENTITY, |node| {
            DMat4::from_cols_array(&node.world_matrix)
        });
    let world_to_local = matrix.inverse();
    let local_is_finite = world_to_local
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite());
    let object_position = if local_is_finite {
        world_to_local.transform_point3(point)
    } else {
        point
    };
    let uv_maps = [("uv_map", uv)];
    let (tangent, bitangent) = triangle_tangent_frame(triangle, normal);
    let mut context = HitContext::new(&doc.node_groups)
        .with_images(images)
        .with_tangent_frame(tangent, bitangent);
    if triangle.uv_available {
        context = context.with_uv_maps(&uv_maps);
    }
    context.uv = uv;
    context.generated = object_position;
    context.position = point;
    context.object = object_position;
    context.normal = normal;
    context.view_direction = -direction;
    let mut params = evaluate_surface(material, &context)?;
    params.alpha = match material.alpha_mode.as_str() {
        "opaque" => 1.0,
        "clip" => {
            if params.alpha >= material.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
        _ => params.alpha.clamp(0.0, 1.0),
    };
    if params.displacement_height != 0.0 && material.displacement_method != "displacement" {
        params.normal = displacement_bump_normal(material, context, tangent, bitangent, normal)?;
    }
    params.normal = if params.normal.is_finite() {
        params.normal.normalize_or_zero()
    } else {
        normal
    };
    if params.normal.dot(-direction) < 0.0 {
        params.normal = -params.normal;
    }
    Ok(Surface {
        point,
        normal: params.normal,
        params,
        object_index: triangle.object_index,
        element_index: triangle.element_index,
        distance: hit.distance,
    })
}

fn displacement_bump_normal(
    material: &crate::model::Material,
    context: HitContext<'_>,
    tangent: DVec3,
    bitangent: DVec3,
    normal: DVec3,
) -> Result<DVec3> {
    let step = 1.0e-3;
    let positive_u = displacement_sample(material, context, [step, 0.0], tangent, bitangent)?;
    let negative_u = displacement_sample(material, context, [-step, 0.0], tangent, bitangent)?;
    let positive_v = displacement_sample(material, context, [0.0, step], tangent, bitangent)?;
    let negative_v = displacement_sample(material, context, [0.0, -step], tangent, bitangent)?;
    let gradient_u = (positive_u - negative_u) / (2.0 * step);
    let gradient_v = (positive_v - negative_v) / (2.0 * step);
    Ok((normal - tangent * gradient_u - bitangent * gradient_v).normalize_or_zero())
}

fn displacement_sample(
    material: &crate::model::Material,
    context: HitContext<'_>,
    uv_offset: [f64; 2],
    tangent: DVec3,
    bitangent: DVec3,
) -> Result<f64> {
    let displacement = tangent * uv_offset[0] + bitangent * uv_offset[1];
    let mut sample_context = context;
    sample_context.uv[0] += uv_offset[0];
    sample_context.uv[1] += uv_offset[1];
    sample_context.generated += displacement;
    sample_context.position += displacement;
    sample_context.object += displacement;
    evaluate_surface(material, &sample_context).map(|params| params.displacement_height)
}

fn object_for_index(objects: &[ObjectRecord], object_index: u32) -> Option<&ObjectRecord> {
    object_index
        .checked_sub(1)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| objects.get(index))
}

fn triangle_tangent_frame(triangle: &super::raster::Triangle, normal: DVec3) -> (DVec3, DVec3) {
    let edge1 = triangle.positions[1] - triangle.positions[0];
    let edge2 = triangle.positions[2] - triangle.positions[0];
    let uv1 = [
        triangle.uv[1][0] - triangle.uv[0][0],
        triangle.uv[1][1] - triangle.uv[0][1],
    ];
    let uv2 = [
        triangle.uv[2][0] - triangle.uv[0][0],
        triangle.uv[2][1] - triangle.uv[0][1],
    ];
    let determinant = uv1[0] * uv2[1] - uv1[1] * uv2[0];
    let (raw_tangent, raw_bitangent) =
        if determinant.is_finite() && determinant.abs() > f64::EPSILON {
            (
                (edge1 * uv2[1] - edge2 * uv1[1]) / determinant,
                (edge2 * uv1[0] - edge1 * uv2[0]) / determinant,
            )
        } else {
            (edge1, normal.cross(edge1))
        };
    let tangent = (raw_tangent - normal * raw_tangent.dot(normal)).normalize_or_zero();
    let fallback = if normal.z.abs() < 0.9 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let tangent = if tangent == DVec3::ZERO {
        fallback
    } else {
        tangent
    };
    let bitangent =
        (raw_bitangent - normal * raw_bitangent.dot(normal) - tangent * raw_bitangent.dot(tangent))
            .normalize_or_zero();
    let bitangent = if bitangent == DVec3::ZERO {
        normal.cross(tangent).normalize_or_zero()
    } else {
        bitangent
    };
    (tangent, bitangent)
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
    images: &BTreeMap<String, ImageData>,
    importance: Option<&WorldImportanceSampler>,
    surface: &Surface,
    incoming_direction: DVec3,
    environment: DVec3,
    rng: &mut Pcg32,
) -> Result<DVec3> {
    let normal = surface.normal;
    let view = -incoming_direction;
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
            images,
            surface.point,
            normal,
            sample.direction,
            sample.distance,
            rng,
        )?;
        if visibility <= 0.0 {
            continue;
        }
        contribution += evaluate_bsdf(&surface.params, normal, view, sample.direction)
            * sample.radiance
            * (cosine * visibility);
    }
    let (direction, cosine_pdf) = sample_cosine_hemisphere(normal, rng);
    let cosine = normal.dot(direction).max(0.0);
    if cosine > 0.0 && cosine_pdf > 0.0 {
        let environment_sample =
            world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
        let visibility = shadow_transmittance(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            images,
            surface.point,
            normal,
            direction,
            f64::INFINITY,
            rng,
        )?;
        let environment_pdf = importance.map_or(0.0, |sampler| sampler.pdf(direction));
        let weight = importance.map_or(1.0, |_| power_heuristic(cosine_pdf, environment_pdf));
        contribution += evaluate_bsdf(&surface.params, normal, view, direction)
            * environment_sample
            * (cosine / cosine_pdf * visibility * 0.5 * weight);
    }
    if let Some(sampler) = importance {
        let (direction, environment_pdf) = sampler.sample(rng);
        let cosine = normal.dot(direction).max(0.0);
        if cosine > 0.0 && environment_pdf > 0.0 {
            let environment_sample =
                world_radiance(doc, &snapshot.scene_id, images, direction, environment)?;
            let visibility = shadow_transmittance(
                doc,
                snapshot,
                geometry,
                bvh,
                volumes,
                images,
                surface.point,
                normal,
                direction,
                f64::INFINITY,
                rng,
            )?;
            let cosine_pdf = cosine / std::f64::consts::PI;
            let weight = power_heuristic(environment_pdf, cosine_pdf);
            contribution += evaluate_bsdf(&surface.params, normal, view, direction)
                * environment_sample
                * (cosine / environment_pdf * visibility * 0.5 * weight);
        }
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
    volumes: &[VolumeInstance<'_>],
    images: &BTreeMap<String, ImageData>,
    importance: Option<&WorldImportanceSampler>,
    event: VolumeEvent,
    incoming_direction: DVec3,
    environment: DVec3,
    rng: &mut Pcg32,
) -> Result<DVec3> {
    let mut contribution = DVec3::ZERO;
    for light in lights {
        let Some(sample) = sample_light(light, event.point, rng) else {
            continue;
        };
        if sample.radiance.max_element() <= 0.0 {
            continue;
        }
        let visibility = shadow_transmittance(
            doc,
            snapshot,
            geometry,
            bvh,
            volumes,
            images,
            event.point,
            DVec3::ZERO,
            sample.direction,
            sample.distance,
            rng,
        )?;
        let phase = henyey_greenstein(incoming_direction.dot(sample.direction), event.anisotropy);
        contribution += sample.radiance * (phase * visibility * VOLUME_DIRECT_LIGHT_SCALE);
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
            images,
            event.point,
            DVec3::ZERO,
            direction,
            f64::INFINITY,
            rng,
        )?;
        let phase = henyey_greenstein(incoming_direction.dot(direction), event.anisotropy);
        contribution += incoming * (phase * visibility * 0.5 / pdf);
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
    images: &BTreeMap<String, ImageData>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
    rng: &mut Pcg32,
) -> Result<f64> {
    let mut transmittance = 1.0;
    let mut origin = offset_origin(point, direction, normal);
    let mut remaining = if distance.is_finite() {
        (distance - RAY_EPSILON).max(0.0)
    } else {
        f64::INFINITY
    };
    for _ in 0..MAX_SHADOW_LAYERS {
        let Some(hit) = closest_hit(geometry, bvh, origin, direction, RAY_EPSILON, remaining)
        else {
            break;
        };
        let surface = make_surface(doc, snapshot, geometry, hit, origin, direction, images)?;
        let alpha = surface.params.alpha.clamp(0.0, 1.0);
        let transmission = surface.params.transmission.clamp(0.0, 1.0);
        if alpha >= 1.0 && transmission <= 0.0 {
            return Ok(0.0);
        }
        let tint = DVec3::new(
            surface.params.base_color[0],
            surface.params.base_color[1],
            surface.params.base_color[2],
        );
        transmittance *= (1.0 - alpha) + alpha * transmission * tint.max_element().clamp(0.0, 1.0);
        if transmittance <= 1.0e-5 {
            return Ok(0.0);
        }
        let advance = surface.distance + RAY_EPSILON;
        origin += direction * advance;
        if remaining.is_finite() {
            remaining = (remaining - advance).max(0.0);
            if remaining <= RAY_EPSILON {
                break;
            }
        }
    }
    let volume_transmittance = march_volumes(volumes, origin, direction, remaining, Some(rng))?;
    Ok((transmittance * volume_transmittance).clamp(0.0, 1.0))
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
        });
    }
    let target = if light.kind == LightType::Area && light.radius > 0.0 {
        let tangent = if light.direction.z.abs() < 0.9 {
            light.direction.cross(DVec3::Z).normalize_or_zero()
        } else {
            light.direction.cross(DVec3::Y).normalize_or_zero()
        };
        let bitangent = light.direction.cross(tangent).normalize_or_zero();
        let radius = light.radius * rng.next_f64().sqrt();
        let angle = rng.next_f64() * std::f64::consts::TAU;
        light.position + tangent * (radius * angle.cos()) + bitangent * (radius * angle.sin())
    } else if light.kind == LightType::Point && light.radius > 0.0 {
        let z = 1.0 - 2.0 * rng.next_f64();
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let radial = (1.0 - z * z).max(0.0).sqrt();
        light.position + DVec3::new(radial * angle.cos(), radial * angle.sin(), z) * light.radius
    } else {
        light.position
    };
    let offset = target - point;
    let distance_squared = offset.length_squared();
    if distance_squared <= f64::EPSILON || !distance_squared.is_finite() {
        return None;
    }
    let distance = distance_squared.sqrt();
    let direction = offset / distance;
    let mut attenuation = 1.0 / distance_squared;
    if light.kind == LightType::Spot {
        let cosine = light.direction.normalize_or_zero().dot(-direction);
        let outer = (light.spot_size.clamp(0.0, std::f64::consts::TAU) * 0.5).cos();
        let blend = light.spot_blend.clamp(0.0, 1.0) * (1.0 - outer);
        let inner = (outer + blend).min(1.0);
        if cosine <= outer {
            return None;
        }
        attenuation *= if inner <= outer {
            1.0
        } else {
            ((cosine - outer) / (inner - outer)).clamp(0.0, 1.0)
        };
    }
    Some(LightSample {
        direction,
        distance,
        radiance: light.color * (light.energy * attenuation),
    })
}

fn sample_bsdf(params: &BsdfParams, view: DVec3, rng: &mut Pcg32) -> (DVec3, DVec3, bool) {
    let diffuse_weight = ((1.0 - params.metallic) * (1.0 - params.transmission)).max(0.0);
    let transmission_weight = params.transmission.clamp(0.0, 1.0);
    let specular_weight = (params.metallic.clamp(0.0, 1.0)
        + 0.04 * (1.0 - params.metallic.clamp(0.0, 1.0)))
    .max(1.0e-4);
    let total = diffuse_weight + transmission_weight + specular_weight;
    let p_diffuse = diffuse_weight / total;
    let p_transmission = transmission_weight / total;
    let choice = rng.next_f64();
    if choice < p_transmission {
        let direction = refract(-view, params.normal, 1.0 / params.ior.max(1.0e-3))
            .unwrap_or_else(|| reflect(-view, params.normal));
        let tint = DVec3::new(
            params.base_color[0],
            params.base_color[1],
            params.base_color[2],
        );
        return (
            direction,
            tint * (transmission_weight / p_transmission.max(1.0e-8)),
            false,
        );
    }
    if choice >= p_transmission + p_diffuse {
        let half_vector = sample_ggx_half(params.normal, params.roughness, rng);
        let direction = reflect(-view, half_vector).normalize_or_zero();
        let cosine = params.normal.dot(direction).max(0.0);
        let half_cosine = params.normal.dot(half_vector).max(0.0);
        let alpha = params.roughness.max(0.02).powi(2);
        let denominator = half_cosine * half_cosine * (alpha * alpha - 1.0) + 1.0;
        let distribution =
            alpha * alpha / (std::f64::consts::PI * denominator * denominator).max(1.0e-12);
        let pdf_specular =
            distribution * half_cosine / (4.0 * view.dot(half_vector).abs()).max(1.0e-8);
        let pdf = (1.0 - p_transmission) * pdf_specular;
        let bsdf = evaluate_bsdf(params, params.normal, view, direction);
        return (direction, bsdf * (cosine / pdf.max(1.0e-8)), false);
    }
    let (direction, _) = sample_cosine_hemisphere(params.normal, rng);
    let cosine = params.normal.dot(direction).max(0.0);
    let pdf = p_diffuse * cosine / std::f64::consts::PI;
    let bsdf = evaluate_bsdf(params, params.normal, view, direction);
    (direction, bsdf * (cosine / pdf.max(1.0e-8)), true)
}

fn evaluate_bsdf(params: &BsdfParams, normal: DVec3, view: DVec3, light: DVec3) -> DVec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return DVec3::ZERO;
    }
    let half = (view + light).normalize_or_zero();
    let n_dot_h = normal.dot(half).max(0.0);
    let v_dot_h = view.dot(half).max(0.0);
    let albedo = DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
    .max(DVec3::ZERO);
    let f0 = DVec3::splat(0.04 * params.specular_ior_level.clamp(0.0, 2.0))
        .lerp(albedo, params.metallic.clamp(0.0, 1.0));
    let fresnel = f0 + (DVec3::ONE - f0) * (1.0 - v_dot_h).powi(5);
    let alpha = params.roughness.clamp(0.02, 1.0).powi(2);
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    let distribution =
        alpha_squared / (std::f64::consts::PI * denominator * denominator).max(1.0e-12);
    let geometry_v = smith_g1(n_dot_v, alpha_squared);
    let geometry_l = smith_g1(n_dot_l, alpha_squared);
    let specular =
        fresnel * (distribution * geometry_v * geometry_l / (4.0 * n_dot_v * n_dot_l).max(1.0e-8));
    let diffuse = albedo
        * ((1.0 - params.metallic.clamp(0.0, 1.0)) * (1.0 - params.transmission.clamp(0.0, 1.0))
            / std::f64::consts::PI);
    diffuse + specular
}

fn smith_g1(cosine: f64, alpha_squared: f64) -> f64 {
    let cosine = cosine.max(0.0);
    2.0 * cosine
        / (cosine + (alpha_squared + (1.0 - alpha_squared) * cosine * cosine).sqrt()).max(1.0e-8)
}

fn sample_ggx_half(normal: DVec3, roughness: f64, rng: &mut Pcg32) -> DVec3 {
    let alpha = roughness.max(0.02).powi(2);
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
                spot_size: data.spot_size,
                spot_blend: data.spot_blend,
            })
        })
        .collect()
}

fn collect_volumes<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    meters_per_unit: f64,
    images: &BTreeMap<String, ImageData>,
) -> Result<Vec<VolumeInstance<'a>>> {
    let mut volumes = Vec::new();
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
        let material = node
            .materials
            .first()
            .and_then(|material_id| doc.materials.get(material_id));
        let evaluated_material = if let Some(material) = material {
            let local_center = (bounds_min + bounds_max) * 0.5;
            let mut context = HitContext::new(&doc.node_groups).with_images(images);
            context.object = local_center;
            context.generated = local_center;
            context.position = matrix.transform_point3(local_center);
            Some(evaluate_surface(material, &context)?)
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
        let scattering = params.is_none_or(|params| params.volume_scattering) || simple_volume;
        let density_scale = params.map_or(1.0, |params| {
            if has_volume_output || simple_volume {
                params.volume_density.max(0.0)
            } else {
                1.0
            }
        });
        let albedo = if scattering {
            params.map_or(DVec3::splat(0.5), |params| {
                DVec3::from_array(params.volume_color).clamp(DVec3::ZERO, DVec3::ONE)
            })
        } else {
            DVec3::ZERO
        };
        let absorption_tint = (!scattering && has_volume_output).then(|| {
            DVec3::from_array(params.map_or([1.0; 3], |params| params.volume_color))
                .clamp(DVec3::splat(1.0e-8), DVec3::ONE)
        });
        let emission = if has_volume_output || simple_volume {
            params.map_or(DVec3::ZERO, |params| {
                DVec3::from_array(params.volume_emission_color)
                    * params.volume_emission_strength.max(0.0)
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
            density_scale,
            albedo,
            anisotropy: params.map_or(0.0, |params| params.volume_anisotropy),
            emission,
            scattering,
            absorption_tint,
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

fn volume_extinction_density(instance: &VolumeInstance<'_>, density: f64) -> f64 {
    let absorption_scale = instance.absorption_tint.map_or(1.0, |tint| {
        (-(tint.x.ln() + tint.y.ln() + tint.z.ln()) / 3.0).max(0.0)
    });
    density.max(0.0) * instance.density_scale.max(0.0) * absorption_scale
}

fn trace_volume_segment(
    volumes: &[VolumeInstance<'_>],
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    rng: &mut Pcg32,
) -> Result<VolumeSegment> {
    let mut entry = f64::INFINITY;
    let mut exit = 0.0_f64;
    let mut minimum_step = f64::INFINITY;
    let mut stochastic = false;
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
                && instance.density_scale > 0.0;
        }
    }
    if !entry.is_finite() || exit <= entry {
        return Ok(VolumeSegment {
            transmittance: 1.0,
            emission: DVec3::ZERO,
            event: None,
            stochastic: false,
        });
    }
    let segment_length = exit - entry;
    let steps = (segment_length / minimum_step)
        .ceil()
        .clamp(1.0, MAX_VOLUME_STEPS as f64) as usize;
    let step = segment_length / steps as f64;
    let meters_per_unit = volumes
        .first()
        .map_or(1.0, |instance| instance.meters_per_unit);
    let free_flight = if stochastic {
        -rng.next_f64().max(f64::MIN_POSITIVE).ln()
    } else {
        f64::INFINITY
    };
    let mut accumulated_depth = 0.0;
    let mut emitted = DVec3::ZERO;
    for sample_index in 0..steps {
        let distance = entry + (sample_index as f64 + 0.5) * step;
        let point = origin + direction * distance;
        let mut extinction = 0.0;
        let mut source = DVec3::ZERO;
        for instance in volumes {
            let density = volume_density_at(instance, point)?;
            extinction += volume_extinction_density(instance, density);
            source += instance.emission * (density * instance.density_scale.max(0.0));
        }
        if extinction <= 0.0 {
            continue;
        }
        let step_in_meters = step * meters_per_unit;
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
            let event_world_distance = distance - step * 0.5 + traveled_meters / meters_per_unit;
            let event_point = origin + direction * event_world_distance;
            let selected = rng.next_f64() * extinction;
            let mut weight = 0.0;
            let mut chosen = None;
            for instance in volumes {
                let density = volume_density_at(instance, point)?;
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
                stochastic,
            });
        }
    }
    Ok(VolumeSegment {
        transmittance: (-accumulated_depth).exp(),
        emission: emitted,
        event: None,
        stochastic,
    })
}

fn volume_density_at(instance: &VolumeInstance<'_>, world_position: DVec3) -> Result<f64> {
    let local_position = instance.world_to_local.transform_point3(world_position);
    let density = sample_density(local_position, instance.volume);
    if !density.is_finite() {
        sample_density_checked(local_position, instance.volume)?;
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
                    let density = sample_density(local_position, instance.volume);
                    if !density.is_finite() {
                        sample_density_checked(local_position, instance.volume)?;
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
            let density = sample_density(local_position, instance.volume);
            if !density.is_finite() {
                sample_density_checked(local_position, instance.volume)?;
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
            density_scale: 1.0,
            albedo: DVec3::ZERO,
            anisotropy: 0.0,
            emission: DVec3::ZERO,
            scattering: false,
            absorption_tint: None,
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
