//! The world's radiance as a cube map, built once per render.
//!
//! EEVEE renders the world into one sphere probe, and every surface reads that probe. `LightProbeModule::sync_world`
//! (`eevee_lightprobe.cc`) marks the world probe for rendering only when the world changes, so the World node graph is
//! not evaluated per surface sample. Realtime Beauty's environment term used to evaluate the graph at each of its taps,
//! three times per shaded sample, so a render's cost grew with its sample count. The probe holds `evaluate_world` at the
//! centre of each texel, and a lookup returns the texel that holds the direction.

use std::collections::BTreeMap;

use glam::{DVec3, Vec3};

use crate::{
    error::Result,
    graph::NodeGroup,
    image::ImageData,
    model::{Registry, World},
    shader::evaluate_world,
};

/// Texels along each cube face's edge.
const FACE_TEXELS: usize = 64;
/// EEVEE's sphere-probe capture face extent at the default 512 cubemap setting.
const VOLUME_PROBE_FACE_TEXELS: usize = 256;
const SPHERE_PROBE_REMAP_GROUP_SIZE: usize = 32;
const SPHERE_PROBE_SH_GROUP_SIZE: usize = 256;

/// The world's radiance by direction, sampled at the centre of each texel of a cube map.
pub(super) struct WorldProbe {
    /// The six faces in the order of `face_direction`, each `FACE_TEXELS` by `FACE_TEXELS`, row by row.
    radiance: Vec<DVec3>,
    /// EEVEE's non-directional spherical-harmonic L0 for sphere-thickness and isotropic volume lighting.
    average_irradiance: DVec3,
}

impl WorldProbe {
    /// Evaluates the world at every texel centre.
    pub(super) fn build(
        world: &World,
        groups: &Registry<NodeGroup>,
        images: &BTreeMap<String, ImageData>,
    ) -> Result<Self> {
        let mut radiance = Vec::with_capacity(6 * FACE_TEXELS * FACE_TEXELS);
        let average_irradiance =
            sample_world_probe(world, groups, images, |value| radiance.push(value))?;
        Ok(Self {
            radiance,
            average_irradiance,
        })
    }

    /// The radiance of the texel that `direction` falls in.
    pub(super) fn radiance(&self, direction: DVec3) -> DVec3 {
        let (face, s, t) = cube_coordinates(direction);
        let index = (face * FACE_TEXELS + texel(t)) * FACE_TEXELS + texel(s);
        self.radiance[index]
    }
    /// EEVEE's non-directional world-probe SH L0.
    pub(super) fn average_irradiance(&self) -> DVec3 {
        self.average_irradiance
    }
}
/// EEVEE's isotropic volume-world result: project the half-float sphere probe to L0, then dot
/// its half-float volume-probe coefficient with the isotropic phase (`eevee_volume.bsl.hh`).
/// The second result reports sampled directional variation that can contribute through SH L1.
pub(super) fn volume_ambient_l0(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &BTreeMap<String, ImageData>,
) -> Result<(DVec3, bool)> {
    let mut cubemap = Vec::with_capacity(6 * VOLUME_PROBE_FACE_TEXELS * VOLUME_PROBE_FACE_TEXELS);
    let mut first_radiance: Option<[f32; 3]> = None;
    let mut nonuniform = false;
    for face in 0..6 {
        for row in 0..VOLUME_PROBE_FACE_TEXELS {
            for column in 0..VOLUME_PROBE_FACE_TEXELS {
                let direction = face_direction(
                    face,
                    texel_centre_at(column, VOLUME_PROBE_FACE_TEXELS),
                    texel_centre_at(row, VOLUME_PROBE_FACE_TEXELS),
                );
                let radiance = evaluate_world(world, groups, images, direction, false)?;
                let radiance = [
                    exr::prelude::f16::from_f32(radiance.x as f32).to_f32(),
                    exr::prelude::f16::from_f32(radiance.y as f32).to_f32(),
                    exr::prelude::f16::from_f32(radiance.z as f32).to_f32(),
                ];
                if let Some(first) = first_radiance {
                    nonuniform |= first.map(f32::to_bits) != radiance.map(f32::to_bits);
                } else {
                    first_radiance = Some(radiance);
                }
                cubemap.push(radiance);
            }
        }
    }

    // EEVEE projects the half-float cubemap through its double-resolution octahedral atlas.
    let extent = 2 * VOLUME_PROBE_FACE_TEXELS;
    let groups_per_axis = extent / SPHERE_PROBE_REMAP_GROUP_SIZE;
    let mut coefficients = vec![[0.0_f32; 3]; groups_per_axis * groups_per_axis];
    let mut samples = [[0.0_f32; 3]; SPHERE_PROBE_REMAP_GROUP_SIZE * SPHERE_PROBE_REMAP_GROUP_SIZE];
    let l0 = 0.282_094_8_f32;
    for group_y in 0..groups_per_axis {
        for group_x in 0..groups_per_axis {
            for local_y in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                for local_x in 0..SPHERE_PROBE_REMAP_GROUP_SIZE {
                    let local_index = local_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let column = group_x * SPHERE_PROBE_REMAP_GROUP_SIZE + local_x;
                    let row = group_y * SPHERE_PROBE_REMAP_GROUP_SIZE + local_y;
                    let weight = octahedral_texel_solid_angle(column, row, extent);
                    if weight == 0.0 {
                        samples[local_index] = [0.0; 3];
                        continue;
                    }
                    let denominator = (extent - 2) as f32;
                    let direction =
                        octahedral_direction(column as f32 / denominator, row as f32 / denominator);
                    let radiance = cube_probe_sample(&cubemap, direction);
                    samples[local_index] = radiance.map(|channel| channel * weight);
                }
            }
            let mut stride = samples.len() / 2;
            while stride != 0 {
                for index in 0..stride {
                    let source = samples[index + stride];
                    for (target, source) in samples[index].iter_mut().zip(source) {
                        *target += source;
                    }
                }
                stride /= 2;
            }
            coefficients[group_y * groups_per_axis + group_x] =
                samples[0].map(|radiance| radiance * l0);
        }
    }
    let mut stride = SPHERE_PROBE_SH_GROUP_SIZE / 2;
    while stride != 0 {
        for index in 0..stride {
            let source = coefficients[index + stride];
            for (target, source) in coefficients[index].iter_mut().zip(source) {
                *target += source;
            }
        }
        stride /= 2;
    }
    let coefficient = coefficients[0];
    Ok((
        DVec3::new(
            f64::from(exr::prelude::f16::from_f32(coefficient[0]).to_f32() * l0),
            f64::from(exr::prelude::f16::from_f32(coefficient[1]).to_f32() * l0),
            f64::from(exr::prelude::f16::from_f32(coefficient[2]).to_f32() * l0),
        ),
        nonuniform,
    ))
}

fn octahedral_direction(u: f32, v: f32) -> Vec3 {
    let x = 2.0 * u - 1.0;
    let y = 2.0 * v - 1.0;
    let mut direction = Vec3::new(x, y, 1.0 - (x.abs() + y.abs()));
    if x.abs() + y.abs() > 1.0 {
        let sign_x = if x >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if y >= 0.0 { 1.0 } else { -1.0 };
        direction.x = (y.abs() - 1.0) * -sign_x;
        direction.y = (x.abs() - 1.0) * -sign_y;
    }
    direction
}

fn octahedral_texel_solid_angle(column: usize, row: usize, extent: usize) -> f32 {
    if column == extent - 1 || row == extent - 1 {
        return 0.0;
    }
    let half = extent / 2;
    let padded = extent - 2;
    let wrapped_x = if column >= half {
        padded - column
    } else {
        column
    };
    let wrapped_y = if row >= half { padded - row } else { row };
    let clamp = |coordinate: f32| coordinate.clamp(0.0, (half - 1) as f32);
    let direction = |x: f32, y: f32| {
        octahedral_direction(
            clamp(x) / (extent - 2) as f32,
            clamp(y) / (extent - 2) as f32,
        )
        .normalize()
    };
    let x = wrapped_x as f32;
    let y = wrapped_y as f32;
    let v00 = direction(x - 0.5, y - 0.5);
    let v10 = direction(x + 0.5, y - 0.5);
    let v01 = direction(x - 0.5, y + 0.5);
    let v11 = direction(x + 0.5, y + 0.5);
    let triangle_area = |a: Vec3, b: Vec3, c: Vec3| {
        2.0 * a
            .dot(b.cross(c))
            .abs()
            .atan2(1.0 + b.dot(c) + a.dot(c) + a.dot(b))
    };
    let mut area = triangle_area(v00, v10, v01) + triangle_area(v01, v10, v11);
    if wrapped_x == half - 1 {
        area *= 2.0;
    }
    if wrapped_y == half - 1 {
        area *= 2.0;
    }
    area
}

fn cube_probe_sample(cubemap: &[[f32; 3]], direction: Vec3) -> [f32; 3] {
    let extent = VOLUME_PROBE_FACE_TEXELS as f32;
    let (face, s, t) = cube_coordinates(direction.as_dvec3());
    let x = (f64::midpoint(s, 1.0) * f64::from(extent) - 0.5) as f32;
    let y = (f64::midpoint(t, 1.0) * f64::from(extent) - 0.5) as f32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fraction_x = x - x.floor();
    let fraction_y = y - y.floor();
    let p00 = cube_probe_tap(cubemap, face, x0, y0);
    let p10 = cube_probe_tap(cubemap, face, x0 + 1, y0);
    let p01 = cube_probe_tap(cubemap, face, x0, y0 + 1);
    let p11 = cube_probe_tap(cubemap, face, x0 + 1, y0 + 1);
    std::array::from_fn(|channel| {
        let lower = p00[channel] * (1.0 - fraction_x) + p10[channel] * fraction_x;
        let upper = p01[channel] * (1.0 - fraction_x) + p11[channel] * fraction_x;
        lower * (1.0 - fraction_y) + upper * fraction_y
    })
}

fn cube_probe_tap(cubemap: &[[f32; 3]], face: usize, column: i32, row: i32) -> [f32; 3] {
    let extent = VOLUME_PROBE_FACE_TEXELS as i32;
    if (0..extent).contains(&column) && (0..extent).contains(&row) {
        return cubemap
            [(face * extent as usize + row as usize) * extent as usize + column as usize];
    }
    let s = 2.0 * (column as f64 + 0.5) / f64::from(extent) - 1.0;
    let t = 2.0 * (row as f64 + 0.5) / f64::from(extent) - 1.0;
    let direction = face_direction(face, s, t);
    let (face, s, t) = cube_coordinates(direction);
    let column = (f64::midpoint(s, 1.0) * f64::from(extent)).floor() as usize;
    let row = (f64::midpoint(t, 1.0) * f64::from(extent)).floor() as usize;
    cubemap[(face * extent as usize + row.min(extent as usize - 1)) * extent as usize
        + column.min(extent as usize - 1)]
}

fn sample_world_probe(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &BTreeMap<String, ImageData>,
    mut capture_radiance: impl FnMut(DVec3),
) -> Result<DVec3> {
    let mut average_irradiance = DVec3::ZERO;
    let mut total_solid_angle = 0.0;
    for face in 0..6 {
        for row in 0..FACE_TEXELS {
            for column in 0..FACE_TEXELS {
                let direction = face_direction(face, texel_centre(column), texel_centre(row));
                let radiance = evaluate_world(world, groups, images, direction, false)?;
                let solid_angle = cube_texel_solid_angle(column, row);
                average_irradiance +=
                    super::eevee_transmission::irradiance_probe_radiance(radiance) * solid_angle;
                total_solid_angle += solid_angle;
                capture_radiance(radiance);
            }
        }
    }
    Ok(average_irradiance / total_solid_angle)
}
fn cube_texel_solid_angle(column: usize, row: usize) -> f64 {
    cube_texel_solid_angle_at(column, row, FACE_TEXELS)
}

fn cube_texel_solid_angle_at(column: usize, row: usize, face_texels: usize) -> f64 {
    let edge = face_texels as f64;
    let s0 = 2.0 * column as f64 / edge - 1.0;
    let s1 = 2.0 * (column + 1) as f64 / edge - 1.0;
    let t0 = 2.0 * row as f64 / edge - 1.0;
    let t1 = 2.0 * (row + 1) as f64 / edge - 1.0;
    let area = |s: f64, t: f64| (s * t).atan2((1.0 + s * s + t * t).sqrt());
    area(s1, t1) - area(s0, t1) - area(s1, t0) + area(s0, t0)
}

/// The centre of texel `index` along a face edge, in `-1..1`.
fn texel_centre(index: usize) -> f64 {
    texel_centre_at(index, FACE_TEXELS)
}

fn texel_centre_at(index: usize, face_texels: usize) -> f64 {
    2.0 * (index as f64 + 0.5) / face_texels as f64 - 1.0
}

/// The texel index that holds `coordinate`, in `-1..1`.
fn texel(coordinate: f64) -> usize {
    let index = (f64::midpoint(coordinate, 1.0) * FACE_TEXELS as f64).floor();
    (index.max(0.0) as usize).min(FACE_TEXELS - 1)
}

/// The direction through point `(s, t)` on cube face `face`: +X, -X, +Y, -Y, +Z, -Z in that order.
fn face_direction(face: usize, s: f64, t: f64) -> DVec3 {
    match face {
        0 => DVec3::new(1.0, -t, -s),
        1 => DVec3::new(-1.0, -t, s),
        2 => DVec3::new(s, 1.0, t),
        3 => DVec3::new(s, -1.0, -t),
        4 => DVec3::new(s, -t, 1.0),
        _ => DVec3::new(-s, -t, -1.0),
    }
}

/// The face and the `(s, t)` coordinates that `direction` pierces: the inverse of `face_direction`.
fn cube_coordinates(direction: DVec3) -> (usize, f64, f64) {
    let magnitude = direction.abs();
    if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
        if direction.x > 0.0 {
            (0, -direction.z / magnitude.x, -direction.y / magnitude.x)
        } else {
            (1, direction.z / magnitude.x, -direction.y / magnitude.x)
        }
    } else if magnitude.y >= magnitude.z {
        if direction.y > 0.0 {
            (2, direction.x / magnitude.y, direction.z / magnitude.y)
        } else {
            (3, direction.x / magnitude.y, -direction.z / magnitude.y)
        }
    } else if direction.z > 0.0 {
        (4, direction.x / magnitude.z, -direction.y / magnitude.z)
    } else {
        (5, -direction.x / magnitude.z, -direction.y / magnitude.z)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Registry, World};

    #[test]
    fn cube_coordinates_invert_face_directions() {
        for face in 0..6 {
            for &s in &[-0.9, -0.3, 0.0, 0.4, 0.95] {
                for &t in &[-0.8, -0.1, 0.2, 0.7] {
                    let direction = face_direction(face, s, t).normalize();
                    let (found_face, found_s, found_t) = cube_coordinates(direction);
                    assert_eq!(found_face, face, "face of ({s}, {t})");
                    assert!(
                        (found_s - s).abs() < 1e-9,
                        "s of face {face}: {found_s} vs {s}"
                    );
                    assert!(
                        (found_t - t).abs() < 1e-9,
                        "t of face {face}: {found_t} vs {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn octahedral_probe_texel_weights_cover_the_sphere() {
        let extent = 32;
        let mut total = 0.0_f32;
        for row in 0..extent {
            for column in 0..extent {
                total += octahedral_texel_solid_angle(column, row, extent);
            }
        }
        assert!((f64::from(total) - std::f64::consts::TAU * 2.0).abs() < 1.0e-4);
    }

    #[test]
    fn a_constant_world_is_its_colour_in_every_direction() -> Result<()> {
        let world = World {
            color: [0.2, 0.4, 0.6],
            strength: 2.0,
            ..World::default()
        };
        let probe = WorldProbe::build(&world, &Registry::default(), &BTreeMap::new())?;
        for direction in [DVec3::X, -DVec3::Y, DVec3::new(0.3, -0.5, 0.8).normalize()] {
            assert_eq!(probe.radiance(direction), DVec3::new(0.4, 0.8, 1.2));
        }
        let expected =
            crate::render::eevee_transmission::irradiance_probe_radiance(DVec3::new(0.4, 0.8, 1.2));
        assert!((probe.average_irradiance() - expected).abs().max_element() < 1.0e-12);
        Ok(())
    }
}
