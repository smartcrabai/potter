//! The world's radiance as a cube map, built once per render.
//!
//! EEVEE renders the world into one sphere probe, and every surface reads that probe. `LightProbeModule::sync_world`
//! (`eevee_lightprobe.cc`) marks the world probe for rendering only when the world changes, so the World node graph is
//! not evaluated per surface sample. Realtime Beauty's environment term used to evaluate the graph at each of its taps,
//! three times per shaded sample, so a render's cost grew with its sample count. The probe holds `evaluate_world` at the
//! centre of each texel, and a lookup returns the texel that holds the direction.

use std::collections::BTreeMap;

use glam::DVec3;

use crate::{
    error::Result,
    graph::NodeGroup,
    image::ImageData,
    model::{Registry, World},
    shader::evaluate_world,
};

/// Texels along each cube face's edge.
const FACE_TEXELS: usize = 64;

/// The world's radiance by direction, sampled at the centre of each texel of a cube map.
pub(super) struct WorldProbe {
    /// The six faces in the order of `face_direction`, each `FACE_TEXELS` by `FACE_TEXELS`, row by row.
    radiance: Vec<DVec3>,
    /// EEVEE's non-directional spherical-harmonic L0, used by sphere-thickness translucent closures.
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
        let mut average_irradiance = DVec3::ZERO;
        let mut total_solid_angle = 0.0;
        for face in 0..6 {
            for row in 0..FACE_TEXELS {
                for column in 0..FACE_TEXELS {
                    let s = texel_centre(column);
                    let t = texel_centre(row);
                    let value = evaluate_world(world, groups, images, face_direction(face, s, t))?;
                    let solid_angle = cube_texel_solid_angle(column, row);
                    average_irradiance +=
                        super::eevee_transmission::irradiance_probe_radiance(value) * solid_angle;
                    total_solid_angle += solid_angle;
                    radiance.push(value);
                }
            }
        }
        Ok(Self {
            radiance,
            average_irradiance: average_irradiance / total_solid_angle,
        })
    }

    /// The radiance of the texel that `direction` falls in.
    pub(super) fn radiance(&self, direction: DVec3) -> DVec3 {
        let (face, s, t) = cube_coordinates(direction);
        let index = (face * FACE_TEXELS + texel(t)) * FACE_TEXELS + texel(s);
        self.radiance[index]
    }
    /// EEVEE's non-directional world-probe value for sphere-thickness translucency.
    pub(super) fn average_irradiance(&self) -> DVec3 {
        self.average_irradiance
    }
}
fn cube_texel_solid_angle(column: usize, row: usize) -> f64 {
    let edge = FACE_TEXELS as f64;
    let s0 = 2.0 * column as f64 / edge - 1.0;
    let s1 = 2.0 * (column + 1) as f64 / edge - 1.0;
    let t0 = 2.0 * row as f64 / edge - 1.0;
    let t1 = 2.0 * (row + 1) as f64 / edge - 1.0;
    let area = |s: f64, t: f64| (s * t).atan2((1.0 + s * s + t * t).sqrt());
    area(s1, t1) - area(s0, t1) - area(s1, t0) + area(s0, t0)
}

/// The centre of texel `index` along a face edge, in `-1..1`.
fn texel_centre(index: usize) -> f64 {
    2.0 * (index as f64 + 0.5) / FACE_TEXELS as f64 - 1.0
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
