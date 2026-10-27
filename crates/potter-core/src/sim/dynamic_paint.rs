//! Deterministic vertex colors from point-brush proximity and volume-mesh interior coverage.

use glam::{DMat4, DVec3, Vec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

const MAX_DISTANCE_CHECKS: usize = 20_000_000;
const MAX_PAINT_VERTICES: usize = 1_000_000;
const MAX_BRUSH_POSITIONS: usize = 1_000_000;

/// Persistent PAINT/VERTEX state for one canvas vertex.

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PaintPoint {
    dry: [f32; 4],
    wet: [f32; 4],
    wetness: f32,
    painted: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintSettings {
    pub(crate) influence_scale: f32,
    pub(crate) use_drying: bool,
    pub(crate) dry_speed: f32,
    pub(crate) dry_log: bool,
    pub(crate) color_dry_threshold: f32,
    pub(crate) use_dissolve: bool,
    pub(crate) dissolve_speed: f32,
    pub(crate) dissolve_log: bool,
}

/// Advance one Blender PAINT/VERTEX frame. Blender dries and dissolves old paint before brushes
/// update the surface, then exposes the blended color and the independent wetness value.
pub(crate) fn paint_step(
    canvas: &Mesh,
    canvas_world: DMat4,
    brushes: &[super::DynamicPaintBrush],
    points: &mut [PaintPoint],
    settings: PaintSettings,
) -> Result<()> {
    if !canvas_world.is_finite()
        || !settings.influence_scale.is_finite()
        || settings.influence_scale < 0.0
        || points.len() != canvas.vertices.len()
        || (settings.use_drying && (!settings.dry_speed.is_finite() || settings.dry_speed <= 0.0))
        || (settings.use_dissolve
            && (!settings.dissolve_speed.is_finite() || settings.dissolve_speed <= 0.0))
        || brushes.iter().any(|brush| {
            brush
                .volume_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != canvas.vertices.len())
        })
    {
        return Err(invalid("dynamic-paint paint inputs are invalid"));
    }
    if canvas.vertices.len() > MAX_PAINT_VERTICES || brushes.len() > MAX_BRUSH_POSITIONS {
        return Err(limit("dynamic paint input exceeds limits"));
    }
    let checks = canvas
        .vertices
        .len()
        .checked_mul(brushes.len())
        .ok_or_else(|| limit("dynamic paint distance work exceeds limits"))?;
    if checks > MAX_DISTANCE_CHECKS {
        return Err(limit("dynamic paint distance work exceeds limits"));
    }
    if brushes.iter().any(|brush| {
        !brush.position.is_finite()
            || !brush.radius.is_finite()
            || brush.radius < 0.0
            || !brush.strength.is_finite()
            || !brush.wetness.is_finite()
            || !(0.0..=1.0).contains(&brush.strength)
            || brush.color.iter().any(|component| !component.is_finite())
            || !(brush.wetness as f32).is_finite()
    }) || points.iter().any(|point| {
        point
            .dry
            .iter()
            .chain(&point.wet)
            .any(|value| !value.is_finite())
            || !point.wetness.is_finite()
    }) {
        return Err(invalid("dynamic-paint paint values must be finite"));
    }
    let has_point_brush = brushes.iter().any(|brush| brush.volume_mask.is_none());

    for (vertex_index, (vertex, point)) in canvas.vertices.iter().zip(points.iter_mut()).enumerate()
    {
        if !vertex.co.is_finite() {
            return Err(invalid("dynamic paint canvas contains a non-finite vertex"));
        }
        let position = if has_point_brush {
            let position = canvas_world.transform_point3(vertex.co);
            if !position.is_finite() {
                return Err(invalid(
                    "dynamic paint transform produced a non-finite vertex",
                ));
            }
            position
        } else {
            DVec3::ZERO
        };
        prepare_paint_point(point, settings);

        for brush in brushes {
            let influence = if let Some(mask) = &brush.volume_mask {
                if !mask[vertex_index] {
                    continue;
                }
                1.0
            } else {
                if brush.radius == 0.0 {
                    continue;
                }
                let distance = position.distance(brush.position);
                if distance > brush.radius {
                    continue;
                }
                if brush.constant_falloff {
                    1.0
                } else {
                    1.0 - distance / brush.radius
                }
            };
            if influence < 0.001 {
                continue;
            }
            let strength = (influence * brush.strength * f64::from(settings.influence_scale))
                .clamp(0.0, 1.0) as f32;
            apply_paint_brush(point, brush, strength);
        }
    }
    if points.iter().any(|point| {
        point
            .dry
            .iter()
            .chain(&point.wet)
            .any(|value| !value.is_finite())
            || !point.wetness.is_finite()
    }) {
        return Err(invalid("dynamic-paint paint output is not finite"));
    }
    Ok(())
}
const IMAGE_GAUSSIAN: [f32; 5] = [0.996_849, 0.596_145, 0.596_145, 0.596_145, 0.524_141];
const IMAGE_GAUSSIAN_TOTAL: f32 = 3.309_425;

fn prepare_paint_point(point: &mut PaintPoint, settings: PaintSettings) {
    if settings.use_drying {
        if point.wetness >= 0.001 {
            let previous_wetness = point.wetness;
            point.wetness =
                dissolve_value(point.wetness, settings.dry_speed, settings.dry_log).max(0.0);
            if point.wetness < settings.color_dry_threshold {
                let blended = blend_paint_colors(point.dry, point.wet);
                let dry_ratio = point.wetness / previous_wetness;
                point.wet[3] *= dry_ratio;
                point.dry[3] = (blended[3] - point.wet[3]) / (1.0 - point.wet[3]);
                if point.dry[3] != 0.0 {
                    for component in 0..3 {
                        point.dry[component] = (blended[component] * blended[3]
                            - point.wet[component] * point.wet[3])
                            / (point.dry[3] * (1.0 - point.wet[3]));
                    }
                }
            }
        } else if point.painted {
            point.dry = blend_paint_colors(point.dry, point.wet);
            point.wetness = 0.0;
            point.wet[3] = 0.0;
        }
    }
    if settings.use_dissolve {
        point.dry[3] =
            dissolve_value(point.dry[3], settings.dissolve_speed, settings.dissolve_log).max(0.0);
        point.wet[3] =
            dissolve_value(point.wet[3], settings.dissolve_speed, settings.dissolve_log).max(0.0);
    }
}

fn apply_paint_brush(point: &mut PaintPoint, brush: &super::DynamicPaintBrush, strength: f32) {
    let paint_wetness = brush.wetness as f32 * strength;
    let previous_alpha = point.wet[3];
    let mixed_alpha = previous_alpha * (1.0 - strength) + strength;
    if mixed_alpha != 0.0 {
        for component in 0..3 {
            point.wet[component] = (point.wet[component] * previous_alpha * (1.0 - strength)
                + brush.color[component] as f32 * strength)
                / mixed_alpha;
        }
    }
    if brush.absolute_alpha {
        point.wet[3] = previous_alpha.max(strength);
        point.wetness = point.wetness.max(paint_wetness);
    } else {
        point.wet[3] = mixed_alpha;
        let wetness = paint_wetness.clamp(0.0, 1.0);
        point.wetness = point.wetness * (1.0 - wetness) + wetness;
    }
    point.wetness = point.wetness.max(0.001);
    point.painted = true;
}

/// Advance one Blender PAINT/IMAGE frame using pixel-center POINT samples or Gaussian volume
/// supersampling at the stored UV barycentrics.
pub(crate) fn paint_image_step(
    sample_positions: &[[f32; 3]],
    samples_per_pixel: usize,
    brushes: &[super::DynamicPaintBrush],
    points: &mut [PaintPoint],
    settings: PaintSettings,
) -> Result<()> {
    if !matches!(samples_per_pixel, 1 | 5)
        || points
            .len()
            .checked_mul(samples_per_pixel)
            .is_none_or(|count| count != sample_positions.len())
        || !settings.influence_scale.is_finite()
        || settings.influence_scale < 0.0
        || (settings.use_drying && (!settings.dry_speed.is_finite() || settings.dry_speed <= 0.0))
        || (settings.use_dissolve
            && (!settings.dissolve_speed.is_finite() || settings.dissolve_speed <= 0.0))
        || brushes.iter().any(|brush| {
            brush
                .volume_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != sample_positions.len())
        })
    {
        return Err(invalid("dynamic-paint image inputs are invalid"));
    }
    let checks = points
        .len()
        .checked_mul(brushes.len())
        .ok_or_else(|| limit("dynamic-paint image brush work exceeds limits"))?;
    if checks > MAX_DISTANCE_CHECKS {
        return Err(limit("dynamic-paint image brush work exceeds limits"));
    }
    if sample_positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
        || brushes.iter().any(|brush| {
            !brush.position.is_finite()
                || !brush.radius.is_finite()
                || brush.radius < 0.0
                || !brush.strength.is_finite()
                || !brush.wetness.is_finite()
                || !(0.0..=1.0).contains(&brush.strength)
                || brush.color.iter().any(|value| !value.is_finite())
        })
        || points.iter().any(|point| {
            point
                .dry
                .iter()
                .chain(&point.wet)
                .any(|value| !value.is_finite())
                || !point.wetness.is_finite()
        })
    {
        return Err(invalid("dynamic-paint image values must be finite"));
    }

    for (pixel_index, point) in points.iter_mut().enumerate() {
        prepare_paint_point(point, settings);
        let sample_start = pixel_index * samples_per_pixel;
        for brush in brushes {
            let influence = if let Some(mask) = &brush.volume_mask {
                if samples_per_pixel == 1 {
                    if mask[sample_start] { 1.0 } else { 0.0 }
                } else {
                    let mut strength = 0.0_f32;
                    for (sample_index, &weight) in IMAGE_GAUSSIAN.iter().enumerate() {
                        if mask[sample_start + sample_index] {
                            strength += weight;
                        }
                    }
                    strength / IMAGE_GAUSSIAN_TOTAL
                }
            } else {
                if brush.radius == 0.0 {
                    continue;
                }
                let point = Vec3::from_array(sample_positions[sample_start]);
                let distance = point.distance(brush.position.as_vec3());
                let radius = brush.radius as f32;
                if distance > radius {
                    continue;
                }
                if brush.constant_falloff {
                    1.0
                } else {
                    1.0 - distance / radius
                }
            };
            if influence >= 0.001 {
                let strength =
                    (influence * settings.influence_scale * brush.strength as f32).clamp(0.0, 1.0);
                apply_paint_brush(point, brush, strength);
            }
        }
    }
    if points.iter().any(|point| {
        point
            .dry
            .iter()
            .chain(&point.wet)
            .any(|value| !value.is_finite())
            || !point.wetness.is_finite()
    }) {
        return Err(invalid("dynamic-paint image output is not finite"));
    }
    Ok(())
}

pub(crate) fn paint_image_output(
    points: &[PaintPoint],
    surface: &super::dynamic_paint_image::ImageSurface,
) -> Vec<[f32; 4]> {
    let mut pixels = vec![[0.0; 4]; surface.resolution * surface.resolution];
    for (point, image_pixel) in points.iter().zip(&surface.pixels) {
        let mut color = blend_paint_colors(point.dry, point.wet);
        let alpha = color[3];
        for component in &mut color[..3] {
            *component *= alpha;
        }
        pixels[image_pixel.pixel_index] = color;
    }
    pixels
}

pub(crate) fn paint_image_wetness_output(
    points: &[PaintPoint],
    surface: &super::dynamic_paint_image::ImageSurface,
) -> Vec<[f32; 4]> {
    let mut pixels = vec![[0.0; 4]; surface.resolution * surface.resolution];
    for (point, image_pixel) in points.iter().zip(&surface.pixels) {
        let wetness = point.wetness.min(1.0);
        pixels[image_pixel.pixel_index] = [wetness, wetness, wetness, 1.0];
    }
    pixels
}

pub(crate) fn paint_output(points: &[PaintPoint]) -> Vec<[f64; 4]> {
    points
        .iter()
        .map(|point| vertex_color_attribute(blend_paint_colors(point.dry, point.wet)))
        .collect()
}

pub(crate) fn paint_wetness_output(points: &[PaintPoint]) -> Vec<f64> {
    points
        .iter()
        .map(|point| f64::from(point.wetness))
        .collect()
}
pub(crate) fn wetness_color_attribute(wetness: f64) -> [f64; 4] {
    let channel = srgb_byte_to_linear(wetness as f32);
    [
        f64::from(channel),
        f64::from(channel),
        f64::from(channel),
        1.0,
    ]
}

fn vertex_color_attribute(color: [f32; 4]) -> [f64; 4] {
    [
        f64::from(srgb_byte_to_linear(color[0])),
        f64::from(srgb_byte_to_linear(color[1])),
        f64::from(srgb_byte_to_linear(color[2])),
        f64::from(unit_float_to_uchar_clamp(color[3])),
    ]
}

fn srgb_byte_to_linear(value: f32) -> f32 {
    let srgb = unit_float_to_uchar_clamp(value);
    if srgb < 0.04045 {
        srgb * (1.0 / 12.92)
    } else {
        ((srgb + 0.055) * (1.0 / 1.055)).powf(2.4)
    }
}

fn unit_float_to_uchar_clamp(value: f32) -> f32 {
    (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
}

fn dissolve_value(value: f32, time: f32, is_log: bool) -> f32 {
    if is_log {
        value * 0.001_f32.powf(1.0 / (1.2 * time))
    } else {
        value - 1.0 / time
    }
}

fn blend_paint_colors(dry: [f32; 4], wet: [f32; 4]) -> [f32; 4] {
    let inverse_alpha = 1.0 - wet[3];
    let alpha = dry[3] * inverse_alpha + wet[3];
    let mut result = [0.0; 4];
    if alpha == 0.0 {
        result[..3].copy_from_slice(&dry[..3]);
    } else {
        for component in 0..3 {
            result[component] =
                (dry[component] * dry[3] * inverse_alpha + wet[component] * wet[3]) / alpha;
        }
    }
    result[3] = alpha;
    result
}
/// Paint vertex colors using the strongest brush influence at each mesh vertex.
///
/// Results follow `canvas.vertices` storage order. Uncovered vertices are transparent
/// zero color; covered vertices retain `color` with alpha weighted by falloff and strength.
pub fn paint(
    canvas: &Mesh,
    canvas_world: DMat4,
    brushes: &[DVec3],
    radius: f64,
    color: [f64; 4],
    strength: f64,
) -> Result<Vec<[f64; 4]>> {
    if !canvas_world.is_finite() {
        return Err(invalid("dynamic paint canvas transform must be finite"));
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(invalid(
            "dynamic paint radius must be finite and non-negative",
        ));
    }
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(invalid(
            "dynamic paint strength must be between zero and one",
        ));
    }
    if color.iter().any(|component| !component.is_finite()) {
        return Err(invalid("dynamic paint color must be finite"));
    }
    if canvas.vertices.len() > MAX_PAINT_VERTICES || brushes.len() > MAX_BRUSH_POSITIONS {
        return Err(limit(
            "dynamic paint is limited to one million vertices and brushes",
        ));
    }
    if brushes.iter().any(|brush| !brush.is_finite()) {
        return Err(invalid("dynamic paint brush positions must be finite"));
    }
    let checks = canvas
        .vertices
        .len()
        .checked_mul(brushes.len())
        .ok_or_else(|| limit("dynamic paint distance work exceeds limits"))?;
    if checks > MAX_DISTANCE_CHECKS {
        return Err(limit(
            "dynamic paint is limited to 20 million brush-vertex checks",
        ));
    }

    let mut output = Vec::with_capacity(canvas.vertices.len());
    for vertex in &canvas.vertices {
        if !vertex.co.is_finite() {
            return Err(invalid("dynamic paint canvas contains a non-finite vertex"));
        }
        let position = canvas_world.transform_point3(vertex.co);
        if !position.is_finite() {
            return Err(invalid(
                "dynamic paint transform produced a non-finite vertex",
            ));
        }
        let mut influence: f64 = 0.0;
        if strength > 0.0 {
            for brush in brushes {
                let offset = position - *brush;
                if !offset.is_finite() {
                    continue;
                }
                let scale = offset.abs().max_element();
                let falloff = if radius == 0.0 {
                    if scale == 0.0 { 1.0 } else { 0.0 }
                } else if scale > radius {
                    0.0
                } else {
                    let normalized_distance = (offset / radius).length();
                    if normalized_distance >= 1.0 {
                        0.0
                    } else {
                        let linear = 1.0 - normalized_distance;
                        linear * linear * (3.0 - 2.0 * linear)
                    }
                };
                influence = influence.max(falloff * strength);
                if influence >= strength {
                    break;
                }
            }
        }
        output.push(if influence > 0.0 {
            [color[0], color[1], color[2], color[3] * influence]
        } else {
            [0.0; 4]
        });
    }
    Ok(output)
}
/// Apply one Blender WEIGHT frame's point- or volume-brush updates to persistent vertex weights.
pub(crate) fn paint_weight_step(
    canvas: &Mesh,
    canvas_world: DMat4,
    brushes: &[super::DynamicPaintBrush],
    weights: &mut [f64],
    influence_scale: f64,
) -> Result<()> {
    if !canvas_world.is_finite()
        || !influence_scale.is_finite()
        || influence_scale < 0.0
        || weights.len() != canvas.vertices.len()
        || brushes.iter().any(|brush| {
            brush
                .volume_mask
                .as_ref()
                .is_some_and(|mask| mask.len() != canvas.vertices.len())
        })
    {
        return Err(invalid("dynamic-paint weight inputs are invalid"));
    }
    if canvas.vertices.len() > MAX_PAINT_VERTICES || brushes.len() > MAX_BRUSH_POSITIONS {
        return Err(limit("dynamic paint input exceeds limits"));
    }
    let checks = canvas
        .vertices
        .len()
        .checked_mul(brushes.len())
        .ok_or_else(|| limit("dynamic paint distance work exceeds limits"))?;
    if checks > MAX_DISTANCE_CHECKS {
        return Err(limit("dynamic paint distance work exceeds limits"));
    }
    if brushes.iter().any(|brush| {
        !brush.position.is_finite()
            || !brush.radius.is_finite()
            || brush.radius < 0.0
            || !brush.strength.is_finite()
            || !(0.0..=1.0).contains(&brush.strength)
    }) || weights
        .iter()
        .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
    {
        return Err(invalid("dynamic-paint weight values are invalid"));
    }
    let has_point_brush = brushes.iter().any(|brush| brush.volume_mask.is_none());

    for (vertex_index, (vertex, weight)) in canvas.vertices.iter().zip(weights).enumerate() {
        let point = if has_point_brush {
            let point = canvas_world.transform_point3(vertex.co);
            if !point.is_finite() {
                return Err(invalid("dynamic-paint canvas points must be finite"));
            }
            point
        } else {
            DVec3::ZERO
        };
        for brush in brushes {
            let influence = if let Some(mask) = &brush.volume_mask {
                if !mask[vertex_index] {
                    continue;
                }
                1.0
            } else {
                if brush.radius == 0.0 {
                    continue;
                }
                let distance = point.distance(brush.position);
                if distance > brush.radius {
                    continue;
                }
                if brush.constant_falloff {
                    1.0
                } else {
                    1.0 - distance / brush.radius
                }
            };
            if influence < 0.001 {
                continue;
            }
            let strength = (influence * brush.strength * influence_scale).clamp(0.0, 1.0);
            if brush.erase {
                *weight = (*weight * (1.0 - strength)).max(0.0);
            } else {
                *weight = weight.max(strength);
            }
        }
    }
    Ok(())
}

fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "solver tests use valid fixture inputs")]
    use glam::{DMat4, DVec3};

    use crate::{
        geom::{Mesh, Vertex},
        model::Id,
        sim::DynamicPaintBrush,
    };

    use super::{paint, paint_weight_step};

    #[test]
    fn world_space_brush_falloff_is_storage_ordered_and_bounded() {
        let canvas = Mesh {
            vertices: vec![
                Vertex {
                    id: 9,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 2,
                    co: DVec3::X,
                },
                Vertex {
                    id: 7,
                    co: DVec3::new(2.0, 0.0, 0.0),
                },
            ],
            ..Mesh::default()
        };
        let colors = paint(
            &canvas,
            DMat4::from_translation(DVec3::new(10.0, 0.0, 0.0)),
            &[DVec3::new(10.0, 0.0, 0.0)],
            2.0,
            [0.2, 0.4, 0.6, 1.0],
            0.5,
        )
        .expect("painting succeeds");
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], [0.2, 0.4, 0.6, 0.5]);
        assert!(colors[1][3] > 0.0 && colors[1][3] < colors[0][3]);
        assert_eq!(colors[2], [0.0; 4]);
        assert_eq!(
            paint(
                &canvas,
                DMat4::IDENTITY,
                &[],
                2.0,
                [0.2, 0.4, 0.6, 1.0],
                0.5
            )
            .expect("painting without brushes succeeds"),
            vec![[0.0; 4]; 3]
        );
    }
    #[test]
    fn weight_brush_accumulates_clamps_and_erases_with_linear_falloff() {
        let canvas = Mesh {
            vertices: vec![
                Vertex {
                    id: 0,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 1,
                    co: DVec3::X,
                },
                Vertex {
                    id: 2,
                    co: DVec3::new(2.0, 0.0, 0.0),
                },
            ],
            ..Mesh::default()
        };
        let brush = DynamicPaintBrush {
            node_id: Id::new("brush".to_owned()).expect("valid test brush ID"),
            position: DVec3::ZERO,
            radius: 2.0,
            color: [0.0; 4],
            strength: 0.8,
            wetness: 1.0,
            absolute_alpha: true,
            erase: false,
            constant_falloff: false,
            volume_mask: None,
        };
        let mut weights = vec![0.0; 3];
        paint_weight_step(
            &canvas,
            DMat4::IDENTITY,
            std::slice::from_ref(&brush),
            &mut weights,
            1.0,
        )
        .expect("painting weights succeeds");
        assert_eq!(weights, [0.8, 0.4, 0.0]);

        let eraser = DynamicPaintBrush {
            strength: 0.5,
            erase: true,
            ..brush.clone()
        };
        paint_weight_step(&canvas, DMat4::IDENTITY, &[eraser], &mut weights, 1.0)
            .expect("erasing weights succeeds");
        assert!((weights[0] - 0.4).abs() < 1.0e-12);
        assert!((weights[1] - 0.3).abs() < 1.0e-12);
        assert_eq!(weights[2], 0.0);

        paint_weight_step(&canvas, DMat4::IDENTITY, &[brush], &mut weights, 4.0)
            .expect("influence scale succeeds");
        assert_eq!(weights, [1.0, 1.0, 0.0]);
    }
}
