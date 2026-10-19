//! Blender 5.2.2 Clay Strips, Inflate, and Layer as dab-sequential brushes.
//!
//! Blender applies one `do_brush_action` per mouse event of a DOTS stroke. Each sample here is one
//! such dab at its `position`, applied in order to the positions the previous dabs left behind.
//! Vertex normals are the stroke-start normals for every dab: Blender's pbvh normals are not
//! refreshed between dabs of one stroke, which the Inflate and Layer oracles confirm.
//! Mirror symmetry runs each dab once per pass, in Blender's order, against the positions the
//! earlier passes left behind. The strength of every pass is scaled by Blender's symmetry feather.
//!
//! Potter's stroke has no brush settings, so the Blender 5.2.2 defaults for these brushes are
//! fixed here: the constants below and the strength, pressure, and falloff a sample carries.
//!
//! Blender sources: `mesh/brushes/clay_strips.cc`, `inflate.cc`, `layer.cc`; `sculpt.cc`
//! `brush_strength` (2329), `calc_area_normal_and_center` (2185), `calc_brush_plane` (3048),
//! `calc_brush_cube_distances` (7503), `apply_hardness_to_distances` (7543); `brush.cc`
//! `BKE_brush_calc_curve_factors` (1478).

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};

use super::sculpt::{DabBrush, Falloff, Stroke, invalid};
use crate::{error::Result, geom::Mesh};

/// Clay Strips `plane_offset` default: the plane sits this many radii above the area center.
const CLAY_STRIPS_PLANE_OFFSET: f64 = 0.15;
/// Clay Strips `tip_roundness` default: the rounded-square tip's corner radius.
const CLAY_STRIPS_TIP_ROUNDNESS: f64 = 0.15;
/// `normal_radius_factor` default. Clay Strips uses it for both the area normal and the area
/// center (`area_normal_and_center_get_position_radius`, non-plane branch).
const CLAY_STRIPS_AREA_RADIUS_FACTOR: f64 = 1.2;
/// Layer `height` default: the displacement factor scales this along the stroke-start normal.
const LAYER_HEIGHT: f64 = 0.05;
/// Layer `hardness` default.
const LAYER_HARDNESS: f64 = 0.35;

/// Blender's `symm` bit mask for the stroke's mirror axes: X = 1, Y = 2, Z = 4.
fn symmetry_mask(axes: &[String]) -> Result<u8> {
    let mut mask = 0_u8;
    for axis in axes {
        mask |= match axis.as_str() {
            "x" => 1,
            "y" => 2,
            "z" => 4,
            _ => return Err(invalid("symmetry axes must be x, y, or z")),
        };
    }
    Ok(mask)
}

/// Blender's `is_symmetry_iteration_valid` (`paint_intern.hh`): the symmetry passes of a `symm`
/// mask, as per-axis sign vectors. Pass 0 is the unflipped brush. Blender's
/// `do_symmetrical_brush_actions` runs the whole brush once per pass, in order, with the location
/// and grab delta flipped.
fn symmetry_passes(symm: u8) -> Vec<DVec3> {
    let sign = |bit: u8, i: u8| if i & bit != 0 { -1.0 } else { 1.0 };
    (0..=symm)
        .filter(|&i| {
            i == 0
                || ((symm & i) != 0 && !(symm == 5 && i == 3) && !(symm == 6 && (i == 3 || i == 5)))
        })
        .map(|i| DVec3::new(sign(1, i), sign(2, i), sign(4, i)))
        .collect()
}

/// Blender's `calc_symmetry_feather` with the default `use_symmetry_feather`: each symmetry pass
/// contributes an overlap of how close its mirrored dab center is to the original, and every
/// pass's strength is divided by the total overlap (`sculpt.cc` `calc_overlap`).
fn symmetry_feather(center: DVec3, radius: f64, passes: &[DVec3]) -> f64 {
    let overlap: f64 = passes
        .iter()
        .map(|sign| {
            let distance_sq = (center * *sign).distance_squared(center);
            if distance_sq <= 4.0 * radius * radius {
                (2.0 * radius - distance_sq.sqrt()) / (2.0 * radius)
            } else {
                0.0
            }
        })
        .sum();
    1.0 / overlap
}

/// Apply every sample as one dab, in order, running each dab's symmetry passes in Blender's order.
/// Returns whether any vertex moved.
pub(super) fn apply_dabs(
    mesh: &mut Mesh,
    stroke: &Stroke,
    dab: DabBrush,
    normals: &BTreeMap<u32, DVec3>,
    masks: &BTreeMap<u32, f64>,
) -> Result<bool> {
    let passes = symmetry_passes(symmetry_mask(&stroke.symmetry)?);
    let original: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let stroke_normals: Vec<DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| normals.get(&vertex.id).copied().unwrap_or(DVec3::Z))
        .collect();
    // Blender's `fill_factor_from_hide_and_mask`: a masked vertex keeps `1 - mask` of each dab.
    let mask_factors: Vec<f64> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            1.0 - masks
                .get(&vertex.id)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
        })
        .collect();
    let mut current = original.clone();
    let mut layer_displacement = vec![0.0_f64; original.len()];
    let mut previous: Option<DVec3> = None;
    for sample in &stroke.samples {
        let center = sample.position;
        // Blender's `sculpt.cc` `brush_strength` with alpha = strength squared, flip = 1 and
        // overlap = 1 (DOTS dabs without space attenuation), times the symmetry feather.
        let alpha = sample.strength * sample.strength;
        let feather = symmetry_feather(center, sample.radius, &passes);
        let grab = previous.replace(center).map(|last| center - last);
        match dab {
            DabBrush::ClayStrips => {
                // Clay Strips has no strip direction before the second dab, and `calc_node_mask`
                // returns no nodes for a zero grab delta.
                let Some(grab) = grab.filter(|grab| *grab != DVec3::ZERO) else {
                    continue;
                };
                // Clay Strips strength curve: pressure^1.5 and a 0.3 factor (`brush_strength`).
                let strength = 0.3 * alpha * sample.pressure.powf(1.5) * feather;
                clay_strips_dab(
                    &mut current,
                    &stroke_normals,
                    &mask_factors,
                    stroke.falloff,
                    &passes,
                    ClayStripsDab {
                        center,
                        grab,
                        radius: sample.radius,
                        strength,
                    },
                );
            }
            DabBrush::Inflate => {
                // Inflate's `brush_strength` case: 0.25 * alpha * pressure for a positive flip.
                let strength = 0.25 * alpha * sample.pressure * feather;
                for sign in &passes {
                    inflate_dab(
                        &mut current,
                        &stroke_normals,
                        &mask_factors,
                        stroke.falloff,
                        center * *sign,
                        sample.radius,
                        strength,
                    );
                }
            }
            DabBrush::Layer => {
                let strength = alpha * sample.pressure * feather;
                for sign in &passes {
                    layer_dab(
                        &mut current,
                        &original,
                        &stroke_normals,
                        &mut layer_displacement,
                        &mask_factors,
                        stroke.falloff,
                        center * *sign,
                        sample.radius,
                        strength,
                    );
                }
            }
        }
    }
    let changed = current
        .iter()
        .zip(&original)
        .any(|(after, before)| after != before);
    if changed {
        for (vertex, co) in mesh.vertices.iter_mut().zip(current) {
            vertex.co = co;
        }
    }
    Ok(changed)
}

/// One Clay Strips dab: the stroke-relative grab delta, dab center, radius, and bstrength.
struct ClayStripsDab {
    center: DVec3,
    grab: DVec3,
    radius: f64,
    strength: f64,
}

/// Blender 5.2.2 Clay Strips for one dab, run once per symmetry pass. Pass 0 takes the plane from
/// the area normal and center around the dab (`calc_brush_plane`, AREA). A mirror pass flips that
/// normal, center, and the grab delta instead of recomputing them (`sculpt.cc`, non-main-pass
/// branch). The strip frame has x = normal × grab, y = normal × x, and z = -normal, so vertices
/// below the plane have positive local z. Each vertex moves along the plane normal by
/// `strength * radius` times its depth and tip factors (`clay_strips.cc`).
fn clay_strips_dab(
    positions: &mut [DVec3],
    normals: &[DVec3],
    mask_factors: &[f64],
    falloff: Falloff,
    passes: &[DVec3],
    dab: ClayStripsDab,
) {
    let Some((normal, area_center)) = area_plane(
        positions,
        normals,
        dab.center,
        dab.radius * CLAY_STRIPS_AREA_RADIUS_FACTOR,
    ) else {
        return;
    };
    for sign in passes {
        let plane_normal = normal * *sign;
        let plane_center =
            area_center * *sign + plane_normal * (dab.radius * CLAY_STRIPS_PLANE_OFFSET);
        let grab = dab.grab * *sign;
        let x_axis = plane_normal.cross(grab).normalize_or_zero();
        if x_axis == DVec3::ZERO {
            continue;
        }
        let y_axis = plane_normal.cross(x_axis).normalize_or_zero();
        let z_axis = -plane_normal;
        for (index, position) in positions.iter_mut().enumerate() {
            let offset = *position - plane_center;
            let local_z = offset.dot(z_axis) / dab.radius;
            let depth = (local_z * (1.0 - local_z)).max(0.0) * mask_factors[index];
            if depth == 0.0 {
                continue;
            }
            let tip = cube_distance(
                offset.dot(x_axis) / dab.radius,
                offset.dot(y_axis) / dab.radius,
                CLAY_STRIPS_TIP_ROUNDNESS,
            );
            if tip >= 1.0 {
                continue;
            }
            let factor = depth * falloff.weight(tip);
            *position += plane_normal * (dab.strength * dab.radius * factor);
        }
    }
}

/// Blender's `calc_area_normal_and_center` over the vertices within `radius` of `center`: normals
/// are summed with a smooth weight, and each vertex's position is pulled toward `center` by the
/// same weight before averaging. Blender keeps only the view-facing group of normals; Potter has
/// no view, so all vertices form one group. Returns `None` when the normal sum is zero.
fn area_plane(
    positions: &[DVec3],
    normals: &[DVec3],
    center: DVec3,
    radius: f64,
) -> Option<(DVec3, DVec3)> {
    let mut normal_sum = DVec3::ZERO;
    let mut center_sum = DVec3::ZERO;
    let mut count = 0_usize;
    for (position, normal) in positions.iter().zip(normals) {
        let distance = position.distance(center);
        if distance > radius {
            continue;
        }
        let weight = Falloff::Smooth.weight(distance / radius);
        normal_sum += *normal * weight;
        center_sum += center + (*position - center) * (1.0 - weight);
        count += 1;
    }
    let normal = normal_sum.normalize_or_zero();
    if normal == DVec3::ZERO {
        return None;
    }
    let area_center = if count == 0 {
        center
    } else {
        center_sum / count as f64
    };
    Some((normal, area_center))
}

/// Blender's `calc_brush_cube_distances` for the Clay Strips tip: a square of half-size one with
/// corners rounded by `roundness`, returning one outside the square.
fn cube_distance(x: f64, y: f64, roundness: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let hardness = 1.0 - roundness;
    let (low, high) = (ax.min(ay), ax.max(ay));
    if high > 1.0 {
        1.0
    } else if low > hardness {
        DVec2::new(ax, ay).distance(DVec2::splat(hardness)) / roundness
    } else if high > hardness {
        (high - hardness) / roundness
    } else {
        0.0
    }
}

/// Blender 5.2.2 Inflate for one dab: each vertex moves along its stroke-start normal by
/// `strength * radius` times the falloff and mask factors (`inflate.cc`).
fn inflate_dab(
    positions: &mut [DVec3],
    normals: &[DVec3],
    mask_factors: &[f64],
    falloff: Falloff,
    center: DVec3,
    radius: f64,
    strength: f64,
) {
    for (index, position) in positions.iter_mut().enumerate() {
        let distance = position.distance(center);
        if distance >= radius {
            continue;
        }
        let factor = falloff.weight(distance / radius) * mask_factors[index];
        *position += normals[index] * (strength * radius * factor);
    }
}

/// Blender's `apply_hardness_to_distances`: distances inside `hardness * radius` collapse to zero,
/// and the rest of the range is stretched back to `[0, radius]`.
fn hardness_distance(distance: f64, radius: f64, hardness: f64) -> f64 {
    if distance < hardness * radius {
        0.0
    } else {
        (distance / radius - hardness) / (1.0 - hardness) * radius
    }
}

/// Blender 5.2.2 Layer for one dab (`layer.cc`, non-persistent base). Falloff uses the stroke-start
/// positions. A per-vertex displacement factor accumulates across the stroke, clamped to the mask
/// limit `1 - mask`, and each vertex moves toward `base + normal * height * factor`.
fn layer_dab(
    positions: &mut [DVec3],
    base: &[DVec3],
    normals: &[DVec3],
    displacement: &mut [f64],
    mask_factors: &[f64],
    falloff: Falloff,
    center: DVec3,
    radius: f64,
    strength: f64,
) {
    for index in 0..positions.len() {
        let distance = base[index].distance(center);
        if distance >= radius {
            continue;
        }
        let hardened = hardness_distance(distance, radius, LAYER_HARDNESS);
        if hardened >= radius {
            continue;
        }
        let factor = falloff.weight(hardened / radius) * mask_factors[index];
        if factor == 0.0 {
            continue;
        }
        let limit = mask_factors[index];
        let grown = displacement[index] + factor * strength * (1.05 - displacement[index].abs());
        displacement[index] = grown.clamp(-limit, limit);
        let target = base[index] + normals[index] * (LAYER_HEIGHT * displacement[index]);
        positions[index] += (target - positions[index]) * factor;
    }
}
