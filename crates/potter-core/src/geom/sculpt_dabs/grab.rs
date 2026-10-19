//! Blender 5.2.2 grab-family dabs: Grab, Snake Hook, Thumb, and Rotate (Twist). Each moves vertices
//! by a per-dab offset scaled by the falloff and mask factors (`mesh_brush_common.cc`
//! `translations_from_offset_and_factors`). Grab and Thumb measure from the stroke's first dab and
//! restore the stroke-start mesh before each dab; Snake Hook carries a lagging center and a rake
//! rotation between dabs; Rotate turns vertices about the stroke origin by a dial angle.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use glam::{DQuat, DVec3};

use super::{DabInput, Field, area_plane};
use crate::geom::sculpt::{DabBrush, Falloff};

/// `SCULPT_RAKE_BRUSH_FACTOR` (`sculpt_intern.hh` 137): the rake follow distance per brush radius.
const RAKE_BRUSH_FACTOR: f64 = 0.25;
/// Snake Hook `rake_factor` in the Blender 5.2.2 essentials asset.
const SNAKE_HOOK_RAKE_FACTOR: f64 = 1.0;
/// The `eps` of Blender's rake update (`sculpt.cc` 4335).
const RAKE_EPSILON: f64 = 1e-5;
/// Thumb `normal_radius_factor` of the essentials asset (`calc_area_normal`, sculpt.cc 2185).
const THUMB_NORMAL_RADIUS_FACTOR: f64 = 0.3;
/// Twist `normal_radius_factor` of the essentials asset (sets its area normal radius).
const TWIST_NORMAL_RADIUS_FACTOR: f64 = 0.5;
/// Per mirror pass, the sign of Blender's rotate angle (`rotate.cc` 181), indexed by the pass's
/// symmetry bits (X = 1, Y = 2, Z = 4).
const ROTATE_FLIP: [f64; 8] = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0];

/// Grab-family state that persists between dabs: Blender `StrokeCache` fields that brushes update
/// per dab (`cache->location` for Snake Hook, `cache->grab_delta`, `rake_data`, `dial`).
#[derive(Default)]
pub(super) struct State {
    /// Whether the stroke's first dab has run.
    started: bool,
    /// Snake Hook `cache->location`: the first dab's center, then grown by the previous grab each
    /// dab (`brush_delta_update`, sculpt.cc 4268), so the center trails the mouse by one dab.
    location: DVec3,
    /// Snake Hook `cache->grab_delta` of the previous dab.
    previous_grab: DVec3,
    /// Snake Hook `rake_data.follow_co`: the point the rake trails behind the mouse.
    follow: DVec3,
    /// Thumb `cache->sculpt_normal` (computed at the first dab), and the view proxy of Twist's dial.
    stroke_normal: Option<DVec3>,
    /// Twist `cache->dial` (`BLI_dial_2d`).
    dial: Dial,
}

/// Blender's `BLI_dial_2d` with threshold zero: Potter has no pixel threshold, so any nonzero move
/// off the dial center starts the dial.
#[derive(Default)]
struct Dial {
    /// Unit direction of the first move off the center, the basis for the angle.
    initial: Option<DVec3>,
    /// The angle returned by the previous move, before full turns are added; used to detect a wrap past +/- pi.
    last_angle: f64,
    /// Full turns detected so far.
    rotations: i32,
}

impl Dial {
    /// `BLI_dial_angle`: the signed angle of `offset` projected onto the plane perpendicular to
    /// `normal`, measured from the initial direction, with `normal` as the positive side.
    fn angle(&mut self, normal: DVec3, offset: DVec3) -> f64 {
        let current = offset - normal * offset.dot(normal);
        let length_squared = current.length_squared();
        if length_squared > 0.0 {
            let current = current / length_squared.sqrt();
            let initial = *self.initial.get_or_insert(current);
            let cos = current.dot(initial);
            let sin = current.cross(initial).dot(normal);
            let angle = sin.atan2(cos);
            if angle * self.last_angle < 0.0 && self.last_angle.abs() > FRAC_PI_2 {
                self.rotations += if self.last_angle < 0.0 { -1 } else { 1 };
            }
            self.last_angle = angle;
            return angle + TAU * f64::from(self.rotations);
        }
        self.last_angle
    }
}

/// Snake Hook's rake rotation for one dab: an axis and an angle, before any symmetry flip.
#[derive(Clone, Copy)]
struct Rake {
    axis: DVec3,
    angle: f64,
}

impl Rake {
    /// Blender's `flip_qt_qt` (sculpt.cc 2980): each mirrored axis negates the axis component and
    /// the angle, so the angle takes the product of the signs.
    fn for_pass(self, sign: DVec3) -> Self {
        Self {
            axis: self.axis * sign,
            angle: self.angle * sign.x * sign.y * sign.z,
        }
    }

    /// Blender's `sculpt_rake_rotate` rotation: the quaternion power `q^factor` is a rotation by
    /// `factor * angle` about the axis.
    fn rotate(self, point: DVec3, factor: f64) -> DVec3 {
        DQuat::from_axis_angle(self.axis, self.angle * factor).mul_vec3(point)
    }
}

/// Blender's `math::AxisAngle(from, to)` for unit vectors: the axis and angle that turn `from`
/// onto `to`.
fn between(from: DVec3, to: DVec3) -> (DVec3, f64) {
    let cos = from.dot(to);
    let cross = from.cross(to);
    let sin = cross.length();
    if sin <= f64::EPSILON {
        if cos > 0.0 {
            return (DVec3::Z, 0.0);
        }
        return (orthogonal(from), PI);
    }
    (cross / sin, sin.atan2(cos))
}

/// Blender's `orthogonal` (`BLI_math_vector.hh`): a unit vector perpendicular to `vector`, chosen
/// from its dominant axis. Only the opposed-vector rake reaches it.
fn orthogonal(vector: DVec3) -> DVec3 {
    let (x, y, z) = (vector.x.abs(), vector.y.abs(), vector.z.abs());
    let dominant = if x > y {
        if x > z { 0 } else { 2 }
    } else if y > z {
        1
    } else {
        2
    };
    let perpendicular = match dominant {
        0 => DVec3::new(-vector.y - vector.z, vector.x, vector.x),
        1 => DVec3::new(vector.y, -vector.x - vector.z, vector.y),
        _ => DVec3::new(vector.z, vector.z, -vector.x - vector.y),
    };
    perpendicular.normalize_or_zero()
}

/// Blender's pass index for a symmetry sign vector: the symm bits of the mirrored axes.
fn pass_index(sign: DVec3) -> usize {
    usize::from(sign.x < 0.0) | (usize::from(sign.y < 0.0) << 1) | (usize::from(sign.z < 0.0) << 2)
}

/// One vertex's factor for a dab centered at `center`: zero at or beyond `radius`, otherwise the
/// falloff of `distance / radius` times `1 - mask` (`filter_distances_with_radius` and
/// `calc_brush_strength_factors`). Hardness is zero for these brushes, so `apply_hardness` is the
/// identity and is not applied.
fn brush_factor(falloff: Falloff, point: DVec3, center: DVec3, radius: f64, mask: f64) -> f64 {
    let distance = point.distance(center);
    if distance >= radius {
        0.0
    } else {
        falloff.weight(distance / radius) * mask
    }
}

/// Blender 5.2.2 Grab for one dab (`grab.cc` `do_grab_brush`). The center is the stroke origin
/// and `grab` is the cumulative move from it (sculpt.cc 4278-4312). Positions were restored to
/// the stroke-start mesh before the dab, so every dab replaces the one before it.
fn grab_dab(field: &mut Field, input: &DabInput) {
    for sign in &field.passes {
        let center = input.center * *sign;
        // `grab_delta = grab_delta_get(brush, cache) * bstrength` (grab.cc 178).
        let offset = input.grab * *sign * input.strength;
        for (index, position) in field.positions.iter_mut().enumerate() {
            let factor = brush_factor(
                field.falloff,
                field.original[index],
                center,
                input.radius,
                field.mask_factors[index],
            );
            *position += offset * factor;
        }
    }
}

/// Blender 5.2.2 Thumb for one dab (`thumb.cc` `do_thumb_brush`). The offset is the grab move
/// projected onto the stroke-start sculpt normal, scaled by `bstrength`. The normal is computed at
/// the first dab and kept for the stroke (`update_sculpt_normal`, sculpt.cc 2729-2739).
fn thumb_dab(field: &mut Field, input: &DabInput, first: bool) {
    if first {
        field.grab.stroke_normal = area_plane(
            field,
            input.center,
            input.radius * THUMB_NORMAL_RADIUS_FACTOR,
            input.radius * THUMB_NORMAL_RADIUS_FACTOR,
            false,
        )
        .map(|area| area.normal);
    }
    let Some(normal) = field.grab.stroke_normal else {
        return;
    };
    for sign in &field.passes {
        let normal = normal * *sign;
        let center = input.center * *sign;
        let grab = input.grab * *sign;
        // `cross(cross(normal, grab_delta), normal) * bstrength` (thumb.cc 146).
        let offset = normal.cross(grab).cross(normal) * input.strength;
        for (index, position) in field.positions.iter_mut().enumerate() {
            let factor = brush_factor(
                field.falloff,
                field.original[index],
                center,
                input.radius,
                field.mask_factors[index],
            );
            *position += offset * factor;
        }
    }
}

/// Blender 5.2.2 Snake Hook for one dab (`snake_hook.cc` `calc_faces`, falloff deform). The center
/// trails the mouse by one dab (`brush_delta_update`, sculpt.cc 4268), the factor reads the current
/// positions, and each vertex moves with the grab plus the rake rotation (sculpt.cc 4329-4358).
fn snake_hook_dab(field: &mut Field, input: &DabInput, first: bool) {
    if first {
        field.grab.location = input.center;
        field.grab.follow = input.center;
    } else {
        field.grab.location += field.grab.previous_grab;
    }
    let grab_location = input.center;
    let follow_distance = input.radius * RAKE_BRUSH_FACTOR;
    let rake = rake_rotation(
        field.grab.follow,
        grab_location,
        input.grab,
        follow_distance,
    );
    // `rake_data_update` (sculpt.cc 4358): the follow point moves toward the mouse by the excess.
    let rake_distance = field.grab.follow.distance(grab_location);
    if rake_distance > follow_distance {
        let excess = rake_distance - follow_distance;
        field.grab.follow += (grab_location - field.grab.follow) * excess;
    }
    field.grab.previous_grab = input.grab;

    let center = field.grab.location;
    for sign in &field.passes {
        let center = center * *sign;
        let grab = input.grab * *sign;
        let rake = rake.map(|rake| rake.for_pass(*sign));
        // Blender gathers the node's positions before it writes any of them.
        let snapshot = field.positions.clone();
        for (index, position) in field.positions.iter_mut().enumerate() {
            let factor = brush_factor(
                field.falloff,
                snapshot[index],
                center,
                input.radius,
                field.mask_factors[index],
            ) * input.strength;
            let mut translation = grab * factor;
            if let Some(rake) = rake {
                let point = snapshot[index];
                translation += rake.rotate(point - center, factor) + center - point;
            }
            *position += translation;
        }
    }
}

/// Blender's rake rotation for one dab (sculpt.cc 4329-4356): the turn from the follow point's
/// direction to the one after the grab, faded near the follow point and scaled by the rake factor.
/// No rake applies when the grab is zero or the directions are degenerate.
fn rake_rotation(
    follow: DVec3,
    grab_location: DVec3,
    grab: DVec3,
    follow_distance: f64,
) -> Option<Rake> {
    if grab == DVec3::ZERO {
        return None;
    }
    let v1 = follow - grab_location;
    let v2 = follow - grab - grab_location;
    let v2_length = v2.length();
    if v2_length <= RAKE_EPSILON {
        return None;
    }
    let v2 = v2 / v2_length;
    let v1_length = v1.length();
    if v1_length <= RAKE_EPSILON {
        return None;
    }
    let v1 = v1 / v1_length;
    if (v1 - v2).length_squared() <= RAKE_EPSILON {
        return None;
    }
    let rake_distance_squared = (follow - grab_location).length_squared();
    let fade = if rake_distance_squared > follow_distance * follow_distance {
        1.0
    } else {
        rake_distance_squared.sqrt() / follow_distance
    };
    let (axis, angle) = between(v1, v2);
    Some(Rake {
        axis,
        angle: angle * SNAKE_HOOK_RAKE_FACTOR * fade,
    })
}

/// Blender 5.2.2 Twist (Rotate) for one dab (`rotate.cc` `calc_translations`). The rotation is
/// about the stroke origin, around the stroke-start sculpt normal, by `factor * vertex_rotation`
/// (`vertex_rotation = -dial_angle * bstrength`, sculpt.cc 5913-5914). The dial's view proxy is
/// the stroke-start normal, since Potter has no view; the dial vector is the origin-relative move.
fn rotate_dab(field: &mut Field, input: &DabInput, first: bool) {
    if first {
        field.grab.stroke_normal = area_plane(
            field,
            input.center,
            input.radius * TWIST_NORMAL_RADIUS_FACTOR,
            input.radius * TWIST_NORMAL_RADIUS_FACTOR,
            false,
        )
        .map(|area| area.normal);
    }
    // `update_sculpt_normal` recomputes the Twist normal every dab (sculpt.cc 2729-2739).
    let normal = area_plane(
        field,
        input.center,
        input.radius * TWIST_NORMAL_RADIUS_FACTOR,
        input.radius * TWIST_NORMAL_RADIUS_FACTOR,
        false,
    )
    .map(|area| area.normal);
    let Some(view) = field.grab.stroke_normal else {
        return;
    };
    let dial = field.grab.dial.angle(view, input.grab);
    let Some(normal) = normal else {
        return;
    };
    let vertex_rotation = -dial * input.strength;
    for sign in &field.passes {
        let center = input.center * *sign;
        let axis = normal * *sign;
        let flip = ROTATE_FLIP[pass_index(*sign)];
        for (index, position) in field.positions.iter_mut().enumerate() {
            let factor = brush_factor(
                field.falloff,
                field.original[index],
                center,
                input.radius,
                field.mask_factors[index],
            );
            let angle = factor * vertex_rotation * flip;
            let original = field.original[index];
            let rotated = DQuat::from_axis_angle(axis, angle).mul_vec3(original - center) + center;
            *position += rotated - original;
        }
    }
}

/// Apply one grab-family dab. `first` is true for the stroke's first dab.
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    let first = !field.grab.started;
    field.grab.started = true;
    match dab {
        DabBrush::Grab => grab_dab(field, input),
        DabBrush::SnakeHook => snake_hook_dab(field, input, first),
        DabBrush::Thumb => thumb_dab(field, input, first),
        DabBrush::Rotate => rotate_dab(field, input, first),
        other => unreachable!("grab::apply received {other:?}"),
    }
}
