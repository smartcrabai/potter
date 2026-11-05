use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CurveKey {
    pub(crate) position: DVec3,
    pub(crate) radius: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CurveHit {
    /// Ray parameter `t` for `ray_origin + t * ray_direction`; metric distance only for unit rays.
    pub(crate) distance: f64,
    pub(crate) u: f64,
    /// Cycles ribbon coordinate in [-1, 1]; thick curves use zero.
    pub(crate) v: f64,
    /// Cycles' unnormalized outward geometric normal.
    pub(crate) geometric_normal: DVec3,
}

const FLOAT_EPSILON: f64 = f32::EPSILON as f64;
const PARALLEL_EPSILON: f64 = 1.0e-18;

/// Intersects the segment `keys[1]`-`keys[2]` and its Cycles-clipped endpoint caps.
///
/// `distance` is the ray parameter in `ray_origin + distance * ray_direction`, not necessarily
/// metric distance. The direction is intentionally not normalized, matching Cycles' ray parameter.
pub(crate) fn linear_curve_intersect(
    keys: [CurveKey; 4],
    ray_origin: DVec3,
    ray_direction: DVec3,
    ray_t_min: f64,
    ray_t_max: f64,
) -> Option<CurveHit> {
    if !ray_origin.is_finite()
        || !ray_direction.is_finite()
        || !ray_t_min.is_finite()
        || ray_t_max.is_nan()
        || ray_t_min > ray_t_max
        || keys
            .iter()
            .any(|key| !key.position.is_finite() || !key.radius.is_finite())
    {
        return None;
    }

    let direction_squared = ray_direction.dot(ray_direction);
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }

    // Recenter near the ray as in Cycles, reducing cancellation for distant geometry.
    let center = (keys[1].position + keys[2].position) * 0.5;
    let offset = (center - ray_origin).dot(ray_direction) / direction_squared;
    if !center.is_finite() || !offset.is_finite() {
        return None;
    }
    let reference = ray_origin + ray_direction * offset;
    if !reference.is_finite() {
        return None;
    }
    let local_keys = keys.map(|key| CurveKey {
        position: key.position - reference,
        radius: key.radius,
    });
    if local_keys.iter().any(|key| !key.position.is_finite()) {
        return None;
    }

    let (local_distance, u, geometric_normal) = cone_sphere_intersect(local_keys, ray_direction)?;
    let distance = local_distance + offset;
    if !distance.is_finite()
        || !u.is_finite()
        || !geometric_normal.is_finite()
        || !inclusive_range(distance, ray_t_min, ray_t_max)
    {
        return None;
    }

    Some(CurveHit {
        distance,
        u,
        v: 0.0,
        geometric_normal,
    })
}

fn cone_sphere_intersect(keys: [CurveKey; 4], ray_direction: DVec3) -> Option<(f64, f64, DVec3)> {
    let r0 = keys[1].radius;
    let r1 = keys[2].radius;
    let dr = r1 - r0;
    let r0dr = r0 * dr;

    let p0 = keys[1].position;
    let p1 = keys[2].position;
    let dp = p1 - p0;
    let origin = -p0;

    let d_od_o = ray_direction.dot(ray_direction);
    let o_dot_d = origin.dot(ray_direction);
    let o_dot_o = origin.dot(origin);
    let d_o_z = dp.dot(ray_direction);
    let o_z = dp.dot(origin);
    let dp_dot_dp = dp.dot(dp);
    let y_p = o_z + r0dr;
    let g = dp_dot_dp - dr * dr;

    let a = g * d_od_o - d_o_z * d_o_z;
    let b = 2.0 * (g * o_dot_d - d_o_z * y_p);
    let c = g * o_dot_o - o_z * o_z - r0 * r0 * dp_dot_dp - 2.0 * r0dr * o_z;
    let discriminant = b * b - 4.0 * a * c;
    if ![
        d_od_o,
        o_dot_d,
        o_dot_o,
        d_o_z,
        o_z,
        dp_dot_dp,
        y_p,
        g,
        a,
        b,
        c,
        discriminant,
    ]
    .into_iter()
    .all(f64::is_finite)
        || discriminant < 0.0
    {
        return None;
    }

    // Cycles rejects the cone-parallel case before checking either spherical cap.
    if a.abs() < PARALLEL_EPSILON {
        return None;
    }

    let root = discriminant.sqrt();
    let t0 = (-b - root) / (2.0 * a);
    let y0 = y_p + t0 * d_o_z;
    if !t0.is_finite() || !y0.is_finite() {
        return None;
    }

    let mut hit = None;
    if y0 > -FLOAT_EPSILON && y0 <= g && g > 0.0 {
        let u = (y0 / g).clamp(0.0, 1.0);
        let normal = origin + t0 * ray_direction - u * dp;
        if normal.is_finite() {
            hit = Some((t0, u, normal));
        }
    }

    // Ending sphere, clipped against the following segment using key 3.
    let end_origin = -p1;
    let end_dot_direction = end_origin.dot(ray_direction);
    let end_h2 =
        end_dot_direction * end_dot_direction - d_od_o * (end_origin.dot(end_origin) - r1 * r1);
    if end_h2.is_finite() && end_h2 >= 0.0 {
        let end_root = end_h2.sqrt();
        let end_t = (-end_dot_direction - end_root) / d_od_o;
        if end_t.is_finite() {
            let next_offset = (end_t * ray_direction - p1).dot(keys[3].position - p1);
            let end_cap = -(r1 * (keys[3].radius - r1));
            let cone_axis_position = y_p + end_t * d_o_z;
            if next_offset.is_finite()
                && end_cap.is_finite()
                && cone_axis_position.is_finite()
                && hit.is_none_or(|(distance, _, _)| end_t <= distance)
                && cone_axis_position > g
                && next_offset <= end_cap
            {
                let normal = end_t * ray_direction - p1;
                if normal.is_finite() {
                    hit = Some((end_t, 1.0, normal));
                }
            }
        }
    }

    // Starting sphere exists only at the first key of a curve, as indicated by equal keys 0/1.
    if keys[0] == keys[1] {
        let start_h2 = o_dot_d * o_dot_d - d_od_o * (o_dot_o - r0 * r0);
        if start_h2.is_finite() && start_h2 >= 0.0 {
            let start_root = start_h2.sqrt();
            let start_t = (-o_dot_d - start_root) / d_od_o;
            let cone_axis_position = y_p + start_t * d_o_z;
            if start_t.is_finite()
                && cone_axis_position.is_finite()
                && hit.is_none_or(|(distance, _, _)| start_t <= distance)
                && cone_axis_position < 0.0
            {
                let normal = start_t * ray_direction - p0;
                if normal.is_finite() {
                    hit = Some((start_t, 0.0, normal));
                }
            }
        }
    }

    hit
}

const CURVE_BEZIER_STEPS: usize = 2;
const CURVE_BEZIER_UNSTABLE_DEPTH: usize = 4;
const CURVE_JACOBIAN_ITERATIONS: usize = 5;
const CURVE_FLOAT_MAX: f64 = f32::MAX as f64;
pub(crate) const CURVE_TYPE_CATMULL_ROM: u8 = 0;
pub(crate) const CURVE_TYPE_LINEAR: u8 = 1;

fn curve_key_add(a: CurveKey, b: CurveKey) -> CurveKey {
    CurveKey {
        position: a.position + b.position,
        radius: a.radius + b.radius,
    }
}

fn curve_key_sub(a: CurveKey, b: CurveKey) -> CurveKey {
    CurveKey {
        position: a.position - b.position,
        radius: a.radius - b.radius,
    }
}

fn curve_key_scale(key: CurveKey, scale: f64) -> CurveKey {
    CurveKey {
        position: key.position * scale,
        radius: key.radius * scale,
    }
}

fn curve_key_combination(curve: &[CurveKey; 4], weights: [f64; 4]) -> CurveKey {
    let mut result = CurveKey {
        position: DVec3::ZERO,
        radius: 0.0,
    };
    for index in 0..4 {
        result = curve_key_add(result, curve_key_scale(curve[index], weights[index]));
    }
    result
}

fn catmull_rom_basis_eval(curve: &[CurveKey; 4], u: f64) -> CurveKey {
    let t = u;
    let s = 1.0 - u;
    let n0 = -t * s * s;
    let n1 = 2.0 + t * t * (3.0 * t - 5.0);
    let n2 = 2.0 + s * s * (3.0 * s - 5.0);
    let n3 = -s * t * t;
    curve_key_scale(curve_key_combination(curve, [n0, n1, n2, n3]), 0.5)
}

fn catmull_rom_basis_derivative(curve: &[CurveKey; 4], u: f64) -> CurveKey {
    let t = u;
    let s = 1.0 - u;
    let n0 = -s * s + 2.0 * s * t;
    let n1 = 2.0 * t * (3.0 * t - 5.0) + 3.0 * t * t;
    let n2 = 2.0 * s * (3.0 * t + 2.0) - 3.0 * s * s;
    let n3 = -2.0 * s * t + t * t;
    curve_key_scale(curve_key_combination(curve, [n0, n1, n2, n3]), 0.5)
}

fn catmull_rom_basis_derivative2(curve: &[CurveKey; 4], u: f64) -> CurveKey {
    let t = u;
    curve_key_combination(
        curve,
        [-3.0 * t + 2.0, 9.0 * t - 5.0, -9.0 * t + 4.0, 3.0 * t - 1.0],
    )
}

pub(crate) fn curve_tangent(keys: &[CurveKey; 4], curve_type: u8, u: f64) -> Option<DVec3> {
    let tangent = match curve_type {
        CURVE_TYPE_CATMULL_ROM => catmull_rom_basis_derivative(keys, u).position,
        CURVE_TYPE_LINEAR => keys[2].position - keys[1].position,
        _ => return None,
    };
    (tangent.is_finite() && tangent.length_squared() > f64::EPSILON).then_some(tangent)
}

pub(crate) fn curve_radius(keys: &[CurveKey; 4], curve_type: u8, u: f64) -> Option<f64> {
    let radius = match curve_type {
        CURVE_TYPE_CATMULL_ROM => catmull_rom_basis_eval(keys, u).radius,
        CURVE_TYPE_LINEAR => mix(keys[1].radius, keys[2].radius, u),
        _ => return None,
    };
    (radius.is_finite() && radius >= 0.0).then_some(radius)
}

pub(crate) fn curve_position(keys: &[CurveKey; 4], curve_type: u8, u: f64) -> Option<DVec3> {
    let position = match curve_type {
        CURVE_TYPE_CATMULL_ROM => catmull_rom_basis_eval(keys, u).position,
        CURVE_TYPE_LINEAR => keys[1].position.lerp(keys[2].position, u),
        _ => return None,
    };
    position.is_finite().then_some(position)
}
fn bezier_to_catmull_keys(keys: [CurveKey; 4]) -> [CurveKey; 4] {
    [
        curve_key_sub(
            keys[3],
            curve_key_scale(curve_key_sub(keys[1], keys[0]), 6.0),
        ),
        keys[0],
        keys[3],
        curve_key_add(
            keys[0],
            curve_key_scale(curve_key_sub(keys[3], keys[2]), 6.0),
        ),
    ]
}

pub(crate) fn bezier_position(keys: [CurveKey; 4], u: f64) -> Option<DVec3> {
    let position = catmull_rom_basis_eval(&bezier_to_catmull_keys(keys), u).position;
    position.is_finite().then_some(position)
}

#[derive(Clone, Debug)]
pub(crate) enum HairCurveShape {
    CatmullRom { keys: Vec<CurveKey>, cyclic: bool },
    Linear { keys: Vec<CurveKey>, cyclic: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HairCurveRenderMode {
    Ribbon,
    Thick,
    ThickLinear,
}

/// Prepared analytic geometry for one serialized Hair Curves spline.
#[derive(Clone, Debug)]
pub(crate) struct HairCurveGeometry {
    pub(crate) shape: HairCurveShape,
    pub(crate) render_mode: HairCurveRenderMode,
    /// Raw Cycles subdivision level; CPU Embree uses `clamp(1 << level, 1, 16) + 1`.
    pub(crate) ribbon_subdivisions: usize,
    pub(crate) object_to_world: glam::DMat4,
    pub(crate) world_to_object: glam::DMat4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HairCurveSample {
    pub(crate) position: DVec3,
    pub(crate) tangent: DVec3,
    /// Radius in the curve's local space. A non-uniform object transform maps its cross-section
    /// to an ellipse, so callers must not scale this into a world-space sphere radius.
    pub(crate) radius: f64,
}

impl HairCurveGeometry {
    pub(crate) fn is_ribbon(&self) -> bool {
        self.render_mode == HairCurveRenderMode::Ribbon
    }

    pub(crate) fn ribbon_shading_normal_world(
        &self,
        span_index: usize,
        u: f64,
        v: f64,
        world_direction: DVec3,
    ) -> Option<DVec3> {
        if !self.is_ribbon()
            || !v.is_finite()
            || !(-1.0..=1.0).contains(&v)
            || !world_direction.is_finite()
        {
            return None;
        }
        let tangent = self
            .sample_local(span_index, u)?
            .tangent
            .normalize_or_zero();
        let direction = self
            .world_to_object
            .transform_vector3(world_direction)
            .normalize_or_zero();
        let bitangent = tangent.cross(-direction).normalize_or_zero();
        let normal = tangent.cross(bitangent).normalize_or_zero();
        let local_normal =
            (bitangent * v - normal * (1.0 - v * v).max(0.0).sqrt()).normalize_or_zero();
        if tangent == DVec3::ZERO
            || direction == DVec3::ZERO
            || bitangent == DVec3::ZERO
            || normal == DVec3::ZERO
            || local_normal == DVec3::ZERO
        {
            return None;
        }
        curve_normal_to_world(self.world_to_object, local_normal)
            .map(DVec3::normalize_or_zero)
            .filter(|normal| *normal != DVec3::ZERO)
    }

    pub(crate) fn closest_hit_world(
        &self,
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
    ) -> Option<(usize, CurveHit)> {
        let (local_origin, local_direction) =
            affine_ray_to_curve_space(self.world_to_object, origin, direction)?;
        let local_hit = match self.render_mode {
            HairCurveRenderMode::Ribbon => {
                let span_count = match &self.shape {
                    HairCurveShape::CatmullRom { keys, cyclic } => {
                        if *cyclic {
                            keys.len()
                        } else {
                            keys.len().saturating_sub(1)
                        }
                    }
                    HairCurveShape::Linear { .. } => 0,
                };
                (0..span_count)
                    .filter_map(|span| {
                        ribbon_curve_intersect(
                            self,
                            span,
                            local_origin,
                            local_direction,
                            t_min,
                            t_max,
                        )
                        .map(|hit| (span, hit))
                    })
                    .min_by(|left, right| left.1.distance.total_cmp(&right.1.distance))
            }
            HairCurveRenderMode::Thick => match &self.shape {
                HairCurveShape::CatmullRom { keys, cyclic } => intersect_key_curve(
                    keys,
                    *cyclic,
                    true,
                    local_origin,
                    local_direction,
                    t_min,
                    t_max,
                ),
                HairCurveShape::Linear { .. } => None,
            },
            HairCurveRenderMode::ThickLinear => match &self.shape {
                HairCurveShape::Linear { keys, cyclic } => intersect_key_curve(
                    keys,
                    *cyclic,
                    false,
                    local_origin,
                    local_direction,
                    t_min,
                    t_max,
                ),
                HairCurveShape::CatmullRom { .. } => None,
            },
        }?;
        let normal = if self.is_ribbon() {
            -direction
        } else {
            curve_normal_to_world(self.world_to_object, local_hit.1.geometric_normal)?
        };
        Some((
            local_hit.0,
            CurveHit {
                geometric_normal: normal,
                ..local_hit.1
            },
        ))
    }
    pub(crate) fn sample_local(&self, span_index: usize, u: f64) -> Option<HairCurveSample> {
        let curve_type = if matches!(&self.shape, HairCurveShape::CatmullRom { .. }) {
            CURVE_TYPE_CATMULL_ROM
        } else {
            CURVE_TYPE_LINEAR
        };
        self.sample_local_with_curve_type(span_index, u, curve_type)
    }

    /// Cycles' curve-shadow bake constructs a synthetic thick primitive even when render geometry is linear.
    pub(crate) fn sample_local_for_shadow_bake(
        &self,
        span_index: usize,
        u: f64,
    ) -> Option<HairCurveSample> {
        self.sample_local_with_curve_type(span_index, u, CURVE_TYPE_CATMULL_ROM)
    }

    fn sample_local_with_curve_type(
        &self,
        span_index: usize,
        u: f64,
        curve_type: u8,
    ) -> Option<HairCurveSample> {
        if !u.is_finite() || !(0.0..=1.0).contains(&u) {
            return None;
        }
        let sample = match &self.shape {
            HairCurveShape::CatmullRom { keys, cyclic }
            | HairCurveShape::Linear { keys, cyclic } => {
                let segment_count = if *cyclic {
                    keys.len()
                } else {
                    keys.len().saturating_sub(1)
                };
                if span_index >= segment_count || keys.len() < 2 {
                    return None;
                }
                let key_at = |offset: isize| {
                    let index = span_index as isize + offset;
                    let index = if *cyclic {
                        index.rem_euclid(keys.len() as isize) as usize
                    } else {
                        index.clamp(0, keys.len() as isize - 1) as usize
                    };
                    keys[index]
                };
                let segment_keys = [key_at(-1), key_at(0), key_at(1), key_at(2)];
                HairCurveSample {
                    position: curve_position(&segment_keys, curve_type, u)?,
                    tangent: curve_tangent(&segment_keys, curve_type, u)?,
                    radius: curve_radius(&segment_keys, curve_type, u)?,
                }
            }
        };
        (sample.position.is_finite() && sample.tangent.is_finite() && sample.radius.is_finite())
            .then_some(sample)
    }
}

const MAX_RIBBON_SUBDIVISION_LEVEL: usize = 24;
const MAX_RIBBON_SUBDIVISIONS: usize = 16;
const MAX_RIBBON_INTERSECTION_SEGMENTS: usize = MAX_RIBBON_SUBDIVISIONS + 1;

fn ribbon_subdivision_count(level: usize) -> Option<usize> {
    if level > MAX_RIBBON_SUBDIVISION_LEVEL {
        None
    } else {
        Some((1usize << level).clamp(1, MAX_RIBBON_SUBDIVISIONS))
    }
}

fn ribbon_intersection_segment_count(level: usize) -> Option<usize> {
    ribbon_subdivision_count(level)?.checked_add(1)
}

#[derive(Clone, Copy)]
struct RibbonPoint {
    position: DVec3,
    tangent: DVec3,
    radius: f64,
}

fn ribbon_curve_intersect(
    geometry: &HairCurveGeometry,
    span: usize,
    ray_origin: DVec3,
    ray_direction: DVec3,
    ray_t_min: f64,
    ray_t_max: f64,
) -> Option<CurveHit> {
    let subdivisions = ribbon_intersection_segment_count(geometry.ribbon_subdivisions)?;
    if !ray_origin.is_finite()
        || !ray_direction.is_finite()
        || !ray_t_min.is_finite()
        || ray_t_max.is_nan()
        || ray_t_min > ray_t_max
    {
        return None;
    }

    let direction_squared = ray_direction.length_squared();
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }
    let direction_length = direction_squared.sqrt();
    let inverse_direction_length = 1.0 / direction_length;
    if !inverse_direction_length.is_finite() {
        return None;
    }
    let unit_direction = ray_direction * inverse_direction_length;
    let dx0 = DVec3::new(0.0, unit_direction.z, -unit_direction.y);
    let dx1 = DVec3::new(-unit_direction.z, 0.0, unit_direction.x);
    let axis_x = if dx0.length_squared() > dx1.length_squared() {
        dx0
    } else {
        dx1
    }
    .normalize_or_zero();
    let axis_y = unit_direction.cross(axis_x).normalize_or_zero();
    // This scaled z axis makes ray-space depth equal the original, possibly affine, ray t.
    let axis_z = unit_direction * inverse_direction_length;
    if axis_x == DVec3::ZERO || axis_y == DVec3::ZERO || !axis_z.is_finite() {
        return None;
    }
    let to_ray_space = |position: DVec3| {
        let offset = position - ray_origin;
        DVec3::new(axis_x.dot(offset), axis_y.dot(offset), axis_z.dot(offset))
    };

    let mut points = [RibbonPoint {
        position: DVec3::ZERO,
        tangent: DVec3::ZERO,
        radius: 0.0,
    }; MAX_RIBBON_INTERSECTION_SEGMENTS + 1];
    let mut scale = 0.0_f64;
    for (index, point_slot) in points.iter_mut().take(subdivisions + 1).enumerate() {
        let u = index as f64 / subdivisions as f64;
        let sample = geometry.sample_local(span, u)?;
        let point = RibbonPoint {
            position: to_ray_space(sample.position),
            tangent: DVec3::new(
                axis_x.dot(sample.tangent),
                axis_y.dot(sample.tangent),
                axis_z.dot(sample.tangent),
            ),
            radius: sample.radius,
        };
        if !point.position.is_finite()
            || !point.tangent.is_finite()
            || !point.radius.is_finite()
            || point.radius < 0.0
        {
            return None;
        }
        scale = scale
            .max(point.position.x.abs())
            .max(point.position.y.abs())
            .max(point.position.z.abs())
            .max(point.radius.abs());
        *point_slot = point;
    }
    let epsilon = 4.0 * FLOAT_EPSILON * scale;
    if !epsilon.is_finite() {
        return None;
    }

    let step_size = 1.0 / subdivisions as f64;
    for index in 0..subdivisions {
        let point0 = points[index];
        let point1 = points[index + 1];
        let delta = point1.position - point0.position;
        let radius = point0.radius.max(point1.radius);
        let culling_numerator = delta.x * point0.position.y - delta.y * point0.position.x;
        let culling_denominator = delta.x * delta.x + delta.y * delta.y;
        let culling_left = culling_numerator * culling_numerator;
        let culling_right = radius * radius * culling_denominator;
        if !culling_numerator.is_finite()
            || !culling_denominator.is_finite()
            || !culling_left.is_finite()
            || !culling_right.is_finite()
            || !radius.is_finite()
            || culling_left > culling_right
        {
            continue;
        }

        let tangent0 = if point0.tangent.abs().max_element() < epsilon {
            delta
        } else {
            point0.tangent
        };
        let tangent1 = if point1.tangent.abs().max_element() < epsilon {
            delta
        } else {
            point1.tangent
        };
        let width0 = DVec3::new(tangent0.y, -tangent0.x, 0.0).normalize_or_zero() * point0.radius;
        let width1 = DVec3::new(tangent1.y, -tangent1.x, 0.0).normalize_or_zero() * point1.radius;
        let Some((u_fraction, v_fraction, distance)) = ribbon_intersect_quad(
            ray_t_min,
            ray_t_max,
            point0.position + width0,
            point1.position + width1,
            point1.position - width1,
            point0.position - width0,
        ) else {
            continue;
        };
        let self_intersection_limit =
            2.0 * mix(point0.radius, point1.radius, u_fraction) * inverse_direction_length;
        if !self_intersection_limit.is_finite() || distance <= self_intersection_limit {
            continue;
        }

        let u = index as f64 * step_size + u_fraction * step_size;
        let v = 2.0 * v_fraction - 1.0;
        if !u.is_finite() || !v.is_finite() || !inclusive_range(distance, ray_t_min, ray_t_max) {
            return None;
        }
        return Some(CurveHit {
            distance,
            u,
            v,
            geometric_normal: -ray_direction,
        });
    }
    None
}

fn ribbon_intersect_quad(
    ray_t_min: f64,
    ray_t_max: f64,
    quad_v0: DVec3,
    quad_v1: DVec3,
    quad_v2: DVec3,
    quad_v3: DVec3,
) -> Option<(f64, f64, f64)> {
    if !quad_v0.is_finite() || !quad_v1.is_finite() || !quad_v2.is_finite() || !quad_v3.is_finite()
    {
        return None;
    }
    let edge_db = quad_v1 - quad_v3;
    let ww = quad_v3.x * edge_db.y - quad_v3.y * edge_db.x;
    if !ww.is_finite() {
        return None;
    }
    let (v0, v1, v2) = if ww <= 0.0 {
        (quad_v0, quad_v1, quad_v3)
    } else {
        (quad_v2, quad_v3, quad_v1)
    };
    let edge0 = v2 - v0;
    let edge1 = v0 - v1;
    let u_edge = v0.x * edge0.y - v0.y * edge0.x;
    let v_edge = v1.x * edge1.y - v1.y * edge1.x;
    if !u_edge.is_finite() || !v_edge.is_finite() || u_edge.max(v_edge) > 0.0 {
        return None;
    }

    let normal = edge1.cross(edge0);
    let denominator = normal.z;
    if !normal.is_finite() || !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let inverse_denominator = 1.0 / denominator;
    let distance = inverse_denominator * v0.dot(normal);
    if !distance.is_finite() || !inclusive_range(distance, ray_t_min, ray_t_max) {
        return None;
    }
    let mut u = u_edge * inverse_denominator;
    let mut v = v_edge * inverse_denominator;
    if ww > 0.0 {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    (u.is_finite() && v.is_finite()).then_some((u, v, distance))
}

fn intersect_key_curve(
    keys: &[CurveKey],
    cyclic: bool,
    catmull_rom: bool,
    origin: DVec3,
    direction: DVec3,
    t_min: f64,
    t_max: f64,
) -> Option<(usize, CurveHit)> {
    if keys.len() < 2 {
        return None;
    }
    let segment_count = if cyclic { keys.len() } else { keys.len() - 1 };
    let mut closest: Option<(usize, CurveHit)> = None;
    for segment in 0..segment_count {
        let key_at = |offset: isize| {
            let index = segment as isize + offset;
            let index = if cyclic {
                index.rem_euclid(keys.len() as isize) as usize
            } else {
                index.clamp(0, keys.len() as isize - 1) as usize
            };
            keys[index]
        };
        let segment_keys = [key_at(-1), key_at(0), key_at(1), key_at(2)];
        let current_max = closest.map_or(t_max, |(_, hit)| hit.distance.min(t_max));
        let hit = if catmull_rom {
            catmull_rom_curve_intersect(segment_keys, origin, direction, t_min, current_max)
        } else {
            linear_curve_intersect(segment_keys, origin, direction, t_min, current_max)
        };
        if let Some(hit) = hit
            && closest.is_none_or(|(_, previous)| hit.distance < previous.distance)
        {
            closest = Some((segment, hit));
        }
    }
    closest
}

/// Maps a world ray into curve-local space without renormalizing its direction.
pub(crate) fn affine_ray_to_curve_space(
    world_to_object: glam::DMat4,
    origin: DVec3,
    direction: DVec3,
) -> Option<(DVec3, DVec3)> {
    let local_origin = world_to_object.transform_point3(origin);
    let local_direction = world_to_object.transform_vector3(direction);
    (local_origin.is_finite()
        && local_direction.is_finite()
        && local_direction.length_squared() > 0.0)
        .then_some((local_origin, local_direction))
}

pub(crate) fn curve_normal_to_world(world_to_object: glam::DMat4, normal: DVec3) -> Option<DVec3> {
    let normal = world_to_object.transpose().transform_vector3(normal);
    normal.is_finite().then_some(normal)
}

fn catmull_rom_bezier_controls(curve: [CurveKey; 4]) -> [CurveKey; 4] {
    [
        curve[1],
        curve_key_add(
            curve[1],
            curve_key_scale(curve_key_sub(curve[2], curve[0]), 1.0 / 6.0),
        ),
        curve_key_sub(
            curve[2],
            curve_key_scale(curve_key_sub(curve[3], curve[1]), 1.0 / 6.0),
        ),
        curve[2],
    ]
}

/// Intersects one Cycles Catmull-Rom thick-curve segment.
///
/// `distance` is the ray parameter in `ray_origin + distance * ray_direction`, not necessarily
/// metric distance. The direction is intentionally not normalized, matching Cycles' ray parameter.
pub(crate) fn catmull_rom_curve_intersect(
    keys: [CurveKey; 4],
    ray_origin: DVec3,
    ray_direction: DVec3,
    ray_t_min: f64,
    ray_t_max: f64,
) -> Option<CurveHit> {
    if !ray_origin.is_finite()
        || !ray_direction.is_finite()
        || !ray_t_min.is_finite()
        || ray_t_max.is_nan()
        || ray_t_min > ray_t_max
        || keys
            .iter()
            .any(|key| !key.position.is_finite() || !key.radius.is_finite())
    {
        return None;
    }
    let direction_squared = ray_direction.dot(ray_direction);
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }

    let center = (keys[0].position + keys[1].position + keys[2].position + keys[3].position) * 0.25;
    let dt = (center - ray_origin).dot(ray_direction) / direction_squared;
    if !center.is_finite() || !dt.is_finite() {
        return None;
    }
    let reference = ray_origin + ray_direction * dt;
    if !reference.is_finite() {
        return None;
    }
    let curve = keys.map(|key| CurveKey {
        position: key.position - reference,
        radius: key.radius,
    });
    if curve.iter().any(|key| !key.position.is_finite()) {
        return None;
    }

    catmull_rom_curve_intersect_recentered(curve, ray_direction, ray_t_min, ray_t_max, dt)
}

fn catmull_rom_curve_intersect_recentered(
    curve: [CurveKey; 4],
    ray_direction: DVec3,
    ray_t_min: f64,
    ray_t_max: f64,
    dt: f64,
) -> Option<CurveHit> {
    let step_size = 1.0 / CURVE_BEZIER_STEPS as f64;
    let mut stack = [SubdivisionState {
        u0: 0.0,
        u1: 1.0,
        step: 0,
    }; CURVE_BEZIER_UNSTABLE_DEPTH];
    let mut depth = 0usize;
    let mut best = None;
    let mut ray_t_max = ray_t_max;
    let mut u0 = 0.0;
    let mut u1 = 1.0;
    let mut step_index = 0usize;

    'walk: loop {
        while step_index < CURVE_BEZIER_STEPS {
            let step = step_index as f64 * step_size;
            let dscale = (u1 - u0) * (1.0 / 3.0) * step_size;
            let vu0 = mix(u0, u1, step);
            let vu1 = mix(u0, u1, step + step_size);
            let p0 = catmull_rom_basis_eval(&curve, vu0);
            let dp0_du = curve_key_scale(catmull_rom_basis_derivative(&curve, vu0), dscale);
            let p3 = catmull_rom_basis_eval(&curve, vu1);
            let dp3_du = curve_key_scale(catmull_rom_basis_derivative(&curve, vu1), dscale);
            let p1 = curve_key_add(p0, dp0_du);
            let p2 = curve_key_sub(p3, dp3_du);

            let Some(rr1) = sqr_point_to_line_distance(dp0_du.position, p3.position - p0.position)
            else {
                step_index += 1;
                continue;
            };
            let Some(rr2) = sqr_point_to_line_distance(dp3_du.position, p3.position - p0.position)
            else {
                step_index += 1;
                continue;
            };
            let maxr12 = rr1.max(rr2).sqrt();
            let one_plus_ulp = 1.0 + 2.0 * FLOAT_EPSILON;
            let one_minus_ulp = 1.0 - 2.0 * FLOAT_EPSILON;
            let r_outer = ([p0.radius, p1.radius, p2.radius, p3.radius]
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                + maxr12)
                * one_plus_ulp;
            let r_inner = (([p0.radius, p1.radius, p2.radius, p3.radius]
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                - maxr12)
                * one_minus_ulp)
                .max(0.0);
            if ![maxr12, r_outer, r_inner].into_iter().all(f64::is_finite) {
                step_index += 1;
                continue;
            }

            let Some(outer) = cylinder_intersect(p0.position, p3.position, r_outer, ray_direction)
            else {
                step_index += 1;
                continue;
            };

            let mut tp_min = (ray_t_min - dt).max(outer.near);
            let mut tp_max = (ray_t_max - dt).min(outer.far);
            let Some(h0) = half_plane_intersect(p0.position, dp0_du.position, ray_direction) else {
                step_index += 1;
                continue;
            };
            tp_min = tp_min.max(h0.0);
            tp_max = tp_max.min(h0.1);
            let Some(h1) = half_plane_intersect(p3.position, -dp3_du.position, ray_direction)
            else {
                step_index += 1;
                continue;
            };
            tp_min = tp_min.max(h1.0);
            tp_max = tp_max.min(h1.1);
            if !matches!(
                tp_min.partial_cmp(&tp_max),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) {
                step_index += 1;
                continue;
            }

            let mut u_outer0 = outer.u_near.clamp(0.0, 1.0);
            let mut u_outer1 = outer.u_far.clamp(0.0, 1.0);
            u_outer0 = mix(u0, u1, (step + u_outer0) / (CURVE_BEZIER_STEPS + 1) as f64);
            u_outer1 = mix(u0, u1, (step + u_outer1) / (CURVE_BEZIER_STEPS + 1) as f64);

            let inner_interval =
                cylinder_intersect(p0.position, p3.position, r_inner, ray_direction)
                    .map_or((CURVE_FLOAT_MAX, -CURVE_FLOAT_MAX), |inner| {
                        (inner.near, inner.far)
                    });
            let tp0 = (tp_min, tp_max.min(inner_interval.0));
            let tp1 = (tp_min.max(inner_interval.1), tp_max);
            let valid0 = (tp0.0 - tp0.1) < 0.001;
            let valid1 = (tp1.0 - tp1.1) < 0.001;
            if !(valid0 || valid1) {
                step_index += 1;
                continue;
            }

            let mut recurse = false;
            if valid0 {
                if depth >= CURVE_BEZIER_UNSTABLE_DEPTH {
                    if let Some(hit) = catmull_rom_curve_intersect_iterative(
                        ray_direction,
                        ray_t_min,
                        &mut ray_t_max,
                        dt,
                        &curve,
                        u_outer0,
                        tp0.0,
                    ) {
                        best = Some(hit);
                    }
                } else {
                    recurse = true;
                }
            }

            let t1 = tp1.0 + dt;
            if valid1 && t1 >= ray_t_min && t1 <= ray_t_max {
                if depth >= CURVE_BEZIER_UNSTABLE_DEPTH {
                    if let Some(hit) = catmull_rom_curve_intersect_iterative(
                        ray_direction,
                        ray_t_min,
                        &mut ray_t_max,
                        dt,
                        &curve,
                        u_outer1,
                        tp1.1,
                    ) {
                        best = Some(hit);
                    }
                } else {
                    recurse = true;
                }
            }

            if recurse {
                stack[depth] = SubdivisionState {
                    u0,
                    u1,
                    step: step_index + 1,
                };
                depth += 1;
                u0 = vu0;
                u1 = vu1;
                step_index = 0;
                continue 'walk;
            }
            step_index += 1;
        }

        if depth == 0 {
            break;
        }
        depth -= 1;
        u0 = stack[depth].u0;
        u1 = stack[depth].u1;
        step_index = stack[depth].step;
    }

    best
}

#[derive(Clone, Copy)]
struct SubdivisionState {
    u0: f64,
    u1: f64,
    step: usize,
}

#[derive(Clone, Copy)]
struct CylinderIntersection {
    near: f64,
    far: f64,
    u_near: f64,
    u_far: f64,
}

fn sqr_point_to_line_distance(point_from_start: DVec3, line_direction: DVec3) -> Option<f64> {
    let direction_squared = line_direction.dot(line_direction);
    if !direction_squared.is_finite() || direction_squared <= 0.0 {
        return None;
    }
    let cross = point_from_start.cross(line_direction);
    let result = cross.dot(cross) / direction_squared;
    result.is_finite().then_some(result)
}

fn cylinder_intersect(
    cylinder_start: DVec3,
    cylinder_end: DVec3,
    cylinder_radius: f64,
    ray_direction: DVec3,
) -> Option<CylinderIntersection> {
    let axis = cylinder_end - cylinder_start;
    let axis_length = axis.length();
    if !axis_length.is_finite() || axis_length <= 0.0 || !cylinder_radius.is_finite() {
        return None;
    }
    let rl = 1.0 / axis_length;
    let d_p = axis * rl;
    let origin = -cylinder_start;
    let d_od_o = ray_direction.dot(ray_direction);
    let o_dot_d = origin.dot(ray_direction);
    let o_dot_o = origin.dot(origin);
    let d_o_z = d_p.dot(ray_direction);
    let o_z = d_p.dot(origin);
    let a = d_od_o - d_o_z * d_o_z;
    let b = 2.0 * (o_dot_d - d_o_z * o_z);
    let c = o_dot_o - o_z * o_z - cylinder_radius * cylinder_radius;
    let discriminant = b * b - 4.0 * a * c;
    if ![d_od_o, o_dot_d, o_dot_o, d_o_z, o_z, a, b, c, discriminant]
        .into_iter()
        .all(f64::is_finite)
        || discriminant < 0.0
    {
        return None;
    }

    let epsilon = 16.0 * FLOAT_EPSILON * d_od_o.abs().max((d_o_z * d_o_z).abs());
    if a.abs() < epsilon {
        if c > 0.0 {
            return None;
        }
        // Axial rays have no finite cylinder roots; the cap planes clamp these endpoint seeds.
        let (u_near, u_far) = if d_o_z >= 0.0 { (0.0, 1.0) } else { (1.0, 0.0) };
        return Some(CylinderIntersection {
            near: -CURVE_FLOAT_MAX,
            far: CURVE_FLOAT_MAX,
            u_near,
            u_far,
        });
    }

    let root = discriminant.sqrt();
    let inverse_2a = 1.0 / (2.0 * a);
    let t0 = (-b - root) * inverse_2a;
    let t1 = (-b + root) * inverse_2a;
    let u_near = (t0 * d_o_z + o_z) * rl;
    let u_far = (t1 * d_o_z + o_z) * rl;
    if ![t0, t1, u_near, u_far].into_iter().all(f64::is_finite) {
        return None;
    }
    Some(CylinderIntersection {
        near: t0,
        far: t1,
        u_near,
        u_far,
    })
}

fn half_plane_intersect(point: DVec3, normal: DVec3, ray_direction: DVec3) -> Option<(f64, f64)> {
    let on = (-point).dot(normal);
    let dn = ray_direction.dot(normal);
    if !on.is_finite() || !dn.is_finite() {
        return None;
    }
    let epsilon = dn.abs() < PARALLEL_EPSILON;
    let t = if epsilon { 0.0 } else { -on / dn };
    let lower = if epsilon || dn < 0.0 {
        -CURVE_FLOAT_MAX
    } else {
        t
    };
    let upper = if epsilon || dn > 0.0 {
        CURVE_FLOAT_MAX
    } else {
        t
    };
    (!lower.is_nan() && !upper.is_nan()).then_some((lower, upper))
}

fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn inclusive_range(value: f64, minimum: f64, maximum: f64) -> bool {
    matches!(
        value.partial_cmp(&minimum),
        Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
    ) && matches!(
        value.partial_cmp(&maximum),
        Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less)
    )
}

fn dnormalize(vector: DVec3, derivative: DVec3) -> Option<DVec3> {
    let squared = vector.dot(vector);
    let denominator = squared * squared.sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let result = (squared * derivative - vector.dot(derivative) * vector) / denominator;
    result.is_finite().then_some(result)
}

fn catmull_rom_curve_intersect_iterative(
    ray_direction: DVec3,
    ray_t_min: f64,
    ray_t_max: &mut f64,
    dt: f64,
    curve: &[CurveKey; 4],
    mut u: f64,
    mut t: f64,
) -> Option<CurveHit> {
    let length_ray_direction = ray_direction.length();
    let box_min = curve[0]
        .position
        .min(curve[1].position)
        .min(curve[2].position.min(curve[3].position));
    let box_max = curve[0]
        .position
        .min(curve[1].position)
        .max(curve[2].position.max(curve[3].position));
    let box_min_radius = curve[0]
        .radius
        .min(curve[1].radius)
        .min(curve[2].radius.min(curve[3].radius));
    let box_max_radius = curve[0]
        .radius
        .min(curve[1].radius)
        .max(curve[2].radius.max(curve[3].radius));
    let box_abs = box_min.abs().max(box_max.abs());
    let p_error = 16.0
        * FLOAT_EPSILON
        * box_abs
            .x
            .max(box_abs.y)
            .max(box_abs.z)
            .max(box_min_radius.abs().max(box_max_radius.abs()));
    let radius_max = box_max_radius;
    if !length_ray_direction.is_finite() || !p_error.is_finite() || !radius_max.is_finite() {
        return None;
    }

    for _ in 0..CURVE_JACOBIAN_ITERATIONS {
        let q = ray_direction * t;
        let q_error = 16.0 * FLOAT_EPSILON * length_ray_direction * t;
        let p4 = catmull_rom_basis_eval(curve, u);
        let dp_du4 = catmull_rom_basis_derivative(curve, u);
        let p = p4.position;
        let dp_du = dp_du4.position;
        let radius = p4.radius;
        let dradius_du = dp_du4.radius;
        let ddp_du = catmull_rom_basis_derivative2(curve, u).position;
        let r = q - p;
        let length_r = r.length();
        let r_error = q_error.max(p_error);
        let dr_du = -dp_du;
        let dr_dt = ray_direction;
        let dp_du_squared = dp_du.dot(dp_du);
        if ![
            u,
            t,
            q_error,
            radius,
            dradius_du,
            length_r,
            r_error,
            dp_du_squared,
        ]
        .into_iter()
        .all(f64::is_finite)
            || dp_du_squared <= 0.0
        {
            return None;
        }
        let tangent = dp_du / dp_du_squared.sqrt();
        let dtangent_du = dnormalize(dp_du, ddp_du)?;
        let cos_error = p_error / dp_du_squared.sqrt();

        let f = r.dot(tangent);
        let f_error = length_r * p_error + r_error + cos_error * (1.0 + length_r);
        let df_du = dr_du.dot(tangent) + r.dot(dtangent_du);
        let df_dt = dr_dt.dot(tangent);
        let k = r.dot(r) - f * f;
        let dk_du = r.dot(dr_du) - f * df_du;
        let dk_dt = r.dot(dr_dt) - f * df_dt;
        if ![f, f_error, df_du, df_dt, k, dk_du, dk_dt]
            .into_iter()
            .all(f64::is_finite)
            || k <= 0.0
        {
            return None;
        }
        let inverse_sqrt_k = 1.0 / k.sqrt();
        let g = k.sqrt() - radius;
        let g_error = r_error + f_error + 16.0 * FLOAT_EPSILON * radius_max;
        let dg_du = dk_du * inverse_sqrt_k - dradius_du;
        let dg_dt = dk_dt * inverse_sqrt_k;
        let determinant = df_du * dg_dt - dg_du * df_dt;
        if ![inverse_sqrt_k, g, g_error, dg_du, dg_dt, determinant]
            .into_iter()
            .all(f64::is_finite)
            || determinant == 0.0
        {
            return None;
        }
        let inverse_determinant = 1.0 / determinant;
        u -= (dg_dt * f - df_dt * g) * inverse_determinant;
        t -= (-dg_du * f + df_du * g) * inverse_determinant;

        if f.abs() < f_error && g.abs() < g_error {
            let distance = t + dt;
            if !distance.is_finite()
                || !u.is_finite()
                || !inclusive_range(distance, ray_t_min, *ray_t_max)
                || !(0.0..=1.0).contains(&u)
            {
                return None;
            }
            let radial = q - p;
            let radial_length = radial.length();
            if !radial_length.is_finite() || radial_length <= 0.0 {
                return None;
            }
            let radial = radial / radial_length;
            let u_axis = dradius_du * radial + dp_du;
            let v_axis = dp_du.cross(radial);
            let geometric_normal = v_axis.cross(u_axis);
            let facing = ray_direction.dot(geometric_normal);
            if !geometric_normal.is_finite() || !facing.is_finite() || facing > 0.0 {
                return None;
            }
            *ray_t_max = distance;
            return Some(CurveHit {
                distance,
                u,
                v: 0.0,
                geometric_normal,
            });
        }
    }
    None
}

const HAIR_BVH_LEAF_SIZE: usize = 4;
const HAIR_BVH_STACK_CAPACITY: usize = usize::BITS as usize + 1;

#[derive(Clone, Copy, Debug)]
struct HairBounds {
    min: DVec3,
    max: DVec3,
}

impl HairBounds {
    // Transform the control hull to world space and bound its affine-scaled radius.
    fn from_spline(curve: &HairCurveGeometry) -> Option<Self> {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut radius = 0.0_f64;
        let mut include = |key: CurveKey| {
            if !key.position.is_finite() || !key.radius.is_finite() {
                return false;
            }
            let position = curve.object_to_world.transform_point3(key.position);
            if !position.is_finite() {
                return false;
            }
            min = min.min(position);
            max = max.max(position);
            radius = radius.max(key.radius.abs());
            true
        };
        match &curve.shape {
            HairCurveShape::CatmullRom { keys, cyclic } => {
                if keys.len() < 2 {
                    return None;
                }
                let span_count = if *cyclic { keys.len() } else { keys.len() - 1 };
                for span in 0..span_count {
                    let key_at = |offset: isize| {
                        let index = span as isize + offset;
                        let index = if *cyclic {
                            index.rem_euclid(keys.len() as isize) as usize
                        } else {
                            index.clamp(0, keys.len() as isize - 1) as usize
                        };
                        keys[index]
                    };
                    for control in
                        catmull_rom_bezier_controls([key_at(-1), key_at(0), key_at(1), key_at(2)])
                    {
                        if !include(control) {
                            return None;
                        }
                    }
                }
            }
            HairCurveShape::Linear { keys, cyclic } => {
                if keys.len() < 2 {
                    return None;
                }
                let span_count = if *cyclic { keys.len() } else { keys.len() - 1 };
                for span in 0..span_count {
                    if !include(keys[span]) || !include(keys[(span + 1) % keys.len()]) {
                        return None;
                    }
                }
            }
        }
        let object_to_world = curve.object_to_world;
        // The Frobenius norm bounds the largest singular value of the affine linear transform.
        let radius_scale = (object_to_world.x_axis.truncate().length_squared()
            + object_to_world.y_axis.truncate().length_squared()
            + object_to_world.z_axis.truncate().length_squared())
        .sqrt();
        let radius = radius * radius_scale;
        let bounds = Self {
            min: min - DVec3::splat(radius),
            max: max + DVec3::splat(radius),
        };
        (radius.is_finite() && bounds.min.is_finite() && bounds.max.is_finite()).then_some(bounds)
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn longest_axis(self) -> usize {
        let extent = self.max - self.min;
        if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        }
    }

    fn centroid(self, axis: usize) -> f64 {
        self.min[axis] * 0.5 + self.max[axis] * 0.5
    }

    fn entry(self, origin: DVec3, direction: DVec3, t_min: f64, t_max: f64) -> Option<f64> {
        let mut near = t_min;
        let mut far = t_max;
        for axis in 0..3 {
            let ray_direction = direction[axis];
            if ray_direction == 0.0 {
                if origin[axis] < self.min[axis] || origin[axis] > self.max[axis] {
                    return None;
                }
                continue;
            }
            let t0 = (self.min[axis] - origin[axis]) / ray_direction;
            let t1 = (self.max[axis] - origin[axis]) / ray_direction;
            if t0.is_nan() || t1.is_nan() {
                return None;
            }
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1));
            if near > far {
                return None;
            }
        }
        Some(near)
    }
}

impl HairCurveGeometry {
    pub(crate) fn world_bounds(&self) -> Option<(DVec3, DVec3)> {
        let bounds = HairBounds::from_spline(self)?;
        Some((bounds.min, bounds.max))
    }
}

#[derive(Clone, Copy, Debug)]
enum HairNodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct HairNode {
    bounds: HairBounds,
    kind: HairNodeKind,
}

/// Median-split acceleration structure for prepared Cycles hair-curve instances.
#[derive(Debug)]
pub(crate) struct HairBvh {
    nodes: Vec<HairNode>,
    curve_indices: Vec<usize>,
    curve_count: usize,
    root: Option<usize>,
}

impl HairBvh {
    pub(crate) fn build_splines<'a>(
        curves: impl IntoIterator<Item = &'a HairCurveGeometry>,
    ) -> Self {
        let bounds: Vec<_> = curves.into_iter().map(HairBounds::from_spline).collect();
        Self::from_bounds(&bounds)
    }

    fn from_bounds(bounds: &[Option<HairBounds>]) -> Self {
        let mut nodes = Vec::new();
        let mut curve_indices = Vec::new();
        curve_indices.extend(
            bounds
                .iter()
                .enumerate()
                .filter_map(|(index, bounds)| bounds.is_some().then_some(index)),
        );
        let curve_count = curve_indices.len();
        let root = if curve_count == 0 {
            None
        } else {
            build_hair_node(&mut nodes, &mut curve_indices, bounds, 0, curve_count)
        };
        Self {
            nodes,
            curve_indices,
            curve_count: bounds.len(),
            root,
        }
    }

    /// Traverses spline control-hull bounds and invokes the exact per-curve intersection for candidates.
    pub(crate) fn closest_spline_hit(
        &self,
        curve_count: usize,
        origin: DVec3,
        direction: DVec3,
        t_min: f64,
        t_max: f64,
        mut intersect: impl FnMut(usize, f64) -> Option<(usize, CurveHit)>,
    ) -> Option<(usize, usize, CurveHit)> {
        if curve_count != self.curve_count
            || !origin.is_finite()
            || !direction.is_finite()
            || !t_min.is_finite()
            || t_max.is_nan()
            || t_min > t_max
        {
            return None;
        }
        let direction_squared = direction.dot(direction);
        if !direction_squared.is_finite() || direction_squared <= 0.0 {
            return None;
        }

        let mut current = self.root?;
        let mut stack = [0usize; HAIR_BVH_STACK_CAPACITY];
        let mut stack_len = 0usize;
        let mut closest: Option<(usize, usize, CurveHit)> = None;

        loop {
            let current_max = closest.map_or(t_max, |(_, _, hit)| hit.distance.min(t_max));
            let node = self.nodes[current];
            if node
                .bounds
                .entry(origin, direction, t_min, current_max)
                .is_none()
            {
                if stack_len == 0 {
                    break;
                }
                stack_len -= 1;
                current = stack[stack_len];
                continue;
            }

            match node.kind {
                HairNodeKind::Leaf { start, end } => {
                    for slot in start..end {
                        let curve_index = self.curve_indices[slot];
                        let current_max =
                            closest.map_or(t_max, |(_, _, hit)| hit.distance.min(t_max));
                        let Some((span_index, hit)) = intersect(curve_index, current_max) else {
                            continue;
                        };
                        if closest.is_none_or(|(previous_curve, previous_span, previous_hit)| {
                            hit.distance < previous_hit.distance
                                || (crate::float::equal_f64(hit.distance, previous_hit.distance)
                                    && (curve_index, span_index) < (previous_curve, previous_span))
                        }) {
                            closest = Some((curve_index, span_index, hit));
                        }
                    }
                    if stack_len == 0 {
                        break;
                    }
                    stack_len -= 1;
                    current = stack[stack_len];
                }
                HairNodeKind::Branch { left, right } => {
                    let current_max = closest.map_or(t_max, |(_, _, hit)| hit.distance.min(t_max));
                    let left_entry =
                        self.nodes[left]
                            .bounds
                            .entry(origin, direction, t_min, current_max);
                    let right_entry =
                        self.nodes[right]
                            .bounds
                            .entry(origin, direction, t_min, current_max);
                    match (left_entry, right_entry) {
                        (Some(left_entry), Some(right_entry)) => {
                            let (near, far) = if left_entry <= right_entry {
                                (left, right)
                            } else {
                                (right, left)
                            };
                            stack[stack_len] = far;
                            stack_len += 1;
                            current = near;
                        }
                        (Some(_), None) => current = left,
                        (None, Some(_)) => current = right,
                        (None, None) => {
                            if stack_len == 0 {
                                break;
                            }
                            stack_len -= 1;
                            current = stack[stack_len];
                        }
                    }
                }
            }
        }
        closest
    }
}

fn build_hair_node(
    nodes: &mut Vec<HairNode>,
    indices: &mut [usize],
    bounds: &[Option<HairBounds>],
    start: usize,
    end: usize,
) -> Option<usize> {
    if start >= end {
        return None;
    }
    let first_index = *indices.get(start)?;
    let mut node_bounds = bounds.get(first_index).copied().flatten()?;
    for index in indices.get(start + 1..end)? {
        node_bounds = node_bounds.union(bounds.get(*index).copied().flatten()?);
    }
    if end - start <= HAIR_BVH_LEAF_SIZE {
        let node_index = nodes.len();
        nodes.push(HairNode {
            bounds: node_bounds,
            kind: HairNodeKind::Leaf { start, end },
        });
        return Some(node_index);
    }

    let axis = node_bounds.longest_axis();
    indices[start..end].sort_unstable_by(|left, right| {
        let left_centroid = bounds
            .get(*left)
            .copied()
            .flatten()
            .map(|bounds| bounds.centroid(axis));
        let right_centroid = bounds
            .get(*right)
            .copied()
            .flatten()
            .map(|bounds| bounds.centroid(axis));
        match (left_centroid, right_centroid) {
            (Some(left_centroid), Some(right_centroid)) => left_centroid
                .total_cmp(&right_centroid)
                .then_with(|| left.cmp(right)),
            _ => left.cmp(right),
        }
    });
    let middle = start + (end - start) / 2;
    let left = build_hair_node(nodes, indices, bounds, start, middle)?;
    let right = build_hair_node(nodes, indices, bounds, middle, end)?;
    let node_index = nodes.len();
    nodes.push(HairNode {
        bounds: node_bounds,
        kind: HairNodeKind::Branch { left, right },
    });
    Some(node_index)
}

#[cfg(test)]
mod tests {
    use super::{
        CurveKey, HairBvh, HairCurveGeometry, HairCurveRenderMode, HairCurveShape,
        catmull_rom_curve_intersect, linear_curve_intersect,
    };
    use glam::{DMat4, DVec3};

    #[test]
    fn intersects_segment_center_and_start_cap() -> Result<(), &'static str> {
        let start = CurveKey {
            position: DVec3::ZERO,
            radius: 1.0,
        };
        let end = CurveKey {
            position: DVec3::new(0.0, 0.0, 2.0),
            radius: 1.0,
        };
        let keys = [start, start, end, end];

        let center = linear_curve_intersect(
            keys,
            DVec3::new(2.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            0.0,
            f64::INFINITY,
        )
        .ok_or("expected center intersection")?;
        assert!((center.distance - 1.0).abs() < 1.0e-12);
        assert!((center.u - 0.5).abs() < 1.0e-12);
        assert!((center.geometric_normal - DVec3::X).length() < 1.0e-12);

        let cap = linear_curve_intersect(
            keys,
            DVec3::new(2.0, 0.0, -0.5),
            DVec3::new(-1.0, 0.0, 0.0),
            0.0,
            f64::INFINITY,
        )
        .ok_or("expected start-cap intersection")?;
        assert!((cap.distance - (2.0 - 0.75_f64.sqrt())).abs() < 1.0e-12);
        assert_eq!(cap.u, 0.0);
        assert!((cap.geometric_normal - DVec3::new(0.75_f64.sqrt(), 0.0, -0.5)).length() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn intersects_catmull_rom_center() -> Result<(), &'static str> {
        let keys = [
            CurveKey {
                position: DVec3::new(0.0, 0.0, -1.0),
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::ZERO,
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::Z,
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 2.0),
                radius: 1.0,
            },
        ];
        let hit = catmull_rom_curve_intersect(
            keys,
            DVec3::new(2.0, 0.0, 0.5),
            DVec3::new(-1.0, 0.0, 0.0),
            0.0,
            f64::INFINITY,
        )
        .ok_or("expected Catmull-Rom intersection")?;
        assert!((hit.distance - 1.0).abs() < 1.0e-3);
        assert!((hit.u - 0.5).abs() < 1.0e-3);
        assert!((hit.geometric_normal - DVec3::X).length() < 1.0e-3);
        Ok(())
    }

    #[test]
    fn ribbon_intersection_preserves_v_and_shading_normal() -> Result<(), &'static str> {
        let keys = [
            CurveKey {
                position: DVec3::ZERO,
                radius: 0.2,
            },
            CurveKey {
                position: DVec3::Z / 3.0,
                radius: 0.2,
            },
            CurveKey {
                position: DVec3::Z * (2.0 / 3.0),
                radius: 0.2,
            },
            CurveKey {
                position: DVec3::Z,
                radius: 0.2,
            },
        ];
        let curve = HairCurveGeometry {
            shape: HairCurveShape::CatmullRom {
                keys: vec![keys[0], keys[3]],
                cyclic: false,
            },
            render_mode: HairCurveRenderMode::Ribbon,
            ribbon_subdivisions: 3,
            object_to_world: DMat4::IDENTITY,
            world_to_object: DMat4::IDENTITY,
        };
        assert!(curve.is_ribbon());
        let (span, hit) = curve
            .closest_hit_world(DVec3::new(2.0, 0.1, 0.5), -DVec3::X, 0.0, f64::INFINITY)
            .ok_or("expected ribbon intersection")?;
        assert_eq!(span, 0);
        assert!((hit.distance - 2.0).abs() < 1.0e-10);
        assert!((hit.u - 0.5).abs() < 1.0e-10);
        assert!((hit.v - 0.5).abs() < 1.0e-10);
        assert_eq!(hit.geometric_normal, DVec3::X);

        let shading_normal = curve
            .ribbon_shading_normal_world(0, hit.u, hit.v, -DVec3::X)
            .ok_or("expected ribbon shading normal")?;
        let expected = DVec3::X * 0.75_f64.sqrt() + DVec3::Y * 0.5;
        assert!((shading_normal - expected).length() < 1.0e-10);
        let scale = DMat4::from_scale(DVec3::new(2.0, 1.0, 1.0));
        let mut scaled_curve = curve.clone();
        scaled_curve.object_to_world = scale;
        scaled_curve.world_to_object = scale.inverse();
        let transformed_normal = scaled_curve
            .ribbon_shading_normal_world(0, hit.u, hit.v, -DVec3::X)
            .ok_or("expected transformed ribbon shading normal")?;
        let expected_transformed =
            (DVec3::X * (0.75_f64.sqrt() / 2.0) + DVec3::Y * 0.5).normalize();
        assert!((transformed_normal - expected_transformed).length() < 1.0e-10);
        Ok(())
    }
    #[test]
    fn ribbon_subdivision_level_matches_cycles_power_of_two_clamp() {
        assert_eq!(super::ribbon_subdivision_count(0), Some(1));
        assert_eq!(super::ribbon_subdivision_count(1), Some(2));
        assert_eq!(super::ribbon_subdivision_count(2), Some(4));
        assert_eq!(super::ribbon_intersection_segment_count(2), Some(5));
        assert_eq!(super::ribbon_subdivision_count(4), Some(16));
        assert_eq!(super::ribbon_intersection_segment_count(24), Some(17));
        assert_eq!(super::ribbon_subdivision_count(25), None);
    }

    #[test]
    fn thick_linear_intersection_keeps_radial_normal_and_zero_v() -> Result<(), &'static str> {
        let start = CurveKey {
            position: DVec3::ZERO,
            radius: 0.2,
        };
        let end = CurveKey {
            position: DVec3::Z,
            radius: 0.2,
        };
        let curve = HairCurveGeometry {
            shape: HairCurveShape::Linear {
                keys: vec![start, end],
                cyclic: false,
            },
            render_mode: HairCurveRenderMode::ThickLinear,
            ribbon_subdivisions: 3,
            object_to_world: DMat4::IDENTITY,
            world_to_object: DMat4::IDENTITY,
        };
        assert!(!curve.is_ribbon());
        let (_, hit) = curve
            .closest_hit_world(DVec3::new(2.0, 0.1, 0.5), -DVec3::X, 0.0, f64::INFINITY)
            .ok_or("expected thick-linear intersection")?;
        assert_eq!(hit.v, 0.0);
        let expected_radial = DVec3::new(0.75_f64.sqrt(), 0.5, 0.0) * 0.2;
        assert!((hit.geometric_normal - expected_radial).length() < 1.0e-10);
        assert!(
            curve
                .ribbon_shading_normal_world(0, hit.u, hit.v, -DVec3::X)
                .is_none()
        );
        Ok(())
    }
    #[test]
    fn affine_catmull_rom_intersection_keeps_local_radius_and_unnormalized_geometric_normal()
    -> Result<(), &'static str> {
        let scale = DMat4::from_scale(DVec3::new(2.0, 1.0, 1.0));
        let keys = [
            CurveKey {
                position: DVec3::ZERO,
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 2.0 / 3.0),
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 4.0 / 3.0),
                radius: 1.0,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 2.0),
                radius: 1.0,
            },
        ];
        let curve = HairCurveGeometry {
            shape: HairCurveShape::CatmullRom {
                keys: keys.to_vec(),
                cyclic: false,
            },
            render_mode: HairCurveRenderMode::Thick,
            ribbon_subdivisions: 3,
            object_to_world: scale,
            world_to_object: scale.inverse(),
        };
        let (_, hit) = curve
            .closest_hit_world(
                DVec3::new(4.0, 0.0, 1.0),
                DVec3::new(-1.0, 0.0, 0.0),
                0.0,
                f64::INFINITY,
            )
            .ok_or("expected affine Catmull-Rom intersection")?;
        assert!((hit.distance - 2.0).abs() < 1.0e-10);
        assert!((hit.geometric_normal - DVec3::X * (2.0 / 9.0)).length() < 1.0e-10);
        Ok(())
    }
    #[test]
    fn spline_bvh_intersects_affine_catmull_rom_with_local_radius() -> Result<(), &'static str> {
        let transform = DMat4::from_scale(DVec3::new(2.0, 1.0, 1.0));
        let keys = [
            CurveKey {
                position: DVec3::new(0.0, 0.0, 0.0),
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 1.0 / 3.0),
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 2.0 / 3.0),
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::new(0.0, 0.0, 1.0),
                radius: 0.1,
            },
        ];
        let curve = HairCurveGeometry {
            shape: HairCurveShape::CatmullRom {
                keys: keys.to_vec(),
                cyclic: false,
            },
            render_mode: HairCurveRenderMode::Thick,
            ribbon_subdivisions: 3,
            object_to_world: transform,
            world_to_object: transform.inverse(),
        };
        let bvh = HairBvh::build_splines([&curve]);
        let closest = bvh
            .closest_spline_hit(
                1,
                DVec3::new(1.0, 0.0, 0.5),
                -DVec3::X,
                0.0,
                f64::INFINITY,
                |curve_index, t_max| {
                    (curve_index == 0)
                        .then(|| {
                            curve.closest_hit_world(
                                DVec3::new(1.0, 0.0, 0.5),
                                -DVec3::X,
                                0.0,
                                t_max,
                            )
                        })
                        .flatten()
                },
            )
            .ok_or("expected affine Catmull-Rom BVH intersection")?;
        assert_eq!((closest.0, closest.1), (0, 1));
        assert!((closest.2.distance - 0.8).abs() < 1.0e-10);
        assert!(
            bvh.closest_spline_hit(
                1,
                DVec3::new(1.0, 1.0, 0.5),
                -DVec3::X,
                0.0,
                f64::INFINITY,
                |_, _| None,
            )
            .is_none()
        );
        Ok(())
    }
    #[test]
    fn linear_render_curve_shadow_bake_uses_thick_catmull_frame() -> Result<(), &'static str> {
        let keys = [
            CurveKey {
                position: DVec3::ZERO,
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::new(1.0, 0.0, 1.0),
                radius: 0.1,
            },
            CurveKey {
                position: DVec3::new(2.0, 0.0, 0.0),
                radius: 0.1,
            },
        ];
        let curve = HairCurveGeometry {
            shape: HairCurveShape::Linear {
                keys: keys.to_vec(),
                cyclic: false,
            },
            render_mode: HairCurveRenderMode::ThickLinear,
            ribbon_subdivisions: 1,
            object_to_world: DMat4::IDENTITY,
            world_to_object: DMat4::IDENTITY,
        };
        assert_eq!(
            curve
                .sample_local(1, 0.0)
                .ok_or("expected local sample")?
                .tangent,
            DVec3::new(1.0, 0.0, -1.0)
        );
        assert_eq!(
            curve
                .sample_local_for_shadow_bake(1, 0.0)
                .ok_or("expected shadow-bake local sample")?
                .tangent,
            DVec3::X
        );
        Ok(())
    }
}
