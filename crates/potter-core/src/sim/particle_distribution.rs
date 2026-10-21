// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org): `distribute_particles`,
// `distribute_binary_search`, `distribute_from_verts_exec`, `distribute_from_faces_exec`,
// `distribute_from_volume_exec` and `psys_uv_to_w` (`particle_distribute.cc`), `psys_interpolate_face`
// and `psys_particle_on_dm` (`particle.cc`), and the `interp_weights_poly_v3`, `isect_ray_tri_v3` and
// normal and area helpers of `math_geom.cc`.

//! Blender's emitter element assignment and birth positions for vertex, face, and volume emission.
//!
//! `distribute_particles` gives every emitting element (a vertex, or a face's legacy tessface) a
//! weight, builds the cumulative weights, and assigns particle `p` an element. With `use_emit_random`
//! (`PART_TRAND`, Blender's default) the position is `BLI_rng_get_float` on a generator seeded from
//! `31415926 + seed`, one draw per particle; without it the particles sit at evenly spaced
//! positions along the cumulative weights. A face then places each particle inside its element by
//! jitter (`PART_DISTR_JIT`) or by random barycentric draws (`PART_DISTR_RAND`). A volume pushes the
//! birth point along the inward normal to half the distance to the nearest other face.
//!
//! Everything here is single precision, as Blender computes it, so birth positions match its
//! output. `PART_DISTR_GRID` is not modelled; the caller refuses it.

use std::collections::HashMap;

use glam::DVec3;

use super::particle_jitter::jitter_table;
use crate::{
    blender_rng::BlenderRng,
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, normals, triangulate},
};

/// A single-precision point or direction, as Blender's `float[3]`.
type Float3 = [f32; 3];

/// Blender's `distribute_binary_search`: the first cumulative weight that reaches `value`.
fn binary_search(sum: &[f32], value: f32) -> usize {
    let mut low = 0;
    let mut high = sum.len() - 1;
    if high == low {
        return low;
    }
    if sum[low] >= value {
        return low;
    }
    if sum[high - 1] < value {
        return high;
    }
    while low < high {
        let mid = (low + high) / 2;
        if sum[mid] >= value && sum[mid - 1] < value {
            return mid;
        }
        if sum[mid] > value {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    low
}

/// Blender's element assignment in `distribute_particles`: the element of each of `count` particles,
/// and for each element the last position written to it (`jitter_offset`, read by the jitter).
///
/// Elements with no weight never take a particle. Returns no elements when none has weight.
fn assign_elements(
    weights: &[f32],
    count: usize,
    system_seed: u32,
    random: bool,
) -> (Vec<usize>, Vec<f32>) {
    let mut jitter_offset = vec![0.0_f32; weights.len()];
    let mut elements = Vec::with_capacity(count);
    let mut totweight = 0.0_f32;
    let mut first = None;
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            totweight += weight;
            if first.is_none() {
                first = Some(index);
            }
        }
    }
    let Some(first) = first else {
        return (elements, jitter_offset);
    };
    let inverse = 1.0_f32 / totweight;
    // Blender drops an element whose addition does not change the sum (`element_map`).
    let mut element_sum = vec![weights[first] * inverse];
    let mut element_map = vec![first];
    for (index, &weight) in weights.iter().enumerate().skip(first + 1) {
        if weight > 0.0 {
            let previous = element_sum[element_sum.len() - 1];
            let next = previous + weight * inverse;
            if next > previous {
                element_sum.push(next);
                element_map.push(index);
            }
        }
    }
    let mapped = element_sum.len();
    if random {
        // `BLI_rng_new_srandom(31415926 + psys->seed)`, one `BLI_rng_get_float` per particle.
        let mut rng = BlenderRng::seed_random(31_415_926_u32.wrapping_add(system_seed));
        for _ in 0..count {
            let position = rng.get_float() * element_sum[mapped - 1];
            let element = element_map[binary_search(&element_sum, position)];
            elements.push(element);
            jitter_offset[element] = position;
        }
    } else {
        let step = if count < 2 {
            0.5_f64
        } else {
            1.0_f64 / count as f64
        };
        let mut position = if count < mapped {
            0.5_f64 / mapped as f64
        } else {
            step * 0.5
        };
        let mut index = 0;
        for _ in 0..count {
            while index + 1 < mapped && position > f64::from(element_sum[index]) {
                index += 1;
            }
            let element = element_map[index];
            elements.push(element);
            jitter_offset[element] = position as f32;
            position += step;
        }
    }
    (elements, jitter_offset)
}

/// The vertex that each of the `count` particles is emitted from (`particle_element` for `PART_FROM_VERT`).
///
/// `vertex_count` is the number of emitting vertices. `system_seed` is `ParticleSystem.seed`.
/// Vertex emission ignores `use_even_distribution`: every vertex weighs `1 / min(vertices, particles)`.
pub(crate) fn vertex_elements(
    vertex_count: usize,
    count: usize,
    system_seed: u32,
    random: bool,
) -> Vec<usize> {
    if vertex_count == 0 || count == 0 {
        return Vec::new();
    }
    let weight = 1.0_f32 / (vertex_count.min(count) as f32);
    let weights = vec![weight; vertex_count];
    assign_elements(&weights, count, system_seed, random).0
}

/// Blender's legacy `MFace` (`BKE_mesh_tessface_calc`): a triangle or quad, in corner order.
#[derive(Clone, Copy, Debug)]
struct Tessface {
    /// Corner indices into [`FaceEmitter::positions`]; the fourth entry is unused by a triangle.
    vertices: [usize; 4],
    quad: bool,
    /// `ME_SMOOTH`: the normal interpolates vertex normals rather than the face normal.
    smooth: bool,
}

/// The emitting mesh in single precision: its tessfaces, vertex positions, and vertex normals.
pub(crate) struct FaceEmitter {
    tessfaces: Vec<Tessface>,
    positions: Vec<Float3>,
    normals: Vec<Float3>,
}

/// The `ParticleSettings` that `distribute_particles` reads for face and volume emission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceSettings {
    /// `PART_FROM_VOLUME`: the birth point is pushed inward along the normal.
    pub(crate) volume: bool,
    /// `PART_DISTR_JIT` (`true`) or `PART_DISTR_RAND`.
    pub(crate) jitter: bool,
    /// `PART_EDISTR`: elements weigh their area.
    pub(crate) even: bool,
    /// `PART_TRAND`: particles are scattered over the elements at random.
    pub(crate) random: bool,
    /// `ParticleSettings.jitter_factor` (`jitfac`).
    pub(crate) jitter_factor: f32,
    /// `ParticleSettings.userjit`: jitter positions per element, or `0` for the automatic count.
    pub(crate) user_jitter: usize,
    /// `ParticleSystem.seed`.
    pub(crate) seed: u32,
    /// Number of particles (`totpart`).
    pub(crate) count: usize,
}

/// One particle's birth on a face or volume emitter: the position and unit normal of
/// `psys_particle_on_dm`, in mesh-local coordinates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceSample {
    pub(crate) position: DVec3,
    pub(crate) normal: DVec3,
}

impl FaceEmitter {
    /// Tessellates `mesh` as Blender's legacy `MFace` list: triangles and quads keep their corner
    /// order, and a larger face splits into the triangles of `loop_triangles`, in face order.
    ///
    /// # Errors
    ///
    /// Returns an error when a face references a vertex the mesh does not have.
    pub(crate) fn new(mesh: &Mesh) -> Result<Self> {
        let index_of: HashMap<u32, usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| {
                        index_of.get(id).copied().ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InvalidArgument,
                                "particle emitter face references a missing vertex",
                            )
                        })
                    })
                    .collect::<Result<Vec<usize>>>()
            })
            .collect::<Result<Vec<Vec<usize>>>>()?;
        let smooth = normals::smooth_face_flags(mesh)?;
        let points: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
        let corner_slices: Vec<&[usize]> = faces.iter().map(Vec::as_slice).collect();
        let vertex_normals: Vec<Float3> = normals::vertex_normals(&points, &corner_slices)
            .into_iter()
            .map(to_float3)
            .collect();
        let positions: Vec<Float3> = points.iter().copied().map(to_float3).collect();

        let mut tessfaces = Vec::with_capacity(faces.len());
        for (face, &smooth_face) in faces.iter().zip(&smooth) {
            match face.len() {
                3 => tessfaces.push(Tessface {
                    vertices: [face[0], face[1], face[2], 0],
                    quad: false,
                    smooth: smooth_face,
                }),
                4 => tessfaces.push(Tessface {
                    vertices: [face[0], face[1], face[2], face[3]],
                    quad: true,
                    smooth: smooth_face,
                }),
                count if count > 4 => {
                    let ring: Vec<[f32; 3]> = face.iter().map(|&index| positions[index]).collect();
                    for [first, second, third] in triangulate::loop_triangles(&ring) {
                        tessfaces.push(Tessface {
                            vertices: [face[first], face[second], face[third], 0],
                            quad: false,
                            smooth: smooth_face,
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(Self {
            tessfaces,
            positions,
            normals: vertex_normals,
        })
    }

    /// The number of tessfaces, the elements that particles are distributed over.
    pub(crate) fn element_count(&self) -> usize {
        self.tessfaces.len()
    }

    /// `distribute_particles`' element weights: each face's area as a share of the total with
    /// `PART_EDISTR`, otherwise `1 / min(elements, particles)` for every element.
    fn element_weights(&self, even: bool, count: usize) -> Vec<f32> {
        let elements = self.tessfaces.len();
        if !even {
            return vec![1.0_f32 / (elements.min(count) as f32); elements];
        }
        let mut weights: Vec<f32> = self.tessfaces.iter().map(|face| self.area(face)).collect();
        let mut total = 0.0_f32;
        for &weight in &weights {
            total += weight;
        }
        for weight in &mut weights {
            *weight /= total;
        }
        weights
    }

    /// Blender's `area_tri_v3` or `area_quad_v3` for the tessface.
    fn area(&self, face: &Tessface) -> f32 {
        let (corners, count) = gather(&self.positions, face);
        if face.quad {
            polygon_area(&corners[..count])
        } else {
            triangle_area(corners[0], corners[1], corners[2])
        }
    }
}

/// Distributes `settings.count` particles over `emitter` (`distribute_particles`), then places each
/// one (`distribute_from_faces_exec`, `distribute_from_volume_exec`, `psys_particle_on_dm`).
///
/// Returns one sample per particle, in particle order, or no samples when no element can take a
/// particle (as when every face has zero area).
pub(crate) fn distribute_faces(emitter: &FaceEmitter, settings: FaceSettings) -> Vec<FaceSample> {
    let elements = emitter.tessfaces.len();
    let count = settings.count;
    if elements == 0 || count == 0 {
        return Vec::new();
    }
    let weights = emitter.element_weights(settings.even, count);
    let (particle_element, jitter_offset) =
        assign_elements(&weights, count, settings.seed, settings.random);
    if particle_element.len() != count {
        return Vec::new();
    }

    // `distribute_particles`: a zero `userjit` sizes the jitter to the particles per element.
    let jitlevel = if settings.user_jitter == 0 {
        let even_factor = if settings.even { 2 } else { 1 };
        (count / elements * even_factor).max(3)
    } else {
        settings.user_jitter
    };
    let jit = if settings.jitter {
        jitter_table(jitlevel, settings.seed, settings.jitter_factor)
    } else {
        Vec::new()
    };

    // `psys_task_init_distribute`: one generator, seeded `31415926 + seed`, skipping three draws per
    // particle, so the sequence does not depend on how the work is split.
    let mut random = BlenderRng::seed(31_415_926_u32.wrapping_add(settings.seed));
    let mut samples = Vec::with_capacity(count);
    for (particle, &element) in particle_element.iter().enumerate() {
        let face = emitter.tessfaces[element];
        let corner_count = if face.quad { 4 } else { 3 };
        let mut drawn = 0;
        let fuv = if !settings.jitter {
            let u = random.get_float();
            let v = random.get_float();
            drawn = 2;
            uv_weights(u, v, face.quad)
        } else if jitlevel == 1 {
            if face.quad {
                uv_weights(0.5, 0.5, true)
            } else {
                uv_weights(1.0 / 3.0, 1.0 / 3.0, false)
            }
        } else {
            let offset = (jitter_offset[element] + particle as f32) % jitlevel as f32;
            let row = (offset as usize).min(jitlevel - 1);
            uv_weights(jit[row][0], jit[row][1], face.quad)
        };

        // `psys_interpolate_face`: the position, and the normal that is flat or smooth per face.
        let (corners, _) = gather(&emitter.positions, &face);
        let position = interpolate(&corners[..corner_count], &fuv[..corner_count]);
        let raw_normal = if face.smooth {
            let (vertex_normals, _) = gather(&emitter.normals, &face);
            interpolate(&vertex_normals[..corner_count], &fuv[..corner_count])
        } else if face.quad {
            quad_normal(corners[0], corners[1], corners[2], corners[3])
        } else {
            triangle_normal(corners[0], corners[1], corners[2])
        };
        let (normal, _) = normalize(raw_normal);

        let mut birth = position;
        if settings.volume {
            // `distribute_from_volume_exec`: the nearest other face along the inward normal sets the
            // depth, halved. The jitter (or a random draw) scales that depth.
            let direction = [-normal[0], -normal[1], -normal[2]];
            let mut nearest = f32::MAX;
            let mut foffset = 0.0_f32;
            let mut hit = false;
            for (other, candidate) in emitter.tessfaces.iter().enumerate() {
                if other == element {
                    continue;
                }
                let point = |slot: usize| emitter.positions[candidate.vertices[slot]];
                let mut distances = [
                    ray_triangle(position, direction, point(1), point(2), point(0)),
                    None,
                ];
                if candidate.quad {
                    distances[1] = ray_triangle(position, direction, point(3), point(0), point(2));
                }
                for distance in distances.into_iter().flatten() {
                    if distance < nearest {
                        nearest = distance;
                        foffset = distance * 0.5;
                        hit = true;
                    }
                }
            }
            if !hit {
                foffset = 0.0;
            } else if settings.jitter {
                let flat = particle % (2 * jitlevel);
                foffset *= jit[flat / 2][flat % 2];
            } else {
                foffset *= random.get_float();
                drawn += 1;
            }
            // `psys_particle_on_dm`: the birth point moves by `normal * -foffset`.
            let inward = -foffset;
            birth = std::array::from_fn(|axis| position[axis] + normal[axis] * inward);
        }
        random.skip(3 - drawn);
        samples.push(FaceSample {
            position: to_dvec3(birth),
            normal: to_dvec3(normal),
        });
    }
    samples
}

/// The corners of a tessface, with the count (three or four) of the ones in use.
fn gather(values: &[Float3], face: &Tessface) -> ([Float3; 4], usize) {
    let count = if face.quad { 4 } else { 3 };
    let mut out = [[0.0_f32; 3]; 4];
    for (slot, &index) in out.iter_mut().zip(&face.vertices[..count]) {
        *slot = values[index];
    }
    (out, count)
}

/// `psys_uv_to_w`: the weights of the face-local point `(u, v)`. A triangle folds `(u, v)` into the
/// half that `(0,0), (1,0), (1,1)` covers; a quad uses `(0,0), (1,0), (1,1), (0,1)`. Unused weights
/// are zero.
fn uv_weights(u: f32, v: f32, quad: bool) -> [f32; 4] {
    if quad {
        interp_weights_poly(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [u, v, 0.0],
        )
    } else {
        let (u, v) = if u + v > 1.0 {
            (u, 1.0 - v)
        } else {
            (1.0 - u, v)
        };
        interp_weights_poly(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [u, v, 0.0],
        )
    }
}

/// Blender's `interp_weights_poly_v3`: mean-value weights of `co` in a polygon of three or four
/// corners. Where `co` lies on an edge or a corner, the linear weights of that edge or corner.
fn interp_weights_poly(polygon: &[Float3], co: Float3) -> [f32; 4] {
    let count = polygon.len();
    let max_value = polygon
        .iter()
        .flat_map(|corner| corner.iter().zip(co.iter()).map(|(a, b)| (a - b).abs()))
        .fold(0.0_f32, f32::max);
    let eps = 16.0 * f32::EPSILON * max_value;
    let eps_squared = eps * eps;

    let mut weights = [0.0_f32; 4];
    let mut totweight = 0.0_f32;
    let mut current = count - 1;
    let mut next = 0;
    let mut current_corner = polygon[current];
    let mut next_corner = polygon[next];
    let mut direction_current = Direction::new(polygon[count - 2], co);
    let mut direction_next = Direction::new(current_corner, co);
    let mut half_tangent_previous = mean_value_half_tangent(direction_current, direction_next);
    let mut hit = None;
    while next < count {
        if direction_next.length < eps {
            hit = Some(Hit::Point);
            break;
        }
        if distance_squared_to_segment(co, current_corner, next_corner) < eps_squared {
            hit = Some(Hit::Segment);
            break;
        }
        direction_current = direction_next;
        direction_next = Direction::new(next_corner, co);
        let half_tangent = mean_value_half_tangent(direction_current, direction_next);
        weights[current] = (half_tangent_previous + half_tangent) / direction_current.length;
        totweight += weights[current];
        current = next;
        next += 1;
        current_corner = next_corner;
        if next < count {
            next_corner = polygon[next];
        }
        half_tangent_previous = half_tangent;
    }
    match hit {
        Some(Hit::Point) => {
            weights = [0.0; 4];
            weights[current] = 1.0;
        }
        Some(Hit::Segment) => {
            weights = [0.0; 4];
            let factor = line_point_factor(co, current_corner, next_corner).clamp(0.0, 1.0);
            weights[current] = 1.0 - factor;
            weights[next] = factor;
        }
        None => {
            if totweight > 0.0 {
                for weight in &mut weights[..count] {
                    *weight /= totweight;
                }
            }
        }
    }
    weights
}

/// Where `interp_weights_poly_v3` stopped on a corner or an edge.
#[derive(Clone, Copy)]
enum Hit {
    Point,
    Segment,
}

/// Blender's `Float3_Len`: a vector from a point to a corner, and its length.
#[derive(Clone, Copy)]
struct Direction {
    vector: Float3,
    length: f32,
}

impl Direction {
    /// `DIR_V3_SET`: `from - to` and its length.
    fn new(from: Float3, to: Float3) -> Self {
        let vector = sub(from, to);
        Self {
            vector,
            length: length(vector),
        }
    }
}

/// Blender's `mean_value_half_tan_v3`: the half tangent between two directions from the point.
fn mean_value_half_tangent(current: Direction, next: Direction) -> f32 {
    let area = length(cross(current.vector, next.vector));
    if area > 0.0 {
        let result = (current.length * next.length - dot(current.vector, next.vector)) / area;
        if result.is_finite() {
            return result;
        }
    }
    0.0
}

/// `dist_squared_to_line_segment_v3`: the squared distance from `point` to the segment, clamped to
/// its ends (`closest_to_line_segment_v3`).
fn distance_squared_to_segment(point: Float3, first: Float3, second: Float3) -> f32 {
    let direction = sub(second, first);
    let squared = dot(direction, direction);
    let lambda = if squared > 0.0 {
        dot(direction, sub(point, first)) / squared
    } else {
        0.0
    };
    let closest = if lambda <= 0.0 {
        first
    } else if lambda >= 1.0 {
        second
    } else {
        [
            first[0] + direction[0] * lambda,
            first[1] + direction[1] * lambda,
            first[2] + direction[2] * lambda,
        ]
    };
    let offset = sub(closest, point);
    dot(offset, offset)
}

/// `line_point_factor_v3`: where `point` projects onto the line through the segment, zero when the
/// segment has no length.
fn line_point_factor(point: Float3, first: Float3, second: Float3) -> f32 {
    let direction = sub(second, first);
    let squared = dot(direction, direction);
    if squared > 0.0 {
        dot(direction, sub(point, first)) / squared
    } else {
        0.0
    }
}

/// Blender's `interp_v3_v3v3v3(v4)`: the weighted sum of corner values, in corner order.
fn interpolate(values: &[Float3], weights: &[f32]) -> Float3 {
    let mut out = [0.0_f32; 3];
    for (axis, slot) in out.iter_mut().enumerate() {
        for (value, &weight) in values.iter().zip(weights) {
            *slot += value[axis] * weight;
        }
    }
    out
}

/// Blender's `isect_ray_tri_v3`: the distance along `direction` from `origin` to the triangle, when
/// the ray meets it at a non-negative distance.
fn ray_triangle(
    origin: Float3,
    direction: Float3,
    first: Float3,
    second: Float3,
    third: Float3,
) -> Option<f32> {
    const EPSILON: f32 = 0.000_000_01;
    let edge_one = sub(second, first);
    let edge_two = sub(third, first);
    let p = cross(direction, edge_two);
    let determinant = dot(edge_one, p);
    if determinant > -EPSILON && determinant < EPSILON {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = sub(origin, first);
    let u = inverse * dot(s, p);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, edge_one);
    let v = inverse * dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let lambda = inverse * dot(edge_two, q);
    if lambda < 0.0 {
        return None;
    }
    Some(lambda)
}

/// `normal_tri_v3`: the unit normal of a triangle.
fn triangle_normal(first: Float3, second: Float3, third: Float3) -> Float3 {
    normalize(cross(sub(first, second), sub(second, third))).0
}

/// `normal_quad_v3`: the unit normal of a quad, from its diagonals.
fn quad_normal(first: Float3, second: Float3, third: Float3, fourth: Float3) -> Float3 {
    normalize(cross(sub(first, third), sub(second, fourth))).0
}

/// `area_tri_v3`: half the length of the triangle's cross product.
fn triangle_area(first: Float3, second: Float3, third: Float3) -> f32 {
    length(cross(sub(first, second), sub(second, third))) * 0.5
}

/// `area_poly_v3`: half the length of the polygon's Newell normal.
fn polygon_area(corners: &[Float3]) -> f32 {
    let mut normal = [0.0_f32; 3];
    let mut previous = corners[corners.len() - 1];
    for &current in corners {
        // `add_newell_cross_v3_v3v3`
        normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
        normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
        normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
        previous = current;
    }
    length(normal) * 0.5
}

/// `normalize_v3`: the unit vector and its length, or the zero vector when `a` is too short.
fn normalize(a: Float3) -> (Float3, f32) {
    let squared = dot(a, a);
    if squared > 1.0e-35 {
        let length = squared.sqrt();
        let scale = 1.0 / length;
        ([a[0] * scale, a[1] * scale, a[2] * scale], length)
    } else {
        ([0.0; 3], 0.0)
    }
}

fn sub(a: Float3, b: Float3) -> Float3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Float3, b: Float3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Float3, b: Float3) -> Float3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: Float3) -> f32 {
    dot(a, a).sqrt()
}

fn to_float3(point: DVec3) -> Float3 {
    [point.x as f32, point.y as f32, point.z as f32]
}

fn to_dvec3(point: Float3) -> DVec3 {
    DVec3::new(
        f64::from(point[0]),
        f64::from(point[1]),
        f64::from(point[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::{interp_weights_poly, uv_weights, vertex_elements};

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn one_vertex_per_particle_when_counts_match_without_randomness() {
        // Evenly spaced positions over equal weights hit each vertex once, in order.
        assert_eq!(vertex_elements(4, 4, 0, false), vec![0, 1, 2, 3]);
    }

    #[test]
    fn every_particle_names_a_vertex() {
        let elements = vertex_elements(8, 20, 7, true);
        assert_eq!(elements.len(), 20);
        assert!(elements.iter().all(|index| *index < 8));
    }

    #[test]
    fn quad_weights_reproduce_the_point_they_name() {
        let (u, v) = (0.3_f32, 0.6_f32);
        let weights = uv_weights(u, v, true);
        assert_close(weights.iter().sum(), 1.0);
        assert_close(weights[1] + weights[2], u);
        assert_close(weights[2] + weights[3], v);
    }

    #[test]
    fn triangle_weights_are_barycentric_in_the_folded_half() {
        // (0.2, 0.3) folds to (0.8, 0.3): barycentric weights of (0,0), (1,0), (1,1) are 0.2, 0.5, 0.3.
        let weights = uv_weights(0.2, 0.3, false);
        assert_close(weights[0], 0.2);
        assert_close(weights[1], 0.5);
        assert_close(weights[2], 0.3);
        assert_close(weights[3], 0.0);
    }

    #[test]
    fn mean_value_weights_hit_a_corner_exactly() {
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let weights = interp_weights_poly(&square, [1.0, 1.0, 0.0]);
        assert_close(weights[2], 1.0);
        assert_close(weights[0] + weights[1] + weights[3], 0.0);
    }
}
