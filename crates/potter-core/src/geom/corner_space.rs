//! Blender's corner spaces for custom normals (`mesh_normals.cc`).
//!
//! Blender stores a custom normal per corner as two 16-bit values: an angle from the corner's
//! fan normal and an angle around it, measured from the fan's reference direction. A modifier
//! that copies those raw values to another corner reinterprets them in the new corner's space,
//! so the normal a copy evaluates to depends on both fans. [`CornerFans`] builds the fans and
//! spaces of a mesh as `normals_calc_corners` does, and encodes and decodes custom normals as
//! `normals_corner_custom_set` and `corner_space_custom_data_to_normal` do.

use std::collections::{HashMap, HashSet};
use std::f64::consts::TAU;

use glam::DVec3;

use super::normals::{face_normal, safe_acos_approx};

/// Blender's `LNOR_SPACE_TRIGO_THRESHOLD`: directions this close to the fan normal are degenerate.
const TRIGO_THRESHOLD: f64 = 1.0 - 1.0e-4;
/// Blender's `SHRT_MAX`: custom normal components are fractions of the fan angles.
const SHORT_MAX: f64 = 32767.0;

/// The space of one fan (`CornerNormalSpace`): the fan normal, an orthonormal reference pair
/// around it, and the fan's angular extents. A degenerate fan has zero extents and a zero normal.
#[derive(Clone, Copy, Debug)]
struct CornerSpace {
    lnor: DVec3,
    vec_ref: DVec3,
    vec_ortho: DVec3,
    ref_alpha: f64,
    ref_beta: f64,
}

impl CornerSpace {
    /// `corner_fan_space_define`: the space of a fan with normal `lnor`, whose first and last
    /// edges point along `vec_ref` and `vec_other`. `edge_vectors` are the fan's edges.
    fn define(lnor: DVec3, vec_ref: DVec3, vec_other: DVec3, edge_vectors: &[DVec3]) -> Self {
        let dtp_ref = vec_ref.dot(lnor);
        let dtp_other = vec_other.dot(lnor);
        if dtp_ref.abs() >= TRIGO_THRESHOLD || dtp_other.abs() >= TRIGO_THRESHOLD {
            return Self {
                lnor: DVec3::ZERO,
                vec_ref: DVec3::ZERO,
                vec_ortho: DVec3::ZERO,
                ref_alpha: 0.0,
                ref_beta: 0.0,
            };
        }
        let ref_alpha = if edge_vectors.is_empty() {
            f64::midpoint(safe_acos_approx(dtp_ref), safe_acos_approx(dtp_other))
        } else {
            edge_vectors
                .iter()
                .map(|edge| safe_acos_approx(edge.dot(lnor)))
                .sum::<f64>()
                / edge_vectors.len() as f64
        };
        let vec_ref = (vec_ref - lnor * dtp_ref).normalize_or_zero();
        let vec_ortho = lnor.cross(vec_ref).normalize_or_zero();
        let other = (vec_other - lnor * dtp_other).normalize_or_zero();
        let dtp = vec_ref.dot(other);
        let ref_beta = if dtp < TRIGO_THRESHOLD {
            let beta = safe_acos_approx(dtp);
            if vec_ortho.dot(other) < 0.0 {
                TAU - beta
            } else {
                beta
            }
        } else {
            TAU
        };
        Self {
            lnor,
            vec_ref,
            vec_ortho,
            ref_alpha,
            ref_beta,
        }
    }

    /// `BKE_lnor_space_custom_normal_to_data`: the two 16-bit components of `normal`.
    fn encode(&self, normal: DVec3) -> [i16; 2] {
        if self.ref_alpha == 0.0 || self.ref_beta == 0.0 {
            return [0, 0];
        }
        if normal == DVec3::ZERO || (self.lnor - normal).abs().max_element() <= 1.0e-4 {
            return [0, 0];
        }
        let cos_alpha = self.lnor.dot(normal);
        let alpha = safe_acos_approx(cos_alpha);
        let alpha_factor = if alpha > self.ref_alpha {
            -(TAU - alpha) / (TAU - self.ref_alpha)
        } else {
            alpha / self.ref_alpha
        };
        let projected = (self.lnor * -cos_alpha + normal).normalize_or_zero();
        let cos_beta = self.vec_ref.dot(projected);
        let beta_factor = if cos_beta < TRIGO_THRESHOLD {
            let mut beta = safe_acos_approx(cos_beta);
            if self.vec_ortho.dot(projected) < 0.0 {
                beta = TAU - beta;
            }
            if beta > self.ref_beta {
                -(TAU - beta) / (TAU - self.ref_beta)
            } else {
                beta / self.ref_beta
            }
        } else {
            0.0
        };
        [to_short(alpha_factor), to_short(beta_factor)]
    }

    /// `corner_space_custom_data_to_normal`: the normal the two 16-bit components stand for.
    fn decode(&self, data: [i16; 2]) -> DVec3 {
        if data[0] == 0 || self.ref_alpha == 0.0 || self.ref_beta == 0.0 {
            return self.lnor;
        }
        let alpha_factor = f64::from(data[0]) / SHORT_MAX;
        let alpha = if alpha_factor > 0.0 {
            self.ref_alpha
        } else {
            TAU - self.ref_alpha
        } * alpha_factor;
        let beta_factor = f64::from(data[1]) / SHORT_MAX;
        let mut normal = self.lnor * alpha.cos();
        if beta_factor == 0.0 {
            normal += self.vec_ref * alpha.sin();
        } else {
            let beta = if beta_factor > 0.0 {
                self.ref_beta
            } else {
                TAU - self.ref_beta
            } * beta_factor;
            normal +=
                self.vec_ref * alpha.sin() * beta.cos() + self.vec_ortho * alpha.sin() * beta.sin();
        }
        normal
    }
}

/// Blender's `unit_float_to_short`.
fn to_short(value: f64) -> i16 {
    (value * SHORT_MAX + 0.5)
        .floor()
        .clamp(-SHORT_MAX, SHORT_MAX) as i16
}

/// Corner topology of a mesh: corners are numbered in face order.
struct Corners {
    vertex: Vec<usize>,
    face: Vec<usize>,
    prev: Vec<usize>,
    next: Vec<usize>,
    /// Corner indices at each vertex, in face order.
    at_vertex: Vec<Vec<usize>>,
}

/// One fan: the corners sharing a normal around a vertex, in Blender's order, with its space.
#[derive(Clone, Debug)]
struct Fan {
    corners: Vec<usize>,
    space: CornerSpace,
}

/// The fans of every corner of a mesh.
#[derive(Clone, Debug)]
pub(crate) struct CornerFans {
    /// Per corner: index into `fans`.
    fan_of: Vec<usize>,
    fans: Vec<Fan>,
}

impl CornerFans {
    /// Fans of a mesh given as vertex index lists per face, with smooth flags per face and the
    /// sharp edges as sorted vertex pairs. A fan stops at sharp edges, at edges of flat faces, at
    /// non-manifold edges, and at edges whose two faces wind the same way (`add_corner_to_edge`).
    pub(crate) fn new(
        faces: &[Vec<usize>],
        positions: &[DVec3],
        smooth: &[bool],
        sharp: &HashSet<(usize, usize)>,
    ) -> Self {
        let mut corners = Corners {
            vertex: Vec::new(),
            face: Vec::new(),
            prev: Vec::new(),
            next: Vec::new(),
            at_vertex: vec![Vec::new(); positions.len()],
        };
        let mut face_normals = Vec::with_capacity(faces.len());
        for (face_index, face) in faces.iter().enumerate() {
            let start = corners.vertex.len();
            let count = face.len();
            let points: Vec<DVec3> = face.iter().map(|index| positions[*index]).collect();
            face_normals.push(face_normal(&points));
            for (position, vertex) in face.iter().enumerate() {
                let corner = start + position;
                corners.vertex.push(*vertex);
                corners.face.push(face_index);
                corners.prev.push(start + (position + count - 1) % count);
                corners.next.push(start + (position + 1) % count);
                corners.at_vertex[*vertex].push(corner);
            }
        }

        // `next_link[c]` is the corner whose previous edge is `c`'s next edge, and `prev_link`
        // the reverse: the links of a fan.
        let total = corners.vertex.len();
        let mut next_link: Vec<Option<usize>> = vec![None; total];
        let mut prev_link: Vec<Option<usize>> = vec![None; total];
        for (vertex, at_vertex) in corners.at_vertex.iter().enumerate() {
            // Per other vertex: the corners using that edge at `vertex`, as (corner, winding
            // towards `vertex`). A flat face's corner has no corner: its edges end every fan.
            let mut uses: HashMap<usize, Vec<Option<(usize, bool)>>> = HashMap::new();
            for &corner in at_vertex {
                let usable = smooth[corners.face[corner]].then_some(corner);
                for (other, towards) in [
                    (corners.vertex[corners.next[corner]], false),
                    (corners.vertex[corners.prev[corner]], true),
                ] {
                    uses.entry(other)
                        .or_default()
                        .push(usable.map(|corner| (corner, towards)));
                }
            }
            for (other, entries) in uses {
                let key = (vertex.min(other), vertex.max(other));
                if entries.len() != 2 {
                    continue;
                }
                let (Some(first), Some(second)) = (entries[0], entries[1]) else {
                    continue;
                };
                if sharp.contains(&key) || first.1 == second.1 {
                    continue;
                }
                let (towards, away) = if first.1 {
                    (first, second)
                } else {
                    (second, first)
                };
                next_link[away.0] = Some(towards.0);
                prev_link[towards.0] = Some(away.0);
            }
        }

        let mut fan_of = vec![0; total];
        let mut fans = Vec::new();
        let mut visited = vec![false; total];
        for start in 0..total {
            if visited[start] {
                continue;
            }
            let fan_corners = traverse(start, &next_link, &prev_link);
            for &corner in &fan_corners {
                visited[corner] = true;
                fan_of[corner] = fans.len();
            }
            let space = fan_space(&fan_corners, &corners, &face_normals, positions);
            fans.push(Fan {
                corners: fan_corners,
                space,
            });
        }
        Self { fan_of, fans }
    }

    /// Stored custom data for corner normals `normals` (`normals_corner_custom_set`): each fan
    /// averages its corners' normals and encodes the average in its own space.
    pub(crate) fn encode(&self, normals: &[DVec3]) -> Vec<[i16; 2]> {
        let mut data = vec![[0; 2]; self.fan_of.len()];
        for fan in &self.fans {
            let sum: DVec3 = fan.corners.iter().map(|corner| normals[*corner]).sum();
            let encoded = fan.space.encode(sum / fan.corners.len() as f64);
            for &corner in &fan.corners {
                data[corner] = encoded;
            }
        }
        data
    }

    /// Corner normals of stored custom data (`handle_fan_result_and_custom_normals`): each fan
    /// averages its corners' data with integer division and decodes it in its own space.
    pub(crate) fn decode(&self, data: &[[i16; 2]]) -> Vec<DVec3> {
        let mut normals = vec![DVec3::ZERO; self.fan_of.len()];
        for fan in &self.fans {
            let count = fan.corners.len() as i32;
            let (alpha, beta) = fan.corners.iter().fold((0_i32, 0_i32), |sum, corner| {
                (
                    sum.0 + i32::from(data[*corner][0]),
                    sum.1 + i32::from(data[*corner][1]),
                )
            });
            let normal = fan
                .space
                .decode([(alpha / count) as i16, (beta / count) as i16]);
            for &corner in &fan.corners {
                normals[corner] = normal;
            }
        }
        normals
    }
}

/// The fan through `start` in Blender's order (`traverse_fan_local_corners`). A cycle starts at
/// its smallest corner index. An open chain runs from the corner whose next edge is open to the
/// corner whose previous edge is open.
fn traverse(start: usize, next_link: &[Option<usize>], prev_link: &[Option<usize>]) -> Vec<usize> {
    let mut fan = vec![start];
    let mut current = start;
    let mut cyclic = false;
    while let Some(next) = next_link[current] {
        if next == start {
            cyclic = true;
            break;
        }
        fan.push(next);
        current = next;
    }
    fan.reverse();
    if cyclic {
        let first = (0..fan.len())
            .min_by_key(|index| fan[*index])
            .unwrap_or_default();
        fan.rotate_left(first);
        return fan;
    }
    let mut current = start;
    while let Some(previous) = prev_link[current] {
        fan.push(previous);
        current = previous;
    }
    fan
}

/// The space of a fan (`accumulate_fan_normal` and `corner_fan_space_define`). The fan normal is
/// the angle-weighted sum of its face normals, and the first corner's next edge and the last
/// corner's previous edge set the reference directions.
fn fan_space(
    fan: &[usize],
    corners: &Corners,
    face_normals: &[DVec3],
    positions: &[DVec3],
) -> CornerSpace {
    // Unit direction from `corner`'s vertex to the vertex of `other`.
    let edge = |corner: usize, other: usize| {
        (positions[corners.vertex[other]] - positions[corners.vertex[corner]]).normalize_or_zero()
    };
    let first = fan[0];
    let last = fan[fan.len() - 1];
    let normal = if fan.len() == 1 {
        face_normals[corners.face[first]]
    } else {
        let sum: DVec3 = fan
            .iter()
            .map(|corner| {
                let previous = edge(*corner, corners.prev[*corner]);
                let next = edge(*corner, corners.next[*corner]);
                face_normals[corners.face[*corner]] * safe_acos_approx(previous.dot(next))
            })
            .sum();
        sum.normalize_or_zero()
    };
    let vec_ref = edge(first, corners.next[first]);
    let vec_other = edge(last, corners.prev[last]);
    let mut edge_vectors = Vec::new();
    if fan.len() > 1 {
        edge_vectors.extend(
            fan.iter()
                .map(|corner| edge(*corner, corners.next[*corner])),
        );
        if corners.vertex[corners.prev[last]] != corners.vertex[corners.next[first]] {
            edge_vectors.push(vec_other);
        }
    }
    CornerSpace::define(normal, vec_ref, vec_other, &edge_vectors)
}
