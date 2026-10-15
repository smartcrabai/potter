// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_converter_mesh.cc, source/blender/modifiers/intern/MOD_subsurf.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.
//   OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/fvarLevel.cpp, far/primvarRefiner.h, Apache-2.0.

//! Face-varying UV refinement and bilinear (`SIMPLE`) subdivision.
//!
//! Blender refines each UV layer as an `OpenSubdiv` face-varying channel. A channel value is a position
//! vertex with one UV, so a vertex splits wherever its UVs differ between faces. Each level derives the
//! child channel from the parent topology with the face-varying Catmull-Clark rules, where `uv_smooth`
//! picks the linear option and `boundary_smooth` the boundary-corner option. SIMPLE subdivision uses
//! bilinear positions, and its UVs are linear too (Blender makes every edge and vertex sharp).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::{DVec2, DVec3};
use serde_json::{Map, Value, json};

use super::{
    Polygon, build_mesh, edge_crease_weights, loose_edges, polygons, positions, usize_as_f64,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, edge_key,
        provenance::{Origin, uv_map_entry_layer},
    },
};

/// Face-varying linear interpolation (`OpenSubdiv` `FVarLinearInterpolation`), chosen from `uv_smooth`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FvarLinear {
    /// `uv_smooth` NONE: every edge and vertex is linear, so the UVs interpolate bilinearly.
    All,
    /// `uv_smooth` `SMOOTH_ALL`: smooth everywhere, UV boundaries included.
    SmoothAll,
    /// `uv_smooth` `PRESERVE_CORNERS`: UV corners stay put.
    CornersOnly,
    /// `uv_smooth` `PRESERVE_CORNERS_AND_JUNCTIONS`: corners and junctions stay put.
    CornersPlus1,
    /// `uv_smooth` `PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE`: corners, junctions and concave corners.
    CornersPlus2,
    /// `uv_smooth` `PRESERVE_BOUNDARIES` (Blender default): UV boundaries are linear.
    Boundaries,
}

impl FvarLinear {
    /// The option for a validated `uv_smooth` value; SIMPLE subdivision is always linear.
    pub(super) fn from_uv_smooth(uv_smooth: &str, simple: bool) -> Self {
        match uv_smooth {
            _ if simple => Self::All,
            "NONE" => Self::All,
            "SMOOTH_ALL" => Self::SmoothAll,
            "PRESERVE_CORNERS" => Self::CornersOnly,
            "PRESERVE_CORNERS_AND_JUNCTIONS" => Self::CornersPlus1,
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE" => Self::CornersPlus2,
            _ => Self::Boundaries,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct UvOptions {
    pub(super) linear: FvarLinear,
    /// `boundary_smooth` `PRESERVE_CORNERS`: single-face boundary vertices are sharp.
    pub(super) preserve_corners: bool,
    /// Whether `crease_edge` sharpens edges (`use_creases`).
    pub(super) use_creases: bool,
}

/// One UV layer as a face-varying channel: `corners[face][corner]` names the value at that corner.
#[derive(Clone)]
struct Channel {
    corners: Vec<Vec<usize>>,
    uvs: Vec<DVec2>,
    /// The level-0 value each value descends from through vertex points.
    origin: Vec<Option<usize>>,
}

/// Where a channel's UVs are written back.
enum Target {
    /// Index into `blender_uv_layers`.
    Blender(usize),
    /// Layer name in `uv_map` entries.
    UvMap(String),
}

struct Layer {
    target: Target,
    /// Level-0 channel and, per level-0 value, the two values at the ends of its crease (if any).
    base: Channel,
    base_ends: Vec<Option<[usize; 2]>>,
    current: Channel,
}

/// Face-varying refinement state for every UV layer of one subdivision.
pub(super) struct UvSubdivision {
    options: UvOptions,
    layers: Vec<Layer>,
}

/// Rule that refines one value's vertex point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// Sharp corner: the value does not move.
    Corner,
    /// Smooth boundary: `3/4 v + 1/8 (a + b)` with the values at the ends of its sharp edges.
    Crease([usize; 2]),
    /// Smooth interior value: Catmull-Clark vertex rule.
    Smooth,
}

struct Topology {
    rules: Vec<Rule>,
    /// The two value indices of each face-varying edge.
    edge_ends: Vec<[usize; 2]>,
    /// The `(face, corner)` sides of each face-varying edge.
    edge_sides: Vec<Vec<(usize, usize)>>,
    /// Face-varying edge of the side leaving each `(face, corner)` towards its next corner.
    side_edge: Vec<Vec<usize>>,
    /// Whether the edge point is the smooth average (continuous, manifold, sharp-free edge).
    edge_smooth: Vec<bool>,
}

/// Per value: corner count, sum of face points, sum of edge-neighbour values (each neighbour counted
/// once per corner, so twice per edge), and sum of opposite corner values.
#[derive(Clone, Copy, Default)]
struct Sums {
    count: usize,
    face: DVec2,
    neighbours: DVec2,
    opposite: DVec2,
}

/// Maximal run of one value around a position vertex, with the statistics `OpenSubdiv` tags.
struct Span {
    value: usize,
    size: usize,
    /// Sharp interior edges inside the run.
    inf: usize,
    /// Discontinuous interior edges, plus one per extra run of the same value (non-manifold).
    disct: usize,
    /// Values across the run's two end edges.
    ends: Option<[usize; 2]>,
}

/// A position vertex's faces in fan order. `corners[i]` and `corners[i + 1]` share an edge; a closed
/// fan wraps around, an open one starts and ends on a boundary.
struct Fan {
    corners: Vec<(usize, usize)>,
    closed: bool,
}

/// The edge between consecutive fan corners (`index - 1` and `index`); boundary links sit at the ends
/// of an open fan.
#[derive(Clone, Copy)]
struct Link {
    mismatch: bool,
    sharp: bool,
}

/// Mesh topology of one level, used to walk fans and read sharp edges.
struct Walk<'a> {
    polys: &'a [Vec<usize>],
    values: &'a [Vec<usize>],
    side_edge: &'a [Vec<usize>],
    position_sides: &'a BTreeMap<(usize, usize), Vec<(usize, usize)>>,
    creases: &'a BTreeMap<(usize, usize), f64>,
}

impl Walk<'_> {
    fn value(&self, (face, corner): (usize, usize)) -> usize {
        self.values[face][corner]
    }

    fn next(&self, (face, corner): (usize, usize)) -> usize {
        let size = self.values[face].len();
        self.values[face][(corner + 1) % size]
    }

    fn prev(&self, (face, corner): (usize, usize)) -> usize {
        let size = self.values[face].len();
        self.values[face][(corner + size - 1) % size]
    }

    /// Faces sharing the position edge that leaves `(face, corner)` towards its next corner.
    fn side_faces(&self, (face, corner): (usize, usize)) -> usize {
        let poly = &self.polys[face];
        let edge = edge_key(poly[corner], poly[(corner + 1) % poly.len()]);
        self.position_sides.get(&edge).map_or(0, Vec::len)
    }

    /// Whether the position edge leaving `(face, corner)` towards its next corner is sharp.
    fn sharp(&self, (face, corner): (usize, usize)) -> bool {
        let poly = &self.polys[face];
        let edge = edge_key(poly[corner], poly[(corner + 1) % poly.len()]);
        self.creases
            .get(&edge)
            .is_some_and(|sharpness| *sharpness >= 1.0)
    }

    /// The corner at `vertex` on the other face of the edge leaving `corner` forward or backward.
    /// `None` on a boundary or non-manifold edge.
    fn across(
        &self,
        (face, corner): (usize, usize),
        vertex: usize,
        forward: bool,
    ) -> Option<(usize, usize)> {
        let poly = &self.polys[face];
        let side = if forward {
            corner
        } else {
            (corner + poly.len() - 1) % poly.len()
        };
        let sides = self
            .position_sides
            .get(&edge_key(poly[side], poly[(side + 1) % poly.len()]))?;
        if sides.len() != 2 {
            return None;
        }
        let (other_face, other_side) =
            *sides.iter().find(|candidate| **candidate != (face, side))?;
        let other = &self.polys[other_face];
        let at_vertex = if other[other_side] == vertex {
            other_side
        } else {
            (other_side + 1) % other.len()
        };
        Some((other_face, at_vertex))
    }

    /// The fan around `vertex` through its `corners`; `None` when they do not form one manifold fan.
    fn fan(&self, vertex: usize, corners: &[(usize, usize)]) -> Option<Fan> {
        for corner in corners {
            let size = self.polys[corner.0].len();
            if self.side_faces(*corner) > 2
                || self.side_faces((corner.0, (corner.1 + size - 1) % size)) > 2
            {
                return None;
            }
        }
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let mut fans: Vec<Fan> = Vec::new();
        for &start in corners {
            if seen.contains(&start) {
                continue;
            }
            let mut first = start;
            let mut closed = false;
            for _ in 0..corners.len() {
                match self.across(first, vertex, false) {
                    None => break,
                    Some(previous) if previous == start => {
                        closed = true;
                        break;
                    }
                    Some(previous) => first = previous,
                }
            }
            let mut chain = vec![if closed { start } else { first }];
            loop {
                let last = chain[chain.len() - 1];
                match self.across(last, vertex, true) {
                    None => break,
                    Some(next) if next == chain[0] => break,
                    Some(next) => {
                        if chain.len() >= corners.len() {
                            return None;
                        }
                        chain.push(next);
                    }
                }
            }
            seen.extend(chain.iter().copied());
            fans.push(Fan {
                corners: chain,
                closed,
            });
        }
        (fans.len() == 1).then(|| fans.pop()).flatten()
    }
}

/// Values of every face corner of a channel, used to group the channel's values into faces.
fn face_points(channel: &Channel) -> Vec<DVec2> {
    channel
        .corners
        .iter()
        .map(|values| {
            values
                .iter()
                .fold(DVec2::ZERO, |sum, value| sum + channel.uvs[*value])
                / usize_as_f64(values.len())
        })
        .collect()
}

fn sums(channel: &Channel, face_points: &[DVec2]) -> Vec<Sums> {
    let uvs = &channel.uvs;
    let mut sums = vec![Sums::default(); uvs.len()];
    for (face, values) in channel.corners.iter().enumerate() {
        let size = values.len();
        for corner in 0..size {
            let sum = &mut sums[values[corner]];
            sum.count += 1;
            sum.face += face_points[face];
            sum.neighbours +=
                uvs[values[(corner + 1) % size]] + uvs[values[(corner + size - 1) % size]];
            sum.opposite += uvs[values[(corner + 2) % size]];
        }
    }
    sums
}

fn link(walk: &Walk, fan: &Fan, index: usize) -> Link {
    let size = fan.corners.len();
    if !fan.closed && (index == 0 || index == size) {
        return Link {
            mismatch: false,
            sharp: true,
        };
    }
    let before = fan.corners[(index + size - 1) % size];
    let after = fan.corners[index % size];
    let after_size = walk.polys[after.0].len();
    let after_back = walk.side_edge[after.0][(after.1 + after_size - 1) % after_size];
    Link {
        mismatch: walk.side_edge[before.0][before.1] != after_back,
        sharp: walk.sharp(before),
    }
}

/// The value across link `index`'s end, on the face of the corner before it, for a crease.
fn link_end(walk: &Walk, fan: &Fan, index: usize) -> usize {
    let size = fan.corners.len();
    if !fan.closed && index == 0 {
        walk.prev(fan.corners[0])
    } else if !fan.closed && index == size {
        walk.next(fan.corners[size - 1])
    } else {
        walk.next(fan.corners[(index + size - 1) % size])
    }
}

/// Sets the rule of every value at one manifold position vertex, from `OpenSubdiv`'s value tags
/// (`vtr::FVarLevel::completeTopologyFromFaceValues`) or, when the vertex is not split, its geometry
/// rule (`vtr::Refinement`).
fn vertex_rules(walk: &Walk, fan: &Fan, options: UvOptions, rules: &mut [Rule]) {
    let size = fan.corners.len();
    let closed = fan.closed;
    let boundary = !closed;
    let link_count = if closed { size } else { size + 1 };
    let links: Vec<Link> = (0..link_count)
        .map(|index| link(walk, fan, index))
        .collect();
    let linear = options.linear;
    let fvar_sharp = linear != FvarLinear::SmoothAll;
    let vertex_sharp = boundary && size == 1 && options.preserve_corners;
    let processed = links.iter().any(|link| link.mismatch)
        || (boundary && linear == FvarLinear::Boundaries)
        || (boundary && size == 1 && !options.preserve_corners && fvar_sharp);

    if !processed {
        let value = walk.value(fan.corners[0]);
        let sharp: Vec<usize> = (0..link_count)
            .filter(|index| links[*index].sharp)
            .collect();
        rules[value] = if vertex_sharp || sharp.len() > 2 {
            Rule::Corner
        } else if let &[first, second] = sharp.as_slice() {
            Rule::Crease([link_end(walk, fan, first), link_end(walk, fan, second)])
        } else {
            Rule::Smooth
        };
        return;
    }

    let values: Vec<usize> = fan
        .corners
        .iter()
        .map(|corner| walk.value(*corner))
        .collect();
    let mut spans: Vec<Span> = Vec::new();
    if closed && values.iter().all(|value| *value == values[0]) {
        // One value around an interior vertex: a single mismatched edge leaves a span covering the fan;
        // two or more make it non-manifold.
        let mismatched: Vec<usize> = (0..size).filter(|index| links[*index].mismatch).collect();
        spans.push(match mismatched.as_slice() {
            &[index] => Span {
                value: values[0],
                size,
                inf: (0..size)
                    .filter(|other| *other != index && links[*other].sharp)
                    .count(),
                disct: 0,
                ends: Some([
                    walk.prev(fan.corners[index]),
                    walk.next(fan.corners[(index + size - 1) % size]),
                ]),
            },
            _ => Span {
                value: values[0],
                size,
                inf: 0,
                disct: 1,
                ends: None,
            },
        });
    } else {
        // Runs of one value, rotated so that a closed fan starts at a change of value.
        let start = if closed {
            (0..size)
                .find(|&t| values[(t + size - 1) % size] != values[t])
                .unwrap_or(0)
        } else {
            0
        };
        let corner_of = |position: usize| {
            if closed {
                (start + position) % size
            } else {
                position
            }
        };
        let mut position = 0;
        while position < size {
            let first = position;
            let value = values[corner_of(first)];
            while position + 1 < size && values[corner_of(position + 1)] == value {
                position += 1;
            }
            let last = position;
            position += 1;
            let interior = first + 1..=last;
            let inf = interior
                .clone()
                .filter(|p| links[corner_of(*p)].sharp)
                .count();
            let disct = interior.filter(|p| links[corner_of(*p)].mismatch).count();
            let ends = Some([
                walk.prev(fan.corners[corner_of(first)]),
                walk.next(fan.corners[corner_of(last)]),
            ]);
            match spans.iter_mut().find(|span| span.value == value) {
                Some(span) => {
                    span.size += last - first + 1;
                    span.inf += inf;
                    span.disct += disct + 1;
                }
                None => spans.push(Span {
                    value,
                    size: last - first + 1,
                    inf,
                    disct,
                    ends,
                }),
            }
        }
    }

    let dependent = matches!(linear, FvarLinear::CornersPlus1 | FvarLinear::CornersPlus2);
    let sharpen_darts = matches!(linear, FvarLinear::CornersPlus2 | FvarLinear::Boundaries);
    let value_count = spans.len();
    let mut all_sharp = linear == FvarLinear::Boundaries
        || vertex_sharp
        || (dependent && value_count > 2)
        || (sharpen_darts && value_count == 1 && closed);
    if !all_sharp && dependent && value_count == 2 {
        all_sharp = spans.iter().any(|span| span.inf > 0 || span.disct > 0)
            || (linear == FvarLinear::CornersPlus2 && spans.iter().any(|span| span.size == 1));
    }
    for span in &spans {
        rules[span.value] =
            if span.disct > 0 || all_sharp || span.inf > 0 || (span.size == 1 && fvar_sharp) {
                Rule::Corner
            } else {
                span.ends.map_or(Rule::Corner, Rule::Crease)
            };
    }
}

/// Face-varying topology and rules of one level, from the channel and the level's mesh.
fn classify(
    channel: &Channel,
    polys: &[Vec<usize>],
    creases: &BTreeMap<(usize, usize), f64>,
    vertex_count: usize,
    options: UvOptions,
) -> Topology {
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_ends: Vec<[usize; 2]> = Vec::new();
    let mut edge_sides: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut side_edge: Vec<Vec<usize>> = Vec::with_capacity(polys.len());
    let mut position_sides: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    let mut corners_at: Vec<Vec<(usize, usize)>> = vec![Vec::new(); vertex_count];
    for (face, poly) in polys.iter().enumerate() {
        let size = poly.len();
        let mut sides = Vec::with_capacity(size);
        for corner in 0..size {
            let next = (corner + 1) % size;
            let key = edge_key(channel.corners[face][corner], channel.corners[face][next]);
            let edge = if let Some(&edge) = edge_index.get(&key) {
                edge
            } else {
                edge_index.insert(key, edge_ends.len());
                edge_ends.push([key.0, key.1]);
                edge_sides.push(Vec::new());
                edge_ends.len() - 1
            };
            edge_sides[edge].push((face, corner));
            sides.push(edge);
            position_sides
                .entry(edge_key(poly[corner], poly[next]))
                .or_default()
                .push((face, corner));
            corners_at[poly[corner]].push((face, corner));
        }
        side_edge.push(sides);
    }

    let (rules, edge_smooth) = {
        let walk = Walk {
            polys,
            values: &channel.corners,
            side_edge: &side_edge,
            position_sides: &position_sides,
            creases,
        };
        let mut rules = vec![Rule::Corner; channel.uvs.len()];
        if options.linear != FvarLinear::All {
            for (vertex, corners) in corners_at.iter().enumerate() {
                if let Some(fan) = walk.fan(vertex, corners) {
                    vertex_rules(&walk, &fan, options, &mut rules);
                }
            }
        }
        let edge_smooth = edge_sides
            .iter()
            .map(|sides| {
                options.linear != FvarLinear::All
                    && sides.len() == 2
                    && walk.side_faces(sides[0]) == 2
                    && !walk.sharp(sides[0])
            })
            .collect();
        (rules, edge_smooth)
    };
    Topology {
        rules,
        edge_ends,
        edge_sides,
        side_edge,
        edge_smooth,
    }
}

/// One refinement level of a channel: vertex points, face-varying edge points, and face points, in
/// the child order of `subdivide_once` (each parent corner becomes one quad).
fn refine_channel(channel: &Channel, topology: &Topology) -> Channel {
    let uvs = &channel.uvs;
    let value_count = uvs.len();
    let face_points = face_points(channel);
    let sums = sums(channel, &face_points);
    let mut child_uvs: Vec<DVec2> = (0..value_count)
        .map(|value| {
            let uv = uvs[value];
            match topology.rules[value] {
                Rule::Corner => uv,
                Rule::Crease([first, second]) => uv * 0.75 + (uvs[first] + uvs[second]) * 0.125,
                Rule::Smooth => {
                    let sum = sums[value];
                    if sum.count == 0 {
                        uv
                    } else {
                        let count = usize_as_f64(sum.count);
                        let edge_midpoints = (uv * count + sum.neighbours * 0.5) / (2.0 * count);
                        (sum.face / count + edge_midpoints * 2.0 + uv * (count - 3.0)) / count
                    }
                }
            }
        })
        .collect();
    for (edge, [first, second]) in topology.edge_ends.iter().enumerate() {
        child_uvs.push(if topology.edge_smooth[edge] {
            let sides = &topology.edge_sides[edge];
            (uvs[*first] + uvs[*second] + face_points[sides[0].0] + face_points[sides[1].0]) * 0.25
        } else {
            (uvs[*first] + uvs[*second]) * 0.5
        });
    }
    let face_base = value_count + topology.edge_ends.len();
    child_uvs.extend(face_points);

    let mut origin = channel.origin.clone();
    origin.resize(child_uvs.len(), None);
    let mut corners = Vec::with_capacity(channel.corners.iter().map(Vec::len).sum());
    for (face, values) in channel.corners.iter().enumerate() {
        let size = values.len();
        for (corner, &value) in values.iter().enumerate() {
            corners.push(vec![
                value,
                value_count + topology.side_edge[face][corner],
                face_base + face,
                value_count + topology.side_edge[face][(corner + size - 1) % size],
            ]);
        }
    }
    Channel {
        corners,
        uvs: child_uvs,
        origin,
    }
}

impl Layer {
    /// Limit values of the final level. A single-face crease value follows the boundary chain of its
    /// level-0 value, iterated `quality` times.
    fn limit(&self, topology: &Topology, sums: &[Sums], quality: u64) -> Vec<DVec2> {
        let uvs = &self.current.uvs;
        (0..uvs.len())
            .map(|value| {
                let sum = sums[value];
                match topology.rules[value] {
                    Rule::Corner => uvs[value],
                    Rule::Crease([first, second]) if sum.count == 1 => {
                        self.corner_limit(value, [first, second], quality)
                    }
                    Rule::Crease([first, second]) => {
                        (uvs[value] * 4.0 + uvs[first] + uvs[second]) / 6.0
                    }
                    Rule::Smooth if sum.count == 0 => uvs[value],
                    Rule::Smooth => {
                        let count = usize_as_f64(sum.count);
                        (uvs[value] * (count * count) + sum.neighbours * 2.0 + sum.opposite)
                            / (count * (count + 5.0))
                    }
                }
            })
            .collect()
    }

    fn corner_limit(&self, value: usize, ends: [usize; 2], quality: u64) -> DVec2 {
        let uvs = &self.current.uvs;
        let regular = (uvs[value] * 4.0 + uvs[ends[0]] + uvs[ends[1]]) / 6.0;
        let Some(base) = self.current.origin[value] else {
            return regular;
        };
        let Some(base_ends) = self.base_ends[base] else {
            return regular;
        };
        let (mut point, mut first, mut second) = (
            self.base.uvs[base],
            self.base.uvs[base_ends[0]],
            self.base.uvs[base_ends[1]],
        );
        for _ in 0..quality {
            (point, first, second) = (
                point * 0.75 + (first + second) * 0.125,
                (point + first) * 0.5,
                (point + second) * 0.5,
            );
        }
        point
    }
}

/// Polygon vertex indices of `mesh`, in face and corner order.
fn polygon_vertices(mesh: &Mesh) -> Result<Vec<Vec<usize>>> {
    Ok(polygons(mesh)?
        .into_iter()
        .map(|(vertices, _)| vertices)
        .collect())
}

fn uv_of(value: &Value) -> DVec2 {
    value
        .as_array()
        .and_then(|pair| Some(DVec2::new(pair.first()?.as_f64()?, pair.get(1)?.as_f64()?)))
        .unwrap_or(DVec2::ZERO)
}

/// Blender's UV connection tolerance (`STD_UV_CONNECT_LIMIT`).
const UV_CONNECT_LIMIT: f64 = 0.0001;

/// Whether a face's UVs wind negatively, by the trapezium rule Blender's `cross_poly_v2` uses.
fn uv_winding(uvs: &[DVec2]) -> bool {
    let Some(&last) = uvs.last() else {
        return false;
    };
    let mut previous = last;
    let mut cross = 0.0;
    for uv in uvs {
        cross += (previous.x - uv.x) * (uv.y + previous.y);
        previous = *uv;
    }
    cross < 0.0
}

/// Channel whose values are Blender's UV vertices (`calc_uv_verts` with winding): per position
/// vertex, corners whose UVs connect and whose UV windings agree share one value, grouped greedily in
/// corner order. A value keeps the UV of its first corner.
fn channel_from(vertex_count: usize, polys: &[Vec<usize>], uvs: &[Vec<DVec2>]) -> Channel {
    let winding: Vec<bool> = uvs.iter().map(|face| uv_winding(face)).collect();
    let mut corners_at: Vec<Vec<(usize, usize)>> = vec![Vec::new(); vertex_count];
    for (face, poly) in polys.iter().enumerate() {
        for (corner, vertex) in poly.iter().enumerate() {
            corners_at[*vertex].push((face, corner));
        }
    }
    let mut channel = Channel {
        corners: polys
            .iter()
            .map(|poly| vec![usize::MAX; poly.len()])
            .collect(),
        uvs: Vec::new(),
        origin: Vec::new(),
    };
    for corners in &corners_at {
        for (index, &(face, corner)) in corners.iter().enumerate() {
            if channel.corners[face][corner] != usize::MAX {
                continue;
            }
            let value = channel.uvs.len();
            let uv = uvs[face][corner];
            channel.uvs.push(uv);
            channel.origin.push(Some(value));
            channel.corners[face][corner] = value;
            for &(other_face, other_corner) in &corners[index + 1..] {
                let other = uvs[other_face][other_corner];
                if channel.corners[other_face][other_corner] == usize::MAX
                    && (other.x - uv.x).abs() < UV_CONNECT_LIMIT
                    && (other.y - uv.y).abs() < UV_CONNECT_LIMIT
                    && winding[other_face] == winding[face]
                {
                    channel.corners[other_face][other_corner] = value;
                }
            }
        }
    }
    channel
}

/// Every UV layer of `mesh` as `(where it is written, per-face corner UVs)`: `blender_uv_layers` by
/// index, then `uv_map` layers by name. Missing corner UVs are zero.
fn layer_sources(mesh: &Mesh, polys: &[Vec<usize>]) -> Vec<(Target, Channel)> {
    let mut sources = Vec::new();
    if let Some(layers) = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
    {
        for (index, layer) in layers.iter().enumerate() {
            let values = layer.get("values").and_then(Value::as_array);
            let mut loop_index = 0;
            let per_face: Vec<Vec<DVec2>> = polys
                .iter()
                .map(|poly| {
                    let row = (0..poly.len())
                        .map(|corner| {
                            values
                                .and_then(|values| values.get(loop_index + corner))
                                .map_or(DVec2::ZERO, uv_of)
                        })
                        .collect();
                    loop_index += poly.len();
                    row
                })
                .collect();
            sources.push((
                Target::Blender(index),
                channel_from(mesh.vertices.len(), polys, &per_face),
            ));
        }
    }
    if let Some(entries) = mesh.attributes.get("uv_map").and_then(Value::as_array) {
        let face_index: HashMap<u64, usize> = mesh
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| (u64::from(face.id), index))
            .collect();
        let mut by_layer: BTreeMap<String, Vec<Vec<DVec2>>> = BTreeMap::new();
        for entry in entries {
            let Some(face) = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .and_then(|id| face_index.get(&id))
            else {
                continue;
            };
            let layer = uv_map_entry_layer(entry);
            let uv: Vec<DVec2> = entry
                .get("uv")
                .and_then(Value::as_array)
                .map(|row| row.iter().map(uv_of).collect())
                .unwrap_or_default();
            by_layer
                .entry(layer.to_owned())
                .or_insert_with(|| vec![Vec::new(); polys.len()])[*face] = uv;
        }
        for (name, faces) in by_layer {
            let per_face: Vec<Vec<DVec2>> = polys
                .iter()
                .zip(&faces)
                .map(|(poly, uv)| {
                    (0..poly.len())
                        .map(|corner| uv.get(corner).copied().unwrap_or(DVec2::ZERO))
                        .collect()
                })
                .collect();
            sources.push((
                Target::UvMap(name),
                channel_from(mesh.vertices.len(), polys, &per_face),
            ));
        }
    }
    sources
}

impl UvSubdivision {
    /// Face-varying channels of every UV layer of `mesh` (the level-0 state).
    pub(super) fn new(mesh: &Mesh, options: UvOptions) -> Result<Self> {
        let polys = polygon_vertices(mesh)?;
        let sources = layer_sources(mesh, &polys);
        if sources.is_empty() {
            return Ok(Self {
                options,
                layers: Vec::new(),
            });
        }
        let creases = edge_crease_weights(mesh, options.use_creases)?;
        let layers = sources
            .into_iter()
            .map(|(target, base)| {
                let topology = classify(&base, &polys, &creases, mesh.vertices.len(), options);
                let base_ends = topology
                    .rules
                    .iter()
                    .map(|rule| match rule {
                        Rule::Crease(ends) => Some(*ends),
                        _ => None,
                    })
                    .collect();
                Layer {
                    target,
                    current: base.clone(),
                    base,
                    base_ends,
                }
            })
            .collect();
        Ok(Self { options, layers })
    }

    /// Refines every channel from `parent`, the mesh the channels currently describe.
    pub(super) fn refine(&mut self, parent: &Mesh) -> Result<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        let polys = polygon_vertices(parent)?;
        let creases = edge_crease_weights(parent, self.options.use_creases)?;
        for layer in &mut self.layers {
            let topology = classify(
                &layer.current,
                &polys,
                &creases,
                parent.vertices.len(),
                self.options,
            );
            layer.current = refine_channel(&layer.current, &topology);
        }
        Ok(())
    }

    /// Writes the channels to the final mesh's UV attributes. `quality` gives the limit surface
    /// (`use_limit_surface`), or `None` for the refined values.
    pub(super) fn write_back(&self, result: &mut Mesh, quality: Option<u64>) -> Result<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        let polys = polygon_vertices(result)?;
        let creases = match quality {
            Some(_) => edge_crease_weights(result, self.options.use_creases)?,
            None => BTreeMap::new(),
        };
        let face_index: HashMap<u64, usize> = result
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| (u64::from(face.id), index))
            .collect();
        for layer in &self.layers {
            if layer.current.corners.len() != polys.len() {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "subdivision UV channel does not match the output faces",
                ));
            }
            let uvs = match quality {
                Some(quality) => {
                    let topology = classify(
                        &layer.current,
                        &polys,
                        &creases,
                        result.vertices.len(),
                        self.options,
                    );
                    let sums = sums(&layer.current, &face_points(&layer.current));
                    layer.limit(&topology, &sums, quality)
                }
                None => layer.current.uvs.clone(),
            };
            let per_face: Vec<Vec<Value>> = layer
                .current
                .corners
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .map(|value| json!([uvs[*value].x, uvs[*value].y]))
                        .collect()
                })
                .collect();
            match &layer.target {
                Target::Blender(index) => {
                    let flat: Vec<Value> = per_face.into_iter().flatten().collect();
                    if let Some(object) = result
                        .attributes
                        .get_mut("blender_uv_layers")
                        .and_then(Value::as_array_mut)
                        .and_then(|layers| layers.get_mut(*index))
                        .and_then(Value::as_object_mut)
                    {
                        object.insert("values".to_owned(), Value::Array(flat));
                    }
                }
                Target::UvMap(name) => {
                    let Some(entries) = result
                        .attributes
                        .get_mut("uv_map")
                        .and_then(Value::as_array_mut)
                    else {
                        continue;
                    };
                    for entry in entries.iter_mut() {
                        if uv_map_entry_layer(entry) != name.as_str() {
                            continue;
                        }
                        let Some(face) = entry
                            .get("face_id")
                            .and_then(Value::as_u64)
                            .and_then(|id| face_index.get(&id))
                        else {
                            continue;
                        };
                        if let Some(object) = entry.as_object_mut() {
                            object.insert("uv".to_owned(), Value::Array(per_face[*face].clone()));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// One level of bilinear (`SIMPLE`) subdivision: original points stay, each edge gets its midpoint,
/// and each face its centroid. Every corner becomes one quad, as in `subdivide_once`.
pub(super) fn subdivide_simple_once(mesh: &Mesh, use_creases: bool) -> Result<Mesh> {
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_edges = loose_edges(mesh)?;
    let mut edges = BTreeSet::new();
    for (face, _) in &source_polygons {
        for (first, second) in super::cyclic_pairs(face) {
            edges.insert(edge_key(*first, *second));
        }
    }
    for [first, second] in &source_edges {
        edges.insert(edge_key(*first, *second));
    }
    let edge_creases = edge_crease_weights(mesh, use_creases)?;
    let face_points: Vec<DVec3> = source_polygons
        .iter()
        .map(|(face, _)| {
            face.iter()
                .fold(DVec3::ZERO, |sum, index| sum + source_points[*index])
                / usize_as_f64(face.len())
        })
        .collect();
    let mut points = source_points.clone();
    let mut edge_point_indices = BTreeMap::new();
    for edge in &edges {
        edge_point_indices.insert(*edge, points.len());
        points.push((source_points[edge.0] + source_points[edge.1]) * 0.5);
    }
    points.extend(face_points);
    subdivided_mesh(
        mesh,
        points,
        &source_polygons,
        &source_edges,
        &edge_point_indices,
        &edge_creases,
        use_creases,
    )
}

/// One subdivision level's mesh from refined points, shared by SIMPLE and Catmull-Clark. `points` holds
/// the refined vertices, then one point per edge in `edge_points` key order, then one per source face.
/// Every source corner becomes one quad `[corner, next edge point, face point, previous edge point]`,
/// faces in source order. Loose edges split at their edge point, and each output vertex records its
/// source origin. With `use_creases`, child edges keep the crease minus one level.
pub(super) fn subdivided_mesh(
    mesh: &Mesh,
    points: Vec<DVec3>,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    edge_points: &BTreeMap<(usize, usize), usize>,
    edge_creases: &BTreeMap<(usize, usize), f64>,
    use_creases: bool,
) -> Result<Mesh> {
    let face_point_base = points.len() - source_polygons.len();
    let mut output_polygons =
        Vec::with_capacity(source_polygons.iter().map(|(face, _)| face.len()).sum());
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        for corner in 0..face.len() {
            let previous = face[(corner + face.len() - 1) % face.len()];
            let current = face[corner];
            let next = face[(corner + 1) % face.len()];
            output_polygons.push((
                vec![
                    current,
                    edge_points[&edge_key(current, next)],
                    face_point_base + face_index,
                    edge_points[&edge_key(previous, current)],
                ],
                *material,
            ));
        }
    }
    let mut output_loose_edges = Vec::with_capacity(source_edges.len() * 2);
    for &[first, second] in source_edges {
        let midpoint = edge_points[&edge_key(first, second)];
        output_loose_edges.push([first, midpoint]);
        output_loose_edges.push([midpoint, second]);
    }

    let mut vertex_origins: Vec<Vec<(usize, f64)>> = (0..mesh.vertices.len())
        .map(|vertex| vec![(vertex, 1.0)])
        .collect();
    vertex_origins.extend(
        edge_points
            .keys()
            .map(|edge| vec![(edge.0, 0.5), (edge.1, 0.5)]),
    );
    vertex_origins.extend(source_polygons.iter().map(|(face, _)| {
        let share = 1.0 / usize_as_f64(face.len());
        face.iter().map(|&vertex| (vertex, share)).collect()
    }));
    let origin = Origin {
        vertices: vertex_origins,
        faces: source_polygons
            .iter()
            .enumerate()
            .flat_map(|(index, (face, _))| std::iter::repeat_n(Some(index), face.len()))
            .collect(),
        corners: None,
    };
    let mut output = build_mesh(points, output_polygons, output_loose_edges, &origin, mesh)?;
    if use_creases && mesh.attributes.contains_key("crease_edge") {
        let mut next_creases = BTreeMap::new();
        for (edge, sharpness) in edge_creases {
            let remaining = sharpness - 1.0;
            if remaining <= 0.0 {
                continue;
            }
            let midpoint = edge_points[edge];
            let first_id = output.vertices[edge.0].id;
            let second_id = output.vertices[edge.1].id;
            let midpoint_id = output.vertices[midpoint].id;
            let ordered = |first: u32, second: u32| {
                if first <= second {
                    (first, second)
                } else {
                    (second, first)
                }
            };
            next_creases.insert(ordered(first_id, midpoint_id), remaining);
            next_creases.insert(ordered(midpoint_id, second_id), remaining);
        }
        let mut values = Map::new();
        for edge in &output.edges {
            let ordered = if edge.vertices[0] <= edge.vertices[1] {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            if let Some(weight) = next_creases.get(&ordered) {
                values.insert(format!("e{}", edge.id), json!(weight));
            }
        }
        output.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain":"edge","type":"float","values":values}),
        );
    }
    Ok(output)
}
