// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_converter_mesh.cc, source/blender/modifiers/intern/MOD_subsurf.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.
//   OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/fvarLevel.cpp, far/primvarRefiner.h, Apache-2.0.

//! Face-varying UV and loop-normal refinement, and bilinear (`SIMPLE`) subdivision.
//!
//! Blender refines each UV layer as an `OpenSubdiv` face-varying channel. A channel value is a position
//! vertex with one UV, so a vertex splits wherever its UVs differ between faces. Each level derives the
//! child channel from the parent topology with the face-varying Catmull-Clark rules, where `uv_smooth`
//! picks the linear option and `boundary_smooth` the boundary-corner option. SIMPLE subdivision uses
//! bilinear positions, and its UVs are linear too (Blender makes every edge and vertex sharp).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::{DVec2, DVec3, Vec3};
use serde_json::{Map, json};

use super::{
    NonManifold, Polygon, build_mesh, edge_crease_weights, loose_edges, polygons, positions,
    usize_as_f64,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        clnor::{self, NormalCodeOptions, SetterInput},
        edge_key,
        provenance::Origin,
        uv_layers::{self, UvLayer},
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

/// One UV layer as a face-varying channel. Corners are global: face `f` owns
/// `corners[starts[f]..starts[f + 1]]`, and `corners[g]` names the value at corner `g`. The layout
/// matches the position topology of the mesh the channel describes.
struct Channel {
    starts: Vec<usize>,
    corners: Vec<usize>,
    uvs: Vec<DVec2>,
    /// The level-0 value each value descends from through vertex points.
    origin: Vec<Option<usize>>,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            starts: vec![0],
            corners: Vec::new(),
            uvs: Vec::new(),
            origin: Vec::new(),
        }
    }
}

impl Channel {
    fn face_count(&self) -> usize {
        self.starts.len() - 1
    }

    fn face(&self, face: usize) -> &[usize] {
        &self.corners[self.starts[face]..self.starts[face + 1]]
    }
}

struct Layer {
    name: String,
    /// Per-face corner pins, refined independently of the face-varying UV values.
    pins: Vec<Option<Vec<bool>>>,
    /// Level-0 value UVs, read by `corner_limit` after the channel has been refined.
    base_uvs: Vec<DVec2>,
    /// Per level-0 value, the two values at the ends of its crease (if any).
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

/// Per-layer face-varying topology of one level.
struct Topology {
    rules: Vec<Rule>,
    /// The two value indices of each face-varying edge.
    edge_ends: Vec<[usize; 2]>,
    /// Face-varying edge of each corner's side, towards its next corner.
    side_edge: Vec<usize>,
    /// For each face-varying edge whose edge point is the smooth average (continuous, manifold,
    /// sharp-free), the faces on its two sides.
    edge_smooth: Vec<Option<[usize; 2]>>,
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

/// A position vertex's corners in fan order. `corners[i]` and `corners[i + 1]` share an edge; a closed
/// fan wraps around, an open one starts and ends on a boundary.
struct Fan {
    corners: Vec<usize>,
    closed: bool,
}

/// The edge between consecutive fan corners (`index - 1` and `index`); boundary links sit at the ends
/// of an open fan.
#[derive(Clone, Copy)]
struct Link {
    mismatch: bool,
    sharp: bool,
}

/// Position topology of one level, built once and shared by every UV layer. Corners are global
/// indices, as in [`Channel`].
struct Position {
    /// First corner of each face; `starts[f + 1]` ends face `f`.
    starts: Vec<usize>,
    /// Face owning each corner.
    face_of: Vec<usize>,
    /// Vertex at each corner.
    corner_vertex: Vec<usize>,
    /// Position edge of each corner's side, towards its next corner.
    corner_edge: Vec<usize>,
    /// Sides of position edge `e` are `edge_sides[edge_starts[e]..edge_starts[e + 1]]`, in corner order.
    edge_starts: Vec<usize>,
    edge_sides: Vec<usize>,
    /// Whether each position edge is sharp (crease weight at least 1).
    edge_sharp: Vec<bool>,
    /// Corners at vertex `v` are `vertex_corners[vertex_starts[v]..vertex_starts[v + 1]]`, in corner order.
    vertex_starts: Vec<usize>,
    vertex_corners: Vec<usize>,
    /// Fan of each vertex (`None` when it is not one manifold fan). Empty unless built with fans.
    fans: Vec<Option<Fan>>,
}

/// Groups the indices `0..keys.len()` by `keys[index]` into `buckets` buckets: bucket `b` holds
/// `members[starts[b]..starts[b + 1]]`, in ascending index order.
fn group(keys: &[usize], buckets: usize) -> (Vec<usize>, Vec<usize>) {
    let mut starts = vec![0; buckets + 1];
    for &key in keys {
        starts[key + 1] += 1;
    }
    let mut running = 0;
    for start in &mut starts {
        running += *start;
        *start = running;
    }
    let mut cursor = starts.clone();
    let mut members = vec![0; keys.len()];
    for (index, &key) in keys.iter().enumerate() {
        members[cursor[key]] = index;
        cursor[key] += 1;
    }
    (starts, members)
}

impl Position {
    /// Position topology of a mesh's polygons. `with_fans` computes every vertex's fan, which only
    /// the non-linear rules read.
    fn new(
        polys: &[Vec<usize>],
        creases: &BTreeMap<(usize, usize), f64>,
        vertex_count: usize,
        with_fans: bool,
    ) -> Self {
        // OpenSubdiv makes non-manifold edges infinitely sharp in the face-varying topology too.
        let creases =
            NonManifold::new(vertex_count, polys.iter().map(Vec::as_slice)).sharpened(creases);
        let total: usize = polys.iter().map(Vec::len).sum();
        let mut starts = Vec::with_capacity(polys.len() + 1);
        let mut face_of = Vec::with_capacity(total);
        let mut corner_vertex = Vec::with_capacity(total);
        starts.push(0);
        for (face, poly) in polys.iter().enumerate() {
            corner_vertex.extend_from_slice(poly);
            face_of.extend(std::iter::repeat_n(face, poly.len()));
            starts.push(corner_vertex.len());
        }

        let mut edge_index: HashMap<(usize, usize), usize> = HashMap::with_capacity(total / 2 + 1);
        let mut corner_edge = Vec::with_capacity(total);
        let mut edge_sharp = Vec::new();
        for poly in polys {
            let size = poly.len();
            for corner in 0..size {
                let key = edge_key(poly[corner], poly[(corner + 1) % size]);
                let edge = *edge_index.entry(key).or_insert_with(|| {
                    edge_sharp.push(creases.get(&key).is_some_and(|sharpness| *sharpness >= 1.0));
                    edge_sharp.len() - 1
                });
                corner_edge.push(edge);
            }
        }
        let (edge_starts, edge_sides) = group(&corner_edge, edge_sharp.len());
        let (vertex_starts, vertex_corners) = group(&corner_vertex, vertex_count);

        let mut position = Self {
            starts,
            face_of,
            corner_vertex,
            corner_edge,
            edge_starts,
            edge_sides,
            edge_sharp,
            vertex_starts,
            vertex_corners,
            fans: Vec::new(),
        };
        if with_fans {
            position.fans = (0..vertex_count)
                .map(|vertex| position.fan(vertex))
                .collect();
        }
        position
    }

    fn corner_count(&self) -> usize {
        self.corner_vertex.len()
    }

    fn next(&self, corner: usize) -> usize {
        let face = self.face_of[corner];
        if corner + 1 == self.starts[face + 1] {
            self.starts[face]
        } else {
            corner + 1
        }
    }

    fn prev(&self, corner: usize) -> usize {
        let face = self.face_of[corner];
        if corner == self.starts[face] {
            self.starts[face + 1] - 1
        } else {
            corner - 1
        }
    }

    fn edge_sides_of(&self, edge: usize) -> &[usize] {
        &self.edge_sides[self.edge_starts[edge]..self.edge_starts[edge + 1]]
    }

    fn corners_at(&self, vertex: usize) -> &[usize] {
        &self.vertex_corners[self.vertex_starts[vertex]..self.vertex_starts[vertex + 1]]
    }

    /// Faces sharing the position edge that leaves `corner` towards its next corner.
    fn side_faces(&self, corner: usize) -> usize {
        self.edge_sides_of(self.corner_edge[corner]).len()
    }

    /// Whether the position edge that leaves `corner` towards its next corner is sharp.
    fn sharp(&self, corner: usize) -> bool {
        self.edge_sharp[self.corner_edge[corner]]
    }

    /// The corner at `vertex` on the other face of the edge leaving `corner` forward or backward.
    /// `None` on a boundary or non-manifold edge.
    fn across(&self, corner: usize, vertex: usize, forward: bool) -> Option<usize> {
        let side = if forward { corner } else { self.prev(corner) };
        let sides = self.edge_sides_of(self.corner_edge[side]);
        if sides.len() != 2 {
            return None;
        }
        let other = *sides.iter().find(|candidate| **candidate != side)?;
        Some(if self.corner_vertex[other] == vertex {
            other
        } else {
            self.next(other)
        })
    }

    /// The fan around `vertex`; `None` when its corners do not form one manifold fan.
    fn fan(&self, vertex: usize) -> Option<Fan> {
        let corners = self.corners_at(vertex);
        for &corner in corners {
            if self.side_faces(corner) > 2 || self.side_faces(self.prev(corner)) > 2 {
                return None;
            }
        }
        let mut seen: Vec<usize> = Vec::new();
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

/// Per-layer walk: channel values over the shared position topology.
struct Walk<'a> {
    position: &'a Position,
    channel: &'a Channel,
    side_edge: &'a [usize],
}

impl Walk<'_> {
    fn value(&self, corner: usize) -> usize {
        self.channel.corners[corner]
    }

    fn prev_value(&self, corner: usize) -> usize {
        self.value(self.position.prev(corner))
    }

    fn next_value(&self, corner: usize) -> usize {
        self.value(self.position.next(corner))
    }
}

/// Values of every face of a channel, used to group the channel's values into faces.
fn face_points(channel: &Channel) -> Vec<DVec2> {
    (0..channel.face_count())
        .map(|face| {
            let values = channel.face(face);
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
    for (face, &face_point) in face_points.iter().enumerate().take(channel.face_count()) {
        let values = channel.face(face);
        let size = values.len();
        for corner in 0..size {
            let sum = &mut sums[values[corner]];
            sum.count += 1;
            sum.face += face_point;
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
    let after_back = walk.side_edge[walk.position.prev(after)];
    Link {
        mismatch: walk.side_edge[before] != after_back,
        sharp: walk.position.sharp(before),
    }
}

/// The value across link `index`'s end, on the face of the corner before it, for a crease.
fn link_end(walk: &Walk, fan: &Fan, index: usize) -> usize {
    let size = fan.corners.len();
    if !fan.closed && index == 0 {
        walk.prev_value(fan.corners[0])
    } else if !fan.closed && index == size {
        walk.next_value(fan.corners[size - 1])
    } else {
        walk.next_value(fan.corners[(index + size - 1) % size])
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
                    walk.prev_value(fan.corners[index]),
                    walk.next_value(fan.corners[(index + size - 1) % size]),
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
                walk.prev_value(fan.corners[corner_of(first)]),
                walk.next_value(fan.corners[corner_of(last)]),
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

/// Face-varying topology and rules of one level, from one layer's channel and the shared position
/// topology. `position` must have fans exactly when `options.linear` is not `All`.
fn classify(channel: &Channel, position: &Position, options: UvOptions) -> Topology {
    let mut edge_index: HashMap<(usize, usize), usize> =
        HashMap::with_capacity(channel.corners.len() / 2 + 1);
    let mut edge_ends: Vec<[usize; 2]> = Vec::new();
    // Per face-varying edge: its side count, and its first two sides (corners).
    let mut side_counts: Vec<usize> = Vec::new();
    let mut first_sides: Vec<[usize; 2]> = Vec::new();
    let mut side_edge = Vec::with_capacity(channel.corners.len());
    for face in 0..channel.face_count() {
        let (start, end) = (channel.starts[face], channel.starts[face + 1]);
        for corner in start..end {
            let next = if corner + 1 == end { start } else { corner + 1 };
            let key = edge_key(channel.corners[corner], channel.corners[next]);
            let edge = *edge_index.entry(key).or_insert_with(|| {
                edge_ends.push([key.0, key.1]);
                side_counts.push(0);
                first_sides.push([usize::MAX; 2]);
                edge_ends.len() - 1
            });
            if side_counts[edge] < 2 {
                first_sides[edge][side_counts[edge]] = corner;
            }
            side_counts[edge] += 1;
            side_edge.push(edge);
        }
    }

    let mut rules = vec![Rule::Corner; channel.uvs.len()];
    if options.linear != FvarLinear::All {
        let walk = Walk {
            position,
            channel,
            side_edge: &side_edge,
        };
        for fan in position.fans.iter().flatten() {
            vertex_rules(&walk, fan, options, &mut rules);
        }
    }
    let edge_smooth = (0..edge_ends.len())
        .map(|edge| {
            let [first, second] = first_sides[edge];
            (options.linear != FvarLinear::All
                && side_counts[edge] == 2
                && position.side_faces(first) == 2
                && !position.sharp(first))
            .then(|| [position.face_of[first], position.face_of[second]])
        })
        .collect();
    Topology {
        rules,
        edge_ends,
        side_edge,
        edge_smooth,
    }
}

/// One refinement level of a channel: vertex points, face-varying edge points, and face points, in
/// the child order of `subdivide_once` (each parent corner becomes one quad).
fn refine_channel(channel: Channel, topology: &Topology) -> Channel {
    let value_count = channel.uvs.len();
    let face_centers = face_points(&channel);
    let sums = sums(&channel, &face_centers);
    let uvs = &channel.uvs;
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
        child_uvs.push(match topology.edge_smooth[edge] {
            Some([first_face, second_face]) => {
                (uvs[*first] + uvs[*second] + face_centers[first_face] + face_centers[second_face])
                    * 0.25
            }
            None => (uvs[*first] + uvs[*second]) * 0.5,
        });
    }
    let face_base = value_count + topology.edge_ends.len();
    child_uvs.extend(face_centers);

    let Channel {
        starts,
        corners,
        mut origin,
        ..
    } = channel;
    origin.resize(child_uvs.len(), None);
    let mut child_corners = Vec::with_capacity(corners.len() * 4);
    for face in 0..starts.len() - 1 {
        let (start, end) = (starts[face], starts[face + 1]);
        for (offset, &value) in corners[start..end].iter().enumerate() {
            let corner = start + offset;
            let previous = if corner == start { end - 1 } else { corner - 1 };
            child_corners.extend([
                value,
                value_count + topology.side_edge[corner],
                face_base + face,
                value_count + topology.side_edge[previous],
            ]);
        }
    }
    Channel {
        starts: (0..=corners.len()).map(|corner| 4 * corner).collect(),
        corners: child_corners,
        uvs: child_uvs,
        origin,
    }
}

/// Refines loop pins like Blender's boolean corner interpolation: any pinned source with a
/// positive contribution pins the child corner.
fn refine_pins(pins: Vec<Option<Vec<bool>>>, starts: &[usize]) -> Vec<Option<Vec<bool>>> {
    let mut child = Vec::with_capacity(starts.last().copied().unwrap_or_default());
    for (face, row) in pins.into_iter().enumerate() {
        let (start, end) = (starts[face], starts[face + 1]);
        let Some(row) = row else {
            child.extend(std::iter::repeat_n(None, end - start));
            continue;
        };
        let face_pin = row.iter().any(|pinned| *pinned);
        for corner in 0..end - start {
            let next = (corner + 1) % (end - start);
            let previous = if corner == 0 {
                end - start - 1
            } else {
                corner - 1
            };
            child.push(Some(vec![
                row[corner],
                row[corner] || row[next],
                face_pin,
                row[previous] || row[corner],
            ]));
        }
    }
    child
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
            self.base_uvs[base],
            self.base_uvs[base_ends[0]],
            self.base_uvs[base_ends[1]],
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
fn channel_from(position: &Position, corner_uv: &[DVec2]) -> Channel {
    let winding: Vec<bool> = position
        .starts
        .windows(2)
        .map(|range| uv_winding(&corner_uv[range[0]..range[1]]))
        .collect();
    let mut corners = vec![usize::MAX; corner_uv.len()];
    let mut uvs = Vec::new();
    let mut origin = Vec::new();
    for vertex in 0..position.vertex_starts.len() - 1 {
        let at_vertex = position.corners_at(vertex);
        for (index, &corner) in at_vertex.iter().enumerate() {
            if corners[corner] != usize::MAX {
                continue;
            }
            let value = uvs.len();
            let uv = corner_uv[corner];
            uvs.push(uv);
            origin.push(Some(value));
            corners[corner] = value;
            for &other in &at_vertex[index + 1..] {
                let other_uv = corner_uv[other];
                if corners[other] == usize::MAX
                    && (other_uv.x - uv.x).abs() < UV_CONNECT_LIMIT
                    && (other_uv.y - uv.y).abs() < UV_CONNECT_LIMIT
                    && winding[position.face_of[other]] == winding[position.face_of[corner]]
                {
                    corners[other] = value;
                }
            }
        }
    }
    Channel {
        starts: position.starts.clone(),
        corners,
        uvs,
        origin,
    }
}

/// One layer's corner UVs in global corner order, zero where the layer has no value.
fn corner_uvs(position: &Position, layer: &UvLayer) -> Vec<DVec2> {
    let mut uvs = vec![DVec2::ZERO; position.corner_count()];
    for (face, values) in layer.faces.iter().enumerate() {
        if let Some(values) = values {
            let start = position.starts[face];
            for (corner, uv) in values.iter().enumerate() {
                uvs[start + corner] = DVec2::new(uv[0], uv[1]);
            }
        }
    }
    uvs
}

/// The pins of layer `name` that the rebuild carried onto `faces`. A face keeps its pins when they
/// cover its corners; otherwise it has none.
pub(super) fn carried_pins(
    carried: &[UvLayer],
    name: &str,
    faces: &[Option<Vec<[f64; 2]>>],
) -> Vec<Option<Vec<bool>>> {
    let known = carried.iter().find(|layer| layer.name == name);
    faces
        .iter()
        .enumerate()
        .map(|(face, uv)| {
            let corners = uv.as_ref()?.len();
            known?
                .pinned
                .get(face)?
                .clone()
                .filter(|pins| pins.len() == corners)
        })
        .collect()
}

impl UvSubdivision {
    /// Face-varying channels of every UV layer of `mesh` (the level-0 state).
    pub(super) fn new(mesh: &Mesh, options: UvOptions) -> Result<Self> {
        let layers = uv_layers::read_layers(mesh);
        if layers.is_empty() {
            return Ok(Self {
                options,
                layers: Vec::new(),
            });
        }
        let polys = polygon_vertices(mesh)?;
        let creases = edge_crease_weights(mesh, options.use_creases)?;
        let position = Position::new(
            &polys,
            &creases,
            mesh.vertices.len(),
            options.linear != FvarLinear::All,
        );
        let layers = layers
            .into_iter()
            .map(|layer| {
                let base = channel_from(&position, &corner_uvs(&position, &layer));
                let pins = layer.pinned;
                let topology = classify(&base, &position, options);
                let base_ends = topology
                    .rules
                    .iter()
                    .map(|rule| match rule {
                        Rule::Crease(ends) => Some(*ends),
                        _ => None,
                    })
                    .collect();
                Layer {
                    name: layer.name,
                    pins,
                    base_uvs: base.uvs.clone(),
                    base_ends,
                    current: base,
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
        let position = Position::new(
            &polys,
            &creases,
            parent.vertices.len(),
            self.options.linear != FvarLinear::All,
        );
        for layer in &mut self.layers {
            let topology = classify(&layer.current, &position, self.options);
            layer.pins = refine_pins(std::mem::take(&mut layer.pins), &layer.current.starts);
            layer.current = refine_channel(std::mem::take(&mut layer.current), &topology);
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
        let limit = match quality {
            Some(quality) => {
                let creases = edge_crease_weights(result, self.options.use_creases)?;
                let position = Position::new(
                    &polys,
                    &creases,
                    result.vertices.len(),
                    self.options.linear != FvarLinear::All,
                );
                Some((quality, position))
            }
            None => None,
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            if layer.current.face_count() != polys.len() || layer.pins.len() != polys.len() {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "subdivision UV channel does not match the output faces",
                ));
            }
            let limited;
            let uvs: &[DVec2] = match &limit {
                Some((quality, position)) => {
                    let topology = classify(&layer.current, position, self.options);
                    let sums = sums(&layer.current, &face_points(&layer.current));
                    limited = layer.limit(&topology, &sums, *quality);
                    &limited
                }
                None => &layer.current.uvs,
            };
            let faces: Vec<Option<Vec<[f64; 2]>>> = (0..layer.current.face_count())
                .map(|face| {
                    Some(
                        layer
                            .current
                            .face(face)
                            .iter()
                            .map(|value| [uvs[*value].x, uvs[*value].y])
                            .collect(),
                    )
                })
                .collect();
            let mut written = UvLayer::new(layer.name.clone(), faces);
            written.pinned.clone_from(&layer.pins);
            layers.push(written);
        }
        uv_layers::write_layers(result, &layers);
        Ok(())
    }
}

/// One sample of a Subsurf face on Blender's ptex grid: the input corner whose ptex face holds it, the
/// lower-left grid point of its square, and the grid slot of its first corner.
#[derive(Clone, Copy)]
struct NormalSample {
    corner: usize,
    x: usize,
    y: usize,
    rotation: usize,
}

/// Grid offset of each slot's corner within its square, counter-clockwise from the lower-left corner.
const SLOT_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (1, 1), (0, 1)];

/// Corner range of one input face.
struct NormalFace {
    start: usize,
    count: usize,
}

/// Custom-normal state of one Subsurf modifier, advanced once per level beside the UV channels.
///
/// Vectors follow Blender's loop-normal interpolation (`subdiv_mesh.cc`): a quad face is one ptex patch of
/// its raw corner vectors, an n-gon corner is the patch `[corner, edge to next, face mean, edge to
/// previous]` with each mixed value normalized, and each output loop is the normalized bilinear mix at its
/// grid point. Automatic corners carry their evaluated fan normal, so they take part whenever the input has
/// a custom layer. The loop numbering places each sample on Blender's grid (`subdiv_foreach.cc`): a quad is
/// one ptex face of side `2^level`, an n-gon corner one ptex face of side `2^(level-1)`.
pub(super) struct NormalSubdivision {
    /// Interpolation patch of each input corner in slot order; `None` without an enabled custom layer.
    patches: Option<Vec<[DVec3; 4]>>,
    faces: Vec<NormalFace>,
    /// Face of each input corner.
    corner_face: Vec<usize>,
    /// Blender loop position of each input corner within its face.
    corner_local: Vec<usize>,
    samples: Vec<NormalSample>,
    /// Corner grid side of the current level: 0 before the first refinement, then `2^(level - 1)`.
    grid: usize,
}

impl NormalSubdivision {
    /// The input's custom-normal patches, with automatic corners evaluated by the input fans, and its loop
    /// basis. Patches exist only when `use_custom_normals` is set and the input has a custom layer.
    pub(super) fn new(mesh: &Mesh, use_custom_normals: bool) -> Result<Self> {
        let corner_local = input_locals(mesh)?;
        let mut faces = Vec::with_capacity(mesh.faces.len());
        let mut corner_face = Vec::with_capacity(corner_local.len());
        for (index, face) in mesh.faces.iter().enumerate() {
            let count = face.vertices.len();
            faces.push(NormalFace {
                start: corner_face.len(),
                count,
            });
            corner_face.extend(std::iter::repeat_n(index, count));
        }
        let patches = match clnor::corner_codes(mesh)? {
            Some(codes) if use_custom_normals => {
                let vectors = clnor::evaluate_codes(
                    mesh,
                    &codes,
                    &positions(mesh),
                    NormalCodeOptions {
                        corner_order: None,
                        setter_input: SetterInput::NormalizeInput,
                    },
                )?;
                Some(corner_patches(&vectors, &faces, &corner_local))
            }
            _ => None,
        };
        Ok(Self {
            patches,
            faces,
            corner_face,
            corner_local,
            samples: Vec::new(),
            grid: 0,
        })
    }

    /// Advances the numbering one level. The first level has one sample per input corner: a quad corner is
    /// the child of its face's 2x2 grid at its slot, and an n-gon corner is its whole ptex face.
    pub(super) fn refine(&mut self) {
        self.samples = if self.grid == 0 {
            (0..self.corner_face.len())
                .map(|corner| self.root(corner))
                .collect()
        } else {
            self.samples
                .iter()
                .flat_map(|sample| {
                    (0..4).map(move |corner| {
                        let slot = (sample.rotation + corner) % 4;
                        let (dx, dy) = SLOT_OFFSETS[slot];
                        NormalSample {
                            corner: sample.corner,
                            x: sample.x * 2 + dx,
                            y: sample.y * 2 + dy,
                            rotation: slot,
                        }
                    })
                })
                .collect()
        };
        self.grid = if self.grid == 0 { 1 } else { self.grid * 2 };
    }

    /// First sample of an input corner.
    fn root(&self, corner: usize) -> NormalSample {
        let face = &self.faces[self.corner_face[corner]];
        if face.count == 4 {
            let slot = (self.corner_local[face.start] + corner - face.start) % 4;
            let (x, y) = SLOT_OFFSETS[slot];
            NormalSample {
                corner,
                x,
                y,
                rotation: slot,
            }
        } else {
            NormalSample {
                corner,
                x: 0,
                y: 0,
                rotation: 0,
            }
        }
    }

    /// Grid side of a face's ptex squares at the current level: a quad is one ptex face of twice the
    /// corner side.
    fn side(&self, face: usize) -> usize {
        if self.faces[face].count == 4 {
            2 * self.grid
        } else {
            self.grid
        }
    }

    /// Blender face index and loop-start slot of each sample in the current ptex grid.
    fn blender_faces(&self) -> Vec<(usize, usize)> {
        let mut offsets = Vec::with_capacity(self.faces.len());
        let mut total = 0;
        for (index, face) in self.faces.iter().enumerate() {
            let side = self.side(index);
            let ptex_faces = if face.count == 4 { 1 } else { face.count };
            offsets.push(total);
            total += ptex_faces * side * side;
        }
        self.samples
            .iter()
            .map(|sample| {
                let face = self.corner_face[sample.corner];
                let side = self.side(face);
                if self.faces[face].count == 4 {
                    (
                        offsets[face] + quad_index(sample.x, sample.y, side),
                        boundary_run(sample.x, sample.y, side),
                    )
                } else {
                    let (index, run) = ngon_square_order(
                        self.corner_local[sample.corner],
                        sample.x,
                        sample.y,
                        side,
                        self.faces[face].count,
                    );
                    (offsets[face] + index, run)
                }
            })
            .collect()
    }

    /// Blender loop ordinal of each output corner, in Potter order.
    fn loop_order(&self) -> Vec<usize> {
        let faces = self.blender_faces();
        let mut order = Vec::with_capacity(self.samples.len() * 4);
        for (sample, &(face, run)) in self.samples.iter().zip(&faces) {
            for corner in 0..4 {
                order.push(4 * face + (sample.rotation + corner + 4 - run) % 4);
            }
        }
        order
    }

    /// Writes the output custom normals through the Subdivision setter, with the output loop numbering as
    /// the basis when it differs from storage order. Without a custom layer only that basis remains.
    pub(super) fn write_back(&self, result: &mut Mesh) -> Result<()> {
        let order = self.loop_order();
        let identity = order.iter().enumerate().all(|(index, &ordinal)| index == ordinal);
        let options = NormalCodeOptions {
            corner_order: (!identity).then_some(order.as_slice()),
            setter_input: SetterInput::AlreadyNormalized,
        };
        let Some(patches) = &self.patches else {
            return clnor::write_codes(result, None, options);
        };
        let normals: Vec<Option<DVec3>> = self
            .samples
            .iter()
            .enumerate()
            .flat_map(|(sample_index, sample)| {
                let order = &order;
                let side = usize_as_f32(self.side(self.corner_face[sample.corner]));
                let patch = patches[sample.corner];
                (0..4).map(move |corner| {
                    let loop_slot = order[sample_index * 4 + corner] % 4;
                    let (dx, dy) = SLOT_OFFSETS[loop_slot];
                    let u = usize_as_f32(sample.x + dx) / side;
                    let v = usize_as_f32(sample.y + dy) / side;
                    let weights = [
                        (1.0 - u) * (1.0 - v),
                        u * (1.0 - v),
                        u * v,
                        (1.0 - u) * v,
                    ];
                    let mut value = Vec3::ZERO;
                    for (&point, weight) in patch.iter().zip(weights) {
                        value += point.as_vec3() * weight;
                    }
                    Some(clnor::normalize(value).as_dvec3())
                })
            })
            .collect();
        clnor::encode_vectors(result, Some(normals.as_slice()), options)
    }
}

/// Blender loop position of each input corner within its face, from the input's loop order (storage order
/// without one). Blender numbers a face's loops as one rotation of its own cycle, so a reversed or
/// interleaved run has no polygon numbering.
fn input_locals(mesh: &Mesh) -> Result<Vec<usize>> {
    let order = clnor::corner_order(mesh)?;
    let mut locals = Vec::new();
    let mut start = 0;
    for face in &mesh.faces {
        let count = face.vertices.len();
        let ordinals: Vec<usize> = (start..start + count)
            .map(|corner| order.as_ref().map_or(corner, |order| order[corner]))
            .collect();
        let base = ordinals.iter().copied().min().unwrap_or(start);
        let local: Vec<usize> = ordinals.iter().map(|ordinal| ordinal - base).collect();
        let rotation = local.first().copied().unwrap_or(0);
        if local
            .iter()
            .enumerate()
            .any(|(corner, &index)| index != (rotation + corner) % count)
        {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "custom normal corner_order is not a Blender polygon loop order",
            ));
        }
        locals.extend(local);
        start += count;
    }
    Ok(locals)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Subdivision interpolation matches Blender's f32 loop weights"
)]
fn usize_as_f32(value: usize) -> f32 {
    value as f32
}

/// Blender's interpolation patch of every input corner. A quad face is one patch of its raw corners in
/// slot order; an n-gon corner is `[corner, edge to next, face mean, edge to previous]`, with mixed values
/// normalized as `mix_normals` does.
fn corner_patches(vectors: &[DVec3], faces: &[NormalFace], locals: &[usize]) -> Vec<[DVec3; 4]> {
    let mut patches = vec![[DVec3::ZERO; 4]; vectors.len()];
    for face in faces {
        let corners = &vectors[face.start..face.start + face.count];
        if face.count == 4 {
            let rotation = locals[face.start];
            let patch: [DVec3; 4] = std::array::from_fn(|slot| corners[(slot + 4 - rotation) % 4]);
            patches[face.start..face.start + 4].fill(patch);
        } else {
            let weight = 1.0 / usize_as_f32(face.count);
            let mut sum = Vec3::ZERO;
            for &corner in corners {
                sum += corner.as_vec3() * weight;
            }
            let mean = clnor::normalize(sum).as_dvec3();
            for (index, &current) in corners.iter().enumerate() {
                let next = corners[(index + 1) % face.count];
                let previous = corners[(index + face.count - 1) % face.count];
                patches[face.start + index] = [
                    current,
                    clnor::normalize(current.as_vec3() * 0.5 + next.as_vec3() * 0.5)
                        .as_dvec3(),
                    mean,
                    clnor::normalize(previous.as_vec3() * 0.5 + current.as_vec3() * 0.5)
                        .as_dvec3(),
                ];
            }
        }
    }
    patches
}

/// The boundary run of a square on a `side × side` patch: 0 bottom, 1 right, 2 top, 3 left. Interior
/// squares report 0, where loops start at the lower-left corner.
fn boundary_run(x: usize, y: usize, side: usize) -> usize {
    let last = side - 1;
    if y == 0 && x < last {
        0
    } else if x == last && y < last {
        1
    } else if y == last && x >= 1 {
        2
    } else if x == 0 && y >= 1 {
        3
    } else {
        0
    }
}

/// Blender's special n-gon ptex traversal order: inner squares, patch connections, centre squares, then
/// U and V boundary squares. Returns the face index within the parent polygon and loop-start slot.
fn ngon_square_order(
    corner: usize,
    x: usize,
    y: usize,
    side: usize,
    corners: usize,
) -> (usize, usize) {
    let inner = side.saturating_sub(1);
    let boundary = side.saturating_sub(2);
    let inner_per_corner = inner * boundary;
    let inner_faces = corners * inner_per_corner;
    let connection_faces = corners * boundary;
    let centre_per_corner = usize::from(side >= 2);
    let centre_faces = corners * centre_per_corner;

    if side >= 2 && (1..side).contains(&x) && (1..side - 1).contains(&y) {
        return (
            corner * inner_per_corner + (y - 1) * inner + (x - 1),
            0,
        );
    }
    if side >= 3 && y == side - 1 && (1..side - 1).contains(&x) {
        return (inner_faces + corner * boundary + x - 1, 1);
    }
    if side >= 2 && x == side - 1 && y == side - 1 {
        return (inner_faces + connection_faces + corner, 2);
    }

    let boundary_start = inner_faces + connection_faces + centre_faces;
    if y == 0 {
        (boundary_start + corner * side + x, 0)
    } else {
        (
            boundary_start + corners * side + corner * inner + (y - 1),
            3,
        )
    }
}

/// Blender's face index of square `(x, y)` in a quad's `side × side` ptex grid (`subdiv_foreach.cc`
/// `loops_regular`): interior squares row by row, then the boundary runs of corners 0 to 3.
fn quad_index(x: usize, y: usize, side: usize) -> usize {
    let last = side - 1;
    let inner = side - 2;
    if (1..last).contains(&x) && (1..last).contains(&y) {
        return (y - 1) * inner + (x - 1);
    }
    let ring = inner * inner;
    match boundary_run(x, y, side) {
        0 => ring + x,
        1 => ring + last + y,
        2 => ring + 2 * last + (last - x),
        _ => ring + 3 * last + (last - y),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn quads(count: usize) -> NormalSubdivision {
        NormalSubdivision {
            patches: None,
            faces: (0..count)
                .map(|index| NormalFace {
                    start: index * 4,
                    count: 4,
                })
                .collect(),
            corner_face: (0..count * 4).map(|corner| corner / 4).collect(),
            corner_local: (0..count * 4).map(|corner| corner % 4).collect(),
            samples: Vec::new(),
            grid: 0,
        }
    }

    fn ngon() -> NormalSubdivision {
        NormalSubdivision {
            patches: None,
            faces: vec![NormalFace { start: 0, count: 5 }],
            corner_face: vec![0; 5],
            corner_local: (0..5).collect(),
            samples: Vec::new(),
            grid: 0,
        }
    }

    #[test]
    fn loop_numbering_is_identity_on_level_one_and_a_permutation_on_level_two() {
        let mut state = quads(2);
        state.refine();
        assert_eq!(state.loop_order(), (0..32).collect::<Vec<_>>());
        state.refine();
        let mut order = state.loop_order();
        order.sort_unstable();
        assert_eq!(order, (0..128).collect::<Vec<_>>());
    }

    #[test]
    fn ngon_loop_numbering_covers_blender_ptex_passes() {
        let mut state = ngon();
        state.refine();
        assert_eq!(state.loop_order(), (0..20).collect::<Vec<_>>());
        state.refine();
        let mut order = state.loop_order();
        order.sort_unstable();
        assert_eq!(order, (0..80).collect::<Vec<_>>());
        state.refine();
        let mut order = state.loop_order();
        order.sort_unstable();
        assert_eq!(order, (0..320).collect::<Vec<_>>());
    }

    #[test]
    fn quad_patch_is_read_in_slot_order_from_the_input_rotation() {
        let corners = [DVec3::X, DVec3::Y, DVec3::Z, DVec3::NEG_X];
        let faces = [NormalFace { start: 0, count: 4 }];
        let patches = corner_patches(&corners, &faces, &[1, 2, 3, 0]);
        // Corner 0 sits in slot 1, so slot 0 holds corner 3.
        assert_eq!(patches[0][0], DVec3::NEG_X);
        assert_eq!(patches[0][1], DVec3::X);
    }
}
