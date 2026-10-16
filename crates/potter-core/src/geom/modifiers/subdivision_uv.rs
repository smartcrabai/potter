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

use std::collections::{BTreeMap, HashMap};

use glam::{DVec2, DVec3};
use serde_json::{Map, json};

use super::{
    NonManifold, Polygon, blender_edge_crease_from_sharpness, build_mesh, positions, usize_as_f64,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, edge_key,
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

#[derive(Clone, Copy)]
enum Rule {
    /// Sharp corner: the value does not move.
    Corner,
    /// Semi-sharp corner blended from the smooth rule.
    BlendedCorner(f64),
    /// Smooth boundary blended from the smooth rule according to crease sharpness.
    Crease([usize; 2], f64),
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
    /// For each face-varying edge with a smooth portion, adjacent faces and crease weight.
    edge_smooth: Vec<Option<([usize; 2], f64)>>,
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

#[derive(Clone, Copy)]
struct Link {
    mismatch: bool,
    sharpness: f64,
}

/// Position topology of one level, shared by geometry and all face-varying layers.
pub(super) struct Position {
    pub(super) starts: Vec<usize>,
    pub(super) face_of: Vec<usize>,
    pub(super) corner_vertex: Vec<usize>,
    pub(super) corner_edge: Vec<usize>,
    pub(super) edge_starts: Vec<usize>,
    pub(super) edge_sides: Vec<usize>,
    pub(super) edge_ends: Vec<(usize, usize)>,
    pub(super) edge_order: Vec<usize>,
    pub(super) edge_sharp: Vec<f64>,
    pub(super) vertex_starts: Vec<usize>,
    pub(super) vertex_corners: Vec<usize>,
    pub(super) vertex_edge_starts: Vec<usize>,
    pub(super) vertex_edges: Vec<usize>,
    pub(super) non_manifold_corners: Vec<bool>,
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
    /// Position topology shared by the geometry and every face-varying channel at this level.
    pub(super) fn new(
        polys: &[Polygon],
        loose_edges: &[[usize; 2]],
        creases: &BTreeMap<(usize, usize), f64>,
        vertex_count: usize,
        with_fans: bool,
    ) -> Self {
        let non_manifold = NonManifold::new(
            vertex_count,
            polys.iter().map(|(face, _)| face.as_slice()),
        );
        let sharpness = non_manifold.sharpened(creases);
        let total: usize = polys.iter().map(|(face, _)| face.len()).sum();
        let mut starts = Vec::with_capacity(polys.len() + 1);
        let mut face_of = Vec::with_capacity(total);
        let mut corner_vertex = Vec::with_capacity(total);
        starts.push(0);
        for (face_index, (poly, _)) in polys.iter().enumerate() {
            corner_vertex.extend_from_slice(poly);
            face_of.extend(std::iter::repeat_n(face_index, poly.len()));
            starts.push(corner_vertex.len());
        }

        let mut edge_index: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(total / 2 + loose_edges.len());
        let mut edge_ends = Vec::with_capacity(total / 2 + loose_edges.len());
        let mut edge_sharp = Vec::new();
        let mut corner_edge = Vec::with_capacity(total);
        for (poly, _) in polys {
            for corner in 0..poly.len() {
                let key = edge_key(poly[corner], poly[(corner + 1) % poly.len()]);
                let edge = *edge_index.entry(key).or_insert_with(|| {
                    edge_ends.push(key);
                    edge_sharp.push(sharpness.get(&key).copied().unwrap_or(0.0));
                    edge_ends.len() - 1
                });
                corner_edge.push(edge);
            }
        }
        for &[first, second] in loose_edges {
            let key = edge_key(first, second);
            edge_index.entry(key).or_insert_with(|| {
                edge_ends.push(key);
                edge_sharp.push(sharpness.get(&key).copied().unwrap_or(0.0));
                edge_ends.len() - 1
            });
        }
        let (edge_starts, edge_sides) = group(&corner_edge, edge_ends.len());
        let (vertex_starts, vertex_corners) = group(&corner_vertex, vertex_count);
        let endpoints = edge_ends
            .iter()
            .flat_map(|(first, second)| [*first, *second])
            .collect::<Vec<_>>();
        let (vertex_edge_starts, vertex_edges) = group(&endpoints, vertex_count);
        let mut edge_order: Vec<usize> = (0..edge_ends.len()).collect();
        edge_order.sort_unstable_by_key(|edge| edge_ends[*edge]);

        let mut position = Self {
            starts,
            face_of,
            corner_vertex,
            corner_edge,
            edge_starts,
            edge_sides,
            edge_ends,
            edge_order,
            edge_sharp,
            vertex_starts,
            vertex_corners,
            vertex_edge_starts,
            vertex_edges,
            non_manifold_corners: (0..vertex_count)
                .map(|vertex| non_manifold.corner(vertex))
                .collect(),
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


    fn sharpness(&self, corner: usize) -> f64 {
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
        let next = self.next(side);
        let other_next = self.next(other);
        if self.corner_vertex[side] != self.corner_vertex[other_next]
            || self.corner_vertex[next] != self.corner_vertex[other]
        {
            return None;
        }
        Some(if self.corner_vertex[other] == vertex {
            other
        } else {
            other_next
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
            sharpness: f64::INFINITY,
        };
    }
    let before = fan.corners[(index + size - 1) % size];
    let after = fan.corners[index % size];
    let after_back = walk.side_edge[walk.position.prev(after)];
    Link {
        mismatch: walk.side_edge[before] != after_back,
        sharpness: walk.position.sharpness(before),
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
        let mut sharp: Vec<(usize, f64)> = (0..link_count)
            .filter_map(|index| {
                (links[index].sharpness > 0.0).then_some((index, links[index].sharpness))
            })
            .collect();
        sharp.sort_unstable_by(|first, second| second.1.total_cmp(&first.1));
        rules[value] = if vertex_sharp {
            Rule::Corner
        } else if sharp.len() > 2 {
            if sharp[2].1 >= 1.0 {
                Rule::Corner
            } else {
                Rule::BlendedCorner(sharp[2].1)
            }
        } else if let &[first, second] = sharp.as_slice() {
            Rule::Crease(
                [link_end(walk, fan, first.0), link_end(walk, fan, second.0)],
                first.1.min(second.1),
            )
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
                    .filter(|other| *other != index && links[*other].sharpness > 0.0)
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
                .filter(|p| links[corner_of(*p)].sharpness > 0.0)
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
                span.ends
                    .map_or(Rule::Corner, |ends| Rule::Crease(ends, f64::INFINITY))
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
                && position.side_faces(first) == 2)
                .then(|| {
                    (
                        [position.face_of[first], position.face_of[second]],
                        position.sharpness(first),
                    )
                })
        })
        .collect();
    Topology {
        rules,
        edge_ends,
        side_edge,
        edge_smooth,
    }
}
fn smooth_vertex(uv: DVec2, sum: Sums) -> DVec2 {
    if sum.count == 0 {
        return uv;
    }
    let count = usize_as_f64(sum.count);
    let edge_midpoints = (uv * count + sum.neighbours * 0.5) / (2.0 * count);
    (sum.face / count + edge_midpoints * 2.0 + uv * (count - 3.0)) / count
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
            let smooth = smooth_vertex(uv, sums[value]);
            match topology.rules[value] {
                Rule::Corner => uv,
                Rule::BlendedCorner(weight) => smooth.lerp(uv, weight.min(1.0)),
                Rule::Crease([first, second], weight) => {
                    let crease = uv * 0.75 + (uvs[first] + uvs[second]) * 0.125;
                    smooth.lerp(crease, weight.min(1.0))
                }
                Rule::Smooth => smooth,
            }
        })
        .collect();
    for (edge, [first, second]) in topology.edge_ends.iter().enumerate() {
        child_uvs.push(match topology.edge_smooth[edge] {
            Some(([first_face, second_face], weight)) => {
                let smooth =
                    (uvs[*first] + uvs[*second] + face_centers[first_face] + face_centers[second_face])
                        * 0.25;
                let linear = (uvs[*first] + uvs[*second]) * 0.5;
                smooth.lerp(linear, weight.min(1.0))
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

impl Layer {
    /// Limit values of the final level. A single-face crease value follows the boundary chain of its
    /// level-0 value, iterated `quality` times.
    fn limit(&self, topology: &Topology, sums: &[Sums], quality: u64) -> Vec<DVec2> {
        let uvs = &self.current.uvs;
        (0..uvs.len())
            .map(|value| {
                let uv = uvs[value];
                let sum = sums[value];
                let smooth = if sum.count == 0 {
                    uv
                } else {
                    let count = usize_as_f64(sum.count);
                    (uv * (count * count) + sum.neighbours * 2.0 + sum.opposite)
                        / (count * (count + 5.0))
                };
                match topology.rules[value] {
                    Rule::Corner => uv,
                    Rule::BlendedCorner(weight) => smooth.lerp(uv, weight.min(1.0)),
                    Rule::Crease([first, second], weight) if sum.count == 1 => {
                        let regular = (uv * 4.0 + uvs[first] + uvs[second]) / 6.0;
                        let corner = self.corner_limit(value, [first, second], quality);
                        regular.lerp(corner, weight.min(1.0))
                    }
                    Rule::Crease([first, second], weight) => {
                        let crease = (uv * 4.0 + uvs[first] + uvs[second]) / 6.0;
                        smooth.lerp(crease, weight.min(1.0))
                    }
                    Rule::Smooth => smooth,
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

impl UvSubdivision {
    /// Face-varying channels of every UV layer of `mesh` (the level-0 state).
    pub(super) fn new(
        mesh: &Mesh,
        options: UvOptions,
        position: &Position,
    ) -> Result<Self> {
        let layers = uv_layers::read_layers(mesh);
        if layers.is_empty() {
            return Ok(Self {
                options,
                layers: Vec::new(),
            });
        }
        let layers = layers
            .into_iter()
            .map(|layer| {
                let base = channel_from(position, &corner_uvs(position, &layer));
                let topology = classify(&base, position, options);
                let base_ends = topology
                    .rules
                    .iter()
                    .map(|rule| match rule {
                        Rule::Crease(ends, _) => Some(*ends),
                        _ => None,
                    })
                    .collect();
                Layer {
                    name: layer.name,
                    base_uvs: base.uvs.clone(),
                    base_ends,
                    current: base,
                }
            })
            .collect();
        Ok(Self { options, layers })
    }

    /// Refines every channel over the shared position topology for its parent level.
    pub(super) fn refine(&mut self, position: &Position) -> Result<()> {
        for layer in &mut self.layers {
            let topology = classify(&layer.current, position, self.options);
            layer.current = refine_channel(std::mem::take(&mut layer.current), &topology);
        }
        Ok(())
    }

    /// Writes the channels to the final mesh's UV attributes. `quality` gives the limit surface
    /// (`use_limit_surface`), or `None` for the refined values.
    pub(super) fn write_back(
        &self,
        result: &mut Mesh,
        position: &Position,
        quality: Option<u64>,
    ) -> Result<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        let limit = quality.map(|quality| (quality, position));
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            if layer.current.face_count() != position.starts.len() - 1 {
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
            let faces = (0..layer.current.face_count())
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
            layers.push(UvLayer {
                name: layer.name.clone(),
                faces,
            });
        }
        uv_layers::write_layers(result, &layers);
        Ok(())
    }
}

pub(super) fn subdivide_simple_once(
    mesh: &Mesh,
    position: &Position,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    edge_creases: &BTreeMap<(usize, usize), f64>,
    use_creases: bool,
    raw_normals: bool,
) -> Result<Mesh> {
    let source_points = positions(mesh);
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
    for &edge_index in &position.edge_order {
        let edge = position.edge_ends[edge_index];
        edge_point_indices.insert(edge, points.len());
        points.push((source_points[edge.0] + source_points[edge.1]) * 0.5);
    }
    points.extend(face_points);
    subdivided_mesh(
        mesh,
        points,
        source_polygons,
        source_edges,
        &edge_point_indices,
        edge_creases,
        use_creases,
        raw_normals,
    )
}

/// One subdivision level's mesh from refined points, shared by SIMPLE and Catmull-Clark. `points` holds
/// the refined vertices, then one point per edge in `edge_points` key order, then one per source face.
/// Every source corner becomes one quad `[corner, next edge point, face point, previous edge point]`,
/// faces in source order. Loose edges split at their edge point, and each output vertex records its
/// source origin. Internal sharpness decays by one level and is stored as Blender's crease value.
pub(super) fn subdivided_mesh(
    mesh: &Mesh,
    points: Vec<DVec3>,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    edge_points: &BTreeMap<(usize, usize), usize>,
    edge_creases: &BTreeMap<(usize, usize), f64>,
    use_creases: bool,
    raw_normals: bool,
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
        raw_normals,
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
            if let Some(sharpness) = next_creases.get(&ordered) {
                values.insert(
                    format!("e{}", edge.id),
                    json!(blender_edge_crease_from_sharpness(*sharpness)),
                );
            }
        }
        output.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain":"edge","type":"float","values":values}),
        );
    }
    Ok(output)
}
