//! Blender's bevel front end (`bmesh_bevel.cc`): per-vertex edge halves, boundary vertices, offset
//! meeting points, profile parameters, width adjustment, and geometry-collision clamping.
//!
//! Blender links `EdgeHalf`, `BoundVert`, and `BevVert` with pointers. Here they live in index
//! arenas: an edge half or boundary vertex is referenced by its arena index. `BevVert` lists its
//! edge halves in CCW order and its boundary ring in Blender's order; `bevel_rings.rs` reads the
//! finished rings to build the vertex meshes.
//!
//! Blender's `build_boundary(construct = false)` refreshes coordinates on the ring that
//! `construct = true` made, retaining its topology and boundary metadata.

use std::collections::{HashMap, HashSet};

use glam::DVec3;

use super::Polygon;
use super::bevel_profile::CurveProfile;
use super::bevel_vmesh::{even_chords, power_of_two_at_least};
use crate::error::{ErrorCode, PotError, Result};
use crate::geom::edge_key;

/// Builds the error for a broken internal invariant of the front end. Reaching it means the
/// front end's own bookkeeping is inconsistent, so it surfaces as an `InternalError`.
#[must_use]
pub(super) fn internal_invariant(message: &'static str) -> PotError {
    PotError::new(ErrorCode::InternalError, message)
}

/// Blender's `BEVEL_EPSILON`, rounded to binary32 as Blender's float code uses it.
pub(super) const EPSILON: f64 = 1.0e-6_f32 as f64;
/// Blender's `BEVEL_EPSILON_BIG`.
pub(super) const EPSILON_BIG: f64 = 1.0e-4_f32 as f64;
/// Blender's `BEVEL_EPSILON_SQ`.
pub(super) const EPSILON_SQ: f64 = 1.0e-12_f32 as f64;
/// Blender's `BEVEL_EPSILON_ANG` (2 degrees).
pub(super) const EPSILON_ANG: f64 = (2.0_f32 * std::f32::consts::PI / 180.0_f32) as f64;
/// Blender's `BEVEL_SMALL_ANG` (10 degrees).
const SMALL_ANG: f64 = (10.0_f32 * std::f32::consts::PI / 180.0_f32) as f64;
/// Blender's `BEVEL_GOOD_ANGLE`.
const GOOD_ANGLE: f64 = 0.1_f32 as f64;
/// Blender's `BEVEL_MATCH_SPEC_WEIGHT`.
const MATCH_SPEC_WEIGHT: f64 = 0.2;
/// Blender's `BEVEL_EPSILON_D`.
const EPSILON_D: f64 = 1.0e-6_f32 as f64;
/// `FLT_EPSILON`: the parallel-line threshold of `isect_line_plane_v3`.
const FLT_EPSILON: f64 = f32::EPSILON as f64;

/// Superellipse exponent of a square profile (Blender `PRO_SQUARE_R`).
pub(super) const PRO_SQUARE_R: f64 = 1.0e4;
/// Superellipse exponent of a circular profile (Blender `PRO_CIRCLE_R`).
pub(super) const PRO_CIRCLE_R: f64 = 2.0;
/// Superellipse exponent of a linear profile (Blender `PRO_LINE_R`).
pub(super) const PRO_LINE_R: f64 = 1.0;
/// Superellipse exponent of an inward square profile (Blender `PRO_SQUARE_IN_R`).
pub(super) const PRO_SQUARE_IN_R: f64 = 0.0;

/// Blender's `BEVEL_AMT_*`: how the user width becomes a distance along each adjacent edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OffsetType {
    Offset,
    Width,
    Depth,
    Percent,
    Absolute,
}

/// Blender's `BEVEL_MITER_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MiterKind {
    Sharp,
    Patch,
    Arc,
}

/// Blender's `BEVEL_VMESH_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VmeshMethod {
    Adj,
    Cutoff,
}

/// Blender's vertex-mesh kinds (`M_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MeshKind {
    None,
    Poly,
    Adj,
    Cutoff,
    TriFan,
}

/// Blender's `AngleKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AngleKind {
    Smaller,
    Straight,
    Larger,
}

/// Bevel settings the front end reads (Blender's `BevelParams`).
#[derive(Clone, Debug)]
pub(super) struct Params {
    pub offset: f64,
    pub offset_type: OffsetType,
    pub seg: usize,
    pub profile: f64,
    /// `bp->pro_super_r` after Blender's special-case snapping (see [`super_r_of`]).
    pub pro_super_r: f64,
    /// The authored curve when `profile_type == CUSTOM`.
    pub custom: Option<CurveProfile>,
    /// `affect == VERTICES`.
    pub affect_vertices: bool,
    pub loop_slide: bool,
    pub miter_outer: MiterKind,
    pub miter_inner: MiterKind,
    pub spread: f64,
    pub vmesh_method: VmeshMethod,
    pub mark_seam: bool,
    pub mark_sharp: bool,
    /// `use_clamp_overlap`: clamp the offset to avoid geometry collisions.
    pub limit_offset: bool,
}

impl Params {
    /// Blender's `offset_adjust` for EDGES: a width-adjust pass unless the offset is a leg slide.
    fn offset_adjust(&self) -> bool {
        !self.affect_vertices
            && !matches!(self.offset_type, OffsetType::Percent | OffsetType::Absolute)
    }

    /// Blender's odd-segment VERTICES condition.
    fn affect_vertices_odd(&self) -> bool {
        self.affect_vertices && self.seg % 2 == 1
    }
}

/// Input geometry and selection, in mesh point indices.
#[derive(Clone, Debug)]
pub(super) struct Geometry {
    pub points: Vec<DVec3>,
    pub polygons: Vec<Polygon>,
    pub face_normals: Vec<DVec3>,
    /// Edges that fail Blender's `contig_ldata_across_edge` test, including edges without a loop pair.
    pub seam_edges: HashSet<(usize, usize)>,
    /// Discontinuous `contig_ldata_around_vert` flags; absent when no UV layer needs checking.
    pub seam_vertices: Option<Vec<bool>>,
    /// Original `(sharp, seam)` edge flags in `edge_order` order.
    pub input_edge_flags: Vec<(bool, bool)>,
    /// Manifold beveled edges (EDGES), as canonical keys.
    pub selected_edges: HashSet<(usize, usize)>,
    /// All mesh edges in mesh order, each with its stored endpoint direction.
    pub edge_order: Vec<(usize, usize)>,
    /// Beveled vertices (VERTICES).
    pub selected_vertices: HashSet<usize>,
    /// Edge bevel weights (`bweight_offset_edge`); `None` when weights are not used.
    pub edge_weight: Option<HashMap<(usize, usize), f64>>,
    /// Vertex weights (vertex group or vertex bevel weight); `None` when weights are not used.
    pub vertex_weight: Option<HashMap<usize, f64>>,
}
/// Blender's `Profile`: the arc of one boundary vertex's profile from `start` to `end`.
#[derive(Clone, Debug, Default)]
pub(super) struct BoundProfile {
    pub super_r: f64,
    pub start: DVec3,
    pub middle: DVec3,
    pub end: DVec3,
    pub plane_co: DVec3,
    pub plane_no: DVec3,
    pub proj_dir: DVec3,
    pub special_params: bool,
    pub height: f64,
    /// Profile points for `seg` segments (empty when `seg == 1`).
    pub prof_co: Vec<DVec3>,
    /// Profile points for the power-of-two `seg_2` sampling (empty when `seg == 1`).
    pub prof_co_2: Vec<DVec3>,
}

/// Blender's `ProfileSpacing`: unit-profile sample coordinates for `seg` and `seg_2` segments.
#[derive(Clone, Debug, Default)]
pub(super) struct ProfileSpacing {
    pub seg_2: usize,
    pub xvals: Vec<f64>,
    pub yvals: Vec<f64>,
    pub xvals_2: Vec<f64>,
    pub yvals_2: Vec<f64>,
}

impl ProfileSpacing {
    /// Blender's `set_profile_spacing`. A `CUSTOM` profile supplies its curve samples; other
    /// profiles use the even superellipse chords of exponent `super_r`.
    fn new(seg: usize, super_r: f64, custom: Option<&CurveProfile>) -> Self {
        if seg <= 1 {
            return Self::default();
        }
        let seg_2 = power_of_two_at_least(seg).max(4);
        let (xvals, yvals) = samples(seg, super_r, custom);
        let (xvals_2, yvals_2) = if seg_2 == seg {
            (xvals.clone(), yvals.clone())
        } else {
            samples(seg_2, super_r, custom)
        };
        Self {
            seg_2,
            xvals,
            yvals,
            xvals_2,
            yvals_2,
        }
    }
}

/// Unit-profile chords for `n` segments. The curve's samples are read as Blender reads
/// `segments[i]`: x from `.y` and y from `.x`.
fn samples(n: usize, super_r: f64, custom: Option<&CurveProfile>) -> (Vec<f64>, Vec<f64>) {
    match custom {
        Some(curve) => {
            let points = curve.sample(n);
            (
                points.iter().map(|point| point.1).collect(),
                points.iter().map(|point| point.0).collect(),
            )
        }
        None => even_chords(n, super_r),
    }
}

/// Blender's `EdgeHalf`: one edge at a beveled vertex.
#[derive(Clone, Debug)]
pub(super) struct EdgeHalf {
    /// The beveled vertex this half belongs to.
    pub vert: usize,
    /// The vertex at the other end of the edge.
    pub other: usize,
    /// Blender's `is_rev`: the beveled vertex is the edge's second endpoint.
    pub is_rev: bool,
    pub is_bev: bool,
    pub seg: usize,
    /// The faces on either side of the edge at this vertex.
    pub fprev: Option<usize>,
    pub fnext: Option<usize>,
    /// Arena ids of the CCW neighbors of this edge half at the vertex.
    pub next: usize,
    pub prev: usize,
    /// Arena ids of the boundary vertices on the left and right of this edge.
    pub leftv: Option<usize>,
    pub rightv: Option<usize>,
    /// Index in `Geometry::edge_order`, preserving source mesh edge identity.
    pub edge_id: usize,
    pub offset_l_spec: f64,
    pub offset_r_spec: f64,
    pub offset_l: f64,
    pub offset_r: f64,
    pub is_seam: bool,
    pub profile_index: usize,
    /// Blender's `visited_rpo`: profile orientation was assigned for this edge half.
    pub visited_rpo: bool,
}

/// Blender's `BoundVert`: one point on the boundary of a beveled vertex.
#[derive(Clone, Debug)]
pub(super) struct BoundVert {
    pub next: usize,
    pub prev: usize,
    pub co: DVec3,
    pub efirst: Option<usize>,
    pub elast: Option<usize>,
    pub ebev: Option<usize>,
    pub eon: Option<usize>,
    pub sinratio: f64,
    pub adjchain: Option<usize>,
    pub visited: bool,
    pub is_arc_start: bool,
    pub is_patch_start: bool,
    pub is_profile_start: bool,
    pub any_seam: bool,
    pub seam_len: usize,
    pub sharp_len: usize,
    pub profile: BoundProfile,
}

impl BoundVert {
    fn new(co: DVec3) -> Self {
        Self {
            next: 0,
            prev: 0,
            co,
            efirst: None,
            elast: None,
            ebev: None,
            eon: None,
            sinratio: 1.0,
            adjchain: None,
            visited: false,
            is_arc_start: false,
            is_patch_start: false,
            is_profile_start: false,
            any_seam: false,
            seam_len: 0,
            sharp_len: 0,
            profile: BoundProfile::default(),
        }
    }
}

/// Blender's `BevVert`: a beveled input vertex with its edge halves and boundary ring.
#[derive(Clone, Debug)]
pub(super) struct BevVert {
    /// Arena ids of the edge halves in CCW order.
    pub edges: Vec<usize>,
    pub selcount: usize,
    pub offset: f64,
    /// Arena ids of the boundary ring; `ring[0]` is Blender's `boundstart`.
    pub ring: Vec<usize>,
    pub mesh_kind: MeshKind,
    pub any_seam: bool,
}

/// Blender's `BM_mesh_bevel` front end: every beveled vertex with its edge halves and boundary.
pub(super) struct Front {
    pub p: Params,
    pub geo: Geometry,
    pub vertex_normals: Vec<DVec3>,
    pub pro_spacing: ProfileSpacing,
    pub pro_spacing_miter: Option<ProfileSpacing>,
    pub edges: Vec<EdgeHalf>,
    edge_ids: HashMap<(usize, usize), usize>,
    edge_faces_by_id: Vec<Vec<usize>>,
    incident_edges: Vec<Vec<usize>>,
    pub bounds: Vec<BoundVert>,
    /// `BevVert` per input vertex index; `None` for vertices that are not beveled.
    pub verts: Vec<Option<BevVert>>,
}

/// The superellipse exponent of a profile fraction, after Blender's `BM_mesh_bevel` snapping of
/// the special cases: square (`>= 0.95`), circle, line, and inward square.
pub(super) fn super_r_of(profile: f64) -> f64 {
    let profile_f32 = profile as f32;
    if profile_f32 >= 0.950 {
        return PRO_SQUARE_R;
    }
    let exponent = -(2.0_f32.ln()) / profile_f32.sqrt().ln();
    if (exponent - PRO_CIRCLE_R as f32).abs() < 1.0e-4_f32 {
        PRO_CIRCLE_R
    } else if (exponent - PRO_LINE_R as f32).abs() < 1.0e-4_f32 {
        PRO_LINE_R
    } else if exponent < 1.0e-4_f32 {
        PRO_SQUARE_IN_R
    } else {
        f64::from(exponent)
    }
}

/// Blender's `isect_line_line_v3`: `(kind, r1, r2)` for the lines `v1`-`v2` and `v3`-`v4`.
/// `kind` is 0 for parallel lines, 1 for coplanar lines meeting at one point, and 2 for skew
/// lines, whose nearest points are `r1` and `r2`.
pub(super) fn isect_line_line(v1: DVec3, v2: DVec3, v3: DVec3, v4: DVec3) -> (u8, DVec3, DVec3) {
    let a = f32_sub(v2, v1);
    let b = f32_sub(v4, v3);
    let ab = f32_cross(a, b);
    let div = f32_dot(ab, ab);
    if div == 0.0 {
        return (0, DVec3::ZERO, DVec3::ZERO);
    }
    let c = f32_sub(v3, v1);
    if f32_dot(c, ab).abs() <= EPSILON {
        let point = f32_add(v1, f32_mul(a, f32_div(f32_dot(f32_cross(c, b), ab), div)));
        return (1, point, point);
    }
    let offset = f32_mul(ab, f32_div(f32_dot(f32_sub(v1, v3), ab), div));
    let c = f32_add(c, offset);
    let r1 = f32_add(v1, f32_mul(a, f32_div(f32_dot(f32_cross(c, b), ab), div)));
    (2, r1, f32_sub(r1, offset))
}

/// Blender's `isect_line_plane_v3`: where the line `l1`-`l2` meets the plane, or `None` if the
/// line is parallel to it.
fn isect_line_plane(l1: DVec3, l2: DVec3, plane_co: DVec3, plane_no: DVec3) -> Option<DVec3> {
    let direction = f32_sub(l2, l1);
    let denominator = f32_dot(plane_no, direction);
    if denominator.abs() < FLT_EPSILON {
        return None;
    }
    let t = f32_div(f32_dot(plane_no, f32_sub(plane_co, l1)), denominator);
    Some(f32_add(l1, f32_mul(direction, t)))
}

fn f32_vec(value: DVec3) -> DVec3 {
    value.as_vec3().as_dvec3()
}

fn f32_add(a: DVec3, b: DVec3) -> DVec3 {
    (a.as_vec3() + b.as_vec3()).as_dvec3()
}

fn f32_sub(a: DVec3, b: DVec3) -> DVec3 {
    (a.as_vec3() - b.as_vec3()).as_dvec3()
}

fn f32_mul(value: DVec3, scale: f64) -> DVec3 {
    (value.as_vec3() * scale as f32).as_dvec3()
}

fn f32_dot(a: DVec3, b: DVec3) -> f64 {
    f64::from(a.as_vec3().dot(b.as_vec3()))
}

fn f32_cross(a: DVec3, b: DVec3) -> DVec3 {
    a.as_vec3().cross(b.as_vec3()).as_dvec3()
}

fn f32_length(value: DVec3) -> f64 {
    f64::from(value.as_vec3().length())
}

fn f32_normalize(value: DVec3) -> DVec3 {
    value.as_vec3().normalize_or_zero().as_dvec3()
}

fn f32_sin(value: f64) -> f64 {
    f64::from((value as f32).sin())
}

fn f32_cos(value: f64) -> f64 {
    f64::from((value as f32).cos())
}

fn f32_div(a: f64, b: f64) -> f64 {
    f64::from(a as f32 / b as f32)
}

fn f32_mul_scalar(a: f64, b: f64) -> f64 {
    f64::from(a as f32 * b as f32)
}

fn f32_add_scalar(a: f64, b: f64) -> f64 {
    f64::from(a as f32 + b as f32)
}

fn f32_sub_scalar(a: f64, b: f64) -> f64 {
    f64::from(a as f32 - b as f32)
}

fn f32_asin(value: f64) -> f64 {
    f64::from((value as f32).clamp(-1.0, 1.0).asin())
}

fn angle_normalized_f32(a: DVec3, b: DVec3) -> f64 {
    let dot = f32_dot(a, b);
    if dot >= 0.0 {
        f32_mul_scalar(2.0, f32_asin(f32_div(f32_length(f32_sub(a, b)), 2.0)))
    } else {
        let opposite = f32_mul(b, -1.0);
        let half_angle = f32_asin(f32_div(f32_length(f32_sub(a, opposite)), 2.0));
        f32_sub_scalar(std::f32::consts::PI as f64, f32_mul_scalar(2.0, half_angle))
    }
}

fn angle_f32(a: DVec3, b: DVec3) -> f64 {
    angle_normalized_f32(f32_normalize(a), f32_normalize(b))
}

fn f32_lerp(a: DVec3, b: DVec3, factor: f64) -> DVec3 {
    let t = factor as f32;
    let s = 1.0_f32 - t;
    (a.as_vec3() * s + b.as_vec3() * t).as_dvec3()
}

/// Blender's `angle_v3v3`: the angle between two vectors, in `[0, pi]`.
fn angle(a: DVec3, b: DVec3) -> f64 {
    angle_f32(a, b)
}

/// Blender's `angle_v3v3v3`: the angle at `b` of the corner `a`-`b`-`c`.
fn angle3(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    angle_f32(f32_sub(a, b), f32_sub(c, b))
}

/// Blender's `nearly_parallel`: the directions are within `EPSILON_ANG` of parallel or opposite.
fn nearly_parallel(d1: DVec3, d2: DVec3) -> bool {
    let ang = angle_f32(d1, d2);
    ang.abs() < EPSILON_ANG || f32_sub_scalar(ang, std::f32::consts::PI as f64).abs() < EPSILON_ANG
}

/// Blender's `nearly_parallel_normalized`, for unit vectors.
fn nearly_parallel_normalized(d1: DVec3, d2: DVec3) -> bool {
    let epsilon_ang_dot = f64::from(1.0_f32 - (EPSILON_ANG as f32).cos());
    f32_sub_scalar(f32_dot(d1, d2).abs(), 1.0).abs() <= epsilon_ang_dot
}

/// Blender's `make_unit_square_map` as `c0 * x + c1 * y + c3` for a planar profile point
/// `(x, y)`. `None` when the start, middle, and end are degenerate.
fn unit_square_map(start: DVec3, middle: DVec3, end: DVec3) -> Option<[DVec3; 3]> {
    let va_vmid = f32_sub(middle, start);
    let vb_vmid = f32_sub(middle, end);
    if va_vmid == DVec3::ZERO || vb_vmid == DVec3::ZERO {
        return None;
    }
    let angle = angle_f32(va_vmid, vb_vmid);
    if f32_sub_scalar(angle, std::f32::consts::PI as f64).abs() <= EPSILON_ANG {
        return None;
    }
    Some([
        f32_sub(middle, start),
        f32_sub(middle, end),
        f32_sub(f32_add(start, end), middle),
    ])
}

impl Front {
    /// Blender's `BM_mesh_bevel` front end, in its order: construct every beveled vertex, clamp
    /// the offset if requested, build every boundary, adjust widths, orient custom profiles, and
    /// compute the profile points.
    pub(super) fn build(p: Params, geo: Geometry) -> Result<Self> {
        let mut front = Self::new(p, geo);
        let mut beveled: Vec<usize> = if front.p.affect_vertices {
            front.geo.selected_vertices.iter().copied().collect()
        } else {
            front
                .geo
                .selected_edges
                .iter()
                .flat_map(|(a, b)| [*a, *b])
                .collect()
        };
        beveled.sort_unstable();
        beveled.dedup();
        for vertex in &beveled {
            front.construct_vert(*vertex)?;
        }
        let vertices: Vec<usize> = front
            .verts
            .iter()
            .enumerate()
            .filter_map(|(index, bv)| bv.as_ref().map(|_| index))
            .collect();
        if front.p.limit_offset {
            front.limit_offset(&vertices)?;
        }
        front.build_all(&vertices, true)?;
        if front.p.offset_adjust() {
            front.adjust_offsets(&vertices)?;
        }
        if front.p.custom.is_some() {
            front.regularize_profiles()?;
        }
        for vertex in &vertices {
            front.calculate_vm_profiles(*vertex)?;
        }
        Ok(front)
    }
    fn new(mut p: Params, mut geo: Geometry) -> Self {
        for point in &mut geo.points {
            *point = f32_vec(*point);
        }
        for normal in &mut geo.face_normals {
            *normal = f32_vec(*normal);
        }
        p.seg = p.seg.max(1);
        p.offset = f64::from(p.offset as f32);
        p.profile = f64::from(p.profile as f32);
        p.spread = f64::from(p.spread as f32);
        p.pro_super_r = super_r_of(p.profile);
        if p.vmesh_method == VmeshMethod::Cutoff {
            p.miter_outer = MiterKind::Sharp;
            p.miter_inner = MiterKind::Sharp;
        }
        let pro_spacing = ProfileSpacing::new(p.seg, p.pro_super_r, p.custom.as_ref());
        let pro_spacing_miter = (p.custom.is_some()
            && (p.miter_inner != MiterKind::Sharp || p.miter_outer != MiterKind::Sharp))
            .then(|| ProfileSpacing::new(p.seg, p.pro_super_r, None));
        let vertex_count = geo.points.len();
        if let Some(seam_vertices) = &geo.seam_vertices {
            assert_eq!(
                seam_vertices.len(),
                vertex_count,
                "one loop-data seam flag is required per source vertex",
            );
        }
        let edge_ids: HashMap<_, _> = geo
            .edge_order
            .iter()
            .enumerate()
            .map(|(index, &(a, b))| (edge_key(a, b), index))
            .collect();
        let mut edge_faces_by_id = vec![Vec::new(); geo.edge_order.len()];
        for (face, (vertices, _)) in geo.polygons.iter().enumerate() {
            for index in 0..vertices.len() {
                let key = edge_key(vertices[index], vertices[(index + 1) % vertices.len()]);
                if let Some(&edge_id) = edge_ids.get(&key) {
                    edge_faces_by_id[edge_id].push(face);
                }
            }
        }
        let mut incident_edges = vec![Vec::new(); vertex_count];
        for (edge_id, &(a, b)) in geo.edge_order.iter().enumerate() {
            incident_edges[a].push(edge_id);
            incident_edges[b].push(edge_id);
        }
        let vertex_normals = vertex_normals(&geo);
        Self {
            p,
            geo,
            vertex_normals,
            pro_spacing,
            pro_spacing_miter,
            edges: Vec::new(),
            edge_ids,
            edge_faces_by_id,
            incident_edges,
            bounds: Vec::new(),
            verts: vec![None; vertex_count],
        }
    }
    /// The original `(sharp, seam)` flags of an input edge in mesh order.
    pub(super) fn input_edge_flags_by_id(&self, edge_id: usize) -> (bool, bool) {
        self.geo.input_edge_flags[edge_id]
    }

    /// The mesh edge index of the input edge between two points, if it exists.
    pub(super) fn input_edge_id(&self, first: usize, second: usize) -> Option<usize> {
        self.edge_ids.get(&edge_key(first, second)).copied()
    }

    /// The original `(sharp, seam)` flags of the input edge between two points.
    pub(super) fn input_edge_flags_between(&self, first: usize, second: usize) -> (bool, bool) {
        self.edge_ids
            .get(&edge_key(first, second))
            .map_or((false, false), |edge_id| {
                self.input_edge_flags_by_id(*edge_id)
            })
    }

    /// Blender's `set_bound_vert_seams`: preserve UV seam representatives and, when requested,
    /// calculate the marked input-edge spans extended over the output boundary.
    fn set_bound_vert_seams(&mut self, vertex: usize) -> Result<()> {
        let ring = self.bev(vertex)?.ring.clone();
        let mut any = false;
        for id in ring {
            let mut flag = false;
            if let Some(first) = self.bounds[id].efirst {
                let last = self.bounds[id].elast;
                let mut current = Some(first);
                while let Some(e) = current {
                    flag |= self.edges[e].is_seam;
                    if Some(e) == last {
                        break;
                    }
                    current = Some(self.edges[e].next);
                }
            }
            self.bounds[id].any_seam = flag;
            self.bounds[id].seam_len = 0;
            self.bounds[id].sharp_len = 0;
            any |= flag;
        }
        if self.p.affect_vertices_odd()
            && !any
            && let Some(seam_vertices) = &self.geo.seam_vertices
        {
            any = seam_vertices[vertex];
        }
        self.bev_mut(vertex)?.any_seam = any;
        if self.p.mark_seam {
            self.check_edge_data_seam_sharp_edges(vertex, true)?;
        }
        if self.p.mark_sharp {
            self.check_edge_data_seam_sharp_edges(vertex, false)?;
        }
        Ok(())
    }

    /// Blender's `check_edge_data_seam_sharp_edges`: count intervening beveled edges after each
    /// marked input edge, storing that span on the edge's right boundary vertex.
    fn check_edge_data_seam_sharp_edges(&mut self, vertex: usize, seam: bool) -> Result<()> {
        let Some(mut edge) = self.bev(vertex)?.edges.first().copied() else {
            return Ok(());
        };
        let start = edge;
        while !self.edge_has_mark(edge, seam) {
            edge = self.edges[edge].next;
            if edge == start {
                return Ok(());
            }
        }
        let first_marked = edge;
        loop {
            let mut flag_count = 0;
            let mut next = self.edges[edge].next;
            while !self.edge_has_mark(next, seam) && next != first_marked {
                if self.edges[next].is_bev {
                    flag_count += 1;
                }
                next = self.edges[next].next;
            }
            if next == edge || (next == first_marked && !self.edge_has_mark(first_marked, seam)) {
                break;
            }
            if let Some(bound) = self.edges[edge].rightv {
                if seam {
                    self.bounds[bound].seam_len = flag_count;
                } else {
                    self.bounds[bound].sharp_len = flag_count;
                }
            }
            edge = next;
            if edge == first_marked {
                break;
            }
        }
        Ok(())
    }

    fn edge_has_mark(&self, edge: usize, seam: bool) -> bool {
        let (sharp, input_seam) = self.input_edge_flags_by_id(self.edges[edge].edge_id);
        if seam { input_seam } else { sharp }
    }

    fn edge_loop_successors(
        &self,
        edge_id: usize,
        vertex: usize,
        tagged: &HashSet<usize>,
    ) -> Vec<usize> {
        let edge = edge_key(
            self.geo.edge_order[edge_id].0,
            self.geo.edge_order[edge_id].1,
        );
        let mut successors = Vec::new();
        // `BM_LOOPS_OF_EDGE` visits the newest radial loop first, then the older loops in creation order
        // (`bmesh_radial_loop_append`, bmesh_structure.cc:377-391): [last, first, ..., penultimate].
        let edge_faces = self.edge_faces(edge_id);
        for radial_index in 0..edge_faces.len() {
            let face = edge_faces[if radial_index == 0 {
                edge_faces.len() - 1
            } else {
                radial_index - 1
            }];
            let vertices = &self.geo.polygons[face].0;
            let count = vertices.len();
            for index in 0..count {
                if edge_key(vertices[index], vertices[(index + 1) % count]) != edge {
                    continue;
                }
                let next = if vertices[index] == vertex {
                    edge_key(vertices[(index + count - 1) % count], vertex)
                } else {
                    edge_key(vertex, vertices[(index + 2) % count])
                };
                if let Some(&successor) = self.edge_ids.get(&next)
                    && !tagged.contains(&successor)
                {
                    successors.push(successor);
                }
                break;
            }
        }
        successors
    }

    fn edges_face_connected_at_vert(&self, first: usize, second: usize) -> bool {
        let first_key = edge_key(self.geo.edge_order[first].0, self.geo.edge_order[first].1);
        let second_key = edge_key(self.geo.edge_order[second].0, self.geo.edge_order[second].1);
        for &face in self.edge_faces(first) {
            let vertices = &self.geo.polygons[face].0;
            let count = vertices.len();
            for index in 0..count {
                if edge_key(vertices[index], vertices[(index + 1) % count]) == first_key
                    && (edge_key(vertices[(index + count - 1) % count], vertices[index])
                        == second_key
                        || edge_key(vertices[(index + 1) % count], vertices[(index + 2) % count])
                            == second_key)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Blender's recursive `bevel_edge_order_extend`, including its longest-path tie break.
    fn edge_order_extend(
        &self,
        vertex: usize,
        path: &mut Vec<usize>,
        tagged: &mut HashSet<usize>,
    ) -> usize {
        let start = path.len() - 1;
        let current = path[start];
        let successors = self.edge_loop_successors(current, vertex, tagged);
        let mut best_end = start;
        let mut best_path = Vec::new();
        for successor in successors {
            if !tagged.insert(successor) {
                continue;
            }
            path.push(successor);
            let end = self.edge_order_extend(vertex, path, tagged);
            let better = end > best_end
                || (end == best_end && self.edges_face_connected_at_vert(path[end], path[0]));
            if better {
                best_end = end;
                best_path.clear();
                best_path.extend_from_slice(&path[start + 1..=end]);
            }
            for edge in &path[start + 1..=end] {
                tagged.remove(edge);
            }
            path.truncate(start + 1);
        }
        for &edge in &best_path {
            path.push(edge);
            tagged.insert(edge);
        }
        best_end
    }

    /// Blender's `fast_bevel_edge_order` followed by `find_bevel_edge_order`.
    fn bevel_edge_order(&self, vertex: usize, incident: &[usize]) -> Result<Vec<usize>> {
        // Blender tags selected edges only for EDGES affect (`bmesh_bevel.cc` vert construct, the `nsel` and
        // first-edge test), so a VERTICES bevel starts from a face-count-one edge or the vertex's first edge.
        let mut first = None;
        for &edge_id in incident {
            if first.is_none() && !self.p.affect_vertices && {
                let (a, b) = self.geo.edge_order[edge_id];
                self.is_selected(a, b)
            } {
                first = Some(edge_id);
            }
            if self.edge_face_count(edge_id) == 1 {
                first = Some(edge_id);
            }
        }
        let first = first.unwrap_or(incident[0]);
        let mut order = vec![first];
        let mut tagged = HashSet::from([first]);
        let mut fast = !self.edge_faces(first).is_empty();
        if fast {
            for index in 1..incident.len() {
                let previous = order[index - 1];
                let mut shared_count = 0;
                let mut first_successor = None;
                'successors: for &candidate in incident {
                    if tagged.contains(&candidate) {
                        continue;
                    }
                    for &face in self.edge_faces(candidate) {
                        if self.face_has_edge(face, previous) {
                            shared_count += 1;
                            first_successor.get_or_insert(candidate);
                            if shared_count >= 3 {
                                break 'successors;
                            }
                        }
                    }
                }
                if shared_count == 1 || (index == 1 && shared_count == 2) {
                    let successor = first_successor
                        .ok_or_else(|| internal_invariant("shared face has a successor"))?;
                    tagged.insert(successor);
                    order.push(successor);
                } else {
                    fast = false;
                    break;
                }
            }
        }
        if fast {
            return Ok(order);
        }

        order.truncate(1);
        tagged.clear();
        tagged.insert(first);
        while order.len() < incident.len() {
            let mut path = order.clone();
            let end = self.edge_order_extend(vertex, &mut path, &mut tagged);
            if end + 1 > order.len() {
                order = path;
                if order.len() == incident.len() {
                    break;
                }
            }
            let next = incident
                .iter()
                .copied()
                .filter(|edge_id| !tagged.contains(edge_id))
                .find(|edge_id| self.edge_face_count(*edge_id) == 1)
                .or_else(|| {
                    incident
                        .iter()
                        .copied()
                        .find(|edge_id| !tagged.contains(edge_id))
                })
                .ok_or_else(|| internal_invariant("remaining incident edge"))?;
            tagged.insert(next);
            order.push(next);
        }

        Ok(order)
    }

    fn shared_face(&self, first: usize, second: usize, vertex: usize) -> Option<usize> {
        let first_key = edge_key(self.geo.edge_order[first].0, self.geo.edge_order[first].1);
        let second_key = edge_key(self.geo.edge_order[second].0, self.geo.edge_order[second].1);
        let mut best = None;
        for &face in self.edge_faces(first) {
            let vertices = &self.geo.polygons[face].0;
            let count = vertices.len();
            for index in 0..count {
                if edge_key(vertices[index], vertices[(index + 1) % count]) != first_key {
                    continue;
                }
                let previous = edge_key(vertices[(index + count - 1) % count], vertices[index]);
                let next = edge_key(vertices[(index + 1) % count], vertices[(index + 2) % count]);
                if (previous == second_key || next == second_key)
                    && (best.is_none() || vertices[index] == vertex)
                {
                    best = Some(face);
                }
                break;
            }
        }
        best
    }

    fn edge_ccw_test(&self, first: usize, second: usize, face: usize) -> i32 {
        let first_key = edge_key(self.geo.edge_order[first].0, self.geo.edge_order[first].1);
        let second_key = edge_key(self.geo.edge_order[second].0, self.geo.edge_order[second].1);
        let vertices = &self.geo.polygons[face].0;
        let count = vertices.len();
        let first_loop = (0..count)
            .find(|&index| edge_key(vertices[index], vertices[(index + 1) % count]) == first_key);
        let second_loop = (0..count)
            .find(|&index| edge_key(vertices[index], vertices[(index + 1) % count]) == second_key);
        match (first_loop, second_loop) {
            (Some(first_loop), Some(second_loop)) if (second_loop + 1) % count == first_loop => 1,
            (Some(_), Some(_)) => -1,
            _ => 0,
        }
    }

    /// Creates Blender's circular `EdgeHalf` order and its fnext/fprev links at one vertex.
    fn construct_edgehalves(&mut self, vertex: usize) -> Result<Option<Vec<usize>>> {
        let affect_vertices = self.p.affect_vertices;
        let incident: Vec<usize> = self.incident_edges[vertex]
            .iter()
            .copied()
            .filter(|edge_id| affect_vertices || self.edge_face_count(*edge_id) > 0)
            .collect();
        if incident.len() < 2
            || (!affect_vertices
                && !incident.iter().any(|edge_id| {
                    let (a, b) = self.geo.edge_order[*edge_id];
                    self.is_selected(a, b)
                }))
        {
            return Ok(None);
        }

        let mut edge_ids = self.bevel_edge_order(vertex, &incident)?;
        let count = edge_ids.len();
        let mut gaps = vec![None; count];
        let mut previous_faces = vec![None; count];
        for index in 0..count {
            let next = (index + 1) % count;
            if gaps[index].is_some() || previous_faces[next].is_some() {
                continue;
            }
            if let Some(face) = self.shared_face(edge_ids[index], edge_ids[next], vertex) {
                gaps[index] = Some(face);
                previous_faces[next] = Some(face);
            }
        }
        let mut ccw_sum = 0_i32;
        for index in 0..count {
            if let Some(face) = gaps[index] {
                ccw_sum += self.edge_ccw_test(edge_ids[index], edge_ids[(index + 1) % count], face);
            }
        }
        if ccw_sum < 0 {
            edge_ids.reverse();
            gaps = reversed_gaps(&gaps);
        }

        let first_id = self.edges.len();
        for (index, edge_id) in edge_ids.iter().copied().enumerate() {
            let (a, b) = self.geo.edge_order[edge_id];
            let other = if a == vertex { b } else { a };
            let is_bev = !affect_vertices && self.is_selected(a, b);
            let fnext = gaps[index];
            let fprev = gaps[(index + count - 1) % count];
            let is_seam = match (fprev, fnext) {
                (Some(_), Some(_)) => self.geo.seam_edges.contains(&edge_key(a, b)),
                _ => true,
            };
            self.edges.push(EdgeHalf {
                vert: vertex,
                other,
                edge_id,
                is_rev: b == vertex,
                is_bev,
                seg: if is_bev { self.p.seg } else { 0 },
                fprev,
                fnext,
                next: first_id + (index + 1) % count,
                prev: first_id + (index + count - 1) % count,
                leftv: None,
                rightv: None,
                offset_l_spec: 0.0,
                offset_r_spec: 0.0,
                offset_l: 0.0,
                offset_r: 0.0,
                is_seam,
                profile_index: 0,
                visited_rpo: false,
            });
        }
        Ok(Some((first_id..first_id + count).collect()))
    }

    fn point(&self, vertex: usize) -> DVec3 {
        self.geo.points[vertex]
    }

    fn face_normal(&self, face: usize) -> DVec3 {
        self.geo.face_normals[face]
    }

    fn bev(&self, vertex: usize) -> Result<&BevVert> {
        self.verts[vertex]
            .as_ref()
            .ok_or_else(|| internal_invariant("beveled vertex"))
    }

    fn bev_mut(&mut self, vertex: usize) -> Result<&mut BevVert> {
        self.verts[vertex]
            .as_mut()
            .ok_or_else(|| internal_invariant("beveled vertex"))
    }

    fn is_selected(&self, a: usize, b: usize) -> bool {
        self.geo.selected_edges.contains(&edge_key(a, b))
    }

    fn key_length(&self, a: usize, b: usize) -> f64 {
        f32_length(f32_sub(self.point(a), self.point(b)))
    }

    /// Length of the edge half's edge.
    fn edge_length(&self, id: usize) -> f64 {
        let e = &self.edges[id];
        self.key_length(e.vert, e.other)
    }

    /// Blender's `edge_face_angle`: the angle between the two faces at an edge half.
    fn edge_face_angle(&self, id: usize) -> f64 {
        let e = &self.edges[id];
        match (e.fprev, e.fnext) {
            (Some(prev), Some(next)) => f32_sub_scalar(
                std::f32::consts::PI as f64,
                angle_normalized_f32(self.face_normal(prev), self.face_normal(next)),
            ),
            _ => 0.0,
        }
    }

    /// Blender's `eh_on_plane`: the faces on either side of the edge half are coplanar.
    fn eh_on_plane(&self, id: usize) -> bool {
        let e = &self.edges[id];
        match (e.fprev, e.fnext) {
            (Some(prev), Some(next)) => {
                let dot = f32_dot(self.face_normal(prev), self.face_normal(next));
                f32_sub_scalar(dot, -1.0).abs() <= EPSILON_BIG
                    || f32_sub_scalar(dot, 1.0).abs() <= EPSILON_BIG
            }
            _ => false,
        }
    }

    /// Blender's `find_edge_half`: the edge half of `vertex` on the undirected edge `edge`.
    fn find_edge_half(&self, vertex: usize, edge: (usize, usize)) -> Option<usize> {
        let bv = self.verts[vertex].as_ref()?;
        bv.edges.iter().copied().find(|id| {
            let e = &self.edges[*id];
            edge_key(e.vert, e.other) == edge
        })
    }

    /// Blender's `find_other_end_edge_half`: the edge half at the far end of `edge`, with the
    /// vertex it belongs to, when that vertex is beveled.
    fn find_other_end(&self, edge: usize) -> Option<(usize, usize)> {
        let e = &self.edges[edge];
        let other = e.other;
        let half = self.find_edge_half(other, edge_key(e.vert, other))?;
        Some((half, other))
    }

    /// Blender's `next_bev`: the first beveled edge half, scanning from the last entry.
    fn next_bev(&self, bv: &BevVert) -> Option<usize> {
        let count = bv.edges.len();
        (0..count)
            .map(|step| bv.edges[(count - 1 + step) % count])
            .find(|id| self.edges[*id].is_bev)
    }

    /// Blender's `is_outside_edge`: the nearer endpoint of the edge `a`-`b` when `co` lies beyond
    /// it, else `None`.
    fn is_outside_edge(&self, a: usize, b: usize, co: DVec3) -> Option<usize> {
        let (pa, pb) = (self.point(a), self.point(b));
        let vector = f32_sub(pb, pa);
        let lenu = f32_length(vector);
        let u = f32_normalize(vector);
        let lambda = f32_dot(u, f32_sub(co, pa));
        if lambda <= -f32_mul_scalar(EPSILON_BIG, lenu) {
            return Some(a);
        }
        if lambda >= f32_mul_scalar(f32_add_scalar(1.0, EPSILON_BIG), lenu) {
            return Some(b);
        }
        None
    }

    /// Blender's `slide_dist`: the point at distance `d` from `vertex` along the edge `a`-`b`,
    /// clamped just short of the far endpoint.
    fn slide_key(&self, a: usize, b: usize, vertex: usize, d: f64) -> DVec3 {
        let other = if vertex == a { b } else { a };
        let v = self.point(vertex);
        let direction = f32_sub(v, self.point(other));
        let len = f32_length(direction);
        let direction = f32_normalize(direction);
        let d = if d > len {
            f64::from(len as f32 - (50.0 * EPSILON_D) as f32)
        } else {
            f64::from(d as f32)
        };
        f32_add(v, f32_mul(direction, -d))
    }

    /// Blender's `slide_dist` for an edge half at its own beveled vertex.
    fn slide_dist(&self, id: usize, d: f64) -> DVec3 {
        let e = &self.edges[id];
        self.slide_key(e.vert, e.other, e.vert, d)
    }

    /// Blender's `edges_angle_kind`: whether the angle at `vertex` between the edge halves `e1`
    /// and `e2` is smaller, larger, or straight.
    pub(super) fn edges_angle_kind(&self, e1: usize, e2: usize, vertex: usize) -> AngleKind {
        let v = self.point(vertex);
        let dir1 = f32_normalize(f32_sub(v, self.point(self.edges[e1].other)));
        let dir2 = f32_normalize(f32_sub(v, self.point(self.edges[e2].other)));
        if nearly_parallel_normalized(dir1, dir2) {
            return AngleKind::Straight;
        }
        let cross = f32_normalize(f32_cross(dir1, dir2));
        let no = if let Some(face) = self.edges[e1].fnext {
            self.face_normal(face)
        } else if let Some(face) = self.edges[e2].fprev {
            self.face_normal(face)
        } else {
            self.vertex_normals[vertex]
        };
        if f32_dot(cross, no) < 0.0 {
            AngleKind::Larger
        } else {
            AngleKind::Smaller
        }
    }

    /// Blender's `point_between_edges`: whether `co`, on `face`, lies between `e1` and `e2`
    /// sweeping CCW around `vertex`.
    fn point_between_edges(
        &self,
        co: DVec3,
        vertex: usize,
        face: usize,
        e1: usize,
        e2: usize,
    ) -> bool {
        let v = self.point(vertex);
        let dir1 = f32_normalize(f32_sub(v, self.point(self.edges[e1].other)));
        let dir2 = f32_normalize(f32_sub(v, self.point(self.edges[e2].other)));
        let dirco = f32_normalize(f32_sub(v, co));
        let fno = self.face_normal(face);
        let mut ang11 = angle_normalized_f32(dir1, dir2);
        let mut ang1co = angle_normalized_f32(dir1, dirco);
        let full_turn = f64::from((std::f64::consts::PI * 2.0) as f32);
        if f32_dot(f32_cross(dir1, dir2), fno) < 0.0 {
            ang11 = f32_sub_scalar(full_turn, ang11);
        }
        if f32_dot(f32_cross(dir1, dirco), fno) < 0.0 {
            ang1co = f32_sub_scalar(full_turn, ang1co);
        }
        f32_sub_scalar(ang11, ang1co) > -EPSILON_ANG
    }

    /// Blender's `edge_edge_angle_less_than_180`: the angle swept CCW from edge `e1` to edge `e2`
    /// on `face` is neither reflex nor straight. The two edges share one endpoint.
    fn edge_edge_angle_less_than_180(
        &self,
        e1: (usize, usize),
        e2: (usize, usize),
        face: usize,
    ) -> bool {
        let (shared, first, second) = if e1.0 == e2.0 {
            (e1.0, e1.1, e2.1)
        } else if e1.0 == e2.1 {
            (e1.0, e1.1, e2.0)
        } else if e1.1 == e2.0 {
            (e1.1, e1.0, e2.1)
        } else {
            (e1.1, e1.0, e2.0)
        };
        let v = self.point(shared);
        let dir1 = f32_sub(self.point(first), v);
        let dir2 = f32_sub(self.point(second), v);
        f32_dot(f32_cross(dir1, dir2), self.face_normal(face)) > 0.0
    }

    /// Blender's `offset_in_plane`: the point offset by the edge half's left or right width, in
    /// the plane with normal `plane_no` (or an arbitrary plane when `None`).
    fn offset_in_plane(&self, id: usize, plane_no: Option<DVec3>, left: bool) -> DVec3 {
        let e = &self.edges[id];
        let v = self.point(e.vert);
        let dir = f32_normalize(f32_sub(self.point(e.other), v));
        let no = match plane_no {
            Some(no) => no,
            None => {
                if dir.x.abs() < dir.y.abs() {
                    DVec3::X
                } else {
                    DVec3::Y
                }
            }
        };
        let fdir = f32_normalize(if left {
            f32_cross(dir, no)
        } else {
            f32_cross(no, dir)
        });
        f32_add(v, f32_mul(fdir, if left { e.offset_l } else { e.offset_r }))
    }

    /// Blender's `project_to_edge`: the point on the edge `a`-`b` nearest the line `co_a`-`co_b`.
    fn project_to_edge(&self, a: usize, b: usize, co_a: DVec3, co_b: DVec3) -> DVec3 {
        let (pa, pb) = (self.point(a), self.point(b));
        let (kind, projco, _) = isect_line_line(pa, pb, co_a, co_b);
        if kind == 0 { pa } else { projco }
    }

    /// Blender's `offset_meet_edge`: where the offset line of `e1` meets the edge `e2` when one of
    /// the two offsets is zero. Returns `(met, meeting point, angle)`.
    fn offset_meet_edge(&self, e1: usize, e2: usize, vertex: usize) -> (bool, DVec3, f64) {
        let v = self.point(vertex);
        let dir1 = f32_normalize(f32_sub(self.point(self.edges[e1].other), v));
        let dir2 = f32_normalize(f32_sub(self.point(self.edges[e2].other), v));
        let ang = angle_normalized_f32(dir1, dir2);
        if ang.abs() < GOOD_ANGLE {
            return (false, DVec3::ZERO, 0.0);
        }
        if f32_dot(f32_cross(dir1, dir2), self.vertex_normals[vertex]) < 0.0 {
            return (
                false,
                DVec3::ZERO,
                f32_sub_scalar(f64::from(2.0_f32 * std::f32::consts::PI), ang),
            );
        }
        if f32_sub_scalar(ang, std::f32::consts::PI as f64).abs() < GOOD_ANGLE {
            return (false, DVec3::ZERO, ang);
        }
        let sinang = f32_sin(ang);
        let meetco = if self.edges[e1].offset_r == 0.0 {
            f32_add(v, f32_mul(dir1, f32_div(self.edges[e2].offset_l, sinang)))
        } else {
            f32_add(v, f32_mul(dir2, f32_div(self.edges[e1].offset_r, sinang)))
        };
        (true, meetco, ang)
    }

    /// Blender's `good_offset_on_edge_between`.
    fn good_offset_on_edge_between(
        &self,
        e1: usize,
        e2: usize,
        emid: usize,
        vertex: usize,
    ) -> bool {
        self.offset_meet_edge(e1, emid, vertex).0 && self.offset_meet_edge(emid, e2, vertex).0
    }

    /// Blender's `offset_on_edge_between`: the boundary point on the unbeveled edge `emid` between
    /// the beveled edge halves `e1` and `e2`. Returns `(placed, point, sinratio)`.
    fn offset_on_edge_between(
        &self,
        e1: usize,
        e2: usize,
        emid: usize,
        vertex: usize,
    ) -> (bool, DVec3, f64) {
        let (ok1, meet1, ang1) = self.offset_meet_edge(e1, emid, vertex);
        let (ok2, meet2, ang2) = self.offset_meet_edge(emid, e2, vertex);
        let sinratio = if ang1 == 0.0 {
            1.0
        } else {
            f32_div(f32_sin(ang2), f32_sin(ang1))
        };
        if matches!(
            self.p.offset_type,
            OffsetType::Percent | OffsetType::Absolute
        ) {
            let v = self.point(vertex);
            let v2 = self.point(self.edges[emid].other);
            let meetco = if self.p.offset_type == OffsetType::Percent {
                let sum = self.edge_weight_of(e1) as f32 + self.edge_weight_of(e2) as f32;
                let wt = f64::from(0.5_f32 * sum);
                f32_lerp(v, v2, f32_div(f32_mul_scalar(wt, self.p.offset), 100.0))
            } else {
                f32_add(v, f32_mul(f32_normalize(f32_sub(v2, v)), self.p.offset))
            };
            return (true, meetco, sinratio);
        }
        if ok1 && ok2 {
            (true, f32_mul(f32_add(meet1, meet2), 0.5), sinratio)
        } else if ok1 {
            (false, meet1, 0.0)
        } else if ok2 {
            (false, meet2, 0.0)
        } else {
            (false, self.slide_dist(emid, self.edges[e1].offset_r), 0.0)
        }
    }

    /// The bevel weight of an edge half: 1 without weights, and 0 for a weighted edge with no
    /// recorded weight (Blender's missing-attribute value).
    fn edge_weight_of(&self, id: usize) -> f64 {
        let e = &self.edges[id];
        match &self.geo.edge_weight {
            None => 1.0,
            Some(weights) => f64::from(
                weights
                    .get(&edge_key(e.vert, e.other))
                    .copied()
                    .unwrap_or(0.0) as f32,
            ),
        }
    }
    pub(super) fn edge_faces(&self, edge_id: usize) -> &[usize] {
        &self.edge_faces_by_id[edge_id]
    }

    fn face_has_edge(&self, face: usize, edge_id: usize) -> bool {
        self.edge_faces(edge_id).contains(&face)
    }

    fn edge_face_count(&self, edge_id: usize) -> usize {
        self.edge_faces(edge_id).len()
    }

    /// The vertices around `vertex` in `face`: `(previous, next)`.
    fn face_neighbors(&self, face: usize, vertex: usize) -> Result<(usize, usize)> {
        let vertices = &self.geo.polygons[face].0;
        let count = vertices.len();
        let position = vertices
            .iter()
            .position(|value| *value == vertex)
            .ok_or_else(|| internal_invariant("vertex in face"))?;
        Ok((
            vertices[(position + count - 1) % count],
            vertices[(position + 1) % count],
        ))
    }

    /// Blender's `offset_meet_lines_percent_or_absolute`: the lines `(l1a, l1b)` and `(l2a, l2b)`
    /// whose intersection is the meeting point for percent and absolute widths.
    fn offset_meet_lines_percent_or_absolute(
        &self,
        e1: usize,
        e2: usize,
        vertex: usize,
    ) -> Result<[DVec3; 4]> {
        let v = self.point(vertex);
        let v1 = self.edges[e1].other;
        let v2 = self.edges[e2].other;
        let plain = [v, self.point(v1), v, self.point(v2)];
        let (Some(f1), Some(f2)) = (self.edges[e1].fnext, self.edges[e2].fprev) else {
            return Ok(plain);
        };
        // e0: the edge after v1 around f1. e4: the edge before v around f1.
        let (_, after_v1) = self.face_neighbors(f1, v1)?;
        let (before_v, _) = self.face_neighbors(f1, vertex)?;
        // e3: the edge before v2 around f2. e5: the edge after v around f2.
        let (before_v2, _) = self.face_neighbors(f2, v2)?;
        let (_, after_v) = self.face_neighbors(f2, vertex)?;
        let e0 = edge_key(v1, after_v1);
        let e3 = edge_key(before_v2, v2);
        let e4 = edge_key(before_v, vertex);
        let e5 = edge_key(vertex, after_v);
        let e1_key = edge_key(vertex, v1);
        let e2_key = edge_key(vertex, v2);
        let no_offsets = !self.edge_edge_angle_less_than_180(e0, e1_key, f1)
            || !self.edge_edge_angle_less_than_180(e1_key, e4, f1)
            || !self.edge_edge_angle_less_than_180(e2_key, e3, f2)
            || !self.edge_edge_angle_less_than_180(e5, e2_key, f1);
        if no_offsets {
            return Ok(plain);
        }
        let (d0, d3, d4, d5) = if self.p.offset_type == OffsetType::Absolute {
            (self.p.offset, self.p.offset, self.p.offset, self.p.offset)
        } else {
            (
                f32_div(
                    f32_mul_scalar(self.p.offset, self.key_length(e0.0, e0.1)),
                    100.0,
                ),
                f32_div(
                    f32_mul_scalar(self.p.offset, self.key_length(e3.0, e3.1)),
                    100.0,
                ),
                f32_div(
                    f32_mul_scalar(self.p.offset, self.key_length(e4.0, e4.1)),
                    100.0,
                ),
                f32_div(
                    f32_mul_scalar(self.p.offset, self.key_length(e5.0, e5.1)),
                    100.0,
                ),
            )
        };
        let e1_wt = self.edge_weight_of(e1);
        let e2_wt = self.edge_weight_of(e2);
        Ok([
            self.slide_key(e4.0, e4.1, vertex, f32_mul_scalar(d4, e1_wt)),
            self.slide_key(e0.0, e0.1, v1, f32_mul_scalar(d0, e1_wt)),
            self.slide_key(e5.0, e5.1, vertex, f32_mul_scalar(d5, e2_wt)),
            self.slide_key(e3.0, e3.1, v2, f32_mul_scalar(d3, e2_wt)),
        ])
    }

    /// Blender's `offset_meet`: where the offset lines of `e1` and `e2` meet at `vertex`. `face`
    /// is the face between them when they share one; `edges_between` says edges lie between them;
    /// `e_in_plane` is an in-plane edge whose neighboring faces the drop must avoid.
    fn offset_meet(
        &self,
        e1: usize,
        e2: usize,
        vertex: usize,
        face: Option<usize>,
        edges_between: bool,
        e_in_plane: Option<usize>,
    ) -> Result<DVec3> {
        let v = self.point(vertex);
        let v1 = self.edges[e1].other;
        let v2 = self.edges[e2].other;
        let mut dir1 = f32_sub(v, self.point(v1));
        let dir2 = f32_sub(self.point(v2), v);
        let (dir1n, dir2p) = if edges_between {
            let e1next = self.edges[e1].next;
            let e2prev = self.edges[e2].prev;
            (
                f32_sub(self.point(self.edges[e1next].other), v),
                f32_sub(v, self.point(self.edges[e2prev].other)),
            )
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };
        let ang = angle_f32(dir1, dir2);
        if ang < EPSILON_ANG {
            let norm_v = if let Some(face) = face {
                self.face_normal(face)
            } else {
                let mut sum = DVec3::ZERO;
                let mut count = 0;
                let mut eloop = e1;
                while eloop != e2 {
                    if let Some(next) = self.edges[eloop].fnext {
                        sum = f32_add(sum, self.face_normal(next));
                        count += 1;
                    }
                    eloop = self.edges[eloop].next;
                }
                if count == 0 {
                    self.vertex_normals[vertex]
                } else {
                    f32_mul(sum, f32_div(1.0, count as f64))
                }
            };
            dir1 = f32_add(dir1, dir2);
            let norm_perp1 = f32_normalize(f32_cross(dir1, norm_v));
            let d = f32_div(
                self.edges[e1].offset_r.max(self.edges[e2].offset_l),
                f32_cos(f32_div(ang, 2.0)),
            );
            return Ok(f32_add(v, f32_mul(norm_perp1, d)));
        }
        if f32_sub_scalar(ang, std::f32::consts::PI as f64).abs() < EPSILON_ANG {
            let d = self.edges[e1].offset_r.max(self.edges[e2].offset_l);
            return Ok(self.slide_dist(e2, d));
        }
        let reference = |face: Option<usize>| match face {
            Some(f) => self.face_normal(f),
            None => self.vertex_normals[vertex],
        };
        let (norm_v1, norm_v2) = if let (Some(f), true) = (face, ang < SMALL_ANG) {
            (self.face_normal(f), self.face_normal(f))
        } else if !edges_between {
            let mut norm_v1 = f32_normalize(f32_cross(dir2, dir1));
            if f32_dot(norm_v1, reference(face)) < 0.0 {
                norm_v1 = -norm_v1;
            }
            (norm_v1, norm_v1)
        } else {
            let mut norm_v1 = f32_normalize(f32_cross(dir1n, dir1));
            if f32_dot(norm_v1, reference(self.edges[e1].fnext)) < 0.0 {
                norm_v1 = -norm_v1;
            }
            let mut norm_v2 = f32_normalize(f32_cross(dir2, dir2p));
            if f32_dot(norm_v2, reference(self.edges[e2].fprev)) < 0.0 {
                norm_v2 = -norm_v2;
            }
            (norm_v1, norm_v2)
        };
        let norm_perp1 = f32_normalize(f32_cross(dir1, norm_v1));
        let norm_perp2 = f32_normalize(f32_cross(dir2, norm_v2));
        let (off1a, off1b, off2a, off2b) = if matches!(
            self.p.offset_type,
            OffsetType::Percent | OffsetType::Absolute
        ) {
            let [a, b, c, d] = self.offset_meet_lines_percent_or_absolute(e1, e2, vertex)?;
            (a, b, c, d)
        } else {
            let off1a = f32_add(v, f32_mul(norm_perp1, self.edges[e1].offset_r));
            let off2a = f32_add(v, f32_mul(norm_perp2, self.edges[e2].offset_l));
            (off1a, f32_add(off1a, dir1), off2a, f32_add(off2a, dir2))
        };
        let (kind, meet, isect2) = isect_line_line(off1a, off1b, off2a, off2b);
        if kind == 0 {
            return Ok(off1a);
        }
        let mut meetco = meet;
        if self.edges[e1].offset_r == 0.0
            && let Some(closer) = self.is_outside_edge(v1, vertex, meetco)
        {
            meetco = self.point(closer);
        }
        if self.edges[e2].offset_l == 0.0
            && let Some(closer) = self.is_outside_edge(vertex, v2, meetco)
        {
            meetco = self.point(closer);
        }
        if edges_between && self.edges[e1].offset_r > 0.0 && self.edges[e2].offset_l > 0.0 {
            if kind == 2 {
                meetco = f32_mul(f32_add(meetco, isect2), 0.5);
            }
            let mut e = e1;
            while e != e2 {
                if let Some(fnext) = self.edges[e].fnext {
                    let fno = self.face_normal(fnext);
                    let dropco = f32_sub(meetco, f32_mul(fno, f32_dot(fno, f32_sub(meetco, v))));
                    let skip = e_in_plane.is_some_and(|in_plane| {
                        self.edges[in_plane].fnext.is_some_and(|other| {
                            let face_angle = angle_f32(fno, self.face_normal(other));
                            face_angle.abs() < SMALL_ANG
                                || f32_sub_scalar(face_angle, std::f32::consts::PI as f64).abs()
                                    < SMALL_ANG
                        })
                    });
                    if !skip
                        && self.point_between_edges(dropco, vertex, fnext, e, self.edges[e].next)
                    {
                        meetco = dropco;
                        break;
                    }
                }
                e = self.edges[e].next;
            }
        }
        Ok(meetco)
    }

    fn construct_vert(&mut self, vertex: usize) -> Result<()> {
        let Some(ids) = self.construct_edgehalves(vertex)? else {
            return Ok(());
        };
        let affect_vertices = self.p.affect_vertices;
        let selcount = ids.iter().filter(|id| self.edges[**id].is_bev).count();
        let mut offset = self.p.offset;
        if affect_vertices && let Some(weights) = &self.geo.vertex_weight {
            offset = f32_mul_scalar(offset, weights.get(&vertex).copied().unwrap_or(0.0));
        }
        self.verts[vertex] = Some(BevVert {
            edges: ids.clone(),
            selcount,
            offset,
            ring: Vec::new(),
            mesh_kind: MeshKind::None,
            any_seam: false,
        });
        self.set_edge_offsets(vertex, &ids)
    }

    /// Blender's width specs in `bevel_vert_construct`: how the user width becomes the left and
    /// right offsets of each edge half, by offset type.
    fn set_edge_offsets(&mut self, vertex: usize, ids: &[usize]) -> Result<()> {
        let v = self.point(vertex);
        let bv_offset = self.bev(vertex)?.offset;
        let vertices = self.p.affect_vertices;
        let mut vert_axis = DVec3::ZERO;
        if vertices && matches!(self.p.offset_type, OffsetType::Width | OffsetType::Depth) {
            for id in ids {
                vert_axis = f32_add(
                    vert_axis,
                    f32_normalize(f32_sub(v, self.point(self.edges[*id].other))),
                );
            }
        }
        for &id in ids {
            let (left, right) = if self.edges[id].is_bev {
                let (left, right) = match self.p.offset_type {
                    OffsetType::Offset | OffsetType::Absolute => (self.p.offset, self.p.offset),
                    OffsetType::Width => {
                        let z =
                            f32_mul_scalar(2.0, f32_sin(f32_div(self.edge_face_angle(id), 2.0)))
                                .abs();
                        let amount = if z < EPSILON {
                            f32_mul_scalar(0.01, self.p.offset)
                        } else {
                            f32_div(self.p.offset, z)
                        };
                        (amount, amount)
                    }
                    OffsetType::Depth => {
                        let z = f32_cos(f32_div(self.edge_face_angle(id), 2.0)).abs();
                        let amount = if z < EPSILON {
                            f32_mul_scalar(0.01, self.p.offset)
                        } else {
                            f32_div(self.p.offset, z)
                        };
                        (amount, amount)
                    }
                    OffsetType::Percent => {
                        let previous = self.edges[id].prev;
                        let next = self.edges[id].next;
                        (
                            f32_div(
                                f32_mul_scalar(self.edge_length(previous), self.p.offset),
                                100.0,
                            ),
                            f32_div(f32_mul_scalar(self.edge_length(next), self.p.offset), 100.0),
                        )
                    }
                };
                if self.geo.edge_weight.is_some() {
                    let weight = self.edge_weight_of(id);
                    (f32_mul_scalar(left, weight), f32_mul_scalar(right, weight))
                } else {
                    (left, right)
                }
            } else if vertices {
                let edge_dir = f32_sub(v, self.point(self.edges[id].other));
                let l = match self.p.offset_type {
                    OffsetType::Offset | OffsetType::Absolute => bv_offset,
                    OffsetType::Width => {
                        let z = f32_mul_scalar(2.0, f32_sin(angle(vert_axis, edge_dir))).abs();
                        if z < EPSILON {
                            f32_mul_scalar(0.01, self.p.offset)
                        } else {
                            f32_div(self.p.offset, z)
                        }
                    }
                    OffsetType::Depth => {
                        let z = f32_cos(angle(vert_axis, edge_dir)).abs();
                        if z < EPSILON {
                            f32_mul_scalar(0.01, self.p.offset)
                        } else {
                            f32_div(self.p.offset, z)
                        }
                    }
                    OffsetType::Percent => {
                        f32_div(f32_mul_scalar(self.edge_length(id), bv_offset), 100.0)
                    }
                };
                (l, l)
            } else {
                (0.0, 0.0)
            };
            let e = &mut self.edges[id];
            e.offset_l_spec = f64::from(left as f32);
            e.offset_r_spec = f64::from(right as f32);
            e.offset_l = e.offset_l_spec;
            e.offset_r = e.offset_r_spec;
        }
        Ok(())
    }

    /// Appends a boundary vertex at `co` to `ring` and returns its arena id.
    fn add_bound(&mut self, ring: &mut Vec<usize>, co: DVec3) -> usize {
        let id = self.bounds.len();
        self.bounds.push(BoundVert::new(co));
        ring.push(id);
        id
    }
    fn add_or_update_bound(
        &mut self,
        construct: bool,
        ring: &mut Vec<usize>,
        existing: Option<usize>,
        co: DVec3,
    ) -> Result<usize> {
        if construct {
            Ok(self.add_bound(ring, co))
        } else {
            let bound = existing.ok_or_else(|| internal_invariant("existing boundary vertex"))?;
            self.bounds[bound].co = co;
            Ok(bound)
        }
    }

    /// Links a boundary ring's `next` and `prev` pointers in order.
    fn link_ring(&mut self, ring: &[usize]) {
        let count = ring.len();
        for index in 0..count {
            let id = ring[index];
            self.bounds[id].next = ring[(index + 1) % count];
            self.bounds[id].prev = ring[(index + count - 1) % count];
        }
    }

    /// Records the finished boundary ring of `vertex` and its vertex-mesh kind.
    fn store_ring(&mut self, vertex: usize, ring: Vec<usize>, mesh_kind: MeshKind) -> Result<()> {
        self.bev_mut(vertex)?.ring = ring;
        self.bev_mut(vertex)?.mesh_kind = mesh_kind;
        self.set_bound_vert_seams(vertex)
    }

    /// Blender's `build_boundary`: the boundary ring of the beveled vertex `vertex`, from its edge
    /// halves and current widths.
    fn build_boundary(&mut self, vertex: usize, construct: bool) -> Result<()> {
        let bv = self.bev(vertex)?.clone();
        if bv.edges.len() <= 1 {
            return Ok(());
        }
        if self.p.affect_vertices {
            return self.build_boundary_vertex_only(vertex, construct);
        }
        let Some(efirst) = self.next_bev(&bv) else {
            return Ok(());
        };
        if bv.selcount == 1 {
            return self.build_boundary_terminal_edge(vertex, efirst, construct);
        }

        let miter_outer = if bv.selcount >= 3 {
            self.p.miter_outer
        } else {
            MiterKind::Sharp
        };
        let miter_inner = self.p.miter_inner;
        let mut ring = Vec::new();
        let mut emiter = None;
        let mut e = efirst;
        loop {
            let mut in_plane = 0;
            let mut not_in_plane = 0;
            let mut enip = None;
            let mut eip = None;
            let mut e2 = self.edges[e].next;
            while !self.edges[e2].is_bev {
                if self.eh_on_plane(e2) {
                    in_plane += 1;
                    eip = Some(e2);
                } else {
                    not_in_plane += 1;
                    enip = Some(e2);
                }
                e2 = self.edges[e2].next;
            }

            let mut eon = None;
            let mut sinratio = 1.0;
            let co = if in_plane == 0 && not_in_plane == 0 {
                self.offset_meet(e, e2, vertex, self.edges[e].fnext, false, None)?
            } else if not_in_plane > 0 {
                match enip {
                    Some(mid)
                        if self.p.loop_slide
                            && not_in_plane == 1
                            && self.good_offset_on_edge_between(e, e2, mid, vertex) =>
                    {
                        let (placed, co, ratio) = self.offset_on_edge_between(e, e2, mid, vertex);
                        if placed {
                            eon = Some(mid);
                            sinratio = ratio;
                        }
                        co
                    }
                    _ => self.offset_meet(e, e2, vertex, None, true, eip)?,
                }
            } else {
                match eip {
                    Some(mid)
                        if self.p.loop_slide
                            && in_plane == 1
                            && self.good_offset_on_edge_between(e, e2, mid, vertex) =>
                    {
                        let (placed, co, ratio) = self.offset_on_edge_between(e, e2, mid, vertex);
                        if placed {
                            eon = Some(mid);
                            sinratio = ratio;
                        }
                        co
                    }
                    _ => self.offset_meet(e, e2, vertex, self.edges[e].fnext, false, None)?,
                }
            };

            let v = if construct {
                self.add_bound(&mut ring, co)
            } else {
                let v = self.edges[e]
                    .rightv
                    .ok_or_else(|| internal_invariant("existing boundary vertex"))?;
                self.bounds[v].co = co;
                v
            };
            if construct {
                let bound = &mut self.bounds[v];
                bound.efirst = Some(e);
                bound.elast = Some(e2);
                bound.ebev = Some(e2);
                bound.eon = eon;
                if eon.is_some() {
                    bound.sinratio = sinratio;
                }
                self.edges[e].rightv = Some(v);
                self.edges[e2].leftv = Some(v);
                let mut e3 = self.edges[e].next;
                while e3 != e2 {
                    self.edges[e3].leftv = Some(v);
                    self.edges[e3].rightv = Some(v);
                    e3 = self.edges[e3].next;
                }
            }

            let ang_kind = self.edges_angle_kind(e, e2, vertex);
            let outer = miter_outer != MiterKind::Sharp
                && emiter.is_none()
                && ang_kind == AngleKind::Larger;
            let inner = miter_inner != MiterKind::Sharp && ang_kind == AngleKind::Smaller;
            if outer || inner {
                if ang_kind == AngleKind::Larger {
                    emiter = Some(e);
                }
                let patch = ang_kind == AngleKind::Larger && miter_outer == MiterKind::Patch;
                if construct {
                    self.bounds[v].ebev = None;
                    let v2 = patch.then(|| self.add_bound(&mut ring, co));
                    let v3 = self.add_bound(&mut ring, co);
                    {
                        let bound = &mut self.bounds[v3];
                        bound.ebev = Some(e2);
                        bound.efirst = Some(e2);
                        bound.elast = Some(e2);
                        bound.eon = None;
                    }
                    self.edges[e2].leftv = Some(v3);
                    if let Some(v2) = v2 {
                        self.bounds[v].is_patch_start = true;
                        self.bounds[v2].eon = self.bounds[v].eon;
                        self.bounds[v2].sinratio = self.bounds[v].sinratio;
                        self.bounds[v2].ebev = None;
                        self.bounds[v].eon = None;
                        self.bounds[v].sinratio = 1.0;
                        self.bounds[v].elast = Some(e);
                        if self.edges[e].next == e2 {
                            self.bounds[v2].efirst = None;
                            self.bounds[v2].elast = None;
                        } else {
                            let mut e3 = self.edges[e].next;
                            self.bounds[v2].efirst = Some(e3);
                            while e3 != e2 {
                                self.edges[e3].leftv = Some(v2);
                                self.edges[e3].rightv = Some(v2);
                                self.bounds[v2].elast = Some(e3);
                                e3 = self.edges[e3].next;
                            }
                        }
                    } else {
                        self.bounds[v].is_arc_start = true;
                        self.bounds[v].profile.middle = co;
                        if self.edges[e].next == e2 {
                            self.bounds[v].elast = self.bounds[v].efirst;
                        } else {
                            let between = in_plane + not_in_plane;
                            let half = between / 2;
                            let odd = between % 2 == 1;
                            let mut index = 0;
                            let mut e3 = self.edges[e].next;
                            while e3 != e2 {
                                self.bounds[v].elast = Some(e3);
                                self.edges[e3].profile_index = if index < half {
                                    0
                                } else if odd && index == half {
                                    self.p.seg / 2
                                } else {
                                    self.p.seg
                                };
                                index += 1;
                                e3 = self.edges[e3].next;
                            }
                        }
                    }
                } else {
                    let v3 = if patch {
                        self.bounds[self.bounds[v].next].next
                    } else {
                        self.bounds[v].next
                    };
                    self.bounds[v].co = co;
                    if patch {
                        let v2 = self.bounds[v].next;
                        self.bounds[v2].co = co;
                    }
                    self.bounds[v3].co = co;
                }
            }
            e = e2;
            if e == efirst {
                break;
            }
        }

        if construct {
            self.link_ring(&ring);
            self.bev_mut(vertex)?.ring.clone_from(&ring);
        }
        if miter_inner != MiterKind::Sharp {
            self.adjust_miter_inner_coords(vertex, emiter)?;
        }
        if let Some(emiter) = emiter {
            self.adjust_miter_coords(vertex, emiter)?;
        }
        if construct {
            let mesh_kind = if ring.len() == 2 {
                MeshKind::None
            } else if self.edges[efirst].seg == 1 {
                MeshKind::Poly
            } else {
                match self.p.vmesh_method {
                    VmeshMethod::Adj => MeshKind::Adj,
                    VmeshMethod::Cutoff => MeshKind::Cutoff,
                }
            };
            self.store_ring(vertex, ring, mesh_kind)?;
        }
        Ok(())
    }

    /// Blender's `build_boundary_vertex_only`: the boundary of a VERTICES bevel, one point per edge
    /// half, slid along that edge by its width.
    fn build_boundary_vertex_only(&mut self, vertex: usize, construct: bool) -> Result<()> {
        let efirst = self.bev(vertex)?.edges[0];
        let mut ring = Vec::new();
        let mut e = efirst;
        loop {
            let co = self.slide_dist(e, self.edges[e].offset_l);
            if construct {
                let bound = self.add_bound(&mut ring, co);
                self.bounds[bound].efirst = Some(e);
                self.bounds[bound].elast = Some(e);
                self.edges[e].leftv = Some(bound);
                self.edges[e].rightv = Some(bound);
            } else {
                let bound = self.edges[e]
                    .leftv
                    .ok_or_else(|| internal_invariant("existing vertex boundary"))?;
                self.bounds[bound].co = co;
            }
            e = self.edges[e].next;
            if e == efirst {
                break;
            }
        }
        if construct {
            self.link_ring(&ring);
            let mesh_kind = if ring.len() == 2 {
                MeshKind::None
            } else if self.p.seg == 1 {
                MeshKind::Poly
            } else {
                MeshKind::Adj
            };
            self.store_ring(vertex, ring, mesh_kind)?;
        }
        Ok(())
    }

    /// Blender's `build_boundary_terminal_edge`: the boundary of a vertex with one beveled edge.
    fn build_boundary_terminal_edge(
        &mut self,
        vertex: usize,
        efirst: usize,
        construct: bool,
    ) -> Result<()> {
        let edgecount = self.bev(vertex)?.edges.len();
        let mut ring = Vec::new();
        let mut e = efirst;
        if edgecount == 2 {
            let no = self.edges[e]
                .fprev
                .or(self.edges[e].fnext)
                .map(|f| self.face_normal(f));
            let co = self.offset_in_plane(e, no, true);
            let existing = self.edges[e].leftv;
            let left = self.add_or_update_bound(construct, &mut ring, existing, co)?;
            if construct {
                self.bounds[left].efirst = Some(e);
                self.bounds[left].elast = Some(e);
                self.bounds[left].ebev = Some(e);
                self.edges[e].leftv = Some(left);
            }
            let no = self.edges[e]
                .fnext
                .or(self.edges[e].fprev)
                .map(|f| self.face_normal(f));
            let co = self.offset_in_plane(e, no, false);
            let existing = self.edges[e].rightv;
            let right = self.add_or_update_bound(construct, &mut ring, existing, co)?;
            if construct {
                self.bounds[right].efirst = Some(e);
                self.bounds[right].elast = Some(e);
                self.edges[e].rightv = Some(right);
            }
            let next = self.edges[e].next;
            let co = self.slide_dist(next, self.edges[e].offset_l);
            let existing = self.edges[next].leftv;
            let far = self.add_or_update_bound(construct, &mut ring, existing, co)?;
            if construct {
                self.bounds[far].efirst = Some(next);
                self.bounds[far].elast = Some(next);
                self.edges[next].leftv = Some(far);
                self.edges[next].rightv = Some(far);
            }
        } else {
            let leg_slide = matches!(
                self.p.offset_type,
                OffsetType::Percent | OffsetType::Absolute
            );
            let prev = self.edges[e].prev;
            let co = if leg_slide {
                self.slide_dist(prev, self.edges[e].offset_l)
            } else {
                self.offset_meet(prev, e, vertex, self.edges[e].fprev, false, None)?
            };
            let existing = self.edges[e].leftv;
            let left = self.add_or_update_bound(construct, &mut ring, existing, co)?;
            if construct {
                self.bounds[left].efirst = Some(prev);
                self.bounds[left].elast = Some(e);
                self.bounds[left].ebev = Some(e);
                self.edges[e].leftv = Some(left);
                self.edges[prev].leftv = Some(left);
                self.edges[prev].rightv = Some(left);
            }
            let beveled = e;
            e = self.edges[e].next;
            let co = if leg_slide {
                self.slide_dist(e, self.edges[beveled].offset_r)
            } else {
                self.offset_meet(beveled, e, vertex, self.edges[e].fprev, false, None)?
            };
            let existing = self.edges[e].leftv;
            let second = self.add_or_update_bound(construct, &mut ring, existing, co)?;
            if construct {
                self.bounds[second].efirst = Some(beveled);
                self.bounds[second].elast = Some(e);
                self.edges[e].leftv = Some(second);
                self.edges[e].rightv = Some(second);
                self.edges[beveled].rightv = Some(second);
            }
            let mut d = self.edges[efirst].offset_l_spec;
            if self.p.custom.is_some() || self.p.profile < 0.25 {
                d = f32_mul_scalar(d, f64::from(2.0_f32.sqrt()));
            }
            e = self.edges[e].next;
            while self.edges[e].next != efirst {
                let co = self.slide_dist(e, d);
                let existing = self.edges[e].leftv;
                let id = self.add_or_update_bound(construct, &mut ring, existing, co)?;
                if construct {
                    self.bounds[id].efirst = Some(e);
                    self.bounds[id].elast = Some(e);
                    self.edges[e].leftv = Some(id);
                    self.edges[e].rightv = Some(id);
                }
                e = self.edges[e].next;
            }
        }
        if construct {
            self.link_ring(&ring);
        }
        if edgecount >= 3 {
            let bndv = if construct {
                ring[0]
            } else {
                self.bev(vertex)?.ring[0]
            };
            self.set_profile_params(vertex, bndv)?;
            self.move_profile_plane(bndv, self.point(vertex));
        }
        if construct {
            let mesh_kind = if ring.len() == 2 && edgecount == 3 {
                MeshKind::None
            } else if ring.len() == 3 {
                let mut use_tri_fan = true;
                if self.p.custom.is_some() {
                    let left_id = self.edges[efirst]
                        .leftv
                        .ok_or_else(|| internal_invariant("leftv"))?;
                    let left = self.bounds[left_id].profile.clone();
                    let right_id = self.edges[efirst]
                        .rightv
                        .ok_or_else(|| internal_invariant("rightv"))?;
                    let right_next = self.bounds[self.bounds[right_id].next].co;
                    if dist_squared_to_plane(right_next, left.plane_co, left.plane_no) < EPSILON_BIG
                    {
                        use_tri_fan = false;
                    }
                }
                if use_tri_fan {
                    MeshKind::TriFan
                } else {
                    MeshKind::Poly
                }
            } else {
                MeshKind::Poly
            };
            self.store_ring(vertex, ring, mesh_kind)?;
        }
        Ok(())
    }

    /// Blender's `adjust_miter_coords`: moves the outer miter's boundary points onto the lines
    /// through the miter edge, offset by the miter distance.
    fn adjust_miter_coords(&mut self, vertex: usize, emiter: usize) -> Result<()> {
        let v1 = self.edges[emiter]
            .rightv
            .ok_or_else(|| internal_invariant("miter right vertex"))?;
        let v3 = if self.p.miter_outer == MiterKind::Patch {
            let v2 = self.bounds[v1].next;
            self.bounds[v2].next
        } else {
            self.bounds[v1].next
        };
        let v1prev = self.bounds[v1].prev;
        let v3next = self.bounds[v3].next;
        let co2 = self.bounds[v1].co;
        if self.bounds[v1].is_arc_start {
            self.bounds[v1].profile.middle = co2;
        }
        let v = self.point(vertex);
        let half_segments = f64::from(self.p.seg as f32 / 2.0_f32);
        let d = f32_div(self.p.offset, half_segments);
        let edge_dir = f32_normalize(f32_sub(v, self.point(self.edges[emiter].other)));
        let line_p = f32_add(co2, f32_mul(edge_dir, d));
        let co1 = isect_line_plane(co2, line_p, self.bounds[v1prev].co, edge_dir).unwrap_or(line_p);
        self.bounds[v1].co = co1;
        // Blender leaves co3 unset if the plane intersection fails; line_p is the defined fallback.
        let emiter_other = self.bounds[v3]
            .elast
            .ok_or_else(|| internal_invariant("miter other edge"))?;
        let edge_dir = f32_normalize(f32_sub(v, self.point(self.edges[emiter_other].other)));
        let line_p = f32_add(co2, f32_mul(edge_dir, d));
        let co3 = isect_line_plane(co2, line_p, self.bounds[v3next].co, edge_dir).unwrap_or(line_p);
        self.bounds[v3].co = co3;
        Ok(())
    }

    /// Blender's `adjust_miter_inner_coords`: moves the inner miter's arc boundary points along
    /// their edges by `spread`.
    fn adjust_miter_inner_coords(&mut self, vertex: usize, emiter: Option<usize>) -> Result<()> {
        let ring = self.bev(vertex)?.ring.clone();
        let start = ring[0];
        let v_point = self.point(vertex);
        let mut v = start;
        loop {
            if self.bounds[v].is_arc_start {
                let v3 = self.bounds[v].next;
                let co = self.bounds[v].co;
                if let Some(e) = self.bounds[v].efirst.filter(|e| Some(*e) != emiter) {
                    let edge_dir = f32_normalize(f32_sub(self.point(self.edges[e].other), v_point));
                    self.bounds[v].co = f32_add(co, f32_mul(edge_dir, self.p.spread));
                    if let Some(e3) = self.bounds[v3].elast {
                        let edge_dir =
                            f32_normalize(f32_sub(self.point(self.edges[e3].other), v_point));
                        self.bounds[v3].co = f32_add(co, f32_mul(edge_dir, self.p.spread));
                    }
                }
                v = self.bounds[v3].next;
            } else {
                v = self.bounds[v].next;
            }
            if v == start {
                break;
            }
        }
        Ok(())
    }

    /// Blender's `set_profile_params`: the profile of boundary vertex `bndv`, from its beveled edge
    /// (if it has one) or from the vertex.
    fn set_profile_params(&mut self, vertex: usize, bndv: usize) -> Result<()> {
        let mut do_linear_interp = true;
        let edge = self.bounds[bndv].ebev;
        let start = self.bounds[bndv].co;
        let end = self.bounds[self.bounds[bndv].next].co;
        let mut pro = self.bounds[bndv].profile.clone();
        let v = self.point(vertex);
        if let Some(e) = edge {
            do_linear_interp = false;
            pro.super_r = self.p.pro_super_r;
            let other = self.edges[e].other;
            pro.proj_dir = f32_normalize(f32_sub(v, self.point(other)));
            let (first, second) = self.geo.edge_order[self.edges[e].edge_id];
            pro.middle = self.project_to_edge(first, second, start, end);
            pro.start = start;
            pro.end = end;
            let mut d1 = f32_normalize(f32_sub(pro.middle, start));
            let mut d2 = f32_normalize(f32_sub(pro.middle, end));
            pro.plane_no = f32_normalize(f32_cross(d1, d2));
            if nearly_parallel(d1, d2) {
                pro.middle = v;
                let prev_edge = self.edges[e].prev;
                let next_edge = self.edges[e].next;
                if self.edges[prev_edge].is_bev
                    && self.edges[next_edge].is_bev
                    && self.bev(vertex)?.selcount >= 3
                {
                    let (prev_a, prev_b) = self.geo.edge_order[self.edges[prev_edge].edge_id];
                    let (next_a, next_b) = self.geo.edge_order[self.edges[next_edge].edge_id];
                    let d3 = f32_normalize(f32_sub(self.point(prev_a), self.point(prev_b)));
                    let d4 = f32_normalize(f32_sub(self.point(next_a), self.point(next_b)));
                    if nearly_parallel(d3, d4) {
                        pro.middle = f32_mul(f32_add(start, end), 0.5);
                        do_linear_interp = true;
                    } else {
                        let co3 = f32_add(start, d3);
                        let co4 = f32_add(end, d4);
                        let (kind, meetco, _) = isect_line_line(start, co3, end, co4);
                        if kind != 0 {
                            pro.middle = meetco;
                        } else {
                            pro.middle = f32_mul(f32_add(start, end), 0.5);
                            do_linear_interp = true;
                        }
                    }
                }
                pro.end = end;
                d1 = f32_normalize(f32_sub(pro.middle, start));
                d2 = f32_normalize(f32_sub(pro.middle, end));
                pro.plane_no = f32_normalize(f32_cross(d1, d2));
                if nearly_parallel(d1, d2) {
                    do_linear_interp = true;
                } else {
                    pro.proj_dir = pro.plane_no;
                }
            }
            pro.plane_co = start;
        } else if self.bounds[bndv].is_arc_start {
            pro.start = start;
            pro.end = end;
            pro.super_r = PRO_CIRCLE_R;
            pro.plane_co = DVec3::ZERO;
            pro.plane_no = DVec3::ZERO;
            pro.proj_dir = DVec3::ZERO;
            do_linear_interp = false;
        } else if self.p.affect_vertices {
            pro.start = start;
            pro.middle = v;
            pro.end = end;
            pro.super_r = self.p.pro_super_r;
            pro.plane_co = DVec3::ZERO;
            pro.plane_no = DVec3::ZERO;
            pro.proj_dir = DVec3::ZERO;
            do_linear_interp = false;
        }
        if do_linear_interp {
            pro.super_r = PRO_LINE_R;
            pro.start = start;
            pro.end = end;
            pro.middle = f32_mul(f32_add(start, end), 0.5);
            pro.plane_co = DVec3::ZERO;
            pro.plane_no = DVec3::ZERO;
            pro.proj_dir = DVec3::ZERO;
        }
        self.bounds[bndv].profile = pro;
        Ok(())
    }

    /// Blender's `move_profile_plane`: moves a terminal boundary vertex's profile plane to the
    /// plane through its start, end, and the beveled vertex, when that plane is well defined.
    fn move_profile_plane(&mut self, bndv: usize, bmvert: DVec3) {
        let pro = self.bounds[bndv].profile.clone();
        if pro.proj_dir == DVec3::ZERO {
            return;
        }
        let d1 = f32_normalize(f32_sub(bmvert, pro.start));
        let d2 = f32_normalize(f32_sub(bmvert, pro.end));
        let mut no = f32_cross(d1, d2);
        let mut no2 = f32_cross(d1, pro.proj_dir);
        let mut no3 = f32_cross(d2, pro.proj_dir);
        if f32_length(no) > EPSILON_BIG
            && f32_length(no2) > EPSILON_BIG
            && f32_length(no3) > EPSILON_BIG
        {
            no = f32_normalize(no);
            no2 = f32_normalize(no2);
            no3 = f32_normalize(no3);
            let dot2 = f32_dot(no, no2);
            let dot3 = f32_dot(no, no3);
            let dot_limit = f32_sub_scalar(1.0, EPSILON_BIG);
            if dot2.abs() < dot_limit && dot3.abs() < dot_limit {
                self.bounds[bndv].profile.plane_no = no;
            }
        }
        self.bounds[bndv].profile.special_params = true;
    }

    /// Blender's `calculate_vm_profiles`: the profile of every boundary vertex of `vertex`.
    fn calculate_vm_profiles(&mut self, vertex: usize) -> Result<()> {
        let ring = self.bev(vertex)?.ring.clone();
        for bndv in ring {
            if !self.bounds[bndv].profile.special_params {
                self.set_profile_params(vertex, bndv)?;
            }
            let mut miter_profile = false;
            let mut reverse_profile = false;
            if self.p.custom.is_some() {
                miter_profile = self.bounds[bndv].is_arc_start || self.bounds[bndv].is_patch_start;
                reverse_profile = !self.bounds[bndv].is_profile_start && !miter_profile;
            }
            self.calculate_profile(bndv, reverse_profile, miter_profile);
        }
        Ok(())
    }

    /// Blender's `calculate_profile`: the profile points of one boundary vertex, for `seg` and
    /// `seg_2` segments.
    fn calculate_profile(&mut self, bndv: usize, reversed: bool, miter: bool) {
        if self.p.seg == 1 {
            return;
        }
        let mut pro = self.bounds[bndv].profile.clone();
        // A superellipse profile that is linear is not mapped; everything else maps its unit
        // profile into the profile plane.
        let map = if self.p.custom.is_none() && pro.super_r == PRO_LINE_R {
            None
        } else {
            unit_square_map(pro.start, pro.middle, pro.end)
        };
        if self.p.vmesh_method == VmeshMethod::Cutoff
            && let Some([c0, c1, _]) = map
        {
            // The profile's height is the distance between its two unit-square corners.
            pro.height = (c0 + c1).length();
        }
        let spacing = if miter {
            self.pro_spacing_miter.as_ref().unwrap_or(&self.pro_spacing)
        } else {
            &self.pro_spacing
        };
        let prof_co = profile_points(
            &pro,
            map,
            reversed,
            self.p.seg,
            &spacing.xvals,
            &spacing.yvals,
        );
        let need_2 = self.p.seg != self.pro_spacing.seg_2;
        pro.prof_co_2 = if need_2 {
            profile_points(
                &pro,
                map,
                reversed,
                spacing.seg_2,
                &spacing.xvals_2,
                &spacing.yvals_2,
            )
        } else {
            prof_co.clone()
        };
        pro.prof_co = prof_co;
        self.bounds[bndv].profile = pro;
    }

    /// Blender's `geometry_collide_offset`: the largest offset at which the edge half `eb` does not
    /// collide with its neighbors.
    fn geometry_collide_offset(&self, eb: usize) -> f64 {
        let no_collide_offset = f32_add_scalar(self.p.offset, 1.0e6);
        if self.p.offset == 0.0 {
            return no_collide_offset;
        }
        let kb = self.edges[eb].offset_l_spec;
        let ea = self.edges[eb].next;
        let ka = self.edges[ea].offset_r_spec;
        // Blender: with `is_rev`, vb is the stored v2 and vc the stored v1; otherwise the reverse.
        let (vb, mut vc) = {
            let (first, second) = self.geo.edge_order[self.edges[eb].edge_id];
            if self.edges[eb].is_rev {
                (second, first)
            } else {
                (first, second)
            }
        };
        let va = self.edges[ea].other;
        let ebother = self.find_other_end(eb);
        if matches!(
            self.p.offset_type,
            OffsetType::Percent | OffsetType::Absolute
        ) {
            if let Some((other, _)) = ebother {
                let ebprev = self.edges[other].prev;
                if self.edges[ea].is_bev && self.edges[ebprev].is_bev {
                    if self.p.offset_type == OffsetType::Percent {
                        return 50.0;
                    }
                    let blen = self.edge_length(eb);
                    let half_length = f32_div(blen, 2.0);
                    return if self.p.offset > half_length {
                        half_length
                    } else {
                        blen
                    };
                }
            }
            return no_collide_offset;
        }
        let (kc, ec_key, vd) = if let Some((other, far)) = ebother {
            // Blender: an opposite beveled endpoint overrides vc with its vertex.
            vc = far;
            let ec = self.edges[other].prev;
            let vd = self.edges[ec].other;
            (self.edges[ec].offset_l_spec, Some(edge_key(vc, vd)), vd)
        } else {
            let Some(face) = self.edges[eb].fnext else {
                return no_collide_offset;
            };
            let face_vertices = &self.geo.polygons[face].0;
            let Some(i) = loop_index(face_vertices, vb, vc) else {
                return no_collide_offset;
            };
            let count = face_vertices.len();
            let next = face_vertices[(i + 1) % count];
            let vd = if next == vc {
                face_vertices[(i + 2) % count]
            } else if face_vertices[i] == vc {
                face_vertices[(i + count - 1) % count]
            } else {
                return no_collide_offset;
            };
            (0.0, None, vd)
        };
        if edge_key(vb, va) == edge_key(vb, vc) || ec_key == Some(edge_key(vb, vc)) {
            return no_collide_offset;
        }
        let th1 = angle3(self.point(va), self.point(vb), self.point(vc));
        let th2 = angle3(self.point(vb), self.point(vc), self.point(vd));
        let (sin1, sin2, cos1, cos2) = (f32_sin(th1), f32_sin(th2), f32_cos(th1), f32_cos(th2));
        let safe_divide = |a: f64, b: f64| if b == 0.0 { 0.0 } else { f32_div(a, b) };
        let projected = f32_add_scalar(
            safe_divide(f32_add_scalar(ka, f32_mul_scalar(cos1, kb)), sin1),
            safe_divide(f32_add_scalar(kc, f32_mul_scalar(cos2, kb)), sin2),
        );
        let mut limit = no_collide_offset;
        if projected > EPSILON {
            let candidate = f32_mul_scalar(self.p.offset, f32_div(self.edge_length(eb), projected));
            if candidate > EPSILON {
                limit = candidate;
            }
        }
        if kb > f64::from(f32::EPSILON) && (ka == 0.0 || kc == 0.0) {
            let kb = f32_div(self.p.offset, kb);
            let Some(face) = self.edges[eb].fnext else {
                return limit;
            };
            let face_vertices = &self.geo.polygons[face].0;
            if ka == 0.0
                && let Some(i) = loop_index(face_vertices, vb, va)
            {
                let a_side = slide_side(self, face_vertices, i, th1, true);
                limit = limit.min(f32_mul_scalar(a_side, kb));
            }
            if kc == 0.0
                && let Some(i) = loop_index(face_vertices, vb, vc)
            {
                let c_side = slide_side(
                    self,
                    face_vertices,
                    (i + 1) % face_vertices.len(),
                    th2,
                    false,
                );
                limit = limit.min(f32_mul_scalar(c_side, kb));
            }
        }
        limit
    }

    /// Blender's `vertex_collide_offset`: the offset at which the vertex-only bevel along edge half
    /// `ea` meets the bevel at its other end.
    fn vertex_collide_offset(&self, ea: usize) -> f64 {
        let no_collide_offset = f32_add_scalar(self.p.offset, 1.0e6);
        if self.p.offset == 0.0 {
            return no_collide_offset;
        }
        let ka = f32_div(self.edges[ea].offset_l_spec, self.p.offset);
        let kb = self.find_other_end(ea).map_or(0.0, |(other, _)| {
            f32_div(self.edges[other].offset_l_spec, self.p.offset)
        });
        let kab = f32_add_scalar(ka, kb);
        if kab <= 0.0 {
            return no_collide_offset;
        }
        f32_div(self.edge_length(ea), kab)
    }

    /// Blender's `bevel_limit_offset`: clamps the offset to avoid collisions and scales all specs.
    fn limit_offset(&mut self, vertices: &[usize]) -> Result<()> {
        let mut limited = self.p.offset;
        for vertex in vertices {
            for id in self.bev(*vertex)?.edges.clone() {
                let collision = if self.p.affect_vertices {
                    self.vertex_collide_offset(id)
                } else {
                    self.geometry_collide_offset(id)
                };
                limited = limited.min(collision);
            }
        }
        if limited < self.p.offset {
            let factor = f32_div(limited, self.p.offset);
            for vertex in vertices {
                for id in self.bev(*vertex)?.edges.clone() {
                    let e = &mut self.edges[id];
                    e.offset_l_spec = f32_mul_scalar(e.offset_l_spec, factor);
                    e.offset_r_spec = f32_mul_scalar(e.offset_r_spec, factor);
                    e.offset_l = f32_mul_scalar(e.offset_l, factor);
                    e.offset_r = f32_mul_scalar(e.offset_r, factor);
                }
            }
            self.p.offset = limited;
        }
        Ok(())
    }

    /// Rebuilds every boundary from the current widths.
    fn build_all(&mut self, vertices: &[usize], construct: bool) -> Result<()> {
        for vertex in vertices {
            self.build_boundary(*vertex, construct)?;
        }
        Ok(())
    }

    /// Blender's `adjust_offsets`: groups the boundary vertices that depend on one another into
    /// chains and cycles, and solves each for widths that match their specs as closely as possible.
    fn adjust_offsets(&mut self, vertices: &[usize]) -> Result<()> {
        for vertex in vertices {
            let start = self.bev(*vertex)?.ring[0];
            let mut anchor = start;
            loop {
                if !self.bounds[anchor].visited && self.bounds[anchor].eon.is_some() {
                    self.adjust_chain_from(anchor)?;
                }
                anchor = self.bounds[anchor].next;
                if anchor == start {
                    break;
                }
            }
        }
        self.build_all(vertices, false)?;
        Ok(())
    }

    /// One anchor's step of `adjust_offsets`: find the chain or cycle through `anchor` and solve it.
    fn adjust_chain_from(&mut self, anchor: usize) -> Result<()> {
        let mut v = anchor;
        let mut vchainstart = anchor;
        let mut vchainend = anchor;
        let mut iscycle = false;
        let mut chainlen = 1;
        // Follow paired edges left to right.
        while self.bounds[v].eon.is_some() && !self.bounds[v].visited && !iscycle {
            self.bounds[v].visited = true;
            let Some(efirst) = self.bounds[v].efirst else {
                break;
            };
            let Some((enext, _)) = self.find_other_end(efirst) else {
                break;
            };
            let Some(vnext) = self.edges[enext].leftv else {
                break;
            };
            self.bounds[v].adjchain = Some(vnext);
            vchainend = vnext;
            chainlen += 1;
            if self.bounds[vnext].visited {
                if vnext != vchainstart {
                    break;
                }
                self.adjust_cycle_or_chain(vchainstart, true)?;
                iscycle = true;
            }
            v = vnext;
        }
        if iscycle {
            return Ok(());
        }
        // Then right to left, moving the chain start at each step.
        self.bounds[v].adjchain = None;
        v = vchainstart;
        loop {
            self.bounds[v].visited = true;
            let Some(elast) = self.bounds[v].elast else {
                break;
            };
            let Some((enext, _)) = self.find_other_end(elast) else {
                break;
            };
            let Some(vnext) = self.edges[enext].rightv else {
                break;
            };
            self.bounds[vnext].adjchain = Some(v);
            chainlen += 1;
            vchainstart = vnext;
            v = vnext;
            if self.bounds[v].visited || self.bounds[v].eon.is_none() {
                break;
            }
        }
        if chainlen >= 3
            && self.bounds[vchainstart].eon.is_none()
            && self.bounds[vchainend].eon.is_none()
        {
            self.adjust_cycle_or_chain(vchainstart, false)?;
        }
        Ok(())
    }

    /// Blender's `adjust_the_cycle_or_chain`: least-squares widths for one chain or cycle of
    /// boundary vertices linked through `adjchain`.
    fn adjust_cycle_or_chain(&mut self, vstart: usize, iscycle: bool) -> Result<()> {
        let mut chain = Vec::new();
        let mut current = Some(vstart);
        while let Some(id) = current {
            chain.push(id);
            current = self.bounds[id].adjchain;
            if current == Some(vstart) {
                break;
            }
        }
        let np = chain.len();
        let nrows = if iscycle { 3 * np } else { 3 * np - 3 };
        let mut rows: Vec<(Vec<(usize, f64)>, f64)> = vec![(Vec::new(), 0.0); nrows];
        let weight = MATCH_SPEC_WEIGHT;
        for (i, &v) in chain.iter().enumerate() {
            if iscycle || i < np - 1 {
                let link = self.bounds[v]
                    .adjchain
                    .ok_or_else(|| internal_invariant("chain link"))?;
                let eright = self.bounds[v]
                    .efirst
                    .ok_or_else(|| internal_invariant("chain efirst"))?;
                let enextleft = self.bounds[link]
                    .elast
                    .ok_or_else(|| internal_invariant("chain elast"))?;
                let sinratio = self.bounds[v].sinratio;
                // Residue i: width difference between eright and the next vertex's eleft.
                rows[i].0.push((i, 1.0));
                if iscycle {
                    let row = if i > 0 { i - 1 } else { np - 1 };
                    rows[row].0.push((i, -sinratio));
                } else if i > 0 {
                    rows[i - 1].0.push((i, -sinratio));
                }
                // The right offset of parameter i matches its spec.
                let row = if iscycle { np + 2 * i } else { np - 1 + 2 * i };
                rows[row].0.push((i, weight));
                rows[row].1 += weight * self.edges[eright].offset_r;
                // The left offset of the next vertex matches its spec.
                let next_index = if i == np - 1 { 0 } else { i + 1 };
                rows[row + 1]
                    .0
                    .push((next_index, weight * self.bounds[link].sinratio));
                rows[row + 1].1 += weight * self.edges[enextleft].offset_l;
            } else {
                // The last vertex of a chain: the second part of residue i.
                rows[i - 1].0.push((i, -1.0));
            }
        }
        let solution = solve_least_squares(np, &rows);
        for (i, &v) in chain.iter().enumerate() {
            let val = solution[i];
            if iscycle || i < np - 1 {
                let eright = self.bounds[v]
                    .efirst
                    .ok_or_else(|| internal_invariant("chain efirst"))?;
                let eleft = self.bounds[v]
                    .elast
                    .ok_or_else(|| internal_invariant("chain elast"))?;
                self.edges[eright].offset_r = f64::from(val as f32);
                if iscycle || i > 0 {
                    self.edges[eleft].offset_l = f64::from((self.bounds[v].sinratio * val) as f32);
                }
            } else {
                let eleft = self.bounds[v]
                    .elast
                    .ok_or_else(|| internal_invariant("chain elast"))?;
                self.edges[eleft].offset_l = f64::from(val as f32);
            }
        }
        Ok(())
    }

    /// Blender's `regularize_profile_orientation` over every beveled edge, in mesh edge order:
    /// marks which boundary vertex of each custom profile is its start, consistently along chains.
    fn regularize_profiles(&mut self) -> Result<()> {
        for index in 0..self.geo.edge_order.len() {
            let (a, b) = self.geo.edge_order[index];
            let Some(start) = self.find_edge_half(a, edge_key(a, b)) else {
                continue;
            };
            if !self.edges[start].is_bev || self.edges[start].visited_rpo {
                continue;
            }
            // Start from the boundary vertex highest on Z: any consistent rule works.
            let left = self.edges[start]
                .leftv
                .ok_or_else(|| internal_invariant("leftv"))?;
            let right = self.edges[start]
                .rightv
                .ok_or_else(|| internal_invariant("rightv"))?;
            let right_highest = self.bounds[left].co.z < self.bounds[right].co.z;
            self.bounds[left].is_profile_start = right_highest;
            self.edges[start].visited_rpo = true;
            for direction in 0..2 {
                let mut toward_bv = direction == 0;
                let mut bv = a;
                let mut current = self.next_edgehalf_bev(start, toward_bv, &mut bv)?;
                while let Some(edge) = current {
                    if self.edges[edge].visited_rpo {
                        break;
                    }
                    let leftv = self.edges[edge]
                        .leftv
                        .ok_or_else(|| internal_invariant("leftv"))?;
                    self.bounds[leftv].is_profile_start = if direction == 0 {
                        toward_bv ^ right_highest
                    } else {
                        !toward_bv ^ right_highest
                    };
                    toward_bv = !toward_bv;
                    self.edges[edge].visited_rpo = true;
                    current = self.next_edgehalf_bev(edge, toward_bv, &mut bv)?;
                }
            }
        }
        Ok(())
    }

    /// Blender's `next_edgehalf_bev`: the next beveled edge half along a chain or cycle. Stepping
    /// `toward_bv` stays at the same vertex and moves to the most parallel beveled edge.
    fn next_edgehalf_bev(
        &self,
        start: usize,
        toward_bv: bool,
        bv: &mut usize,
    ) -> Result<Option<usize>> {
        if !toward_bv {
            let Some((half, vertex)) = self.find_other_end(start) else {
                return Ok(None);
            };
            *bv = vertex;
            return Ok(Some(half));
        }
        let selcount = self.bev(*bv)?.selcount;
        if selcount == 1 {
            return Ok(None);
        }
        if selcount == 2 {
            let mut new_edge = self.edges[start].next;
            while !self.edges[new_edge].is_bev {
                new_edge = self.edges[new_edge].next;
            }
            return Ok(Some(new_edge));
        }
        let vertex = *bv;
        let v = self.point(vertex);
        let dir_start = f32_normalize(f32_sub(v, self.point(self.edges[start].other)));
        let mut new_edge = self.edges[start].next;
        let mut second_best_dot = 0.0;
        let mut best_dot = 0.0;
        let mut next_edge = None;
        while new_edge != start {
            if !self.edges[new_edge].is_bev {
                new_edge = self.edges[new_edge].next;
                continue;
            }
            let dir_new = f32_normalize(f32_sub(self.point(self.edges[new_edge].other), v));
            let new_dot = f32_dot(dir_new, dir_start);
            if new_dot > best_dot {
                second_best_dot = best_dot;
                best_dot = new_dot;
                next_edge = Some(new_edge);
            } else if new_dot > second_best_dot {
                second_best_dot = new_dot;
            }
            new_edge = self.edges[new_edge].next;
        }
        let small_ang_dot = f64::from(1.0_f32 - (SMALL_ANG as f32).cos());
        if next_edge.is_some() && f32_sub_scalar(best_dot, second_best_dot).abs() < small_ang_dot {
            return Ok(None);
        }
        Ok(next_edge)
    }
}

/// The slide length along a face beyond a beveled corner (Blender's `A_side_slide` and
/// `C_side_slide`). `index` is the loop that starts the walk; `walk_back` walks the face against
/// its loop order (the A side) rather than along it (the C side).
fn slide_side(
    front: &Front,
    face: &[usize],
    index: usize,
    corner_angle: f64,
    walk_back: bool,
) -> f64 {
    let count = face.len();
    let mut side = 0.0_f32;
    let mut exterior = 0.0_f32;
    let mut first = true;
    let mut current = index;
    while exterior < 1.0e-4_f32 {
        if first {
            exterior = std::f32::consts::PI - corner_angle as f32;
            first = false;
        } else if walk_back {
            current = (current + count - 1) % count;
            exterior += std::f32::consts::PI
                - angle3(
                    front.point(face[current]),
                    front.point(face[(current + 1) % count]),
                    front.point(face[(current + 2) % count]),
                ) as f32;
        } else {
            current = (current + 1) % count;
            exterior += std::f32::consts::PI
                - angle3(
                    front.point(face[(current + count - 1) % count]),
                    front.point(face[current]),
                    front.point(face[(current + 1) % count]),
                ) as f32;
        }
        side +=
            front.key_length(face[current], face[(current + 1) % count]) as f32 * exterior.sin();
    }
    f64::from(side)
}

/// Blender reverses the circular edge array and swaps each edge's faces.
fn reversed_gaps(gaps: &[Option<usize>]) -> Vec<Option<usize>> {
    let count = gaps.len();
    (0..count)
        .map(|index| gaps[(2 * count - 2 - index) % count])
        .collect()
}

/// Index of the loop in `face` that runs along the edge between `a` and `b`, in either order.
fn loop_index(face: &[usize], a: usize, b: usize) -> Option<usize> {
    let count = face.len();
    (0..count).find(|&i| {
        let (x, y) = (face[i], face[(i + 1) % count]);
        (x == a && y == b) || (x == b && y == a)
    })
}

/// Blender's `dist_squared_to_plane_v3` (`math_geom.cc:487-493`): the side is divided by the plane normal's
/// squared length, so a zero normal gives NaN, which fails every `<` test, as in Blender.
fn dist_squared_to_plane(co: DVec3, plane_co: DVec3, plane_no: DVec3) -> f64 {
    let len_sq = plane_no.length_squared();
    let side = plane_no.dot(co - plane_co);
    let fac = side / len_sq;
    len_sq * (fac * fac)
}

/// The profile points of `pro` for `ns` segments, projected onto the profile plane.
fn profile_points(
    pro: &BoundProfile,
    map: Option<[DVec3; 3]>,
    reversed: bool,
    ns: usize,
    xvals: &[f64],
    yvals: &[f64],
) -> Vec<DVec3> {
    (0..=ns)
        .map(|k| {
            let co = if k == 0 {
                pro.start
            } else if k == ns {
                pro.end
            } else if let Some([c0, c1, c3]) = map {
                let (x, y) = if reversed {
                    (yvals[ns - k], xvals[ns - k])
                } else {
                    (xvals[k], yvals[k])
                };
                f32_add(f32_add(f32_mul(c0, x), f32_mul(c1, y)), c3)
            } else {
                let factor = f32_div(k as f64, ns as f64);
                f32_add(pro.start, f32_mul(f32_sub(pro.end, pro.start), factor))
            };
            if pro.proj_dir == DVec3::ZERO {
                co
            } else {
                let co2 = f32_add(co, pro.proj_dir);
                isect_line_plane(co, co2, pro.plane_co, pro.plane_no).unwrap_or(co)
            }
        })
        .collect()
}

/// Least-squares solution of the sparse system `rows` (each row is `(entries, right-hand side)`)
/// for `n` unknowns: the normal equations, solved by Gauss-Jordan elimination with partial pivoting.
fn solve_least_squares(n: usize, rows: &[(Vec<(usize, f64)>, f64)]) -> Vec<f64> {
    let mut normal = vec![vec![0.0; n + 1]; n];
    for (entries, rhs) in rows {
        for &(i, a) in entries {
            for &(j, b) in entries {
                normal[i][j] += a * b;
            }
            normal[i][n] += a * rhs;
        }
    }
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|a, b| normal[*a][pivot].abs().total_cmp(&normal[*b][pivot].abs()))
            .unwrap_or(pivot);
        normal.swap(pivot, best);
        let divisor = normal[pivot][pivot];
        if divisor.abs() < 1.0e-300 {
            continue;
        }
        // Row `pivot` is read while the other rows change, so it is taken out for the loop.
        let pivot_row = std::mem::take(&mut normal[pivot]);
        for (row, target_row) in normal.iter_mut().enumerate() {
            if row == pivot {
                continue;
            }
            let factor = target_row[pivot] / divisor;
            if factor == 0.0 {
                continue;
            }
            for (target, value) in target_row[pivot..=n].iter_mut().zip(&pivot_row[pivot..=n]) {
                *target -= factor * value;
            }
        }
        normal[pivot] = pivot_row;
    }
    (0..n)
        .map(|i| {
            let divisor = normal[i][i];
            if divisor.abs() < 1.0e-300 {
                0.0
            } else {
                normal[i][n] / divisor
            }
        })
        .collect()
}

/// Blender's angle-weighted mesh vertex normals, used when an edge side has no face.
fn vertex_normals(geo: &Geometry) -> Vec<DVec3> {
    let mut normals = vec![DVec3::ZERO; geo.points.len()];
    for (face, (vertices, _)) in geo.polygons.iter().enumerate() {
        for index in 0..vertices.len() {
            let vertex = vertices[index];
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let next = vertices[(index + 1) % vertices.len()];
            let dir_prev = f32_normalize(f32_sub(geo.points[previous], geo.points[vertex]));
            let dir_next = f32_normalize(f32_sub(geo.points[next], geo.points[vertex]));
            let factor =
                crate::geom::normals::float_safe_acos_approx(f32_dot(dir_prev, dir_next) as f32);
            normals[vertex] = f32_add(
                normals[vertex],
                f32_mul(geo.face_normals[face], f64::from(factor)),
            );
        }
    }
    normals
        .iter()
        .map(|normal| f32_normalize(*normal))
        .collect()
}
